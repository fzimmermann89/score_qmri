"""Per-dataset-kind running normalization with EMA."""

import torch
import torch.distributed as dist
from torch import Tensor

from score_qmri.datasets.kind import NUM_DATASET_KINDS


class RunningNormalizer(torch.nn.Module):
    """Track per-kind running mean/std and normalize to zero mean, unit variance.

    Uses exponential moving average. Updates are frozen in eval mode.
    Statistics are tracked separately for each channel.
    """

    def __init__(
        self,
        num_kinds: int = NUM_DATASET_KINDS,
        momentum: float = 0.001,
        eps: float = 1e-6,
    ) -> None:
        """
        Parameters
        ----------
        num_kinds
            Number of DatasetKind values.
        momentum
            EMA momentum factor. Effective window is approximately 1/momentum samples.
        eps
            Numerical stability constant for std division.
        """
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.num_kinds = num_kinds

        for k in range(num_kinds):
            self.register_buffer(f"mean_{k}", torch.zeros(2))
            self.register_buffer(f"var_{k}", torch.ones(2))
            self.register_buffer(f"n_{k}", torch.zeros(1, dtype=torch.long))

    def _get_buffers(self, k: int) -> tuple[Tensor, Tensor, Tensor]:
        """Get (mean, var, n) buffers for kind k."""
        return (
            getattr(self, f"mean_{k}"),
            getattr(self, f"var_{k}"),
            getattr(self, f"n_{k}"),
        )

    @torch.no_grad()
    def _update(self, x: Tensor, mask: Tensor, kind: Tensor) -> None:
        """Update running stats from batch. Only masked-in pixels contribute."""
        C = x.shape[1]

        for k in kind.unique():
            k_val = k.item()
            idx = kind == k
            x_k = x[idx]
            m_k = mask[idx].expand_as(x_k)

            n_valid = m_k.sum(dim=(0, 2, 3))
            if n_valid.min() < 1:
                continue

            batch_mean = (x_k * m_k).sum(dim=(0, 2, 3)) / n_valid
            centered = x_k - batch_mean[None, :, None, None]
            batch_var = (centered.pow(2) * m_k).sum(dim=(0, 2, 3)) / n_valid

            mean, var, n = self._get_buffers(k_val)

            if n.item() == 0:
                mean[:C].copy_(batch_mean)
                var[:C].copy_(batch_var.clamp(min=self.eps))
            else:
                mean[:C].lerp_(batch_mean, self.momentum)
                var[:C].lerp_(batch_var, self.momentum)

            n.add_(1)

    def _get_stats(self, k: int, num_channels: int) -> tuple[Tensor, Tensor]:
        """Get mean and std for kind k."""
        mean, var, _ = self._get_buffers(k)
        std = var[:num_channels].sqrt().clamp(min=self.eps)
        return mean[:num_channels], std

    def normalize(self, x: Tensor, mask: Tensor, kind: Tensor) -> Tensor:
        """Update stats (if training) then normalize.

        Parameters
        ----------
        x
            Input tensor (B, C, H, W) where C is 1 or 2.
        mask
            Boolean mask (B, 1, H, W) or (B, H, W).
        kind
            Dataset kind indices (B,).

        Returns
        -------
        Normalized tensor with approximately zero mean and unit std per kind/channel.
        """
        if mask.ndim == 3:
            mask = mask[:, None, :, :]

        if self.training:
            self._update(x, mask, kind)

        C = x.shape[1]
        result = torch.empty_like(x)

        for k in kind.unique():
            k_val = k.item()
            idx = kind == k
            mean, std = self._get_stats(k_val, C)
            result[idx] = (x[idx] - mean[None, :, None, None]) / std[None, :, None, None]

        return result

    def denormalize(self, x: Tensor, kind: Tensor) -> Tensor:
        """Reverse normalization.

        Parameters
        ----------
        x
            Normalized tensor (B, 2, H, W).
        kind
            Dataset kind indices (B,).

        Returns
        -------
        Denormalized tensor in original scale.
        """
        result = x.clone()

        for k in kind.unique():
            k_val = k.item()
            idx = kind == k
            mean, std = self._get_stats(k_val, 2)
            result[idx] = x[idx] * std[None, :, None, None] + mean[None, :, None, None]

        return result

    @torch.no_grad()
    def sync_stats(self) -> None:
        """Synchronize running statistics across all DDP ranks.

        All ranks must call this together to avoid deadlock. Uses count-weighted
        averaging for correct mean/variance merging across ranks with different
        sample counts.
        """
        if not dist.is_initialized():
            return

        world_size = dist.get_world_size()
        if world_size == 1:
            return

        for k in range(self.num_kinds):
            mean, var, n = self._get_buffers(k)

            n_total = n.clone().float()
            dist.all_reduce(n_total, op=dist.ReduceOp.SUM)

            if n_total.item() == 0:
                continue

            weighted_mean = mean * n.float()
            weighted_var = var * n.float()
            dist.all_reduce(weighted_mean, op=dist.ReduceOp.SUM)
            dist.all_reduce(weighted_var, op=dist.ReduceOp.SUM)

            mean.copy_(weighted_mean / n_total)
            var.copy_((weighted_var / n_total).clamp(min=self.eps))
