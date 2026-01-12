"""Per-dataset-kind running normalization with EMA."""

import torch
import torch.distributed as dist
from torch import Tensor

from score_qmri.datasets.kind import NUM_DATASET_KINDS, is_complex_kind


class RunningNormalizer(torch.nn.Module):
    """Track per-kind running mean/std and normalize to zero mean, unit variance.

    Uses exponential moving average. Updates are frozen in eval mode.
    Statistics are tracked separately for each channel.

    For real kinds (magnitude/mapping), imaginary channel stats are frozen
    at mean=0, var=1 to maintain identity transform on that channel.
    """

    def __init__(
        self,
        num_kinds: int = NUM_DATASET_KINDS,
        momentum: float = 0.001,
        eps: float = 1e-5,
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

        self.register_buffer("mean", torch.zeros(num_kinds, 2))
        self.register_buffer("var", torch.ones(num_kinds, 2))
        self.register_buffer("n", torch.zeros(num_kinds, dtype=torch.long))

    mean: Tensor
    var: Tensor
    n: Tensor

    @torch.no_grad()
    def _update(self, x: Tensor, mask: Tensor, kind: Tensor) -> None:
        """Update running stats from batch. Only masked-in pixels contribute."""
        for k in kind.unique():
            k_idx = int(k.item())
            idx = kind == k
            x_k = x[idx]
            m_k = mask[idx].expand_as(x_k)

            n_valid = m_k.sum(dim=(0, 2, 3))
            if n_valid.min() < 1:
                continue

            batch_mean = (x_k * m_k).sum(dim=(0, 2, 3)) / n_valid
            centered = x_k - batch_mean[None, :, None, None]
            batch_var = (centered.pow(2) * m_k).sum(dim=(0, 2, 3)) / n_valid
            batch_var = batch_var.clamp(min=self.eps)

            if not is_complex_kind(k_idx):
                batch_mean[1] = 0.0
                batch_var[1] = 1.0

            if self.n[k_idx] == 0:
                self.mean[k_idx].copy_(batch_mean)
                self.var[k_idx].copy_(batch_var)
            else:
                self.mean[k_idx].lerp_(batch_mean, self.momentum)
                self.var[k_idx].lerp_(batch_var, self.momentum)

            self.n[k_idx] += 1

    def __getitem__(self, k: int) -> tuple[Tensor, Tensor]:
        std = self.var[k].sqrt().clamp(min=self.eps)
        return self.mean[k], std

    def normalize(self, x: Tensor, mask: Tensor, kind: Tensor) -> Tensor:
        """Update stats (if training) then normalize.

        Parameters
        ----------
        x
            Input tensor (B, 2, H, W).
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

        result = torch.empty_like(x)

        for k in kind.unique():
            k_idx = int(k.item())
            idx = kind == k
            mean, std = self[k_idx]
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
            k_idx = int(k.item())
            idx = kind == k
            mean, std = self[k_idx]
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

        n_total = self.n.clone().float()
        dist.all_reduce(n_total, op=dist.ReduceOp.SUM)

        weighted_mean = self.mean * self.n[:, None].float()
        weighted_var = self.var * self.n[:, None].float()
        dist.all_reduce(weighted_mean, op=dist.ReduceOp.SUM)
        dist.all_reduce(weighted_var, op=dist.ReduceOp.SUM)

        valid = n_total > 0
        self.mean[valid] = weighted_mean[valid] / n_total[valid, None]
        self.var[valid] = (weighted_var[valid] / n_total[valid, None]).clamp(min=self.eps)

        for k_idx in range(self.num_kinds):
            if not is_complex_kind(k_idx):
                self.mean[k_idx, 1] = 0.0
                self.var[k_idx, 1] = 1.0
