"""Per-dataset-kind running normalization with EMA using 3-sigma scaling."""

import torch
import torch.distributed as dist
from torch import Tensor

from score_qmri.datasets.kind import NUM_DATASET_KINDS, is_complex_kind

EPS = 1e-6


class RunningNormalizer(torch.nn.Module):
    """Track per-kind running mean/std and normalize using 3-sigma scaling.

    Uses exponential moving average. Updates are frozen in eval mode.
    Statistics are tracked separately for each channel.

    Normalization formula:
    - Complex kinds: scale = 3 * std + eps, x_norm = x / scale
    - Real kinds: scale = mean + 3 * std + eps, x_norm = x / scale

    For real kinds (magnitude/mapping), imaginary channel stats are frozen
    at mean=0, var=1 to maintain identity transform on that channel.
    """

    def __init__(
        self,
        num_kinds: int = NUM_DATASET_KINDS,
        momentum: float = 0.001,
    ) -> None:
        """
        Parameters
        ----------
        num_kinds
            Number of DatasetKind values.
        momentum
            EMA momentum factor. Effective window is approximately 1/momentum samples.
        """
        super().__init__()
        self.momentum = momentum
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
            batch_var = batch_var.clamp(min=EPS)

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

    def get_scale(self, k: int) -> Tensor:
        """Get normalization scale for a dataset kind.

        Returns scale = mean + 3*std + eps for real kinds, 3*std + eps for complex.
        """
        std = self.var[k].sqrt().clamp(min=EPS)
        if is_complex_kind(k):
            return 3.0 * std + EPS
        mean = self.mean[k].clamp(min=0.0)
        return mean + 3.0 * std + EPS

    def normalize(self, x: Tensor, mask: Tensor, kind: Tensor) -> Tensor:
        """Update stats (if training) then normalize using 3-sigma scaling.

        Parameters
        ----------
        x
            Input tensor (B, 2, H, W).
        mask
            Boolean mask (B, 1, H, W) or (B, H, W). True = valid signal.
        kind
            Dataset kind indices (B,).

        Returns
        -------
        Normalized tensor scaled to approximately [-1, 1] or [0, 1] range.
        """
        if mask.ndim == 3:
            mask = mask[:, None, :, :]

        if self.training:
            self._update(x, mask, kind)

        result = torch.empty_like(x)

        for k in kind.unique():
            k_idx = int(k.item())
            idx = kind == k
            scale = self.get_scale(k_idx)
            result[idx] = x[idx] / scale[None, :, None, None]

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
            scale = self.get_scale(k_idx)
            result[idx] = x[idx] * scale[None, :, None, None]

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
        self.var[valid] = (weighted_var[valid] / n_total[valid, None]).clamp(min=EPS)

        for k_idx in range(self.num_kinds):
            if not is_complex_kind(k_idx):
                self.mean[k_idx, 1] = 0.0
                self.var[k_idx, 1] = 1.0
