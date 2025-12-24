"""Training configuration."""

import math
from collections.abc import Callable
from dataclasses import dataclass, field


def _default_loss_schedule(step: int) -> float:
    """Default schedule: MSE for first 10k steps, then SSIM+L1."""
    return 0.0 if step < 10000 else 1.0


@dataclass
class TrainingConfig:
    """Configuration for diffusion training.

    Attributes
    ----------
    unet_base_channels : int
        Base channel count for UNet.
    unet_num_blocks : int
        Number of down/up blocks in UNet.
    num_kinds : int
        Number of dataset kinds for FiLM conditioning.
    num_timesteps : int
        Diffusion timesteps.
    beta_start, beta_end : float
        Linear noise schedule endpoints.
    batch_size : int
        Training batch size.
    learning_rate : float
        Adam learning rate.
    num_workers : int
        DataLoader workers.
    max_steps : int
        Maximum training steps.
    validation_freq : int
        Steps between validation runs.
    gradient_clip_val : float | None
        Gradient clipping value.
    loss_schedule_fn : Callable[[int], float]
        Function mapping step to SSIM+L1 weight (0=MSE, 1=SSIM+L1).
    ssim_weight, l1_weight : float
        Weights for SSIM+L1 loss components.
    phase_aug_prob : float
        Probability of applying phase augmentation.
    phase_mean_range : tuple[float, float]
        Range for mean phase offset.
    phase_poly_coeff_range : tuple[float, float]
        Range for polynomial coefficients.
    wandb_project : str
        W&B project name.
    wandb_run_name : str | None
        W&B run name.
    wandb_entity : str | None
        W&B entity.
    checkpoint_dir : str
        Directory for checkpoints.
    save_top_k : int
        Number of best checkpoints to keep.
    resume_from_checkpoint : str | None
        Path to resume checkpoint.
    """

    unet_base_channels: int = 64
    unet_num_blocks: int = 4
    num_kinds: int = 3
    num_timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02

    batch_size: int = 8
    learning_rate: float = 1e-4
    num_workers: int = 4
    max_steps: int = 100000
    validation_freq: int = 1000
    gradient_clip_val: float | None = 1.0

    loss_schedule_fn: Callable[[int], float] = field(default=_default_loss_schedule)
    ssim_weight: float = 0.5
    l1_weight: float = 0.5

    phase_aug_prob: float = 0.5
    phase_mean_range: tuple[float, float] = (0.0, 2.0 * math.pi)
    phase_poly_coeff_range: tuple[float, float] = (-0.1, 0.1)

    wandb_project: str = "mri-diffusion"
    wandb_run_name: str | None = None
    wandb_entity: str | None = None

    checkpoint_dir: str = "./checkpoints"
    save_top_k: int = 3
    resume_from_checkpoint: str | None = None
