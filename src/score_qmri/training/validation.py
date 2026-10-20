"""Fixed-case denoiser validation and auxiliary/main backbone gradient checks."""

import numpy as np
import torch
from torch import Tensor

from score_qmri.models.diffusion import DiffusionModel
from score_qmri.training.losses import create_channel_mask, l1_loss, noise_loss, ssim_loss

VALIDATION_SIGMAS = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 2.0, 20.0)


def validation_reference(model: DiffusionModel, batch: dict[str, Tensor]) -> Tensor:
    """Apply the same clean-target normalization, background zeroing and clipping as training."""
    clean = model.normalize(batch["image"], batch["kind"])
    clean = torch.where(batch["background_mask"].unsqueeze(1), 0, clean)
    return model.clamp_x0(clean, batch["kind"], limit=model.train_clip)


@torch.no_grad()
def denoising_metrics(model: DiffusionModel, clean: Tensor, kind: Tensor, sigma: float, noise: Tensor) -> dict[str, Tensor]:
    """Return per-case unclipped main-head and zero-network EDM MSE with identical noise."""
    if not model.sigma_min <= sigma <= model.sigma_max:
        raise ValueError("Validation sigma outside model support")
    channel = create_channel_mask(kind).to(clean)
    point = clean + sigma * noise * channel
    prediction = model.denoise(point, clean.new_full((len(clean),), sigma), kind)
    weight = (sigma**2 + model.sigma_data**2) / (sigma * model.sigma_data) ** 2
    baseline = model.sigma_data**2 / (sigma**2 + model.sigma_data**2) * point
    return {
        "main_edm_mse": noise_loss(prediction, clean, None, channel, reduction="none") * weight,
        "baseline_edm_mse": noise_loss(baseline, clean, None, channel, reduction="none") * weight,
    }


@torch.enable_grad()
def backbone_gradient_diagnostics(
    model: DiffusionModel,
    batch: dict[str, Tensor],
    sigma: float,
    noise: Tensor,
    *,
    aux_weight: float = 0.1,
    main_weight: float = 1.0,
    ssim_weight: float = 0.7,
    l1_weight: float = 0.3,
    background_weight: float = 0.5,
    penumbra_weight: float = 0.01,
) -> dict[str, float]:
    """Measure separate loss gradients on shared parameters, without changing .grad.

    Use trained eval-mode weights: zero-initialized heads give zero backbone
    gradients. Ratios are raw gradients, not Adam-preconditioned update ratios.
    """
    if len(batch["image"]) != 1:
        raise ValueError("Backbone diagnostics evaluate one case at a time")
    if not model.sigma_min <= sigma <= model.sigma_max:
        raise ValueError("Diagnostic sigma outside model support")
    clean = validation_reference(model, batch)
    kind = batch["kind"]
    channel = create_channel_mask(kind).to(clean)
    output = model(clean + sigma * noise * channel, clean.new_full((len(clean),), sigma), kind)
    weight = (sigma**2 + model.sigma_data**2) / (sigma * model.sigma_data) ** 2
    main = main_weight * weight * noise_loss(output.x0_main, clean, None, channel)
    spatial = torch.where(batch["background_mask"], background_weight, 1.0)
    spatial = torch.where(batch["penumbra_mask"], penumbra_weight, spatial).unsqueeze(1)
    auxiliary = (
        aux_weight
        * weight**0.5
        * (ssim_weight * ssim_loss(output.x0_aux, clean, spatial, channel) + l1_weight * l1_loss(output.x0_aux, clean, spatial, channel))
    )
    trunk = [p for name, p in model.named_parameters() if p.requires_grad and "main_head" not in name and "aux_head" not in name]
    main_grads = torch.autograd.grad(main, trunk, retain_graph=True, allow_unused=True)
    aux_grads = torch.autograd.grad(auxiliary, trunk, allow_unused=True)
    zero = clean.new_zeros(())
    main_sq = sum((g.detach().square().sum() for g in main_grads if g is not None), zero)
    aux_sq = sum((g.detach().square().sum() for g in aux_grads if g is not None), zero)
    dot = sum(
        (m.detach().mul(a.detach()).sum() for m, a in zip(main_grads, aux_grads, strict=True) if m is not None and a is not None), zero
    )
    result = {
        "main_backbone_norm": float(main_sq.sqrt()),
        "aux_backbone_norm": float(aux_sq.sqrt()),
        "ratio_defined": float(main_sq > 0),
        "aux_over_main_backbone_norm": float((aux_sq / main_sq.clamp_min(1e-30)).sqrt()),
        "backbone_gradient_cosine": float(dot / (main_sq * aux_sq).sqrt().clamp_min(1e-30)),
    }
    if not all(np.isfinite(value) for value in result.values()):
        raise FloatingPointError("Nonfinite backbone gradient diagnostic")
    return result
