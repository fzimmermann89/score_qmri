"""Fixed-noise metrics and genuine shared-backbone gradient controls."""

import pytest
import torch

from score_qmri.models.diffusion import DiffusionConfig, DiffusionModel
from score_qmri.models.unet import UNet
from score_qmri.training.validation import backbone_gradient_diagnostics, denoising_metrics, validation_reference


def model_and_batch():
    model = DiffusionModel(
        unet=UNet(base_channels=8, num_res_blocks=1, channel_mult=(1,), attention_levels=(), cond_dim=16),
        diffusion=DiffusionConfig(cond_dim=16),
        compile_model=False,
    ).eval()
    batch = {
        "image": torch.full((1, 2, 32, 32), 0.1),
        "kind": torch.tensor([1]),
        "background_mask": torch.zeros(1, 32, 32, dtype=torch.bool),
        "penumbra_mask": torch.zeros(1, 32, 32, dtype=torch.bool),
    }
    return model, batch


def test_zero_network_matches_baseline_and_fixed_noise_does_not_consume_rng():
    model, batch = model_and_batch()
    clean = validation_reference(model, batch)
    noise = torch.randn(clean.shape, generator=torch.Generator().manual_seed(42))
    before = torch.get_rng_state().clone()
    first = denoising_metrics(model, clean, batch["kind"], 0.01, noise)
    second = denoising_metrics(model, clean, batch["kind"], 0.01, noise)
    assert torch.equal(before, torch.get_rng_state())
    torch.testing.assert_close(first["main_edm_mse"], first["baseline_edm_mse"])
    torch.testing.assert_close(first["main_edm_mse"], second["main_edm_mse"], rtol=0, atol=0)


def test_backbone_ratio_scales_with_aux_weight_without_modifying_gradients():
    model, batch = model_and_batch()
    # Both heads start at zero; nonzero trained-head features are needed to
    # measure backbone interference rather than a trivial zero gradient.
    with torch.no_grad():
        model.unet.main_head.weight.normal_(std=0.01)
        model.unet.aux_head.weight.normal_(std=0.01)
    noise = torch.randn(batch["image"].shape, generator=torch.Generator().manual_seed(3))
    original = {name: p.detach().clone() for name, p in model.named_parameters()}
    full = backbone_gradient_diagnostics(model, batch, 0.01, noise, aux_weight=0.1)
    half = backbone_gradient_diagnostics(model, batch, 0.01, noise, aux_weight=0.05)
    assert full["ratio_defined"] == 1
    assert full["main_backbone_norm"] > 0 and full["aux_backbone_norm"] > 0
    assert half["aux_over_main_backbone_norm"] == pytest.approx(full["aux_over_main_backbone_norm"] / 2, rel=1e-5)
    assert half["backbone_gradient_cosine"] == pytest.approx(full["backbone_gradient_cosine"], abs=1e-6)
    for name, p in model.named_parameters():
        assert p.grad is None
        torch.testing.assert_close(p, original[name], rtol=0, atol=0)


def test_normalized_map_reference_is_not_normalized_twice():
    model, batch = model_and_batch()
    batch["kind"] = torch.tensor([6])
    batch["image"][:, 1] = 0
    clean = validation_reference(model, batch)
    noise = torch.ones_like(clean)
    with torch.no_grad():
        model.unet.main_head.weight.normal_(std=0.01)
        model.unet.aux_head.weight.normal_(std=0.01)
    expected = backbone_gradient_diagnostics(model, batch, 0.01, noise)
    normalized_batch = {**batch, "image": clean}
    actual = backbone_gradient_diagnostics(model, normalized_batch, 0.01, noise, normalized_clean=clean)
    assert actual == pytest.approx(expected)
