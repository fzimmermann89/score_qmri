"""Tests for sample generation utilities."""

from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from score_qmri.datasets.kind import DatasetKind
from score_qmri.models.diffusion import DiffusionConfig, DiffusionModel, NormalizationConfig
from score_qmri.models.unet import UNet
from score_qmri.sampling.generate import generate_samples
from score_qmri.utils.plot import plot_samples_grid


@pytest.fixture
def small_model() -> DiffusionModel:
    """Create a small model for fast testing."""
    unet = UNet(
        base_channels=16,
        num_res_blocks=1,
        channel_mult=(1, 2),
        cond_dim=32,
        attention_levels=(),
    )
    diffusion = DiffusionConfig(cond_dim=32)
    model = DiffusionModel(unet=unet, diffusion=diffusion)
    model.eval()
    return model


class TestGenerateSamples:
    """Tests for generate_samples function."""

    def test_generate_magnitude_samples(self, small_model: DiffusionModel) -> None:
        samples = generate_samples(
            model=small_model,
            kind=DatasetKind.T1W_MAGNITUDE,
            n_samples=2,
            size=32,
            num_steps=3,
            seed=42,
        )
        assert samples.shape == (2, 32, 32)
        assert not samples.is_complex()
        assert torch.isfinite(samples).all()

    def test_generate_complex_samples(self, small_model: DiffusionModel) -> None:
        samples = generate_samples(
            model=small_model,
            kind=DatasetKind.COMPLEX_T1W_COIL_COMBINED,
            n_samples=2,
            size=32,
            num_steps=3,
            seed=42,
        )
        assert samples.shape == (2, 32, 32)
        assert samples.is_complex()

    def test_reproducibility_with_seed(self, small_model: DiffusionModel) -> None:
        samples1 = generate_samples(
            model=small_model,
            kind=DatasetKind.T1W_MAGNITUDE,
            n_samples=1,
            size=32,
            num_steps=3,
            seed=123,
        )
        samples2 = generate_samples(
            model=small_model,
            kind=DatasetKind.T1W_MAGNITUDE,
            n_samples=1,
            size=32,
            num_steps=3,
            seed=123,
        )
        assert torch.allclose(samples1, samples2)

    def test_invalid_sampler_raises(self, small_model: DiffusionModel) -> None:
        with pytest.raises(ValueError, match="Unknown sampler"):
            generate_samples(
                model=small_model,
                kind=DatasetKind.T1W_MAGNITUDE,
                n_samples=1,
                size=32,
                sampler_type="invalid",
            )


class TestPlotSamplesGrid:
    """Tests for plot_samples_grid."""

    def test_saves_png(self, tmp_path: Path) -> None:
        samples = torch.randn(4, 32, 32)
        output = tmp_path / "test.png"

        plot_samples_grid(samples, output, title="Test")

        assert output.exists()
        assert output.stat().st_size > 0

    def test_handles_complex(self, tmp_path: Path) -> None:
        samples = torch.complex(
            torch.randn(4, 32, 32),
            torch.randn(4, 32, 32),
        )
        output = tmp_path / "complex.png"

        plot_samples_grid(samples, output)

        assert output.exists()

    def test_auto_grid_layout(self, tmp_path: Path) -> None:
        for n in [1, 3, 4, 7, 9, 16]:
            samples = torch.randn(n, 32, 32)
            output = tmp_path / f"grid_{n}.png"
            plot_samples_grid(samples, output)
            assert output.exists()

    def test_single_sample(self, tmp_path: Path) -> None:
        samples = torch.randn(1, 64, 64)
        output = tmp_path / "single.png"

        plot_samples_grid(samples, output)

        assert output.exists()


class TestLoadCheckpoint:
    """Tests for DiffusionModel.load_checkpoint."""

    def test_loads_ema_weights(self, tmp_path: Path, small_model: DiffusionModel) -> None:
        ckpt_path = tmp_path / "test.ckpt"
        ema_state = {"shadow": small_model.state_dict(), "decay": 0.9999}
        torch.save({"ema": ema_state}, ckpt_path)

        unet = UNet(
            base_channels=16,
            num_res_blocks=1,
            channel_mult=(1, 2),
            cond_dim=32,
            attention_levels=(),
        )
        diffusion = DiffusionConfig(cond_dim=32)
        fresh_model = DiffusionModel(unet=unet, diffusion=diffusion)

        fresh_model.load_checkpoint(ckpt_path)

        for (n1, p1), (_n2, p2) in zip(
            small_model.named_parameters(),
            fresh_model.named_parameters(),
            strict=True,
        ):
            assert torch.allclose(p1, p2), f"Parameter {n1} mismatch"

    def test_from_checkpoint_reconstructs_model(self, tmp_path: Path, small_model: DiffusionModel) -> None:
        ckpt_path = tmp_path / "test.ckpt"
        unet = {
            "class_path": "score_qmri.models.unet.UNet",
            "init_args": {
                "in_channels": 2,
                "base_channels": 16,
                "num_res_blocks": 1,
                "channel_mult": (1, 2),
                "attention_levels": (),
            },
        }
        diffusion = DiffusionConfig(
            cond_dim=32,
            sigma_min=0.01,
            sigma_max=5.0,
            sigma_data=0.3,
            p_mean=-1.1,
            p_std=1.1,
            rho=5.5,
            normalization=NormalizationConfig(
                train_clip=1.3,
                sample_clip=0.9,
                kind_scales={"T1MAP": 0.2},
            ),
        )
        hparams = {"diffusion": asdict(diffusion)}
        state_dict = {f"model.{k}": v for k, v in small_model.state_dict().items()}
        cli_config = {"model": {"unet": unet, "diffusion": asdict(diffusion)}}
        torch.save({"state_dict": state_dict, "hyper_parameters": hparams, "cli_config": cli_config}, ckpt_path)

        model = DiffusionModel.from_checkpoint(ckpt_path)

        assert model.sigma_min == 0.01
        assert model.sigma_max == 5.0
        assert model.sigma_data == 0.3
        assert model.p_mean == -1.1
        assert model.p_std == 1.1
        assert model.rho == 5.5
        assert model.train_clip == 1.3
        assert model.sample_clip == 0.9
        assert model._kind_scale_lookup[int(DatasetKind.T1MAP)] == 0.2

        for (n1, p1), (_n2, p2) in zip(
            small_model.named_parameters(),
            model.named_parameters(),
            strict=True,
        ):
            assert torch.allclose(p1, p2), f"Parameter {n1} mismatch"

    def test_fallback_to_state_dict(self, tmp_path: Path, small_model: DiffusionModel) -> None:
        ckpt_path = tmp_path / "test.ckpt"
        state_dict = {f"model.{k}": v for k, v in small_model.state_dict().items()}
        torch.save({"state_dict": state_dict}, ckpt_path)

        unet = UNet(
            base_channels=16,
            num_res_blocks=1,
            channel_mult=(1, 2),
            cond_dim=32,
            attention_levels=(),
        )
        diffusion = DiffusionConfig(cond_dim=32)
        fresh_model = DiffusionModel(unet=unet, diffusion=diffusion)

        fresh_model.load_checkpoint(ckpt_path)

        for (n1, p1), (_n2, p2) in zip(
            small_model.named_parameters(),
            fresh_model.named_parameters(),
            strict=True,
        ):
            assert torch.allclose(p1, p2), f"Parameter {n1} mismatch"
