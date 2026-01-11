"""Tests for sample generation utilities."""

from pathlib import Path

import pytest
import torch

from score_qmri.datasets.kind import DatasetKind
from score_qmri.models.diffusion import DiffusionModel
from score_qmri.sampling.generate import generate_samples
from score_qmri.utils.plot import plot_samples_grid


@pytest.fixture
def small_model() -> DiffusionModel:
    """Create a small model for fast testing."""
    model = DiffusionModel(
        base_channels=16,
        num_res_blocks=1,
        channel_mult=(1, 2),
        cond_dim=32,
    )
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

    def test_ddpm_sampler(self, small_model: DiffusionModel) -> None:
        samples = generate_samples(
            model=small_model,
            kind=DatasetKind.T1MAP,
            n_samples=1,
            size=32,
            sampler_type="ddpm",
            num_steps=3,
            seed=42,
        )
        assert samples.shape == (1, 32, 32)

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

    def test_ddim_eta_parameter(self, small_model: DiffusionModel) -> None:
        samples = generate_samples(
            model=small_model,
            kind=DatasetKind.T1W_MAGNITUDE,
            n_samples=1,
            size=32,
            sampler_type="ddim",
            num_steps=3,
            eta=1.0,
            seed=42,
        )
        assert samples.shape == (1, 32, 32)


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

        fresh_model = DiffusionModel(
            base_channels=16,
            num_res_blocks=1,
            channel_mult=(1, 2),
            cond_dim=32,
        )

        fresh_model.load_checkpoint(ckpt_path)

        for (n1, p1), (_n2, p2) in zip(
            small_model.named_parameters(),
            fresh_model.named_parameters(),
            strict=True,
        ):
            assert torch.allclose(p1, p2), f"Parameter {n1} mismatch"

    def test_fallback_to_state_dict(self, tmp_path: Path, small_model: DiffusionModel) -> None:
        ckpt_path = tmp_path / "test.ckpt"
        state_dict = {f"model.{k}": v for k, v in small_model.state_dict().items()}
        torch.save({"state_dict": state_dict}, ckpt_path)

        fresh_model = DiffusionModel(
            base_channels=16,
            num_res_blocks=1,
            channel_mult=(1, 2),
            cond_dim=32,
        )

        fresh_model.load_checkpoint(ckpt_path)

        for (n1, p1), (_n2, p2) in zip(
            small_model.named_parameters(),
            fresh_model.named_parameters(),
            strict=True,
        ):
            assert torch.allclose(p1, p2), f"Parameter {n1} mismatch"
