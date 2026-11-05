"""Keep shared references consistent when comparing joint reconstructions."""

from pathlib import Path
from runpy import run_path

import numpy as np
import pytest
import torch
from mr2.operators import FastFourierOp, models

from score_qmri.models.diffusion import DiffusionConfig, DiffusionModel, NormalizationConfig
from score_qmri.models.unet import UNet
from score_qmri.sampling import MRIProblem


@pytest.mark.parametrize("changed", [{"seed": 43}, {"regularization": 0.1}])
def test_joint_example_rejects_inconsistent_shared_results(tmp_path, monkeypatch, changed):
    example = run_path(str(Path(__file__).parents[2] / "examples/qmri_joint.py"))

    def stop_before_sampling(*args, **kwargs):
        raise RuntimeError("sampling stopped")

    model = DiffusionModel(
        unet=UNet(base_channels=8, num_res_blocks=1, channel_mult=(1,), attention_levels=(), cond_dim=16),
        diffusion=DiffusionConfig(cond_dim=16),
        compile_model=False,
    )
    monkeypatch.setattr(example["DiffusionModel"], "from_checkpoint", lambda *args, **kwargs: model)
    monkeypatch.setattr(example["DPSSampler"], "_step", stop_before_sampling)
    monkeypatch.setattr(example["DAPSSampler"], "_conditional", stop_before_sampling)
    case = tmp_path / "input.npz"
    np.savez(
        case,
        t1=np.full((8, 8), 1.2, dtype=np.float32),
        m0=np.ones((8, 8), dtype=np.complex64),
        times=np.array([0.5, 1.0, 2.0, 8.0], dtype=np.float32),
        mask=np.ones((4, 8, 8), dtype=np.float32),
        csm=np.ones((1, 8, 8), dtype=np.complex64),
    )
    output = tmp_path / "results"
    arguments = (tmp_path / "model.ckpt", case, output, 0.2383)
    with monkeypatch.context() as loading:
        loading.setattr(example["DiffusionModel"], "from_checkpoint", stop_before_sampling)
        with pytest.raises(RuntimeError, match="sampling stopped"):
            example["sample"](*arguments, device="cpu")
    assert not any(output.iterdir())
    with pytest.raises(ValueError, match="finite and positive"):
        example["sample"](*arguments[:3], 0.0, device="cpu")
    assert not any(output.iterdir())
    with pytest.raises(RuntimeError, match="sampling stopped"):
        example["sample"](*arguments, device="cpu")
    np.savez(output / "dps.npz", t1=np.zeros((2, 8, 8)))
    saved = {path: path.read_bytes() for path in output.iterdir()}

    with pytest.raises(ValueError, match="use a new output directory"):
        example["sample"](*arguments, device="cpu", **changed)
    assert all(path.read_bytes() == contents for path, contents in saved.items())

    with pytest.raises(RuntimeError, match="sampling stopped"):
        example["sample"](*arguments, device="cpu", method="daps")
    assert all(path.read_bytes() == contents for path, contents in saved.items())


@pytest.mark.parametrize("t1_scale", [0.25, 0.5])
def test_joint_example_respects_checkpoint_t1_scale(t1_scale):
    example = run_path(str(Path(__file__).parents[2] / "examples/qmri_joint.py"))
    model = DiffusionModel(
        unet=UNet(base_channels=8, num_res_blocks=1, channel_mult=(1,), attention_levels=(), cond_dim=16),
        diffusion=DiffusionConfig(cond_dim=16, normalization=NormalizationConfig(kind_scales={"T1MAP": t1_scale})),
        compile_model=False,
    )
    prior = example["JointPrior"](model, 0.2383)
    assert prior.t1_scale == t1_scale
    state = torch.tensor([1.2 * t1_scale, 0, 1, 0.2], dtype=torch.float64)[None, :, None, None].expand(1, 4, 2, 3)
    state = state.clone().requires_grad_(True)
    signal = models.SaturationRecovery(torch.tensor([0.5, 1.0, 2.0, 8.0], dtype=torch.float64))
    encoding = FastFourierOp(dim=(-2, -1))
    m0 = torch.complex(state[:, 2].detach(), state[:, 3].detach())
    measurement = encoding(signal(m0, torch.full_like(m0.real, 1.2))[0])[0] + 0.01j
    physical = MRIProblem(encoding, measurement, 0.03, signal_model=signal, measurement_batch_dim=1)
    joint = example["JointProblem"](physical, 1, prior.t1_scale)
    torch.testing.assert_close(joint.decode(state)[0], torch.full_like(m0.real, 1.2))
    loss = joint.residual(state).abs().square().sum() / (2 * physical.noise_std**2)
    (expected,) = torch.autograd.grad(loss, state)
    torch.testing.assert_close(joint.gradient(state), expected, rtol=1e-12, atol=1e-12)


def test_joint_plot_rejects_one_draw(tmp_path):
    example = run_path(str(Path(__file__).parents[2] / "examples/qmri_joint.py"))
    maps = {"t1": np.ones((8, 8)), "m0": np.ones((8, 8))}
    np.savez(tmp_path / "case.npz", **maps)
    np.savez(tmp_path / "cg-fit.npz", **maps)
    np.savez(tmp_path / "dps.npz", **{key: value[None] for key, value in maps.items()})
    with pytest.raises(ValueError, match="At least two draws"):
        example["plot"](tmp_path, tmp_path / "overview")
