"""Keep shared references consistent when comparing joint reconstructions."""

from pathlib import Path
from runpy import run_path

import numpy as np
import pytest


@pytest.mark.parametrize("changed", [{"seed": 43}, {"regularization": 0.1}])
def test_joint_example_rejects_inconsistent_shared_results(tmp_path, monkeypatch, changed):
    example = run_path(str(Path(__file__).parents[2] / "examples/qmri_joint.py"))

    def stop_before_sampling(*args, **kwargs):
        raise RuntimeError("checkpoint unavailable")

    monkeypatch.setattr(example["DiffusionModel"], "from_checkpoint", stop_before_sampling)
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
    with pytest.raises(RuntimeError, match="checkpoint unavailable"):
        example["sample"](*arguments, device="cpu")
    np.savez(output / "dps.npz", t1=np.zeros((2, 8, 8)))
    saved = {path: path.read_bytes() for path in output.iterdir()}

    with pytest.raises(ValueError, match="use a new output directory"):
        example["sample"](*arguments, device="cpu", **changed)
    assert all(path.read_bytes() == contents for path, contents in saved.items())

    with pytest.raises(RuntimeError, match="checkpoint unavailable"):
        example["sample"](*arguments, device="cpu", method="daps")
    assert all(path.read_bytes() == contents for path, contents in saved.items())
