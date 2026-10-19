"""The smoke CLI must accept its documented multi-checkpoint invocation."""

import runpy
from pathlib import Path


def test_multiple_checkpoint_and_training_paths():
    script = Path(__file__).resolve().parents[2] / "scripts" / "reconstruct_smoke.py"
    parse = runpy.run_path(str(script))["parse_arguments"]
    args = parse(
        [
            "--checkpoint",
            "/tmp/first.ckpt",
            "/tmp/second.ckpt",
            "--training-h5",
            "/tmp/train.h5",
            "/tmp/train2.h5",
            "--h5",
            "/tmp/case.h5",
            "--output",
            "/tmp/result",
        ]
    )
    assert args.checkpoint == [Path("/tmp/first.ckpt"), Path("/tmp/second.ckpt")]
    assert args.training_h5 == [Path("/tmp/train.h5"), Path("/tmp/train2.h5")]
    assert args.h5 == Path("/tmp/case.h5")
    assert args.output == Path("/tmp/result")
