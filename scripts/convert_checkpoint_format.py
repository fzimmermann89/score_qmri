"""Convert a trusted checkpoint to portable metadata without overwriting inputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from score_qmri.utils.checkpoint import portable_checkpoint


def convert_checkpoint(input_path: Path, output_path: Path) -> None:
    """Preserve model/EMA/optimizer states and normalize recorded configuration."""
    if input_path.resolve() == output_path.resolve() or output_path.exists():
        raise FileExistsError(f"Choose a new output file; refusing to overwrite {output_path}")
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False, mmap=True)
    converted = portable_checkpoint(checkpoint)
    with output_path.open("xb") as output:
        torch.save(converted, output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Trusted checkpoint created by this project")
    parser.add_argument("--output", type=Path, help="New output path; existing files are refused")
    args = parser.parse_args()
    output = args.output or args.input.with_suffix(f".converted{args.input.suffix or '.ckpt'}")
    convert_checkpoint(args.input, output)


if __name__ == "__main__":
    main()
