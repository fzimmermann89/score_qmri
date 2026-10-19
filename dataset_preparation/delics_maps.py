"""Prepare Deli-CS T1/T2/complex-M0 from the original qmaps NPZ files.

Source: datacheck/v15.py. The original used filesystem enumeration order.
The bundled relative-path manifest records that order, verified against all
360 original target planes and all three output maps. Override it only for a
different input revision; alphabetical order does not reproduce this dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .common import exclusive_h5, write_dataset
from .nifti import clean_mask_2d


def plane(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        t1, t2, real, imag = torch.tensor(archive["qmaps"])
    m0 = (real + 1j * imag) / 0.03
    mask = clean_mask_2d(m0.abs() > 0.1)
    return {
        "t1map": (t1 * mask).to(torch.float32).numpy(),
        "t2map": (t2 * mask).to(torch.float32).numpy(),
        "complexm0": (m0 * mask).to(torch.complex64).numpy(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--order-manifest",
        type=Path,
        default=Path(__file__).with_name("delics_order.json"),
        help="JSON list of paths relative to input-root",
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.order_manifest:
        paths = [args.input_root / name for name in json.loads(args.order_manifest.read_text())]
        if any(not path.resolve().is_relative_to(args.input_root.resolve()) for path in paths):
            parser.error("Order manifest paths must stay within input-root")
    else:
        paths = sorted(p for p in args.input_root.rglob("qmap*.npz") if p.is_file())
    if args.limit is not None:
        if args.limit <= 0:
            parser.error("limit must be positive")
        paths = paths[: args.limit]
    if not paths:
        parser.error("No qmap NPZ files found")
    # The source writes one stacked 'data' dataset, not a subject per NPZ.
    arrays = [plane(path) for path in paths]
    for modality in ("t1map", "t2map", "complexm0"):
        with exclusive_h5(args.output_dir / f"delics_{modality}.h5") as handle:
            write_dataset(handle, "data", np.stack([data[modality] for data in arrays]))


if __name__ == "__main__":
    main()
