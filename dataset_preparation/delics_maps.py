"""Build delics_{t1map,t2map,complexm0}.h5 from qmaps NPZ files.
Get datasets/Deli-CS/qmaps from https://github.com/p-mayo/mrf-diph
and pass that directory as --input-root. Keep delics_order.json next to this
script to retain the checked 360-plane order.
Run: python delics_maps.py --input-root QMAPS --output-dir NEW."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import scipy.ndimage as ndi
import torch


def exclusive_h5(path: str | Path) -> h5py.File:
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return h5py.File(path, "x")


def write_dataset(handle: h5py.Group, name: str, array: np.ndarray) -> h5py.Dataset:
    return handle.create_dataset(name, data=array, chunks=(1, *array.shape[1:]), compression="lzf", shuffle=True)


def clean_mask_2d(mask: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Original opening, closing, dilation, hole fill and largest component."""
    structure = ndi.generate_binary_structure(2, 1)
    result = ndi.binary_opening(np.asarray(mask).astype(bool), structure=structure)
    result = ndi.binary_closing(result, structure=structure, iterations=1)
    result = ndi.binary_dilation(result, structure=structure, iterations=2)
    result = ndi.binary_fill_holes(result)
    labels, n_labels = ndi.label(result, structure=structure)
    if n_labels:
        result = labels == (np.bincount(labels.ravel())[1:].argmax() + 1)
    return torch.tensor(result, dtype=torch.bool)


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
    paths = [args.input_root / name for name in json.loads(args.order_manifest.read_text())]
    if any(not path.resolve().is_relative_to(args.input_root.resolve()) for path in paths):
        parser.error("Order manifest paths must stay within input-root")
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
