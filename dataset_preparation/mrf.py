"""Prepare Zenodo8183344 NIfTI and NIST MRF DICOM maps on the CPU.

Zenodo pipeline: datacheck/v10.py. NIST T1/T2: IPython session215
lines156/163; M0: datacheck/v11.py. NIST intentionally retains the first
five original directory-order filenames before sorting, matching the source.
Zenodo defaults to the canonical curated slice filter; --stage prepared
retains original pre-HPC slices. NIST arrays are unchanged in either stage.
Fresh downloads can enumerate differently; use --dicom-order-manifest for
an explicitly recorded list if exact historical reproduction is required.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from contextlib import ExitStack
from pathlib import Path

import torch

from .clean_h5 import retained_indices
from .common import exclusive_h5, write_dataset
from .openneuro import centered, clean_mask_2d, first, load_nifti

MODALITIES = {"zenodo8183344": ("t1map", "t2map", "t1w", "t2w"), "nist": ("t1map", "t2map", "m0map")}


def zenodo_subject(root: Path, which: str, modalities: set[str]) -> Iterator[tuple[str, str, torch.Tensor]]:
    """Original v10 quantitative seconds and independent image masks/crops."""
    name = "_".join([*root.parts[-2:], which])
    mprage, _ = load_nifti(first(root, f"*{which}*mprage.nii"))
    mask = torch.stack([clean_mask_2d(m > 10) for m in mprage])
    for modality in ("t1map", "t2map"):
        if modality in modalities:
            image, _ = load_nifti(first(root, f"*{which}*{modality[:2]}.nii"))
            yield modality, name, (image * mask.float() / 1000)[7:-5].float()
    if "t1w" in modalities:
        yield "t1w", name, (mprage * mask.float() / mprage.max())[3:-5].float()
    if "t2w" in modalities:
        image, _ = load_nifti(first(root, f"*{which}*tse.nii"))
        mask = torch.stack([clean_mask_2d(m > 50) for m in image])
        yield "t2w", name, (image * mask.float() / image.max())[3:-5].float()


def nist_series(root: Path, modality: str, dicom_order: dict[str, list[str]] | None = None) -> torch.Tensor:
    """Read the original five DICOM candidates and sort by SliceLocation."""
    import pydicom

    if dicom_order is None:
        files = sorted(list(root.rglob("*.IMA"))[:5])
    else:
        name = "_".join(root.parts[-2:])
        if name not in dicom_order:
            raise ValueError(f"No DICOM filename order for {name}")
        files = [root / filename for filename in dicom_order[name]]
        if any(not file.is_relative_to(root) or ".." in file.parts for file in files):
            raise ValueError("DICOM manifest paths must stay within the series root")
    images = []
    for path in files:
        dcm = pydicom.dcmread(path)
        image = torch.tensor(dcm.pixel_array, dtype=torch.float32)
        mask = clean_mask_2d(image > 0) * (image > 0)
        if mask.sum() < 2000:
            continue
        images.append((float(dcm.SliceLocation), image * mask / 1000))
    if not images:
        raise ValueError(f"No retained slices in {root}")
    image = torch.stack([x[1] for x in sorted(images, key=lambda x: x[0])])
    image = centered(image, (image.shape[0], 224, 224))
    if modality == "m0map":
        image /= image.max()
    return image


def select_roots(paths: Iterator[Path], subjects: list[str] | None, limit: int | None) -> list[Path]:
    """Select subject ancestors as well as series/session names."""
    if limit is not None and limit < 1:
        raise ValueError("--limit must be positive")
    roots = sorted(set(paths))
    if subjects:
        roots = [p for p in roots if set(p.parts) & set(subjects) or p.stem in subjects]
    return roots[:limit] if limit is not None else roots


def prepare(
    dataset: str,
    input_root: Path,
    output_dir: Path,
    *,
    subjects: list[str] | None = None,
    limit: int | None = None,
    modalities: list[str] | None = None,
    dicom_order_manifest: Path | None = None,
    stage: str = "curated",
) -> None:
    """Create new output files; never rewrite raw inputs or existing HDF5."""
    if stage not in {"prepared", "curated"}:
        raise ValueError("stage must be prepared or curated")
    wanted = set(modalities or MODALITIES[dataset])
    if wanted - set(MODALITIES[dataset]):
        raise ValueError(f"Recovered modalities: {MODALITIES[dataset]}")
    if not input_root.is_dir():
        raise FileNotFoundError(input_root)
    order = json.loads(dicom_order_manifest.read_text()) if dicom_order_manifest else None
    prefix = "mrf_zenodo8183344" if dataset == "zenodo8183344" else "mrf_nist"
    with ExitStack() as stack:
        outputs = {m: stack.enter_context(exclusive_h5(output_dir / f"{prefix}_{m}.h5")) for m in sorted(wanted)}
        if dataset == "zenodo8183344":
            roots = select_roots(input_root.rglob("MR?"), subjects, limit)
            if not roots:
                raise ValueError(f"No selected MR? sessions in {input_root}")
            for root in roots:
                for which in ("original", "reposition"):
                    for modality, name, image in zenodo_subject(root, which, wanted):
                        array = image.numpy()
                        if stage == "curated":
                            array = array[retained_indices(array)]
                        write_dataset(outputs[modality], name, array)
        else:
            for modality in sorted(wanted):
                tag = {"t1map": "T1", "t2map": "T2", "m0map": "M0"}[modality]
                roots = select_roots((p for p in input_root.rglob(f"*{tag}*") if p.is_dir()), subjects, limit)
                if not roots:
                    raise ValueError(f"No selected *{tag}* series in {input_root}")
                for root in roots:
                    image = nist_series(root, modality, order)
                    write_dataset(outputs[modality], "_".join(root.parts[-2:]), image.numpy())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(MODALITIES))
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--modalities", nargs="+")
    parser.add_argument("--dicom-order-manifest", type=Path)
    parser.add_argument(
        "--stage",
        choices=["prepared", "curated"],
        default="curated",
        help="Zenodo: original pre-HPC arrays or canonical slice-filtered arrays; NIST stays unchanged",
    )
    torch.set_num_threads(1)
    prepare(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
