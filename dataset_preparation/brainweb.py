"""Build brainweb.h5 using the selected newer little-endian recipe.
Use pack with the original newer sXX.h5 files, or raw with cached
sXX/CLASS.gz raw-short tissue files from https://brainweb.bic.mni.mcgill.ca/.
raw --download fetches missing gzip classes. Keep the original uint16 baseline
subtraction, including its rare underflow, to match the selected H5.
Run: python brainweb.py pack --input-root SUBJECTS --output NEW.h5."""

from __future__ import annotations

import argparse
import gzip
import os
import shutil
import tempfile
import urllib.request
from collections.abc import Iterator, Sequence
from contextlib import ExitStack
from pathlib import Path
from typing import BinaryIO, cast

import h5py
import numpy as np

CLASSES = ("bck", "skl", "gry", "wht", "csf", "mrw", "dura", "fat", "fat2", "mus", "m-s", "ves")
SUBJECTS = ("04", "05", "06", "18", "20", "38", "41", "42", "43", "44", "45", "46", "47", "48", "49", "50", "51", "52", "53", "54")
SHAPE = (362, 434, 362)
URL = (
    "https://brainweb.bic.mni.mcgill.ca/cgi/brainweb1?"
    "do_download_alias=subject{subject}_{tissue}&format_value=raw_short"
    "&zip_value=gnuzip&download_for_real=%5BStart+download%21%5D"
)


def exclusive_h5(path: str | Path) -> h5py.File:
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return h5py.File(str(path), "w-")


def normalize_slab(values: Sequence[np.ndarray], baselines: Sequence[int]) -> np.ndarray:
    """Normalize one spatial slab using the original whole-volume baseline."""
    corrected = []
    for values_i, baseline in zip(values, baselines, strict=True):
        corrected.append(np.clip(values_i - np.asarray(baseline, dtype=values_i.dtype), 0, 4096))
    # Python sum preserves the original uint16 accumulation. 12*4096 < 65536.
    denominator = sum(corrected)
    normalized = [np.divide(x, denominator, out=np.zeros_like(x, dtype=float), where=denominator != 0) for x in corrected[1:]]
    return np.stack([(x * 255).astype(np.uint8) for x in normalized], axis=-1)


def _read_plane(stream: BinaryIO | gzip.GzipFile, shape: tuple[int, int, int], z: int) -> np.ndarray:
    size = shape[1] * shape[2] * 2
    stream.seek(z * size)
    data = stream.read(size)
    if len(data) != size:
        raise ValueError(f"Truncated raw gzip at slice {z}.")
    plane = np.frombuffer(data, dtype="<u2").reshape(shape[1:])
    if plane.max(initial=0) > 4095:
        raise ValueError("Raw values exceed 4095; check the source and byte order.")
    return cast(np.ndarray, plane)


def iter_raw_slabs(
    paths: Sequence[Path],
    shape: tuple[int, int, int] = SHAPE,
    baseline_slice: int = 50,
) -> Iterator[tuple[int, np.ndarray]]:
    """Stream gzip classes with bounded memory and validate the raw lengths."""
    if len(paths) != len(CLASSES) or not 0 <= baseline_slice < shape[0]:
        raise ValueError("Invalid class count or baseline slice.")
    with ExitStack() as stack:
        streams = [stack.enter_context(gzip.open(path, "rb")) for path in paths]
        baselines = [int(_read_plane(stream, shape, baseline_slice).min()) for stream in streams]
        for stream in streams:
            stream.seek(0)
        for start in range(shape[0]):
            values = [_read_plane(stream, shape, start)[None] for stream in streams]
            yield start, normalize_slab(values, baselines)
        if any(stream.read(1) for stream in streams):
            raise ValueError("Raw gzip has more voxels than the expected Brainweb shape.")


def _create_classes(group: h5py.Group, shape: tuple[int, ...]) -> h5py.Dataset:
    return group.create_dataset(
        "classes", shape=shape, dtype=np.uint8, chunks=(1, min(64, shape[1]), min(64, shape[2]), shape[3]), compression="lzf", shuffle=True
    )


def pack_subjects(input_root: Path, output: Path, subjects: Sequence[str]) -> None:
    """Losslessly pack flat newer subject files into the training schema."""
    if not subjects or len(set(subjects)) != len(subjects):
        raise ValueError("Select distinct subjects.")
    sources = [input_root / f"s{subject}.h5" for subject in subjects]
    for source, subject in zip(sources, subjects, strict=True):
        with h5py.File(str(source), "r") as file:
            data = cast(h5py.Dataset, file["classes"])
            if data.shape != (*SHAPE, len(CLASSES) - 1) or data.dtype != np.uint8:
                raise ValueError(f"Unexpected newer Brainweb schema: {source}")
            if tuple(cast(Sequence[str], file.attrs["classnames"])) != CLASSES[1:] or int(cast(int, file.attrs["subject"])) != int(subject):
                raise ValueError(f"Unexpected class names/subject: {source}")
    with exclusive_h5(output) as target:
        for source, subject in zip(sources, subjects, strict=True):
            with h5py.File(str(source), "r") as file:
                group = target.create_group(f"subject_{subject}")
                for name, value in file.attrs.items():
                    group.attrs[name] = value
                src = cast(h5py.Dataset, file["classes"])
                dst = _create_classes(group, src.shape)
                for name, value in src.attrs.items():
                    dst.attrs[name] = value
                for start in range(src.shape[0]):
                    dst[start] = src[start]


def _cache_paths(root: Path, subject: str, download: bool) -> list[Path]:
    folder = root / f"s{subject}"
    if download:
        folder.mkdir(parents=True, exist_ok=True)
    paths = [folder / f"{tissue}.gz" for tissue in CLASSES]
    for path, tissue in zip(paths, CLASSES, strict=True):
        if not path.exists() and download:
            with tempfile.NamedTemporaryFile(dir=folder, prefix=path.name + ".", suffix=".part") as target:
                with urllib.request.urlopen(URL.format(subject=subject, tissue=tissue), timeout=120) as response:
                    shutil.copyfileobj(response, target, length=1024 * 1024)
                target.flush()
                target.seek(0)
                if target.read(2) != b"\x1f\x8b":
                    raise ValueError(f"Not a gzip download: {path}")
                os.link(target.name, path)
        if not path.is_file():
            raise FileNotFoundError(f"{path}: provide cached raw classes or explicitly pass --download.")
    return paths


def prepare_raw(cache: Path, output: Path, subjects: Sequence[str], download: bool = False) -> None:
    """Prepare a packed H5 from raw gzip inputs, downloading only when requested."""
    if not subjects or len(set(subjects)) != len(subjects):
        raise ValueError("Select distinct subjects.")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    paths = [_cache_paths(cache, subject, download) for subject in subjects]
    with exclusive_h5(output) as target:
        for subject, class_paths in zip(subjects, paths, strict=True):
            group = target.create_group(f"subject_{subject}")
            group.attrs["classnames"] = CLASSES[1:]
            group.attrs["subject"] = int(subject)
            group.attrs["version"] = 1
            data = _create_classes(group, (*SHAPE, len(CLASSES) - 1))
            for start, values in iter_raw_slabs(class_paths):
                data[start : start + len(values)] = values


def main(argv: Sequence[str] | None = None) -> None:
    """Run an explicitly selected packing or raw preparation operation."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("pack", "raw"))
    parser.add_argument("--input-root", type=Path, required=True, help="Flat subject H5 directory for pack; gzip cache for raw.")
    parser.add_argument("--output", type=Path, required=True, help="New H5 path; existing paths are refused.")
    parser.add_argument("--subjects", nargs="+", default=list(SUBJECTS), choices=SUBJECTS)
    parser.add_argument("--download", action="store_true", help="Explicitly allow missing raw gzip downloads, sequentially.")
    args = parser.parse_args(argv)
    subjects = [subject for subject in SUBJECTS if subject in args.subjects]
    if args.mode == "pack":
        if args.download:
            parser.error("pack accepts --input-root and preserves existing normalization; raw options are invalid.")
        pack_subjects(args.input_root, args.output, subjects)
    else:
        prepare_raw(args.input_root, args.output, subjects, args.download)


if __name__ == "__main__":
    main()
