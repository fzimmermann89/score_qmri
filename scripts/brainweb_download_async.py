#!/usr/bin/env python3
# /// script
# dependencies = [
#   "aiohttp>=3.9.0",
#   "h5py>=3.8.0",
#   "numpy>=1.24.0",
# ]
# ///
"""Async Brainweb data downloader.

Downloads tissue class data for Brainweb subjects and saves as multi-subject HDF5.

Usage:
    uv run scripts/brainweb_download_async.py --out data/brainweb/brainweb.h5
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import io
import re
from pathlib import Path

import aiohttp
import h5py
import numpy as np

OVERVIEW_URL = "http://brainweb.bic.mni.mcgill.ca/brainweb/anatomic_normal_20.html"
URL_TEMPLATE = (
    "http://brainweb.bic.mni.mcgill.ca/cgi/brainweb1?"
    "do_download_alias=subject{subject}_{c}"
    "&format_value=raw_short"
    "&zip_value=gnuzip"
    "&download_for_real=%5BStart+download%21%5D"
)

ALL_CLASSES = ("bck", "skl", "gry", "wht", "csf", "mrw", "dura", "fat", "fat2", "mus", "m-s", "ves")
VERSION = 1
BRAINWEBSHAPE = (362, 434, 362)


async def fetch_available_subjects(session: aiohttp.ClientSession, timeout: float) -> list[str]:
    """Fetch list of available subject IDs from Brainweb overview page."""
    async with session.get(OVERVIEW_URL, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
        resp.raise_for_status()
        text = await resp.text()
    return re.findall(r"value=?\"?(\d+)\"?", text)


async def download_class(
    session: aiohttp.ClientSession,
    subject: str,
    class_name: str,
    semaphore: asyncio.Semaphore,
    timeout: float,
) -> tuple[str, bytes]:
    """Download and decompress a single tissue class file."""
    url = URL_TEMPLATE.format(subject=subject, c=class_name)
    async with semaphore, session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
        resp.raise_for_status()
        compressed = await resp.read()
    try:
        with gzip.open(io.BytesIO(compressed)) as f:
            decompressed = f.read()
    except OSError as exc:
        raise ValueError(f"Failed to decompress {class_name} for subject {subject}") from exc
    return class_name, decompressed


def normalize_classes(raw_data: dict[str, bytes]) -> np.ndarray:
    """Normalize tissue class data to fractional probabilities.

    Returns
    -------
    Array of shape (Z, Y, X, C) with C=11 classes (background removed), dtype uint8.
    """
    values: list[np.ndarray] = []
    expected_bytes = int(np.prod(BRAINWEBSHAPE)) * 2
    for c in ALL_CLASSES:
        raw = raw_data[c]
        if len(raw) != expected_bytes:
            raise ValueError(f"Unexpected data size for {c}: {len(raw)} bytes, expected {expected_bytes}")
        arr = np.frombuffer(raw, dtype=">u2").reshape(BRAINWEBSHAPE)
        arr = arr.astype(np.float32)
        # Per-class min clipping at slice 50
        min_val = arr[50].min()
        arr = np.clip(arr - min_val, 0, 4096)
        values.append(arr)

    # Sum across all classes
    sum_values = sum(values)

    # Remove background class
    bck_idx = ALL_CLASSES.index("bck")
    values.pop(bck_idx)

    # Normalize to sum=1 and convert to uint8
    result = []
    for arr in values:
        with np.errstate(divide="ignore", invalid="ignore"):
            normalized = np.divide(arr, sum_values, out=np.zeros_like(arr), where=sum_values != 0)
        normalized = (normalized * 255).astype(np.uint8)
        result.append(normalized)

    # Stack along last axis: (Z, Y, X, C)
    return np.stack(result, axis=-1)


def write_to_h5_group(h5_file: h5py.File, group_name: str, data: np.ndarray, subject: int) -> None:
    """Write processed data to an HDF5 group with LZF compression and axial-slice chunking."""
    classnames = [c for c in ALL_CLASSES if c != "bck"]
    z, y, x, c = data.shape
    chunk_size = (1, y, x, c)  # Axial slices

    group = h5_file.create_group(group_name)
    group.create_dataset(
        "classes",
        data=data,
        dtype=np.uint8,
        chunks=chunk_size,
        compression="lzf",
    )
    group.attrs["classnames"] = classnames
    group.attrs["subject"] = subject
    group.attrs["version"] = VERSION


async def download_subject_data(
    session: aiohttp.ClientSession,
    subject: int,
    workers: int,
    timeout: float,
) -> np.ndarray:
    """Download and normalize tissue class data for a subject."""
    subject_str = f"{subject:02d}"
    semaphore = asyncio.Semaphore(workers)
    tasks = [download_class(session, subject_str, c, semaphore, timeout) for c in ALL_CLASSES]

    print(f"Downloading {len(ALL_CLASSES)} tissue classes for subject {subject}...")
    results = await asyncio.gather(*tasks)

    raw_data = dict(results)
    print("Normalizing data...")
    return normalize_classes(raw_data)


async def download_all_subjects(
    output_path: Path,
    workers: int,
    timeout: float,
) -> None:
    """Download all available subjects to a single HDF5 file with groups."""
    async with aiohttp.ClientSession() as session:
        print("Fetching available subjects...")
        available = await fetch_available_subjects(session, timeout)

        subjects = sorted({int(s) for s in available})
        print(f"Found {len(subjects)} subjects: {subjects}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(str(output_path), "w") as f:
            f.attrs["version"] = VERSION
            f.attrs["num_subjects"] = len(subjects)

        for idx, subject in enumerate(subjects, start=1):
            try:
                print(f"\n{'=' * 60}")
                print(f"Processing subject {subject} ({idx}/{len(subjects)})")
                print(f"{'=' * 60}")
                normalized = await download_subject_data(session, subject, workers, timeout)

                group_name = f"subject_{subject:02d}"
                print(f"Writing {group_name} to {output_path}...")
                with h5py.File(str(output_path), "a") as f:
                    write_to_h5_group(f, group_name, normalized, subject)

            except Exception as e:
                print(f"Error downloading subject {subject}: {e}")
                continue


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Brainweb data.")
    parser.add_argument("--out", type=str, default="data/brainweb/brainweb.h5", help="Output HDF5 file path")
    parser.add_argument("--workers", type=int, default=4, help="Number of concurrent downloads")
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout in seconds")
    args = parser.parse_args()

    asyncio.run(download_all_subjects(Path(args.out), args.workers, args.timeout))


if __name__ == "__main__":
    main()
