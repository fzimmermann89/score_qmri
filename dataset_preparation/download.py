"""Explicit downloads to a new destination; never run during preparation.

OpenNeuro uses DataLad/git-annex, so a Git clone alone does not fetch images.
Archive mode saves the original archive without extracting or deleting it.
Rockland mode uses the original aws_links.csv file list (anonymous S3).
Private/expired fastMRI signed URLs are intentionally not embedded here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path


def download_file(url: str, output: Path, sha256: str | None = None, max_bytes: int | None = None) -> dict[str, str | int]:
    """Stream an explicitly requested public URL into an exclusive new file."""
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if urllib.parse.urlsplit(url).scheme != "https":
        raise ValueError("Downloads require HTTPS")
    if max_bytes is not None and max_bytes <= 0:
        raise ValueError("max-bytes must be positive")
    output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    count = 0
    with urllib.request.urlopen(url, timeout=120) as response, output.open("xb") as handle:
        length = response.headers.get("Content-Length")
        if max_bytes is not None and length is not None and int(length) > max_bytes:
            raise ValueError("Download exceeds the requested byte budget")
        while block := response.read(1024 * 1024):
            if max_bytes is not None and count + len(block) > max_bytes:
                raise ValueError("Download exceeds the requested byte budget; partial new file retained")
            handle.write(block)
            digest.update(block)
            count += len(block)
    actual = digest.hexdigest()
    if sha256 and actual.lower() != sha256.lower():
        raise ValueError(f"SHA-256 mismatch: {actual}; new file retained for inspection")
    return {"path": str(output), "bytes": count, "sha256": actual}


def rockland_plan(manifest: Path, subjects: list[str], modalities: list[str], limit: int | None = None) -> list[tuple[str, Path]]:
    """Select anatomical rows from the original Rockland CSV, preserving keys."""
    prefix = "data/Projects/RocklandSample/RawDataBIDSLatest/"
    result = []
    with manifest.open(newline="") as handle:
        for row in csv.DictReader(handle):
            source = row["filepath"]
            key = source.removeprefix("s3://fcp-indi/")
            if not key.startswith(prefix):
                continue
            relative = Path(key.removeprefix(prefix))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe manifest path")
            if not any(s in relative.parts or "sub-" + s in relative.parts for s in subjects):
                continue
            if "/anat/" not in "/" + str(relative) or not any(m in relative.name for m in modalities):
                continue
            result.append(("https://fcp-indi.s3.amazonaws.com/" + urllib.parse.quote(key), relative))
    result = sorted(set(result))
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        result = result[:limit]
    if not result:
        raise ValueError("No matching Rockland files")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    archive = modes.add_parser("archive", help="Save an explicitly supplied HTTPS archive, without extraction")
    archive.add_argument("--url", required=True)
    archive.add_argument("--output", type=Path, required=True)
    archive.add_argument("--sha256", help="Expected archive SHA-256 from a trusted source")
    archive.add_argument("--max-bytes", type=int, required=True)
    bids = modes.add_parser("openneuro", help="Clone into a new path and optionally fetch selected annex files")
    bids.add_argument(
        "--dataset", required=True, choices=["ds000221", "ds002685", "ds002785", "ds002790", "ds003967", "ds004611", "ds004889", "ds005851"]
    )
    bids.add_argument("--output", type=Path, required=True)
    bids.add_argument("--revision", required=True, help="Explicit Git revision/tag for the source dataset")
    bids.add_argument("--get", nargs="+", default=[], help="Relative file/subject paths to fetch; no implicit full download")
    rockland = modes.add_parser("rockland", help="Download selected anatomy from the historical public S3 CSV")
    rockland.add_argument("--manifest", type=Path, required=True)
    rockland.add_argument("--subjects", nargs="+", required=True)
    rockland.add_argument("--modalities", nargs="+", choices=["T1w", "T2w", "FLAIR"], required=True)
    rockland.add_argument("--limit", type=int)
    rockland.add_argument("--output", type=Path, required=True)
    rockland.add_argument("--max-bytes-per-file", type=int, required=True)
    rockland.add_argument("--download", action="store_true", help="Without this flag, print the plan only")
    args = parser.parse_args()
    if args.mode == "archive":
        print(json.dumps(download_file(args.url, args.output, args.sha256, args.max_bytes), indent=2))
    elif args.mode == "openneuro":
        if args.output.exists() or args.output.is_symlink():
            raise FileExistsError(args.output)
        if any(Path(p).is_absolute() or ".." in Path(p).parts or p.startswith("-") for p in args.get):
            parser.error("--get paths must be relative paths within the new dataset")
        subprocess.run(["datalad", "clone", f"https://github.com/OpenNeuroDatasets/{args.dataset}", str(args.output)], check=True)
        subprocess.run(["git", "-C", str(args.output), "checkout", args.revision], check=True)
        if args.get:
            subprocess.run(["datalad", "get", "-d", str(args.output), "--", *(str(args.output / p) for p in args.get)], check=True)
    else:
        plan = rockland_plan(args.manifest, args.subjects, args.modalities, args.limit)
        print(json.dumps([{"relative_path": str(p), "url": u} for u, p in plan], indent=2))
        if args.download:
            if args.output.exists() or args.output.is_symlink():
                raise FileExistsError(args.output)
            args.output.mkdir(parents=True)
            for url, relative in plan:
                print(json.dumps(download_file(url, args.output / relative, max_bytes=args.max_bytes_per_file)))


if __name__ == "__main__":
    main()
