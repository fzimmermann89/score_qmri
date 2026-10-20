"""Download selected OpenNeuro annex contents or NKI-Rockland anatomy.
OpenNeuro requires separately installed DataLad/git-annex. Rockland requires
the original aws_links.csv from the NKI-Rockland download instructions at
https://fcon_1000.projects.nitrc.org/indi/enhanced/. Use a new --output path.
Run: python download.py --help."""

import argparse
import csv
import shutil
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path


def download_file(url: str, output: Path) -> None:
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response, output.open("xb") as target:
        shutil.copyfileobj(response, target, length=1024 * 1024)


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
    bids = modes.add_parser("openneuro")
    bids.add_argument(
        "--dataset", required=True, choices=["ds000221", "ds002685", "ds002785", "ds002790", "ds003967", "ds004611", "ds004889", "ds005851"]
    )
    bids.add_argument("--output", type=Path, required=True)
    bids.add_argument("--revision", required=True, help="Source dataset Git revision/tag")
    bids.add_argument("--get", nargs="+", required=True, help="Relative annex file/subject paths to download")
    rockland = modes.add_parser("rockland")
    rockland.add_argument("--manifest", type=Path, required=True)
    rockland.add_argument("--subjects", nargs="+", required=True)
    rockland.add_argument("--modalities", nargs="+", choices=["T1w", "T2w", "FLAIR"], required=True)
    rockland.add_argument("--limit", type=int)
    rockland.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    if args.mode == "openneuro":
        if any(Path(p).is_absolute() or ".." in Path(p).parts or p.startswith("-") for p in args.get):
            parser.error("--get paths must be relative to the new dataset")
        subprocess.run(["datalad", "clone", f"https://github.com/OpenNeuroDatasets/{args.dataset}", str(args.output)], check=True)
        subprocess.run(["git", "-C", str(args.output), "checkout", args.revision], check=True)
        subprocess.run(["datalad", "get", "-d", str(args.output), "--", *(str(args.output / p) for p in args.get)], check=True)
    else:
        for url, relative in rockland_plan(args.manifest, args.subjects, args.modalities, args.limit):
            print(relative, flush=True)
            download_file(url, args.output / relative)


if __name__ == "__main__":
    main()
