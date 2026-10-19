"""List per-HDF5 recipes, check YAML coverage, or run one explicit recipe."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


def load_catalog() -> dict:
    return json.loads(Path(__file__).with_name("catalog.json").read_text())


def yaml_files(path: Path) -> set[str]:
    """Read plain YAML without importing the training package or constructing objects."""
    import yaml

    names = set()

    def walk(value: object) -> None:
        if isinstance(value, dict):
            if "h5_path" in value:
                names.add(Path(value["h5_path"]).name)
            for entry in value.values():
                walk(entry)
        elif isinstance(value, list):
            for entry in value:
                walk(entry)

    walk(yaml.safe_load(path.read_text()))
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", nargs="?", help="Exact output filename, e.g. wand_t1map.h5")
    parser.add_argument("--yaml", type=Path, help="Check that every configured H5 filename has a recipe")
    parser.add_argument("--input-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--lut", type=Path, help="Complete WAND NPZ lookup table")
    parser.add_argument("--limit", type=int, default=1, help="Bounded subject count for the selected command")
    parser.add_argument("--execute", action="store_true", help="Run the printed recipe; default prints only")
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Choose an explicitly recorded raw reconstruction alternative instead of an archived-intermediate recipe",
    )
    args = parser.parse_args()
    catalog = load_catalog()
    recipes = {row["file"]: row for row in catalog["recipes"]}
    if len(recipes) != len(catalog["recipes"]):
        raise ValueError("Duplicate recipe filenames")
    if args.yaml:
        configured = yaml_files(args.yaml)
        missing = configured - set(recipes)
        print(
            json.dumps(
                {
                    "configured": len(configured),
                    "recipes": len(recipes),
                    "missing": sorted(missing),
                    "additional": sorted(set(recipes) - configured),
                },
                indent=2,
            )
        )
        if missing:
            raise SystemExit(1)
    if not args.file:
        if not args.yaml:
            for file, row in recipes.items():
                print(f"{file}\t{row['module']}\t{row['validation_status']}")
        return
    if args.file not in recipes:
        parser.error("Unknown output filename; run without a filename to list recipes")
    recipe = recipes[args.file]
    print(recipe["validation_note"])
    if args.raw:
        if "raw_recipe" not in recipe:
            parser.error("This entry has no separate raw reconstruction alternative")
        print(recipe.get("raw_validation_note", "Raw reproduction is not certified; inspect the recipe notes."))
        recipe = recipe["raw_recipe"]
    if args.limit <= 0:
        parser.error("limit must be positive")
    if args.execute and (args.input_root is None or args.output_dir is None):
        parser.error("--execute requires --input-root and --output-dir")
    substitutions = {
        "input": str(args.input_root or "/path/to/downloaded-input"),
        "output": str(args.output_dir or "/path/to/new-output"),
        "limit": str(args.limit),
        "lut": str(args.lut or "/path/to/complete-wand-lut.npz"),
    }
    if args.execute and any("{lut}" in value for value in recipe["arguments"]) and args.lut is None:
        parser.error("This WAND recipe also requires --lut")
    command = [sys.executable, "-m", recipe["module"], *(value.format_map(substitutions) for value in recipe["arguments"])]
    print(shlex.join(command), flush=True)
    if args.execute:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
