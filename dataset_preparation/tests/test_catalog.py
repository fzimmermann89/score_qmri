"""Ensure every configured filename has an executable entry point."""

import importlib.util
from pathlib import Path

from dataset_preparation.catalog import load_catalog, yaml_files


def test_catalog_covers_configuration_and_wand_correction() -> None:
    catalog = load_catalog()
    rows = catalog["recipes"]
    names = {row["file"] for row in rows}
    assert len(rows) == len(names) == 69
    assert yaml_files(Path(__file__).parents[2] / "data.yaml") <= names
    assert {"wand_t1w.h5", "wand_t1map.h5"} <= names
    for row in rows:
        assert importlib.util.find_spec(row["module"]) is not None
        assert row["original_sources"]
        assert row["validation_status"] and row["validation_note"]
        assert row["file"] in row["also_writes"]
        assert len(row["reference_h5"]["sha256"]) == 64


def test_every_catalog_command_honors_explicit_work_limit() -> None:
    """A bounded catalog run must never expand into a whole-source conversion."""
    for row in load_catalog()["recipes"]:
        arguments = row["arguments"]
        assert "--limit" in arguments, row["file"]
        index = arguments.index("--limit")
        assert arguments[index + 1] == "{limit}", row["file"]
        if row["module"] == "dataset_preparation.delics_maps":
            assert "planes" in row["validation_note"]
