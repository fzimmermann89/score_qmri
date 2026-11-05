"""Annotation storage for dataset debug viewer."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

FieldType = Literal["bool", "number", "text"]


@dataclass(frozen=True)
class AnnotationField:
    """Definition of a dataset-level annotation field."""

    id: str
    label: str
    field_type: FieldType


class AnnotationStore:
    """Persist annotation fields and dataset values to JSON."""

    def __init__(self, storage_dir: Path, config_path: Path) -> None:
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.storage_dir / "annotations.json"
        self.config_path = config_path
        self._data = self._load()

    def get_fields(self) -> list[AnnotationField]:
        return [AnnotationField(**field) for field in self._data["fields"]]

    def add_field(self, field: AnnotationField) -> None:
        if any(existing["id"] == field.id for existing in self._data["fields"]):
            raise ValueError(f"Field '{field.id}' already exists")
        self._data["fields"].append({"id": field.id, "label": field.label, "field_type": field.field_type})
        self._save()

    def remove_field(self, field_id: str) -> None:
        self._data["fields"] = [field for field in self._data["fields"] if field["id"] != field_id]
        for dataset_id, values in self._data["datasets"].items():
            if field_id in values:
                values.pop(field_id)
            self._data["datasets"][dataset_id] = values
        self._save()

    def get_dataset_annotations(self, dataset_idx: int) -> dict[str, Any]:
        return dict(self._data["datasets"].get(str(dataset_idx), {}))

    def set_dataset_annotations(self, dataset_idx: int, values: dict[str, Any]) -> None:
        allowed_fields = {field.id for field in self.get_fields()}
        filtered = {key: value for key, value in values.items() if key in allowed_fields}
        self._data["datasets"][str(dataset_idx)] = filtered
        self._save()

    def export(self) -> dict[str, Any]:
        return dict(self._data)

    def _load(self) -> dict[str, Any]:
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            data.setdefault("fields", [])
            data.setdefault("datasets", {})
            data.setdefault("metadata", {})
            return data

        return {
            "fields": [],
            "datasets": {},
            "metadata": {
                "config_path": str(self.config_path),
                "created_at": self._timestamp(),
                "last_modified": self._timestamp(),
            },
        }

    def _save(self) -> None:
        self._data.setdefault("metadata", {})
        self._data["metadata"]["config_path"] = str(self.config_path)
        self._data["metadata"]["last_modified"] = self._timestamp()
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()
