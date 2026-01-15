"""Mock dataset manager for when real data is unavailable."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .image_utils import render_images, render_mask_overlay
from .mock_dataset import MockDataset, create_mock_datasets


@dataclass(frozen=True)
class MockDatasetStructure:
    subject_keys: list[str]
    slices_per_subject: list[int]
    coils_per_subject: list[int] | None
    index_to_position: list[tuple[int, int, int | None]]
    position_to_index: dict[tuple[int, int, int | None], int]

    @property
    def subject_count(self) -> int:
        return len(self.subject_keys)

    @property
    def has_coils(self) -> bool:
        return self.coils_per_subject is not None


class MockDatasetManager:
    def __init__(self) -> None:
        self._datasets = create_mock_datasets()
        self._structures: dict[int, MockDatasetStructure] = {}

    def list_datasets(self) -> list[dict[str, Any]]:
        result = []
        for idx, ds in enumerate(self._datasets):
            result.append({
                "dataset_idx": idx,
                "total_samples": len(ds),
                "config": {"mock": True, "name": ds.name},
                "h5_path": "mock://phantom",
                "h5_filename": f"mock_{ds.kind.lower()}.h5",
                "kind": ds.kind,
                "type": "complex" if ds.is_complex else "magnitude",
                "mode": None,
                "scaling_factor": 1.0,
                "penumbra_width": 30,
                "schedule": {"type": None, "weight": None, "steps": None},
                "augmentation_config": None,
                "is_mock": True,
            })
        return result

    def random_index(self, dataset_idx: int) -> int:
        ds = self._datasets[dataset_idx]
        return random.randrange(len(ds))

    def get_sample(
        self,
        dataset_idx: int,
        sample_idx: int,
        augment: bool,
        seed: int | None,
        display_mode: str,
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> dict[str, Any]:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        ds = self._datasets[dataset_idx]
        sample = ds[sample_idx]
        image = sample["image"]
        penumbra = sample["penumbra_mask"]
        background = sample["background_mask"]

        is_complex = image.is_complex()
        image_left, image_right, left_label, right_label, colorbar_range, raw_left, raw_right = render_images(
            image,
            display_mode=display_mode,
            is_complex=is_complex,
            vmin=vmin,
            vmax=vmax,
        )

        penumbra_overlay = render_mask_overlay(penumbra, (255, 165, 0, 120))
        background_overlay = render_mask_overlay(background, (0, 128, 255, 90))

        stats = self._compute_stats(image, background)
        metadata = self._build_metadata(dataset_idx, sample_idx, ds, image)
        navigation = self._build_navigation(dataset_idx, sample_idx)

        return {
            "index": sample_idx,
            "total_samples": len(ds),
            "image_left_base64": image_left,
            "image_right_base64": image_right,
            "image_left_label": left_label,
            "image_right_label": right_label,
            "colorbar_range": colorbar_range,
            "penumbra_overlay_base64": penumbra_overlay,
            "background_overlay_base64": background_overlay,
            "stats": stats,
            "metadata": metadata,
            "navigation": navigation,
            "raw_data_left": raw_left,
            "raw_data_right": raw_right,
            "is_mock": True,
        }

    def get_sample_by_hierarchy(
        self,
        dataset_idx: int,
        subject_idx: int,
        slice_idx: int,
        coil_idx: int | None,
        augment: bool,
        seed: int | None,
        display_mode: str,
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> dict[str, Any]:
        structure = self._get_structure(dataset_idx)
        position = (subject_idx, slice_idx, None)
        if position not in structure.position_to_index:
            raise ValueError("Invalid hierarchy position")
        linear_idx = structure.position_to_index[position]
        return self.get_sample(dataset_idx, linear_idx, augment, seed, display_mode, vmin, vmax)

    def get_histogram(self, dataset_idx: int, sample_idx: int, augment: bool, seed: int | None) -> dict[str, Any]:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        ds = self._datasets[dataset_idx]
        sample = ds[sample_idx]
        image = sample["image"]

        values = image.abs().float().cpu().numpy().ravel() if image.is_complex() else image.float().cpu().numpy().ravel()
        values = values[np.isfinite(values)]
        if values.size == 0:
            bins = np.linspace(0.0, 1.0, 2)
            counts = np.zeros(1)
        else:
            counts, bin_edges = np.histogram(values, bins=128)
            bins = bin_edges.tolist()
        stats = self._compute_stats(image, sample["background_mask"])
        return {
            "bins": bins,
            "counts": counts.tolist(),
            "stats": stats,
        }

    def _get_structure(self, dataset_idx: int) -> MockDatasetStructure:
        if dataset_idx in self._structures:
            return self._structures[dataset_idx]
        ds = self._datasets[dataset_idx]
        structure = self._build_structure(ds)
        self._structures[dataset_idx] = structure
        return structure

    def _build_structure(self, ds: MockDataset) -> MockDatasetStructure:
        subject_keys = list(ds._subject_keys)
        slices_per_subject = [ds.slices_per_subject] * len(subject_keys)

        index_to_position: list[tuple[int, int, int | None]] = []
        position_to_index: dict[tuple[int, int, int | None], int] = {}

        for idx, (subject_key, slice_idx) in enumerate(ds._slice_index):
            subject_idx = subject_keys.index(subject_key)
            index_to_position.append((subject_idx, slice_idx, None))
            position_to_index[(subject_idx, slice_idx, None)] = idx

        return MockDatasetStructure(
            subject_keys=subject_keys,
            slices_per_subject=slices_per_subject,
            coils_per_subject=None,
            index_to_position=index_to_position,
            position_to_index=position_to_index,
        )

    def _compute_stats(self, image: torch.Tensor, background_mask: torch.Tensor) -> dict[str, float]:
        values = image.abs().float() if image.is_complex() else image.float()
        values = values[torch.isfinite(values)]
        if values.numel() == 0:
            return {"min": 0.0, "max": 0.0, "mean": 0.0, "variance": 0.0, "std": 0.0, "signal_fraction": 0.0}

        variance = torch.var(values, unbiased=False).item()
        std = float(np.sqrt(variance))
        signal_fraction = (~background_mask).float().mean().item()
        return {
            "min": values.min().item(),
            "max": values.max().item(),
            "mean": values.mean().item(),
            "variance": variance,
            "std": std,
            "signal_fraction": signal_fraction,
        }

    def _build_metadata(
        self,
        dataset_idx: int,
        sample_idx: int,
        ds: MockDataset,
        image: torch.Tensor,
    ) -> dict[str, Any]:
        structure = self._get_structure(dataset_idx)
        subject_idx, slice_idx, coil_idx = structure.index_to_position[sample_idx]

        return {
            "dataset_idx": dataset_idx,
            "h5_path": "mock://phantom",
            "h5_filename": f"mock_{ds.kind.lower()}.h5",
            "dataset_type": "complex" if ds.is_complex else "magnitude",
            "kind": ds.kind,
            "mode": None,
            "subject_key": structure.subject_keys[subject_idx],
            "subject_index": subject_idx,
            "slice_index": slice_idx,
            "coil_index": coil_idx,
            "is_complex": bool(image.is_complex()),
            "shape": list(image.shape[-2:]),
            "total_samples": len(ds),
            "is_mock": True,
        }

    def _build_navigation(self, dataset_idx: int, sample_idx: int) -> dict[str, Any]:
        structure = self._get_structure(dataset_idx)
        subject_idx, slice_idx, coil_idx = structure.index_to_position[sample_idx]
        slice_count = structure.slices_per_subject[subject_idx]

        ds = self._datasets[dataset_idx]

        return {
            "dataset_type": "flat",
            "linear_index": sample_idx,
            "total_samples": len(ds),
            "subject_count": structure.subject_count,
            "current_subject_index": subject_idx,
            "current_subject_key": structure.subject_keys[subject_idx],
            "slice_count": slice_count,
            "current_slice_index": slice_idx,
            "coil_count": None,
            "current_coil_index": None,
            "can_next_coil": False,
            "can_prev_coil": False,
            "can_next_slice": slice_idx + 1 < slice_count,
            "can_prev_slice": slice_idx > 0,
            "can_next_subject": subject_idx + 1 < structure.subject_count,
            "can_prev_subject": subject_idx > 0,
        }
