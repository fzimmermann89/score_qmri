"""Dataset manager for debug viewer."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from score_qmri.datasets import ComplexImageDataset, FlatH5Dataset, MagnitudeImageDataset, MappingDataset

from .config_loader import LoadedConfig, clone_without_augmentation
from .image_utils import render_images, render_mask_overlay


@dataclass(frozen=True)
class DatasetStructure:
    """Hierarchical structure for navigation."""

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


class DatasetManager:
    """Load datasets and provide sampling utilities."""

    def __init__(self, loaded: LoadedConfig) -> None:
        self.configs = loaded.dataset_configs
        self.raw_configs = loaded.raw_dataset_configs
        self._datasets: dict[int, ComplexImageDataset | FlatH5Dataset] = {}
        self._datasets_no_aug: dict[int, ComplexImageDataset | FlatH5Dataset] = {}
        self._structures: dict[int, DatasetStructure] = {}
        self._summaries: dict[int, dict[str, Any]] = {}

    def list_datasets(self) -> list[dict[str, Any]]:
        return [self._get_dataset_summary(i) for i in range(len(self.configs))]

    def random_index(self, dataset_idx: int) -> int:
        dataset = self._get_dataset(dataset_idx, augment=False)
        return random.randrange(len(dataset))

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
        dataset = self._get_dataset(dataset_idx, augment=augment)
        self._seed_rng(seed)
        sample = dataset[sample_idx]
        image = sample["image"]
        penumbra = sample["penumbra_mask"]
        background = sample["background_mask"]

        is_complex = bool(image.is_complex())
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
        metadata = self._build_metadata(dataset_idx, sample_idx, dataset, image)
        navigation = self._build_navigation(dataset_idx, sample_idx)

        return {
            "index": sample_idx,
            "total_samples": len(dataset),
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
        position = (subject_idx, slice_idx, coil_idx if structure.has_coils else None)
        if position not in structure.position_to_index:
            raise ValueError("Invalid hierarchy position")
        linear_idx = structure.position_to_index[position]
        return self.get_sample(dataset_idx, linear_idx, augment, seed, display_mode, vmin, vmax)

    def get_batch_data(
        self,
        dataset_idx: int,
        indices: list[int],
        augment: bool,
        seed: int | None,
        display_mode: str,
    ) -> dict[str, Any]:
        dataset = self._get_dataset(dataset_idx, augment=augment)
        self._seed_rng(seed)

        if not indices:
            return {
                "samples": [],
                "global_stats": {
                    "min": 0.0,
                    "max": 0.0,
                    "mean": 0.0,
                    "std": 0.0,
                    "variance": 0.0,
                    "signal_fraction": 0.0,
                    "p99": 0.0,
                },
                "global_histogram": {
                    "bins": [],
                    "counts": [],
                    "stats": {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0},
                },
                "global_range": (0.0, 1.0),
            }

        images_to_stack = []
        background_masks = []

        for idx in indices:
            sample = dataset[idx]
            img = sample["image"]
            images_to_stack.append(img)
            background_masks.append(sample["background_mask"])

        batch_tensor = torch.stack(images_to_stack)
        hist_values = batch_tensor.abs().float() if batch_tensor.is_complex() else batch_tensor.float()
        flat_tensor = hist_values.flatten()
        flat_tensor = flat_tensor[torch.isfinite(flat_tensor)]

        signal_fraction = 0.0
        if background_masks:
            mask_tensor = torch.stack(background_masks)
            signal_fraction = (~mask_tensor).float().mean().item()

        if flat_tensor.numel() == 0:
            hist_bins = np.linspace(0.0, 1.0, 2).tolist()
            hist_counts = [0]
            stats = {
                "min": 0.0,
                "max": 0.0,
                "mean": 0.0,
                "std": 0.0,
                "variance": 0.0,
                "signal_fraction": signal_fraction,
                "p99": 0.0,
            }
            render_min = 0.0
            render_max = 1.0
        else:
            variance = torch.var(flat_tensor, unbiased=False).item()
            stats = {
                "min": flat_tensor.min().item(),
                "max": flat_tensor.max().item(),
                "mean": flat_tensor.mean().item(),
                "std": torch.sqrt(torch.tensor(variance)).item(),
                "variance": variance,
                "signal_fraction": signal_fraction,
                "p99": torch.quantile(flat_tensor, 0.99).item(),
            }

            counts, bin_edges = np.histogram(flat_tensor.cpu().numpy(), bins=128)
            hist_bins = bin_edges.tolist()
            hist_counts = counts.tolist()

            if batch_tensor.is_complex() and display_mode == "real_imag":
                max_abs = torch.maximum(batch_tensor.real.abs(), batch_tensor.imag.abs()).max().item()
                render_min = -max_abs
                render_max = max_abs
            elif batch_tensor.is_complex():
                render_min = 0.0
                render_max = flat_tensor.max().item()
            else:
                render_min = stats["min"]
                render_max = stats["max"]

        rendered_samples = []
        for i, img in enumerate(images_to_stack):
            is_complex = bool(img.is_complex())
            left_img, right_img, left_lbl, right_lbl, ranges, _, _ = render_images(
                img,
                display_mode=display_mode,
                is_complex=is_complex,
                vmin=render_min,
                vmax=render_max,
            )

            rendered_samples.append({
                "index": indices[i],
                "image_left_base64": left_img,
                "image_right_base64": right_img,
                "image_left_label": left_lbl,
                "image_right_label": right_lbl,
                "colorbar_range": ranges,
                "shape": list(img.shape[-2:]),
            })

        return {
            "samples": rendered_samples,
            "global_stats": stats,
            "global_histogram": {
                "bins": hist_bins,
                "counts": hist_counts,
                "stats": {
                    "min": stats["min"],
                    "max": stats["max"],
                    "mean": stats["mean"],
                    "std": stats["std"],
                },
            },
            "global_range": (render_min, render_max),
        }


    def navigate_next(self, dataset_idx: int, sample_idx: int, nav_type: str) -> int:
        return self._navigate(dataset_idx, sample_idx, nav_type, direction=1)

    def navigate_prev(self, dataset_idx: int, sample_idx: int, nav_type: str) -> int:
        return self._navigate(dataset_idx, sample_idx, nav_type, direction=-1)

    def get_histogram(self, dataset_idx: int, sample_idx: int, augment: bool, seed: int | None) -> dict[str, Any]:
        dataset = self._get_dataset(dataset_idx, augment=augment)
        self._seed_rng(seed)
        sample = dataset[sample_idx]
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

    def _navigate(self, dataset_idx: int, sample_idx: int, nav_type: str, direction: int) -> int:
        dataset = self._get_dataset(dataset_idx, augment=False)
        structure = self._get_structure(dataset_idx)
        total = len(dataset)

        if nav_type == "sample":
            return max(0, min(total - 1, sample_idx + direction))

        subject_idx, slice_idx, coil_idx = structure.index_to_position[sample_idx]
        if nav_type == "coil" and structure.has_coils and structure.coils_per_subject is not None:
            coil_count = structure.coils_per_subject[subject_idx]
            next_coil = coil_idx + direction if coil_idx is not None else None
            if next_coil is None or next_coil < 0 or next_coil >= coil_count:
                return sample_idx
            return structure.position_to_index[(subject_idx, slice_idx, next_coil)]

        if nav_type == "slice":
            next_slice = slice_idx + direction
            slice_count = structure.slices_per_subject[subject_idx]
            if next_slice < 0 or next_slice >= slice_count:
                return sample_idx
            return structure.position_to_index[(subject_idx, next_slice, coil_idx if structure.has_coils else None)]

        if nav_type == "subject":
            next_subject = subject_idx + direction
            if next_subject < 0 or next_subject >= structure.subject_count:
                return sample_idx
            new_coil = 0 if structure.has_coils else None
            return structure.position_to_index[(next_subject, 0, new_coil)]

        return sample_idx

    def _get_dataset(self, dataset_idx: int, augment: bool) -> ComplexImageDataset | FlatH5Dataset:
        cache = self._datasets if augment else self._datasets_no_aug
        if dataset_idx not in cache:
            config = self.configs[dataset_idx]
            if not augment:
                config = clone_without_augmentation(config)
            cache[dataset_idx] = config.create()
        return cache[dataset_idx]

    def _get_structure(self, dataset_idx: int) -> DatasetStructure:
        if dataset_idx in self._structures:
            return self._structures[dataset_idx]
        dataset = self._get_dataset(dataset_idx, augment=False)
        structure = self._build_structure(dataset)
        self._structures[dataset_idx] = structure
        return structure

    def _build_structure(self, dataset: ComplexImageDataset | FlatH5Dataset) -> DatasetStructure:
        subject_keys = list(dataset._subject_keys)
        subject_key_to_index = {key: idx for idx, key in enumerate(subject_keys)}

        slices_per_subject: dict[str, int] = dict.fromkeys(subject_keys, 0)
        coils_per_subject: dict[str, int] | None = (
            dict.fromkeys(subject_keys, 0) if isinstance(dataset, ComplexImageDataset) and dataset.mode == "coilwise" else None
        )

        index_to_position: list[tuple[int, int, int | None]] = []
        position_to_index: dict[tuple[int, int, int | None], int] = {}

        if isinstance(dataset, ComplexImageDataset):
            for idx, (subject_key, slice_idx, coil_idx) in enumerate(dataset._slice_index):
                subject_idx = subject_key_to_index[subject_key]
                index_to_position.append((subject_idx, slice_idx, coil_idx))
                position_to_index[(subject_idx, slice_idx, coil_idx)] = idx
                slices_per_subject[subject_key] = max(slices_per_subject[subject_key], slice_idx + 1)
                if coils_per_subject is not None:
                    coils_per_subject[subject_key] = max(coils_per_subject[subject_key], coil_idx + 1)
        else:
            for idx, (subject_key, slice_idx) in enumerate(dataset._slice_index):
                subject_idx = subject_key_to_index[subject_key]
                index_to_position.append((subject_idx, slice_idx, None))
                position_to_index[(subject_idx, slice_idx, None)] = idx
                slices_per_subject[subject_key] = max(slices_per_subject[subject_key], slice_idx + 1)

        slices_list = [slices_per_subject[key] for key in subject_keys]
        coils_list = [coils_per_subject[key] for key in subject_keys] if coils_per_subject is not None else None

        return DatasetStructure(
            subject_keys=subject_keys,
            slices_per_subject=slices_list,
            coils_per_subject=coils_list,
            index_to_position=index_to_position,
            position_to_index=position_to_index,
        )

    def _get_dataset_summary(self, dataset_idx: int) -> dict[str, Any]:
        if dataset_idx in self._summaries:
            return self._summaries[dataset_idx]
        dataset = self._get_dataset(dataset_idx, augment=False)
        config = self.configs[dataset_idx]
        raw_config = self.raw_configs[dataset_idx]
        summary = {
            "dataset_idx": dataset_idx,
            "total_samples": len(dataset),
            "config": raw_config,
            "h5_path": getattr(config, "h5_path", ""),
            "h5_filename": getattr(config, "h5_path", "").split("/")[-1],
            "kind": config.kind.name,
            "type": getattr(config, "type", ""),
            "mode": getattr(config, "mode", None),
            "scaling_factor": getattr(config, "scaling_factor", 1.0),
            "penumbra_width": getattr(config, "penumbra_width", None),
            "schedule": {
                "type": config.schedule.type if config.schedule is not None else None,
                "weight": config.schedule.weight if config.schedule is not None else None,
                "steps": getattr(config.schedule, "steps", None) if config.schedule is not None else None,
            },
            "augmentation_config": raw_config.get("augmentation_config"),
        }
        self._summaries[dataset_idx] = summary
        return summary

    def _compute_stats(self, image: torch.Tensor, background_mask: torch.Tensor) -> dict[str, float]:
        values = image.abs().float() if image.is_complex() else image.float()
        values = values[torch.isfinite(values)]
        if values.numel() == 0:
            return {"min": 0.0, "max": 0.0, "mean": 0.0, "variance": 0.0, "std": 0.0, "signal_fraction": 0.0, "p99": 0.0}

        variance = torch.var(values, unbiased=False).item()
        std = torch.sqrt(torch.tensor(variance)).item()
        signal_fraction = (~background_mask).float().mean().item()
        return {
            "min": values.min().item(),
            "max": values.max().item(),
            "mean": values.mean().item(),
            "variance": variance,
            "std": std,
            "signal_fraction": signal_fraction,
            "p99": torch.quantile(values, 0.99).item(),
        }

    def _build_metadata(
        self,
        dataset_idx: int,
        sample_idx: int,
        dataset: ComplexImageDataset | FlatH5Dataset,
        image: torch.Tensor,
    ) -> dict[str, Any]:
        config = self.configs[dataset_idx]
        structure = self._get_structure(dataset_idx)
        subject_idx, slice_idx, coil_idx = structure.index_to_position[sample_idx]

        return {
            "dataset_idx": dataset_idx,
            "h5_path": getattr(config, "h5_path", ""),
            "h5_filename": getattr(config, "h5_path", "").split("/")[-1],
            "dataset_type": getattr(config, "type", ""),
            "kind": config.kind.name,
            "mode": getattr(config, "mode", None),
            "subject_key": structure.subject_keys[subject_idx],
            "subject_index": subject_idx,
            "slice_index": slice_idx,
            "coil_index": coil_idx,
            "is_complex": bool(image.is_complex()),
            "shape": list(image.shape[-2:]),
            "total_samples": len(dataset),
        }

    def _build_navigation(self, dataset_idx: int, sample_idx: int) -> dict[str, Any]:
        structure = self._get_structure(dataset_idx)
        subject_idx, slice_idx, coil_idx = structure.index_to_position[sample_idx]
        has_coils = structure.has_coils
        coil_count = structure.coils_per_subject[subject_idx] if has_coils and structure.coils_per_subject is not None else None
        slice_count = structure.slices_per_subject[subject_idx]

        dataset = self._get_dataset(dataset_idx, False)
        dataset_type = dataset.mode if isinstance(dataset, ComplexImageDataset) else "flat"

        return {
            "dataset_type": dataset_type,
            "linear_index": sample_idx,
            "total_samples": len(self._get_dataset(dataset_idx, False)),
            "subject_count": structure.subject_count,
            "current_subject_index": subject_idx,
            "current_subject_key": structure.subject_keys[subject_idx],
            "slice_count": slice_count,
            "current_slice_index": slice_idx,
            "coil_count": coil_count,
            "current_coil_index": coil_idx,
            "can_next_coil": has_coils and coil_count is not None and coil_idx is not None and coil_idx + 1 < coil_count,
            "can_prev_coil": has_coils and coil_count is not None and coil_idx is not None and coil_idx > 0,
            "can_next_slice": slice_idx + 1 < slice_count,
            "can_prev_slice": slice_idx > 0,
            "can_next_subject": subject_idx + 1 < structure.subject_count,
            "can_prev_subject": subject_idx > 0,
        }

    def _seed_rng(self, seed: int | None) -> None:
        if seed is None:
            return
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


DatasetTypes = ComplexImageDataset | MagnitudeImageDataset | MappingDataset
