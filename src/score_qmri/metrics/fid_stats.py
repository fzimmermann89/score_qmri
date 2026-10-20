"""Compute FID reference statistics from training data."""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import yaml
from jsonargparse import ArgumentParser
from torch import Tensor
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm

from score_qmri.datasets.batch_sampler import ScheduledBatchSampler
from score_qmri.datasets.kind import DatasetKind
from score_qmri.datasets.lightning import ConstantSchedule, DatasetConfig
from score_qmri.metrics.fid import compute_statistics_torch, prepare_inception_input
from score_qmri.metrics.inception import InceptionV3Features


def _parse_config(config_path: Path) -> dict[str, Any]:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    data_cfg = config.get("data", {})
    init_args = data_cfg.get("init_args", data_cfg)
    return {
        "dataset_configs_raw": init_args.get("dataset_configs", []),
        "batch_size": init_args.get("batch_size", 64),
        "base_seed": init_args.get("base_seed", 42),
    }


def _instantiate_dataset_configs(
    raw_configs: list[dict[str, Any]],
    disable_augmentation: bool,
) -> list[DatasetConfig]:
    parser = ArgumentParser()
    parser.add_argument("--cfg", type=DatasetConfig)
    configs: list[DatasetConfig] = []
    for raw in raw_configs:
        parsed = parser.parse_object({"cfg": raw})
        instantiated = parser.instantiate_classes(parsed)
        cfg: DatasetConfig = instantiated.cfg
        if disable_augmentation and cfg.augmentation_config is not None:
            aug = cfg.augmentation_config
            new_aug = replace(
                aug,
                geometric_prob=0.0,
                contrast_prob=0.0,
                phase_prob=0.0,
                flip_prob=0.0,
                roll_prob=0.0,
                intensity_scale_prob=0.0,
            )
            cfg = replace(cfg, augmentation_config=new_aug)
        configs.append(cfg)
    return configs


def _group_by_kind(configs: list[DatasetConfig]) -> dict[DatasetKind, list[DatasetConfig]]:
    groups: dict[DatasetKind, list[DatasetConfig]] = defaultdict(list)
    for cfg in configs:
        groups[cfg.kind].append(cfg)
    return groups


def _fid_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    images_2ch: list[Tensor] = []
    for sample in batch:
        img = sample["image"]
        img_2ch = torch.moveaxis(torch.view_as_real(img), -1, 0) if img.is_complex() else torch.stack([img, torch.zeros_like(img)], dim=0)
        images_2ch.append(img_2ch)

    return {"image": torch.stack(images_2ch)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute FID reference statistics")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--inception-weights",
        type=Path,
        default=None,
        help="Path to FID Inception weights (downloaded if missing)",
    )
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config_data = _parse_config(args.config)
    batch_size = args.batch_size or config_data["batch_size"]
    seed = config_data["base_seed"] if args.seed is None else args.seed

    configs = _instantiate_dataset_configs(config_data["dataset_configs_raw"], args.no_augmentation)
    kind_groups = _group_by_kind(configs)

    device = torch.device(args.device)
    model = InceptionV3Features(weights_path=args.inception_weights).to(device)
    model.eval()

    stats: dict[str, Tensor] = {}
    for kind, kind_configs in kind_groups.items():
        # Compute frozen weights at step=0
        frozen_weights = []
        for cfg in kind_configs:
            schedule = cfg.schedule if cfg.schedule is not None else ConstantSchedule()
            frozen_weights.append(schedule(step=0))

        total_weight = sum(frozen_weights)
        if total_weight <= 0:
            print(f"[SKIP] {kind.name}: all datasets have zero weight")
            continue

        datasets = [cfg.create() for cfg in kind_configs]
        concat_ds: ConcatDataset = ConcatDataset(datasets)
        dataset_lengths = [len(ds) for ds in datasets]
        total_samples = sum(dataset_lengths)
        print(f"[START] {kind.name}: {len(datasets)} dataset(s), {total_samples} total samples")

        schedules = [ConstantSchedule(weight=w) for w in frozen_weights]

        num_batches = math.ceil(args.num_samples / batch_size)
        sampler = ScheduledBatchSampler(
            concat_ds,
            dataset_lengths,
            dataset_kinds=[cfg.kind for cfg in kind_configs],
            schedules=schedules,
            batch_size=batch_size,
            kind_normalization=0.0,
            base_seed=seed,
            rank=0,
            max_steps=num_batches,
        )

        loader = DataLoader(
            concat_ds,
            batch_sampler=sampler,
            collate_fn=_fid_collate,
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
            pin_memory=device.type == "cuda" and torch.cuda.is_available(),
            multiprocessing_context="spawn" if args.num_workers > 0 else None,
        )

        features: list[Tensor] = []
        for batch in tqdm(loader, desc=kind.name):
            images = batch["image"].to(device)
            is_complex = images.new_full((images.shape[0],), kind.is_complex, dtype=torch.bool)
            rgb = prepare_inception_input(images, is_complex)
            with torch.inference_mode():
                feats = model(rgb)
            features.append(feats.cpu())

        all_features = torch.cat(features, dim=0)[: args.num_samples]
        mu, sigma = compute_statistics_torch(all_features.double())
        stats[f"{kind.name}_mu"] = mu.cpu()
        stats[f"{kind.name}_sigma"] = sigma.cpu()
        print(f"[DONE] {kind.name}: computed stats from {all_features.shape[0]} samples")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stats, args.output)


if __name__ == "__main__":
    main()
