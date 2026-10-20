"""Build ldm_t1w.h5 by repacking the original samples_500.h5.
To generate new samples, manually download the bundle and model weights from
https://huggingface.co/MONAI/brain_image_synthesis_latent_diffusion_model
at revision 55d88321e08930ba3a71d084ed94afaab0b29320; pass --bundle-root.
The historical seed is unknown, so new draws will not match the training H5.
Run: python latent_t1.py repack --input-root samples_500.h5 --output NEW.h5."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np
import scipy.ndimage as ndi
import torch


def clean_mask_2d(mask: np.ndarray | torch.Tensor) -> torch.Tensor:
    """Original opening, closing, dilation, hole fill and largest component."""
    structure = ndi.generate_binary_structure(2, 1)
    result = ndi.binary_opening(np.asarray(mask).astype(bool), structure=structure)
    result = ndi.binary_closing(result, structure=structure, iterations=1)
    result = ndi.binary_dilation(result, structure=structure, iterations=2)
    result = ndi.binary_fill_holes(result)
    labels, n_labels = ndi.label(result, structure=structure)
    if n_labels:
        result = labels == (np.bincount(labels.ravel())[1:].argmax() + 1)
    return torch.tensor(result, dtype=torch.bool)


def repack_samples(source: Path, output: Path, subjects: Sequence[str] | None = None) -> None:
    """Copy saved arrays and metadata in slabs, refusing every existing output."""
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    with h5py.File(str(source), "r") as original:
        names = sorted(original, key=lambda key: (0, int(key)) if key.isdecimal() else (1, key))
        if subjects:
            unknown = set(subjects) - set(names)
            if unknown:
                raise ValueError(f"Unknown saved samples: {sorted(unknown)}")
            names = [name for name in names if name in subjects]
        if not names:
            raise ValueError("No selected saved samples.")
        output.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(str(output), "w-") as target:
            for name, value in original.attrs.items():
                target.attrs[name] = value
            target.attrs["n_samples"] = len(names)
            for name in names:
                src = original[name]
                if not isinstance(src, h5py.Dataset) or src.ndim != 3 or src.dtype != np.float32:
                    raise ValueError(f"{name}: expected a float32 slice volume.")
                dst = target.create_dataset(
                    name, shape=src.shape, dtype=src.dtype, compression="lzf", shuffle=True, chunks=(1, *src.shape[1:])
                )
                for key, value in src.attrs.items():
                    dst.attrs[key] = value
                for z in range(len(src)):
                    dst[z] = src[z]


def prepare_generated_sample(volume: torch.Tensor) -> torch.Tensor:
    """Apply the original generated-volume orientation, slices, mask and scaling."""
    x = volume.transpose(0, 2).flip(-2)[40:-40:4].cpu().float().detach()
    x = torch.nn.functional.pad(x[None], (16, 16, 16, 16))[0]
    mask = torch.stack([clean_mask_2d(m) for m in x > 0.05])
    x -= torch.quantile(x[~mask], 0.95)
    x /= torch.quantile(x[mask], 0.99)
    x[~mask] = 0
    x.clamp_max_(1.2)
    return x


def generate_samples(bundle_root: Path, output: Path, seed: int = 0, device: str = "cpu") -> None:
    """Generate new explicitly seeded samples from an already downloaded bundle."""
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    config = bundle_root / "configs/inference.json"
    required = [config, bundle_root / "scripts/sampler.py", bundle_root / "models/model.pt", bundle_root / "models/autoencoder.pt"]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    from monai.bundle import ConfigParser

    torch.manual_seed(seed)
    selected_device = torch.device(device)
    parser = ConfigParser()
    parser.read_config(str(config))
    parser.update(
        {
            "bundle_root": str(bundle_root.resolve()),
            "device": selected_device,
            "load_autoencoder": "$@load_autoencoder_func(torch.load(@load_autoencoder_path, map_location='cpu'))",
            "load_diffusion": "$@load_diffusion_func(torch.load(@load_diffusion_path, map_location='cpu'))",
        }
    )
    # The pinned MONAI config imports scripts.sampler from its own bundle directory.
    sys.path.insert(0, str(bundle_root.resolve()))
    try:
        scheduler = parser.get_parsed_content("scheduler")
        autoencoder = parser.get_parsed_content("autoencoder").eval()
        diffusion = parser.get_parsed_content("diffusion").eval()
        sampler = parser.get_parsed_content("sampler")
        with torch.inference_mode(), h5py.File(str(output), "w-") as target:
            target.attrs["n_samples"] = 500
            target.attrs["seed"] = seed
            for idx in range(500):
                gender = torch.randint(0, 2, (1, 1, 1), device=selected_device).to(torch.float32)
                other = torch.rand((1, 1, 3), device=selected_device, dtype=torch.float32)
                conditioning = torch.cat([gender, other], dim=-1)
                conditioning[..., -1] = 1.0
                noise = torch.randn((1, 3, 20, 28, 20), device=selected_device, dtype=torch.float32)
                sampled = sampler.sampling_fn(noise, autoencoder, diffusion, scheduler, conditioning)
                image = prepare_generated_sample(sampled[0, 0])
                data = target.create_dataset(
                    str(idx), data=image.numpy(), dtype=np.float32, compression="lzf", shuffle=True, chunks=(1, *image.shape[1:])
                )
                for name, value in zip(("gender", "age", "ventricular_vol", "brain_vol"), conditioning.squeeze().cpu(), strict=True):
                    data.attrs[name] = float(value)
    finally:
        sys.path.pop(0)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("repack", "generate"))
    parser.add_argument("--input-root", type=Path, help="Existing samples_500.h5 for repack.")
    parser.add_argument("--bundle-root", type=Path, help="Local pinned MONAI bundle for generate.")
    parser.add_argument("--output", required=True, type=Path, help="New output H5, normally ldm_t1w.h5.")
    parser.add_argument("--subjects", nargs="+", help="Saved sample keys, repack only.")
    parser.add_argument("--seed", type=int, default=0, help="New generation seed; historical seed was not saved.")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if args.mode == "repack":
        if args.input_root is None or args.bundle_root is not None:
            parser.error("repack requires --input-root and no --bundle-root.")
        repack_samples(args.input_root, args.output, args.subjects)
    else:
        if args.bundle_root is None or args.input_root is not None or args.subjects:
            parser.error("generate requires --bundle-root and no --input-root/--subjects.")
        torch.set_num_threads(1)
        generate_samples(args.bundle_root, args.output, args.seed, args.device)


if __name__ == "__main__":
    main()
