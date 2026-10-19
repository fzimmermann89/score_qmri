"""Saved-sample preservation and the recovered MONAI spatial recipe."""

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from dataset_preparation import latent_t1
from dataset_preparation.raw_mri import clean_mask_2d


def test_repack_preserves_samples_attributes_and_inputs(tmp_path: Path) -> None:
    source = tmp_path / "saved.h5"
    values = np.arange(3 * 8 * 8, dtype=np.float32).reshape(3, 8, 8)
    with h5py.File(source, "x") as file:
        file.attrs["n_samples"] = 2
        for key in ["0", "1"]:
            data = file.create_dataset(key, data=values)
            data.attrs["gender"] = 0.0
            data.attrs["age"] = 0.3
    before = source.read_bytes()
    output = tmp_path / "ldm_t1w.h5"
    latent_t1.repack_samples(source, output, limit=1)
    with h5py.File(output, "r") as file:
        assert list(file) == ["0"]
        assert file.attrs["n_samples"] == 1
        np.testing.assert_array_equal(file["0"][:], values)
        assert file["0"].attrs["age"] == 0.3
    assert source.read_bytes() == before
    packed_before = output.read_bytes()
    with pytest.raises(FileExistsError):
        latent_t1.repack_samples(source, output)
    assert output.read_bytes() == packed_before


def test_preparation_matches_original_permutation_and_normalization() -> None:
    coordinates = torch.linspace(-1, 1, 64)
    x, y = torch.meshgrid(coordinates, coordinates, indexing="ij")
    image = 0.005 + 0.8 * torch.exp(-8 * (x * x + y * y))
    volume = image[:, :, None].repeat(1, 1, 88)
    expected = volume.permute(2, 1, 0).flip(-2)[40:-40:4].cpu().float().detach()
    expected = torch.nn.functional.pad(expected[None], (16, 16, 16, 16))[0]
    mask = torch.stack([clean_mask_2d(m) for m in expected > 0.05])
    expected -= torch.quantile(expected[~mask], 0.95)
    expected /= torch.quantile(expected[mask], 0.99)
    expected[~mask] = 0
    expected.clamp_max_(1.2)
    torch.testing.assert_close(latent_t1.prepare_generated_sample(volume), expected, rtol=0, atol=0)


def test_generation_refuses_output_before_loading_models(tmp_path: Path) -> None:
    output = tmp_path / "preserve.h5"
    output.write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        latent_t1.generate_samples(tmp_path, output)
    assert output.read_bytes() == b"preserve"
    new = tmp_path / "new.h5"
    with pytest.raises(FileNotFoundError):
        latent_t1.generate_samples(tmp_path, new)
    assert not new.exists()
