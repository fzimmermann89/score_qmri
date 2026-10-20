"""Check non-obvious scientific conventions against independent expectations."""

from pathlib import Path

import numpy as np
import pytest
import torch

from dataset_preparation.download import download_file, rockland_plan
from dataset_preparation.ibc_t2 import pad_or_crop
from dataset_preparation.wand import compute_psir_mp2rage, fit_t1_m0


def test_signed_complex_mp2rage_and_lut_matching(tmp_path: Path) -> None:
    # Opposite phase carries the inversion sign; magnitudes alone cannot.
    s1 = torch.tensor([-2 + 0j, 2 + 0j])
    s2 = torch.tensor([4 + 0j, 4 + 0j])
    psir, mp2 = compute_psir_mp2rage(s1, s2)
    torch.testing.assert_close(psir, torch.tensor([-1 / 3, 1 / 3]))
    torch.testing.assert_close(mp2, torch.tensor([-0.4, 0.4]))
    lut = tmp_path / "lut.npz"
    inv1 = np.full((3, 3), 0.71)
    inv2 = np.full((3, 3), 0.71 * 2)
    p = np.zeros((3, 3))
    m = p.copy()
    p[:, 1:] = [-1 / 3, 1 / 3]
    m[:, 1:] = [-0.4, 0.4]
    np.savez(lut, inv1=inv1, inv2=inv2, psir=p, mp2rage=m)
    t1, m0, _ = fit_t1_m0(psir, mp2, s1, s2, torch.ones(2), str(lut), "cpu", 1)
    torch.testing.assert_close(t1, torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(m0.abs(), torch.tensor([2.0, 2.0]))


def test_mrpro_crop_odd_negative_padding() -> None:
    # Crop 3 pixels: historical mrpro removes one on the left, two right.
    data = np.arange(49).reshape(7, 7)
    np.testing.assert_array_equal(pad_or_crop(data, (4, 4)), data[1:5, 1:5])


def test_rockland_download_plan_is_bounded_and_keeps_anatomy(tmp_path: Path) -> None:
    path = tmp_path / "links.csv"
    path.write_text(
        "filepath\ns3://fcp-indi/data/Projects/RocklandSample/RawDataBIDSLatest/sub-A1/ses-BAS1/anat/a_T1w.nii.gz\ns3://fcp-indi/data/Projects/RocklandSample/RawDataBIDSLatest/sub-A1/ses-BAS1/func/a_bold.nii.gz\n"
    )
    plan = rockland_plan(path, ["A1"], ["T1w"], 1)
    assert len(plan) == 1
    assert plan[0][1] == Path("sub-A1/ses-BAS1/anat/a_T1w.nii.gz")
    target = tmp_path / "exists"
    target.write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        download_file("https://example.invalid/data", target)
    assert target.read_bytes() == b"preserve"
