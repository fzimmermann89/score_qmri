"""Check the recovered viewer against the current dataset and server contracts."""

import json

import h5py
import numpy as np
import pytest
import yaml

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from score_qmri.viewer import server
from score_qmri.viewer.config_loader import load_config
from score_qmri.viewer.sampling import SampleManager


def test_split_config_loads_real_samples_without_workers(tmp_path):
    image_path = tmp_path / "images.h5"
    with h5py.File(image_path, "w") as file:
        file["subject"] = np.ones((2, 32, 32), dtype=np.complex64)
    config_path = tmp_path / "data.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "class_path": "score_qmri.datasets.lightning.MRIDataModule",
                    "init_args": {
                        "sample_size": [32, 32],
                        "num_workers": 0,
                        "dataset_configs": [
                            {
                                "type": "complex",
                                "h5_path": str(image_path),
                                "mode": "flat",
                                "kind": "COMPLEX_T1W_COIL_COMBINED",
                                "augmentation_config": None,
                                "penumbra_width": 0,
                            }
                        ],
                    },
                }
            }
        )
    )
    manager = SampleManager(load_config(config_path), prewarm=False)
    result = manager.sample_dataset(0, n=1)
    assert result.images.shape == (1, 2, 96, 96)
    assert np.isfinite(result.images).all()
    assert result.kinds == ["COMPLEX_T1W_COIL_COMBINED"]
    loader = manager._get_dataloader(0)
    assert loader.num_workers == 0
    assert not loader.pin_memory
    assert manager.sample_dataset(0, n=3).images.shape[0] == 3


def test_missing_configuration_serves_labelled_mock_samples(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "_config_path", str(tmp_path / "missing.yaml"))
    with TestClient(server.app) as client:
        status = client.get("/api/status").json()
        assert status["ready"] and status["mock_mode"]
        assert client.get("/").status_code == 200
        datasets = client.get("/api/datasets")
        assert datasets.status_code == 200
        response = client.get("/api/samples/dataset/0?n=1")
        assert response.status_code == 200
        shape = json.loads(response.headers["X-Shape"])
        assert len(response.content) > int(np.prod(shape)) * 4
