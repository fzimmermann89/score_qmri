"""Mock sample manager for when real datasets are unavailable."""

from __future__ import annotations

import random
from typing import Any

import numpy as np

from score_qmri.datasets.kind import DatasetKind

from .sampling import OUTPUT_SIZE, SampleResult


def generate_shepp_logan_phantom(size: int = 256) -> np.ndarray:
    """Generate a Shepp-Logan phantom image."""
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)

    ellipses = [
        (1.0, 0.69, 0.92, 0, 0, 0),
        (-0.8, 0.6624, 0.874, 0, -0.0184, 0),
        (-0.2, 0.11, 0.31, 0.22, 0, -18),
        (-0.2, 0.16, 0.41, -0.22, 0, 18),
        (0.1, 0.21, 0.25, 0, 0.35, 0),
        (0.1, 0.046, 0.046, 0, 0.1, 0),
        (0.1, 0.046, 0.046, 0, -0.1, 0),
        (0.1, 0.046, 0.023, -0.08, -0.605, 0),
        (0.1, 0.023, 0.023, 0, -0.606, 0),
        (0.1, 0.023, 0.046, 0.06, -0.605, 0),
    ]

    phantom = np.zeros((size, size), dtype=np.float32)
    for intensity, a, b, x0, y0, theta_deg in ellipses:
        theta = np.deg2rad(theta_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        x_rot = cos_t * (X - x0) + sin_t * (Y - y0)
        y_rot = -sin_t * (X - x0) + cos_t * (Y - y0)
        mask = (x_rot / a) ** 2 + (y_rot / b) ** 2 <= 1
        phantom[mask] += intensity

    return np.clip(phantom, 0, 1)


class MockSampleManager:
    """Mock sample manager for when real data is unavailable."""

    MOCK_KINDS = [
        ("T1W_MAGNITUDE", False),
        ("T2W_MAGNITUDE", False),
        ("T1MAP", False),
        ("COMPLEX_T1W_COIL_COMBINED", True),
    ]

    def __init__(self) -> None:
        self.output_size = OUTPUT_SIZE

    def list_datasets(self) -> list[dict[str, Any]]:
        """List mock datasets."""
        return [
            {
                "dataset_idx": i,
                "total_samples": 100,
                "h5_path": f"mock://{name.lower()}.h5",
                "h5_filename": f"mock_{name.lower()}",
                "kind": name,
                "type": "complex" if is_complex else "magnitude",
                "scaling_factor": 1.0,
                "penumbra_width": 30,
                "schedule": {"type": "constant", "weight": 1.0},
                "has_augmentation": False,
                "is_mock": True,
            }
            for i, (name, is_complex) in enumerate(self.MOCK_KINDS)
        ]

    def get_unique_kinds(self) -> list[dict[str, Any]]:
        """Get unique kinds with their dataset counts."""
        return [{"name": name, "count": 1} for name, _ in self.MOCK_KINDS]

    def sample_dataset(self, dataset_idx: int, n: int = 32) -> SampleResult:
        """Sample from mock dataset."""
        name, is_complex = self.MOCK_KINDS[dataset_idx % len(self.MOCK_KINDS)]
        return self._generate_samples(n, name, is_complex)

    def sample_kind(self, kind: DatasetKind, n: int = 32) -> SampleResult:
        """Sample from mock dataset by kind."""
        is_complex = kind.is_complex
        return self._generate_samples(n, kind.name, is_complex)

    def _generate_samples(self, n: int, kind_name: str, is_complex: bool) -> SampleResult:
        """Generate mock samples."""
        h = w = self.output_size
        images = np.zeros((n, 2, h, w), dtype=np.float32)
        penumbra = np.zeros((n, h, w), dtype=np.uint8)
        background = np.zeros((n, h, w), dtype=np.uint8)

        for i in range(n):
            # Generate phantom with slight variation
            phantom = generate_shepp_logan_phantom(h)
            scale = 0.8 + 0.4 * random.random()
            phantom = phantom * scale

            # Add some noise
            noise = np.random.randn(h, w).astype(np.float32) * 0.02
            phantom = np.clip(phantom + noise, 0, 1)

            if is_complex:
                phase = np.random.uniform(-0.5, 0.5) + 0.1 * np.random.randn(h, w).astype(np.float32)
                images[i, 0] = phantom * np.cos(phase)
                images[i, 1] = phantom * np.sin(phase)
            else:
                images[i, 0] = phantom
                images[i, 1] = 0

            # Generate masks
            background[i] = (phantom < 0.01).astype(np.uint8)
            penumbra[i] = ((phantom >= 0.01) & (phantom < 0.1)).astype(np.uint8)

        return SampleResult(
            images=images,
            penumbra_masks=penumbra,
            background_masks=background,
            labels=[f"mock_{kind_name.lower()}"] * n,
            kinds=[kind_name] * n,
        )

    def clear_cache(self) -> None:
        """No-op for mock manager."""
        pass
