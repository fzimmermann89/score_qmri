"""Mock dataset for demo/development when real data is unavailable."""

from __future__ import annotations

import numpy as np
import torch


def generate_shepp_logan_phantom(size: int = 256) -> np.ndarray:
    """Generate a Shepp-Logan phantom image."""
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)

    ellipses = [
        # (intensity, a, b, x0, y0, theta)
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

    phantom = np.clip(phantom, 0, 1)
    return phantom


def generate_brain_phantom(size: int = 256, slice_idx: int = 0, num_slices: int = 20) -> np.ndarray:
    """Generate a simplified brain-like phantom with slice variation."""
    base = generate_shepp_logan_phantom(size)

    z_factor = 1.0 - 0.3 * abs(slice_idx - num_slices // 2) / (num_slices // 2)
    z_factor = max(0.3, z_factor)

    scaled = base * z_factor

    np.random.seed(42 + slice_idx)
    # Generate noise but only apply it where the phantom has signal
    # This ensures a perfectly black background for mask detection
    noise = np.random.randn(size, size).astype(np.float32) * 0.02
    
    # Create a mask for where signal exists (with a small epsilon for safety)
    signal_mask = scaled > 1e-6
    
    result = scaled.copy()
    result[signal_mask] += noise[signal_mask]
    
    return np.clip(result, 0, 1).astype(np.float32)


def generate_complex_phantom(size: int = 256, slice_idx: int = 0, num_slices: int = 20) -> np.ndarray:
    """Generate a complex-valued phantom (for complex MRI data)."""
    magnitude = generate_brain_phantom(size, slice_idx, num_slices)

    y, x = np.ogrid[:size, :size]
    cx, cy = size // 2, size // 2
    np.random.seed(123 + slice_idx)
    phase_offset = np.random.uniform(-0.5, 0.5)
    phase = 0.3 * ((x - cx) / size + (y - cy) / size) + phase_offset
    phase = phase.astype(np.float32)

    real = magnitude * np.cos(phase)
    imag = magnitude * np.sin(phase)
    return (real + 1j * imag).astype(np.complex64)


class MockDataset:
    """A mock dataset that generates phantom images for UI development."""

    def __init__(
        self,
        name: str,
        kind: str,
        is_complex: bool = False,
        num_subjects: int = 3,
        slices_per_subject: int = 20,
        image_size: int = 256,
    ) -> None:
        self.name = name
        self.kind = kind
        self.is_complex = is_complex
        self.num_subjects = num_subjects
        self.slices_per_subject = slices_per_subject
        self.image_size = image_size
        self.mode = "flat"

        self._subject_keys = [f"phantom_sub_{i:03d}" for i in range(num_subjects)]
        self._slice_index = [
            (subj, slice_idx)
            for subj in self._subject_keys
            for slice_idx in range(slices_per_subject)
        ]

    def __len__(self) -> int:
        return len(self._slice_index)

    def __getitem__(self, idx: int) -> dict:
        subject_key, slice_idx = self._slice_index[idx]
        subject_idx = self._subject_keys.index(subject_key)

        seed = subject_idx * 1000 + slice_idx
        np.random.seed(seed)

        if self.is_complex:
            image_np = generate_complex_phantom(
                self.image_size, slice_idx, self.slices_per_subject
            )
            image = torch.from_numpy(image_np)
        else:
            image_np = generate_brain_phantom(
                self.image_size, slice_idx, self.slices_per_subject
            )
            image = torch.from_numpy(image_np)

        magnitude = np.abs(image_np) if self.is_complex else image_np
        background_mask = torch.from_numpy(magnitude < 0.01)
        penumbra_mask = torch.from_numpy((magnitude >= 0.01) & (magnitude < 0.1))

        return {
            "image": image,
            "penumbra_mask": penumbra_mask,
            "background_mask": background_mask,
            "kind": self.kind,
        }


def create_mock_datasets() -> list[MockDataset]:
    """Create a set of mock datasets for demonstration."""
    return [
        MockDataset(
            name="Mock T1w Magnitude",
            kind="T1W_MAGNITUDE",
            is_complex=False,
            num_subjects=3,
            slices_per_subject=20,
        ),
        MockDataset(
            name="Mock T2w Complex",
            kind="T2W_COMPLEX_COMBINED",
            is_complex=True,
            num_subjects=2,
            slices_per_subject=15,
        ),
        MockDataset(
            name="Mock T1 Map",
            kind="T1MAP",
            is_complex=False,
            num_subjects=2,
            slices_per_subject=25,
        ),
    ]
