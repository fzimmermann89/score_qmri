"""Tests for RunningNormalizer."""

import pytest
import torch

from score_qmri.models.normalization import RunningNormalizer


@pytest.fixture
def normalizer():
    return RunningNormalizer(num_kinds=3, momentum=0.1)


class TestRunningNormalizer:
    def test_first_update_initializes_stats(self, normalizer):
        normalizer.train()
        x = torch.ones(2, 2, 8, 8) * 5.0
        x[:, 1] = 10.0
        mask = torch.ones(2, 8, 8, dtype=torch.bool)
        kind = torch.tensor([0, 0])

        normalizer.normalize(x, mask, kind)

        assert normalizer.n[0].item() == 1
        assert torch.isclose(normalizer.mean[0, 0], torch.tensor(5.0), atol=1e-5)
        assert torch.isclose(normalizer.mean[0, 1], torch.tensor(10.0), atol=1e-5)

    def test_ema_update_blends_correctly(self, normalizer):
        normalizer.train()
        mask = torch.ones(2, 8, 8, dtype=torch.bool)
        kind = torch.tensor([0, 0])

        x1 = torch.ones(2, 2, 8, 8) * 10.0
        normalizer.normalize(x1, mask, kind)

        x2 = torch.ones(2, 2, 8, 8) * 20.0
        normalizer.normalize(x2, mask, kind)

        assert normalizer.n[0].item() == 2
        expected = 10.0 * 0.9 + 20.0 * 0.1
        assert torch.isclose(normalizer.mean[0, 0], torch.tensor(expected), atol=1e-5)

    def test_mask_excludes_invalid_pixels(self, normalizer):
        normalizer.train()
        x = torch.zeros(1, 2, 8, 8)
        x[:, :, :4, :] = 100.0
        x[:, :, 4:, :] = 0.0

        mask = torch.zeros(1, 8, 8, dtype=torch.bool)
        mask[:, :4, :] = True

        kind = torch.tensor([0])
        normalizer.normalize(x, mask, kind)

        assert torch.isclose(normalizer.mean[0, 0], torch.tensor(100.0), atol=1e-5)

    def test_per_kind_independence(self, normalizer):
        normalizer.train()
        mask = torch.ones(1, 8, 8, dtype=torch.bool)

        x0 = torch.ones(1, 2, 8, 8) * 5.0
        normalizer.normalize(x0, mask, torch.tensor([0]))

        x1 = torch.ones(1, 2, 8, 8) * 50.0
        normalizer.normalize(x1, mask, torch.tensor([1]))

        assert torch.isclose(normalizer.mean[0, 0], torch.tensor(5.0), atol=1e-5)
        assert torch.isclose(normalizer.mean[1, 0], torch.tensor(50.0), atol=1e-5)

    def test_normalize_denormalize_roundtrip(self, normalizer):
        normalizer.train()
        x = torch.randn(2, 2, 16, 16) * 3.0 + 7.0
        mask = torch.ones(2, 16, 16, dtype=torch.bool)
        kind = torch.tensor([0, 0])

        x_norm = normalizer.normalize(x, mask, kind)
        x_denorm = normalizer.denormalize(x_norm, kind)

        assert torch.allclose(x, x_denorm, atol=1e-5)

    def test_eval_mode_freezes_updates(self, normalizer):
        normalizer.train()
        mask = torch.ones(1, 8, 8, dtype=torch.bool)
        kind = torch.tensor([0])

        x1 = torch.ones(1, 2, 8, 8) * 10.0
        normalizer.normalize(x1, mask, kind)

        normalizer.eval()

        x2 = torch.ones(1, 2, 8, 8) * 100.0
        normalizer.normalize(x2, mask, kind)

        assert normalizer.n[0].item() == 1

    def test_state_dict_persistence(self, normalizer):
        normalizer.train()
        x = torch.ones(1, 2, 8, 8) * 42.0
        mask = torch.ones(1, 8, 8, dtype=torch.bool)
        kind = torch.tensor([0])

        normalizer.normalize(x, mask, kind)
        state = normalizer.state_dict()

        new_normalizer = RunningNormalizer(num_kinds=3, momentum=0.1)
        new_normalizer.load_state_dict(state)

        assert new_normalizer.n[0].item() == 1
        assert torch.isclose(new_normalizer.mean[0, 0], torch.tensor(42.0), atol=1e-5)

    def test_separate_channel_stats(self, normalizer):
        normalizer.train()
        x = torch.zeros(1, 2, 8, 8)
        x[:, 0] = 10.0
        x[:, 1] = 20.0
        mask = torch.ones(1, 8, 8, dtype=torch.bool)
        kind = torch.tensor([0])

        normalizer.normalize(x, mask, kind)

        assert torch.isclose(normalizer.mean[0, 0], torch.tensor(10.0), atol=1e-5)
        assert torch.isclose(normalizer.mean[0, 1], torch.tensor(20.0), atol=1e-5)

    def test_real_kind_imag_channel_frozen(self):
        """For real kinds (non-complex), imag channel stats stay at mean=0, var=1."""
        from score_qmri.datasets.kind import NUM_DATASET_KINDS

        normalizer = RunningNormalizer(num_kinds=NUM_DATASET_KINDS, momentum=0.1)
        normalizer.train()
        # kind=6 is T1MAP (real kind)
        x = torch.zeros(1, 2, 8, 8)
        x[:, 0] = 10.0
        x[:, 1] = 20.0  # Imag channel has non-zero data but should be ignored
        mask = torch.ones(1, 8, 8, dtype=torch.bool)
        kind = torch.tensor([6])

        normalizer.normalize(x, mask, kind)

        assert torch.isclose(normalizer.mean[6, 0], torch.tensor(10.0), atol=1e-5)
        assert torch.isclose(normalizer.mean[6, 1], torch.tensor(0.0), atol=1e-5)
        assert torch.isclose(normalizer.var[6, 1], torch.tensor(1.0), atol=1e-5)

    def test_normalize_produces_zero_mean(self):
        normalizer = RunningNormalizer(num_kinds=1, momentum=0.5)
        normalizer.train()

        x = torch.randn(4, 2, 8, 8) * 3.0 + 10.0
        mask = torch.ones(4, 8, 8, dtype=torch.bool)
        kind = torch.tensor([0, 0, 0, 0])

        x_norm = normalizer.normalize(x, mask, kind)

        assert x_norm.mean().abs() < 0.5

    def test_mixed_kinds_in_batch(self, normalizer):
        normalizer.train()
        x = torch.zeros(4, 2, 8, 8)
        x[:2] = 10.0
        x[2:] = 50.0
        mask = torch.ones(4, 8, 8, dtype=torch.bool)
        kind = torch.tensor([0, 0, 1, 1])

        normalizer.normalize(x, mask, kind)

        assert torch.isclose(normalizer.mean[0, 0], torch.tensor(10.0), atol=1e-5)
        assert torch.isclose(normalizer.mean[1, 0], torch.tensor(50.0), atol=1e-5)
