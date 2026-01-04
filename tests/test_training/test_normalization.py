"""Tests for RunningNormalizer."""

import pytest
import torch

from score_qmri.models.normalization import RunningNormalizer


@pytest.fixture
def normalizer():
    """Create normalizer with 3 kinds for testing."""
    return RunningNormalizer(num_kinds=3, momentum=0.1)


class TestRunningNormalizer:
    """Tests for RunningNormalizer."""

    def test_first_update_initializes_stats(self, normalizer):
        """First batch sets mean/var directly without EMA blend."""
        normalizer.train()
        x = torch.ones(2, 2, 8, 8) * 5.0
        x[:, 1] = 10.0
        mask = torch.ones(2, 8, 8, dtype=torch.bool)
        kind = torch.tensor([0, 0])

        normalizer.normalize(x, mask, kind)

        mean_0, var_0, n_0 = normalizer._get_buffers(0)
        assert n_0.item() == 1
        assert torch.isclose(mean_0[0], torch.tensor(5.0), atol=1e-5)
        assert torch.isclose(mean_0[1], torch.tensor(10.0), atol=1e-5)

    def test_ema_update_blends_correctly(self, normalizer):
        """Subsequent batches blend with momentum."""
        normalizer.train()
        mask = torch.ones(2, 8, 8, dtype=torch.bool)
        kind = torch.tensor([0, 0])

        x1 = torch.ones(2, 2, 8, 8) * 10.0
        normalizer.normalize(x1, mask, kind)

        x2 = torch.ones(2, 2, 8, 8) * 20.0
        normalizer.normalize(x2, mask, kind)

        mean_0, _, n_0 = normalizer._get_buffers(0)
        assert n_0.item() == 2
        expected = 10.0 * 0.9 + 20.0 * 0.1
        assert torch.isclose(mean_0[0], torch.tensor(expected), atol=1e-5)

    def test_mask_excludes_invalid_pixels(self, normalizer):
        """Only masked-in pixels contribute to statistics."""
        normalizer.train()
        x = torch.zeros(1, 2, 8, 8)
        x[:, :, :4, :] = 100.0
        x[:, :, 4:, :] = 0.0

        mask = torch.zeros(1, 8, 8, dtype=torch.bool)
        mask[:, :4, :] = True

        kind = torch.tensor([0])
        normalizer.normalize(x, mask, kind)

        mean_0, _, _ = normalizer._get_buffers(0)
        assert torch.isclose(mean_0[0], torch.tensor(100.0), atol=1e-5)

    def test_per_kind_independence(self, normalizer):
        """Different kinds maintain separate statistics."""
        normalizer.train()
        mask = torch.ones(1, 8, 8, dtype=torch.bool)

        x0 = torch.ones(1, 2, 8, 8) * 5.0
        normalizer.normalize(x0, mask, torch.tensor([0]))

        x1 = torch.ones(1, 2, 8, 8) * 50.0
        normalizer.normalize(x1, mask, torch.tensor([1]))

        mean_0, _, _ = normalizer._get_buffers(0)
        mean_1, _, _ = normalizer._get_buffers(1)

        assert torch.isclose(mean_0[0], torch.tensor(5.0), atol=1e-5)
        assert torch.isclose(mean_1[0], torch.tensor(50.0), atol=1e-5)

    def test_normalize_denormalize_roundtrip(self, normalizer):
        """denormalize(normalize(x)) approximately equals x."""
        normalizer.train()
        x = torch.randn(2, 2, 16, 16) * 3.0 + 7.0
        mask = torch.ones(2, 16, 16, dtype=torch.bool)
        kind = torch.tensor([0, 0])

        x_norm = normalizer.normalize(x, mask, kind)
        x_denorm = normalizer.denormalize(x_norm, kind)

        assert torch.allclose(x, x_denorm, atol=1e-5)

    def test_eval_mode_freezes_updates(self, normalizer):
        """Stats don't update when module is in eval mode."""
        normalizer.train()
        mask = torch.ones(1, 8, 8, dtype=torch.bool)
        kind = torch.tensor([0])

        x1 = torch.ones(1, 2, 8, 8) * 10.0
        normalizer.normalize(x1, mask, kind)

        normalizer.eval()

        x2 = torch.ones(1, 2, 8, 8) * 100.0
        normalizer.normalize(x2, mask, kind)

        _, _, n_0 = normalizer._get_buffers(0)
        assert n_0.item() == 1

    def test_state_dict_persistence(self, normalizer):
        """Stats survive save/load cycle."""
        normalizer.train()
        x = torch.ones(1, 2, 8, 8) * 42.0
        mask = torch.ones(1, 8, 8, dtype=torch.bool)
        kind = torch.tensor([0])

        normalizer.normalize(x, mask, kind)
        state = normalizer.state_dict()

        new_normalizer = RunningNormalizer(num_kinds=3, momentum=0.1)
        new_normalizer.load_state_dict(state)

        mean_0, _, n_0 = new_normalizer._get_buffers(0)
        assert n_0.item() == 1
        assert torch.isclose(mean_0[0], torch.tensor(42.0), atol=1e-5)

    def test_separate_channel_stats(self, normalizer):
        """Real and imaginary channels have independent stats."""
        normalizer.train()
        x = torch.zeros(1, 2, 8, 8)
        x[:, 0] = 10.0
        x[:, 1] = 20.0
        mask = torch.ones(1, 8, 8, dtype=torch.bool)
        kind = torch.tensor([0])

        normalizer.normalize(x, mask, kind)

        mean_0, _, _ = normalizer._get_buffers(0)
        assert torch.isclose(mean_0[0], torch.tensor(10.0), atol=1e-5)
        assert torch.isclose(mean_0[1], torch.tensor(20.0), atol=1e-5)

    def test_single_channel_input(self, normalizer):
        """Single channel input only updates first channel stats."""
        normalizer.train()
        x = torch.ones(1, 1, 8, 8) * 15.0
        mask = torch.ones(1, 8, 8, dtype=torch.bool)
        kind = torch.tensor([0])

        normalizer.normalize(x, mask, kind)

        mean_0, var_0, _ = normalizer._get_buffers(0)
        assert torch.isclose(mean_0[0], torch.tensor(15.0), atol=1e-5)
        assert torch.isclose(mean_0[1], torch.tensor(0.0), atol=1e-5)
        assert torch.isclose(var_0[1], torch.tensor(1.0), atol=1e-5)

    def test_denormalize_single_channel_kind(self, normalizer):
        """Denormalize works correctly for kinds trained with 1 channel."""
        normalizer.train()

        x_train = torch.randn(4, 1, 8, 8) * 2.0 + 10.0
        mask = torch.ones(4, 8, 8, dtype=torch.bool)
        kind = torch.tensor([0, 0, 0, 0])

        normalizer.normalize(x_train, mask, kind)

        x_sample = torch.zeros(1, 2, 8, 8)
        x_denorm = normalizer.denormalize(x_sample, torch.tensor([0]))

        mean_0, _, _ = normalizer._get_buffers(0)
        assert torch.isclose(x_denorm[0, 0].mean(), mean_0[0], atol=0.1)
        assert torch.isclose(x_denorm[0, 1].mean(), torch.tensor(0.0), atol=1e-5)

    def test_normalize_produces_zero_mean(self):
        """After first update, normalization produces approximately zero mean."""
        normalizer = RunningNormalizer(num_kinds=1, momentum=0.5)
        normalizer.train()

        x = torch.randn(4, 2, 8, 8) * 3.0 + 10.0
        mask = torch.ones(4, 8, 8, dtype=torch.bool)
        kind = torch.tensor([0, 0, 0, 0])

        x_norm = normalizer.normalize(x, mask, kind)

        assert x_norm.mean().abs() < 0.5

    def test_mixed_kinds_in_batch(self, normalizer):
        """Batch with multiple kinds updates each correctly."""
        normalizer.train()
        x = torch.zeros(4, 2, 8, 8)
        x[:2] = 10.0
        x[2:] = 50.0
        mask = torch.ones(4, 8, 8, dtype=torch.bool)
        kind = torch.tensor([0, 0, 1, 1])

        normalizer.normalize(x, mask, kind)

        mean_0, _, _ = normalizer._get_buffers(0)
        mean_1, _, _ = normalizer._get_buffers(1)

        assert torch.isclose(mean_0[0], torch.tensor(10.0), atol=1e-5)
        assert torch.isclose(mean_1[0], torch.tensor(50.0), atol=1e-5)
