"""Tests for FiLM conditioning."""

import torch

from models.film_conditioning import FilmConditioning, apply_film


def test_apply_film_scales_and_shifts():
    """Test apply_film applies scale and shift correctly."""
    features = torch.ones(2, 4, 8, 8)
    scale = torch.full((2, 4), 2.0)
    shift = torch.full((2, 4), 1.0)

    out = apply_film(features, scale, shift)
    expected = 2.0 * 1.0 + 1.0  # scale * features + shift
    assert torch.allclose(out, torch.full_like(out, expected))


def test_film_conditioning_output_shape():
    """Test FilmConditioning outputs correct shapes."""
    film = FilmConditioning(num_kinds=3, num_blocks=4)
    kind = torch.tensor([0, 1, 2])

    out = film(kind)
    assert out["scales"].shape == (3, 4)
    assert out["shifts"].shape == (3, 4)


def test_film_conditioning_different_kinds_differ():
    """Test different kinds produce different parameters."""
    film = FilmConditioning(num_kinds=3, num_blocks=2)

    out0 = film(torch.tensor([0]))
    out1 = film(torch.tensor([1]))

    # Parameters should differ for different kinds
    assert not torch.allclose(out0["scales"], out1["scales"])
