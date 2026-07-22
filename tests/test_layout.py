import numpy as np
import pytest

from locium.layout import fit_to_rect, ring_layout, umap_layout
from locium.models import Rect

RECT = Rect(100.0, 200.0, 300.0, 400.0)


def _vectors(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 384)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_ring_layout_of_zero_is_empty():
    assert ring_layout(0, RECT).shape == (0, 2)


def test_ring_layout_of_one_is_centred():
    coords = ring_layout(1, RECT)
    assert coords.shape == (1, 2)
    assert coords[0][0] == pytest.approx(250.0)
    assert coords[0][1] == pytest.approx(400.0)


def test_ring_layout_stays_inside_rect():
    coords = ring_layout(7, RECT)
    assert (coords[:, 0] >= RECT.x).all() and (coords[:, 0] <= RECT.x + RECT.w).all()
    assert (coords[:, 1] >= RECT.y).all() and (coords[:, 1] <= RECT.y + RECT.h).all()


def test_ring_layout_is_deterministic():
    assert np.array_equal(ring_layout(9, RECT), ring_layout(9, RECT))


def test_fit_to_rect_maps_into_padded_bounds():
    raw = np.array([[-5.0, -5.0], [5.0, 5.0], [0.0, 0.0]], dtype=np.float32)
    out = fit_to_rect(raw, RECT, pad=0.05)
    assert out[:, 0].min() == pytest.approx(RECT.x + 15.0)
    assert out[:, 0].max() == pytest.approx(RECT.x + RECT.w - 15.0)
    assert out[:, 1].min() == pytest.approx(RECT.y + 20.0)


def test_fit_to_rect_centres_a_degenerate_axis():
    raw = np.array([[1.0, 0.0], [1.0, 5.0]], dtype=np.float32)
    out = fit_to_rect(raw, RECT, pad=0.05)
    assert out[0][0] == pytest.approx(RECT.x + RECT.w / 2)
    assert out[1][0] == pytest.approx(RECT.x + RECT.w / 2)


def test_tiny_wings_use_the_ring_fallback():
    for n in (0, 1, 2, 4):
        coords = umap_layout(_vectors(n), RECT, seed=42, threshold=10)
        assert coords.shape == (n, 2)
        assert np.array_equal(coords, ring_layout(n, RECT))


def test_umap_runs_above_the_threshold_and_is_deterministic():
    vectors = _vectors(40)
    first = umap_layout(vectors, RECT, seed=42, threshold=10)
    second = umap_layout(vectors, RECT, seed=42, threshold=10)
    assert first.shape == (40, 2)
    assert np.array_equal(first, second)
    assert (first[:, 0] >= RECT.x).all() and (first[:, 0] <= RECT.x + RECT.w).all()
