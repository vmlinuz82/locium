import numpy as np
import pytest

from locium.stability import merge_coords, place_into_existing


def _unit(rows) -> np.ndarray:
    v = np.asarray(rows, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_merge_keeps_existing_coordinates_untouched():
    previous = {"a": [1.0, 2.0], "b": [3.0, 4.0]}
    fresh = {"a": [99.0, 99.0], "b": [88.0, 88.0], "c": [5.0, 6.0]}
    merged = merge_coords(previous, fresh, refit=False)
    assert merged["a"] == [1.0, 2.0]
    assert merged["b"] == [3.0, 4.0]


def test_merge_adds_new_drawers():
    merged = merge_coords({"a": [1.0, 2.0]}, {"a": [9.0, 9.0], "c": [5.0, 6.0]}, refit=False)
    assert merged["c"] == [5.0, 6.0]


def test_merge_drops_deleted_drawers():
    merged = merge_coords({"a": [1.0, 2.0], "gone": [7.0, 7.0]}, {"a": [9.0, 9.0]}, refit=False)
    assert "gone" not in merged


def test_refit_replaces_everything():
    merged = merge_coords({"a": [1.0, 2.0]}, {"a": [9.0, 9.0]}, refit=True)
    assert merged["a"] == [9.0, 9.0]


def test_new_drawer_lands_near_its_nearest_placed_neighbour():
    placed_vectors = _unit([[1.0, 0.0], [0.0, 1.0]])
    placed_coords = np.array([[0.0, 0.0], [500.0, 500.0]], dtype=np.float32)
    new_vectors = _unit([[0.99, 0.01]])

    coords = place_into_existing(
        new_vectors, placed_vectors, placed_coords, seed=42, k=1, jitter=0.0
    )
    assert np.allclose(coords[0], [0.0, 0.0], atol=1e-3)


def test_placement_is_deterministic_under_a_fixed_seed():
    placed_vectors = _unit(np.random.default_rng(1).normal(size=(20, 8)))
    placed_coords = np.random.default_rng(2).uniform(0, 500, size=(20, 2)).astype(np.float32)
    new_vectors = _unit(np.random.default_rng(3).normal(size=(5, 8)))

    first = place_into_existing(new_vectors, placed_vectors, placed_coords, seed=42)
    second = place_into_existing(new_vectors, placed_vectors, placed_coords, seed=42)
    assert np.array_equal(first, second)


def test_placing_nothing_returns_an_empty_array():
    placed_vectors = _unit([[1.0, 0.0]])
    placed_coords = np.array([[0.0, 0.0]], dtype=np.float32)
    out = place_into_existing(
        np.zeros((0, 2), dtype=np.float32), placed_vectors, placed_coords, seed=42
    )
    assert out.shape == (0, 2)


def test_k_larger_than_placed_drawers_does_not_raise():
    placed_vectors = _unit([[1.0, 0.0], [0.0, 1.0]])
    placed_coords = np.array([[0.0, 0.0], [500.0, 500.0]], dtype=np.float32)
    new_vectors = _unit([[0.99, 0.01], [0.5, 0.5]])

    coords = place_into_existing(
        new_vectors, placed_vectors, placed_coords, seed=42, k=10, jitter=0.0
    )
    assert coords.shape == (2, 2)


def test_placing_into_an_empty_space_raises():
    placed_vectors = np.zeros((0, 8), dtype=np.float32)
    placed_coords = np.zeros((0, 2), dtype=np.float32)
    new_vectors = _unit([[1.0, 0.0]])

    with pytest.raises(ValueError):
        place_into_existing(new_vectors, placed_vectors, placed_coords, seed=42)


def test_non_positive_similarity_preserves_ordering():
    # Both neighbours are non-positive similarity but differ in magnitude:
    # sim ~ -0.2 (less dissimilar) vs sim ~ -0.8 (more dissimilar). The old
    # clip-to-1e-6 implementation floors both to the same value, collapsing
    # them to equal weights (an unweighted centroid); the remap keeps the
    # -0.2 neighbour's coordinate closer.
    placed_vectors = _unit([[-0.2, 0.9798], [-0.8, 0.6]])
    placed_coords = np.array([[0.0, 0.0], [1000.0, 1000.0]], dtype=np.float32)
    new_vectors = _unit([[1.0, 0.0]])

    coords = place_into_existing(
        new_vectors, placed_vectors, placed_coords, seed=42, k=2, jitter=0.0
    )

    near = placed_coords[0]
    far = placed_coords[1]
    dist_to_near = np.linalg.norm(coords[0] - near)
    dist_to_far = np.linalg.norm(coords[0] - far)
    assert dist_to_near < dist_to_far
