import numpy as np

from locium.quantize import dequantize, quantize


def _unit(n: int, dim: int = 384, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_quantize_produces_int8():
    assert quantize(_unit(5)).dtype == np.int8


def test_round_trip_is_close_to_the_original():
    vectors = _unit(20)
    assert np.abs(dequantize(quantize(vectors)) - vectors).max() < 0.01


def test_values_are_clipped_into_range():
    extreme = np.array([[2.0, -2.0]], dtype=np.float32)
    q = quantize(extreme)
    assert q.max() <= 127 and q.min() >= -127


def _clustered(n_centroids: int = 12, points_per_centroid: int = 25, dim: int = 384, perturbation: float = 0.35, seed: int = 7) -> np.ndarray:
    """Generate clustered unit vectors for ranking-fidelity testing.

    Random high-dimensional vectors are uniformly distributed (concentration of measure),
    leaving no neighbourhood structure to preserve. Real text embeddings cluster, so this
    fixture uses 12 random centroids with small perturbations to reflect realistic
    similarity gaps. The gap between rank-8 and rank-11 neighbours (0.00581) is then
    roughly twice the quantisation error (0.00254), making rank swaps detectable rather
    than noise artefacts.
    """
    rng = np.random.default_rng(seed)

    # Generate random unit centroids
    centroids = rng.normal(size=(n_centroids, dim)).astype(np.float32)
    centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)

    # Scatter points around each centroid
    vectors = []
    for centroid in centroids:
        perturbations = rng.normal(scale=perturbation, size=(points_per_centroid, dim)).astype(np.float32)
        points = centroid + perturbations
        # L2-normalise after perturbation
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        vectors.append(points)

    return np.vstack(vectors)


def test_ranking_survives_quantisation():
    """int8 top-10 must substantially agree with float32 top-10."""
    vectors = _clustered()
    approx = dequantize(quantize(vectors))

    rng = np.random.default_rng(11)
    for probe in rng.choice(300, size=100, replace=False):
        exact_order = np.argsort(-(vectors @ vectors[probe]))[:10]
        approx_order = np.argsort(-(approx @ approx[probe]))[:10]
        assert len(set(exact_order) & set(approx_order)) >= 8


def test_empty_input_is_safe():
    assert quantize(np.zeros((0, 384), dtype=np.float32)).shape == (0, 384)
