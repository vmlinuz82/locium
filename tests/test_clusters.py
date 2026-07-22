import numpy as np
import pytest

from locium.clusters import cluster_labels


def _two_blobs(per_blob: int = 12):
    rng = np.random.default_rng(0)
    left = rng.normal(loc=[0.0, 0.0], scale=1.0, size=(per_blob, 2))
    right = rng.normal(loc=[100.0, 100.0], scale=1.0, size=(per_blob, 2))
    coords = np.vstack([left, right]).astype(np.float32)
    texts = ["docker compose container build"] * per_blob + [
        "voucher affiliate discount checkout"
    ] * per_blob
    return coords, texts


def test_finds_two_clusters():
    clusters, assignments = cluster_labels(*_two_blobs(), min_cluster_size=5)
    assert len(clusters) == 2
    assert set(assignments) == {0, 1}


def test_labels_use_distinguishing_terms():
    clusters, _ = cluster_labels(*_two_blobs(), min_cluster_size=5)
    joined = " ".join(c["label"] for c in clusters)
    assert "docker" in joined
    assert "voucher" in joined


def test_centroid_sits_inside_its_blob():
    clusters, assignments = cluster_labels(*_two_blobs(), min_cluster_size=5)
    coords, _ = _two_blobs()
    for c in clusters:
        members = coords[assignments == c["cluster"]]
        assert abs(c["centroid"][0] - members[:, 0].mean()) < 1e-3


def test_too_few_points_yields_no_clusters():
    coords = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    clusters, assignments = cluster_labels(coords, ["a", "b"], min_cluster_size=5)
    assert clusters == []
    assert list(assignments) == [-1, -1]


def test_empty_input_is_safe():
    clusters, assignments = cluster_labels(np.zeros((0, 2), dtype=np.float32), [])
    assert clusters == []
    assert len(assignments) == 0


def test_stop_word_only_text_yields_empty_labels():
    coords, _ = _two_blobs()
    texts = ["a is to of an it be"] * 12 + ["the or if to a an"] * 12
    clusters, _ = cluster_labels(coords, texts, min_cluster_size=5)
    assert len(clusters) == 2
    assert all(c["label"] == "" for c in clusters)


def test_mismatched_lengths_raise_value_error():
    coords, texts = _two_blobs()
    with pytest.raises(ValueError) as excinfo:
        cluster_labels(coords, texts[:12], min_cluster_size=5)
    assert "24" in str(excinfo.value)
    assert "12" in str(excinfo.value)
