import numpy as np
import pytest

from locium.clusters import assign_clusters, cluster_count, label_clusters


def _two_blobs(n_per_blob: int) -> np.ndarray:
    """Two well-separated groups of unit vectors in 8d."""
    rng = np.random.default_rng(7)
    a = rng.normal(loc=(3, 0, 0, 0, 0, 0, 0, 0), scale=0.1, size=(n_per_blob, 8))
    b = rng.normal(loc=(0, 3, 0, 0, 0, 0, 0, 0), scale=0.1, size=(n_per_blob, 8))
    vectors = np.vstack([a, b]).astype(np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def test_cluster_count_grows_slowly_and_respects_the_cap():
    cases = [
        (150, 6, 2),   # the threshold chamber gets the minimum split
        (992, 6, 4),   # the dense qa-workbench chamber
        (1721, 6, 5),  # the largest real chamber
        (100000, 6, 6),  # the cap holds no matter the size
        (200, 3, 2),
    ]
    for n, cap, expected in cases:
        assert cluster_count(n, cap) == expected, (n, cap)


def test_assign_clusters_separates_two_obvious_blobs():
    vectors = _two_blobs(20)
    assignments = assign_clusters(vectors, 2, seed=42)

    first = set(assignments[:20])
    second = set(assignments[20:])
    assert len(first) == 1
    assert len(second) == 1
    assert first != second


def test_assign_clusters_is_deterministic_for_a_seed():
    vectors = _two_blobs(15)
    once = assign_clusters(vectors, 2, seed=42)
    again = assign_clusters(vectors, 2, seed=42)
    assert (once == again).all()


def test_label_clusters_names_what_distinguishes_each_cluster():
    texts = ["deployment pipeline failed on staging server"] * 10 + [
        "invoice payment reconciliation ledger"
    ] * 10
    assignments = np.array([0] * 10 + [1] * 10)

    labels = label_clusters(texts, assignments, 2)

    assert len(labels) == 2
    assert any(word in labels[0] for word in ("deployment", "pipeline", "staging"))
    assert any(word in labels[1] for word in ("invoice", "payment", "ledger"))


def test_label_clusters_degrades_to_empty_labels_on_an_empty_vocabulary():
    # Pure stopword texts produce no TF-IDF vocabulary; the build must not die.
    texts = ["the and of to"] * 6
    assignments = np.array([0, 0, 0, 1, 1, 1])
    assert label_clusters(texts, assignments, 2) == ["", ""]
