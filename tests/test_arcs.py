import numpy as np

from locium.arcs import compute_arcs


def _unit(rows) -> np.ndarray:
    v = np.asarray(rows, dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_arcs_never_connect_drawers_in_the_same_wing():
    vectors = _unit([[1.0, 0.0], [1.0, 0.01], [0.0, 1.0]])
    wings = ["a", "a", "b"]
    for src, dst, _ in compute_arcs(vectors, wings, 2.0, 3, 100):
        assert wings[src] != wings[dst]


def test_close_cross_wing_pair_produces_an_arc():
    vectors = _unit([[1.0, 0.0], [0.999, 0.01]])
    arcs = compute_arcs(vectors, ["a", "b"], 0.45, 3, 100)
    assert len(arcs) == 1
    assert arcs[0][0] == 0 and arcs[0][1] == 1


def test_distant_pairs_are_filtered_by_threshold():
    vectors = _unit([[1.0, 0.0], [0.0, 1.0]])  # cosine distance 1.0
    assert compute_arcs(vectors, ["a", "b"], 0.45, 3, 100) == []


def test_pairs_are_deduplicated_and_ordered():
    vectors = _unit([[1.0, 0.0], [0.99, 0.01], [0.98, 0.02]])
    arcs = compute_arcs(vectors, ["a", "b", "c"], 2.0, 3, 100)
    keys = [(a, b) for a, b, _ in arcs]
    assert len(keys) == len(set(keys))
    for src, dst, _ in arcs:
        assert src < dst


def test_results_are_sorted_nearest_first():
    distances = [d for _, _, d in compute_arcs(
        _unit([[1.0, 0.0], [0.99, 0.02], [0.8, 0.6]]), ["a", "b", "c"], 2.0, 3, 100
    )]
    assert distances == sorted(distances)


def test_global_cap_is_honoured():
    rng = np.random.default_rng(0)
    vectors = _unit(rng.normal(size=(40, 16)))
    wings = ["a" if i % 2 else "b" for i in range(40)]
    assert len(compute_arcs(vectors, wings, 2.0, 3, 5)) == 5


def test_chunking_does_not_change_the_result():
    rng = np.random.default_rng(1)
    vectors = _unit(rng.normal(size=(30, 16)))
    wings = ["a" if i % 3 else "b" for i in range(30)]
    assert compute_arcs(vectors, wings, 2.0, 3, 100, chunk=4) == compute_arcs(
        vectors, wings, 2.0, 3, 100, chunk=1000
    )


def test_empty_input_is_safe():
    assert compute_arcs(np.zeros((0, 8), dtype=np.float32), [], 0.45, 3, 100) == []


def test_single_wing_produces_no_arcs():
    vectors = _unit([[1.0, 0.0], [0.99, 0.01]])
    assert compute_arcs(vectors, ["a", "a"], 2.0, 3, 100) == []


def test_infinite_max_distance_never_connects_same_wing():
    vectors = _unit([[1.0, 0.0], [1.0, 0.01], [0.0, 1.0]])
    wings = ["a", "a", "b"]
    for src, dst, _ in compute_arcs(vectors, wings, float("inf"), 3, 100):
        assert wings[src] != wings[dst]


def test_global_cap_zero_returns_empty_list():
    vectors = _unit([[1.0, 0.0], [0.999, 0.01]])
    assert compute_arcs(vectors, ["a", "b"], 2.0, 3, 0) == []
