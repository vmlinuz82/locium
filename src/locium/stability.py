"""Keep loci stable across rebuilds.

A memory palace depends on things staying where you left them. Once a drawer
has a coordinate it keeps it forever; new drawers are placed into the space
that already exists rather than triggering a re-projection. Only an explicit
--refit moves anything.
"""

import numpy as np


def place_into_existing(
    new_vectors: np.ndarray,
    placed_vectors: np.ndarray,
    placed_coords: np.ndarray,
    seed: int,
    k: int = 5,
    jitter: float = 1.5,
) -> np.ndarray:
    """Position new drawers near their nearest already-placed neighbours.

    Vectors are assumed L2-normalised, so a dot product is cosine similarity
    in the signed range [-1, 1]. Similarities are remapped from [-1, 1] onto
    [0, 1] before normalising into weights, so ordering among neighbours is
    preserved across the whole signed range, not just the positive half. A
    small deterministic jitter keeps identical drawers from stacking into a
    single unclickable point.
    """
    if len(new_vectors) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    if len(placed_vectors) == 0:
        raise ValueError("cannot place into an empty space")

    neighbours = min(k, len(placed_vectors))
    sims = new_vectors @ placed_vectors.T
    top = np.argsort(-sims, axis=1)[:, :neighbours]

    weights = (1.0 + np.take_along_axis(sims, top, axis=1)) / 2.0
    row_sums = weights.sum(axis=1, keepdims=True)
    uniform = np.full_like(weights, 1.0 / neighbours)
    weights = np.where(row_sums > 0, weights / np.where(row_sums > 0, row_sums, 1.0), uniform)
    coords = (placed_coords[top] * weights[:, :, None]).sum(axis=1)

    if jitter > 0:
        rng = np.random.default_rng(seed)
        coords = coords + rng.normal(0.0, jitter, coords.shape)
    return coords.astype(np.float32)


def merge_coords(
    previous: dict[str, list[float]],
    fresh: dict[str, list[float]],
    refit: bool,
) -> dict[str, list[float]]:
    """Combine persisted coordinates with freshly computed ones.

    Without ``refit``, previously placed drawers keep their exact coordinates
    and only drawers absent from the index take their fresh position. Drawers
    that have left the palace are dropped.
    """
    if refit:
        return dict(fresh)

    merged = {did: xy for did, xy in previous.items() if did in fresh}
    for did, xy in fresh.items():
        merged.setdefault(did, xy)
    return merged
