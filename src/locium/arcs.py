"""Cross-wing arcs: semantic adjacency the hybrid layout would otherwise hide.

Two drawers sitting in different wings can be nearly identical in meaning.
Because wings occupy fixed regions, that closeness cannot show up as
proximity, so it is recorded as an explicit edge instead. These arcs are also
the candidate set for tunnel confirmation.
"""

import numpy as np


def compute_arcs(
    vectors: np.ndarray,
    wings: list[str],
    max_distance: float,
    per_drawer: int,
    global_cap: int,
    chunk: int = 512,
) -> list[list]:
    """Find each drawer's nearest neighbours in *other* wings.

    Vectors must be L2-normalised, so a dot product is cosine similarity and
    ``1 - similarity`` is cosine distance. Pairs are symmetric and returned
    once, nearest first, capped globally.
    """
    n = len(vectors)
    if n < 2:
        return []

    codes = np.unique(np.asarray(wings), return_inverse=True)[1]
    kth = min(per_drawer, n - 1)
    pairs: dict[tuple[int, int], float] = {}

    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        distances = 1.0 - (vectors[start:stop] @ vectors.T)
        distances[codes[start:stop][:, None] == codes[None, :]] = np.inf

        for row in range(stop - start):
            i = start + row
            candidates = np.argpartition(distances[row], kth)[:kth]
            for j in candidates:
                distance = float(distances[row][j])
                if distance > max_distance:
                    continue
                key = (i, int(j)) if i < int(j) else (int(j), i)
                if key not in pairs or distance < pairs[key]:
                    pairs[key] = distance

    ordered = sorted(pairs.items(), key=lambda item: item[1])[:global_cap]
    return [[src, dst, round(distance, 4)] for (src, dst), distance in ordered]
