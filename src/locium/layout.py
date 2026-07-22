"""Micro layout: position drawers inside their wing's rectangle.

UMAP is fitted per wing rather than globally. One room holds the majority of
the palace, and a global fit would let that mass dominate the manifold and
squash every other wing into the margins.
"""

import numpy as np

from .models import Rect


def ring_layout(n: int, rect: Rect) -> np.ndarray:
    """Deterministic fallback for wings too small for UMAP to mean anything."""
    if n <= 0:
        return np.zeros((0, 2), dtype=np.float32)

    cx = rect.x + rect.w / 2.0
    cy = rect.y + rect.h / 2.0
    if n == 1:
        return np.array([[cx, cy]], dtype=np.float32)

    radius = 0.35 * min(rect.w, rect.h)
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return np.stack(
        [cx + radius * np.cos(angles), cy + radius * np.sin(angles)], axis=1
    ).astype(np.float32)


def fit_to_rect(coords: np.ndarray, rect: Rect, pad: float = 0.05) -> np.ndarray:
    """Scale arbitrary 2D coordinates into a padded rectangle.

    An axis with no spread (every point identical) is centred rather than
    collapsed onto the left or bottom edge.
    """
    if len(coords) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    span = hi - lo
    unit = np.where(span > 1e-9, (coords - lo) / np.where(span > 1e-9, span, 1.0), 0.5)

    pad_x = rect.w * pad
    pad_y = rect.h * pad
    x = rect.x + pad_x + unit[:, 0] * (rect.w - 2.0 * pad_x)
    y = rect.y + pad_y + unit[:, 1] * (rect.h - 2.0 * pad_y)
    return np.stack([x, y], axis=1).astype(np.float32)


def umap_layout(
    vectors: np.ndarray, rect: Rect, seed: int, threshold: int
) -> np.ndarray:
    """Project a wing's vectors into its rectangle.

    Wings below ``threshold`` drawers use the ring fallback: UMAP requires
    n_neighbors < n_samples and produces nothing meaningful at that size.
    """
    n = len(vectors)
    if n < threshold:
        return ring_layout(n, rect)

    import umap

    reducer = umap.UMAP(
        n_components=2,
        random_state=seed,
        n_neighbors=min(15, n - 1),
        metric="cosine",
    )
    projected = np.asarray(reducer.fit_transform(vectors), dtype=np.float32)
    return fit_to_rect(projected, rect)
