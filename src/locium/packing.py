"""Scatter a chamber's drawers inside its walls.

Candidates are sampled and the one furthest from the nearest already-placed dot
wins, which reads as packed rather than typeset. Only a recent window of placed
points is consulted, so packing a full chamber stays linear rather than
quadratic.

Passing ``placed`` returns those coordinates untouched and packs new dots around
them: this is where coordinate stability lives now that nothing is projected.
"""

import random

from .models import Rect

CANDIDATES = 9
NEIGHBOUR_WINDOW = 42
INSET = 2.6


def pack_chamber(
    n: int,
    rect: Rect,
    seed: int,
    placed: list[tuple[float, float]] | None = None,
) -> list[tuple[float, float]]:
    if n <= 0:
        return []

    out = list(placed or [])[:n]
    if len(out) >= n:
        return out

    rng = random.Random(seed)
    # Burn the draws that produced the existing points so growth is stable.
    for _ in range(len(out) * CANDIDATES * 2):
        rng.random()

    inset = min(INSET, rect.w / 3.0, rect.h / 3.0)
    ax, ay = rect.x + inset, rect.y + inset
    aw = max(rect.w - 2 * inset, 1e-6)
    ah = max(rect.h - 2 * inset, 1e-6)

    while len(out) < n:
        best, best_d = None, -1.0
        for _ in range(CANDIDATES):
            px, py = ax + rng.random() * aw, ay + rng.random() * ah
            if not out:
                best = (px, py)
                break
            window = out[-NEIGHBOUR_WINDOW:]
            d = min((px - qx) ** 2 + (py - qy) ** 2 for qx, qy in window)
            if d > best_d:
                best, best_d = (px, py), d
        out.append(best)
    return out
