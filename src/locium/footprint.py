"""The building's plan: one connected mass of blocks, not a filled rectangle.

Each block shares a full edge with the core, so the building reads as a single
structure. The blocks sit at different depths and do not line up, so the outline
steps in and out rather than squaring off.

Sizes are log-compressed throughout. One wing holds 88% of the real store, and
area proportional to count makes most chambers smaller than their own labels.
A floorplan does not size rooms by how full they are; density shows that.
"""

import math

import squarify

from .models import Rect


class EmptyBlock(Exception):
    """Raised when a block would be left with no wings, erasing its step."""


# index 0 is the core; the rest each share a full edge with it
BLOCKS = [
    Rect(168.0, 296.0, 496.0, 424.0),   # core
    Rect(262.0, 128.0, 402.0, 168.0),   # north, set back from the core's face
    Rect(664.0, 366.0, 236.0, 306.0),   # east, projecting
    Rect(238.0, 720.0, 372.0, 158.0),   # south, offset left
    Rect(76.0, 392.0, 92.0, 236.0),     # west annexe, shallow
]


def subdivide(weights: dict[str, int], rect: Rect, pad: float) -> dict[str, Rect]:
    """Squarify members into a rect, log-compressed and in a stable order."""
    live = [n for n in sorted(weights, key=lambda k: (-weights[k], k)) if weights[n] > 0]
    if not live:
        return {}

    vals = [math.log(weights[n] + 1.0) + 0.9 for n in live]
    aw = max(rect.w - 2 * pad, 1.0)
    ah = max(rect.h - 2 * pad, 1.0)
    raw = squarify.squarify(
        squarify.normalize_sizes(vals, aw, ah), rect.x + pad, rect.y + pad, aw, ah
    )
    return {n: Rect(r["x"], r["y"], r["dx"], r["dy"]) for n, r in zip(live, raw)}


def building_footprint(counts: dict[str, int]) -> dict[str, Rect]:
    """Place every wing into the building.

    The largest wing takes the core on its own. The rest are seeded one per
    remaining block — largest block first — then the remainder goes to whichever
    block is furthest below its share of floor area.
    """
    if not counts:
        return {}

    names = sorted(counts, key=lambda k: (-counts[k], k))
    placed = {names[0]: BLOCKS[0]}
    rest = names[1:]
    if not rest:
        return placed

    ranges = BLOCKS[1:]
    areas = [b.area for b in ranges]
    weights = {n: math.log(counts[n] + 1.0) + 1.0 for n in rest}

    groups: list[list[str]] = [[] for _ in ranges]
    cursor = 0
    for k in sorted(range(len(ranges)), key=lambda j: -areas[j]):
        if cursor < len(rest):
            groups[k].append(rest[cursor])
            cursor += 1

    load = [sum(weights[n] for n in g) for g in groups]
    while cursor < len(rest):
        k = min(range(len(ranges)), key=lambda j: load[j] / areas[j])
        groups[k].append(rest[cursor])
        load[k] += weights[rest[cursor]]
        cursor += 1

    for block, group in zip(ranges, groups):
        if not group:
            continue
        placed.update(subdivide({n: counts[n] for n in group}, block, 0.0))
    return placed
