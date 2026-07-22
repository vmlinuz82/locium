"""Macro layout: a squarified treemap allocates each wing a rectangle.

Wings are laid out in alphabetical order so the partition does not reshuffle
when drawer counts drift. A gutter strip is reserved at the bottom of the
canvas so wings added later can be placed without moving existing ones.
"""

import squarify

from .config import MIN_WING_SIDE
from .models import Rect


class RefitRequired(Exception):
    """Raised when new wings cannot be placed without a full re-layout."""


def _usable_height(canvas: Rect, gutter_fraction: float) -> float:
    return canvas.h * (1.0 - gutter_fraction)


def _squarify_into(counts: dict[str, int], rect: Rect) -> dict[str, Rect]:
    live = [(name, counts[name]) for name in sorted(counts) if counts[name] > 0]
    if not live:
        return {}

    sizes = [count for _, count in live]
    normed = squarify.normalize_sizes(sizes, rect.w, rect.h)
    raw = squarify.squarify(normed, rect.x, rect.y, rect.w, rect.h)
    return {
        name: Rect(r["x"], r["y"], r["dx"], r["dy"])
        for (name, _), r in zip(live, raw)
    }


def wing_rects(
    counts: dict[str, int], canvas: Rect, gutter_fraction: float
) -> dict[str, Rect]:
    """Allocate each wing a rectangle, area proportional to drawer count."""
    usable = Rect(canvas.x, canvas.y, canvas.w, _usable_height(canvas, gutter_fraction))
    return _squarify_into(counts, usable)


def gutter_rect(canvas: Rect, gutter_fraction: float) -> Rect:
    """The reserved strip where wings discovered after the first build go."""
    usable_h = _usable_height(canvas, gutter_fraction)
    return Rect(canvas.x, canvas.y + usable_h, canvas.w, canvas.h - usable_h)


def carve_gutter(counts: dict[str, int], gutter: Rect) -> dict[str, Rect]:
    """Place newly discovered wings inside the gutter.

    Raises RefitRequired when the gutter is too crowded to give every new wing
    a usable rectangle — a loud failure is better than silently moving every
    existing drawer.
    """
    rects = _squarify_into(counts, gutter)
    for name, rect in rects.items():
        if min(rect.w, rect.h) < MIN_WING_SIDE:
            raise RefitRequired(
                f"wing {name!r} would be {rect.w:.0f}x{rect.h:.0f}, below the "
                f"{MIN_WING_SIDE:.0f} minimum. The gutter is exhausted; "
                f"run: locium build --refit"
            )
    return rects
