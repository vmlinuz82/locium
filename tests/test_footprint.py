import pytest

from locium.footprint import BLOCKS, building_footprint, subdivide
from locium.models import Rect


def _overlaps(a: Rect, b: Rect) -> bool:
    return not (
        a.x + a.w <= b.x + 1e-6 or b.x + b.w <= a.x + 1e-6
        or a.y + a.h <= b.y + 1e-6 or b.y + b.h <= a.y + 1e-6
    )


def _counts(n: int) -> dict[str, int]:
    """One dominant wing plus a long tail — the real palace's shape."""
    sizes = [5785, 199, 197, 131, 90, 52, 34, 31, 12, 9, 8, 7, 6, 4, 4, 2, 1]
    return {f"w{i}": sizes[i % len(sizes)] for i in range(n)}


def test_wings_never_overlap():
    rects = list(building_footprint(_counts(17)).values())
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            assert not _overlaps(rects[i], rects[j])


def test_every_block_is_occupied():
    """An empty block silently deletes its step from the outline."""
    placed = building_footprint(_counts(17)).values()
    for block in BLOCKS:
        assert any(
            r.x >= block.x - 1 and r.y >= block.y - 1
            and r.x + r.w <= block.x + block.w + 1
            and r.y + r.h <= block.y + block.h + 1
            for r in placed
        ), f"block at {block.x},{block.y} got no wings"


def test_largest_wing_takes_the_core_alone():
    rects = building_footprint(_counts(17))
    biggest = max(_counts(17), key=lambda k: _counts(17)[k])
    assert rects[biggest] == BLOCKS[0]


def test_outline_is_not_a_rectangle():
    """Blocks sit at different depths, so the silhouette steps."""
    rects = list(building_footprint(_counts(17)).values())
    tops = {round(r.y) for r in rects}
    lefts = {round(r.x) for r in rects}
    assert len(tops) > 3
    assert len(lefts) > 3


def test_fewer_wings_than_blocks_still_places_them_all():
    rects = building_footprint(_counts(3))
    assert len(rects) == 3


def test_single_wing_takes_the_core():
    rects = building_footprint({"only": 42})
    assert rects["only"] == BLOCKS[0]


def test_no_wings_is_empty():
    assert building_footprint({}) == {}


def test_subdivide_is_deterministic():
    weights = {"a": 100, "b": 30, "c": 4}
    rect = Rect(0.0, 0.0, 200.0, 120.0)
    assert subdivide(weights, rect, 4.0) == subdivide(weights, rect, 4.0)


def test_subdivide_compresses_a_dominant_member():
    """log sizing keeps a tiny room legible beside a huge one."""
    rects = subdivide({"huge": 5000, "tiny": 1}, Rect(0.0, 0.0, 400.0, 400.0), 0.0)
    ratio = rects["huge"].area / rects["tiny"].area
    assert ratio < 12.0


def test_subdivide_stays_inside_its_rect():
    rect = Rect(10.0, 20.0, 300.0, 200.0)
    for r in subdivide({"a": 9, "b": 3, "c": 1}, rect, 5.0).values():
        assert r.x >= rect.x + 5.0 - 1e-6
        assert r.y >= rect.y + 5.0 - 1e-6
        assert r.x + r.w <= rect.x + rect.w - 5.0 + 1e-6
        assert r.y + r.h <= rect.y + rect.h - 5.0 + 1e-6
