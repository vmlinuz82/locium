import pytest

from locium.models import Rect
from locium.treemap import RefitRequired, carve_gutter, gutter_rect, wing_rects

CANVAS = Rect(0.0, 0.0, 1000.0, 1000.0)


def _overlaps(a: Rect, b: Rect) -> bool:
    return not (
        a.x + a.w <= b.x + 1e-6
        or b.x + b.w <= a.x + 1e-6
        or a.y + a.h <= b.y + 1e-6
        or b.y + b.h <= a.y + 1e-6
    )


def test_rects_do_not_overlap():
    rects = list(wing_rects({"a": 100, "b": 50, "c": 25}, CANVAS, 0.15).values())
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            assert not _overlaps(rects[i], rects[j])


def test_area_is_proportional_to_count():
    rects = wing_rects({"big": 200, "small": 50}, CANVAS, 0.15)
    ratio = rects["big"].area / rects["small"].area
    assert ratio == pytest.approx(4.0, rel=0.02)


def test_rects_stay_inside_the_usable_area():
    usable_h = CANVAS.h * 0.85
    for r in wing_rects({"a": 3, "b": 2, "c": 1}, CANVAS, 0.15).values():
        assert r.x >= -1e-6
        assert r.y >= -1e-6
        assert r.x + r.w <= CANVAS.w + 1e-6
        assert r.y + r.h <= usable_h + 1e-6


def test_layout_is_alphabetical_and_stable_under_count_drift():
    before = wing_rects({"alpha": 100, "beta": 50, "gamma": 25}, CANVAS, 0.15)
    after = wing_rects({"alpha": 104, "beta": 50, "gamma": 25}, CANVAS, 0.15)
    assert list(before) == list(after) == ["alpha", "beta", "gamma"]


def test_zero_count_wings_are_dropped():
    assert "empty" not in wing_rects({"a": 5, "empty": 0}, CANVAS, 0.15)


def test_no_wings_yields_empty_mapping():
    assert wing_rects({}, CANVAS, 0.15) == {}


def test_gutter_sits_below_the_usable_area():
    g = gutter_rect(CANVAS, 0.15)
    assert g.y == pytest.approx(850.0)
    assert g.h == pytest.approx(150.0)
    assert g.w == pytest.approx(1000.0)


def test_carve_gutter_places_new_wings():
    rects = carve_gutter({"new_one": 10, "new_two": 10}, gutter_rect(CANVAS, 0.15))
    assert set(rects) == {"new_one", "new_two"}
    for r in rects.values():
        assert r.y >= 850.0 - 1e-6


def test_carve_gutter_refuses_when_slivers_result():
    with pytest.raises(RefitRequired, match="--refit"):
        carve_gutter({f"w{i}": 1 for i in range(400)}, gutter_rect(CANVAS, 0.15))
