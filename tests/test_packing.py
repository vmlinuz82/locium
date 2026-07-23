import pytest

from locium.models import Rect
from locium.packing import pack_chamber

RECT = Rect(100.0, 200.0, 60.0, 40.0)


def test_returns_one_point_per_drawer():
    assert len(pack_chamber(25, RECT, seed=42)) == 25


def test_points_stay_inside_the_chamber():
    for x, y in pack_chamber(40, RECT, seed=42):
        assert RECT.x <= x <= RECT.x + RECT.w
        assert RECT.y <= y <= RECT.y + RECT.h


def test_packing_is_deterministic():
    assert pack_chamber(30, RECT, seed=42) == pack_chamber(30, RECT, seed=42)


def test_a_different_seed_packs_differently():
    assert pack_chamber(30, RECT, seed=42) != pack_chamber(30, RECT, seed=7)


def test_zero_drawers_is_empty():
    assert pack_chamber(0, RECT, seed=42) == []


def test_existing_points_are_returned_unchanged():
    first = pack_chamber(10, RECT, seed=42)
    grown = pack_chamber(14, RECT, seed=42, placed=first)
    assert grown[:10] == first
    assert len(grown) == 14


def test_growth_does_not_stack_new_points_on_old():
    first = pack_chamber(8, RECT, seed=42)
    grown = pack_chamber(12, RECT, seed=42, placed=first)
    for nx, ny in grown[8:]:
        assert all(abs(nx - ox) > 1e-9 or abs(ny - oy) > 1e-9 for ox, oy in first)


def test_a_tiny_chamber_still_packs():
    assert len(pack_chamber(5, Rect(0.0, 0.0, 2.0, 2.0), seed=42)) == 5
