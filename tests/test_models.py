from locium.models import Drawer, Rect, make_preview


def test_rect_area():
    assert Rect(0.0, 0.0, 4.0, 5.0).area == 20.0


def test_drawer_fields():
    d = Drawer(
        id="abc",
        text="hello",
        wing="wing_kosio",
        hall="memory",
        room="diary",
        created_at="2026-05-02T14:59:47",
        source_file="x.jsonl",
    )
    assert d.wing == "wing_kosio"
    assert d.hall == "memory"
    assert d.room == "diary"


def test_preview_collapses_whitespace():
    assert make_preview("a\n\n  b\tc", limit=200) == "a b c"


def test_preview_returns_short_text_unchanged():
    assert make_preview("short text", limit=200) == "short text"


def test_preview_cuts_at_sentence_boundary():
    text = "First sentence. " + ("x" * 300)
    assert make_preview(text, limit=200) == "First sentence."


def test_preview_hard_cuts_when_no_boundary():
    out = make_preview("y" * 300, limit=50)
    assert len(out) == 51  # 50 chars plus the ellipsis
    assert out.endswith("…")
