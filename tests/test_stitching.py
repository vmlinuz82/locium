from locium.models import Drawer
from locium.stitching import build_families


def _drawer(id, text, source="conv.jsonl", chunk=None):
    return Drawer(
        id=id, text=text, wing="w", hall="h", room="r",
        created_at="2026-07-01T00:00:00", source_file=source, chunk_index=chunk,
    )


def test_a_split_exchange_forms_one_family_in_chunk_order():
    drawers = [
        _drawer("b", "middle of the answer, ", chunk=1),
        _drawer("a", "> what happens next? The handler ", chunk=0),
        _drawer("c", "and the end of it.", chunk=2),
    ]

    stitches = build_families(drawers)

    assert list(stitches["families"].values()) == [["a", "b", "c"]]
    key = stitches["member"]["a"]
    assert stitches["member"]["b"] == key
    assert stitches["member"]["c"] == key


def test_a_new_exchange_head_closes_the_previous_family():
    drawers = [
        _drawer("a", "> first question. Answer part one ", chunk=0),
        _drawer("b", "answer part two.", chunk=1),
        _drawer("c", "> second question, fits in one drawer", chunk=2),
    ]

    stitches = build_families(drawers)

    assert list(stitches["families"].values()) == [["a", "b"]]
    assert "c" not in stitches["member"]


def test_an_unsplit_exchange_is_not_a_family():
    drawers = [
        _drawer("a", "> short question. Short answer.", chunk=0),
        _drawer("b", "> another one. Also short.", chunk=1),
    ]
    assert build_families(drawers) == {"families": {}, "member": {}, "docs": []}


def test_an_index_gap_breaks_the_family():
    """The miner drops sub-minimum slivers, leaving index gaps. Text across a
    gap is not contiguous, so stitching it would fabricate a sentence."""
    drawers = [
        _drawer("a", "> question. Start of the answer ", chunk=0),
        _drawer("b", "continues here ", chunk=1),
        _drawer("d", "but this is after a dropped sliver.", chunk=3),
    ]

    stitches = build_families(drawers)

    assert list(stitches["families"].values()) == [["a", "b"]]
    assert "d" not in stitches["member"]


def test_a_headless_file_forms_one_document_family():
    """Paragraph-mode files (tool-output dumps, pasted docs) have no '>'
    markers; all their slices group into one document family, flagged in
    ``docs`` because they cannot be reassembled by plain concatenation."""
    drawers = [
        _drawer("a", "one slice of a diff", chunk=0),
        _drawer("b", "another slice", chunk=2),  # gaps are fine for documents
        _drawer("c", "a third slice", chunk=5),
    ]

    stitches = build_families(drawers)

    assert list(stitches["families"].values()) == [["a", "b", "c"]]
    key = stitches["member"]["a"]
    assert stitches["docs"] == [key]


def test_a_headless_single_chunk_stays_single():
    drawers = [_drawer("a", "the whole small document", chunk=0)]
    assert build_families(drawers) == {"families": {}, "member": {}, "docs": []}


def test_families_never_cross_source_files():
    drawers = [
        _drawer("a", "> q one. Long answer part ", source="one.jsonl", chunk=0),
        _drawer("b", "one continues.", source="one.jsonl", chunk=1),
        _drawer("x", "> q two. Other answer ", source="two.jsonl", chunk=0),
        _drawer("y", "in the other file.", source="two.jsonl", chunk=1),
    ]

    stitches = build_families(drawers)

    assert sorted(stitches["families"].values()) == [["a", "b"], ["x", "y"]]
    assert stitches["member"]["b"] != stitches["member"]["y"]


def test_curated_drawers_without_chunk_metadata_are_ignored():
    drawers = [
        _drawer("a", "> hand-written note", source="", chunk=None),
        _drawer("b", "> another", source="notes.md", chunk=None),
    ]
    assert build_families(drawers) == {"families": {}, "member": {}, "docs": []}
