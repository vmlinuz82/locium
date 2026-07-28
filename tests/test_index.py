import json
import os
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from locium.index import (
    META_NAME,
    TEXTS_NAME,
    VECTORS_NAME,
    index_exists,
    read_meta,
    read_text,
    read_vectors,
    search_texts,
    snippet,
    write_index,
)


def _meta() -> dict:
    return {
        "built_at": "2026-07-22T10:00:00+00:00",
        "palace_mtime": 1753180800.0,
        "drawer_count": 2,
        "vector_dim": 4,
        "seed": 42,
        "mempalace_version": "3.3.3",
        "drawers": [
            {"id": "d1", "wing": "a", "room": "diary", "date": "2026-05-02",
             "x": 1.0, "y": 2.0, "preview": "hello", "cluster": 0},
            {"id": "d2", "wing": "b", "room": "technical", "date": "2026-05-03",
             "x": 3.0, "y": 4.0, "preview": "world", "cluster": -1},
        ],
        "wings": [{"name": "a", "rect": [0, 0, 10, 10], "count": 1}],
        "clusters": [{"wing": "a", "cluster": 0, "label": "docker compose",
                      "centroid": [1.0, 2.0]}],
        "arcs": [[0, 1, 0.31]],
    }


def test_write_creates_both_files(tmp_path):
    write_index(tmp_path / "idx", _meta(), np.ones((2, 4), dtype=np.int8))
    assert (tmp_path / "idx" / META_NAME).exists()
    assert (tmp_path / "idx" / VECTORS_NAME).exists()


def test_meta_round_trips(tmp_path):
    write_index(tmp_path / "idx", _meta(), np.ones((2, 4), dtype=np.int8))
    assert read_meta(tmp_path / "idx") == _meta()


def test_vectors_round_trip(tmp_path):
    vectors = np.array([[1, -2, 3, -4], [5, 6, 7, 8]], dtype=np.int8)
    write_index(tmp_path / "idx", _meta(), vectors)
    assert np.array_equal(read_vectors(tmp_path / "idx", 2, 4), vectors)


def test_write_rejects_non_int8(tmp_path):
    with pytest.raises(ValueError, match="int8"):
        write_index(tmp_path / "idx", _meta(), np.ones((2, 4), dtype=np.float32))


def test_index_exists_reports_missing(tmp_path):
    assert not index_exists(tmp_path / "nope")
    write_index(tmp_path / "idx", _meta(), np.ones((2, 4), dtype=np.int8))
    assert index_exists(tmp_path / "idx")


def test_write_is_atomic_leaving_no_temp_files(tmp_path):
    write_index(tmp_path / "idx", _meta(), np.ones((2, 4), dtype=np.int8))
    assert [p.name for p in sorted((tmp_path / "idx").iterdir())] == [
        META_NAME,
        VECTORS_NAME,
    ]


def test_empty_index_is_valid(tmp_path):
    meta = _meta() | {"drawers": [], "wings": [], "clusters": [], "arcs": [],
                      "drawer_count": 0}
    write_index(tmp_path / "idx", meta, np.zeros((0, 4), dtype=np.int8))
    assert read_meta(tmp_path / "idx")["drawers"] == []
    assert read_vectors(tmp_path / "idx", 0, 4).shape == (0, 4)


def test_meta_is_human_readable(tmp_path):
    write_index(tmp_path / "idx", _meta(), np.ones((2, 4), dtype=np.int8))
    raw = (tmp_path / "idx" / META_NAME).read_text(encoding="utf-8")
    assert "\n" in raw
    json.loads(raw)


def test_crash_mid_write_leaves_previous_artifact_intact(tmp_path, monkeypatch):
    original_vectors = np.array([[1, -2, 3, -4], [5, 6, 7, 8]], dtype=np.int8)
    write_index(tmp_path / "idx", _meta(), original_vectors)

    def boom(self, data):
        raise OSError("disk full")

    # Patch the staging write of vectors.bin: meta.json for the new index
    # will already be staged when this fires, but the real target directory
    # is never touched until the final os.replace. This crash point was
    # already safe by construction (staging is a separate directory the
    # target never depends on) -- it doesn't exercise the inter-rename
    # window the atomic-staging fix introduced. See
    # test_crash_between_renames_recovers_and_retry_preserves_backup for
    # a test of that window.
    monkeypatch.setattr(Path, "write_bytes", boom)

    new_meta = _meta() | {"drawer_count": 99}
    with pytest.raises(OSError, match="disk full"):
        write_index(tmp_path / "idx", new_meta, np.zeros((2, 4), dtype=np.int8))

    assert index_exists(tmp_path / "idx")
    assert read_meta(tmp_path / "idx") == _meta()
    assert np.array_equal(read_vectors(tmp_path / "idx", 2, 4), original_vectors)
    assert not (tmp_path / "idx.old").exists()


def test_crash_between_renames_recovers_and_retry_preserves_backup(
    tmp_path, monkeypatch
):
    idx_path = tmp_path / "idx"
    old_path = tmp_path / "idx.old"
    original_vectors = np.array([[1, -2, 3, -4], [5, 6, 7, 8]], dtype=np.int8)
    write_index(idx_path, _meta(), original_vectors)

    new_meta = _meta() | {"drawer_count": 99}
    new_vectors = np.zeros((2, 4), dtype=np.int8)

    real_replace = os.replace
    call_count = {"n": 0}

    def flaky_replace(src, dst):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise OSError("power cut")
        return real_replace(src, dst)

    # Fault the swap between the two renames: the first os.replace (moving
    # `idx` aside to `idx.old`) succeeds, the second (committing staging
    # into `idx`) fails. This leaves `idx` missing, `idx.old` holding the
    # last-good artifact, and the fully-written staging dir still on disk.
    with patch("os.replace", side_effect=flaky_replace):
        with pytest.raises(OSError, match="power cut"):
            write_index(idx_path, new_meta, new_vectors)

    # The last-good artifact must still be recoverable, whether directly
    # at idx_path or in the sibling backup.
    if index_exists(idx_path):
        assert read_meta(idx_path) == _meta()
        assert np.array_equal(read_vectors(idx_path, 2, 4), original_vectors)
    else:
        assert index_exists(old_path)
        assert read_meta(old_path) == _meta()
        assert np.array_equal(read_vectors(old_path, 2, 4), original_vectors)

    # Guard shutil.rmtree so it's proven that `idx.old` is only ever
    # removed once `idx_path` exists again -- i.e. recovery ran before any
    # destructive cleanup, so this retry cannot destroy the sole surviving
    # copy while it is the only one left.
    real_rmtree = shutil.rmtree

    def guarded_rmtree(target, *args, **kwargs):
        if Path(target) == old_path:
            assert idx_path.exists(), "backup removed while idx_path was missing"
        return real_rmtree(target, *args, **kwargs)

    monkeypatch.setattr(shutil, "rmtree", guarded_rmtree)

    write_index(idx_path, new_meta, new_vectors)

    assert index_exists(idx_path)
    assert read_meta(idx_path) == new_meta
    assert np.array_equal(read_vectors(idx_path, 2, 4), new_vectors)
    assert not old_path.exists()


def test_read_vectors_rejects_byte_length_mismatch(tmp_path):
    write_index(tmp_path / "idx", _meta(), np.ones((2, 4), dtype=np.int8))
    (tmp_path / "idx" / VECTORS_NAME).write_bytes(b"\x00" * 5)

    with pytest.raises(ValueError, match="8.*5|5.*8"):
        read_vectors(tmp_path / "idx", 2, 4)


def test_texts_are_written_inside_the_staged_directory(tmp_path):
    texts = {"d1": "full text for d1", "d2": "full text for d2"}
    write_index(tmp_path / "idx", _meta(), np.ones((2, 4), dtype=np.int8), texts)
    assert [p.name for p in sorted((tmp_path / "idx").iterdir())] == [
        META_NAME,
        TEXTS_NAME,
        VECTORS_NAME,
    ]


def test_read_text_round_trips(tmp_path):
    texts = {"d1": "full text for d1", "d2": "full text for d2"}
    write_index(tmp_path / "idx", _meta(), np.ones((2, 4), dtype=np.int8), texts)
    assert read_text(tmp_path / "idx", "d1") == "full text for d1"


def test_read_text_returns_none_for_unknown_id(tmp_path):
    texts = {"d1": "full text for d1"}
    write_index(tmp_path / "idx", _meta(), np.ones((2, 4), dtype=np.int8), texts)
    assert read_text(tmp_path / "idx", "does-not-exist") is None


def test_search_texts_finds_an_identifier_the_preview_would_miss(tmp_path):
    texts = {"d1": "a" * 500 + " ticket EM-4103 raised", "d2": "unrelated"}
    write_index(tmp_path / "idx", _meta(), np.ones((2, 4), dtype=np.int8), texts)
    assert search_texts(tmp_path / "idx", "EM-4103", 10) == ["d1"]


def test_search_texts_is_case_insensitive(tmp_path):
    texts = {"d1": "ticket em-4103 in lower case", "d2": "unrelated"}
    write_index(tmp_path / "idx", _meta(), np.ones((2, 4), dtype=np.int8), texts)
    assert search_texts(tmp_path / "idx", "EM-4103", 10) == ["d1"]


def test_search_texts_stops_at_the_limit(tmp_path):
    texts = {f"d{i}": "a shared token" for i in range(5)}
    write_index(tmp_path / "idx", _meta(), np.ones((2, 4), dtype=np.int8), texts)
    assert len(search_texts(tmp_path / "idx", "shared", 3)) == 3


def test_search_texts_survives_a_missing_texts_file(tmp_path):
    write_index(tmp_path / "idx", _meta(), np.ones((2, 4), dtype=np.int8))
    assert search_texts(tmp_path / "idx", "anything", 10) == []


def test_snippet_returns_short_text_untouched():
    assert snippet("a short drawer", "drawer", width=100) == "a short drawer"


def test_snippet_centres_on_the_match_rather_than_the_start():
    text = "x" * 2000 + " NEEDLE-42 " + "y" * 2000
    out = snippet(text, "NEEDLE-42", width=300)

    assert "NEEDLE-42" in out
    assert len(out) <= 302  # the window plus a leading and trailing ellipsis
    assert out.startswith("…") and out.endswith("…")


def test_snippet_falls_back_to_the_head_when_the_needle_is_absent():
    text = "z" * 1000
    out = snippet(text, "not-here", width=200)

    assert out.startswith("z")
    assert out.endswith("…")


def test_snippet_at_the_very_end_still_fills_the_window():
    """A match in the last characters must not yield a stub of a snippet."""
    text = "q" * 1000 + " TAIL-TOKEN"
    out = snippet(text, "TAIL-TOKEN", width=300)

    assert "TAIL-TOKEN" in out
    assert len(out) >= 290


def test_search_texts_sees_a_rebuild(tmp_path):
    """The parsed texts are cached, so a rebuild must invalidate the entry."""
    index = tmp_path / "idx"
    vectors = np.ones((2, 4), dtype=np.int8)
    write_index(index, _meta(), vectors, {"d1": "before"})
    assert search_texts(index, "before", 10) == ["d1"]

    write_index(index, _meta(), vectors, {"d1": "after"})
    assert search_texts(index, "before", 10) == []
    assert search_texts(index, "after", 10) == ["d1"]


def test_read_text_returns_none_when_texts_file_missing(tmp_path):
    write_index(tmp_path / "idx", _meta(), np.ones((2, 4), dtype=np.int8))
    assert read_text(tmp_path / "idx", "d1") is None


def test_read_text_returns_none_on_malformed_json(tmp_path):
    write_index(tmp_path / "idx", _meta(), np.ones((2, 4), dtype=np.int8), {"d1": "hi"})
    (tmp_path / "idx" / TEXTS_NAME).write_text("not json", encoding="utf-8")
    assert read_text(tmp_path / "idx", "d1") is None


def test_load_stitches_is_empty_when_the_file_is_absent(tmp_path):
    write_index(tmp_path / "idx", _meta(), np.ones((2, 4), dtype=np.int8))
    from locium.index import load_stitches

    assert load_stitches(tmp_path / "idx") == {"families": {}, "member": {}}


def test_stitches_round_trip_and_survive_a_rebuild(tmp_path):
    from locium.index import load_stitches

    index = tmp_path / "idx"
    vectors = np.ones((2, 4), dtype=np.int8)
    first = {"families": {"f0": ["d1", "d2"]}, "member": {"d1": "f0", "d2": "f0"}}
    write_index(index, _meta(), vectors, stitches=first)
    assert load_stitches(index) == first

    second = {"families": {}, "member": {}}
    write_index(index, _meta(), vectors, stitches=second)
    assert load_stitches(index) == second
