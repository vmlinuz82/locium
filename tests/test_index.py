import json
from pathlib import Path

import numpy as np
import pytest

from locium.index import (
    META_NAME,
    VECTORS_NAME,
    index_exists,
    read_meta,
    read_vectors,
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
    # is never touched until the final os.replace, so this exercises exactly
    # the window the atomic-staging fix closes.
    monkeypatch.setattr(Path, "write_bytes", boom)

    new_meta = _meta() | {"drawer_count": 99}
    with pytest.raises(OSError, match="disk full"):
        write_index(tmp_path / "idx", new_meta, np.zeros((2, 4), dtype=np.int8))

    assert index_exists(tmp_path / "idx")
    assert read_meta(tmp_path / "idx") == _meta()
    assert np.array_equal(read_vectors(tmp_path / "idx", 2, 4), original_vectors)
    assert not (tmp_path / "idx.old").exists()


def test_read_vectors_rejects_byte_length_mismatch(tmp_path):
    write_index(tmp_path / "idx", _meta(), np.ones((2, 4), dtype=np.int8))
    (tmp_path / "idx" / VECTORS_NAME).write_bytes(b"\x00" * 5)

    with pytest.raises(ValueError, match="8.*5|5.*8"):
        read_vectors(tmp_path / "idx", 2, 4)
