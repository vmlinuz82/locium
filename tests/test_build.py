import numpy as np
import pytest

from locium.build import build_index
from locium.config import Tuning
from locium.extract import PalaceNotFound
from locium.index import read_meta, read_vectors


def _add_drawers(palace, ids, wing):
    import chromadb

    collection = chromadb.PersistentClient(path=str(palace)).get_or_create_collection(
        name="mempalace_drawers"
    )
    rng = np.random.default_rng(len(ids))
    vectors = rng.normal(size=(len(ids), 8)).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    collection.add(
        ids=ids,
        documents=[f"text for {i} about docker" for i in ids],
        embeddings=[v.tolist() for v in vectors],
        metadatas=[
            {"wing": wing, "room": "technical", "source_file": "f.jsonl",
             "created_at": "2026-06-01T00:00:00"}
            for _ in ids
        ],
    )


def test_build_writes_a_readable_index(fake_palace, tmp_path):
    meta = build_index(fake_palace, tmp_path / "idx")
    assert meta["drawer_count"] == 6
    assert len(read_meta(tmp_path / "idx")["drawers"]) == 6


def test_every_drawer_gets_coordinates(fake_palace, tmp_path):
    meta = build_index(fake_palace, tmp_path / "idx")
    for drawer in meta["drawers"]:
        assert isinstance(drawer["x"], float)
        assert isinstance(drawer["y"], float)


def test_vectors_are_written_as_int8(fake_palace, tmp_path):
    meta = build_index(fake_palace, tmp_path / "idx")
    vectors = read_vectors(tmp_path / "idx", meta["drawer_count"], meta["vector_dim"])
    assert vectors.dtype == np.int8
    assert vectors.shape == (6, 8)


def test_build_is_deterministic(fake_palace, tmp_path):
    first = build_index(fake_palace, tmp_path / "a")
    second = build_index(fake_palace, tmp_path / "b")
    assert [(d["x"], d["y"]) for d in first["drawers"]] == [
        (d["x"], d["y"]) for d in second["drawers"]
    ]


def test_existing_coordinates_survive_a_rebuild(fake_palace, tmp_path):
    index_path = tmp_path / "idx"
    before = {d["id"]: (d["x"], d["y"]) for d in build_index(fake_palace, index_path)["drawers"]}

    _add_drawers(fake_palace, ["new1", "new2", "new3"], "alpha")
    after = {d["id"]: (d["x"], d["y"]) for d in build_index(fake_palace, index_path)["drawers"]}

    for drawer_id, coords in before.items():
        assert after[drawer_id] == coords, f"{drawer_id} moved"


def test_rebuild_adds_the_new_drawers(fake_palace, tmp_path):
    index_path = tmp_path / "idx"
    build_index(fake_palace, index_path)
    _add_drawers(fake_palace, ["new1"], "alpha")
    ids = {d["id"] for d in build_index(fake_palace, index_path)["drawers"]}
    assert "new1" in ids


def test_refit_is_allowed_to_move_drawers(fake_palace, tmp_path):
    index_path = tmp_path / "idx"
    build_index(fake_palace, index_path)
    _add_drawers(fake_palace, [f"extra{i}" for i in range(15)], "alpha")
    meta = build_index(fake_palace, index_path, refit=True)
    assert meta["drawer_count"] == 21


def test_a_new_wing_is_placed_without_disturbing_existing_wings(fake_palace, tmp_path):
    index_path = tmp_path / "idx"
    before = build_index(fake_palace, index_path)
    before_rects = {w["name"]: w["rect"] for w in before["wings"]}

    _add_drawers(fake_palace, ["g1", "g2"], "gamma")
    meta = build_index(fake_palace, index_path)

    assert any(w["name"] == "gamma" for w in meta["wings"])
    for w in meta["wings"]:
        if w["name"] in before_rects:
            assert w["rect"] == before_rects[w["name"]]


def test_meta_records_provenance(fake_palace, tmp_path):
    meta = build_index(fake_palace, tmp_path / "idx")
    assert meta["seed"] == 42
    assert meta["palace_mtime"] > 0
    assert meta["built_at"]
    assert meta["vector_dim"] == 8


def test_build_index_raises_palace_not_found_when_palace_missing(tmp_path):
    """Verify build_index raises PalaceNotFound (not bare FileNotFoundError) for missing palace."""
    missing_palace = tmp_path / "nonexistent"
    index_path = tmp_path / "idx"
    with pytest.raises(PalaceNotFound) as exc_info:
        build_index(missing_palace, index_path)
    assert "not found" in str(exc_info.value)
    assert str(missing_palace) in str(exc_info.value)


def test_meta_carries_the_building(fake_palace, tmp_path):
    meta = build_index(fake_palace, tmp_path / "idx")
    assert meta["wings"] and meta["halls"] and meta["chambers"]
    for chamber in meta["chambers"]:
        assert {"name", "wing", "hall", "rect", "count", "capped"} <= set(chamber)


def test_every_drawer_sits_inside_its_chamber(fake_palace, tmp_path):
    meta = build_index(fake_palace, tmp_path / "idx")
    boxes = {
        (c["wing"], c["hall"], c["name"]): c["rect"] for c in meta["chambers"]
    }
    for d in meta["drawers"]:
        x, y, w, h = boxes[(d["wing"], d["hall"], d["room"])]
        assert x <= d["x"] <= x + w
        assert y <= d["y"] <= y + h


def test_chamber_over_the_cap_is_marked(fake_palace, tmp_path):
    meta = build_index(fake_palace, tmp_path / "idx", tuning=Tuning(dot_cap=1))
    assert any(c["capped"] for c in meta["chambers"])


def test_capped_chamber_still_reports_its_true_count(fake_palace, tmp_path):
    # Keyed by the full (wing, hall, name) triple, not name alone: room names
    # are shared across wings/halls (e.g. two unrelated "technical" chambers),
    # so keying by name alone would sum drawers from different chambers.
    meta = build_index(fake_palace, tmp_path / "idx", tuning=Tuning(dot_cap=1))
    capped = [c for c in meta["chambers"] if c["capped"]]
    drawn = {(c["wing"], c["hall"], c["name"]): 0 for c in capped}
    for d in meta["drawers"]:
        key = (d["wing"], d["hall"], d["room"])
        if key in drawn:
            drawn[key] += 1
    for c in capped:
        assert c["count"] > drawn[(c["wing"], c["hall"], c["name"])]
