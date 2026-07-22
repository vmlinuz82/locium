import numpy as np
import pytest

from locium.build import build_index
from locium.index import read_meta, read_vectors
from locium.treemap import RefitRequired


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


def test_a_new_wing_lands_in_the_gutter(fake_palace, tmp_path):
    index_path = tmp_path / "idx"
    build_index(fake_palace, index_path)
    _add_drawers(fake_palace, ["g1", "g2"], "gamma")
    meta = build_index(fake_palace, index_path)

    gutter_top = 1000.0 * (1.0 - 0.15)
    gamma = next(w for w in meta["wings"] if w["name"] == "gamma")
    assert gamma["rect"][1] >= gutter_top - 1e-6


def test_meta_records_provenance(fake_palace, tmp_path):
    meta = build_index(fake_palace, tmp_path / "idx")
    assert meta["seed"] == 42
    assert meta["palace_mtime"] > 0
    assert meta["built_at"]
    assert meta["vector_dim"] == 8
