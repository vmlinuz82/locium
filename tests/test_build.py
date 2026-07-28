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


def _make_palace(path, entries):
    """Build a palace from (id, wing, hall, room) tuples, added in list order.

    Chroma preserves add-call order on plain get(), which is what lets these
    tests control the read order deterministically instead of relying on the
    incidental order a real palace happens to produce.
    """
    import chromadb

    path.mkdir(parents=True, exist_ok=True)
    collection = chromadb.PersistentClient(path=str(path)).get_or_create_collection(
        name="mempalace_drawers"
    )
    ids = [e[0] for e in entries]
    rng = np.random.default_rng(len(ids))
    vectors = rng.normal(size=(len(ids), 8)).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    collection.add(
        ids=ids,
        documents=[f"text for {i} about docker" for i in ids],
        embeddings=[v.tolist() for v in vectors],
        metadatas=[
            {"wing": w, "hall": h, "room": r, "source_file": "f.jsonl",
             "created_at": "2026-06-01T00:00:00"}
            for _, w, h, r in entries
        ],
    )


def _overlaps(rect_a, rect_b):
    x1, y1, w1, h1 = rect_a
    x2, y2, w2, h2 = rect_b
    return x1 < x2 + w2 and x2 < x1 + w1 and y1 < y2 + h2 and y2 < y1 + h1


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


def test_existing_coordinates_survive_a_rebuild(tmp_path):
    """Regression guard for #1: pairing must be by drawer id, not position.

    d1/d2 and the new drawers all share the exact same (wing, hall, room)
    chamber, and the new drawers are inserted so they read BEFORE d1/d2 in
    the second build. A positional zip(shown, points) would hand d1's old
    coordinate to a new drawer and reassign d1/d2 fresh coordinates instead
    of keeping their own.
    """
    palace = tmp_path / "palace"
    _make_palace(palace, [("d1", "w", "h", "r"), ("d2", "w", "h", "r")])
    index_path = tmp_path / "idx"
    before = {d["id"]: (d["x"], d["y"]) for d in build_index(palace, index_path)["drawers"]}

    import chromadb

    collection = chromadb.PersistentClient(path=str(palace)).get_or_create_collection(
        name="mempalace_drawers"
    )
    collection.delete(ids=["d1", "d2"])
    all_ids = ["new1", "new2", "d1", "d2"]
    rng = np.random.default_rng(7)
    vectors = rng.normal(size=(len(all_ids), 8)).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    collection.add(
        ids=all_ids,
        documents=[f"text for {i} about docker" for i in all_ids],
        embeddings=[v.tolist() for v in vectors],
        metadatas=[
            {"wing": "w", "hall": "h", "room": "r", "source_file": "f.jsonl",
             "created_at": "2026-06-01T00:00:00"}
            for _ in all_ids
        ],
    )

    after = {d["id"]: (d["x"], d["y"]) for d in build_index(palace, index_path)["drawers"]}

    assert after["d1"] == before["d1"]
    assert after["d2"] == before["d2"]
    assert after["new1"] not in (before["d1"], before["d2"])
    assert after["new2"] not in (before["d1"], before["d2"])
    assert after["new1"] != after["new2"]


def test_drawer_moved_to_a_new_chamber_is_repacked_inside_it(tmp_path):
    """Regression guard for #2: a stale coordinate must never survive a move.

    m1 starts in room r1 and is moved to r2 before the rebuild. Its old
    coordinate (valid for r1's old rect) must not be reused for r2.
    """
    palace = tmp_path / "palace"
    _make_palace(
        palace,
        [
            ("m1", "w", "h", "r1"), ("m2", "w", "h", "r1"), ("m3", "w", "h", "r1"),
            ("s1", "w", "h", "r2"), ("s2", "w", "h", "r2"), ("s3", "w", "h", "r2"),
        ],
    )
    index_path = tmp_path / "idx"
    before = build_index(palace, index_path)
    before_coord = {d["id"]: (d["x"], d["y"]) for d in before["drawers"]}

    import chromadb

    collection = chromadb.PersistentClient(path=str(palace)).get_or_create_collection(
        name="mempalace_drawers"
    )
    collection.update(ids=["m1"], metadatas=[{"room": "r2"}])

    after = build_index(palace, index_path)
    boxes = {(c["wing"], c["hall"], c["name"]): c["rect"] for c in after["chambers"]}
    for d in after["drawers"]:
        x, y, w, h = boxes[(d["wing"], d["hall"], d["room"])]
        assert x <= d["x"] <= x + w
        assert y <= d["y"] <= y + h

    moved = next(d for d in after["drawers"] if d["id"] == "m1")
    assert moved["room"] == "r2"
    assert (moved["x"], moved["y"]) != before_coord["m1"]


def test_a_new_wing_does_not_overlap_existing_wings(tmp_path):
    """Regression guard for #3: building_footprint must run on the FULL wing set.

    Running it on only the newly-discovered wing lands that wing on the core
    block, which is also whatever wing already occupies the core.
    """
    palace = tmp_path / "palace"
    _make_palace(
        palace,
        [
            ("a1", "alpha", "h", "r"), ("a2", "alpha", "h", "r"),
            ("b1", "beta", "h", "r"), ("b2", "beta", "h", "r"),
        ],
    )
    index_path = tmp_path / "idx"
    build_index(palace, index_path)

    import chromadb

    collection = chromadb.PersistentClient(path=str(palace)).get_or_create_collection(
        name="mempalace_drawers"
    )
    rng = np.random.default_rng(3)
    vectors = rng.normal(size=(2, 8)).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    collection.add(
        ids=["c1", "c2"],
        documents=["text for c1 about docker", "text for c2 about docker"],
        embeddings=[v.tolist() for v in vectors],
        metadatas=[
            {"wing": "gamma", "hall": "h", "room": "r", "source_file": "f.jsonl",
             "created_at": "2026-06-01T00:00:00"}
            for _ in range(2)
        ],
    )

    meta = build_index(palace, index_path)
    rects = [w["rect"] for w in meta["wings"]]
    assert len(rects) == 3
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            assert not _overlaps(rects[i], rects[j]), (
                f"{meta['wings'][i]['name']} overlaps {meta['wings'][j]['name']}"
            )


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


def test_a_new_wing_is_added_alongside_existing_wings(fake_palace, tmp_path):
    """Geometry is recomputed fresh every build (see #3), so an existing wing's
    rect is no longer guaranteed to stay byte-identical once a new wing is
    added -- only that every wing, including the new one, is present and
    that the layout stays overlap-free (covered separately)."""
    index_path = tmp_path / "idx"
    before = build_index(fake_palace, index_path)
    before_names = {w["name"] for w in before["wings"]}

    _add_drawers(fake_palace, ["g1", "g2"], "gamma")
    meta = build_index(fake_palace, index_path)

    assert any(w["name"] == "gamma" for w in meta["wings"])
    assert before_names <= {w["name"] for w in meta["wings"]}


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


def test_vectors_and_arcs_are_aligned_to_drawn_drawers(fake_palace, tmp_path):
    """meta.drawers, vectors.bin and arcs must share one index space."""
    index_path = tmp_path / "idx"
    meta = build_index(fake_palace, index_path)

    assert len(meta["drawers"]) == meta["drawer_count"]

    vectors = read_vectors(index_path, meta["drawer_count"], meta["vector_dim"])
    assert vectors.shape[0] == len(meta["drawers"])

    for src, dst, _distance in meta["arcs"]:
        assert 0 <= src < len(meta["drawers"])
        assert 0 <= dst < len(meta["drawers"])


def test_vectors_and_arcs_stay_aligned_when_a_chamber_is_capped(tmp_path):
    """Regression guard: dropping capped-out drawers from meta.drawers must
    also drop them from vectors.bin and arcs, or the three counts diverge.

    Before the fix, vectors.bin and arcs were built from ALL drawers read
    from the palace, while meta.drawers only kept the drawn (uncapped)
    subset -- so with a chamber over the cap, vectors.bin's row count would
    exceed len(meta.drawers) and this test would fail.
    """
    palace = tmp_path / "palace"
    _make_palace(
        palace,
        [(f"d{i}", "w", "h", "r") for i in range(5)],
    )
    index_path = tmp_path / "idx"
    meta = build_index(palace, index_path, tuning=Tuning(dot_cap=2))

    assert len(meta["drawers"]) == 2
    assert meta["drawer_count"] == 2

    vectors = read_vectors(index_path, meta["drawer_count"], meta["vector_dim"])
    assert vectors.shape[0] == len(meta["drawers"])

    for src, dst, _distance in meta["arcs"]:
        assert 0 <= src < len(meta["drawers"])
        assert 0 <= dst < len(meta["drawers"])


def test_dot_cap_zero_draws_every_drawer(tmp_path):
    """dot_cap <= 0 means no cap: every drawer gets a dot so every search hit
    is reachable, and no chamber is marked capped."""
    palace = tmp_path / "palace"
    _make_palace(palace, [(f"d{i}", "w", "h", "r") for i in range(50)])
    meta = build_index(palace, tmp_path / "idx", tuning=Tuning(dot_cap=0))

    assert len(meta["drawers"]) == 50
    assert meta["drawer_count"] == 50
    assert not any(c["capped"] for c in meta["chambers"])
    # the one chamber reports all 50 and draws all 50
    chamber = next(c for c in meta["chambers"] if c["name"] == "r")
    assert chamber["count"] == 50


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


def test_saturated_chamber_is_sub_clustered(tmp_path):
    """A chamber past cluster_min gets labelled zones and every drawer lands
    inside the chamber; smaller chambers stay untouched."""
    palace = tmp_path / "palace"
    entries = [(f"big{i}", "solo", "technical", "technical") for i in range(24)]
    entries += [(f"small{i}", "solo", "technical", "problems") for i in range(3)]
    _make_palace(palace, entries)

    meta = build_index(palace, tmp_path / "idx", tuning=Tuning(cluster_min=20))

    big = next(c for c in meta["chambers"] if c["name"] == "technical")
    small = next(c for c in meta["chambers"] if c["name"] == "problems")
    assert "clusters" not in small

    clusters = big["clusters"]
    assert len(clusters) >= 2
    assert sum(c["count"] for c in clusters) == 24
    # Zones subdivide the chamber, so they must all lie inside its rect.
    bx, by, bw, bh = big["rect"]
    for c in clusters:
        x, y, w, h = c["rect"]
        assert x >= bx - 1e-6 and y >= by - 1e-6
        assert x + w <= bx + bw + 1e-6 and y + h <= by + bh + 1e-6

    # Every drawer still lands inside the chamber that owns it.
    for d in meta["drawers"]:
        if d["room"] != "technical":
            continue
        assert bx <= d["x"] <= bx + bw
        assert by <= d["y"] <= by + bh


def test_sub_clustering_does_not_move_kept_drawers(tmp_path):
    """Coordinate stability outranks cluster purity: a rebuild that starts
    clustering a chamber must leave previously placed drawers where they were."""
    palace = tmp_path / "palace"
    entries = [(f"d{i}", "solo", "technical", "technical") for i in range(24)]
    _make_palace(palace, entries)

    index = tmp_path / "idx"
    before = build_index(palace, index, tuning=Tuning(cluster_min=1000))
    assert "clusters" not in before["chambers"][0]

    after = build_index(palace, index, tuning=Tuning(cluster_min=20))
    assert "clusters" in after["chambers"][0]

    old = {d["id"]: (d["x"], d["y"]) for d in before["drawers"]}
    for d in after["drawers"]:
        assert (d["x"], d["y"]) == old[d["id"]]


def test_build_snapshots_the_knowledge_graph(tmp_path):
    from tests.test_kg import _make_kg

    palace = tmp_path / "palace"
    _make_palace(palace, [("d0", "solo", "technical", "technical")])
    _make_kg(tmp_path, [("solo-project", "uses", "docker", "2026-05-01", None)])

    meta = build_index(palace, tmp_path / "idx")

    assert meta["kg"]["entity_count"] == 2
    assert meta["kg"]["triples"][0]["s"] == "solo-project"


def test_build_without_a_knowledge_graph_gets_an_empty_snapshot(tmp_path):
    palace = tmp_path / "palace"
    _make_palace(palace, [("d0", "solo", "technical", "technical")])

    meta = build_index(palace, tmp_path / "idx")

    assert meta["kg"] == {"entity_count": 0, "triples": []}


def test_build_writes_the_stitching_map(tmp_path):
    """Chunked siblings land in stitches.json; whole exchanges stay out."""
    import json

    import chromadb

    palace = tmp_path / "palace"
    palace.mkdir()
    collection = chromadb.PersistentClient(path=str(palace)).get_or_create_collection(
        name="mempalace_drawers"
    )
    rng = np.random.default_rng(5)
    vectors = rng.normal(size=(3, 8)).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    rows = [("h", "> q. Part one ", 0), ("t", "and part two.", 1), ("s", "> whole.", 2)]
    collection.add(
        ids=[r[0] for r in rows],
        documents=[r[1] for r in rows],
        embeddings=[v.tolist() for v in vectors],
        metadatas=[
            {"wing": "solo", "hall": "technical", "room": "technical",
             "source_file": "c.jsonl", "chunk_index": r[2],
             "created_at": "2026-06-01T00:00:00"}
            for r in rows
        ],
    )

    build_index(palace, tmp_path / "idx")
    stitches = json.loads((tmp_path / "idx" / "stitches.json").read_text())

    assert list(stitches["families"].values()) == [["h", "t"]]
    assert set(stitches["member"]) == {"h", "t"}


def test_build_ships_the_recall_gap_material(tmp_path):
    """Split exchanges get index-based families in meta plus one whole-message
    embedding each; a palace without splits ships neither."""
    import json

    import chromadb

    palace = tmp_path / "palace"
    palace.mkdir()
    collection = chromadb.PersistentClient(path=str(palace)).get_or_create_collection(
        name="mempalace_drawers"
    )
    rng = np.random.default_rng(9)
    vectors = rng.normal(size=(3, 8)).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    rows = [("h", "> q. Part one ", 0), ("t", "and part two.", 1), ("s", "> whole.", 2)]
    collection.add(
        ids=[r[0] for r in rows],
        documents=[r[1] for r in rows],
        embeddings=[v.tolist() for v in vectors],
        metadatas=[
            {"wing": "solo", "hall": "technical", "room": "technical",
             "source_file": "c.jsonl", "chunk_index": r[2],
             "created_at": "2026-06-01T00:00:00"}
            for r in rows
        ],
    )

    meta = build_index(palace, tmp_path / "idx")

    # Families hold positions into meta["drawers"], and they resolve back to
    # the drawers that were stitched.
    (family,) = meta["stitch_families"]
    ids = [meta["drawers"][i]["id"] for i in family]
    assert ids == ["h", "t"]

    dim = meta["family_vector_dim"]
    assert dim == 384
    blob = (tmp_path / "idx" / "family_vectors.bin").read_bytes()
    assert len(blob) == 1 * dim

    # And the no-splits case ships neither the file nor a dimension.
    no_split = tmp_path / "plain"
    _make_palace(no_split, [("a", "solo", "technical", "technical")])
    meta2 = build_index(no_split, tmp_path / "idx2")
    assert meta2["stitch_families"] == []
    assert meta2["family_vector_dim"] == 0
    assert not (tmp_path / "idx2" / "family_vectors.bin").exists()


def test_build_tags_content_kind_only_when_it_says_something(tmp_path):
    import chromadb

    palace = tmp_path / "palace"
    palace.mkdir()
    collection = chromadb.PersistentClient(path=str(palace)).get_or_create_collection(
        name="mempalace_drawers"
    )
    rng = np.random.default_rng(11)
    vectors = rng.normal(size=(2, 8)).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    csv_row = '"11/26/2025","hacked again","09/20/2025",,22,21,437 '
    collection.add(
        ids=["prose", "dump"],
        documents=["a decision about the messenger transport and why", csv_row * 12],
        embeddings=[v.tolist() for v in vectors],
        metadatas=[
            {"wing": "solo", "hall": "technical", "room": "technical",
             "created_at": "2026-06-01T00:00:00"}
            for _ in range(2)
        ],
    )

    meta = build_index(palace, tmp_path / "idx")

    by_id = {d["id"]: d for d in meta["drawers"]}
    assert "kind" not in by_id["prose"]
    assert by_id["dump"]["kind"] == "data"
