import numpy as np
import pytest

from locium.extract import PAGE_SIZE, PalaceNotFound, palace_mtime, read_drawers, snapshot_palace


def _build_palace(tmp_path, count, dim=8, name="palace"):
    """Build a real Chroma collection with ``count`` drawers, same schema as fake_palace."""
    import chromadb

    palace = tmp_path / name
    palace.mkdir()
    client = chromadb.PersistentClient(path=str(palace))
    collection = client.get_or_create_collection(name="mempalace_drawers")

    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(count, dim)).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    collection.add(
        ids=[f"d{i}" for i in range(count)],
        documents=[f"document number {i}" for i in range(count)],
        embeddings=[v.tolist() for v in vectors],
        metadatas=[
            {
                "wing": "alpha",
                "room": "technical",
                "source_file": f"f{i}.jsonl",
                "created_at": "2026-05-01T10:00:00",
            }
            for i in range(count)
        ],
    )
    return palace


def _install_corrupt_get(monkeypatch, bad_start, bad_end):
    """Make Collection.get raise InternalError for any range touching [bad_start, bad_end).

    Returns the list of (offset, limit) pairs every call was made with, so tests
    can confirm the salvage recursion actually visited sub-page-sized slices.
    """
    import chromadb
    from chromadb.api.models.Collection import Collection

    original_get = Collection.get
    calls = []

    def fake_get(self, *, limit=None, offset=None, include=None, **kwargs):
        offset = offset or 0
        calls.append((offset, limit))
        if offset < bad_end and offset + limit > bad_start:
            raise chromadb.errors.InternalError("simulated corruption")
        return original_get(self, limit=limit, offset=offset, include=include, **kwargs)

    monkeypatch.setattr(Collection, "get", fake_get)
    return calls


def test_snapshot_creates_a_separate_copy(fake_palace):
    copy = snapshot_palace(fake_palace)
    assert copy.exists()
    assert copy != fake_palace
    assert not str(copy).startswith(str(fake_palace))


def test_snapshot_of_a_missing_palace_raises(tmp_path):
    with pytest.raises(PalaceNotFound, match="not found"):
        snapshot_palace(tmp_path / "nowhere")


def test_read_drawers_returns_every_drawer(fake_palace):
    drawers, vectors = read_drawers(snapshot_palace(fake_palace))
    assert len(drawers) == 6
    assert vectors.shape == (6, 8)


def test_read_drawers_preserves_metadata(fake_palace):
    drawers, _ = read_drawers(snapshot_palace(fake_palace))
    by_id = {d.id: d for d in drawers}
    assert by_id["d0"].wing == "alpha"
    assert by_id["d5"].wing == "beta"
    assert by_id["d0"].room == "technical"
    assert by_id["d0"].source_file == "f0.jsonl"


def test_vectors_are_normalised(fake_palace):
    _, vectors = read_drawers(snapshot_palace(fake_palace))
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_drawers_and_vectors_stay_aligned(fake_palace):
    drawers, vectors = read_drawers(snapshot_palace(fake_palace))
    assert len(drawers) == len(vectors)


def test_missing_metadata_falls_back_to_defaults(tmp_path):
    import chromadb

    palace = tmp_path / "sparse"
    palace.mkdir()
    collection = chromadb.PersistentClient(path=str(palace)).get_or_create_collection(
        name="mempalace_drawers"
    )
    collection.add(
        ids=["only"],
        documents=["text"],
        embeddings=[[1.0, 0.0]],
        metadatas=[{"wing": "solo"}],
    )
    drawers, _ = read_drawers(snapshot_palace(palace))
    assert drawers[0].room == "general"
    assert drawers[0].created_at == ""


def test_filed_at_becomes_created_at(tmp_path):
    """MemPalace's real key: 'filed_at' populates Drawer.created_at."""
    import chromadb

    palace = tmp_path / "sparse"
    palace.mkdir()
    collection = chromadb.PersistentClient(path=str(palace)).get_or_create_collection(
        name="mempalace_drawers"
    )
    collection.add(
        ids=["only"],
        documents=["text"],
        embeddings=[[1.0, 0.0]],
        metadatas=[{"wing": "solo", "filed_at": "2026-05-08T07:52:22.226459"}],
    )
    drawers, _ = read_drawers(snapshot_palace(palace))
    assert drawers[0].created_at == "2026-05-08T07:52:22.226459"


def test_legacy_created_at_is_used_when_filed_at_is_absent(tmp_path):
    """Fallback path: stores without 'filed_at' still populate Drawer.created_at."""
    import chromadb

    palace = tmp_path / "sparse"
    palace.mkdir()
    collection = chromadb.PersistentClient(path=str(palace)).get_or_create_collection(
        name="mempalace_drawers"
    )
    collection.add(
        ids=["only"],
        documents=["text"],
        embeddings=[[1.0, 0.0]],
        metadatas=[{"wing": "solo", "created_at": "2026-05-01T10:00:00"}],
    )
    drawers, _ = read_drawers(snapshot_palace(palace))
    assert drawers[0].created_at == "2026-05-01T10:00:00"


def test_palace_mtime_is_a_timestamp(fake_palace):
    assert palace_mtime(fake_palace) > 0


def test_paging_returns_everything_when_nothing_is_broken(tmp_path):
    count = int(PAGE_SIZE * 2.4)  # spans three pages
    palace = _build_palace(tmp_path, count=count)
    drawers, vectors = read_drawers(snapshot_palace(palace))
    assert len(drawers) == count
    assert vectors.shape == (count, 8)
    assert {d.id for d in drawers} == {f"d{i}" for i in range(count)}


def test_failing_region_is_skipped_and_rest_survives(tmp_path, monkeypatch):
    palace = _build_palace(tmp_path, count=30)
    copy = snapshot_palace(palace)
    all_ids = {f"d{i}" for i in range(30)}
    bad_ids = {f"d{i}" for i in range(10, 15)}

    calls = _install_corrupt_get(monkeypatch, bad_start=10, bad_end=15)
    with pytest.warns(UserWarning, match="Skipped 5"):
        drawers, vectors = read_drawers(copy)

    assert {d.id for d in drawers} == all_ids - bad_ids
    assert len(drawers) == len(vectors) == 25

    # Confirm the recursion genuinely narrowed into sub-page slices rather than
    # jumping straight from the whole page (30) to single records: at least one
    # *successful* fetch covered more than one record but fewer than all of them.
    successful_partial = [
        (offset, limit)
        for offset, limit in calls
        if 1 < limit < 30 and not (offset < 15 and offset + limit > 10)
    ]
    assert successful_partial, "expected sub-slice salvage, not just page/singleton reads"


def test_read_drawers_warns_when_records_are_skipped(tmp_path, monkeypatch):
    palace = _build_palace(tmp_path, count=30)
    copy = snapshot_palace(palace)
    _install_corrupt_get(monkeypatch, bad_start=10, bad_end=15)

    with pytest.warns(UserWarning, match="Skipped 5"):
        read_drawers(copy)


def test_read_drawers_does_not_warn_when_nothing_is_skipped(fake_palace, recwarn):
    read_drawers(snapshot_palace(fake_palace))
    assert not any("Skipped" in str(w.message) for w in recwarn.list)
