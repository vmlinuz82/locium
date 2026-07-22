import numpy as np
import pytest

from locium.extract import PalaceNotFound, palace_mtime, read_drawers, snapshot_palace


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


def test_palace_mtime_is_a_timestamp(fake_palace):
    assert palace_mtime(fake_palace) > 0
