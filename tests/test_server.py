import os
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from locium.build import build_index


@pytest.fixture
def client(fake_palace, tmp_path):
    index_path = tmp_path / "idx"
    build_index(fake_palace, index_path)

    from locium.server import create_app

    return TestClient(create_app(index_path, fake_palace))


def test_search_text_endpoint_finds_every_literal_match(client):
    body = client.post("/api/search-text", json={"query": "docker"}).json()
    assert len(body["ids"]) == 6
    assert body["truncated"] is False


def test_search_text_endpoint_is_empty_when_the_string_is_absent(client):
    body = client.post("/api/search-text", json={"query": "EM-4103"}).json()
    assert body["ids"] == []


def test_search_text_endpoint_rejects_a_blank_query(client):
    assert client.post("/api/search-text", json={"query": "   "}).status_code == 422


def test_snippets_endpoint_returns_text_for_the_ids_given(client):
    body = client.post("/api/snippets", json={"ids": ["d0", "d1"], "query": "docker"}).json()
    assert set(body["snippets"]) == {"d0", "d1"}
    assert "docker" in body["snippets"]["d0"]


def test_snippets_endpoint_skips_unknown_ids(client):
    body = client.post("/api/snippets", json={"ids": ["d0", "nope"], "query": ""}).json()
    assert list(body["snippets"]) == ["d0"]


def test_index_endpoint_returns_meta(client):
    body = client.get("/api/index").json()
    assert body["drawer_count"] == 6
    assert len(body["drawers"]) == 6
    assert "stale" in body


def test_stale_flag_reflects_palace_modification(fake_palace, tmp_path):
    """Verify staleness detection in both directions: not stale immediately after build, stale after modification."""
    index_path = tmp_path / "idx"
    build_index(fake_palace, index_path)

    from locium.server import create_app

    app = create_app(index_path, fake_palace)
    client = TestClient(app)

    # Check that immediately after build, index is not stale
    body = client.get("/api/index").json()
    assert body["stale"] is False, "Index should not be stale immediately after build"

    # Modify the palace by touching a file with an explicitly future timestamp
    # to avoid filesystem timestamp granularity issues
    test_file = fake_palace / "test_marker.txt"
    test_file.write_text("marker")
    future_time = time.time() + 10.0  # 10 seconds in the future
    os.utime(test_file, (future_time, future_time))

    # Check that after modification, index is stale
    body = client.get("/api/index").json()
    assert body["stale"] is True, "Index should be stale after palace modification"


def test_vectors_endpoint_returns_int8_bytes(client):
    response = client.get("/api/vectors")
    assert response.headers["content-type"] == "application/octet-stream"
    assert np.frombuffer(response.content, dtype=np.int8).shape == (48,)  # 6 x 8


def test_search_returns_a_normalised_vector(client):
    body = client.post("/api/search", json={"query": "docker"}).json()
    vector = np.array(body["vector"], dtype=np.float32)
    assert vector.shape == (384,)
    assert abs(float(np.linalg.norm(vector)) - 1.0) < 1e-5


def test_search_rejects_an_empty_query(client):
    assert client.post("/api/search", json={"query": "   "}).status_code == 422


def test_rebuild_returns_the_new_count(client):
    assert client.post("/api/rebuild").json()["drawer_count"] == 6


def test_root_serves_the_viewer(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "<canvas" in response.text


def test_missing_index_refuses_to_start(tmp_path, fake_palace):
    from locium.server import create_app

    with pytest.raises(FileNotFoundError, match="locium build"):
        create_app(tmp_path / "absent", fake_palace)


def test_drawer_endpoint_returns_full_text(client):
    meta = client.get("/api/index").json()
    drawer_id = meta["drawers"][0]["id"]
    body = client.get(f"/api/drawer/{drawer_id}").json()
    assert body["id"] == drawer_id
    assert {"wing", "hall", "room", "date", "text"} <= set(body)
    assert len(body["text"]) >= len(meta["drawers"][0]["preview"])


def test_unknown_drawer_is_404(client):
    assert client.get("/api/drawer/does-not-exist").status_code == 404


def _split_exchange_palace(tmp_path):
    """A palace holding one exchange split across two chunks, plus a whole one."""
    import chromadb

    palace = tmp_path / "split-palace"
    palace.mkdir()
    collection = chromadb.PersistentClient(path=str(palace)).get_or_create_collection(
        name="mempalace_drawers"
    )
    rng = np.random.default_rng(3)
    vectors = rng.normal(size=(3, 8)).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    rows = [
        ("c0", "> how does dispatch work? The handler fans out into RecordSentEve", 0),
        ("c1", "nt.php messages, one per batch of entities.", 1),
        ("solo", "> unrelated question. Unrelated answer.", 2),
    ]
    collection.add(
        ids=[r[0] for r in rows],
        documents=[r[1] for r in rows],
        embeddings=[v.tolist() for v in vectors],
        metadatas=[
            {"wing": "alpha", "hall": "technical", "room": "technical",
             "source_file": "conv.jsonl", "chunk_index": r[2],
             "filed_at": "2026-06-01T10:00:00"}
            for r in rows
        ],
    )
    return palace


def test_drawer_endpoint_stitches_a_split_exchange(tmp_path):
    palace = _split_exchange_palace(tmp_path)
    index_path = tmp_path / "idx"
    build_index(palace, index_path)

    from locium.server import create_app

    client = TestClient(create_app(index_path, palace))

    first_len = len("> how does dispatch work? The handler fans out into RecordSentEve")
    for drawer_id, part, offset in (("c0", 1, 0), ("c1", 2, first_len)):
        body = client.get(f"/api/drawer/{drawer_id}").json()
        # Concatenation heals the word the miner split ("RecordSentEve|nt.php").
        assert "RecordSentEvent.php" in body["message"]
        assert body["message_part"] == part
        assert body["message_chunks"] == 2
        # The marker span is exactly this drawer's slice of the message.
        assert body["message_offset"] == offset
        span = body["message"][offset:offset + body["message_span_length"]]
        assert span == body["text"]

    solo = client.get("/api/drawer/solo").json()
    assert "message" not in solo


def test_family_vectors_endpoint_serves_the_gap_material(tmp_path):
    palace = _split_exchange_palace(tmp_path)
    index_path = tmp_path / "idx"
    meta = build_index(palace, index_path)

    from locium.server import create_app

    client = TestClient(create_app(index_path, palace))
    body = client.get("/api/family-vectors")
    assert body.headers["content-type"] == "application/octet-stream"
    assert len(body.content) == len(meta["stitch_families"]) * meta["family_vector_dim"]


def test_family_vectors_endpoint_is_empty_without_splits(client):
    assert client.get("/api/family-vectors").content == b""


def test_drawer_endpoint_rejoins_a_document_family_on_newlines(tmp_path):
    """Headless slices are not exact partitions, so they rejoin with a
    separator instead of pretending to be one string."""
    import chromadb

    palace = tmp_path / "doc-palace"
    palace.mkdir()
    collection = chromadb.PersistentClient(path=str(palace)).get_or_create_collection(
        name="mempalace_drawers"
    )
    rng = np.random.default_rng(7)
    vectors = rng.normal(size=(2, 8)).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    collection.add(
        ids=["s0", "s1"],
        documents=["first slice of the dump", "second slice of the dump"],
        embeddings=[v.tolist() for v in vectors],
        metadatas=[
            {"wing": "alpha", "hall": "technical", "room": "technical",
             "source_file": "tool-dump.txt", "chunk_index": k,
             "filed_at": "2026-06-01T10:00:00"}
            for k in range(2)
        ],
    )

    index_path = tmp_path / "idx"
    build_index(palace, index_path)

    from locium.server import create_app

    client = TestClient(create_app(index_path, palace))
    body = client.get("/api/drawer/s0").json()
    assert body["message"] == "first slice of the dump\nsecond slice of the dump"
    assert body["message_chunks"] == 2
    assert body["message_offset"] == 0

    # The second slice's offset accounts for the newline joiner.
    second = client.get("/api/drawer/s1").json()
    assert second["message_offset"] == len("first slice of the dump") + 1
    span = second["message"][
        second["message_offset"]:second["message_offset"] + second["message_span_length"]
    ]
    assert span == "second slice of the dump"


def test_static_files_demand_revalidation(client):
    """A stale cached render.js next to a fresh app.js runs a mixed-version
    app; every static must carry no-cache so browsers revalidate."""
    for path in ("/", "/app.js", "/render.js", "/knn.js", "/style.css"):
        response = client.get(path)
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-cache", path


def test_root_stamps_asset_urls_with_their_mtime(client):
    """A changed file must be a changed URL, so a cache entry poisoned before
    no-cache headers existed can never be served again."""
    import re

    html = client.get("/").text
    stamps = dict(re.findall(r'/(\w+\.(?:js|css))\?v=(\d+)', html))
    assert set(stamps) == {"app.js", "render.js", "knn.js", "style.css"}
    # And the stamped URLs actually resolve.
    response = client.get(f"/app.js?v={stamps['app.js']}")
    assert response.status_code == 200
