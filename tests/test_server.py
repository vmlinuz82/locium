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
