import importlib

import numpy as np
import pytest
from fastapi.testclient import TestClient

from locium.build import build_index


@pytest.fixture
def client(fake_palace, tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    import mempalace.palace_graph as palace_graph

    importlib.reload(palace_graph)

    index_path = tmp_path / "idx"
    build_index(fake_palace, index_path)

    from locium.server import create_app

    return TestClient(create_app(index_path, fake_palace))


def test_index_endpoint_returns_meta(client):
    body = client.get("/api/index").json()
    assert body["drawer_count"] == 6
    assert len(body["drawers"]) == 6
    assert "stale" in body


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


def test_tunnel_create_list_delete_round_trip(client):
    created = client.post(
        "/api/tunnel",
        json={
            "source_wing": "alpha", "source_room": "technical",
            "target_wing": "beta", "target_room": "technical",
            "label": "shared", "source_drawer_id": "d0", "target_drawer_id": "d5",
        },
    ).json()
    assert client.get("/api/tunnels").json()["tunnels"][0]["label"] == "shared"

    client.delete(f"/api/tunnel/{created['id']}")
    assert client.get("/api/tunnels").json()["tunnels"] == []


def test_tunnel_creation_requires_wings_and_rooms(client):
    assert client.post("/api/tunnel", json={"source_wing": "alpha"}).status_code == 422


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
