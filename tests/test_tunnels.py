import importlib

import pytest


@pytest.fixture
def tunnels(tmp_path, monkeypatch):
    """Point mempalace's tunnel file at a temp HOME, then reload both modules."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    import mempalace.palace_graph as palace_graph

    importlib.reload(palace_graph)

    import locium.tunnels as module

    importlib.reload(module)
    return module


def test_api_assertion_passes_against_the_installed_mempalace(tunnels):
    tunnels.assert_api()


def test_create_then_list(tunnels):
    tunnels.create("alpha", "diary", "beta", "technical", "shared topic", "d1", "d2")
    stored = tunnels.listing()
    assert len(stored) == 1
    assert stored[0]["label"] == "shared topic"
    assert stored[0]["source"]["drawer_id"] == "d1"
    assert stored[0]["target"]["drawer_id"] == "d2"


def test_same_room_pair_collapses_to_one_record(tunnels):
    tunnels.create("alpha", "diary", "beta", "technical", "first", "d1", "d2")
    tunnels.create("alpha", "diary", "beta", "technical", "second", "d3", "d4")
    stored = tunnels.listing()
    assert len(stored) == 1
    assert stored[0]["label"] == "second"


def test_tunnels_are_symmetric(tunnels):
    tunnels.create("alpha", "diary", "beta", "technical", "forward", "d1", "d2")
    tunnels.create("beta", "technical", "alpha", "diary", "reverse", "d2", "d1")
    assert len(tunnels.listing()) == 1


def test_delete_removes_a_tunnel(tunnels):
    created = tunnels.create("alpha", "diary", "beta", "technical", "x", "d1", "d2")
    tunnels.delete(created["id"])
    assert tunnels.listing() == []


def test_deleting_an_unknown_id_is_a_no_op(tunnels):
    tunnels.create("alpha", "diary", "beta", "technical", "x", "d1", "d2")
    tunnels.delete("does-not-exist")
    assert len(tunnels.listing()) == 1


def test_module_never_references_the_tunnel_file_directly(tunnels):
    """Bypassing create_tunnel would bypass mine_lock and lose updates."""
    source = open(tunnels.__file__, encoding="utf-8").read()
    assert "_TUNNEL_FILE" not in source
    assert "tunnels.json" not in source
