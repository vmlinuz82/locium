import socket
from pathlib import Path

import pytest

from locium.cli import find_free_port, main, resolve_palace


def test_palace_defaults_to_the_standard_location(monkeypatch):
    monkeypatch.delenv("MEMPALACE_PALACE", raising=False)
    assert resolve_palace(None) == Path.home() / ".mempalace" / "palace"


def test_env_var_overrides_the_default(monkeypatch):
    monkeypatch.setenv("MEMPALACE_PALACE", "/tmp/elsewhere")
    assert resolve_palace(None) == Path("/tmp/elsewhere")


def test_explicit_flag_beats_the_env_var(monkeypatch):
    monkeypatch.setenv("MEMPALACE_PALACE", "/tmp/elsewhere")
    assert resolve_palace("/tmp/chosen") == Path("/tmp/chosen")


def test_find_free_port_returns_the_preferred_one_when_available():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        free = probe.getsockname()[1]
    assert find_free_port(free) == free


def test_find_free_port_skips_a_busy_port():
    with socket.socket() as taken:
        taken.bind(("127.0.0.1", 0))
        busy = taken.getsockname()[1]
        taken.listen(1)
        assert find_free_port(busy) != busy


def test_build_command_writes_an_index(fake_palace, tmp_path, capsys):
    code = main(["build", "--palace", str(fake_palace), "--index", str(tmp_path / "idx")])
    assert code == 0
    assert (tmp_path / "idx" / "meta.json").exists()
    assert "6 drawers" in capsys.readouterr().out


def test_build_reports_a_missing_palace(tmp_path, capsys):
    code = main(["build", "--palace", str(tmp_path / "gone"), "--index", str(tmp_path / "i")])
    assert code == 1
    assert "not found" in capsys.readouterr().err


def test_serve_refuses_without_an_index(fake_palace, tmp_path, capsys):
    code = main(["serve", "--palace", str(fake_palace), "--index", str(tmp_path / "absent")])
    assert code == 1
    assert "locium build" in capsys.readouterr().err


def test_unknown_command_exits_nonzero():
    with pytest.raises(SystemExit):
        main(["nonsense"])
