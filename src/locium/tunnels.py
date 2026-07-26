"""Tunnel writes, delegated to mempalace.

Tunnels persist to JSON, not to ChromaDB, and mempalace already serialises the
load-mutate-save cycle behind a file lock with an atomic replace. Calling its
function is therefore exactly as safe as the MCP server doing it — and writing
the file directly would bypass that lock and silently drop concurrent updates.
"""

import inspect

REQUIRED_FUNCTIONS = ("create_tunnel", "delete_tunnel", "list_tunnels")
REQUIRED_CREATE_PARAMS = (
    "source_wing",
    "source_room",
    "target_wing",
    "target_room",
    "label",
    "source_drawer_id",
    "target_drawer_id",
)


class MempalaceApiError(Exception):
    """Raised when the installed mempalace does not expose the expected API."""


def _graph():
    try:
        from mempalace import palace_graph
    except ImportError as exc:
        raise MempalaceApiError(
            "mempalace is not installed; Locium needs it to write tunnels"
        ) from exc
    return palace_graph


def assert_api() -> None:
    """Fail loudly at startup rather than with a stack trace mid-write."""
    graph = _graph()

    missing = [name for name in REQUIRED_FUNCTIONS if not hasattr(graph, name)]
    if missing:
        raise MempalaceApiError(
            f"mempalace.palace_graph is missing {', '.join(missing)}. "
            f"Verified against mempalace 3.3.3."
        )

    params = inspect.signature(graph.create_tunnel).parameters
    absent = [name for name in REQUIRED_CREATE_PARAMS if name not in params]
    if absent:
        raise MempalaceApiError(
            f"create_tunnel is missing parameters: {', '.join(absent)}. "
            f"Verified against mempalace 3.3.3."
        )


def create(
    source_wing: str,
    source_room: str,
    target_wing: str,
    target_room: str,
    label: str = "",
    source_drawer_id: str | None = None,
    target_drawer_id: str | None = None,
) -> dict:
    """Create or update a tunnel.

    Tunnel identity is the sorted wing/room pair, so a second call for the same
    pair updates the existing record rather than adding one.
    """
    return _graph().create_tunnel(
        source_wing=source_wing,
        source_room=source_room,
        target_wing=target_wing,
        target_room=target_room,
        label=label,
        source_drawer_id=source_drawer_id,
        target_drawer_id=target_drawer_id,
    )


def delete(tunnel_id: str) -> dict:
    return _graph().delete_tunnel(tunnel_id)


def listing() -> list[dict]:
    return _graph().list_tunnels()
