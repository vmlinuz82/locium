"""Command line entry point: locium build / locium serve."""

import argparse
import os
import socket
import sys
from pathlib import Path

from .config import DEFAULT_INDEX, DEFAULT_PALACE

DEFAULT_PORT = 7777
PORT_SEARCH_LIMIT = 20


def resolve_palace(explicit: str | None) -> Path:
    """Flag beats environment beats default."""
    if explicit:
        return Path(explicit)
    from_env = os.environ.get("MEMPALACE_PALACE")
    return Path(from_env) if from_env else DEFAULT_PALACE


def find_free_port(preferred: int) -> int:
    for candidate in range(preferred, preferred + PORT_SEARCH_LIMIT):
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", candidate))
                return candidate
            except OSError:
                continue
    raise RuntimeError(f"no free port in {preferred}..{preferred + PORT_SEARCH_LIMIT}")


def _build(args: argparse.Namespace) -> int:
    from .build import build_index
    from .extract import PalaceNotFound

    palace = resolve_palace(args.palace)
    print(f"Building from {palace}", flush=True)
    try:
        meta = build_index(
            palace,
            Path(args.index),
            refit=args.refit,
            # flush: a build is slow enough that a block-buffered pipe would
            # show nothing at all until it finished, which is the problem.
            progress=lambda message: print(message, flush=True),
        )
    except PalaceNotFound as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Indexed {meta['drawer_count']} drawers into {args.index}")
    return 0


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .server import create_app

    try:
        app = create_app(Path(args.index), resolve_palace(args.palace))
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    port = find_free_port(args.port)
    print(f"Locium on http://127.0.0.1:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="locium")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("build", "serve"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--palace", default=None, help="path to the MemPalace store")
        sub.add_argument("--index", default=str(DEFAULT_INDEX), help="index location")

    build_parser = subparsers.choices["build"]
    build_parser.add_argument(
        "--refit",
        action="store_true",
        help="re-project everything; MOVES existing drawers and invalidates the "
        "map you have memorised",
    )

    subparsers.choices["serve"].add_argument("--port", type=int, default=DEFAULT_PORT)

    args = parser.parse_args(argv)
    return _build(args) if args.command == "build" else _serve(args)


if __name__ == "__main__":
    raise SystemExit(main())
