"""Read the palace — always from a throwaway copy, never the live store.

ChromaDB's persistent client makes no multi-process guarantees, and the
MemPalace MCP server holds one open continuously. Copying first makes it
structurally impossible for a build to corrupt the palace.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np

from .config import COLLECTION_NAME
from .models import Drawer


class PalaceNotFound(Exception):
    """Raised when the configured palace path does not exist."""


def snapshot_palace(palace: Path) -> Path:
    """Copy the palace to a temp directory and return the copy's path."""
    if not palace.exists():
        raise PalaceNotFound(
            f"palace not found at {palace}. Pass --palace or set MEMPALACE_PALACE."
        )
    destination = Path(tempfile.mkdtemp(prefix="locium-snapshot-")) / "palace"
    shutil.copytree(palace, destination)
    return destination


def palace_mtime(palace: Path) -> float:
    """Most recent modification time anywhere in the palace directory.

    Uses the filesystem only. Counting drawers would mean opening Chroma,
    which would reintroduce the second-reader risk this module exists to avoid.
    """
    newest = palace.stat().st_mtime
    for child in palace.rglob("*"):
        newest = max(newest, child.stat().st_mtime)
    return newest


def read_drawers(palace_copy: Path) -> tuple[list[Drawer], np.ndarray]:
    """Load every drawer and its embedding from a palace copy."""
    import chromadb

    client = chromadb.PersistentClient(path=str(palace_copy))
    collection = client.get_collection(COLLECTION_NAME)
    payload = collection.get(include=["documents", "metadatas", "embeddings"])

    ids = payload["ids"]
    if not ids:
        return [], np.zeros((0, 0), dtype=np.float32)

    documents = payload["documents"]
    metadatas = payload["metadatas"]

    drawers = [
        Drawer(
            id=drawer_id,
            text=documents[i] or "",
            wing=(metadatas[i] or {}).get("wing", "unknown"),
            room=(metadatas[i] or {}).get("room", "general"),
            created_at=(metadatas[i] or {}).get("created_at", ""),
            source_file=(metadatas[i] or {}).get("source_file", ""),
        )
        for i, drawer_id in enumerate(ids)
    ]

    vectors = np.asarray(payload["embeddings"], dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.where(norms > 1e-9, norms, 1.0)
    return drawers, vectors
