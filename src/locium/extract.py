"""Read the palace — always from a throwaway copy, never the live store.

ChromaDB's persistent client makes no multi-process guarantees, and the
MemPalace MCP server holds one open continuously. Copying first makes it
structurally impossible for a build to corrupt the palace.
"""

import shutil
import tempfile
import warnings
from pathlib import Path

import numpy as np

from .config import COLLECTION_NAME
from .models import Drawer

PAGE_SIZE = 500


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
    """Load every drawer and its embedding from a palace copy.

    Pages through the collection instead of fetching it all in one call —
    against a large real palace a single ``get()`` can raise
    ``chromadb.errors.InternalError`` deep in the storage layer for reasons
    unrelated to our code (observed: genuine data corruption). A page that
    raises is retried in shrinking sub-slices to salvage whatever inside it
    is still readable; a slice that still fails at a single record is
    genuinely unreadable and is skipped. Skipped records are reported via a
    ``UserWarning`` so the loss is visible rather than silent.
    """
    import chromadb

    client = chromadb.PersistentClient(path=str(palace_copy))
    collection = client.get_collection(COLLECTION_NAME)
    total = collection.count()
    if total == 0:
        return [], np.zeros((0, 0), dtype=np.float32)

    include = ["documents", "metadatas", "embeddings"]

    def read_slice(offset: int, limit: int) -> tuple[list, list, list, list, int]:
        """Fetch [offset, offset + limit) records, narrowing on failure.

        Returns (ids, documents, metadatas, embeddings, skipped_count).
        """
        try:
            payload = collection.get(limit=limit, offset=offset, include=include)
        except chromadb.errors.InternalError:
            if limit <= 1:
                return [], [], [], [], 1
            half = limit // 2
            left = read_slice(offset, half)
            right = read_slice(offset + half, limit - half)
            return tuple(a + b for a, b in zip(left, right))
        return (
            payload["ids"],
            payload["documents"],
            payload["metadatas"],
            list(payload["embeddings"]),
            0,
        )

    ids: list = []
    documents: list = []
    metadatas: list = []
    embeddings: list = []
    skipped = 0

    for offset in range(0, total, PAGE_SIZE):
        limit = min(PAGE_SIZE, total - offset)
        page_ids, page_docs, page_meta, page_emb, page_skipped = read_slice(offset, limit)
        ids.extend(page_ids)
        documents.extend(page_docs)
        metadatas.extend(page_meta)
        embeddings.extend(page_emb)
        skipped += page_skipped

    if skipped:
        warnings.warn(
            f"Skipped {skipped} unreadable drawer(s) out of {total} while reading the "
            f"palace ({len(ids)} recovered).",
            stacklevel=2,
        )

    if not ids:
        return [], np.zeros((0, 0), dtype=np.float32)

    drawers = [
        Drawer(
            id=drawer_id,
            text=documents[i] or "",
            wing=(metadatas[i] or {}).get("wing", "unknown"),
            hall=(metadatas[i] or {}).get("hall") or "unfiled",
            room=(metadatas[i] or {}).get("room", "general"),
            # MemPalace writes the timestamp as "filed_at"; "created_at" is kept
            # as a fallback for other stores that may use that key instead.
            created_at=(metadatas[i] or {}).get(
                "filed_at", (metadatas[i] or {}).get("created_at", "")
            ),
            source_file=(metadatas[i] or {}).get("source_file", ""),
        )
        for i, drawer_id in enumerate(ids)
    ]

    vectors = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.where(norms > 1e-9, norms, 1.0)
    return drawers, vectors
