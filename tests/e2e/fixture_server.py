"""Serve Locium against a synthetic palace so e2e runs never touch real data.

Builds its own ChromaDB store and Locium index in a fresh tmp dir, and points
HOME at an isolated directory so nothing can reach the real ``~/.mempalace``.
The already-cached ONNX embedding model is reused via a symlink so search
still works offline instead of trying (and failing) to re-download it under
the isolated HOME.
"""

import os
import shutil
import tempfile
from pathlib import Path

real_home = Path.home()
root = Path(tempfile.mkdtemp(prefix="locium-e2e-"))
fake_home = root / "home"
fake_home.mkdir(parents=True, exist_ok=True)

real_cache = real_home / ".cache" / "chroma"
if real_cache.exists():
    fake_cache_dir = fake_home / ".cache"
    fake_cache_dir.mkdir(parents=True, exist_ok=True)
    os.symlink(real_cache, fake_cache_dir / "chroma")

os.environ["HOME"] = str(fake_home)

import numpy as np
import uvicorn

# (wing, hall, room, count) -- five-plus wings so the footprint's core and
# every side block gets occupied; several distinct halls and rooms; one
# room per wing well past the small dot_cap set below so the capped path
# renders. Counts here are LAYOUT totals; the spec's WING/HALL/CHAMBER
# constants must match this table.
LAYOUT = [
    ("sessions", "technical", "architecture", 20),  # over dot_cap -> capped
    ("sessions", "technical", "problem", 5),
    ("sessions", "diary", "general", 15),  # over dot_cap -> capped
    ("wing_a", "technical", "architecture", 7),
    ("wing_a", "planning", "general", 5),
    ("wing_b", "diary", "general", 10),
    ("wing_c", "technical", "problem", 8),
    ("wing_d", "planning", "architecture", 6),
    ("wing_e", "diary", "problem", 4),
    ("wing_diary", "diary", "diary", 4),  # the diary lens filters to this wing
]
DOT_CAP = 12
PORT = 7799


def make_palace(path: Path) -> None:
    import chromadb

    path.mkdir(parents=True, exist_ok=True)
    collection = chromadb.PersistentClient(path=str(path)).get_or_create_collection(
        name="mempalace_drawers"
    )

    rng = np.random.default_rng(42)
    ids, documents, metadatas = [], [], []
    n = 0
    for wing, hall, room, count in LAYOUT:
        for _ in range(count):
            month = (n % 12) + 1
            day = (n % 27) + 1
            filler = (
                f"Detail sentence number {n} elaborating on {room} matters for "
                "context and testing purposes, well past the preview length. "
            ) * 6
            documents.append(f"Memory {n} filed in {wing}/{hall}/{room}. {filler}")
            metadatas.append(
                {
                    "wing": wing,
                    "hall": hall,
                    "room": room,
                    "source_file": f"f{n}.jsonl",
                    "filed_at": f"2026-{month:02d}-{day:02d}T10:00:00",
                }
            )
            ids.append(f"d{n}")
            n += 1

    vectors = rng.normal(size=(n, 384)).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=[v.tolist() for v in vectors],
        metadatas=metadatas,
    )


def make_kg(beside: Path) -> None:
    """A tiny knowledge graph beside the palace, mempalace's schema."""
    import sqlite3

    db = sqlite3.connect(beside / "knowledge_graph.sqlite3")
    db.execute("CREATE TABLE entities (id TEXT PRIMARY KEY, name TEXT NOT NULL)")
    db.execute(
        "CREATE TABLE triples (id TEXT PRIMARY KEY, subject TEXT, predicate TEXT,"
        " object TEXT, valid_from TEXT, valid_to TEXT,"
        " extracted_at TEXT DEFAULT CURRENT_TIMESTAMP)"
    )
    facts = [
        ("t0", "wing_a", "depends_on", "wing_b", "2026-05-01", None),
        ("t1", "wing_a", "used", "legacy-runner", "2026-01-01", "2026-06-01"),
    ]
    for entity in ("wing_a", "wing_b", "legacy-runner"):
        db.execute("INSERT INTO entities VALUES (?, ?)", (entity, entity))
    db.executemany(
        "INSERT INTO triples (id, subject, predicate, object, valid_from, valid_to)"
        " VALUES (?, ?, ?, ?, ?, ?)",
        facts,
    )
    db.commit()
    db.close()


if __name__ == "__main__":
    try:
        palace = root / "palace"
        index = root / "index"
        make_palace(palace)
        make_kg(root)

        from locium.build import build_index
        from locium.config import Tuning
        from locium.server import create_app

        # cluster_min below the capped chambers' 12 shown drawers so the
        # sub-cluster path renders in e2e as well.
        build_index(palace, index, tuning=Tuning(dot_cap=DOT_CAP, cluster_min=10))

        app = create_app(index, palace)
        uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
    finally:
        shutil.rmtree(root, ignore_errors=True)
