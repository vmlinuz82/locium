"""Read-only snapshot of the MemPalace knowledge graph.

The KG lives beside the palace as knowledge_graph.sqlite3 -- a few hundred
subject/predicate/object triples, which is the second of the three recall
legs an agent runs (search, KG query, diary). Snapshotting it into the index
at build time keeps the viewer's rule intact: serve reads only the artifact,
never the live store. The database is opened read-only and any failure --
absent file, locked WAL, foreign schema -- degrades to an empty snapshot
rather than failing the build; the KG is a lens here, not a requirement.
"""

import sqlite3
from pathlib import Path

KG_FILENAME = "knowledge_graph.sqlite3"

EMPTY = {"entity_count": 0, "triples": []}


def kg_path(palace: Path) -> Path:
    """MemPalace keeps the KG beside the palace directory, not inside it."""
    return palace.parent / KG_FILENAME


def read_kg(palace: Path) -> dict:
    """Snapshot entities count and all triples, or the empty shape."""
    file = kg_path(palace)
    if not file.exists():
        return dict(EMPTY)

    try:
        db = sqlite3.connect(f"file:{file}?mode=ro", uri=True)
        try:
            entity_count = db.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
            rows = db.execute(
                "SELECT subject, predicate, object, valid_from, valid_to "
                "FROM triples ORDER BY extracted_at"
            ).fetchall()
        finally:
            db.close()
    except sqlite3.Error:
        return dict(EMPTY)

    return {
        "entity_count": int(entity_count),
        "triples": [
            {"s": s, "p": p, "o": o, "from": vf, "to": vt}
            for s, p, o, vf, vt in rows
        ],
    }
