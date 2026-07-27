import sqlite3

import pytest

from locium.kg import KG_FILENAME, read_kg


def _make_kg(root, triples):
    """A knowledge_graph.sqlite3 beside the palace, matching mempalace's schema."""
    db = sqlite3.connect(root / KG_FILENAME)
    db.execute("CREATE TABLE entities (id TEXT PRIMARY KEY, name TEXT NOT NULL)")
    db.execute(
        "CREATE TABLE triples (id TEXT PRIMARY KEY, subject TEXT, predicate TEXT,"
        " object TEXT, valid_from TEXT, valid_to TEXT,"
        " extracted_at TEXT DEFAULT CURRENT_TIMESTAMP)"
    )
    for i, (s, p, o, vf, vt) in enumerate(triples):
        db.execute("INSERT OR IGNORE INTO entities VALUES (?, ?)", (s, s))
        db.execute("INSERT OR IGNORE INTO entities VALUES (?, ?)", (o, o))
        db.execute(
            "INSERT INTO triples (id, subject, predicate, object, valid_from, valid_to)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (f"t{i}", s, p, o, vf, vt),
        )
    db.commit()
    db.close()


def test_read_kg_snapshots_triples_beside_the_palace(tmp_path):
    palace = tmp_path / "palace"
    palace.mkdir()
    _make_kg(tmp_path, [
        ("email-tools", "uses", "symfony", "2026-04-22", None),
        ("old-lib", "status", "removed", "2026-01-01", "2026-06-01"),
    ])

    kg = read_kg(palace)

    assert kg["entity_count"] == 4
    assert len(kg["triples"]) == 2
    assert kg["triples"][0] == {
        "s": "email-tools", "p": "uses", "o": "symfony",
        "from": "2026-04-22", "to": None,
    }
    assert kg["triples"][1]["to"] == "2026-06-01"


def test_read_kg_is_empty_when_no_graph_exists(tmp_path):
    palace = tmp_path / "palace"
    palace.mkdir()
    assert read_kg(palace) == {"entity_count": 0, "triples": []}


def test_read_kg_survives_a_foreign_schema(tmp_path):
    palace = tmp_path / "palace"
    palace.mkdir()
    db = sqlite3.connect(tmp_path / KG_FILENAME)
    db.execute("CREATE TABLE something_else (x INTEGER)")
    db.commit()
    db.close()

    assert read_kg(palace) == {"entity_count": 0, "triples": []}
