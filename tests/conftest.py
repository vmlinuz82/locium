"""Shared fixtures. A fake palace is a real ChromaDB store in a tmpdir."""

import numpy as np
import pytest


@pytest.fixture
def fake_palace(tmp_path):
    """Build a small real Chroma collection matching mempalace's schema."""
    import chromadb

    palace = tmp_path / "palace"
    palace.mkdir()
    client = chromadb.PersistentClient(path=str(palace))
    collection = client.get_or_create_collection(name="mempalace_drawers")

    rng = np.random.default_rng(0)
    count = 6
    vectors = rng.normal(size=(count, 8)).astype(np.float32)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    collection.add(
        ids=[f"d{i}" for i in range(count)],
        documents=[f"document number {i} about docker and compose" for i in range(count)],
        embeddings=[v.tolist() for v in vectors],
        metadatas=[
            {
                "wing": "alpha" if i < 3 else "beta",
                "room": "technical",
                "source_file": f"f{i}.jsonl",
                "created_at": f"2026-05-0{i + 1}T10:00:00",
            }
            for i in range(count)
        ],
    )
    return palace
