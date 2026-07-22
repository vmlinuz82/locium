"""The index artifact: everything the viewer needs, and nothing live.

Metadata is JSON so it stays readable and debuggable; only the vectors are
binary. Writes go to temp files and are moved into place, so a crash mid-write
never leaves a half-written index behind.
"""

import json
import os
from pathlib import Path

import numpy as np

META_NAME = "meta.json"
VECTORS_NAME = "vectors.bin"


def write_index(path: Path, meta: dict, vectors: np.ndarray) -> None:
    """Write meta.json and vectors.bin atomically."""
    if vectors.dtype != np.int8:
        raise ValueError(f"vectors must be int8, got {vectors.dtype}")

    path.mkdir(parents=True, exist_ok=True)

    meta_tmp = path / (META_NAME + ".tmp")
    meta_tmp.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    os.replace(meta_tmp, path / META_NAME)

    vectors_tmp = path / (VECTORS_NAME + ".tmp")
    vectors_tmp.write_bytes(np.ascontiguousarray(vectors).tobytes())
    os.replace(vectors_tmp, path / VECTORS_NAME)


def read_meta(path: Path) -> dict:
    return json.loads((path / META_NAME).read_text(encoding="utf-8"))


def read_vectors(path: Path, count: int, dim: int) -> np.ndarray:
    raw = np.frombuffer((path / VECTORS_NAME).read_bytes(), dtype=np.int8)
    return raw.reshape(count, dim)


def index_exists(path: Path) -> bool:
    return (path / META_NAME).exists() and (path / VECTORS_NAME).exists()
