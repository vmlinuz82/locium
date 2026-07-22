"""The index artifact: everything the viewer needs, and nothing live.

Metadata is JSON so it stays readable and debuggable; only the vectors are
binary. The pair is committed as one unit: both files are written into a
staging directory beside the target, then that staging directory is swapped
into place with a single rename. A crash at any point before that rename
leaves the previous artifact (or nothing) in place; a crash after it leaves
the new one. Readers never see a meta.json paired with a stale or missing
vectors.bin.
"""

import json
import os
import shutil
from pathlib import Path

import numpy as np

META_NAME = "meta.json"
VECTORS_NAME = "vectors.bin"


def write_index(path: Path, meta: dict, vectors: np.ndarray) -> None:
    """Write meta.json and vectors.bin as one atomic unit.

    Both files are staged in a sibling ``<name>.new`` directory and only then
    swapped into place, so readers always see either the complete old
    artifact or the complete new one, never a mix of the two.
    """
    if vectors.dtype != np.int8:
        raise ValueError(f"vectors must be int8, got {vectors.dtype}")

    path.parent.mkdir(parents=True, exist_ok=True)

    staging = path.with_name(path.name + ".new")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    (staging / META_NAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (staging / VECTORS_NAME).write_bytes(np.ascontiguousarray(vectors).tobytes())

    old = path.with_name(path.name + ".old")
    if old.exists():
        shutil.rmtree(old)

    moved_old = False
    if path.exists():
        os.replace(path, old)
        moved_old = True

    os.replace(staging, path)

    if moved_old:
        shutil.rmtree(old)


def read_meta(path: Path) -> dict:
    return json.loads((path / META_NAME).read_text(encoding="utf-8"))


def read_vectors(path: Path, count: int, dim: int) -> np.ndarray:
    data = (path / VECTORS_NAME).read_bytes()
    expected = count * dim
    if len(data) != expected:
        raise ValueError(
            f"vectors.bin size mismatch: expected {expected} bytes "
            f"(count={count}, dim={dim}), got {len(data)}"
        )
    raw = np.frombuffer(data, dtype=np.int8)
    return raw.reshape(count, dim)


def index_exists(path: Path) -> bool:
    return (path / META_NAME).exists() and (path / VECTORS_NAME).exists()
