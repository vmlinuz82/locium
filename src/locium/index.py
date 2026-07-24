"""The index artifact: everything the viewer needs, and nothing live.

Metadata is JSON so it stays readable and debuggable; only the vectors are
binary. The artifact is committed as one unit: every file is written into a
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
TEXTS_NAME = "texts.json"


def write_index(
    path: Path, meta: dict, vectors: np.ndarray, texts: dict[str, str] | None = None
) -> None:
    """Write meta.json, vectors.bin and (optionally) texts.json as one atomic unit.

    All files are staged in a sibling ``<name>.new`` directory and only then
    swapped into place, so readers always see either the complete old
    artifact or the complete new one, never a mix of the two. ``texts`` is
    optional so callers that don't have full drawer text (e.g. tests) can
    omit it without producing a texts.json.
    """
    if vectors.dtype != np.int8:
        raise ValueError(f"vectors must be int8, got {vectors.dtype}")

    path.parent.mkdir(parents=True, exist_ok=True)

    old = path.with_name(path.name + ".old")

    # If `path` is missing while `.old` is present, a previous swap was
    # interrupted between moving `path` aside and replacing it with the
    # staged directory: `.old` is the last-known-good artifact, not a
    # stale leftover, so it must be restored before anything else runs.
    # Otherwise the unconditional cleanup below would delete the one
    # surviving copy, and a failure during this retry would then make
    # the loss permanent.
    if not path.exists() and old.exists():
        os.replace(old, path)

    staging = path.with_name(path.name + ".new")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    (staging / META_NAME).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (staging / VECTORS_NAME).write_bytes(np.ascontiguousarray(vectors).tobytes())
    if texts is not None:
        (staging / TEXTS_NAME).write_text(json.dumps(texts, indent=2), encoding="utf-8")

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
    """Read vectors.bin and reshape it to (count, dim).

    Validates that the file's byte length equals ``count * dim``, which
    catches truncation or a caller passing the wrong ``count``/``dim``.
    It cannot by itself detect a same-length but stale ``vectors.bin`` --
    a length check can't distinguish correct new bytes from stale bytes
    of identical length. That case is ruled out structurally by
    ``write_index``'s atomic staging swap, which makes it impossible for
    ``meta.json`` and ``vectors.bin`` to come from different generations.
    """
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


def read_text(path: Path, drawer_id: str) -> str | None:
    """Read one drawer's full text from texts.json.

    Tolerates a missing or malformed texts.json by returning None -- the
    map is a nice-to-have alongside meta.json's preview, not a required
    part of the artifact.
    """
    try:
        texts = json.loads((path / TEXTS_NAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(texts, dict):
        return None
    return texts.get(drawer_id)
