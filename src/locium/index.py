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
STITCHES_NAME = "stitches.json"
FAMILY_VECTORS_NAME = "family_vectors.bin"


def write_index(
    path: Path,
    meta: dict,
    vectors: np.ndarray,
    texts: dict[str, str] | None = None,
    stitches: dict | None = None,
    family_vectors: np.ndarray | None = None,
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
    if family_vectors is not None and family_vectors.dtype != np.int8:
        raise ValueError(f"family_vectors must be int8, got {family_vectors.dtype}")

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
    if stitches is not None:
        (staging / STITCHES_NAME).write_text(
            json.dumps(stitches, indent=2), encoding="utf-8"
        )
    if family_vectors is not None:
        (staging / FAMILY_VECTORS_NAME).write_bytes(
            np.ascontiguousarray(family_vectors).tobytes()
        )

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


# texts.json parsed once per artifact generation, keyed by index path. A
# full-text scan cannot re-parse a multi-megabyte file on every keystroke, so
# the parsed map is held and revalidated against the file's identity.
_TEXTS_CACHE: dict[Path, tuple[tuple[int, int, int], dict[str, str]]] = {}


def _identity(file: Path) -> tuple[int, int, int]:
    """A fingerprint that changes whenever texts.json is rebuilt.

    mtime alone is not enough: Linux stamps inodes from a coarse clock that
    only advances once per timer tick, so two writes a few milliseconds apart
    get byte-identical ``st_mtime_ns`` and a rebuild would go unnoticed. The
    inode is what actually distinguishes them -- ``write_index`` never mutates
    in place, it stages a new file and renames it over the old one.
    """
    stat = file.stat()
    return (stat.st_ino, stat.st_size, stat.st_mtime_ns)


def load_texts(path: Path) -> dict[str, str]:
    """Every drawer's full text, cached. Empty dict if texts.json is unusable."""
    file = path / TEXTS_NAME
    try:
        stamp = _identity(file)
    except OSError:
        return {}

    cached = _TEXTS_CACHE.get(path)
    if cached is not None and cached[0] == stamp:
        return cached[1]

    try:
        texts = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(texts, dict):
        return {}

    _TEXTS_CACHE[path] = (stamp, texts)
    return texts


_STITCHES_CACHE: dict[Path, tuple[tuple[int, int, int], dict]] = {}

_EMPTY_STITCHES = {"families": {}, "member": {}}


def load_stitches(path: Path) -> dict:
    """The reassembly map, cached like texts. Empty shape when absent.

    Same identity-based invalidation as load_texts: mtime alone cannot tell
    two rebuilds apart, the inode can.
    """
    file = path / STITCHES_NAME
    try:
        stamp = _identity(file)
    except OSError:
        return dict(_EMPTY_STITCHES)

    cached = _STITCHES_CACHE.get(path)
    if cached is not None and cached[0] == stamp:
        return cached[1]

    try:
        stitches = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return dict(_EMPTY_STITCHES)
    if not isinstance(stitches, dict) or not isinstance(stitches.get("member"), dict):
        return dict(_EMPTY_STITCHES)

    _STITCHES_CACHE[path] = (stamp, stitches)
    return stitches


# A snippet is for judging a result, not reading it -- the full-text popup is
# for reading. At the panel's width this is roughly five lines, so forty
# results stay scannable. What makes it useful is not its length but that it
# is centred on the match: the first 200 characters of a log or a directory
# listing say nothing about why the drawer came back.
SNIPPET_WIDTH = 240


def snippet(text: str, needle: str, width: int = SNIPPET_WIDTH) -> str:
    """A readable window of ``text``, centred on ``needle`` where it occurs.

    meta.json only carries the first ``preview_chars`` of a drawer, which for
    machine-generated content (a directory listing, a log) is routinely the
    least informative part of it. Centring on the match instead shows why the
    drawer came back at all.
    """
    if len(text) <= width:
        return text

    at = text.lower().find(needle.lower()) if needle else -1
    if at < 0:
        return text[:width].rstrip() + "…"

    # Keep a third of the window ahead of the match so there is context on
    # both sides, then clamp so the tail of a document still fills the window.
    start = max(0, min(at - width // 3, len(text) - width))
    end = min(len(text), start + width)
    lead = "…" if start > 0 else ""
    tail = "…" if end < len(text) else ""
    return f"{lead}{text[start:end].strip()}{tail}"


def snippets(path: Path, drawer_ids: list[str], needle: str) -> dict[str, str]:
    """Snippet per id, skipping ids with no stored text."""
    texts = load_texts(path)
    return {i: snippet(texts[i], needle) for i in drawer_ids if i in texts}


def search_texts(path: Path, needle: str, limit: int) -> list[str]:
    """Ids of drawers whose full text contains ``needle``, case-insensitively.

    A literal substring match, which is what finds an identifier like
    "EM-4103" that the embedding has no reason to rank highly. Hits come back
    in artifact order rather than ranked -- a substring either occurs or it
    does not, so there is no score to sort on.
    """
    lowered = needle.lower()
    hits: list[str] = []
    for drawer_id, text in load_texts(path).items():
        if lowered in text.lower():
            hits.append(drawer_id)
            if len(hits) >= limit:
                break
    return hits


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
