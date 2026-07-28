"""Reassembly map for exchanges the miner split across drawers.

The convo miner slices one user-turn + response exchange into consecutive
``chunk_index`` drawers when it exceeds its chunk size. The slices are exact
partitions of the original text, so concatenating a contiguous run
reconstructs the message — a drawer that reads "…RecordSentEve" is completed
by its sibling starting "nt.php".

This module computes those runs at build time. A family starts at a chunk
whose text carries the "> " user-turn marker (the exchange head) and extends
through contiguous following chunks until the next exchange begins. Anything
ambiguous is left unstitched rather than guessed at: a gap in the index
sequence (the miner skips sub-minimum slivers), a missing head (paragraph-mode
files have no markers), or a lone chunk that already holds its whole exchange.
"""

from collections import defaultdict

from .models import Drawer


def build_families(drawers: list[Drawer]) -> dict:
    """Chunk families: {"families": {...}, "member": {...}, "docs": [...]}.

    ``families`` maps a synthetic key to the ordered drawer ids of one unit;
    ``member`` maps each participating drawer id back to its key. Two kinds
    of unit exist, and ``docs`` lists the keys of the second:

    - exchange families: contiguous runs opened by a "> " head, exact
      partitions of one message (reassembled by plain concatenation);
    - document families: ALL chunks of a source file that has no "> " heads
      at all -- a mined tool-output file or pasted document sliced by the
      paragraph chunker. Slices of one document, but not exact partitions,
      so readers must rejoin them with a separator, not concatenation.

    Drawers without a source_file or chunk_index never participate.
    """
    by_source: dict[str, list[Drawer]] = defaultdict(list)
    for drawer in drawers:
        if drawer.source_file and drawer.chunk_index is not None:
            by_source[drawer.source_file].append(drawer)

    families: dict[str, list[str]] = {}
    member: dict[str, str] = {}
    docs: list[str] = []

    def flush(run: list[Drawer], doc: bool = False) -> None:
        if len(run) < 2:
            return
        key = f"f{len(families)}"
        families[key] = [d.id for d in run]
        for d in run:
            member[d.id] = key
        if doc:
            docs.append(key)

    for chunks in by_source.values():
        chunks.sort(key=lambda d: d.chunk_index)

        if not any(chunk.text.startswith(">") for chunk in chunks):
            flush(chunks, doc=True)
            continue

        run: list[Drawer] = []
        for drawer in chunks:
            starts_exchange = drawer.text.startswith(">")
            contiguous = run and drawer.chunk_index == run[-1].chunk_index + 1

            if starts_exchange or not contiguous:
                flush(run)
                # Only an exchange head may open a run: a continuation chunk
                # after a gap has no head to attach to, so it stays single.
                run = [drawer] if starts_exchange else []
                continue
            run.append(drawer)

        flush(run)

    return {"families": families, "member": member, "docs": docs}
