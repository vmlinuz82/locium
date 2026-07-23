"""Keep loci stable across rebuilds.

A memory palace depends on things staying where you left them. Once a drawer
has a coordinate it keeps it forever; new drawers are placed into the space
that already exists rather than triggering a re-projection. Only an explicit
--refit moves anything.
"""


def merge_coords(
    previous: dict[str, list[float]],
    fresh: dict[str, list[float]],
    refit: bool,
) -> dict[str, list[float]]:
    """Combine persisted coordinates with freshly computed ones.

    Without ``refit``, previously placed drawers keep their exact coordinates
    and only drawers absent from the index take their fresh position. Drawers
    that have left the palace are dropped.
    """
    if refit:
        return dict(fresh)

    merged = {did: xy for did, xy in previous.items() if did in fresh}
    for did, xy in fresh.items():
        merged.setdefault(did, xy)
    return merged
