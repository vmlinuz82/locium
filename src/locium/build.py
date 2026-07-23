"""Orchestrate the build: palace copy in, index artifact out.

The pipeline is snapshot -> extract -> layout -> cluster -> arcs -> quantise
-> write. Layout is the subtle part: wings that already have coordinates only
get their *new* drawers placed, so nothing an existing locus depends on moves.
"""

import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .arcs import compute_arcs
from .clusters import cluster_labels
from .config import CANVAS_H, CANVAS_W, TUNING, Tuning
from .extract import palace_mtime, read_drawers, snapshot_palace
from .index import index_exists, read_meta, write_index
from .layout import umap_layout
from .models import Rect, make_preview
from .quantize import quantize
from .stability import merge_coords, place_into_existing
from .treemap import carve_gutter, gutter_rect, wing_rects

CANVAS = Rect(0.0, 0.0, CANVAS_W, CANVAS_H)


def _previous_state(index_path: Path) -> tuple[dict, dict]:
    """Return (coords by drawer id, rect by wing name) from any existing index."""
    if not index_exists(index_path):
        return {}, {}
    meta = read_meta(index_path)
    coords = {d["id"]: [d["x"], d["y"]] for d in meta["drawers"]}
    rects = {w["name"]: Rect(*w["rect"]) for w in meta["wings"]}
    return coords, rects


def _resolve_rects(
    counts: dict[str, int], previous_rects: dict[str, Rect], refit: bool, tuning: Tuning
) -> dict[str, Rect]:
    """Keep known wings where they are; carve new ones out of the gutter."""
    if refit or not previous_rects:
        return wing_rects(counts, CANVAS, tuning.gutter_fraction)

    known = {name: rect for name, rect in previous_rects.items() if name in counts}
    fresh = {name: count for name, count in counts.items() if name not in known}
    if fresh:
        known.update(carve_gutter(fresh, gutter_rect(CANVAS, tuning.gutter_fraction)))
    return known


def _layout_wing(
    vectors: np.ndarray,
    ids: list[str],
    rect: Rect,
    previous_coords: dict[str, list[float]],
    tuning: Tuning,
) -> np.ndarray:
    """Position one wing's drawers, reusing placements wherever they exist."""
    placed_mask = np.array([did in previous_coords for did in ids])
    if not placed_mask.any():
        return umap_layout(vectors, rect, tuning.seed, tuning.wing_umap_threshold)

    coords = np.zeros((len(ids), 2), dtype=np.float32)
    coords[placed_mask] = np.array(
        [previous_coords[did] for did, keep in zip(ids, placed_mask) if keep],
        dtype=np.float32,
    )
    if (~placed_mask).any():
        coords[~placed_mask] = place_into_existing(
            vectors[~placed_mask],
            vectors[placed_mask],
            coords[placed_mask],
            seed=tuning.seed,
        )
    return coords


def build_index(
    palace: Path,
    index_path: Path,
    refit: bool = False,
    tuning: Tuning = TUNING,
) -> dict:
    """Build the index artifact from the palace. Returns the meta dict."""
    snapshot = snapshot_palace(palace)
    mtime = palace_mtime(palace)
    try:
        drawers, vectors = read_drawers(snapshot)
    finally:
        shutil.rmtree(snapshot.parent, ignore_errors=True)

    previous_coords, previous_rects = _previous_state(index_path)

    by_wing: dict[str, list[int]] = defaultdict(list)
    for position, drawer in enumerate(drawers):
        by_wing[drawer.wing].append(position)

    counts = {wing: len(rows) for wing, rows in by_wing.items()}
    rects = _resolve_rects(counts, previous_rects, refit, tuning)

    fresh_coords: dict[str, list[float]] = {}
    clusters: list[dict] = []
    assignments = np.full(len(drawers), -1, dtype=int)

    for wing, rows in by_wing.items():
        wing_vectors = vectors[rows]
        wing_ids = [drawers[i].id for i in rows]
        placed = _layout_wing(
            wing_vectors,
            wing_ids,
            rects[wing],
            {} if refit else previous_coords,
            tuning,
        )
        for position, row in enumerate(rows):
            fresh_coords[drawers[row].id] = [
                float(placed[position][0]),
                float(placed[position][1]),
            ]

        wing_clusters, wing_assignments = cluster_labels(
            placed, [drawers[i].text for i in rows]
        )
        for cluster in wing_clusters:
            clusters.append({"wing": wing, **cluster})
        for position, row in enumerate(rows):
            assignments[row] = wing_assignments[position]

    coords = merge_coords(previous_coords, fresh_coords, refit)

    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "palace_mtime": mtime,
        "drawer_count": len(drawers),
        "vector_dim": int(vectors.shape[1]) if len(drawers) else 0,
        "seed": tuning.seed,
        "drawers": [
            {
                "id": drawer.id,
                "wing": drawer.wing,
                "room": drawer.room,
                "date": drawer.created_at,
                "x": coords[drawer.id][0],
                "y": coords[drawer.id][1],
                "preview": make_preview(drawer.text, tuning.preview_chars),
                "cluster": int(assignments[position]),
            }
            for position, drawer in enumerate(drawers)
        ],
        "wings": [
            {"name": wing, "rect": [r.x, r.y, r.w, r.h], "count": counts[wing]}
            for wing, r in rects.items()
        ],
        "clusters": clusters,
        "arcs": compute_arcs(
            vectors,
            [d.wing for d in drawers],
            tuning.arc_max_distance,
            tuning.arcs_per_drawer,
            tuning.arc_global_cap,
        ),
    }

    write_index(index_path, meta, quantize(vectors))
    return meta
