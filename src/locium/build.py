"""Orchestrate the build: palace copy in, index artifact out.

The pipeline is snapshot -> extract -> layout -> arcs -> quantise -> write.
Layout walks wing -> hall -> chamber, positioning each drawer with
pack_chamber. Wings already in the index keep their persisted rectangle and
drawers already in the index keep their exact coordinates, so nothing an
existing locus depends on moves on an ordinary rebuild. Only --refit moves
anything.
"""

import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .arcs import compute_arcs
from .config import TUNING, Tuning
from .extract import palace_mtime, read_drawers, snapshot_palace
from .footprint import building_footprint, subdivide
from .index import index_exists, read_meta, write_index
from .models import Rect, make_preview
from .packing import pack_chamber
from .quantize import quantize
from .stability import merge_coords

PAD_HALL, PAD_CHAMBER = 8.5, 5.0


def _rect_list(rect: Rect) -> list[float]:
    return [rect.x, rect.y, rect.w, rect.h]


def _previous_state(index_path: Path) -> tuple[dict, dict]:
    """Return (coords by drawer id, rect by wing name) from any existing index."""
    if not index_exists(index_path):
        return {}, {}
    meta = read_meta(index_path)
    coords = {d["id"]: [d["x"], d["y"]] for d in meta["drawers"]}
    rects = {w["name"]: Rect(*w["rect"]) for w in meta["wings"]}
    return coords, rects


def _resolve_wing_rects(
    counts: dict[str, int], previous_rects: dict[str, Rect], refit: bool
) -> dict[str, Rect]:
    """Keep known wings where they are; place new ones with building_footprint."""
    if refit or not previous_rects:
        return building_footprint(counts)

    known = {name: rect for name, rect in previous_rects.items() if name in counts}
    fresh = {name: counts[name] for name in counts if name not in known}
    if fresh:
        known.update(building_footprint(fresh))
    return known


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

    by_wing: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for position, drawer in enumerate(drawers):
        by_wing[drawer.wing][drawer.hall].append(position)

    wing_counts = {w: sum(len(v) for v in halls.values()) for w, halls in by_wing.items()}
    wing_rects = _resolve_wing_rects(wing_counts, previous_rects, refit)

    wings_meta: list[dict] = []
    halls_meta: list[dict] = []
    chambers_meta: list[dict] = []
    fresh_coords: dict[str, list[float]] = {}

    for wing, wing_rect in wing_rects.items():
        wings_meta.append(
            {"name": wing, "rect": _rect_list(wing_rect), "count": wing_counts[wing]}
        )
        hall_counts = {h: len(rows) for h, rows in by_wing[wing].items()}
        for hall, hall_rect in subdivide(hall_counts, wing_rect, PAD_HALL).items():
            halls_meta.append(
                {
                    "name": hall,
                    "wing": wing,
                    "rect": _rect_list(hall_rect),
                    "count": hall_counts[hall],
                }
            )
            rows_by_room: dict[str, list[int]] = defaultdict(list)
            for row in by_wing[wing][hall]:
                rows_by_room[drawers[row].room].append(row)

            room_counts = {r: len(v) for r, v in rows_by_room.items()}
            for room, chamber in subdivide(room_counts, hall_rect, PAD_CHAMBER).items():
                rows = rows_by_room[room]
                shown = rows[: tuning.dot_cap]
                chambers_meta.append(
                    {
                        "name": room,
                        "wing": wing,
                        "hall": hall,
                        "rect": _rect_list(chamber),
                        "count": len(rows),
                        "capped": len(rows) > tuning.dot_cap,
                    }
                )
                kept = (
                    [
                        previous_coords[drawers[r].id]
                        for r in shown
                        if drawers[r].id in previous_coords
                    ]
                    if not refit
                    else []
                )
                points = pack_chamber(
                    len(shown),
                    chamber,
                    tuning.seed,
                    placed=[tuple(p) for p in kept] or None,
                )
                for row, (px, py) in zip(shown, points):
                    fresh_coords[drawers[row].id] = [round(px, 1), round(py, 1)]

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
                "hall": drawer.hall,
                "room": drawer.room,
                "date": drawer.created_at,
                "x": coords[drawer.id][0],
                "y": coords[drawer.id][1],
                "preview": make_preview(drawer.text, tuning.preview_chars),
            }
            for drawer in drawers
            if drawer.id in coords
        ],
        "wings": wings_meta,
        "halls": halls_meta,
        "chambers": chambers_meta,
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
