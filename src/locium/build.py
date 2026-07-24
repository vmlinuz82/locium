"""Orchestrate the build: palace copy in, index artifact out.

The pipeline is snapshot -> extract -> layout -> arcs -> quantise -> write.
Layout walks wing -> hall -> chamber, positioning each drawer with
pack_chamber. Wing/hall/chamber geometry is recomputed fresh from the full
current set every build, so it is always overlap-free. A drawer's coordinate
is stable only as long as its chamber (wing, hall, room) is unchanged and its
old coordinate still falls inside that chamber's current rectangle; otherwise
it is re-packed. --refit discards all previous coordinates and re-packs every
chamber from scratch.
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


def _rect_list(rect: Rect) -> list[float]:
    return [rect.x, rect.y, rect.w, rect.h]


def _previous_state(index_path: Path) -> dict[str, tuple[str, str, str, list[float]]]:
    """Return, per drawer id, the (wing, hall, room, [x, y]) it had last build."""
    if not index_exists(index_path):
        return {}
    meta = read_meta(index_path)
    return {
        d["id"]: (d["wing"], d["hall"], d["room"], [d["x"], d["y"]])
        for d in meta["drawers"]
    }


def _inside(rect: Rect, x: float, y: float) -> bool:
    return rect.x <= x <= rect.x + rect.w and rect.y <= y <= rect.y + rect.h


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

    previous = {} if refit else _previous_state(index_path)

    by_wing: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for position, drawer in enumerate(drawers):
        by_wing[drawer.wing][drawer.hall].append(position)

    wing_counts = {w: sum(len(v) for v in halls.values()) for w, halls in by_wing.items()}
    wing_rects = building_footprint(wing_counts)

    wings_meta: list[dict] = []
    halls_meta: list[dict] = []
    chambers_meta: list[dict] = []
    fresh_coords: dict[str, list[float]] = {}

    for wing, wing_rect in wing_rects.items():
        wings_meta.append(
            {"name": wing, "rect": _rect_list(wing_rect), "count": wing_counts[wing]}
        )
        hall_counts = {h: len(rows) for h, rows in by_wing[wing].items()}
        for hall, hall_rect in subdivide(hall_counts, wing_rect, tuning.pad_hall).items():
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
            for room, chamber in subdivide(room_counts, hall_rect, tuning.pad_chamber).items():
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

                # A previous coordinate is kept only when it is still valid:
                # the drawer must still belong to this exact chamber and its
                # old point must still fall inside the chamber's current
                # rectangle. Matching is by drawer id, never by position.
                kept_ids: list[str] = []
                kept_points: list[tuple[float, float]] = []
                for row in shown:
                    did = drawers[row].id
                    prev = previous.get(did)
                    if prev is None:
                        continue
                    pwing, phall, proom, (px, py) = prev
                    if (pwing, phall, proom) != (wing, hall, room):
                        continue
                    if not _inside(chamber, px, py):
                        continue
                    kept_ids.append(did)
                    kept_points.append((px, py))

                kept_id_set = set(kept_ids)
                remaining_rows = [r for r in shown if drawers[r].id not in kept_id_set]

                points = pack_chamber(
                    len(shown),
                    chamber,
                    tuning.seed,
                    placed=kept_points or None,
                )
                new_points = points[len(kept_points):]

                for did, (px, py) in zip(kept_ids, kept_points):
                    fresh_coords[did] = [round(px, 1), round(py, 1)]
                for row, (px, py) in zip(remaining_rows, new_points):
                    fresh_coords[drawers[row].id] = [round(px, 1), round(py, 1)]

    coords = fresh_coords

    # meta["drawers"], vectors.bin and arcs must share one index space: the
    # drawn drawers (those with coordinates), in this exact order. Capped-out
    # drawers have no dot on the map, so a neighbour link or arc to them
    # could never be drawn anyway.
    drawn_indices = [i for i, drawer in enumerate(drawers) if drawer.id in coords]
    drawn_drawers = [drawers[i] for i in drawn_indices]
    drawn_vectors = vectors[drawn_indices]

    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "palace_mtime": mtime,
        "drawer_count": len(drawn_drawers),
        "vector_dim": int(drawn_vectors.shape[1]) if len(drawn_drawers) else 0,
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
            for drawer in drawn_drawers
        ],
        "wings": wings_meta,
        "halls": halls_meta,
        "chambers": chambers_meta,
        "arcs": compute_arcs(
            drawn_vectors,
            [d.wing for d in drawn_drawers],
            tuning.arc_max_distance,
            tuning.arcs_per_drawer,
            tuning.arc_global_cap,
        ),
    }

    texts = {drawer.id: drawer.text for drawer in drawn_drawers}
    write_index(index_path, meta, quantize(drawn_vectors), texts)
    return meta
