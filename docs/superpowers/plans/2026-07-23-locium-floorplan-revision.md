# Locium — Floorplan Revision

> Amends `2026-07-23-locium.md`. Tasks 1, 2, 5, 7–15 stand as built. This document
> replaces Tasks 3, 4 and 16, rewrites 17, and retires two modules.

**Date:** 2026-07-23
**Status:** approved from an interactive prototype validated against the real palace

## What changed and why

The original design put drawers where their *meaning* put them: a squarified treemap
gave each wing a rectangle sized by drawer count, and UMAP positioned drawers inside it
so proximity meant similarity.

Two things killed that on real data.

**One wing owns 88% of the store.** `sessions` holds 5,785 of 6,574 drawers. Area
proportional to count gave it 876×850 of a 1000×1000 canvas and squeezed the rest into
slivers — `niamavreme-infra` came out 31×850, a 27:1 aspect ratio. UMAP inside a
31-unit-wide sliver can only draw a vertical line. The rendered map was stripes and one
blob.

**The store already has a building in it.** Every drawer carries `wing`, `hall` and
`room` in its metadata. Locium was using two of the three and had never touched `hall`.
Rendering the actual hierarchy as a floorplan is both truer to the data and immune to
the imbalance, because a floorplan does not have to size rooms by how full they are.

The replacement is an architect's plan view: one connected building, drawers as dots in
their chambers.

### What this costs

Position no longer encodes meaning. Proximity on screen means "same room", not "similar
content". Neighbours remain discoverable by clicking a drawer (client-side k-NN over the
quantised vectors) but not by looking. That trade was made deliberately.

## Measured shape of the real palace

| | |
|---|---|
| Drawers | 6,574 sampled (≈230 unreadable, see [[mempalace-palace-corruption]]) |
| Wings | 17 — `sessions` 5,785, then 199, 197, 131, 90, 52, 34, 31, 12, 9, 8, 7, 6, 4, 4, 2, 1 |
| Halls | 10 distinct names, 60 non-empty (wing, hall) pairs |
| Rooms | 13 distinct names, 138 non-empty (wing, hall, room) chambers |
| Chamber sizes | max 1,656 · median 3 · 81 of 138 hold fewer than 5 |

`wing`/`hall`/`room` are three independent facets, not a strict tree — all 10 halls
appear in all 17 wings. That is fine as architecture: `(sessions → technical)` and
`(niamavreme → technical)` are two distinct rooms that share a name, the way every floor
of a building has its own kitchen.

## Global constraints (additions)

- **Room size is architectural, not proportional.** Blocks are sized on
  `log(count + 1)`, so every chamber stays legible. Proportional sizing would make 81 of
  138 chambers smaller than their own labels. Fullness is shown by dot density.
- **The building is one connected mass.** Every block shares a full edge with the core.
  No courtyards, no free-standing towers, no curtain wall.
- **Every block must be occupied.** An empty block silently deletes its step from the
  outline — the layout does not error, it just stops being the shape that was designed.
- **Text is not embedded in the index.** `meta.json` carries the 200-char preview only;
  full document text is fetched per drawer from the server.
- Deterministic: same input and seed produce byte-identical geometry.

## Retired

- **`src/locium/layout.py`** (Task 4, per-wing UMAP) — nothing positions by embedding any
  more. Delete the module and `tests/test_layout.py`.
- **`src/locium/clusters.py`** (Task 6, HDBSCAN + TF-IDF labels) — chambers now carry
  real names from `room`, so discovered labels have nothing to label. Delete the module
  and `tests/test_clusters.py`.

Both were correct and reviewed; they are removed because the design moved, not because
they were wrong. `arcs.py` and `quantize.py` stay — arcs feed tunnel candidates, and the
quantised vectors still power click-time k-NN.

---

### Task R1: Drawer gains `hall`

**Files:**
- Modify: `src/locium/models.py`, `src/locium/extract.py`
- Test: `tests/test_models.py`, `tests/test_extract.py`, `tests/conftest.py`

**Interfaces:**
- Produces: `Drawer(id, text, wing, hall, room, created_at, source_file)` — `hall` inserted
  after `wing`

- [ ] **Step 1: Write the failing test**

In `tests/test_extract.py`:

```python
def test_hall_is_read_from_metadata(fake_palace):
    drawers, _ = read_drawers(snapshot_palace(fake_palace))
    assert {d.hall for d in drawers} == {"technical", "memory"}


def test_missing_hall_falls_back_to_unfiled(tmp_path):
    import chromadb

    palace = tmp_path / "nohall"
    palace.mkdir()
    collection = chromadb.PersistentClient(path=str(palace)).get_or_create_collection(
        name="mempalace_drawers"
    )
    collection.add(
        ids=["only"], documents=["text"], embeddings=[[1.0, 0.0]],
        metadatas=[{"wing": "solo"}],
    )
    drawers, _ = read_drawers(snapshot_palace(palace))
    assert drawers[0].hall == "unfiled"
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `.venv/bin/python -m pytest tests/test_extract.py -k hall -v`
Expected: FAIL — `Drawer` has no attribute `hall`

- [ ] **Step 3: Implement**

In `src/locium/models.py`, add the field to `Drawer` between `wing` and `room`:

```python
@dataclass(frozen=True)
class Drawer:
    id: str
    text: str
    wing: str
    hall: str
    room: str
    created_at: str
    source_file: str
```

In `src/locium/extract.py`, populate it in the `Drawer(...)` construction:

```python
hall=(metadatas[i] or {}).get("hall") or "unfiled",
```

Use `or "unfiled"` rather than `.get("hall", "unfiled")` — the real store writes an
explicit `None` for some drawers, which a default argument would not catch.

In `tests/conftest.py`, add `"hall"` to the `fake_palace` fixture metadata, alternating
`"technical"` and `"memory"` so grouping is exercised.

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_extract.py tests/test_models.py -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/locium/models.py src/locium/extract.py tests/conftest.py tests/test_extract.py tests/test_models.py
git commit -m "Carry the hall through from palace metadata"
```

---

### Task R2: Building footprint

**Files:**
- Create: `src/locium/footprint.py`
- Delete: `src/locium/treemap.py`, `tests/test_treemap.py`
- Test: `tests/test_footprint.py`

**Interfaces:**
- Consumes: `locium.models.Rect`
- Produces:
  - `BLOCKS: list[Rect]` — the five masses of the building, index 0 is the core
  - `subdivide(weights: dict[str, int], rect: Rect, pad: float) -> dict[str, Rect]`
  - `building_footprint(counts: dict[str, int]) -> dict[str, Rect]`
  - `EmptyBlock(Exception)`

- [ ] **Step 1: Write the failing test**

Create `tests/test_footprint.py`:

```python
import pytest

from locium.footprint import BLOCKS, building_footprint, subdivide
from locium.models import Rect


def _overlaps(a: Rect, b: Rect) -> bool:
    return not (
        a.x + a.w <= b.x + 1e-6 or b.x + b.w <= a.x + 1e-6
        or a.y + a.h <= b.y + 1e-6 or b.y + b.h <= a.y + 1e-6
    )


def _counts(n: int) -> dict[str, int]:
    """One dominant wing plus a long tail — the real palace's shape."""
    sizes = [5785, 199, 197, 131, 90, 52, 34, 31, 12, 9, 8, 7, 6, 4, 4, 2, 1]
    return {f"w{i}": sizes[i % len(sizes)] for i in range(n)}


def test_wings_never_overlap():
    rects = list(building_footprint(_counts(17)).values())
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            assert not _overlaps(rects[i], rects[j])


def test_every_block_is_occupied():
    """An empty block silently deletes its step from the outline."""
    placed = building_footprint(_counts(17)).values()
    for block in BLOCKS:
        assert any(
            r.x >= block.x - 1 and r.y >= block.y - 1
            and r.x + r.w <= block.x + block.w + 1
            and r.y + r.h <= block.y + block.h + 1
            for r in placed
        ), f"block at {block.x},{block.y} got no wings"


def test_largest_wing_takes_the_core_alone():
    rects = building_footprint(_counts(17))
    biggest = max(_counts(17), key=lambda k: _counts(17)[k])
    assert rects[biggest] == BLOCKS[0]


def test_outline_is_not_a_rectangle():
    """Blocks sit at different depths, so the silhouette steps."""
    rects = list(building_footprint(_counts(17)).values())
    tops = {round(r.y) for r in rects}
    lefts = {round(r.x) for r in rects}
    assert len(tops) > 3
    assert len(lefts) > 3


def test_fewer_wings_than_blocks_still_places_them_all():
    rects = building_footprint(_counts(3))
    assert len(rects) == 3


def test_single_wing_takes_the_core():
    rects = building_footprint({"only": 42})
    assert rects["only"] == BLOCKS[0]


def test_no_wings_is_empty():
    assert building_footprint({}) == {}


def test_subdivide_is_deterministic():
    weights = {"a": 100, "b": 30, "c": 4}
    rect = Rect(0.0, 0.0, 200.0, 120.0)
    assert subdivide(weights, rect, 4.0) == subdivide(weights, rect, 4.0)


def test_subdivide_compresses_a_dominant_member():
    """log sizing keeps a tiny room legible beside a huge one."""
    rects = subdivide({"huge": 5000, "tiny": 1}, Rect(0.0, 0.0, 400.0, 400.0), 0.0)
    ratio = rects["huge"].area / rects["tiny"].area
    assert ratio < 12.0


def test_subdivide_stays_inside_its_rect():
    rect = Rect(10.0, 20.0, 300.0, 200.0)
    for r in subdivide({"a": 9, "b": 3, "c": 1}, rect, 5.0).values():
        assert r.x >= rect.x + 5.0 - 1e-6
        assert r.y >= rect.y + 5.0 - 1e-6
        assert r.x + r.w <= rect.x + rect.w - 5.0 + 1e-6
        assert r.y + r.h <= rect.y + rect.h - 5.0 + 1e-6
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `.venv/bin/python -m pytest tests/test_footprint.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'locium.footprint'`

- [ ] **Step 3: Implement**

Create `src/locium/footprint.py`:

```python
"""The building's plan: one connected mass of blocks, not a filled rectangle.

Each block shares a full edge with the core, so the building reads as a single
structure. The blocks sit at different depths and do not line up, so the outline
steps in and out rather than squaring off.

Sizes are log-compressed throughout. One wing holds 88% of the real store, and
area proportional to count makes most chambers smaller than their own labels.
A floorplan does not size rooms by how full they are; density shows that.
"""

import math

import squarify

from .models import Rect


class EmptyBlock(Exception):
    """Raised when a block would be left with no wings, erasing its step."""


# index 0 is the core; the rest each share a full edge with it
BLOCKS = [
    Rect(168.0, 296.0, 496.0, 424.0),   # core
    Rect(262.0, 128.0, 402.0, 168.0),   # north, set back from the core's face
    Rect(664.0, 366.0, 236.0, 306.0),   # east, projecting
    Rect(238.0, 720.0, 372.0, 158.0),   # south, offset left
    Rect(76.0, 392.0, 92.0, 236.0),     # west annexe, shallow
]


def subdivide(weights: dict[str, int], rect: Rect, pad: float) -> dict[str, Rect]:
    """Squarify members into a rect, log-compressed and in a stable order."""
    live = [n for n in sorted(weights, key=lambda k: (-weights[k], k)) if weights[k] > 0]
    if not live:
        return {}

    vals = [math.log(weights[n] + 1.0) + 0.9 for n in live]
    aw = max(rect.w - 2 * pad, 1.0)
    ah = max(rect.h - 2 * pad, 1.0)
    raw = squarify.squarify(
        squarify.normalize_sizes(vals, aw, ah), rect.x + pad, rect.y + pad, aw, ah
    )
    return {n: Rect(r["x"], r["y"], r["dx"], r["dy"]) for n, r in zip(live, raw)}


def building_footprint(counts: dict[str, int]) -> dict[str, Rect]:
    """Place every wing into the building.

    The largest wing takes the core on its own. The rest are seeded one per
    remaining block — largest block first — then the remainder goes to whichever
    block is furthest below its share of floor area.
    """
    if not counts:
        return {}

    names = sorted(counts, key=lambda k: (-counts[k], k))
    placed = {names[0]: BLOCKS[0]}
    rest = names[1:]
    if not rest:
        return placed

    ranges = BLOCKS[1:]
    areas = [b.area for b in ranges]
    weights = {n: math.log(counts[n] + 1.0) + 1.0 for n in rest}

    groups: list[list[str]] = [[] for _ in ranges]
    cursor = 0
    for k in sorted(range(len(ranges)), key=lambda j: -areas[j]):
        if cursor < len(rest):
            groups[k].append(rest[cursor])
            cursor += 1

    load = [sum(weights[n] for n in g) for g in groups]
    while cursor < len(rest):
        k = min(range(len(ranges)), key=lambda j: load[j] / areas[j])
        groups[k].append(rest[cursor])
        load[k] += weights[rest[cursor]]
        cursor += 1

    for block, group in zip(ranges, groups):
        if not group:
            continue
        placed.update(subdivide({n: counts[n] for n in group}, block, 0.0))
    return placed
```

Then delete the retired module:

```bash
git rm src/locium/treemap.py tests/test_treemap.py
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_footprint.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add src/locium/footprint.py tests/test_footprint.py
git commit -m "Replace the treemap with a connected building footprint"
```

---

### Task R3: Chamber packing

**Files:**
- Create: `src/locium/packing.py`
- Delete: `src/locium/layout.py`, `tests/test_layout.py`, `src/locium/clusters.py`, `tests/test_clusters.py`
- Modify: `src/locium/stability.py`
- Test: `tests/test_packing.py`, `tests/test_stability.py`

**Interfaces:**
- Consumes: `locium.models.Rect`
- Produces: `pack_chamber(n: int, rect: Rect, seed: int, placed: list[tuple[float, float]] | None = None) -> list[tuple[float, float]]`

Dots are scattered rather than laid on a lattice: candidates are sampled and the one
furthest from what is already placed wins. A grid reads as typeset; memories should read
as packed.

`placed` carries the positions a chamber already assigned on a previous build. Those
coordinates are returned unchanged and new dots are packed around them — this is how
coordinate stability survives without UMAP.

- [ ] **Step 1: Write the failing test**

Create `tests/test_packing.py`:

```python
import pytest

from locium.models import Rect
from locium.packing import pack_chamber

RECT = Rect(100.0, 200.0, 60.0, 40.0)


def test_returns_one_point_per_drawer():
    assert len(pack_chamber(25, RECT, seed=42)) == 25


def test_points_stay_inside_the_chamber():
    for x, y in pack_chamber(40, RECT, seed=42):
        assert RECT.x <= x <= RECT.x + RECT.w
        assert RECT.y <= y <= RECT.y + RECT.h


def test_packing_is_deterministic():
    assert pack_chamber(30, RECT, seed=42) == pack_chamber(30, RECT, seed=42)


def test_a_different_seed_packs_differently():
    assert pack_chamber(30, RECT, seed=42) != pack_chamber(30, RECT, seed=7)


def test_zero_drawers_is_empty():
    assert pack_chamber(0, RECT, seed=42) == []


def test_existing_points_are_returned_unchanged():
    first = pack_chamber(10, RECT, seed=42)
    grown = pack_chamber(14, RECT, seed=42, placed=first)
    assert grown[:10] == first
    assert len(grown) == 14


def test_growth_does_not_stack_new_points_on_old():
    first = pack_chamber(8, RECT, seed=42)
    grown = pack_chamber(12, RECT, seed=42, placed=first)
    for nx, ny in grown[8:]:
        assert all(abs(nx - ox) > 1e-9 or abs(ny - oy) > 1e-9 for ox, oy in first)


def test_a_tiny_chamber_still_packs():
    assert len(pack_chamber(5, Rect(0.0, 0.0, 2.0, 2.0), seed=42)) == 5
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `.venv/bin/python -m pytest tests/test_packing.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'locium.packing'`

- [ ] **Step 3: Implement**

Create `src/locium/packing.py`:

```python
"""Scatter a chamber's drawers inside its walls.

Candidates are sampled and the one furthest from the nearest already-placed dot
wins, which reads as packed rather than typeset. Only a recent window of placed
points is consulted, so packing a full chamber stays linear rather than
quadratic.

Passing ``placed`` returns those coordinates untouched and packs new dots around
them: this is where coordinate stability lives now that nothing is projected.
"""

import random

from .models import Rect

CANDIDATES = 9
NEIGHBOUR_WINDOW = 42
INSET = 2.6


def pack_chamber(
    n: int,
    rect: Rect,
    seed: int,
    placed: list[tuple[float, float]] | None = None,
) -> list[tuple[float, float]]:
    if n <= 0:
        return []

    out = list(placed or [])[:n]
    if len(out) >= n:
        return out

    rng = random.Random(seed)
    # Burn the draws that produced the existing points so growth is stable.
    for _ in range(len(out) * CANDIDATES * 2):
        rng.random()

    inset = min(INSET, rect.w / 3.0, rect.h / 3.0)
    ax, ay = rect.x + inset, rect.y + inset
    aw = max(rect.w - 2 * inset, 1e-6)
    ah = max(rect.h - 2 * inset, 1e-6)

    while len(out) < n:
        best, best_d = None, -1.0
        for _ in range(CANDIDATES):
            px, py = ax + rng.random() * aw, ay + rng.random() * ah
            if not out:
                best = (px, py)
                break
            window = out[-NEIGHBOUR_WINDOW:]
            d = min((px - qx) ** 2 + (py - qy) ** 2 for qx, qy in window)
            if d > best_d:
                best, best_d = (px, py), d
        out.append(best)
    return out
```

Then delete the retired modules:

```bash
git rm src/locium/layout.py tests/test_layout.py src/locium/clusters.py tests/test_clusters.py
```

`src/locium/stability.py` keeps `merge_coords` — the build still drops drawers that have
left the palace and preserves those that remain. Delete `place_into_existing` and its
tests; new drawers are now positioned by `pack_chamber`'s `placed` argument, not by
embedding proximity. Remove the now-unused numpy import if nothing else needs it.

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_packing.py tests/test_stability.py -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add -A src/locium tests
git commit -m "Pack drawers into chambers and retire the projection modules"
```

---

### Task R4: Build the floorplan artifact

**Files:**
- Modify: `src/locium/build.py`
- Test: `tests/test_build.py`

**Interfaces:**
- Produces: unchanged signature `build_index(palace, index_path, refit=False, tuning=TUNING) -> dict`

`meta.json` gains `halls` and `chambers`, and `drawers` gain `hall`:

```
wings[]:    {name, rect:[x,y,w,h], count}
halls[]:    {name, wing, rect, count}
chambers[]: {name, wing, hall, rect, count, capped}
drawers[]:  {id, wing, hall, room, date, x, y, preview}
arcs[]:     [src, dst, distance]
```

- [ ] **Step 1: Write the failing test**

Add to `tests/test_build.py`:

```python
def test_meta_carries_the_building(fake_palace, tmp_path):
    meta = build_index(fake_palace, tmp_path / "idx")
    assert meta["wings"] and meta["halls"] and meta["chambers"]
    for chamber in meta["chambers"]:
        assert {"name", "wing", "hall", "rect", "count", "capped"} <= set(chamber)


def test_every_drawer_sits_inside_its_chamber(fake_palace, tmp_path):
    meta = build_index(fake_palace, tmp_path / "idx")
    boxes = {
        (c["wing"], c["hall"], c["name"]): c["rect"] for c in meta["chambers"]
    }
    for d in meta["drawers"]:
        x, y, w, h = boxes[(d["wing"], d["hall"], d["room"])]
        assert x <= d["x"] <= x + w
        assert y <= d["y"] <= y + h


def test_chamber_over_the_cap_is_marked(fake_palace, tmp_path):
    meta = build_index(fake_palace, tmp_path / "idx", tuning=Tuning(dot_cap=2))
    assert any(c["capped"] for c in meta["chambers"])


def test_capped_chamber_still_reports_its_true_count(fake_palace, tmp_path):
    meta = build_index(fake_palace, tmp_path / "idx", tuning=Tuning(dot_cap=2))
    capped = [c for c in meta["chambers"] if c["capped"]]
    drawn = {c["name"]: 0 for c in capped}
    for d in meta["drawers"]:
        if d["room"] in drawn:
            drawn[d["room"]] += 1
    for c in capped:
        assert c["count"] > drawn[c["name"]]
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `.venv/bin/python -m pytest tests/test_build.py -k "building or chamber" -v`
Expected: FAIL — `meta` has no `halls`

- [ ] **Step 3: Implement**

Add `dot_cap: int = 300` to `Tuning` in `src/locium/config.py`.

Rewrite the layout section of `build_index` to walk wing → hall → chamber:

```python
from .footprint import building_footprint, subdivide
from .packing import pack_chamber

PAD_HALL, PAD_CHAMBER = 8.5, 5.0

by_wing: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
for position, drawer in enumerate(drawers):
    by_wing[drawer.wing][drawer.hall].append(position)

wing_counts = {w: sum(len(v) for v in halls.values()) for w, halls in by_wing.items()}
wing_rects = _resolve_wing_rects(wing_counts, previous_rects, refit)

wings_meta, halls_meta, chambers_meta = [], [], []
coords: dict[str, list[float]] = {}

for wing, wing_rect in wing_rects.items():
    wings_meta.append(
        {"name": wing, "rect": _rect_list(wing_rect), "count": wing_counts[wing]}
    )
    hall_counts = {h: len(rows) for h, rows in by_wing[wing].items()}
    for hall, hall_rect in subdivide(hall_counts, wing_rect, PAD_HALL).items():
        halls_meta.append({
            "name": hall, "wing": wing,
            "rect": _rect_list(hall_rect), "count": hall_counts[hall],
        })
        rows_by_room = defaultdict(list)
        for row in by_wing[wing][hall]:
            rows_by_room[drawers[row].room].append(row)

        room_counts = {r: len(v) for r, v in rows_by_room.items()}
        for room, chamber in subdivide(room_counts, hall_rect, PAD_CHAMBER).items():
            rows = rows_by_room[room]
            shown = rows[: tuning.dot_cap]
            chambers_meta.append({
                "name": room, "wing": wing, "hall": hall,
                "rect": _rect_list(chamber), "count": len(rows),
                "capped": len(rows) > tuning.dot_cap,
            })
            kept = [previous_coords[drawers[r].id] for r in shown
                    if drawers[r].id in previous_coords] if not refit else []
            points = pack_chamber(
                len(shown), chamber, tuning.seed,
                placed=[tuple(p) for p in kept] or None,
            )
            for row, (px, py) in zip(shown, points):
                coords[drawers[row].id] = [round(px, 1), round(py, 1)]
```

`shown = rows[: tuning.dot_cap]` takes a stable prefix rather than a random sample, so a
capped chamber shows the same drawers on every build.

`_resolve_wing_rects` mirrors the old `_resolve_rects`: on an ordinary rebuild, wings
already in the index keep their persisted rectangle and only new wings are placed by
`building_footprint`; `--refit` recomputes everything.

Only drawers with coordinates go into `meta["drawers"]` — those beyond a chamber's cap
are counted but not drawn.

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_build.py -v`
Expected: all pass

- [ ] **Step 5: Verify against the real palace**

```bash
.venv/bin/python -m locium.cli build --index /tmp/locium-real --refit
```
Expected: ~17 wings, ~60 halls, ~138 chambers, a warning naming the unreadable drawers.

- [ ] **Step 6: Commit**

```bash
git add src/locium/build.py src/locium/config.py tests/test_build.py
git commit -m "Build the floorplan artifact"
```

---

### Task R5: Serve drawer text

**Files:**
- Modify: `src/locium/build.py`, `src/locium/index.py`, `src/locium/server.py`
- Test: `tests/test_server.py`, `tests/test_index.py`

**Interfaces:**
- Produces: `GET /api/drawer/{drawer_id}` → `{"id", "wing", "hall", "room", "date", "source_file", "text"}`

Full text is too large for `meta.json` — roughly 6,500 drawers at 1–2 KB each. The build
writes a sidecar `texts.json` alongside the index (`{drawer_id: text}`) and the server
reads a single entry from it per request.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_server.py`:

```python
def test_drawer_endpoint_returns_full_text(client):
    meta = client.get("/api/index").json()
    drawer_id = meta["drawers"][0]["id"]
    body = client.get(f"/api/drawer/{drawer_id}").json()
    assert body["id"] == drawer_id
    assert {"wing", "hall", "room", "date", "text"} <= set(body)
    assert len(body["text"]) >= len(meta["drawers"][0]["preview"])


def test_unknown_drawer_is_404(client):
    assert client.get("/api/drawer/does-not-exist").status_code == 404
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `.venv/bin/python -m pytest tests/test_server.py -k drawer -v`
Expected: FAIL — 404 on a valid id

- [ ] **Step 3: Implement**

In `src/locium/index.py` add `TEXTS_NAME = "texts.json"`, write it inside the same
staging directory as the other two files so the artifact stays committed as one unit, and
add `read_text(path, drawer_id) -> str | None`.

In `src/locium/build.py`, collect `{drawer.id: drawer.text}` for drawers that got
coordinates and pass it to `write_index`.

In `src/locium/server.py`:

```python
@app.get("/api/drawer/{drawer_id}")
def get_drawer(drawer_id: str) -> dict:
    meta = read_meta(index_path)
    row = next((d for d in meta["drawers"] if d["id"] == drawer_id), None)
    if row is None:
        raise HTTPException(status_code=404, detail=f"no drawer {drawer_id}")
    return {
        "id": drawer_id,
        "wing": row["wing"], "hall": row["hall"], "room": row["room"],
        "date": row["date"], "source_file": row.get("source_file", ""),
        "text": read_text(index_path, drawer_id) or row["preview"],
    }
```

- [ ] **Step 4: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_server.py tests/test_index.py -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/locium/build.py src/locium/index.py src/locium/server.py tests/test_server.py tests/test_index.py
git commit -m "Serve a drawer's full text on demand"
```

---

### Task R6: Blueprint renderer

**Files:**
- Rewrite: `src/locium/static/render.js`, `src/locium/static/style.css`, `src/locium/static/index.html`
- Keep: `src/locium/static/knn.js`

**Interfaces:**
- Produces `Renderer(canvas)` with `.setData(meta)`, `.draw()`, `.fit()`, `.home()`,
  `.zoomBy(factor, px, py)`, `.panBy(dx, dy)`, `.hitTest(px, py)`, `.focusOn(x, y)`,
  `.setTheme(name)`, and settable `.dimmed`, `.highlighted`, `.selected`
- Themes: `light` (ink on paper) and `dark` (light linework on a dark ground)

Drawing rules, in order:

1. Ground fill, then a grain overlay — per-pixel noise at alpha 8 (light) / 13 (dark).
2. Sheet border: two nested strokes at 18.5 and 23.5 px inset.
3. **Poché walls** — each rect drawn as a filled band between an outer and inner edge
   using `evenodd`, not a stroke. Weights: wings 3.0, halls 1.5; chambers get a 0.5 px
   hairline.
4. Dots: radius `clamp(1.35 * scale, 0.8, 4.2)`. **The radius must be capped** — an
   uncapped radius turns a zoomed-in chamber into overlapping blobs.
5. Labels last, in importance order: wings, then halls, then chambers.

Two rules the prototype proved necessary:

**Label size scales with zoom.** `size * clamp(scale / baseScale, 1, 3.4)`. Sizes in
screen pixels while rects are in world units means labels look microscopic the moment you
zoom in.

**Labels never overlap.** Keep a list of the boxes drawn this frame; before drawing, nudge
a colliding label down by its own height up to three times, then drop it rather than
paint over a neighbour. Each label gets a knockout rectangle in the ground colour so walls
never cut through it.

Level of detail: halls appear above `0.75 × base`, chambers above `0.95 × base`.

Value carries recency, and it inverts with the theme — light: recent is dark ink on pale;
dark: recent is bright on dark. Changing theme is a semantic change, not only a cosmetic
one.

- [ ] **Step 1: Write the module**

Port `render.js` from the approved prototype at
`.superpowers/sdd/prototype-castle.py` (the generator that produced it is committed there
for reference), replacing its embedded data with `setData(meta)` and its module-level
globals with instance fields.

- [ ] **Step 2: Verify by hand**

```bash
.venv/bin/python -m locium.cli build --index /tmp/locium-real --refit
.venv/bin/python -m locium.cli serve --index /tmp/locium-real
```

Confirm: one connected building with a stepped outline; all five blocks occupied; poché
walls read as masonry; labels legible at fit and at 6× zoom; both themes; no label
overlaps.

- [ ] **Step 3: Commit**

```bash
git add src/locium/static/
git commit -m "Draw the palace as an architect's floorplan"
```

---

### Task R7: Viewer interaction

**Files:**
- Rewrite: `src/locium/static/app.js`

**Interfaces:**
- Produces `window.__locium = { renderer, state, select, search, setTheme, confirmTunnel }`

The Task 17 draft is preserved at `.superpowers/sdd/task-17-app.js.draft`; its selection,
trail and tunnel logic carries over largely unchanged. What changes is that reading fetches
from `/api/drawer/{id}` instead of showing the preview, and search, zoom and theme are
added.

- **Select** — fetch the drawer, fill the reading panel, highlight the ten nearest
  neighbours via client-side k-NN, append to the trail.
- **Search** — embed the query server-side, score client-side, and additionally match
  literal substrings against wing/hall/room names. Matches go accent-coloured, everything
  else dims to ~10%. **The map never moves.**
- **Zoom/pan** — wheel toward the cursor, drag to pan, buttons for in/out/fit. A drag must
  not select.
- **Theme** — toggle, persisted to `localStorage`, defaulting to `prefers-color-scheme`.
- **Tunnels** — unchanged, including the warning when a room pair already has one.

- [ ] **Step 1: Write the module**
- [ ] **Step 2: Verify each behaviour by hand against the real index**
- [ ] **Step 3: Commit**

```bash
git add src/locium/static/app.js
git commit -m "Add selection, search, zoom, reading and theme to the viewer"
```

---

### Task R8: End-to-end tests and README

**Files:**
- Modify: `tests/e2e/test_ui.spec.js`, `tests/e2e/fixture_server.py`, `README.md`

Update the fixture to build a palace with `wing`, `hall`, `room` and `filed_at` metadata.
Cover: the building renders with every block occupied; clicking a dot opens the panel with
text from `/api/drawer/{id}`; search dims non-matches **without moving any dot**; zoom
changes label size; the theme toggle switches and persists; the trail grows.

- [ ] **Step 1: Update the fixture and specs**
- [ ] **Step 2: Run `npx playwright test`**
- [ ] **Step 3: Rewrite the README's "How it works" for the floorplan model**
- [ ] **Step 4: Commit**

---

## Verification

- `.venv/bin/python -m pytest -q` — all pass, zero warnings
- `npx playwright test` — all pass
- `locium build && locium serve` against the real palace renders the building
- `~/.mempalace/palace` modification times unchanged by a build
