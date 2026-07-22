# Locium — Design

**Date:** 2026-07-22
**Status:** Approved design, pre-implementation

## What it is

Locium is a visual explorer for an agent's memory store. It renders every memory as a
point on a stable two-dimensional map, so a human can see the shape of what has been
remembered, wander the connections between memories, and spot where the memory is
unhealthy.

The first (and only) backend in v1 is MemPalace, a ChromaDB-backed store at
`~/.mempalace/palace`.

### Name

From Latin *locus* → *loci*, the classical **method of loci** — the memory-palace
technique of placing memories in imagined locations. The `-ium` suffix places it with
*museum, lyceum, atrium*: a place where something is housed and viewed. Locium is the
place where the loci are seen.

Pronounced **LOH-see-um**, rhyming with *lyceum*.

Locium is a standalone brand, not a MemPalace sub-feature. The name must work with no
MemPalace context, because the tool may later point at other memory stores. This does
**not** mean v1 abstracts over backends — see Non-goals.

## Goals

1. **Navigation and recall** (primary) — start from a query or a point on the map and
   wander outward through semantic connections, the way an agent explores the palace
   during work, but with the intermediate state made visible and durable.
2. **Inspection** (secondary) — see memory health as geometry: fragmented wings, stale
   regions, orphaned noise, oversized blobs.
3. **Curation** (partly in v1) — promote discovered connections into stored tunnels.
   Retagging, pruning and any editing of drawer content remain phase 2.

## Context: the state of the palace

Measured 2026-07-22:

| Metric | Value |
|---|---|
| Drawers | 5,101 |
| Wings | 16 |
| Rooms | 13 |
| `technical` room | 3,277 drawers (64% of the palace) |
| Derived graph edges | 493 |
| **Explicit tunnels** | **0** |
| Knowledge-graph triples | 14 |
| Embedding model | `all-MiniLM-L6-v2` (Chroma default, ONNX), 384 dims |

Two facts drive the design:

- **There is effectively no stored link layer.** `mempalace_list_tunnels()` returns `[]`.
  The 493 "edges" are a co-occurrence artifact of room labels shared across wings, not
  recorded relationships between memories. The KG has 14 triples. Rendering only what is
  stored would draw the taxonomy, not the memories.
- **The real connective tissue is embedding proximity** — 5,101 vectors, dense and
  meaningful, computed at query time rather than stored. Locium is partly the thing that
  makes this graph visible for the first time.

### Safety constraint

From the palace's own diary, `wing_kosio/diary`, 2026-05-02:

> direct chromadb writes from python while MCP server holds open client = potential
> corruption. avoid future direct manipulation; use MCP tools only.

That entry followed the loss of a palace to a 0-byte `link_lists.bin`, where the backup
was found to carry the same corruption. ChromaDB's `PersistentClient` makes no
multi-process guarantees, and the MCP server holds that client open all day.

**Locium must never be a second process on the live Chroma store.** This is a hard
requirement, not a preference.

The constraint is specifically about **Chroma**. Not all palace state lives there —
explicit tunnels are stored as plain JSON at `~/.mempalace/tunnels.json`
(`palace_graph.py:291`), written under an flock with an atomic tmp-file replace. Writing
a tunnel never opens Chroma and never touches the hnsw index files. See *Writing
tunnels*.

The MCP tools cannot substitute: `mempalace_search` returns text, wing, room and a
distance score, but never the raw embedding vector. Projection needs vectors. Direct read
access is therefore unavoidable — so it happens against a throwaway copy.

## Architecture

Two commands. `build` is the only thing that ever touches the palace; `serve` only ever
reads the artifact. They never run concurrently.

```
locium build     # python: read palace copy → project → write index
locium serve     # python: fastapi serves viewer + index on :7777
```

```
  chroma store                    Locium
  ┌───────────┐                  ┌─────────┐
  │ palace/   │                  │ web ui  │
  └─────┬─────┘                  └────▲────┘
        │ snapshot + read-once        │ serves
        ▼   (offline, throwaway)      │
  ┌──────────────────────────────────┘
  │ ~/.locium/index/
  │   meta.json     coords, metadata, arcs
  │   vectors.bin   int8, N × 384
  └──────────────────────────────────

  MCP server never contended
```

### Build pipeline

1. **Snapshot** — copy `~/.mempalace/palace` to a temp directory. Read the copy, discard
   it. Costs a second or two at current size and makes it structurally impossible for
   Locium to corrupt the palace.
2. **Extract** — ids, document text, embeddings, metadata (`wing`, `room`, `source_file`,
   `created_at`).
3. **Layout** — treemap over wings, UMAP within each wing. See below.
4. **Arcs** — per-drawer k-NN; keep hits whose neighbour is in a *different* wing and
   whose distance is below threshold. Capped per drawer and globally.

   Note there are **two distinct neighbour computations** in this design, and they are
   not interchangeable:

   - **Precomputed `arcs[]`** (this step) — cross-wing only, thresholded and capped,
     baked into the artifact. Powers the always-available overview of cross-wing
     structure and supplies the candidate set for tunnel confirmation. Global, static.
   - **Click-time k-NN** (client-side, from `vectors.bin`) — top 10 neighbours of the
     selected drawer, *including same-wing ones*, computed fresh on each selection.
     Local, dynamic, never stored.

   A drawer's click-time neighbours are therefore usually a superset of its arcs.
5. **Cluster labels** — HDBSCAN over each wing's 2D coordinates, then TF-IDF over each
   cluster's text for its 2–3 most distinguishing terms.
6. **Quantise** — float32 vectors → int8 (~7.8 MB → ~2 MB at current size).
7. **Write** — `meta.json` + `vectors.bin`.

### Layout: semantic within structure

**Macro (structural, stable).** A squarified treemap allocates each wing a rectangle,
area proportional to drawer count, wings in alphabetical order so the partition does not
reshuffle as counts drift. This is the floorplan and it stays put.

**Micro (semantic, discovered).** UMAP runs *per wing*, over only that wing's drawers,
then the result is scaled and translated into that wing's rectangle. Clusters emerge from
content rather than from declared labels.

Per-wing fitting is also more robust than a single global fit: `technical` holds 64% of
the palace, and that mass would swamp the manifold, squashing everything else into the
margins.

**Cross-wing (arcs).** The signal the hybrid layout would otherwise discard returns as
lines: when two drawers in different wings are semantically adjacent, an arc connects
them. Drawn on demand, never always-on.

### Stable loci

`meta.json` persists `drawer_id → (x, y)` and `wing → rect`. On rebuild:

- existing drawers **keep their exact coordinates** — never moved
- new drawers are **placed into the existing space**, not re-fitted
- re-fitting is opt-in via `locium build --refit`, which warns that the memorised map
  will change

A locus that moves is not a locus. This is the requirement the name encodes and the one
the rest of the design bends around.

### Artifact

```
~/.locium/index/
  meta.json     drawers[]: {id, wing, room, date, x, y, preview, cluster}
                wings[]:   {name, rect, count}
                clusters[]:{wing, label, centroid}
                arcs[]:    [src_idx, dst_idx, distance]
                built_at, palace_mtime, drawer_count, seed
  vectors.bin   int8, N × 384
```

Metadata as JSON is ~2 MB at current size and stays readable and debuggable; only the
vectors need binary. Client-side k-NN over 5k int8 vectors takes a few milliseconds, so
revealing a drawer's neighbours needs no server round-trip.

### Stack

- **Build:** Python — forced, since reading Chroma and running UMAP are Python and
  mempalace is already installed as a Python package.
- **Serve:** FastAPI. `GET /index`, `POST /search` (embeds the query), `POST /rebuild`,
  `GET /tunnels`, `POST /tunnel`, `DELETE /tunnel/{id}`.
- **Frontend:** vanilla JS on **Canvas 2D**. SVG gives one DOM node per drawer — fine at
  500, janky at 5k, dead at 50k. The palace grew from 839 to 5,101 drawers in roughly two
  months, so designing for 50k+ is next year, not premature.
- **Query embedding:** the same cached ONNX `all-MiniLM-L6-v2` Chroma already uses
  (`~/.cache/chroma/onnx_models/`). Same model, same vector space, no torch, no download.

## Interaction

### On load

The whole palace at rest: 16 wing rectangles, 5,101 dots.

- **Position** = wing (macro) + meaning (micro)
- **Colour** = room
- **Opacity** = recency, so stale regions visibly fade

### Zoom levels

| Level | Shows |
|---|---|
| Far | Wing rectangles, names, counts, density shading. Dots merge into mass. |
| Mid | Individual dots, cluster labels, colour by room. The working altitude. |
| Near | Dot labels, hover preview (~150 chars, wing/room/date). |

Cluster labels are what turn the map from decoration into a document — a field of
unlabelled dots says nothing.

### Click

Selecting a drawer does three things at once:

1. **Reading pane** — full text, `wing/room`, date, `source_file`, similarity score if
   arrived at via search.
2. **Neighbours light up** — client-side k-NN over the int8 vectors, top 10, arcs drawn
   to each. Cross-wing arcs are the interesting ones.
3. **Trail appends** — the selection joins a breadcrumb strip.

Clicking a highlighted neighbour repeats the cycle. This is the wander, happening on top
of a persistent map rather than a blank canvas, so orientation is never lost.

```
trail:  "chromadb corruption" › #41 palace-rebuild › #883 link_lists › ⌫
```

The trail is re-walkable, and it feeds tunnel creation directly: a path walked twice is
evidence that two areas relate, and any hop in the trail can be confirmed as a tunnel
without leaving the map. See *Writing tunnels*.

### Search: highlight, never reflow

A typed query is embedded server-side, scored against all vectors client-side; matches
light up **in place** and everything else dims. A ranked list appears alongside; clicking
a result flies the camera to its dot.

The map never rearranges. After a few sessions the user knows where a topic lives
spatially, and search must confirm that geography rather than scramble it. This is the
decision that makes a global map beat an expanding-trail canvas for recall: a query
paints onto terrain already known.

### Inspection layer

View toggles, not a separate mode:

- **Colour by age** — active regions vs. stale ones
- **Density** — makes the 3,277-drawer `technical` mass undeniable
- **Orphans** — drawers whose nearest neighbour exceeds a distance threshold; noise
  candidates, typically raw JSONL blobs worth pruning
- **Slivers** — wing fragmentation, e.g. `niamavreme` / `wing_niamavreme` /
  `niamavreme-infra` as three separate rectangles

## Writing tunnels

v1 can record connections, not only display them. This is the one write capability in
scope, and it is deliberately narrow.

### Mechanism

Locium imports `mempalace.palace_graph` and calls `create_tunnel` / `delete_tunnel`
directly. It does **not** speak MCP and does **not** write JSON by hand.

This is safe for a reason specific to tunnels: they persist to
`~/.mempalace/tunnels.json`, not to Chroma. `create_tunnel` serialises its
load → mutate → save cycle under `mine_lock(_TUNNEL_FILE)` and commits via tmp-file +
`os.replace`, so concurrent writers are already handled. Locium calling that function is
byte-for-byte the same operation the MCP server performs, with the same lock. No second
Chroma client is created, and the hnsw index files are never opened.

Calling mempalace's own function rather than reimplementing it is load-bearing — a
hand-rolled writer would not take the lock and would reintroduce lost updates.

### Flow

1. Select an arc on the map (or any two drawers).
2. The label field pre-fills from the cluster labels at both endpoints; edit freely.
3. Confirm. Locium calls `create_tunnel` with both wings, both rooms, both drawer ids and
   the label, all read from `meta.json`.
4. Confirmed tunnels render as a distinct always-on edge style, separate from computed
   arcs — the difference between "the vectors say these are close" and "a human said
   these belong together".

`GET /tunnels` reads the current set on load, so confirmed tunnels survive rebuilds and
appear immediately. Deleting is supported via `delete_tunnel`.

### Known constraint: tunnels collapse per room pair

The canonical tunnel ID is
`sha256(sorted("source_wing/source_room", "target_wing/target_room"))[:16]` —
**drawer ids are not part of the identity** (`palace_graph.py:334`). Consequences:

- Every drawer↔drawer arc between the same two `wing/room` pairs resolves to **one**
  tunnel. A second confirmation updates the existing record's label and drawer ids rather
  than adding a second one.
- Tunnels are **symmetric**: `create_tunnel(A, B)` and `create_tunnel(B, A)` are the same
  record. Arcs are therefore undirected, which matches how they are drawn.

**Decision:** accept this rather than work around it. A tunnel asserts *"these two rooms
relate"*, with the drawer pair recorded as the exemplar that prompted the assertion.
Maintaining a parallel drawer-level store under `~/.locium/` would create a second source
of truth for the same claim.

The UI must make the collapse visible: when the selected arc's room pair already has a
tunnel, show the existing label and state plainly that confirming will update that record
rather than create a new one.

## Non-goals for v1

- **Any write to Chroma.** Drawer creation, editing, retagging and deletion are phase 2.
  Tunnel create/delete is in scope and is *not* a Chroma write — see *Writing tunnels*.
- **Knowledge-graph visualisation.** 14 triples does not earn a view. Revisit at ~500.
- **Other memory backends.** The name must survive repointing; v1 must not abstract for
  it. One backend, no plugin interface.
- **File watching / auto-refresh.** Rebuild is a button and a CLI command.
- **Saved trails, export, sharing, authentication, mobile layout, 3D.** Localhost,
  single user, desktop.

## Failure modes

| Case | Behaviour |
|---|---|
| No index yet | `serve` refuses with `run: locium build`. Never auto-builds — slow and surprising. |
| Stale index | At startup `serve` **stats** the palace directory (filesystem only, never opens Chroma) and compares mtime against `meta.json`. If newer, show a banner and a rebuild button. Using `stat` rather than a drawer count is deliberate: opening the store would reintroduce the second-reader risk. |
| Tiny wings | `wing_mempalace` has 1 drawer, `_hook_stub_no_auto_ingest` has 2, `mempalace-dev` has 4. UMAP requires `n_neighbors < n_samples` and is meaningless below ~10 points. Wings under threshold get a deterministic ring layout inside their rect, no UMAP. Without this the build crashes on the current palace. |
| New wing appears | Existing rects are pinned; new wings are carved from reserved gutter space. When the gutter is exhausted the build refuses and requires `--refit`. Wings have been added repeatedly (`wing_blog`, `mynewlinode`, `wing_infra`), so this path is live, not hypothetical. |
| Wing outgrows its rect | Dots pack denser; no resize until `--refit`. Stability beats tidiness. |
| Drawer deleted from palace | Still displayed (the index is self-contained); disappears at next rebuild. |
| Chroma segfault mid-build | Build reads a throwaway copy, so the crash kills only the build. Error message points at `mempalace repair`. |
| Palace missing or moved | `--palace` flag, `MEMPALACE_PALACE` env var, default `~/.mempalace/palace`. Explicit error, no guessing. |
| Empty palace | Valid empty index, empty-state UI, not a broken canvas. |
| Port in use | `--port`, otherwise the next free port above 7777. |
| **mempalace API drift** | Locium imports `mempalace.palace_graph`. On startup, assert the module exposes `create_tunnel`, `delete_tunnel` and `list_tunnels` with the expected signature, and record the mempalace version in `meta.json`. Fail with a clear message rather than a stack trace mid-write. Verified against mempalace 3.3.3. |
| Tunnel endpoint no longer exists | A tunnel may reference a wing/room/drawer removed since it was created. `GET /tunnels` returns it regardless; the UI renders it as dangling rather than dropping it silently — a dangling tunnel is a finding, not an error. |
| `tunnels.json` malformed | `_load_tunnels` is the sole reader; if it raises, `serve` starts read-only with a banner rather than refusing to boot. The map is still useful without tunnels. |
| Confirming an arc whose room pair is already tunnelled | Show the existing label and state that confirming updates that record. Never silently overwrite. |

## Testing

The build pipeline is a pure function from drawers to an index, so most of the risk is
testable without a browser.

**Layout (pytest, synthetic drawer sets)**

- **Determinism** — same input and seed produce byte-identical coordinates
- **Stability** — build, append drawers, rebuild; existing coordinates must be unchanged.
  This is the requirement the whole design rests on and it gets a test that fails loudly.
- **Degenerate wings** — 0, 1, 2 and 4 drawers do not crash and get deterministic
  fallback positions
- **Treemap invariants** — no overlap, area proportional to count, rects tile the canvas,
  alphabetical order stable under count changes

**Quantisation**

int8 compression could silently degrade neighbour quality while the map still looks
correct. For 100 random drawers, compare top-10 k-NN from int8 against top-10 from
float32 and assert at least 8 overlap.

**API** — FastAPI `TestClient`: `/index` serves valid JSON, `/search` embeds and returns
ranked ids, `/rebuild` shells out correctly.

**Tunnel writes** — against a temporary `HOME` so the real `~/.mempalace/tunnels.json` is
never touched:

- create → `list_tunnels` returns it with both drawer ids and the label
- create the same room pair twice → **one** record, label updated, `created_at` preserved
- create reversed (`B, A` after `A, B`) → still one record, confirming symmetry
- delete → removed; deleting an unknown id is a no-op, not an error
- the write goes through `palace_graph.create_tunnel`, never a hand-rolled JSON write.
  A test asserts Locium holds no direct reference to `_TUNNEL_FILE`, since bypassing the
  lock would reintroduce lost updates.

**Frontend** — Playwright: N dots render, click opens the reading pane, search dims
non-matches, trail appends, zoom switches level of detail.

Because coordinates are deterministic by requirement, screenshot baselines are viable
here — unusual for graph visualisation, where force-directed layouts land differently
every run. One design constraint, two payoffs: stable loci for memory, stable pixels for
CI.

## Tuning defaults

These are decided, not deferred. Implementation starts from these values; any change is
a deliberate revision, not an open question.

| Parameter | Default | Rationale |
|---|---|---|
| Cross-wing arc distance threshold | cosine distance ≤ 0.45 | Observed search results cluster meaningful hits around 0.51–0.68; 0.45 keeps arcs to genuinely tight pairs rather than topical drift. |
| Arcs per drawer | 3 | Enough to show structure, low enough that the overview stays legible. |
| Global arc cap | 20,000 | ~4 per drawer at current size; a hard ceiling on artifact growth. |
| Click-time k-NN | 10 neighbours | Matches the result-set size an agent works with per search step. |
| Wing UMAP threshold | 10 drawers | Below this, UMAP is meaningless; ring layout instead. |
| Treemap gutter | 15% of canvas area | Absorbs roughly a doubling of wing count before `--refit` is forced. |
| `preview` length | 200 chars, sentence-aware | Cut at the last sentence boundary under 200; hard-cut if none. |
| UMAP seed | 42, persisted in `meta.json` | Determinism is a hard requirement; the seed travels with the artifact. |
