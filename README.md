# Locium

A visual explorer for an agent's memory. Locium renders a MemPalace store as
an architect's floorplan — one connected building, wings and halls as blocks,
rooms as chambers, drawers as dots — and records the connections you confirm
between them.

The name comes from *loci* — the classical method of loci, where memories are
placed in imagined locations. Pronounced LOH-see-um.

## Install

    python -m venv .venv
    .venv/bin/pip install -e ".[dev]"

## Use

    locium build          # read the palace, write the index
    locium serve          # open http://127.0.0.1:7777

Options:

    --palace PATH   MemPalace store (default ~/.mempalace/palace,
                    or $MEMPALACE_PALACE)
    --index PATH    index location (default ~/.locium/index)
    --refit         re-pack every chamber; MOVES existing drawers
    --port N        serve on a specific port (default 7777)

## How it works

`build` copies the palace to a temp directory and reads the copy — it never
opens the live ChromaDB store, which the MemPalace MCP server holds open.
Every drawer carries `wing`, `hall` and `room`, the store's own three-level
hierarchy; Locium draws that hierarchy directly as a floorplan: wings and
halls become blocks sharing a full edge with a connected core, rooms become
chambers inside them, and each drawer is a dot scattered inside its chamber.

Block size is `log(count + 1)`, not proportional to how full a room is — one
wing can hold the vast majority of a real store, and sizing by count would
make most chambers smaller than their own labels. Density of dots shows
fullness instead. A dot's value carries recency (older drawers read fainter,
recent ones read stronger), and it inverts with the theme so "recent" always
reads as more present, not just differently coloured.

Position no longer encodes meaning — proximity on screen means "same room",
not "similar content". Clicking a dot fetches that drawer's full text from
the server (`meta.json` only ever carries a 200-character preview) and
highlights its ten nearest neighbours, found by client-side k-NN over
quantised vectors — that's how semantic closeness stays discoverable without
being drawn as distance. Search embeds your query with the same ONNX model
that produced the stored vectors and additionally matches wing/hall/room
names literally; matches go accent-coloured, everything else dims. The map
never moves for a search — no drawer's coordinate changes.

A chamber past `dot_cap` drawers stops adding dots (its true count is still
shown), so one enormous room can't turn the map into an unreadable smear.

## Stable coordinates

A drawer keeps its coordinate as long as its chamber (`wing`, `hall`, `room`)
is unchanged and that point still falls inside the chamber's current
rectangle; wing/hall/chamber geometry itself is recomputed fresh on every
build. New drawers are packed into the space that's left. Only `--refit`
discards everything and repacks from scratch, and it will invalidate the map
you have memorised.

## Tunnels

Confirming a connection calls `mempalace.palace_graph.create_tunnel`, which
writes to `~/.mempalace/tunnels.json` under a file lock. Nothing is ever
written to ChromaDB.

Tunnel identity is the wing/room pair, so one tunnel exists per pair of rooms
and the drawer ids record the exemplar that prompted it.

## Tests

    .venv/bin/python -m pytest      # build pipeline, API, tunnels
    npx playwright test             # browser
