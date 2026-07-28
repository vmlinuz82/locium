"""The local viewer server.

Reads only the index artifact. The one filesystem call it makes against the
real palace is a stat for the staleness banner — counting drawers would mean
opening Chroma, which is exactly what the build pipeline exists to avoid.
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, field_validator

from .extract import palace_mtime
from .index import (
    FAMILY_VECTORS_NAME,
    VECTORS_NAME,
    index_exists,
    load_stitches,
    load_texts,
    read_meta,
    read_text,
    search_texts,
    snippets,
)

STATIC_DIR = Path(__file__).parent / "static"

# Ceiling on literal hits returned for one query. A bare word can occur in
# thousands of drawers, and highlighting thousands of dots says nothing.
TEXT_HIT_LIMIT = 200


class SearchRequest(BaseModel):
    query: str

    @field_validator("query")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value


class SnippetRequest(BaseModel):
    ids: list[str]
    query: str = ""

    @field_validator("ids")
    @classmethod
    def capped(cls, value: list[str]) -> list[str]:
        return value[:TEXT_HIT_LIMIT]


def create_app(index_path: Path, palace: Path) -> FastAPI:
    if not index_exists(index_path):
        raise FileNotFoundError(
            f"no index at {index_path}. Run: locium build"
        )

    app = FastAPI(title="Locium")

    @app.get("/api/index")
    def get_index() -> dict:
        meta = read_meta(index_path)
        stale = palace.exists() and palace_mtime(palace) > meta.get("palace_mtime", 0)
        return meta | {"stale": stale}

    @app.get("/api/vectors")
    def get_vectors() -> Response:
        return Response(
            content=(index_path / VECTORS_NAME).read_bytes(),
            media_type="application/octet-stream",
        )

    @app.get("/api/family-vectors")
    def get_family_vectors() -> Response:
        """Whole-exchange embeddings for the recall-gap verdict.

        Empty body (not 404) when the index predates stitching or has no
        split exchanges -- the client treats both as "no gap to measure".
        """
        file = index_path / FAMILY_VECTORS_NAME
        return Response(
            content=file.read_bytes() if file.exists() else b"",
            media_type="application/octet-stream",
        )

    @app.post("/api/search")
    def search(request: SearchRequest) -> dict:
        from .embed import embed_query

        return {"vector": embed_query(request.query).tolist()}

    @app.post("/api/search-text")
    def search_text(request: SearchRequest) -> dict:
        """Literal substring hits, which the embedding alone cannot find."""
        ids = search_texts(index_path, request.query, TEXT_HIT_LIMIT)
        return {"ids": ids, "truncated": len(ids) >= TEXT_HIT_LIMIT}

    @app.post("/api/snippets")
    def get_snippets(request: SnippetRequest) -> dict:
        """Readable text for a result list, centred on the query where it hits."""
        return {"snippets": snippets(index_path, request.ids, request.query)}

    @app.get("/api/drawer/{drawer_id}")
    def get_drawer(drawer_id: str) -> dict:
        meta = read_meta(index_path)
        row = next((d for d in meta["drawers"] if d["id"] == drawer_id), None)
        if row is None:
            raise HTTPException(status_code=404, detail=f"no drawer {drawer_id}")
        payload = {
            "id": drawer_id,
            "wing": row["wing"], "hall": row["hall"], "room": row["room"],
            "date": row["date"], "source_file": row.get("source_file", ""),
            "text": read_text(index_path, drawer_id) or row["preview"],
        }

        # When the miner split this drawer's exchange across siblings, hand
        # back the reassembled message too. The chunks are exact partitions,
        # so plain concatenation reconstructs the original; a sibling missing
        # from texts.json makes the whole stitch untrustworthy, so none is
        # offered. "text" stays the bare chunk for callers that expect it.
        stitches = load_stitches(index_path)
        family_key = stitches["member"].get(drawer_id)
        if family_key:
            texts = load_texts(index_path)
            sibling_ids = stitches["families"].get(family_key, [])
            parts = [texts.get(i) for i in sibling_ids]
            if sibling_ids and all(part is not None for part in parts):
                # Exchange chunks are exact partitions -- concatenation
                # reconstructs the message. Document slices are not (the
                # paragraph chunker stripped between them), so those rejoin
                # on newlines instead of pretending to be one string.
                is_document = family_key in set(stitches.get("docs", []))
                joiner = "\n" if is_document else ""
                position = sibling_ids.index(drawer_id)
                payload["message"] = joiner.join(parts)
                payload["message_part"] = position + 1
                payload["message_chunks"] = len(sibling_ids)
                # Where THIS drawer's slice sits inside the message, so the
                # reader can mark the part that actually led here. Computed
                # from the parts, not searched for -- sibling chunks can
                # repeat near-identical text.
                payload["message_offset"] = len(joiner.join(parts[:position])) + (
                    len(joiner) if position else 0
                )
                payload["message_span_length"] = len(parts[position])
        return payload

    @app.post("/api/rebuild")
    def rebuild() -> dict:
        from .build import build_index

        try:
            meta = build_index(palace, index_path)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {"drawer_count": meta["drawer_count"]}

    # Two layers against stale statics. no-cache makes browsers revalidate
    # (cheap 304s when unchanged). But copies cached BEFORE that header
    # existed are beyond its reach, so the page also references each asset
    # with a ?v=<mtime> stamp: a changed file is a changed URL, and a stale
    # cache entry simply never gets asked for again. Without both, browsers
    # heuristically kept one file and refetched another, and the app ran a
    # mixed set of versions (observed: app.js fresh, render.js stale,
    # TypeError on a method that "doesn't exist").
    _REVALIDATE = {"Cache-Control": "no-cache"}
    _ASSETS = ("app.js", "render.js", "knn.js", "style.css")

    @app.get("/")
    def root() -> Response:
        html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        for name in _ASSETS:
            stamp = int((STATIC_DIR / name).stat().st_mtime)
            html = html.replace(f"/{name}", f"/{name}?v={stamp}")
        return Response(html, media_type="text/html", headers=_REVALIDATE)

    for name in _ASSETS:
        app.get(f"/{name}")(
            lambda name=name: FileResponse(STATIC_DIR / name, headers=_REVALIDATE)
        )

    return app
