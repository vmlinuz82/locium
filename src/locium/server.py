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
    VECTORS_NAME,
    index_exists,
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
        return {
            "id": drawer_id,
            "wing": row["wing"], "hall": row["hall"], "room": row["room"],
            "date": row["date"], "source_file": row.get("source_file", ""),
            "text": read_text(index_path, drawer_id) or row["preview"],
        }

    @app.post("/api/rebuild")
    def rebuild() -> dict:
        from .build import build_index

        try:
            meta = build_index(palace, index_path)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {"drawer_count": meta["drawer_count"]}

    @app.get("/")
    def root() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    for name in ("app.js", "render.js", "knn.js", "style.css"):
        app.get(f"/{name}")(
            lambda name=name: FileResponse(STATIC_DIR / name)
        )

    return app
