"""The local viewer server.

Reads only the index artifact. The one filesystem call it makes against the
real palace is a stat for the staleness banner — counting drawers would mean
opening Chroma, which is exactly what the build pipeline exists to avoid.
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, field_validator

from . import tunnels as tunnel_api
from .extract import palace_mtime
from .index import VECTORS_NAME, index_exists, read_meta

STATIC_DIR = Path(__file__).parent / "static"


class SearchRequest(BaseModel):
    query: str

    @field_validator("query")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("query must not be blank")
        return value


class TunnelRequest(BaseModel):
    source_wing: str
    source_room: str
    target_wing: str
    target_room: str
    label: str = ""
    source_drawer_id: str | None = None
    target_drawer_id: str | None = None


def create_app(index_path: Path, palace: Path) -> FastAPI:
    if not index_exists(index_path):
        raise FileNotFoundError(
            f"no index at {index_path}. Run: locium build"
        )

    tunnel_api.assert_api()
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

    @app.get("/api/tunnels")
    def get_tunnels() -> dict:
        try:
            return {"tunnels": tunnel_api.listing()}
        except Exception as exc:  # a malformed tunnels.json must not kill the map
            return {"tunnels": [], "error": str(exc)}

    @app.post("/api/tunnel")
    def post_tunnel(request: TunnelRequest) -> dict:
        return tunnel_api.create(**request.model_dump())

    @app.delete("/api/tunnel/{tunnel_id}")
    def remove_tunnel(tunnel_id: str) -> dict:
        return tunnel_api.delete(tunnel_id)

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
