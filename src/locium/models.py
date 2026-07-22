"""Core value types shared across the build pipeline and the server."""

from dataclasses import dataclass

from .config import TUNING


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    w: float
    h: float

    @property
    def area(self) -> float:
        return self.w * self.h


@dataclass(frozen=True)
class Drawer:
    id: str
    text: str
    wing: str
    room: str
    created_at: str
    source_file: str


def make_preview(text: str, limit: int = TUNING.preview_chars) -> str:
    """Collapse whitespace and truncate, preferring a sentence boundary.

    Cuts at the last sentence terminator that falls inside ``limit``. If there
    is none, hard-cuts and appends an ellipsis so truncation is visible.
    """
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed

    window = collapsed[:limit]
    best = max(window.rfind(". "), window.rfind("! "), window.rfind("? "))
    if best > 0:
        return window[: best + 1]
    return window + "…"
