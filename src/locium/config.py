"""Paths and tuning defaults.

Every value in ``Tuning`` comes from the "Tuning defaults" table in the design
spec. Nothing else in the codebase may hard-code these numbers — import TUNING.
"""

from dataclasses import dataclass
from pathlib import Path

DEFAULT_PALACE = Path.home() / ".mempalace" / "palace"
DEFAULT_INDEX = Path.home() / ".locium" / "index"

COLLECTION_NAME = "mempalace_drawers"

# The layout canvas is in abstract units; the viewer scales it to the viewport.
CANVAS_W = 1000.0
CANVAS_H = 1000.0

# A wing rectangle narrower than this in either dimension is unusable.
MIN_WING_SIDE = 20.0


@dataclass(frozen=True)
class Tuning:
    arc_max_distance: float = 0.45
    arcs_per_drawer: int = 3
    arc_global_cap: int = 20000
    knn_k: int = 10
    wing_umap_threshold: int = 10
    gutter_fraction: float = 0.15
    preview_chars: int = 200
    seed: int = 42
    dot_cap: int = 300
    pad_hall: float = 8.5
    pad_chamber: float = 5.0


TUNING = Tuning()
