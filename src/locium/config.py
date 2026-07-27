"""Paths and tuning defaults.

Every value in ``Tuning`` comes from the "Tuning defaults" table in the design
spec. Nothing else in the codebase may hard-code these numbers — import TUNING.
"""

from dataclasses import dataclass
from pathlib import Path

DEFAULT_PALACE = Path.home() / ".mempalace" / "palace"
DEFAULT_INDEX = Path.home() / ".locium" / "index"

COLLECTION_NAME = "mempalace_drawers"


@dataclass(frozen=True)
class Tuning:
    arc_max_distance: float = 0.45
    arcs_per_drawer: int = 3
    arc_global_cap: int = 20000
    preview_chars: int = 200
    seed: int = 42
    dot_cap: int = 0  # 0 = no cap: draw every drawer so every hit has a dot
    pad_hall: float = 8.5
    pad_chamber: float = 5.0
    # Chambers with at least this many drawers get sub-clustered: a mined
    # palace files most of its content under one room name, so past this size
    # the room label carries no information and clusters recover it.
    cluster_min: int = 150
    cluster_k_cap: int = 6
    pad_cluster: float = 1.0


TUNING = Tuning()
