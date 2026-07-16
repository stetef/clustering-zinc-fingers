"""Canonical filesystem locations for the project.

Single source of truth so no module has to guess ``parents[N]``.  Layout::

    <REPO_ROOT>/
        src/zn_cys_his/paths.py   <- this file
        data/                     <- DATA_DIR: pristine, read-only inputs
        cluster-output/           <- CLUSTER_OUTPUT: all generated artifacts,
                                     mirroring data/ one dataset dir at a time

Both data/ and cluster-output/ are gitignored and kept locally.
"""
from __future__ import annotations

from pathlib import Path

# src/zn_cys_his/paths.py -> zn_cys_his -> src -> repo root
REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = REPO_ROOT / "data"
CLUSTER_OUTPUT: Path = REPO_ROOT / "cluster-output"


def mirror_output(base_dir: Path) -> Path:
    """Map an input dataset dir under data/ to its cluster-output/ mirror.

    ``data/4cys-large`` -> ``cluster-output/4cys-large``.  Any path outside
    data/ falls back to ``cluster-output/<name>`` so the function is total.
    """
    base_dir = base_dir.expanduser().resolve()
    try:
        rel = base_dir.relative_to(DATA_DIR)
    except ValueError:
        rel = Path(base_dir.name)
    return CLUSTER_OUTPUT / rel
