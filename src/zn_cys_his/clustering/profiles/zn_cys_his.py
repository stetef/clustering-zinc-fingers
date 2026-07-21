"""The original Zn(Cys/His) chemistry, expressed as a profile.

This profile is a thin wrapper: it reuses the existing parser, dataset gather,
and XYZ writer from :mod:`zn_cys_his.clustering.utils` unchanged, and returns
``build_matcher() -> None``.  A ``None`` matcher tells the shared clustering
functions to take their original residue-permutation code path, so selecting
this profile reproduces the pre-refactor pipeline exactly (the Streamlit app's
inputs are unaffected).
"""
from __future__ import annotations

from pathlib import Path

from zn_cys_his.clustering.utils import gather_structures, write_structure_xyz

from .base import StructureProfile


def _gather(xyz_dir: Path, glob_pat: str) -> tuple:
    return gather_structures(xyz_dir, glob_pat, desc="parsing")


def _build_matcher(structures: list) -> None:
    # None => shared functions use their built-in Cys/His residue matching.
    return None


PROFILE = StructureProfile(
    name="zn_cys_his",
    gather=_gather,
    build_matcher=_build_matcher,
    write_xyz=write_structure_xyz,
    has_metrics=True,
)
