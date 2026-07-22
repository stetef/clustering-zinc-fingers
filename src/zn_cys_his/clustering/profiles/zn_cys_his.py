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


_FAMILY_DOC = (
    "A compact code for the **coordinating motif** — the sequence order of the four "
    "ligand residues and the secondary structure they sit in — in two parts split by "
    "`-`:\n\n"
    "- **Residue order & spacing** (before the `-`): one letter per ligand along the "
    "sequence — **C** = Cys, **H** = His — with `x`*n* giving the number of residues "
    "*between* consecutive ligands. Ligands on separate chains are split by an extra "
    "`-`.\n"
    "- **Secondary structure** (after the `-`): one letter per residue, same order — "
    "**H** = α-helix, **S** = β-sheet, **L** = loop (irregular).\n\n"
    "Example: `Cx5Hx65Cx1C-HHLL` → Cys, 5-residue gap, His, 65-residue gap, Cys, "
    "1-residue gap, Cys; those four residues sit in helix, helix, loop, loop."
)

PROFILE = StructureProfile(
    name="zn_cys_his",
    gather=_gather,
    build_matcher=_build_matcher,
    write_xyz=write_structure_xyz,
    has_metrics=True,
    family_doc=_FAMILY_DOC,
)
