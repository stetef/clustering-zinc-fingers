"""Structure-profile registry.

A profile bundles the chemistry-specific hooks (parse, gather, symmetry matcher,
XYZ writer, whether metrics apply); everything else in the clustering pipeline
is profile-agnostic.  See :mod:`.base` for the interface.

  * ``zn_cys_his`` (default) — the original Zn + Cys/His residue-arm model;
    behaviour-preserving, used by the Streamlit datasets.
  * ``generic`` — arbitrary rigid structures, fixed atom template, identity symmetry.
  * ``heme`` — generic + Fe-centred distance weighting + an axial-ligand flip.
"""
from __future__ import annotations

from .base import GenericStructure, Matcher, StructureProfile

PROFILE_NAMES = ("zn_cys_his", "generic", "heme")


def get_profile(name: str, pdb_dir=None, fetch_pdbs: bool = False) -> StructureProfile:
    """Return the profile registered under ``name`` (imported lazily).

    ``pdb_dir`` / ``fetch_pdbs`` apply only to the generic/heme profiles: they let
    the tag-less-XYZ parser recover atom names from source PDBs (see
    :mod:`.pdb_tags`); they are ignored by ``zn_cys_his``.
    """
    if name == "zn_cys_his":
        from .zn_cys_his import PROFILE
        return PROFILE
    if name == "generic":
        from .generic import make_profile
        return make_profile(pdb_dir=pdb_dir, fetch_pdbs=fetch_pdbs)
    if name == "heme":
        from .heme import make_profile
        return make_profile(pdb_dir=pdb_dir, fetch_pdbs=fetch_pdbs)
    raise ValueError(f"unknown profile {name!r}; choose from {', '.join(PROFILE_NAMES)}")


__all__ = ["get_profile", "PROFILE_NAMES", "StructureProfile",
           "GenericStructure", "Matcher"]
