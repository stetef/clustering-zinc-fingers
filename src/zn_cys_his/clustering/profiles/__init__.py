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


def get_profile(name: str) -> StructureProfile:
    """Return the profile registered under ``name`` (imported lazily)."""
    if name == "zn_cys_his":
        from .zn_cys_his import PROFILE
    elif name == "generic":
        from .generic import PROFILE
    elif name == "heme":
        from .heme import PROFILE
    else:
        raise ValueError(
            f"unknown profile {name!r}; choose from {', '.join(PROFILE_NAMES)}"
        )
    return PROFILE


__all__ = ["get_profile", "PROFILE_NAMES", "StructureProfile",
           "GenericStructure", "Matcher"]
