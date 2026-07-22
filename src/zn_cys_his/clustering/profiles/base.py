"""Structure-profile abstraction shared by every clustering chemistry.

A *profile* owns exactly the pieces of the pipeline that are chemistry-specific:

  * how to parse a source XYZ into a structure object,
  * how to gather + canonicalise a whole dataset,
  * the discrete symmetry (as atom-index permutations) folded out before Kabsch,
  * whether per-structure stats/metrics exist at all,
  * how to write an aligned structure back out to XYZ.

Everything downstream of the feature matrix — z-score, PCA, k-means, the
RMSD-based cluster evaluation, t-SNE, output persistence — is pure geometry and
lives in :mod:`zn_cys_his.clustering.utils`, unchanged, shared by all profiles.

Two structure flavours implement the small interface the shared code needs
(``id``, ``heavy() -> (M,3)``, ``composition() -> hashable``):

  * :class:`zn_cys_his.clustering.utils.Structure` — the original metal +
    coordinating-residue-arms model (Zn/Cys/His), matched by residue permutation.
  * :class:`GenericStructure` — a flat, template-aligned atom cloud used by the
    ``generic`` / ``heme`` profiles, matched by a fixed atom-index symmetry group.

The default (``zn_cys_his``) profile passes ``matcher=None`` through the shared
functions, selecting their original residue-matching code path verbatim, so the
existing pipeline (and the Streamlit app that consumes its output) is byte-for-
byte unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Generic structure: a flat, template-aligned atom cloud
# ---------------------------------------------------------------------------

@dataclass
class GenericStructure:
    """A structure as a fixed-length, template-aligned list of heavy atoms.

    Every structure in a dataset is canonicalised to the SAME ordered atom-name
    template, so slot *i* refers to the same atom for every structure and the
    flattened coordinate vectors are all the same length.  Atoms absent from a
    source file are imputed (see :func:`present`) and carry ``present=False`` so
    the alignment can down-weight them to zero.
    """
    id: str
    coords: np.ndarray                       # (M, 3), canonical template order
    atom_names: list[str]                    # length M
    elements: list[str]                      # length M
    present: np.ndarray                      # (M,) bool; False = imputed/missing
    center_index: Optional[int] = None       # index of the metal/center atom, or None
    template_key: tuple = ()                 # composition signature (shared by dataset)
    family: str = ""                         # categorical label (e.g. heme axial ligand set)

    # --- interface the shared clustering code relies on --------------------
    def heavy(self) -> np.ndarray:
        return self.coords

    def composition(self) -> tuple:
        """A single hashable signature; identical for every structure in a set."""
        return self.template_key

    def n_atoms(self) -> int:
        return len(self.coords)

    # --- transforms used by alignment --------------------------------------
    def transformed(self, R: np.ndarray, t: np.ndarray) -> "GenericStructure":
        return GenericStructure(
            self.id, self.coords @ R.T + t, self.atom_names, self.elements,
            self.present, self.center_index, self.template_key, self.family,
        )

    def reorder_atoms(self, perm: np.ndarray) -> "GenericStructure":
        """Return a copy with atoms reindexed by ``perm`` (new slot i = old perm[i]).

        ``center_index`` is remapped to its new slot so distance weighting stays
        anchored on the same physical atom after a symmetry permutation.
        """
        perm = np.asarray(perm)
        new_center = None
        if self.center_index is not None:
            where = np.where(perm == self.center_index)[0]
            new_center = int(where[0]) if len(where) else self.center_index
        return GenericStructure(
            self.id,
            self.coords[perm],
            [self.atom_names[i] for i in perm],
            [self.elements[i] for i in perm],
            self.present[perm],
            new_center,
            self.template_key,
            self.family,
        )


# ---------------------------------------------------------------------------
# Matcher: the discrete symmetry folded out before Kabsch
# ---------------------------------------------------------------------------

class Matcher:
    """Supplies the atom-index permutations + weighting the aligner searches over.

    A ``matcher`` replaces the residue-permutation machinery (``class_preserving
    _perms`` etc.) with a fixed, template-level symmetry group expressed directly
    as index arrays over ``structure.heavy()``.  The identity permutation is
    always the first element, so a min-RMSD tie resolves to "no permutation".
    """

    def __init__(self, perms: list[np.ndarray], center_index: Optional[int]):
        # Each perm is an (M,) int index array: aligned = heavy()[perm].
        self._perms = [np.asarray(p) for p in perms]
        self.center_index = center_index

    def perms(self, structure) -> list[np.ndarray]:
        return self._perms

    def presence(self, structure) -> np.ndarray:
        """(M,) float weight, 1.0 where the atom is real, 0.0 where imputed."""
        pres = getattr(structure, "present", None)
        if pres is None:
            return np.ones(structure.n_atoms())
        return pres.astype(float)

    def center_dists(self, structure) -> np.ndarray:
        """(M,) per-atom distance to the center atom (1.0 sentinel at center).

        Used by the ``distance`` weight scheme (weight = 1/avg distance), which
        stays meaningful for a rigid molecule: it emphasises the coordination
        core over floppy peripheral substituents.  Falls back to all-ones when
        the profile declares no center.
        """
        if self.center_index is None:
            return np.ones(structure.n_atoms())
        c = structure.heavy()[self.center_index]
        d = np.linalg.norm(structure.heavy() - c, axis=1)
        d[self.center_index] = 1.0
        return d

    def static_w(self, structure, w_type: dict) -> np.ndarray:
        """(M,) static weight for the equal/shell schemes.

        Generic structures have no residue "shell", so every atom is weighted
        equally under both equal and shell; the distance scheme handles emphasis.
        """
        return np.ones(structure.n_atoms())


# ---------------------------------------------------------------------------
# Profile: the pluggable chemistry
# ---------------------------------------------------------------------------

@dataclass
class StructureProfile:
    """Bundle of the chemistry-specific hooks the pipeline dispatches on.

    Attributes
    ----------
    name
        Registry key (``zn_cys_his`` | ``generic`` | ``heme``).
    gather
        ``gather(xyz_dir, glob) -> (structures, report)`` — list, parse, and (for
        template profiles) canonicalise/pad a whole dataset.
    build_matcher
        ``build_matcher(structures) -> Matcher | None``.  ``None`` selects the
        original residue-matching code path in the shared functions (used by
        ``zn_cys_his``); a real :class:`Matcher` selects the atom-index path.
    write_xyz
        ``write_xyz(structure, path)`` — persist an aligned structure.
    has_metrics
        Whether a per-structure stats stage / distribution plots apply.  ``False``
        for generic/heme, so the prep stage and metric plots are skipped.
    """
    name: str
    gather: Callable[[Path, str], tuple]
    build_matcher: Callable[[list], Optional[Matcher]]
    write_xyz: Callable[[object, Path], None]
    has_metrics: bool = False
