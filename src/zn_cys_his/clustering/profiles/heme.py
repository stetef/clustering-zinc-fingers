"""Heme profile: full-structure features + an axial-ligand flip permutation.

A heme is a single *rigid* body (porphyrin + Fe + axial ligands), so — unlike
Zn(Cys/His) with its independent residue arms — there is no free per-arm
permutation.  Atoms keep the standardised HEM naming and a fixed 1:1
correspondence (:mod:`generic`), and the only symmetry we fold out is the
"which axial ligand is up" ambiguity: a 180° flip about an in-plane axis of the
porphyrin.

Why the flip must be a *relabeling*, not a rigid pre-rotation
-------------------------------------------------------------
Weighted Kabsch already finds the optimal rigid rotation for a given atom
correspondence, so applying a 180° rotation to a structure *before* Kabsch adds
nothing — Kabsch would have found that rotation if it lowered RMSD.  The flip
only carries information as an atom **relabeling**: map each atom to its image
under the porphyrin's C2, so a genuinely flipped structure can then be rotated
by Kabsch onto the reference and score a low RMSD.

The flip permutation is discovered geometrically from the reference structure —
find the in-plane C2 axis that best maps the molecule onto itself, then match
each atom to its rotated image — so it needs no hardcoded HEM atom names and
adapts to whatever atom set the files actually contain.  If no clean, bijective
self-image is found (e.g. the axial ligands are chemically different, breaking
the symmetry), the matcher falls back to identity-only and the pipeline still
runs — every structure just clusters on its literal orientation.

NOTE: validate the discovered flip against a handful of real heme XYZ files
(compare identity vs. flip RMSD on a known flipped pair) before trusting it for
production clustering; this profile ships the mechanism, not a tuned constant.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .base import Matcher, StructureProfile
from .generic import make_gather, write_generic_xyz


# Element used as the coordination center (drives distance weighting + the
# porphyrin-plane fit).
_CENTER = "FE"

# Geometric tolerances for the self-image search.
_IMAGE_TOL = 0.8      # Å: max atom-to-image distance to count as a match
_MIN_SELF_FRACTION = 0.9   # fraction of atoms that must map for a valid flip


def _porphyrin_normal(coords: np.ndarray, center_index: int,
                      elements: list[str]) -> Optional[np.ndarray]:
    """Estimate the porphyrin-plane normal from Fe + nearby nitrogens.

    Uses the (up to) four nitrogens closest to Fe — the pyrrole N donors — plus
    Fe itself; the plane normal is the smallest-variance SVD direction.
    """
    fe = coords[center_index]
    n_idx = [i for i, e in enumerate(elements) if e.upper() == "N"]
    if len(n_idx) < 3:
        return None
    n_idx.sort(key=lambda i: np.linalg.norm(coords[i] - fe))
    ring = np.array([coords[i] for i in n_idx[:4]] + [fe])
    ring = ring - ring.mean(0)
    _, _, vt = np.linalg.svd(ring)
    return vt[-1]


def _nearest_image_perm(coords: np.ndarray, R: np.ndarray,
                        center_index: int) -> Optional[np.ndarray]:
    """Permutation mapping each atom to its nearest image under rotation R.

    Returns an (M,) index array (bijective) or None if the image isn't a clean
    self-map (atoms unmatched or matched non-uniquely beyond tolerance).
    """
    fe = coords[center_index]
    img = (coords - fe) @ R.T + fe
    M = len(coords)
    perm = np.full(M, -1, dtype=int)
    used = np.zeros(M, dtype=bool)
    matched = 0
    # Greedy nearest-neighbour matching (M is small for a heme).
    order = list(range(M))
    for i in order:
        d = np.linalg.norm(coords - img[i], axis=1)
        d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] <= _IMAGE_TOL:
            perm[i] = j
            used[j] = True
            matched += 1
    if matched < int(_MIN_SELF_FRACTION * M):
        return None
    # Fill any unmatched slots with identity so the result stays a permutation.
    for i in range(M):
        if perm[i] == -1:
            if not used[i]:
                perm[i] = i
                used[i] = True
            else:
                free = np.where(~used)[0]
                if len(free) == 0:
                    return None
                perm[i] = int(free[0])
                used[free[0]] = True
    if len(set(perm.tolist())) != M:
        return None
    return perm


def _axial_flip_perms(reference) -> list[np.ndarray]:
    """Build the axial-flip permutation set from a reference GenericStructure.

    Searches in-plane C2 axes (toward each nitrogen and each meso/bridge carbon)
    for the 180° rotation that best self-maps the molecule; returns the matching
    relabeling(s).  Empty list => no clean flip found (matcher stays identity).
    """
    coords = reference.heavy()
    ci = reference.center_index
    if ci is None:
        return []
    normal = _porphyrin_normal(coords, ci, reference.elements)
    if normal is None:
        return []
    normal = normal / np.linalg.norm(normal)
    fe = coords[ci]

    # Candidate in-plane axes: projections of atom directions onto the plane.
    perms: list[np.ndarray] = []
    seen: set[tuple] = set()
    for i in range(len(coords)):
        if i == ci:
            continue
        v = coords[i] - fe
        v = v - np.dot(v, normal) * normal   # project into porphyrin plane
        nv = np.linalg.norm(v)
        if nv < 1e-3:
            continue
        a = v / nv
        # 180° rotation about in-plane axis a: R = 2 a aᵀ − I  (flips the normal).
        R = 2.0 * np.outer(a, a) - np.eye(3)
        perm = _nearest_image_perm(coords, R, ci)
        if perm is None:
            continue
        key = tuple(perm.tolist())
        if key not in seen and not np.array_equal(perm, np.arange(len(coords))):
            seen.add(key)
            perms.append(perm)
    # Keep the single best (most atoms are non-fixed → the genuine top/bottom C2)
    if not perms:
        return []
    perms.sort(key=lambda p: -int(np.sum(p != np.arange(len(p)))))
    return perms[:1]


def _build_matcher(structures: list) -> Optional[Matcher]:
    if not structures:
        return None
    M = structures[0].n_atoms()
    # Use the most-complete structure as the geometric reference for the flip.
    ref = max(structures, key=lambda s: int(np.sum(s.present)))
    perms = [np.arange(M)]
    for p in _axial_flip_perms(ref):
        if p.shape == (M,):
            perms.append(p)
    return Matcher(perms, structures[0].center_index)


def make_profile(pdb_dir=None, fetch_pdbs: bool = False) -> StructureProfile:
    from .pdb_tags import MACROCYCLE_RES
    return StructureProfile(
        name="heme",
        gather=make_gather(center_name=_CENTER, pdb_dir=pdb_dir, fetch_pdbs=fetch_pdbs,
                           macrocycle=MACROCYCLE_RES),
        build_matcher=_build_matcher,
        write_xyz=write_generic_xyz,
        has_metrics=False,
    )


PROFILE = make_profile()
