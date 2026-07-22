"""Shared utilities for Zn(Cys/His)₄ featurization and clustering.

Implements the data model, parsing, Kabsch superposition, matching-minimized
structural RMSD, and the clustering/evaluation pipeline shared across all three
featurization approaches (scripts 02–04).

The data model is composition-generic: a Zn site is the metal plus an ordered
list of coordinating residues (Cys and/or His), each keeping its own heavy-atom
arm.  Residue matching uses two classes — Cys and His — so Cys never maps onto
His, while His-ND1 and His-NE2 are matchable (their geometric difference shows
up as RMSD).  A pure-Cys dataset reduces exactly to the original 4×(S,Cβ,Cα)
behaviour.

Public API
----------
Residue, Structure, structure_like, structure_bonds
EQUAL_WEIGHTS, SHELL_WEIGHTS, DISTANCE_WEIGHTS, ARM_ATOMS, ELEMENT_OF
parse_structure(path) -> Structure | None
weighted_kabsch(P, Q, w, allow_reflection) -> (R, t, rmsd)
class_preserving_perms(res_types) -> list[list[int]]
structural_rmsd(A, B, w_type, allow_reflection) -> (rmsd, best_perm, (R, t))
cluster_pipeline(X, k, ...) -> (labels, centroids_pca, pca, (means, stds))
evaluate_clustering(structures, labels, X_pca, centroids, w_type) -> dict  # returns intra/inter/ratio/ch_score
sweep_k(structures, X, k_values, w_type, ...) -> (best_result, all_results)
save_outputs(out_dir, id_list, results_table, best_result)
save_embeddings_and_tsne(out_dir, id_list, X_pca, labels) -> np.ndarray
build_cluster_distribution_plots(out_dir, id_list, labels, stats_csv)
write_structure_xyz(structure, path)
"""
from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations, permutations, product
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL_OK = True
except ImportError:
    _MPL_OK = False

try:
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
except ImportError:
    raise SystemExit("scikit-learn required: uv add scikit-learn")

import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Static weight presets.  Keyed by atom *role*, not atom name, so they apply to
# any coordinating-residue chemistry:
#   "Zn"    — the central metal
#   "coord" — the atom that ligates Zn (SG for Cys; ND1/NE2 for His)
#   "other" — every other heavy atom in the residue arm
EQUAL_WEIGHTS: dict[str, float] = {"Zn": 1.0, "coord": 1.0, "other": 1.0}
SHELL_WEIGHTS: dict[str, float] = {"Zn": 1.0, "coord": 1.0, "other": 0.5}
# Sentinel: weights = 1 / avg_distance_from_Zn per atom pair; Zn itself gets 1.
DISTANCE_WEIGHTS: str = "distance"

# ---------------------------------------------------------------------------
# Coordinating-residue chemistry
# ---------------------------------------------------------------------------
# Heavy-atom "arm" retained for each coordinating residue type, in a fixed
# canonical order.  This order defines the per-residue feature layout and is
# reused everywhere (RMSD, XYZ output, reconstruction).
#
# Cys arm = (SG, CB, CA) — coordinating S plus the two backbone-side anchors.
# His arm = the full imidazole ring (CG, ND1, CD2, CE1, NE2) plus (CB, CA).
# Including the whole ring means a His coordinating via ND1 and one via NE2 use
# the *same* atom set, so they can always be aligned (their geometric
# difference surfaces as RMSD rather than an impossible match).
ARM_ATOMS: dict[str, list[str]] = {
    "CYS": ["SG", "CB", "CA"],
    "HIS": ["CG", "ND1", "CD2", "CE1", "NE2", "CB", "CA"],
}

# Candidate Zn-ligating atoms per residue type.  The actual coordinating atom
# for a given residue is whichever candidate is closest to Zn.
COORD_CANDIDATES: dict[str, list[str]] = {
    "CYS": ["SG"],
    "HIS": ["ND1", "NE2"],
}

# Coordinating water is tolerated (ignored) rather than disqualifying a site.
WATER_RES: set = {"HOH", "WAT", "H2O", "DOD", "SOL", "TIP3"}

# Element symbol for the XYZ element column, keyed by heavy-atom name.
ELEMENT_OF: dict[str, str] = {
    "SG": "S", "ND1": "N", "NE2": "N",
    "CG": "C", "CD2": "C", "CE1": "C", "CB": "C", "CA": "C",
}

# Intra-residue bonds (by atom name) used only for visualisation/reconstruction.
BOND_TEMPLATE: dict[str, list[tuple[str, str]]] = {
    "CYS": [("CA", "CB"), ("CB", "SG")],
    "HIS": [("CA", "CB"), ("CB", "CG"), ("CG", "ND1"), ("ND1", "CE1"),
            ("CE1", "NE2"), ("NE2", "CD2"), ("CD2", "CG")],
}

# Matching classes: His coordinating via ND1 or NE2 are the SAME class ("HIS")
# so a single global reference always exists.  Cys and His never interchange.
def res_class(res_type: str) -> str:
    """Return the matching class for a residue type."""
    return "HIS" if res_type == "HIS" else "CYS"

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Residue:
    """One coordinating residue and its retained heavy-atom arm."""
    res_type: str                       # "CYS" | "HIS"
    coord_atom: str                     # atom name ligating Zn (SG / ND1 / NE2)
    coords: dict                        # atom_name -> np.ndarray(3,); the full arm
    resseq: int = 0
    chain: str = "A"
    res_name: str = "CYS"

    @property
    def cls(self) -> str:
        """Matching class (His-ND1 and His-NE2 collapse to 'HIS')."""
        return res_class(self.res_type)

    def arm_names(self) -> list:
        return ARM_ATOMS[self.res_type]

    def arm(self) -> np.ndarray:
        """(n_atoms, 3) arm coordinates in canonical ARM_ATOMS order."""
        return np.array([self.coords[a] for a in ARM_ATOMS[self.res_type]])

    def transformed(self, R: np.ndarray, t: np.ndarray) -> "Residue":
        return Residue(self.res_type, self.coord_atom,
                       {a: (v @ R.T + t) for a, v in self.coords.items()},
                       self.resseq, self.chain, self.res_name)


@dataclass
class Structure:
    """A Zn site: the metal plus an ordered list of coordinating residues.

    Residues are held in a canonical slot order (Cys before His, then by
    RESSEQ) that is identical for every structure of the same composition, so
    slot i always refers to the same class across the dataset.
    """
    id: str
    zn: np.ndarray                      # (3,)
    residues: list = field(default_factory=list)   # list[Residue]
    zn_resseq: int = 0
    zn_chain:  str = "A"
    zn_res:    str = "ZN"
    # Heavy-atom count of the coordinating residues only (by RESSEQ) + Zn, taken
    # from the source file.  Excludes waters / co-ligands / other residues, so
    # the modal-count filter measures the actual coordination, not pocket
    # contents.  Equals n_atoms() when the file holds only the arm atoms.
    coord_heavy_count: int = 0
    # Names of any COORD=TRUE residue that is neither Cys/His nor water (e.g.
    # GLU, SO4, IMP).  Non-empty ⇒ a mixed-ligand site, not a clean Cys/His
    # coordination — such structures are dropped by gather_structures.
    extra_coord_ligands: list = field(default_factory=list)

    def n_res(self) -> int:
        return len(self.residues)

    def composition(self) -> tuple:
        """Class signature, e.g. ('CYS','CYS','HIS','HIS')."""
        return tuple(r.cls for r in self.residues)

    def heavy(self) -> np.ndarray:
        """(M, 3): Zn followed by each residue's arm, in slot order."""
        return np.vstack([self.zn[None]] + [r.arm() for r in self.residues])

    def n_atoms(self) -> int:
        return 1 + sum(len(r.arm_names()) for r in self.residues)

    def atom_meta(self) -> list:
        """Per heavy-atom (res_idx | -1 for Zn, atom_name, element, is_coord)."""
        meta = [(-1, "ZN", "ZN", False)]
        for i, r in enumerate(self.residues):
            for a in r.arm_names():
                meta.append((i, a, ELEMENT_OF[a], a == r.coord_atom))
        return meta

    def typed_atoms(self) -> list:
        """(label, xyz) for every heavy atom; label = 'ZN' or '<TYPE>_<ATOM>'.

        Used by PIV: labels group chemically-identical atoms so sorting within a
        label-pair block stays permutation invariant and consistent per
        composition.
        """
        out = [("ZN", self.zn)]
        for r in self.residues:
            for a in r.arm_names():
                out.append((f"{r.res_type}_{a}", r.coords[a]))
        return out

    def w_vec(self, w_type: dict) -> np.ndarray:
        """(M,) static weight vector for the equal/shell schemes."""
        w = [w_type["Zn"]]
        for r in self.residues:
            for a in r.arm_names():
                w.append(w_type["coord"] if a == r.coord_atom else w_type["other"])
        return np.array(w, dtype=float)

    def reorder(self, perm: list) -> "Structure":
        """Return a copy with residues reordered: new slot i = old perm[i]."""
        return Structure(self.id, self.zn,
                         [self.residues[perm[i]] for i in range(len(perm))],
                         self.zn_resseq, self.zn_chain, self.zn_res)

    def transformed(self, R: np.ndarray, t: np.ndarray) -> "Structure":
        return Structure(self.id, self.zn @ R.T + t,
                         [r.transformed(R, t) for r in self.residues],
                         self.zn_resseq, self.zn_chain, self.zn_res)


def structure_like(template: "Structure", heavy_coords: np.ndarray,
                   sid: str) -> "Structure":
    """Rebuild a Structure from a flat (M,3) heavy-atom array using template's
    residue layout (types, atom names, metadata).  Used for PCA→XYZ reconstruction.
    """
    zn = heavy_coords[0]
    residues: list = []
    idx = 1
    for r in template.residues:
        names = r.arm_names()
        coords = {a: heavy_coords[idx + k] for k, a in enumerate(names)}
        idx += len(names)
        residues.append(Residue(r.res_type, r.coord_atom, coords,
                                r.resseq, r.chain, r.res_name))
    return Structure(sid, zn, residues,
                     template.zn_resseq, template.zn_chain, template.zn_res)


def structure_bonds(s: "Structure") -> list:
    """List of (i, j) index pairs into heavy() for drawing bonds.

    Includes Zn→coordinating-atom for each residue plus intra-arm bonds from
    BOND_TEMPLATE.
    """
    # Map (res_idx, atom_name) -> flat heavy() index.
    pos: dict = {}
    idx = 1
    for ri, r in enumerate(s.residues):
        for a in r.arm_names():
            pos[(ri, a)] = idx
            idx += 1
    bonds: list = []
    for ri, r in enumerate(s.residues):
        bonds.append((0, pos[(ri, r.coord_atom)]))          # Zn → ligand
        for a, b in BOND_TEMPLATE[r.res_type]:
            if (ri, a) in pos and (ri, b) in pos:
                bonds.append((pos[(ri, a)], pos[(ri, b)]))
    return bonds

# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_atom_name(comment: str) -> Optional[str]:
    """Return the raw heavy-atom name from the 'ATOM=...' EOL tag (upper-cased).

    Bare 'S' is normalised to 'SG' so both spellings map to the Cys sulfur.
    """
    for part in comment.split():
        if part.startswith("ATOM="):
            tag = part[5:].strip().upper()
            return "SG" if tag == "S" else tag
    return None


def _parse_coord_flag(comment: str) -> bool:
    """True if the atom carries COORD=TRUE (marks a coordinating residue)."""
    for part in comment.split():
        if part.startswith("COORD="):
            return part[6:].strip().upper() == "TRUE"
    return False


def _parse_resseq(comment: str) -> Optional[int]:
    for part in comment.split():
        if part.startswith("RESSEQ="):
            try:
                return int(part[7:])
            except ValueError:
                pass
    return None


def _parse_chain(comment: str) -> str:
    for part in comment.split():
        if part.startswith("CHAIN="):
            return part[6:].strip()
    return ""


def _parse_res(comment: str) -> str:
    for part in comment.split():
        if part.startswith("RES="):
            return part[4:].strip()
    return ""


def parse_structure(path: Path) -> Optional[Structure]:
    """Parse an XYZ file with ATOM= EOL tags into a Structure.  None on failure.

    Selects the coordinating residues (Cys/His), keeps each one's canonical arm
    (see ARM_ATOMS), and picks its Zn-ligating atom as the closest candidate.

    Works for two input flavours:
      * cleaned XYZ (only the coordinating heavy atoms present, no COORD tags)
      * extended XYZ (whole pocket; coordinating residues flagged COORD=TRUE)
    When any COORD=TRUE flags are present, only flagged residues are kept.
    """
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return None
    if len(lines) < 2:
        return None

    zn_coords: list[np.ndarray] = []
    zn_comments: list[str] = []
    # (chain, resseq) -> {"_RES": str, "_COORD": bool, atom_name: xyz}
    residues: dict[tuple[str, int], dict] = {}

    for line in lines[2:]:
        line = line.strip()
        if not line:
            continue
        halves = line.split("#", 1)
        coords_part = halves[0].split()
        comment = halves[1] if len(halves) > 1 else ""

        atom_name = _parse_atom_name(comment)
        if atom_name is None or len(coords_part) < 4:
            continue
        try:
            xyz = np.array([float(coords_part[1]), float(coords_part[2]), float(coords_part[3])])
        except ValueError:
            continue

        if atom_name == "ZN":
            zn_coords.append(xyz)
            zn_comments.append(comment)
            continue

        resseq = _parse_resseq(comment)
        if resseq is None:
            continue
        key = (_parse_chain(comment), resseq)
        entry = residues.setdefault(key, {"_RES": _parse_res(comment) or "", "_COORD": False})
        entry[atom_name] = xyz
        if _parse_coord_flag(comment):
            entry["_COORD"] = True

    if len(zn_coords) != 1:
        return None
    zn = zn_coords[0]
    zn_comment = zn_comments[0]

    any_coord_flag = any(v["_COORD"] for v in residues.values())

    built: list[Residue] = []
    for (chain, resseq), v in residues.items():
        res_name = (v.get("_RES") or "").upper()
        rtype = res_name if res_name in ARM_ATOMS else None
        if rtype is None:
            continue
        if any_coord_flag and not v["_COORD"]:
            continue
        arm = ARM_ATOMS[rtype]
        if not all(a in v for a in arm):
            continue
        coords = {a: v[a] for a in arm}
        cand = [a for a in COORD_CANDIDATES[rtype] if a in coords]
        if not cand:
            continue
        coord_atom = min(cand, key=lambda a: float(np.linalg.norm(coords[a] - zn)))
        built.append(Residue(rtype, coord_atom, coords, resseq, chain, res_name))

    if len(built) < 2:
        return None

    # Canonical slot order: Cys before His, then by RESSEQ, then chain.
    # Deterministic and identical for every structure of the same composition.
    _rank = {"CYS": 0, "HIS": 1}
    built.sort(key=lambda r: (_rank[r.res_type], r.resseq, r.chain))

    # Heavy-atom count of the coordinating residues (by RESSEQ) + Zn — counts
    # every heavy atom listed for those residues, but nothing from waters or
    # co-ligands (SO4, GLU, …) that happen to fall inside the cluster cutoff.
    # Count heavy atoms only: hydrogen ATOM tags (H, HA, HB2, HD1, …) all start
    # with H/D, and no Cys/His heavy atom name does, so a prefix test is safe.
    coord_heavy_count = 1  # Zn
    for r in built:
        v = residues[(r.chain, r.resseq)]
        coord_heavy_count += sum(1 for k in v
                                 if not k.startswith(("_", "H", "D")))

    # Disqualifying coordinating ligands: any COORD=TRUE residue that is neither
    # Cys/His nor water.  These make the site a mixed-ligand coordination, not a
    # clean Cys/His set, so it should be rejected (waters are tolerated).
    extra_coord_ligands = sorted({
        rn for v in residues.values()
        if v.get("_COORD") and (rn := (v.get("_RES") or "").upper())
        and rn not in ARM_ATOMS and rn not in WATER_RES
    })

    return Structure(
        id=path.stem, zn=zn, residues=built,
        zn_resseq=_parse_resseq(zn_comment) or 0,
        zn_chain=_parse_chain(zn_comment) or "A",
        zn_res=_parse_res(zn_comment) or "ZN",
        coord_heavy_count=coord_heavy_count,
        extra_coord_ligands=extra_coord_ligands,
    )


def write_structure_xyz(structure: Structure, path: Path) -> None:
    """Write a heavy-atom XYZ preserving RES/CHAIN/RESSEQ metadata.

    Atom count and element symbols follow the structure's composition, so the
    output re-parses back into an equivalent Structure.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    coords = structure.heavy()
    meta = structure.atom_meta()
    with path.open("w", encoding="utf-8") as fh:
        fh.write(f"{len(coords)}\n")
        fh.write(f"id={structure.id}\n")
        for (ridx, aname, elem, is_coord), xyz in zip(meta, coords):
            if ridx < 0:
                fh.write(
                    f"{elem:<2} {xyz[0]:.6f}  {xyz[1]:.6f}  {xyz[2]:.6f}"
                    f"  # RES={structure.zn_res} CHAIN={structure.zn_chain}"
                    f" RESSEQ={structure.zn_resseq} ATOM=ZN\n"
                )
            else:
                r = structure.residues[ridx]
                fh.write(
                    f"{elem:<2} {xyz[0]:.6f}  {xyz[1]:.6f}  {xyz[2]:.6f}"
                    f"  # RES={r.res_name} CHAIN={r.chain} RESSEQ={r.resseq}"
                    f" ATOM={aname} COORD={'TRUE' if is_coord else 'FALSE'}\n"
                )

# ---------------------------------------------------------------------------
# Weighted Kabsch superposition
# ---------------------------------------------------------------------------

def weighted_kabsch(
    P: np.ndarray,
    Q: np.ndarray,
    w: np.ndarray,
    allow_reflection: bool = True,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Weighted Kabsch: find R, t such that P @ R.T + t ≈ Q (minimises weighted RMSD).

    Returns (R, t, rmsd).  If allow_reflection=True, keeps an improper rotation
    when it gives lower RMSD (merges enantiomers).
    """
    w = np.asarray(w, dtype=float)
    w = w / w.sum()
    cp = (w[:, None] * P).sum(0)
    cq = (w[:, None] * Q).sum(0)
    Pc = P - cp
    Qc = Q - cq

    H = (w[:, None] * Pc).T @ Qc
    U, _, Vt = np.linalg.svd(H)

    R_raw = Vt.T @ U.T
    det = np.linalg.det(R_raw)

    # Proper rotation (standard Kabsch diag(1,1,d) correction)
    Vt_corr = Vt.copy()
    if det < 0:
        Vt_corr[-1] *= -1
    R_proper = Vt_corr.T @ U.T

    def _rmsd(R: np.ndarray) -> float:
        return float(np.sqrt((w * ((Pc @ R.T - Qc) ** 2).sum(1)).sum()))

    if allow_reflection and det < 0:
        rmsd_p = _rmsd(R_proper)
        rmsd_r = _rmsd(R_raw)
        R = R_raw if rmsd_r < rmsd_p else R_proper
    else:
        R = R_proper

    t = cq - R @ cp
    return R, t, _rmsd(R)

# ---------------------------------------------------------------------------
# Matching-minimized structural RMSD (metric)
# ---------------------------------------------------------------------------

def class_preserving_perms(res_types: list) -> list:
    """All residue re-orderings that keep each slot's matching class fixed.

    `res_types` is the canonical class list (e.g. ['CYS','CYS','HIS','HIS']).
    A returned perm maps slot i -> source residue index perm[i], permuting only
    within each class.  For 4 identical Cys this yields all 4! = 24 orderings
    (backwards-compatible); for 2 Cys + 2 His it yields 2!·2! = 4, etc.
    """
    by_class: dict = defaultdict(list)
    for i, t in enumerate(res_types):
        by_class[t].append(i)
    classes = sorted(by_class)
    per_class = [list(permutations(by_class[c])) for c in classes]
    perms: list = []
    for combo in product(*per_class):
        perm = [0] * len(res_types)
        for c, order in zip(classes, combo):
            for slot, src in zip(by_class[c], order):
                perm[slot] = src
        perms.append(perm)
    return perms


def _heavy_perm(B: Structure, perm: list[int]) -> np.ndarray:
    """(M, 3) heavy coordinates of B with residues reordered by perm."""
    return np.vstack([B.zn[None]] + [B.residues[perm[i]].arm() for i in range(len(perm))])


def _atom_zn_dists(structure: Structure, order: list) -> np.ndarray:
    """Per heavy-atom distance to Zn, Zn slot = 1.0 sentinel, residues in `order`."""
    d = [1.0]
    for ri in order:
        r = structure.residues[ri]
        for a in r.arm_names():
            d.append(float(np.linalg.norm(r.coords[a] - structure.zn)))
    return np.array(d)


def _zn_distance_weights(A: Structure, B: Structure, perm: list[int]) -> np.ndarray:
    """(M,) weight vector = 1 / avg_Zn_distance for a residue permutation of B.

    Slot i pairs A's residue i with B's residue perm[i] (same class, same arm),
    so the arms line up atom-for-atom.  weight = 1 / ((d_A + d_B) / 2);
    Zn uses a 1.0 Å sentinel → weight 1.0.
    """
    d_A = _atom_zn_dists(A, list(range(A.n_res())))
    d_B = _atom_zn_dists(B, perm)
    return 1.0 / ((d_A + d_B) / 2.0)


def _structural_rmsd_matcher(
    A,
    B,
    w_type: dict | str,
    allow_reflection: bool,
    matcher,
) -> tuple[float, np.ndarray, tuple]:
    """Matcher-based RMSD: minimize over the profile's atom-index permutations.

    Used by the generic/heme profiles.  ``matcher.perms(A)`` supplies index
    arrays over ``heavy()``; imputed/missing atoms (matcher.presence) are
    down-weighted to zero so they never drive the fit.  Returns
    (rmsd, best_atom_perm, (R, t)); best_atom_perm is an (M,) index array.
    """
    if A.composition() != B.composition():
        return math.inf, np.arange(A.n_atoms()), (np.eye(3), np.zeros(3))

    A_heavy = A.heavy()
    B_heavy = B.heavy()
    pres_A = matcher.presence(A)
    pres_B = matcher.presence(B)
    is_dist = (w_type == DISTANCE_WEIGHTS)
    d_A = matcher.center_dists(A) if is_dist else None
    d_B = matcher.center_dists(B) if is_dist else None
    static_w = None if is_dist else matcher.static_w(A, w_type)

    best_rmsd = math.inf
    best_perm = np.arange(len(A_heavy))
    best_Rt: tuple = (np.eye(3), np.zeros(3))
    for perm in matcher.perms(A):
        w = (1.0 / ((d_A + d_B[perm]) / 2.0)) if is_dist else static_w
        w = w * pres_A * pres_B[perm]
        if w.sum() <= 0:
            continue
        R, t, rmsd = weighted_kabsch(B_heavy[perm], A_heavy, w, allow_reflection)
        if rmsd < best_rmsd:
            best_rmsd = rmsd
            best_perm = perm
            best_Rt = (R, t)

    return best_rmsd, best_perm, best_Rt


def structural_rmsd(
    A: Structure,
    B: Structure,
    w_type: dict | str,
    allow_reflection: bool = True,
    matcher=None,
) -> tuple[float, list[int], tuple]:
    """Matching-minimized RMSD over class-preserving residue permutations of B.

    Returns (rmsd, best_perm, (R, t)).  Cys never matches His; His-ND1 and
    His-NE2 are both "His" so they can match (their geometric difference shows
    up in the RMSD).  If A and B have different composition the RMSD is infinite.
    When w_type == DISTANCE_WEIGHTS, per-atom weights are recomputed per perm.

    When ``matcher`` is provided (generic/heme profiles), permutations come from
    the matcher's fixed atom-index symmetry group instead of residue matching.
    """
    if matcher is not None:
        return _structural_rmsd_matcher(A, B, w_type, allow_reflection, matcher)

    if A.composition() != B.composition():
        return math.inf, list(range(A.n_res())), (np.eye(3), np.zeros(3))

    A_heavy = A.heavy()
    res_types = list(A.composition())
    best_rmsd = math.inf
    best_perm = list(range(len(res_types)))
    best_Rt: tuple = (np.eye(3), np.zeros(3))
    _static_w = None if w_type == DISTANCE_WEIGHTS else A.w_vec(w_type)

    for perm in class_preserving_perms(res_types):
        w = _zn_distance_weights(A, B, perm) if w_type == DISTANCE_WEIGHTS else _static_w
        R, t, rmsd = weighted_kabsch(_heavy_perm(B, perm), A_heavy, w, allow_reflection)
        if rmsd < best_rmsd:
            best_rmsd = rmsd
            best_perm = perm
            best_Rt = (R, t)

    return best_rmsd, best_perm, best_Rt

def list_structure_files(xyz_dir: Path, glob_pat: str = "*.xyz") -> list:
    """List candidate XYZ files under xyz_dir.

    `.pc` / `.gzmat` are excluded by the `*.xyz` pattern.  `*-extended.xyz`
    files are rejected by default (the His datasets ship both a full-pocket
    `*_Zn-extended.xyz` and a coordinating-cluster `*_Zn.xyz` for each
    structure — they parse to the same site, so keeping only the non-extended
    one leaves a single file per structure).  If the glob pattern itself asks
    for extended files (contains "extended"), they are kept.
    """
    files = sorted(xyz_dir.glob(glob_pat))
    if "extended" not in glob_pat.lower():
        files = [f for f in files if "-extended" not in f.stem.lower()]
    return files


def count_heavy_atoms(path: Path) -> Optional[int]:
    """Count non-hydrogen atoms in an XYZ file (all atoms, not just the arm)."""
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return None
    n = 0
    for line in lines[2:]:
        parts = line.split("#", 1)[0].split()
        if len(parts) < 4:
            continue
        try:
            float(parts[1]); float(parts[2]); float(parts[3])
        except ValueError:
            continue
        if parts[0].strip().upper() in ("H", "D"):
            continue
        n += 1
    return n


def filter_by_modal_atom_count(paths: list) -> tuple:
    """Keep files whose heavy-atom count equals the modal count across `paths`.

    Files that differ from the mode are likely truncated/incomplete or a
    different coordination (not a true conformation) and are dropped.  Files
    whose atom count can't be read are kept (parse handles them downstream).
    Returns (kept_paths, dropped [(path, count), ...], modal_count).
    """
    counts: dict = {}
    for p in paths:
        c = count_heavy_atoms(p)
        if c is not None:
            counts[p] = c
    if not counts:
        return list(paths), [], None
    mode = Counter(counts.values()).most_common(1)[0][0]
    kept = [p for p in paths if counts.get(p, mode) == mode]
    dropped = [(p, counts[p]) for p in paths if p in counts and counts[p] != mode]
    return kept, dropped, mode


def filter_by_modal_coord_count(structures: list) -> tuple:
    """Keep structures whose coordinating-residue heavy-atom count is modal.

    The count (`Structure.coord_heavy_count`) covers only the coordinating
    residues by RESSEQ + Zn, so waters and co-ligands (SO4, GLU, IMP, …) inside
    the cluster cutoff don't affect it — those files are kept, not dropped.
    A structure is dropped only if its *coordinating* residues have an unusual
    atom count (truncated / altloc / partial-occupancy — "not a true
    conformation").  Returns (kept, dropped [(id, count), ...], modal_count).
    """
    if not structures:
        return [], [], None
    mode = Counter(s.coord_heavy_count for s in structures).most_common(1)[0][0]
    kept    = [s for s in structures if s.coord_heavy_count == mode]
    dropped = [(s.id, s.coord_heavy_count) for s in structures
               if s.coord_heavy_count != mode]
    return kept, dropped, mode


def filter_extra_coord_ligands(structures: list) -> tuple:
    """Drop structures with a coordinating ligand that isn't Cys/His or water.

    A clean Cys/His site coordinates Zn only through Cys/His (and, tolerated,
    water).  A COORD=TRUE GLU/ASP/SO4/IMP/… means it's a mixed-ligand site, not
    the intended coordination, so it's rejected.  Returns (kept, dropped
    [(id, ligand_names), ...]).
    """
    kept    = [s for s in structures if not s.extra_coord_ligands]
    dropped = [(s.id, s.extra_coord_ligands) for s in structures
               if s.extra_coord_ligands]
    return kept, dropped


def gather_structures(xyz_dir: Path, glob_pat: str = "*.xyz",
                      desc: str = "parsing", show_progress: bool = True) -> tuple:
    """List → parse → reject mixed-ligand → modal coord-count → majority-composition.

    Parsing keeps only the coordinating Cys/His arms (waters and non-coordinating
    pocket atoms are discarded); structures with a *coordinating* non-Cys/His
    ligand are rejected outright; the modal filter then measures the true
    coordination.  Returns (structures, report).  `report` keys: n_listed,
    modal_atom_count, dropped_extra_ligand, dropped_atomcount, n_parse_fail,
    dropped_composition, n_kept, composition.
    """
    files = list_structure_files(xyz_dir, glob_pat)
    n_listed = len(files)
    it = tqdm.tqdm(files, desc=desc, leave=False) if show_progress else files
    parsed = [s for p in it if (s := parse_structure(p)) is not None]
    n_parse_fail = len(files) - len(parsed)
    parsed, dropped_extra = filter_extra_coord_ligands(parsed)
    structures, dropped_atomcount, mode = filter_by_modal_coord_count(parsed)
    structures, dropped_comp = filter_majority_composition(structures)
    report = {
        "n_listed": n_listed,
        "modal_atom_count": mode,
        "dropped_extra_ligand": dropped_extra,
        "dropped_atomcount": dropped_atomcount,
        "n_parse_fail": n_parse_fail,
        "dropped_composition": dropped_comp,
        "n_kept": len(structures),
        "composition": structures[0].composition() if structures else None,
    }
    return structures, report


def print_gather_report(report: dict) -> None:
    """Print a human-readable summary of gather_structures()."""
    extra = report.get("dropped_extra_ligand") or []
    if extra:
        ligands = sorted({lig for _, ligs in extra for lig in ligs})
        print(f"  {len(extra)} structures dropped "
              f"(extra coordinating ligand: {', '.join(ligands)})")
    if report["dropped_atomcount"]:
        print(f"  {len(report['dropped_atomcount'])} structures dropped "
              f"(coordinating-atom count ≠ modal {report['modal_atom_count']})")
    if report["n_parse_fail"]:
        print(f"  {report['n_parse_fail']} files failed to parse")
    if report["dropped_composition"]:
        print(f"  {len(report['dropped_composition'])} structures dropped "
              f"(composition ≠ majority)")
    comp = report["composition"]
    print(f"Kept {report['n_kept']} structures" + (f"  Composition: {comp}" if comp else ""))


def filter_majority_composition(structures: list) -> tuple:
    """Keep only structures whose composition matches the most common one.

    All matching/alignment requires a single shared composition (same number of
    Cys and His).  Datasets are organised per composition, so this normally keeps
    everything; it guards against stray files.  Returns (kept, dropped).
    """
    if not structures:
        return [], []
    counts: dict = defaultdict(int)
    for s in structures:
        counts[s.composition()] += 1
    majority = max(counts, key=lambda c: counts[c])
    kept = [s for s in structures if s.composition() == majority]
    dropped = [s for s in structures if s.composition() != majority]
    return kept, dropped


# ---------------------------------------------------------------------------
# Clustering pipeline
# ---------------------------------------------------------------------------

def cluster_pipeline(
    X: np.ndarray,
    k: int,
    variance: float = 0.95,
    var_floor: float = 1e-3,
    random_state: int = 0,
    n_init: int = 10,
    clustering_mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, object, tuple]:
    """Z-score → PCA (95% var) → k-means.

    Returns (labels, centroids_pca, pca_model, (means, stds)).
    centroids_pca[c] is the k-means centroid for cluster c in PCA space.
    """
    Xu = X[:, clustering_mask] if clustering_mask is not None else X
    means = Xu.mean(0)
    stds  = np.maximum(Xu.std(0), var_floor)
    Xs = (Xu - means) / stds

    pca_full = PCA(random_state=random_state)
    pca_full.fit(Xs)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_comp = min(max(1, int(np.searchsorted(cumvar, variance)) + 1), Xs.shape[1])

    pca = PCA(n_components=n_comp, random_state=random_state)
    X_pca = pca.fit_transform(Xs)

    km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
    labels = km.fit_predict(X_pca)

    return labels, km.cluster_centers_, pca, (means, stds), pca_full.explained_variance_ratio_

# ---------------------------------------------------------------------------
# Evaluation metric
# ---------------------------------------------------------------------------

def evaluate_clustering(
    structures: list[Structure],
    labels: np.ndarray,
    X_pca: np.ndarray,
    centroids_pca: np.ndarray,
    w_type: dict | str,
    allow_reflection: bool = True,
    matcher=None,
) -> dict:
    """Intra/inter structural-RMSD evaluation.

    Returns dict with keys: k, intra, inter, ratio, ch_score, medoid_ids, per_cluster_intra.
    ch_score = (inter²/(k−1)) / (intra²/(N−k))  — Calinski-Harabasz analogue for RMSD.
    """
    unique = sorted(set(int(l) for l in labels))

    # Medoid = member nearest centroid in PCA space
    medoid_idx: dict[int, int] = {}
    for c in unique:
        mask = labels == c
        idxs = np.where(mask)[0]
        centroid = centroids_pca[c] if c < len(centroids_pca) else X_pca[mask].mean(0)
        medoid_idx[c] = int(idxs[np.argmin(np.linalg.norm(X_pca[mask] - centroid, axis=1))])

    # Intra: mean structural_rmsd(medoid, member) per cluster
    per_cluster_intra: list[float] = []
    for c in unique:
        idxs = np.where(labels == c)[0]
        med = structures[medoid_idx[c]]
        rmsds = [structural_rmsd(med, structures[i], w_type, allow_reflection, matcher)[0]
                 for i in idxs if i != medoid_idx[c]]
        per_cluster_intra.append(float(np.mean(rmsds)) if rmsds else 0.0)

    intra = float(np.mean(per_cluster_intra)) if per_cluster_intra else 0.0

    # Inter: mean structural_rmsd between every medoid pair
    meds = [structures[medoid_idx[c]] for c in unique]
    inter_vals = [structural_rmsd(meds[i], meds[j], w_type, allow_reflection, matcher)[0]
                  for i, j in combinations(range(len(meds)), 2)]
    inter = float(np.mean(inter_vals)) if inter_vals else 0.0
    ratio = inter / intra if intra > 0 else 0.0

    # Calinski-Harabasz-analogue score (RMSD-based):
    #   ch_score = (inter² / (k-1)) / (intra² / (N-k))
    # Penalizes large k via (k-1) in denominator; peaks at a genuinely good k.
    N = len(structures)
    k = len(unique)
    ch_denom = intra ** 2 * (k - 1)
    ch_score = ((inter ** 2 * (N - k)) / ch_denom) if (ch_denom > 0 and N > k) else 0.0

    return {
        "k": k,
        "intra": intra,
        "inter": inter,
        "ratio": ratio,
        "ch_score": ch_score,
        "medoid_ids": {c: structures[medoid_idx[c]].id for c in unique},
        "per_cluster_intra": per_cluster_intra,
    }

# ---------------------------------------------------------------------------
# k sweep
# ---------------------------------------------------------------------------

def sweep_k(
    structures: list[Structure],
    X: np.ndarray,
    k_values: list[int],
    w_type: dict | str,
    allow_reflection: bool = True,
    clustering_mask: Optional[np.ndarray] = None,
    desc: str = "k sweep",
    matcher=None,
) -> tuple[Optional[dict], list[dict]]:
    """Run cluster_pipeline + evaluate_clustering for each valid k.

    Returns (best_result, all_results).  Each result dict includes labels, X_pca,
    pca, scale in addition to the evaluation fields.
    """
    valid_ks = [k for k in k_values if 2 <= k < len(structures)]
    if not valid_ks:
        return None, []

    results: list[dict] = []
    best_ch = -math.inf
    best: Optional[dict] = None

    for k in tqdm.tqdm(valid_ks, desc=desc, leave=True):
        labels, centroids, pca, scale, evr_full = cluster_pipeline(X, k, clustering_mask=clustering_mask)
        means, stds = scale
        Xu = X[:, clustering_mask] if clustering_mask is not None else X
        X_pca = pca.transform((Xu - means) / stds)

        ev = evaluate_clustering(structures, labels, X_pca, centroids, w_type,
                                 allow_reflection, matcher)
        ev.update({"labels": labels, "X_pca": X_pca, "pca": pca, "scale": scale, "evr_full": evr_full})
        results.append(ev)

        if ev["ch_score"] > best_ch:
            best_ch = ev["ch_score"]
            best = ev

    return best, results

# ---------------------------------------------------------------------------
# Standard output persistence
# ---------------------------------------------------------------------------

def save_outputs(
    out_dir: Path,
    id_list: list[str],
    k_sweep_table: list[dict],
    best: dict,
) -> None:
    """Write labels.csv, medoids.csv, k_sweep.csv, per_cluster_intra.csv."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # k sweep table
    with (out_dir / "k_sweep.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["k", "intra", "inter", "ratio", "ch_score"])
        w.writeheader()
        for r in k_sweep_table:
            w.writerow({"k": r["k"], "intra": f"{r['intra']:.6f}",
                        "inter": f"{r['inter']:.6f}", "ratio": f"{r['ratio']:.6f}",
                        "ch_score": f"{r['ch_score']:.6f}"})

    # Labels
    labels = best["labels"]
    with (out_dir / "labels.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["structure_id", "cluster"])
        for sid, lbl in zip(id_list, labels):
            w.writerow([sid, int(lbl)])

    # Medoids
    with (out_dir / "medoids.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["cluster_id", "medoid_id"])
        for cid, mid in best["medoid_ids"].items():
            w.writerow([int(cid), mid])

    # Per-cluster intra
    with (out_dir / "per_cluster_intra.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["cluster_id", "mean_intra_rmsd"])
        for cid, val in enumerate(best["per_cluster_intra"]):
            w.writerow([cid, f"{val:.6f}"])

    # PCA scree data (full EVR from best-k run; same feature matrix for all k)
    evr_full = best.get("evr_full")
    if evr_full is not None:
        n_retained = best["pca"].n_components_
        cumvar = np.cumsum(evr_full)
        with (out_dir / "pca_scree.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["component", "explained_variance_ratio", "cumulative_variance_ratio", "retained"])
            for i, (evr, cum) in enumerate(zip(evr_full, cumvar)):
                w.writerow([i + 1, f"{evr:.6f}", f"{cum:.6f}", i < n_retained])

    # t-SNE embeddings
    save_embeddings_and_tsne(out_dir, id_list, best["X_pca"], best["labels"])


# ---------------------------------------------------------------------------
# t-SNE embeddings
# ---------------------------------------------------------------------------

def save_embeddings_and_tsne(
    out_dir: Path,
    id_list: list[str],
    X_pca: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Run t-SNE on PCA embeddings; write embeddings.csv and tsne_kmeans.png.

    embeddings.csv columns: id, pc1…pcN, tsne1, tsne2
    tsne_kmeans.png: scatter colored by cluster with tab20.
    Returns the (N, 2) t-SNE coordinate array.
    """
    try:
        from sklearn.manifold import TSNE
    except ImportError:
        print("  t-SNE skipped (scikit-learn TSNE not available)")
        return np.zeros((len(id_list), 2))

    N = len(id_list)
    perplexity = float(min(30, max(5, N // 4)))
    print(f"  Running t-SNE (n={N}, perplexity={perplexity:.0f}) …")
    X_tsne = TSNE(n_components=2, perplexity=perplexity, random_state=0).fit_transform(X_pca)

    n_pca = X_pca.shape[1]
    emb_path = out_dir / "embeddings.csv"
    with emb_path.open("w", newline="", encoding="utf-8") as fh:
        cw = csv.writer(fh)
        cw.writerow(["id"] + [f"pc{i+1}" for i in range(n_pca)] + ["tsne1", "tsne2"])
        for sid, pca_row, (t1, t2) in zip(id_list, X_pca, X_tsne):
            cw.writerow([sid] + [f"{v:.8f}" for v in pca_row] + [f"{t1:.8f}", f"{t2:.8f}"])
    print(f"  Embeddings → {emb_path}")

    if not _MPL_OK:
        return X_tsne

    unique_labels = sorted(set(int(l) for l in labels))
    cmap = plt.get_cmap("tab20")

    fig, ax = plt.subplots(figsize=(9, 7))
    for i, c in enumerate(unique_labels):
        mask = labels == c
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                   c=[cmap(i % 20)], s=15, alpha=0.7, label=str(c))

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(f"t-SNE of PCA space  (k={len(unique_labels)})")
    ncol = max(1, len(unique_labels) // 20)
    ax.legend(title="cluster", bbox_to_anchor=(1.02, 1), loc="upper left",
              fontsize=7, ncol=ncol, markerscale=1.5)
    fig.tight_layout()
    png_path = out_dir / "tsne_kmeans.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  t-SNE plot → {png_path}")

    return X_tsne


# ---------------------------------------------------------------------------
# Cluster distribution plots (requires per-structure stats CSV)
# ---------------------------------------------------------------------------

# Numeric metrics to extract from the stats CSV and plot.
# Each entry: (column_name, display_label) — labels use matplotlib mathtext.
_NUMERIC_PLOT_METRICS: list[tuple[str, str]] = [
    ("volume_A3",              r"Volume ($\AA^3$)"),
    ("q_tetra_coord",          r"$q_\mathrm{tetra}$ (coord)"),
    ("q_tetra_ca",             r"$q_\mathrm{tetra}$ ($C_\alpha$)"),
    ("r_work",                 r"$R_\mathrm{work}$"),
    ("r_free",                 r"$R_\mathrm{free}$"),
    ("zn_bfactor",             r"Zn $B$-factor"),
    ("cys_dihedral_mean_deg",  r"Dihedral mean ($^\circ$)"),
    ("all_coord_res_bfactor_avg", r"Coord-res $\bar{B}$"),
]

# Summary CSV column name for "all_dihedrals_deg" uses cys_dihedral_mean_deg as source.
_DIHEDRAL_COL = "cys_dihedral_mean_deg"
_DIHEDRAL_SUMMARY_KEY = "all_dihedrals_deg"

# Font sizes used in cluster distribution plots.
_FS_LABEL  = 13
_FS_TICK   = 11
_FS_TITLE  = 14
_FS_SUP    = 15
_FS_ANNOT  = 9


def _tab20_hex(idx: int) -> str:
    """Return tab20 color at position idx % 20 as '#rrggbb'."""
    if not _MPL_OK:
        return "#888888"
    rgba = plt.get_cmap("tab20")(idx % 20)
    return "#{:02x}{:02x}{:02x}".format(
        int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255)
    )


def _safe_float(v: str | None) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _coord_res_bfactor_avg(row: dict) -> Optional[float]:
    """Average of per-residue coordinating-residue B-factors."""
    vals = []
    for i in range(1, 5):
        v = _safe_float(row.get(f"coord_cys_{i}_bfactor_avg"))
        if v is not None:
            vals.append(v)
        v = _safe_float(row.get(f"coord_his_{i}_bfactor_avg"))
        if v is not None:
            vals.append(v)
    return float(np.mean(vals)) if vals else None


def _merge_labels_stats(
    id_list: list[str],
    labels: np.ndarray,
    stats_csv: Path,
) -> tuple[list[dict], dict[int, str]]:
    """Read stats CSV, join with labels; return (rows, color_by_cluster).

    Each row has: id, cluster (int), cluster_color (hex), has_stats (0/1),
    plus all columns from the stats CSV, plus all_coord_res_bfactor_avg.
    """
    stats_by_id: dict[str, dict] = {}
    with stats_csv.open(newline="", encoding="utf-8") as fh:
        for srow in csv.DictReader(fh):
            sid = (srow.get("id") or "").strip()
            if sid:
                stats_by_id[sid] = srow

    unique_clusters = sorted(set(int(l) for l in labels))
    color_by_cluster: dict[int, str] = {c: _tab20_hex(i) for i, c in enumerate(unique_clusters)}

    rows: list[dict] = []
    for sid, label in zip(id_list, labels):
        c = int(label)
        srow = stats_by_id.get(sid, {})
        has_stats = 1 if srow else 0

        merged: dict = {
            "id": sid,
            "cluster": c,
            "cluster_color": color_by_cluster[c],
            "has_stats": has_stats,
        }
        merged.update(srow)

        if has_stats:
            # Zn(Cys/His)-only aggregate; omit the column entirely when there are
            # no coord_cys_/coord_his_ B-factors (e.g. heme) so it doesn't surface
            # as an empty metric panel downstream.
            avg = _coord_res_bfactor_avg(srow)
            if avg is not None:
                merged["all_coord_res_bfactor_avg"] = f"{avg:.4f}"

        rows.append(merged)

    return rows, color_by_cluster


def _write_labels_with_stats(out_dir: Path, rows: list[dict]) -> None:
    if not rows:
        return
    all_keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)
    path = out_dir / "kmeans_labels_with_stats.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  Labels+stats → {path}")


def _write_cluster_pdb_family(out_dir: Path, rows: list[dict]) -> None:
    """Write cluster_pdb_family.csv: (cluster, pdb_id, family) sorted by both.

    Auto-generated for every approach (formerly the standalone
    extract_cluster_pdb_family helper).  pdb_id is the 4-char RCSB code.
    """
    if not rows:
        return
    out_rows = [
        {
            "cluster": int(r["cluster"]),
            "pdb_id": (r.get("id") or "")[:4],
            "family": r.get("family", ""),
        }
        for r in rows
    ]
    out_rows.sort(key=lambda r: (r["cluster"], r["pdb_id"]))
    path = out_dir / "cluster_pdb_family.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["cluster", "pdb_id", "family"])
        w.writeheader()
        w.writerows(out_rows)
    print(f"  Cluster/PDB/family → {path}")


def _compute_stats_summary(rows: list[dict], unique_clusters: list[int],
                           metrics: Optional[list[tuple[str, str]]] = None) -> list[dict]:
    """Per-cluster aggregated statistics for numeric metrics."""
    if metrics is None:
        metrics = _NUMERIC_PLOT_METRICS
    rows_by_cluster: dict[int, list[dict]] = {c: [] for c in unique_clusters}
    for row in rows:
        rows_by_cluster[int(row["cluster"])].append(row)

    summary_cols = ["cluster", "n_total", "n_with_stats"]
    numeric_metrics = [col for col, _ in metrics]
    for m in numeric_metrics:
        key = _DIHEDRAL_SUMMARY_KEY if m == _DIHEDRAL_COL else m
        summary_cols += [f"{key}_n", f"{key}_mean", f"{key}_std",
                         f"{key}_min", f"{key}_q25", f"{key}_median",
                         f"{key}_q75", f"{key}_max"]

    summary: list[dict] = []
    for c in unique_clusters:
        cluster_rows = rows_by_cluster[c]
        n_total = len(cluster_rows)
        n_with_stats = sum(int(r.get("has_stats", 0) or 0) for r in cluster_rows)
        srow: dict = {"cluster": c, "n_total": n_total, "n_with_stats": n_with_stats}

        for col, _ in metrics:
            key = _DIHEDRAL_SUMMARY_KEY if col == _DIHEDRAL_COL else col
            vals = [v for r in cluster_rows if (v := _safe_float(r.get(col))) is not None]
            if vals:
                arr = np.array(vals)
                srow[f"{key}_n"] = len(arr)
                srow[f"{key}_mean"] = f"{arr.mean():.6f}"
                srow[f"{key}_std"]  = f"{arr.std():.6f}"
                srow[f"{key}_min"]  = f"{arr.min():.6f}"
                srow[f"{key}_q25"]  = f"{np.percentile(arr, 25):.6f}"
                srow[f"{key}_median"] = f"{np.median(arr):.6f}"
                srow[f"{key}_q75"]  = f"{np.percentile(arr, 75):.6f}"
                srow[f"{key}_max"]  = f"{arr.max():.6f}"
            else:
                for suffix in ("n", "mean", "std", "min", "q25", "median", "q75", "max"):
                    srow[f"{key}_{suffix}"] = ""

        summary.append(srow)
    return summary


def _write_stats_summary(out_dir: Path, summary: list[dict]) -> None:
    if not summary:
        return
    all_keys: list[str] = []
    seen: set[str] = set()
    for row in summary:
        for k in row:
            if k not in seen:
                all_keys.append(k)
                seen.add(k)
    path = out_dir / "kmeans_cluster_stats_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(summary)
    print(f"  Cluster stats summary → {path}")


def _plot_per_cluster_rows(
    rows: list[dict],
    unique_clusters: list[int],
    color_by_cluster: dict[int, str],
    out_dir: Path,
    metrics: Optional[list[tuple[str, str]]] = None,
) -> None:
    if not _MPL_OK:
        return
    if metrics is None:
        metrics = _NUMERIC_PLOT_METRICS
    out_dir.mkdir(parents=True, exist_ok=True)

    from collections import Counter

    # Pre-compute full-dataset values for background histograms.
    all_vals: dict[str, list[float]] = {col: [] for col, _ in metrics}
    all_family: list[str] = []
    for row in rows:
        for col, _ in metrics:
            v = _safe_float(row.get(col))
            if v is not None:
                all_vals[col].append(v)
        f = (row.get("family") or "").strip()
        if f:
            all_family.append(f)

    N_all = len(rows)
    all_families_sorted = sorted(set(all_family))
    has_family = bool(all_families_sorted)
    metrics_with_label = list(metrics) + (
        [("family", "Family")] if has_family else []
    )
    n_metrics = len(metrics_with_label)
    ncols = min(4, n_metrics)
    nrows_fig = math.ceil(n_metrics / ncols)

    for c in unique_clusters:
        cluster_rows = [r for r in rows if int(r["cluster"]) == c]
        color = color_by_cluster[c]

        fig, axes = plt.subplots(nrows_fig, ncols,
                                 figsize=(ncols * 4.5, nrows_fig * 3.5),
                                 squeeze=False)
        fig.suptitle(f"Cluster {c}  (n={len(cluster_rows)})", fontsize=_FS_SUP, fontweight="bold")

        for idx, (col, label) in enumerate(metrics_with_label):
            ax = axes[idx // ncols][idx % ncols]

            if col == "family":
                c_fam = [r.get("family", "").strip() for r in cluster_rows
                         if r.get("family", "").strip()]
                counts = Counter(c_fam)
                if counts:
                    # Show top-20 families; sort by count descending.
                    fams = sorted(counts, key=lambda x: counts[x], reverse=True)[:20]
                    xs = range(len(fams))
                    ax.bar(xs, [counts[f] for f in fams], color=color, alpha=0.85)
                    ax.set_xticks(list(xs))
                    ax.set_xticklabels(fams, rotation=90, ha="center", fontsize=7)
                    # Annotate dominant family in top-right corner.
                    ax.text(0.98, 0.97, fams[0], transform=ax.transAxes,
                            ha="right", va="top", fontsize=_FS_ANNOT,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                      edgecolor="#cccccc", alpha=0.85))
                ax.set_ylabel("count", fontsize=_FS_LABEL)
                ax.set_title("Family", fontsize=_FS_TITLE)
                ax.tick_params(axis="y", labelsize=_FS_TICK)
            else:
                cluster_vals = [v for r in cluster_rows
                                if (v := _safe_float(r.get(col))) is not None]
                overall = all_vals.get(col, [])
                N_col = len(overall)
                if N_col > 0:
                    bins = min(30, max(5, N_col // 10))
                    lo, hi = float(np.min(overall)), float(np.max(overall))
                    if lo == hi:
                        lo, hi = lo - 0.5, hi + 0.5
                    bin_edges = np.linspace(lo, hi, bins + 1)
                    # Both histograms normalised to fraction of N_col so axes match.
                    ax.hist(overall, bins=bin_edges, color="#cccccc", alpha=0.65,
                            weights=np.ones(N_col) / N_col, label="all")
                    if cluster_vals:
                        ax.hist(cluster_vals, bins=bin_edges, color=color, alpha=0.80,
                                weights=np.ones(len(cluster_vals)) / N_col, label=f"c{c}")
                ax.set_xlabel(label, fontsize=_FS_LABEL)
                ax.set_ylabel("fraction of total", fontsize=_FS_LABEL)
                ax.set_title(label, fontsize=_FS_TITLE)
                ax.tick_params(labelsize=_FS_TICK)

        for idx in range(n_metrics, nrows_fig * ncols):
            axes[idx // ncols][idx % ncols].set_visible(False)

        fig.tight_layout()
        fig.savefig(out_dir / f"cluster_{c}_metrics_row.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

    print(f"  Per-cluster row plots → {out_dir}")


def _plot_overlay_metrics(
    rows: list[dict],
    unique_clusters: list[int],
    color_by_cluster: dict[int, str],
    out_dir: Path,
    metrics: Optional[list[tuple[str, str]]] = None,
) -> None:
    """Side-by-side per-cluster subplots for each metric (one file per metric)."""
    if not _MPL_OK:
        return
    if metrics is None:
        metrics = _NUMERIC_PLOT_METRICS
    out_dir.mkdir(parents=True, exist_ok=True)

    from collections import Counter

    rows_by_cluster: dict[int, list[dict]] = {c: [] for c in unique_clusters}
    for row in rows:
        rows_by_cluster[int(row["cluster"])].append(row)

    n_c = len(unique_clusters)
    # Layout: up to 4 columns, wrap to multiple rows.
    ncols_fig = min(4, n_c)
    nrows_fig = math.ceil(n_c / ncols_fig)

    def _make_axes_grid(n_clusters: int, subplot_w: float = 4.0, subplot_h: float = 3.5):
        """Return (fig, axes_flat) with shared x/y axes."""
        nc = min(4, n_clusters)
        nr = math.ceil(n_clusters / nc)
        fig, axes = plt.subplots(
            nr, nc,
            figsize=(nc * subplot_w, nr * subplot_h),
            sharey=True, sharex=True,
            squeeze=False,
        )
        return fig, axes, nr, nc

    # ── Numeric side-by-side overlays ──────────────────────────────────────
    for col, label in metrics:
        all_vals = [v for row in rows if (v := _safe_float(row.get(col))) is not None]
        if not all_vals:
            continue
        N_all = len(all_vals)
        lo, hi = float(np.min(all_vals)), float(np.max(all_vals))
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
        bins = min(30, max(5, N_all // 10))
        bin_edges = np.linspace(lo, hi, bins + 1)

        fig, axes, nr, nc = _make_axes_grid(n_c)
        fig.suptitle(f"{label} — all clusters", fontsize=_FS_SUP, fontweight="bold")

        for i, c in enumerate(unique_clusters):
            ax = axes[i // nc][i % nc]
            cvals = [v for r in rows_by_cluster[c]
                     if (v := _safe_float(r.get(col))) is not None]
            # Gray = all data, both normalised to fraction of N_all.
            ax.hist(all_vals, bins=bin_edges, color="#cccccc", alpha=0.65,
                    weights=np.ones(N_all) / N_all, label="all")
            if cvals:
                ax.hist(cvals, bins=bin_edges, color=color_by_cluster[c], alpha=0.80,
                        weights=np.ones(len(cvals)) / N_all, label=f"c{c}")
            ax.set_title(f"Cluster {c}  (n={len(rows_by_cluster[c])})",
                         fontsize=_FS_TITLE)
            ax.set_xlabel(label, fontsize=_FS_LABEL)
            if i % nc == 0:
                ax.set_ylabel("fraction of total", fontsize=_FS_LABEL)
            ax.tick_params(labelsize=_FS_TICK)

        # Hide empty panels.
        for idx in range(n_c, nr * nc):
            axes[idx // nc][idx % nc].set_visible(False)

        fig.tight_layout()
        metric_key = _DIHEDRAL_SUMMARY_KEY if col == _DIHEDRAL_COL else col
        fig.savefig(out_dir / f"{metric_key}_all_clusters_overlay.png",
                    dpi=130, bbox_inches="tight")
        plt.close(fig)

    # ── Family side-by-side overlays ────────────────────────────────────────
    _MAX_FAM_OVERLAY = 30  # show top-N families in the overlay figure
    all_family_pairs = [(int(row["cluster"]), (row.get("family") or "").strip())
                        for row in rows if (row.get("family") or "").strip()]
    if all_family_pairs:
        # Keep only the top families across all clusters.
        top_fam_overall = [f for f, _ in Counter(fam for _, fam in all_family_pairs)
                           .most_common(_MAX_FAM_OVERLAY)]
        all_families = sorted(top_fam_overall)
        n_fam = len(all_families)
        xs = np.arange(n_fam)
        sp_w = max(4.0, n_fam * 0.45 + 1.5)

        fig, axes, nr, nc = _make_axes_grid(n_c, subplot_w=sp_w, subplot_h=4.0)
        fig.suptitle("Family — all clusters", fontsize=_FS_SUP, fontweight="bold")

        for i, c in enumerate(unique_clusters):
            ax = axes[i // nc][i % nc]
            c_fams = [fam for cl, fam in all_family_pairs if cl == c]
            counts = Counter(c_fams)
            ax.bar(xs, [counts.get(f, 0) for f in all_families],
                   color=color_by_cluster[c], alpha=0.85)
            ax.set_xticks(list(xs))
            ax.set_xticklabels(all_families, rotation=90, ha="center", fontsize=7)
            ax.set_title(f"Cluster {c}  (n={len(rows_by_cluster[c])})",
                         fontsize=_FS_TITLE)
            if i % nc == 0:
                ax.set_ylabel("count", fontsize=_FS_LABEL)
            ax.tick_params(axis="y", labelsize=_FS_TICK)

        for idx in range(n_c, nr * nc):
            axes[idx // nc][idx % nc].set_visible(False)

        fig.tight_layout()
        fig.savefig(out_dir / "family_all_clusters_overlay.png",
                    dpi=130, bbox_inches="tight")
        plt.close(fig)

    print(f"  Overlay plots → {out_dir}")


def write_minimal_labels_with_stats(
    out_dir: Path,
    id_list: list[str],
    labels: np.ndarray,
    family_by_id: Optional[dict[str, str]] = None,
) -> None:
    """Write a metric-free kmeans_labels_with_stats.csv for profiles without stats.

    Columns: id, cluster, cluster_color, pdb_id, family.  This is the minimum the
    Streamlit validation tab needs (t-SNE scatter + optional family bars); the
    numeric-metric panels there filter to whatever columns exist, so their absence
    is handled gracefully.  cluster_color matches the tab20 scheme used elsewhere.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    unique = sorted(set(int(l) for l in labels))
    color = {c: _tab20_hex(i) for i, c in enumerate(unique)}
    family_by_id = family_by_id or {}
    path = out_dir / "kmeans_labels_with_stats.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "cluster", "cluster_color", "pdb_id", "family"])
        for sid, lbl in zip(id_list, labels):
            c = int(lbl)
            w.writerow([sid, c, color[c], sid[:4], family_by_id.get(sid, "")])
    print(f"  Minimal labels+stats → {path}")


def build_cluster_distribution_plots(
    out_dir: Path,
    id_list: list[str],
    labels: np.ndarray,
    stats_csv: Optional[Path],
    metrics: Optional[list[tuple[str, str]]] = None,
) -> None:
    """Generate kmeans_labels_with_stats.csv, stats summary, and histogram PNGs.

    Requires stats_csv (per-structure metadata CSV with an 'id' column).
    Silently skips if stats_csv is None or does not exist.  ``metrics`` (list of
    (column, label)) selects which numeric metrics to bin; defaults to the
    Zn(Cys/His) set.  Non-Zn profiles pass their own smaller set (e.g. heme:
    r_work, r_free, avg B-factor).
    """
    if stats_csv is None or not stats_csv.is_file():
        print(f"  Distribution plots skipped (no stats CSV)")
        return
    if metrics is None:
        metrics = _NUMERIC_PLOT_METRICS

    print(f"\nBuilding cluster distribution plots from {stats_csv.name} …")
    rows, color_by_cluster = _merge_labels_stats(id_list, labels, stats_csv)
    unique_clusters = sorted(set(int(l) for l in labels))

    _write_labels_with_stats(out_dir, rows)
    _write_cluster_pdb_family(out_dir, rows)

    summary = _compute_stats_summary(rows, unique_clusters, metrics)
    _write_stats_summary(out_dir, summary)

    plots_dir = out_dir / "cluster_distribution_plots"
    # Clear stale PNGs so old cluster counts don't bleed into the new report.
    for _subdir in [plots_dir / "per_cluster_rows", plots_dir / "all_cluster_overlays"]:
        if _subdir.is_dir():
            for _f in _subdir.glob("*.png"):
                _f.unlink()
    _plot_per_cluster_rows(rows, unique_clusters, color_by_cluster,
                           plots_dir / "per_cluster_rows", metrics)
    _plot_overlay_metrics(rows, unique_clusters, color_by_cluster,
                          plots_dir / "all_cluster_overlays", metrics)
    print("  Distribution plots done.")
