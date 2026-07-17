#!/usr/bin/env python3
"""Compute per-structure stats from XYZ files and (optionally) PDB files.

For each XYZ file in --xyz-dir computes:

  Geometry (from XYZ coordinates, always):
    volume_A3             Cα tetrahedron volume
    q_tetra_coord         Errington-Debenedetti q using coordinating atoms (SG)
    q_tetra_ca            Same metric using Cα positions
    cys_dihedral_N_deg    Zn→SG→Cβ→Cα dihedral for residue N (N=1..4, sorted by RESSEQ)
    cys_dihedral_mean_deg Mean of the 4 dihedrals

  From XYZ header:
    resolution_A          RESOLUTION_A field in header line

  From PDB (requires --pdb-dir, skipped gracefully otherwise):
    r_work                R VALUE (WORKING SET) from REMARK 3
    r_free                FREE R VALUE from REMARK 3
    zn_bfactor            B-factor of the Zn HETATM record
    coord_cys_N_bfactor_avg  mean(CA, CB, SG) B-factors for coordinating CYS N

Output: --out-csv  (default: <xyz-dir>/structure_stats.csv)

Skips if output already exists unless --force.

Usage
-----
  uv run python -m zn_cys_his.clustering.step02_compute_stats \\
      --xyz-dir  data/1cys3his-large/xyz-files \\
      --pdb-dir  data/4cys-large/pdb-files

  uv run python -m zn_cys_his.clustering.step02_compute_stats \\
      --xyz-dir  data/1cys3his-large/xyz-files \\
      --pdb-dir  data/4cys-large/pdb-files \\
      --force
"""
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import tqdm

from zn_cys_his.clustering.utils import (parse_structure, Structure,
                   list_structure_files, gather_structures, print_gather_report)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _q_tetra(points: np.ndarray, center: np.ndarray) -> Optional[float]:
    """Errington-Debenedetti tetrahedral order parameter.

    q = 1 − (3/8) · Σ_{i<j} (cos θ_ij + 1/3)²
    where θ_ij is the angle center–pointi–pointj (at center).
    Returns None if any point coincides with center.
    """
    if len(points) != 4:
        return None
    unit_vecs = []
    for p in points:
        v = p - center
        n = float(np.linalg.norm(v))
        if n == 0.0:
            return None
        unit_vecs.append(v / n)
    total = 0.0
    for i in range(4):
        for j in range(i + 1, 4):
            cos_t = float(np.dot(unit_vecs[i], unit_vecs[j]))
            cos_t = max(-1.0, min(1.0, cos_t))
            total += (cos_t + 1.0 / 3.0) ** 2
    return 1.0 - (3.0 / 8.0) * total


def _dihedral_deg(p0: np.ndarray, p1: np.ndarray,
                  p2: np.ndarray, p3: np.ndarray) -> Optional[float]:
    """Dihedral angle p0–p1–p2–p3 in degrees (−180 to +180)."""
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2
    n1n = float(np.linalg.norm(b1))
    n2n = float(np.linalg.norm(b2))
    n3n = float(np.linalg.norm(b3))
    if n1n < 1e-10 or n2n < 1e-10 or n3n < 1e-10:
        return None
    b2u = b2 / n2n
    v = b1 - np.dot(b1, b2u) * b2u
    w = b3 - np.dot(b3, b2u) * b2u
    nv, nw = np.linalg.norm(v), np.linalg.norm(w)
    if nv < 1e-10 or nw < 1e-10:
        return None
    cos_t = float(np.dot(v, w) / (nv * nw))
    cos_t = max(-1.0, min(1.0, cos_t))
    angle = math.degrees(math.acos(cos_t))
    if np.dot(np.cross(v, w), b2u) < 0:
        angle = -angle
    return angle


def compute_geometry(s: Structure) -> dict:
    """Compute all geometry stats from a Structure (any Cys/His composition)."""
    zn = s.zn

    ca    = np.array([r.coords["CA"] for r in s.residues])            # (n, 3)
    coord = np.array([r.coords[r.coord_atom] for r in s.residues])    # ligands

    # Cα tetrahedron volume (only well-defined for 4 coordinating residues)
    if len(ca) == 4:
        v1, v2, v3 = ca[1] - ca[0], ca[2] - ca[0], ca[3] - ca[0]
        volume = f"{abs(float(np.dot(v1, np.cross(v2, v3)))) / 6.0:.4f}"
    else:
        volume = ""

    # q_tetra (coordinating atoms — SG/ND1/NE2 — then Cα)
    q_coord = _q_tetra(coord, zn)
    q_ca    = _q_tetra(ca, zn)

    # Zn→ligand→Cβ→Cα dihedral per residue (ligand = SG for Cys, ND1/NE2 for His)
    dihedrals: list[Optional[float]] = [
        _dihedral_deg(zn, r.coords[r.coord_atom], r.coords["CB"], r.coords["CA"])
        for r in s.residues
    ]

    valid_d = [d for d in dihedrals if d is not None]
    dihedral_mean = float(np.mean(valid_d)) if valid_d else None

    row: dict = {
        "volume_A3":            volume,
        "q_tetra_coord":        f"{q_coord:.4f}" if q_coord is not None else "",
        "q_tetra_ca":           f"{q_ca:.4f}"    if q_ca    is not None else "",
        "cys_dihedral_mean_deg": f"{dihedral_mean:.2f}" if dihedral_mean is not None else "",
    }
    for i, d in enumerate(dihedrals, start=1):
        row[f"cys_dihedral_{i}_deg"] = f"{d:.2f}" if d is not None else ""

    return row


# ---------------------------------------------------------------------------
# XYZ header / metadata parsing
# ---------------------------------------------------------------------------

_HEADER_RE = re.compile(r"(\w+)=([^\s]+)")

_SEC_ABBREV: dict[str, str] = {"LOOP": "L", "HELIX": "H", "SHEET": "S"}


def _parse_sec(comment: str) -> str:
    """Return SEC field value (LOOP, HELIX, SHEET) from comment, upper-cased."""
    for part in comment.split():
        if part.startswith("SEC="):
            return part[4:].strip().upper()
    return ""


def _parse_xyz_header(path: Path) -> dict:
    """Extract key=value tokens from the XYZ comment line (line 2)."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    if len(lines) < 2:
        return {}
    return dict(_HEADER_RE.findall(lines[1]))


def _parse_xyz_residues(path: Path) -> dict:
    """Collect Zn plus the coordinating residues (Cys and/or His) and their SEC.

    Returns {"zn": (chain, resseq),
             "coord": [(chain, resseq, res_type), ...]  # Cys before His, by RESSEQ
             "coord_sec": {(chain, resseq): sec_string}}
    When COORD=TRUE flags are present only flagged residues are kept; otherwise
    every Cys/His residue is taken.  Only reads heavy atoms (skips H lines).
    """
    result: dict = {"zn": None, "coord": [], "coord_sec": {}}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return result

    # (chain, resseq) -> {"res":.., "sec":.., "flagged":bool}
    seen: dict[tuple[str, int], dict] = {}
    for line in lines[2:]:
        if "#" not in line:
            continue
        comment = line.split("#", 1)[1]
        meta = dict(_HEADER_RE.findall(comment))
        atom = meta.get("ATOM", "").upper()
        if atom in ("H", ""):
            continue
        res   = meta.get("RES", "").upper()
        chain = meta.get("CHAIN", "")
        try:
            resseq = int(meta.get("RESSEQ", ""))
        except (ValueError, TypeError):
            continue
        if res == "ZN" and atom == "ZN":
            result["zn"] = (chain, resseq)
        elif res in ("CYS", "HIS"):
            key = (chain, resseq)
            e = seen.setdefault(key, {"res": res, "sec": "", "flagged": False})
            if meta.get("COORD", "").upper() == "TRUE":
                e["flagged"] = True
            if atom == "CA":
                e["sec"] = _parse_sec(comment)

    any_flag = any(e["flagged"] for e in seen.values())
    _rank = {"CYS": 0, "HIS": 1}
    coord = [(c, rs, e["res"]) for (c, rs), e in seen.items()
             if (e["flagged"] or not any_flag)]
    coord.sort(key=lambda t: (_rank[t[2]], t[1], t[0]))  # match parse_structure order
    result["coord"] = coord
    for (c, rs), e in seen.items():
        result["coord_sec"][(c, rs)] = e["sec"]
    return result


_RES_LETTER = {"CYS": "C", "HIS": "H"}


def compute_family_string(
    coord_keys: list[tuple[str, int, str]],
    coord_sec: dict[tuple[str, int], str],
) -> str:
    """Build family string from a sorted (chain, resseq, res_type) list + SEC map.

    A one-letter code per coordinating residue (C=Cys, H=His), grouped by chain,
    with residue spacing between consecutive positions.
    Single-chain 4-Cys example:  Cx2Cx2Cx8C-LHLL
    Mixed 2-Cys-2-His example:    Cx2Hx8C-Hx3C-... (sequence order within chain)
    """
    if len(coord_keys) != 4:
        return ""

    # Group by chain, ordering residues within a chain by RESSEQ.
    chain_groups: dict[str, list[tuple[int, str, str]]] = {}
    chain_order: list[str] = []
    for chain, resseq, rtype in coord_keys:
        if chain not in chain_groups:
            chain_groups[chain] = []
            chain_order.append(chain)
        sec = coord_sec.get((chain, resseq), "")
        chain_groups[chain].append((resseq, rtype, sec))
    for chain in chain_groups:
        chain_groups[chain].sort()

    c_parts: list[str] = []
    s_parts: list[str] = []
    for chain in sorted(chain_order):
        residues = chain_groups[chain]
        c_str = _RES_LETTER.get(residues[0][1], "?")
        for i in range(len(residues) - 1):
            spacing = residues[i + 1][0] - residues[i][0] - 1
            c_str += f"x{spacing}{_RES_LETTER.get(residues[i + 1][1], '?')}"
        s_str = "".join(_SEC_ABBREV.get(sec, "?") for _, _, sec in residues)
        c_parts.append(c_str)
        s_parts.append(s_str)

    return "-".join(c_parts) + "-" + "-".join(s_parts)


# ---------------------------------------------------------------------------
# PDB parsing
# ---------------------------------------------------------------------------

# Anchor to the start of the REMARK 3 line so the label is the *whole* field
# name.  Otherwise .search() also matches sub-records like
# "BIN R VALUE (WORKING SET)" and "ESTIMATED ERROR OF ... FREE R VALUE",
# whose values are unrelated to the overall R-factors.
_RWORK_RE = re.compile(r"^REMARK\s+3\s+R VALUE\s+\(WORKING SET\)\s*:\s*([\d.]+)")
_RFREE_RE  = re.compile(r"^REMARK\s+3\s+FREE R VALUE\s*:\s*([\d.]+)")


# Sidechain + Cα heavy-atom names whose B-factors are averaged, per residue type.
_BFACTOR_ATOMS = {
    "CYS": {"CA", "CB", "SG"},
    "HIS": {"CA", "CB", "CG", "ND1", "CD2", "CE1", "NE2"},
}


def _parse_pdb_stats(
    pdb_path: Path,
    zn_chain: str,
    zn_resseq: int,
    coord_list: list[tuple[str, int, str]],
) -> dict:
    """Extract R-factors and B-factors from a PDB file.

    Returns dict with r_work, r_free, zn_bfactor, and per coordinating residue
    coord_cys_N_bfactor_avg / coord_his_N_bfactor_avg (N indexes within each
    type, in coord_list order).  Empty strings for any field not found.
    """
    result: dict = {
        "r_work": "", "r_free": "", "zn_bfactor": "",
    }
    for i in range(1, 5):
        result[f"coord_cys_{i}_bfactor_avg"] = ""
        result[f"coord_his_{i}_bfactor_avg"] = ""

    try:
        text = pdb_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return result

    # R-factors from REMARK 3 (take the first match; anchored regexes reject
    # BIN / ESTIMATED ERROR sub-records).
    r_work_val = r_free_val = None
    for line in text.splitlines():
        if not line.startswith("REMARK   3"):
            continue
        if r_work_val is None:
            m = _RWORK_RE.match(line)
            if m:
                r_work_val = m.group(1)
        if r_free_val is None:
            m = _RFREE_RE.match(line)
            if m:
                r_free_val = m.group(1)

    if r_work_val:
        result["r_work"] = r_work_val
    if r_free_val:
        result["r_free"] = r_free_val

    # Atom-level B-factors
    res_type_by_key = {(c, rs): rt for c, rs, rt in coord_list}
    coord_bf: dict[tuple[str, int], dict[str, float]] = {}
    zn_bf: Optional[float] = None

    for line in text.splitlines():
        rec = line[:6].strip()
        if rec not in ("ATOM", "HETATM"):
            continue
        if len(line) < 66:
            continue
        try:
            atom_name = line[12:16].strip()
            chain     = line[21]
            resseq    = int(line[22:26])
            bf_val    = float(line[60:66])
        except (ValueError, IndexError):
            continue

        # Zn
        if rec == "HETATM" and atom_name.upper() in ("ZN", "ZN2+") \
                and chain == zn_chain and resseq == zn_resseq:
            zn_bf = bf_val

        # Coordinating Cys/His sidechain + Cα atoms
        if rec == "ATOM":
            key = (chain, resseq)
            rt = res_type_by_key.get(key)
            if rt is not None and atom_name in _BFACTOR_ATOMS[rt]:
                coord_bf.setdefault(key, {})[atom_name] = bf_val

    if zn_bf is not None:
        result["zn_bfactor"] = f"{zn_bf:.3f}"

    # Number each coordinating residue within its own type (Cys 1..; His 1..).
    type_counter = {"CYS": 0, "HIS": 0}
    for chain, resseq, rtype in coord_list:
        type_counter[rtype] += 1
        i = type_counter[rtype]
        if i > 4:
            continue
        atoms = coord_bf.get((chain, resseq), {})
        vals = list(atoms.values())
        if vals:
            col = "coord_cys" if rtype == "CYS" else "coord_his"
            result[f"{col}_{i}_bfactor_avg"] = f"{sum(vals) / len(vals):.3f}"

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def compute_stats(
    xyz_dir: Path,
    pdb_dir: Optional[Path],
    out_csv: Path,
    force: bool = False,
    glob_pat: str = "*.xyz",
) -> None:
    if out_csv.exists() and not force:
        print(f"Stats CSV already exists: {out_csv}  (use --force to recalculate)")
        return

    # Apply the same selection the clustering scripts use (reject -extended /
    # .pc / .gzmat, drop off-modal coordinating-atom counts and off-composition
    # structures) so stats cover exactly the clustered structures.
    all_files = list_structure_files(xyz_dir, glob_pat)
    if not all_files:
        raise SystemExit(f"No XYZ files found in {xyz_dir}")
    kept_structs, report = gather_structures(xyz_dir, glob_pat, desc="scanning")
    print_gather_report(report)
    kept_ids = {s.id for s in kept_structs}
    xyz_files = [p for p in all_files if p.stem in kept_ids]

    print(f"Computing stats for {len(xyz_files)} XYZ files …")

    FIELDNAMES = [
        "id", "xyz_path",
        "resolution_A",
        "family",
        "volume_A3",
        "q_tetra_coord", "q_tetra_ca",
        "cys_dihedral_mean_deg",
        "cys_dihedral_1_deg", "cys_dihedral_2_deg",
        "cys_dihedral_3_deg", "cys_dihedral_4_deg",
        "r_work", "r_free",
        "zn_bfactor",
        "coord_cys_1_bfactor_avg", "coord_cys_2_bfactor_avg",
        "coord_cys_3_bfactor_avg", "coord_cys_4_bfactor_avg",
        "coord_his_1_bfactor_avg", "coord_his_2_bfactor_avg",
        "coord_his_3_bfactor_avg", "coord_his_4_bfactor_avg",
    ]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_geom_fail = n_pdb_miss = 0

    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()

        for xyz_path in tqdm.tqdm(xyz_files, desc="stats", leave=True):
            sid = xyz_path.stem
            row: dict = {k: "" for k in FIELDNAMES}
            row["id"]       = sid
            row["xyz_path"] = str(xyz_path.resolve())

            # --- XYZ header ---
            header = _parse_xyz_header(xyz_path)
            row["resolution_A"] = header.get("RESOLUTION_A", "")

            pdb_id = header.get("PDB", sid[:4]).lower()

            # --- Geometry ---
            s = parse_structure(xyz_path)
            if s is None:
                n_geom_fail += 1
                writer.writerow(row)
                continue
            row.update(compute_geometry(s))

            # --- Family string (always computed from XYZ CA atom SEC tags) ---
            res_meta = _parse_xyz_residues(xyz_path)
            row["family"] = compute_family_string(res_meta["coord"], res_meta["coord_sec"])

            # --- PDB stats ---
            if pdb_dir is not None:
                pdb_path = pdb_dir / f"{pdb_id}.pdb"
                if pdb_path.is_file():
                    zn_info    = res_meta["zn"]
                    coord_list = res_meta["coord"]
                    if zn_info is not None:
                        pdb_stats = _parse_pdb_stats(
                            pdb_path, zn_info[0], zn_info[1], coord_list
                        )
                        row.update(pdb_stats)
                else:
                    n_pdb_miss += 1

            n_ok += 1
            writer.writerow(row)

    print(f"\nWrote {n_ok} rows → {out_csv}")
    if n_geom_fail:
        print(f"  {n_geom_fail} XYZ files skipped (parse failed)")
    if n_pdb_miss:
        print(f"  {n_pdb_miss} structures missing PDB file (B-factors / R-factors left empty)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute per-structure stats from XYZ + PDB files."
    )
    parser.add_argument("--xyz-dir", type=Path, required=True,
                        help="Directory containing .xyz files.")
    parser.add_argument("--glob", type=str, default="*.xyz",
                        help="Filename pattern within --xyz-dir (default: *.xyz). "
                             "Use '*_Zn-extended.xyz' for His dirs holding both variants.")
    parser.add_argument("--pdb-dir", type=Path, default=None,
                        help="Directory containing <pdbid>.pdb files for B-factor and R-factor extraction.")
    parser.add_argument("--out-csv", type=Path, default=None,
                        help="Output CSV path (default: <xyz-dir>/structure_stats.csv).")
    parser.add_argument("--force", action="store_true",
                        help="Recalculate even if output already exists.")
    args = parser.parse_args()

    xyz_dir = args.xyz_dir.expanduser().resolve()
    if not xyz_dir.is_dir():
        raise SystemExit(f"--xyz-dir not found: {xyz_dir}")

    pdb_dir = args.pdb_dir.expanduser().resolve() if args.pdb_dir else None
    if pdb_dir is not None and not pdb_dir.is_dir():
        raise SystemExit(f"--pdb-dir not found: {pdb_dir}")

    out_csv = (args.out_csv or xyz_dir / "structure_stats.csv").expanduser().resolve()

    compute_stats(xyz_dir, pdb_dir, out_csv, force=args.force, glob_pat=args.glob)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
