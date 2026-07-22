"""Back-derive atom names for tag-less XYZ files from their source PDB.

Some structure sets ship XYZ files with no ``ATOM=`` end-of-line tags — just an
element and coordinates.  When the filename starts with a 4-character RCSB id
(e.g. ``1a3n_heme1.xyz`` → ``1a3n``) we can recover per-atom names by matching
each XYZ atom to the nearest atom in the deposited PDB (same coordinate frame,
since the XYZ was carved out of it) and copying that atom's name / residue.

This buys two things for the generic/heme profiles:

  * robust cross-file atom correspondence by *name* (survives reordered lines),
  * identification of which atoms belong to the macrocycle (HEM/HEC/…) vs the
    axial ligand residues.

If the id can't be resolved, the PDB can't be fetched, or too few atoms match
cleanly, the caller falls back to positional naming (assume a consistent atom
order) — and, for heme, the geometric flip simply won't be applied if the
resulting geometry has no clean C2 symmetry.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np

from zn_cys_his.clustering.fetch_pdbs import download_pdb

_PDB_ID_RE = re.compile(r"^[0-9][a-zA-Z0-9]{3}$")

# Common heme-type macrocycle residue names (PDB Chemical Component ids).
MACROCYCLE_RES = {"HEM", "HEC", "HEB", "HEA", "HAS", "DHE", "HDD", "SRM", "VER"}


def pdb_id_from_stem(stem: str) -> Optional[str]:
    """Return the leading 4-char RCSB id if the stem starts with one, else None."""
    token = stem.split("_", 1)[0].split("-", 1)[0]
    return token.lower() if _PDB_ID_RE.match(token) else None


def parse_pdb_atoms(pdb_path: Path) -> list[dict]:
    """Parse ATOM/HETATM records → [{name, res_name, element, xyz}].  Empty on failure."""
    out: list[dict] = []
    try:
        lines = pdb_path.read_text(errors="ignore").splitlines()
    except OSError:
        return out
    for ln in lines:
        rec = ln[:6].strip()
        if rec not in ("ATOM", "HETATM"):
            continue
        try:
            x = float(ln[30:38]); y = float(ln[38:46]); z = float(ln[46:54])
        except ValueError:
            continue
        elem = ln[76:78].strip() or ln[12:16].strip()[:1]
        try:
            bfac = float(ln[60:66])
        except ValueError:
            bfac = None
        out.append({
            "name": ln[12:16].strip().upper(),
            "res_name": ln[17:20].strip().upper(),
            "element": elem.capitalize(),
            "xyz": np.array([x, y, z]),
            "bfactor": bfac,
        })
    return out


def resolve_pdb(pdb_id: str, pdb_dir: Optional[Path], fetch: bool) -> Optional[Path]:
    """Locate <pdb_id>.pdb in pdb_dir, downloading it there if fetch and absent."""
    if pdb_dir is None:
        pdb_dir = Path.home() / ".cache" / "generic_pdb_tags"
    pdb_dir.mkdir(parents=True, exist_ok=True)
    dest = pdb_dir / f"{pdb_id}.pdb"
    if dest.is_file():
        return dest
    if fetch and download_pdb(pdb_id, dest):
        return dest
    return dest if dest.is_file() else None


def back_derive(
    raw: dict,
    pdb_atoms: list[dict],
    tol: float = 0.4,
    min_match_fraction: float = 0.7,
) -> Optional[dict]:
    """Assign names/residues to raw XYZ atoms from the nearest PDB atom.

    Returns an enriched copy of ``raw`` with ``names`` replaced by ``<RES>_<NAME>``
    and a new ``res_names`` list, or None if fewer than ``min_match_fraction`` of
    atoms match within ``tol`` Å (frame mismatch → not trustworthy).
    """
    if not pdb_atoms:
        return None
    pdb_xyz = np.array([a["xyz"] for a in pdb_atoms])
    # Map back to the source-PDB frame if the extraction recentered by a CENTROID.
    offset = raw.get("centroid")
    coords = raw["coords"] + offset if offset is not None else raw["coords"]
    names: list[str] = []
    res_names: list[str] = []
    matched = 0
    for c in coords:
        d = np.linalg.norm(pdb_xyz - c, axis=1)
        j = int(np.argmin(d))
        if d[j] <= tol:
            a = pdb_atoms[j]
            names.append(f"{a['res_name']}_{a['name']}")
            res_names.append(a["res_name"])
            matched += 1
        else:
            names.append(None)  # unmatched; filled below
            res_names.append("")
    if matched < int(min_match_fraction * len(raw["coords"])):
        return None
    # Fill unmatched atoms with a positional fallback name so slots stay distinct.
    for i, n in enumerate(names):
        if n is None:
            names[i] = f"{raw['elements'][i].upper()}{i}"
    enriched = dict(raw)
    enriched["names"] = names
    enriched["res_names"] = res_names
    return enriched
