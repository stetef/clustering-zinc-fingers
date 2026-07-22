"""Per-structure metrics for the heme/generic report, from the source PDBs.

The Zn(Cys/His) pipeline computes a rich geometry+quality stats table (step02).
Heme structures share only the *quality* and *environment* metrics that don't
depend on a 4-coordinate tetrahedral site:

  r_work, r_free   crystallographic R-factors (PDB REMARK 3)
  fe_bfactor       B-factor of the Fe atom
  avg_bfactor      mean B-factor over every heavy atom in the extracted cluster
                   EXCEPT the Fe (i.e. "everything other than Fe")
  family           full non-macrocycle residue content (from the RES tags)

B-factors are read from the source PDB and matched to the XYZ atoms by
coordinate (same frame — the XYZ was carved from the PDB).  Works off the raw
(un-aligned) XYZ files, so it must run before/independently of clustering.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Optional

import numpy as np

from .generic import compute_family, read_raw_xyz
from .pdb_tags import parse_pdb_atoms, pdb_id_from_stem, resolve_pdb

_RWORK_RE = re.compile(r"^REMARK\s+3\s+R VALUE\s+\(WORKING SET\)\s*:\s*([\d.]+)")
_RFREE_RE = re.compile(r"^REMARK\s+3\s+FREE R VALUE\s*:\s*([\d.]+)")

# Column -> label; the report/app bin these.
METRICS: tuple = (
    ("r_work", "R_work"),
    ("r_free", "R_free"),
    ("fe_bfactor", "Fe B-factor"),
    ("avg_bfactor", "Avg B-factor (non-Fe)"),
)


def _r_factors(pdb_path: Path) -> tuple[str, str]:
    rw = rf = ""
    try:
        for ln in pdb_path.read_text(errors="ignore").splitlines():
            if not rw and (m := _RWORK_RE.match(ln)):
                rw = m.group(1)
            if not rf and (m := _RFREE_RE.match(ln)):
                rf = m.group(1)
            if rw and rf:
                break
    except OSError:
        pass
    return rw, rf


def _bfactors_for(raw: dict, pdb_atoms: list[dict], center_name: str,
                  tol: float = 0.4) -> tuple[str, str]:
    """(fe_bfactor, avg_bfactor over non-Fe) by matching XYZ atoms to PDB atoms."""
    if not pdb_atoms:
        return "", ""
    pxyz = np.array([a["xyz"] for a in pdb_atoms])
    # Map XYZ coords back to the source-PDB frame (extraction subtracts CENTROID).
    offset = raw.get("centroid")
    coords = raw["coords"] + offset if offset is not None else raw["coords"]
    cn = center_name.upper()
    fe_b = None
    others: list[float] = []
    for c, nm, el in zip(coords, raw["names"], raw["elements"]):
        d = np.linalg.norm(pxyz - c, axis=1)
        j = int(np.argmin(d))
        if d[j] > tol:
            continue
        b = pdb_atoms[j].get("bfactor")
        is_center = (nm.split("#")[0].split("_")[-1] == cn or el.upper() == cn)
        if is_center:
            if b is not None:
                fe_b = b
        elif b is not None:
            others.append(b)
    fe_s = f"{fe_b:.3f}" if fe_b is not None else ""
    avg_s = f"{float(np.mean(others)):.3f}" if others else ""
    return fe_s, avg_s


def make_compute_stats(center_name: str = "FE", macrocycle: Optional[set] = None,
                       pdb_dir: Optional[Path] = None, fetch: bool = False):
    """Return compute_stats(xyz_dir, glob, out_csv) -> Path|None.

    ``pdb_dir`` / ``fetch`` are captured here (same as the profile's gather), so
    step03 only supplies the XYZ dir, glob, and output path.
    """
    def _compute(xyz_dir: Path, glob_pat: str, out_csv: Path) -> Optional[Path]:
        rows: list[dict] = []
        pdb_cache: dict[str, list] = {}
        rfac_cache: dict[str, tuple] = {}
        for f in sorted(xyz_dir.glob(glob_pat)):
            raw = read_raw_xyz(f)
            if raw is None:
                continue
            fam = compute_family(raw, center_name, macrocycle=macrocycle)
            rw = rf = fe_b = avg_b = ""
            pid = pdb_id_from_stem(raw["id"])
            if pid:
                if pid not in pdb_cache:
                    p = resolve_pdb(pid, pdb_dir, fetch)
                    pdb_cache[pid] = parse_pdb_atoms(p) if p else []
                    rfac_cache[pid] = _r_factors(p) if p else ("", "")
                rw, rf = rfac_cache[pid]
                fe_b, avg_b = _bfactors_for(raw, pdb_cache[pid], center_name)
            rows.append({"id": raw["id"], "r_work": rw, "r_free": rf,
                         "fe_bfactor": fe_b, "avg_bfactor": avg_b, "family": fam})
        if not rows:
            return None
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with out_csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["id", "r_work", "r_free",
                                               "fe_bfactor", "avg_bfactor", "family"])
            w.writeheader()
            w.writerows(rows)
        print(f"  Heme stats ({len(rows)} structures) → {out_csv}")
        return out_csv
    return _compute
