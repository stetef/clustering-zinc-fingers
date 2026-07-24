#!/usr/bin/env python3
"""Extract per-PDB PROSITE motif + consensus-name data for the app.

The BLAST/PROSITE pipeline emits ``3Cys1His_pdb_motif_and_enzyme.xlsx`` (built by
``pdb_csv_to_motif_and_enzyme_Sum.py``). Its ``overview`` sheet has one row per
RCSB entry with the consensus enzyme name and the top motifs found:

    group, pdb_id, best_hit_name, consensus_name, motif1, motif2, motif3

This script distils that sheet into a small, tracked ``motifs.csv`` next to the
app module — keyed by the *lowercase* 4-char pdb_id the rest of the app uses — so
both the Unique PDBs tab (app.py) and the Validation tab's "color by motif" view
read a lightweight CSV with no Excel/openpyxl dependency at runtime.

``motif3`` may itself carry a ``"; "``-joined overflow of any motifs beyond the
first two (see build_overview in the generator), so the columns are copied
verbatim; downstream code splits on ``;`` to recover the full per-PDB motif set.

Run (needs openpyxl to read the .xlsx):

    python build_motif_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
try:
    from zn_cys_his.paths import REPO_ROOT
except ModuleNotFoundError:  # run directly without the package installed
    REPO_ROOT = HERE.parents[2]

# The 3Cys1His overview workbook (the one system with a PROSITE motif scan).
XLSX = REPO_ROOT / "blast-output/3cys1his-large/3Cys1His_pdb_motif_and_enzyme.xlsx"
OUT_CSV = HERE / "motifs.csv"

KEEP = ["pdb_id", "consensus_name", "best_hit_name", "motif1", "motif2", "motif3"]


def main() -> None:
    if not XLSX.exists():
        sys.exit(f"ERROR: motif workbook not found: {XLSX}")
    df = pd.read_excel(XLSX, sheet_name="overview")
    missing = [c for c in KEEP if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: overview sheet missing columns: {missing}")

    out = df[KEEP].copy()
    out["pdb_id"] = out["pdb_id"].astype(str).str.strip().str.lower()
    out = out.dropna(subset=["pdb_id"]).drop_duplicates(subset="pdb_id")
    out.to_csv(OUT_CSV, index=False)

    n_motif = out["motif1"].notna().sum()
    print(f"wrote {OUT_CSV.relative_to(REPO_ROOT)}")
    print(f"  {len(out)} PDBs ({n_motif} with at least one motif)")


if __name__ == "__main__":
    main()
