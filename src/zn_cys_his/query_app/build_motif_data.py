#!/usr/bin/env python3
"""Extract per-PDB PROSITE motif + consensus-name data for the app.

The BLAST/PROSITE pipeline emits one ``*_pdb_motif_and_enzyme.xlsx`` per system
(built by ``pdb_csv_to_motif_and_enzyme_Sum.py``). Each workbook's ``overview``
sheet has one row per RCSB entry with the consensus enzyme name and the top
motifs found:

    group, pdb_id, best_hit_name, consensus_name, motif1, motif2, motif3

This script distils those sheets into a small, tracked ``motifs.csv`` next to the
app module — keyed by ``(dataset, lowercase pdb_id)`` — so both the Unique PDBs
tab (app.py) and the Validation tab's "color by motif" view read a lightweight
CSV with no Excel/openpyxl dependency at runtime.

The key includes the dataset because the scans are run **per system**: a PDB that
contributes sites to two systems appears in both workbooks, and its rows can
differ (the BLAST query is the entity carrying that system's site, so a different
chain may be picked). Of the 366 PDBs shared by the 3Cys1His and 4Cys scans, ~17
disagree on ``motif1`` and 8 on ``consensus_name``.

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

# dataset name (as used by the app / validation_data dirs) -> overview workbook.
# Add a line here when another system gets a BLAST/PROSITE scan.
WORKBOOKS: dict[str, Path] = {
    "3cys1his": REPO_ROOT / "blast-output/3cys1his-large/3Cys1His_pdb_motif_and_enzyme.xlsx",
    "4cys": REPO_ROOT / "blast-output/4cys-large/4Cys_pdb_motif_and_enzyme.xlsx",
}
OUT_CSV = HERE / "motifs.csv"

KEEP = ["pdb_id", "consensus_name", "best_hit_name", "motif1", "motif2", "motif3"]


def read_overview(dataset: str, xlsx: Path) -> pd.DataFrame:
    """The workbook's ``overview`` sheet, trimmed to KEEP and tagged with dataset."""
    df = pd.read_excel(xlsx, sheet_name="overview")
    missing = [c for c in KEEP if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: {xlsx.name} overview sheet missing columns: {missing}")
    out = df[KEEP].copy()
    out.insert(0, "dataset", dataset)
    out["pdb_id"] = out["pdb_id"].astype(str).str.strip().str.lower()
    return out.dropna(subset=["pdb_id"]).drop_duplicates(subset="pdb_id")


def main() -> None:
    frames: list[pd.DataFrame] = []
    for dataset, xlsx in WORKBOOKS.items():
        if not xlsx.exists():
            # A partial checkout (or a system whose scan hasn't run) is not fatal:
            # the app just won't offer motif/enzyme views for that dataset.
            print(f"WARNING: skipping {dataset} — workbook not found: {xlsx}")
            continue
        frames.append(read_overview(dataset, xlsx))
    if not frames:
        sys.exit("ERROR: no motif workbooks found; nothing to write.")

    out = pd.concat(frames, ignore_index=True)
    out.to_csv(OUT_CSV, index=False)

    print(f"wrote {OUT_CSV.relative_to(REPO_ROOT)}")
    for dataset, grp in out.groupby("dataset", sort=False):
        n_motif = grp["motif1"].notna().sum()
        print(f"  {dataset}: {len(grp)} PDBs ({n_motif} with at least one motif)")


if __name__ == "__main__":
    main()
