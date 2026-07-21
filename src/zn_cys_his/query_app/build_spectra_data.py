#!/usr/bin/env python3
"""Build the small index that powers the app's Spectra tab.

The FEFF-calculated spectra live under ``data/<ds>/calculated-spectra/<id>/`` as
per-structure ``.dat`` files. Unlike ``cluster-output/`` those files ARE tracked
in git (``.gitignore`` un-ignores ``data/*/calculated-spectra/``), so they ship
with a Streamlit Cloud deploy and the tab reads them directly — no copying.

All this script produces is a tiny join table so the tab needs neither the DB nor
any dataset-name/path guessing at runtime:

  spectra_data/spectra_index.csv   one row per <id> dir that has an ``xas-`` file:
                                   id, pdb_id, dataset, cluster, cluster_color,
                                   and repo-relative paths to its xas / chi-R dat.

Each spectra dir name equals the basename of that structure's ``xyz_path`` in
``structures.db``, which is how cluster id + color are looked up. Run from anywhere:

    python build_spectra_data.py
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

try:  # normal path: run against the installed package
    from zn_cys_his.paths import DATA_DIR, REPO_ROOT
except ModuleNotFoundError:  # run directly without the package installed
    REPO_ROOT = Path(__file__).resolve().parents[3]
    DATA_DIR = REPO_ROOT / "data"

HERE = Path(__file__).resolve().parent
DB_PATH = HERE / "structures.db"
OUT_DIR = HERE / "spectra_data"

# App dataset key -> on-disk dataset dir under data/ (mirrors build_validation_data
# DATASET_DIRS; only datasets with calculated-spectra need listing).
DATASET_DIRS = {
    "4cys": "4cys-large",
}


def _index_by_xyz_basename(con: sqlite3.Connection, dataset: str) -> dict[str, tuple]:
    """xyz basename (== spectra dir name) -> (pdb_id, cluster, cluster_color)."""
    out: dict[str, tuple] = {}
    for pdb_id, cluster, color, xyz in con.execute(
        "SELECT pdb_id, cluster, cluster_color, xyz_path FROM structures WHERE dataset = ?",
        (dataset,),
    ):
        if not xyz:
            continue
        out[Path(xyz).stem] = (pdb_id, cluster, color)
    return out


def build() -> Path:
    if not DB_PATH.exists():
        sys.exit(f"structures.db not found at {DB_PATH}; run build_db.py first.")
    con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)

    rows: list[dict] = []
    for dataset, disk_dir in DATASET_DIRS.items():
        spectra_root = DATA_DIR / disk_dir / "calculated-spectra"
        if not spectra_root.is_dir():
            print(f"! no calculated-spectra for {dataset} ({spectra_root}); skipping")
            continue
        lookup = _index_by_xyz_basename(con, dataset)
        matched = unmatched = 0
        for d in sorted(p for p in spectra_root.iterdir() if p.is_dir()):
            sid = d.name
            xas = d / f"xas-{sid}.dat"
            chir = d / f"chi-R-{sid}.dat"
            if not xas.exists():
                continue
            meta = lookup.get(sid)
            if meta is None:
                unmatched += 1
                continue
            pdb_id, cluster, color = meta
            rows.append({
                "id": sid,
                "pdb_id": pdb_id,
                "dataset": dataset,
                "cluster": int(cluster),
                "cluster_color": color,
                "xas_rel": xas.relative_to(REPO_ROOT).as_posix(),
                "chir_rel": chir.relative_to(REPO_ROOT).as_posix() if chir.exists() else "",
            })
            matched += 1
        print(f"{dataset}: {matched} spectra indexed"
              + (f", {unmatched} dirs with no DB match (skipped)" if unmatched else ""))
    con.close()

    if not rows:
        sys.exit("No spectra indexed; nothing written.")

    OUT_DIR.mkdir(exist_ok=True)
    out_csv = OUT_DIR / "spectra_index.csv"
    import csv as _csv
    fields = ["id", "pdb_id", "dataset", "cluster", "cluster_color", "xas_rel", "chir_rel"]
    with out_csv.open("w", newline="") as fh:
        w = _csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_csv} ({len(rows)} rows)")
    return out_csv


if __name__ == "__main__":
    build()
