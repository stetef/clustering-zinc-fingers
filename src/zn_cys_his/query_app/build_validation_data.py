#!/usr/bin/env python3
"""Copy the CSVs that power the app's Validation tab into a tracked folder.

The interactive validation *reports* (cluster-output/<ds>/validation/.../report_
cluster_distribution*.html) are large, self-contained HTML blobs and cluster-
output/ is gitignored, so neither ships with a Streamlit Cloud deploy. Every
plot in those reports, however, is reproducible from two small CSVs:

  kmeans_labels_with_stats.csv  per-structure cluster id, cluster_color, family,
                                and the numeric metrics the histograms bin
  embeddings.csv                frozen t-SNE coords (tsne1/tsne2) per structure

kmeans_cluster_stats_summary.csv (aggregate per-cluster stats) is copied too when
present; the app can recompute it, but shipping it keeps the tab zero-compute.

This script mirrors build_db.py: it copies those CSVs from cluster-output/ into
``validation_data/<dataset>/<approach>/`` next to the app module, where they are
version-controlled and available to a standalone deploy. Run from anywhere:

    python build_validation_data.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

try:  # normal path: run against the installed package
    from zn_cys_his.paths import CLUSTER_OUTPUT, REPO_ROOT
except ModuleNotFoundError:  # run directly without the package installed
    REPO_ROOT = Path(__file__).resolve().parents[3]
    CLUSTER_OUTPUT = REPO_ROOT / "cluster-output"

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "validation_data"

# App dataset key -> its cluster-output directory (same keys the query app's
# `dataset` column uses; the "-large" suffix is the on-disk pipeline dir).
DATASET_DIRS = {
    "3cys1his": CLUSTER_OUTPUT / "3cys1his-large",
    "4cys": CLUSTER_OUTPUT / "4cys-large",
    "2cys2his": CLUSTER_OUTPUT / "2cys2his-large",
}

# CSVs to copy for each approach. embeddings + labels are required (an approach
# missing either is skipped); the summary is optional.
REQUIRED = ("kmeans_labels_with_stats.csv", "embeddings.csv")
OPTIONAL = ("kmeans_cluster_stats_summary.csv",)


def _has_tsne(embeddings_csv: Path) -> bool:
    """True if embeddings.csv carries the tsne1/tsne2 columns the scatter needs."""
    with embeddings_csv.open("r", encoding="utf-8") as fh:
        header = fh.readline()
    cols = {c.strip() for c in header.split(",")}
    return "tsne1" in cols and "tsne2" in cols


def main() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)  # rebuild from scratch so stale approaches don't linger

    copied = 0
    for ds_key, ds_dir in DATASET_DIRS.items():
        if not ds_dir.is_dir():
            print(f"WARN {ds_key}: {ds_dir} not found, skipping")
            continue

        # Each approach* subdir is a separate clustering; keep them all.
        for approach_dir in sorted(p for p in ds_dir.glob("approach*") if p.is_dir()):
            srcs = {name: approach_dir / name for name in REQUIRED}
            missing = [n for n, p in srcs.items() if not p.exists()]
            if missing:
                continue  # not a completed clustering output
            if not _has_tsne(srcs["embeddings.csv"]):
                print(f"skip {ds_key}/{approach_dir.name}: embeddings.csv has no t-SNE columns")
                continue

            dest_dir = OUT_DIR / ds_key / approach_dir.name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for name in REQUIRED:
                shutil.copyfile(srcs[name], dest_dir / name)
            for name in OPTIONAL:
                opt = approach_dir / name
                if opt.exists():
                    shutil.copyfile(opt, dest_dir / name)
            print(f"copied {ds_key}/{approach_dir.name} -> "
                  f"validation_data/{ds_key}/{approach_dir.name}/")
            copied += 1

    if copied == 0:
        sys.exit("ERROR: no validation CSVs found under cluster-output/")
    print(f"\nwrote {copied} approach(es) under "
          f"{OUT_DIR.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
