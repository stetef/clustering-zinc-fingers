#!/usr/bin/env python3
"""Approach 3 — Permutation Invariant Vector (PIV).

Fully invariant to rotation, translation, and residue permutation by construction.
Independent of any reference structure.

Feature: 78-D vector of sorted pairwise distances grouped by atom-pair type.
Blocks (sorted ascending within each):
  Zn–S (4), Zn–Cβ (4), Zn–Cα (4)
  S–S (6), S–Cβ (16), S–Cα (16)
  Cβ–Cβ (6), Cβ–Cα (16), Cα–Cα (6)
Total: 78

Near-constant covalent-bond columns (S–Cβ, Cβ–Cα) are handled by the variance
floor in z-scoring; no explicit exclusion needed.

Outputs to --out-dir:
  labels.csv, medoids.csv, k_sweep.csv, per_cluster_intra.csv

Usage
-----
  uv run python -m zn_cys_his.clustering.step05_approach3_piv \\
      --xyz-dir data/test-4cys-weighted/initial_xyz_files \\
      --out-dir data/test-4cys-weighted/approach3 \\
      --k-min 2 --k-max 8
"""
from __future__ import annotations

import argparse
import logging
import sys
from itertools import combinations, product
from pathlib import Path

import numpy as np
import tqdm

from zn_cys_his.clustering.utils import (
    EQUAL_WEIGHTS, SHELL_WEIGHTS, DISTANCE_WEIGHTS,
    Structure, parse_structure,
    sweep_k, save_outputs, build_cluster_distribution_plots,
    gather_structures, print_gather_report,
)


# ---------------------------------------------------------------------------
# PIV featurization
# ---------------------------------------------------------------------------

def piv(s: Structure) -> np.ndarray:
    """Compute the PIV feature vector for one structure.

    Heavy atoms are labelled by chemistry ('ZN' or '<RESTYPE>_<ATOM>', e.g.
    'CYS_SG', 'HIS_ND1').  For every unordered pair of labels the pairwise
    distances are collected and sorted ascending — permutation invariant and,
    for a fixed composition, of fixed length.  For a pure-Cys(4) site this
    reproduces the original 78-D, 9-block layout exactly.
    """
    atoms = s.typed_atoms()
    labels = sorted({lbl for lbl, _ in atoms})
    by_label: dict = {lbl: [c for l2, c in atoms if l2 == lbl] for lbl in labels}

    def d(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    blocks: list[np.ndarray] = []
    for i in range(len(labels)):
        for j in range(i, len(labels)):
            la, lb = labels[i], labels[j]
            if la == lb:
                cs = by_label[la]
                ds = [d(cs[a], cs[b]) for a, b in combinations(range(len(cs)), 2)]
            else:
                ds = [d(ca, cb) for ca in by_label[la] for cb in by_label[lb]]
            if ds:
                blocks.append(np.sort(ds))
    return np.concatenate(blocks)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    xyz_dir: Path,
    out_dir: Path,
    k_values: list[int],
    w_type: dict | str = EQUAL_WEIGHTS,
    allow_reflection: bool = True,
    stats_csv: Path | None = None,
    glob_pat: str = "*.xyz",
) -> dict | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    log = out_dir / "approach3_run.log"
    logging.basicConfig(filename=str(log), level=logging.INFO,
                        format="%(asctime)s %(message)s", filemode="w")

    print(f"Gathering structures from {xyz_dir} …")
    structures, report = gather_structures(xyz_dir, glob_pat, desc="parsing")
    print_gather_report(report)

    if len(structures) < 4:
        raise SystemExit("Too few structures to cluster.")

    print("Computing PIV features …")
    X = np.array([piv(s) for s in tqdm.tqdm(structures, desc="PIV", leave=False)])
    print(f"Feature matrix: {X.shape}")

    k_values = [k for k in k_values if 2 <= k < len(structures)]
    if not k_values:
        raise SystemExit("No valid k values for this dataset size.")

    best, table = sweep_k(structures, X, k_values, w_type, allow_reflection, desc="A3 k sweep")
    if best is None:
        print("No valid clustering results.")
        return None

    print(f"Best k={best['k']}  intra={best['intra']:.4f}  "
          f"inter={best['inter']:.4f}  ratio={best['ratio']:.4f}  ch_score={best['ch_score']:.4f}")

    # Save standard outputs (includes embeddings.csv + tsne_kmeans.png)
    save_outputs(out_dir, [s.id for s in structures], table, best)
    # Distribution plots (optional; requires --stats-csv)
    build_cluster_distribution_plots(
        out_dir, [s.id for s in structures], best["labels"], stats_csv
    )
    logging.info("Done. Best k=%d ratio=%.4f ch_score=%.4f", best["k"], best["ratio"], best["ch_score"])
    return best


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Approach 3: PIV (sorted pairwise distances) clustering."
    )
    parser.add_argument("--xyz-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--glob", type=str, default="*.xyz",
                        help="Filename pattern within --xyz-dir (default: *.xyz). "
                             "For His datasets holding both variants, use "
                             "'*_Zn-extended.xyz' to avoid loading each structure twice.")
    parser.add_argument("--k-min", type=int, default=15)
    parser.add_argument("--k-max", type=int, default=35)
    parser.add_argument("--k-step", type=int, default=1)
    parser.add_argument("--no-reflection", action="store_true")
    parser.add_argument("--stats-csv", type=Path, default=None,
                        help="Per-structure metadata CSV (id column required) for "
                             "distribution histograms and stats summary.")
    parser.add_argument("--weight-scheme", choices=["equal", "shell", "distance"],
                        default="distance",
                        help="RMSD atom weighting: equal, shell (coord atom=1, other "
                             "arm atoms=0.5), or distance (1/avg_Zn_distance per atom; default).")
    args = parser.parse_args()

    xyz_dir = args.xyz_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    if not xyz_dir.is_dir():
        raise SystemExit(f"--xyz-dir not found: {xyz_dir}")

    _w_map = {"equal": EQUAL_WEIGHTS, "shell": SHELL_WEIGHTS, "distance": DISTANCE_WEIGHTS}
    stats_csv = args.stats_csv.expanduser().resolve() if args.stats_csv else None
    k_values = list(range(args.k_min, args.k_max + 1, args.k_step))
    run(xyz_dir, out_dir, k_values,
        w_type=_w_map[args.weight_scheme],
        allow_reflection=not args.no_reflection, stats_csv=stats_csv,
        glob_pat=args.glob)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
