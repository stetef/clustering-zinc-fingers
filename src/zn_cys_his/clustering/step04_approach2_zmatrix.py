#!/usr/bin/env python3
"""Approach 2 — Z-matrix internal coordinates (pure numpy).

Uses the global R₀ residue labeling from Approach 1 (rotation not needed;
only residue order matters).  Reads the aligned XYZ files written by
02_approach1_cartesian.py, or re-computes R₀ if given a raw --xyz-dir.

Feature encoding (3N−6 internal DoF for an N-atom site; 33-D for Cys₄, N=13):
  N−1 bond lengths r  (Å, raw)
  N−2 bond angles θ   (°, raw)
  N−3 dihedrals φ     → cos(φ) only  [chirality-insensitive; merges enantiomers]

Construction table (built dynamically from the atom count):
  Atoms are taken in canonical heavy() order (Zn, then each residue's arm).
  Atom i is referenced to the three atoms placed before it (i-1, i-2, i-3).
  Works for any Cys/His composition; see _build_ct below.

No external dependencies beyond numpy (chemcoord is not required).

Outputs to --out-dir:
  labels.csv, medoids.csv, k_sweep.csv, per_cluster_intra.csv

Usage
-----
  # After running approach 1:
  uv run python -m zn_cys_his.clustering.step04_approach2_zmatrix \\
      --aligned-xyz-dir cluster-output/1cys3his-large/approach1/aligned_xyz \\
      --out-dir cluster-output/1cys3his-large/approach2 \\
      --k-min 2 --k-max 8

  # Standalone (re-computes R₀ from raw XYZ):
  uv run python -m zn_cys_his.clustering.step04_approach2_zmatrix \\
      --xyz-dir data/1cys3his-large/xyz-files \\
      --out-dir cluster-output/1cys3his-large/approach2 \\
      --k-min 2 --k-max 8
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import tqdm

from zn_cys_his.clustering.utils import (
    EQUAL_WEIGHTS, SHELL_WEIGHTS, DISTANCE_WEIGHTS,
    Structure, parse_structure, sweep_k, save_outputs,
    build_cluster_distribution_plots, gather_structures, print_gather_report,
)


# ---------------------------------------------------------------------------
# Construction table  (built dynamically from atom count)
# ---------------------------------------------------------------------------
# Atoms are taken in the structure's canonical heavy() order (Zn first, then
# each residue's arm).  Each atom i>=1 is described relative to the three atoms
# placed just before it: bond to i-1, angle to i-2, dihedral to i-3.  This is a
# valid rotation/translation-invariant internal-coordinate encoding for any
# composition and yields 3N-6 features (33 for a 13-atom Cys₄ site).

def _build_ct(n_atoms: int) -> list:
    """Return [(bond_ref, angle_ref, dihedral_ref), ...] (-1 = undefined)."""
    return [
        (i - 1 if i >= 1 else -1,
         i - 2 if i >= 2 else -1,
         i - 3 if i >= 3 else -1)
        for i in range(n_atoms)
    ]


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def _dihedral_cos(p1: np.ndarray, p2: np.ndarray,
                  p3: np.ndarray, p4: np.ndarray) -> float:
    """cos of the dihedral angle of the p1-p2-p3-p4 chain."""
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    n1n, n2n = np.linalg.norm(n1), np.linalg.norm(n2)
    if n1n < 1e-10 or n2n < 1e-10:
        return 1.0
    return float(np.clip(np.dot(n1, n2) / (n1n * n2n), -1.0, 1.0))


def zmat_features(s: Structure) -> np.ndarray:
    """Compute the Z-matrix feature vector (bonds, angles, cos dihedrals).

    Length = 3N-6 for an N-atom site (33 for Cys₄; 57 for a 21-atom 2Cys2His).
    """
    c = s.heavy()
    feats: list[float] = []
    for i, (b, a, d) in enumerate(_build_ct(len(c))):
        if b < 0:
            continue
        feats.append(float(np.linalg.norm(c[i] - c[b])))          # bond (Å)
        if a < 0:
            continue
        v1 = c[i] - c[b]
        v2 = c[a] - c[b]
        cos_a = np.clip(
            np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0
        )
        feats.append(float(np.degrees(np.arccos(cos_a))))          # angle (°)
        if d < 0:
            continue
        feats.append(_dihedral_cos(c[i], c[b], c[a], c[d]))       # cos(φ)
    return np.array(feats, dtype=float)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    structures: list[Structure],
    out_dir: Path,
    k_values: list[int],
    w_type: dict | str = EQUAL_WEIGHTS,
    allow_reflection: bool = True,
    stats_csv: Path | None = None,
) -> dict | None:
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Computing Z-matrix features …")
    X = np.array([zmat_features(s) for s in
                  tqdm.tqdm(structures, desc="zmat", leave=False)])
    print(f"Feature matrix: {X.shape}  (bonds, angles, cos dihedrals)")

    k_values = [k for k in k_values if 2 <= k < len(structures)]
    if not k_values:
        raise SystemExit("No valid k values for this dataset size.")

    best, table = sweep_k(structures, X, k_values, w_type, allow_reflection, desc="A2 k sweep")
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
        description="Approach 2: Z-matrix internal coordinates (numpy; no chemcoord needed)."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--aligned-xyz-dir", type=Path,
                       help="Aligned XYZ from approach 1 (already in R₀ residue order).")
    group.add_argument("--xyz-dir", type=Path,
                       help="Raw XYZ dir — will re-compute R₀ alignment internally.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--glob", type=str, default="*.xyz",
                        help="Filename pattern (default: *.xyz; *-extended.xyz auto-rejected).")
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

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=str(out_dir / "approach2_run.log"), level=logging.INFO,
                        format="%(asctime)s %(message)s", filemode="w")
    stats_csv = args.stats_csv.expanduser().resolve() if args.stats_csv else None
    _w_map = {"equal": EQUAL_WEIGHTS, "shell": SHELL_WEIGHTS, "distance": DISTANCE_WEIGHTS}
    w_type = _w_map[args.weight_scheme]

    if args.aligned_xyz_dir:
        xyz_dir = args.aligned_xyz_dir.expanduser().resolve()
        if not xyz_dir.is_dir():
            raise SystemExit(f"--aligned-xyz-dir not found: {xyz_dir}")
        # Aligned XYZ from approach 1 are already filtered and single-variant.
        structures, report = gather_structures(xyz_dir, args.glob, desc="parsing")
        print_gather_report(report)
    else:
        # Re-compute R₀ via approach 1 module
        from approach1_cartesian import establish_reference_and_label  # type: ignore[import]
        xyz_dir = args.xyz_dir.expanduser().resolve()
        if not xyz_dir.is_dir():
            raise SystemExit(f"--xyz-dir not found: {xyz_dir}")
        raw, report = gather_structures(xyz_dir, args.glob, desc="parsing")
        print_gather_report(report)
        print("Computing R₀ alignment …")
        _, structures = establish_reference_and_label(
            raw, w_type, allow_reflection=not args.no_reflection,
        )

    k_values = list(range(args.k_min, args.k_max + 1, args.k_step))
    run(structures, out_dir, k_values,
        w_type=w_type, allow_reflection=not args.no_reflection, stats_csv=stats_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
