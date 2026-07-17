#!/usr/bin/env python3
"""Approach 1 — Reference alignment with joint rotation + residue matching.

Aligns all structures to a global reference R₀ (dataset medoid in PCA space)
computed by fixed-point iteration.  Rotation and residue matching are solved
jointly by enumerating all 24 residue permutations.  Enantiomers are merged
(reflection allowed and applied).

Outputs to --out-dir:
  labels.csv              structure_id, cluster  (best k from sweep)
  medoids.csv             cluster_id, medoid_id
  k_sweep.csv             k, intra, inter, ratio
  per_cluster_intra.csv   cluster_id, mean_intra_rmsd
  aligned_xyz/            13-atom XYZ per structure in R₀ residue order
  r0_id.txt               stem of the R₀ structure

Usage
-----
  uv run python -m zn_cys_his.clustering.step03_approach1_cartesian \\
      --xyz-dir  data/1cys3his-large/xyz-files \\
      --out-dir  cluster-output/1cys3his-large/approach1 \\
      --k-min 2 --k-max 8
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import numpy as np
import tqdm

# Add script directory to path so utils is importable
from zn_cys_his.clustering.utils import (
    EQUAL_WEIGHTS, SHELL_WEIGHTS, DISTANCE_WEIGHTS,
    Structure, parse_structure, write_structure_xyz,
    weighted_kabsch, class_preserving_perms, _heavy_perm, _zn_distance_weights,
    cluster_pipeline, sweep_k, save_outputs, build_cluster_distribution_plots,
    gather_structures, print_gather_report,
)

try:
    from sklearn.decomposition import PCA
except ImportError:
    raise SystemExit("scikit-learn required: uv add scikit-learn")

# ---------------------------------------------------------------------------
# Joint align-and-match
# ---------------------------------------------------------------------------

def align_to_reference(
    B: Structure,
    R0: Structure,
    w_type: dict | str,
    allow_reflection: bool = True,
) -> tuple[Structure, list[int], float]:
    """Solve rotation + residue matching jointly against R₀.

    Returns (aligned_structure, best_perm, best_rmsd).
    The winning permutation and rotation are applied; enantiomers are collapsed
    onto R₀'s handedness when allow_reflection=True.  Residue matching preserves
    class (Cys↔Cys, His↔His); B must share R₀'s composition.
    """
    if B.composition() != R0.composition():
        # Should not happen within a single-composition dataset; skip cleanly.
        return B, list(range(B.n_res())), math.inf

    R0_heavy = R0.heavy()
    _static_w = None if w_type == DISTANCE_WEIGHTS else R0.w_vec(w_type)
    res_types = list(R0.composition())
    best_rmsd = math.inf
    best_perm = list(range(len(res_types)))
    best_R = np.eye(3)
    best_t = np.zeros(3)

    for perm in class_preserving_perms(res_types):
        w = _zn_distance_weights(R0, B, perm) if w_type == DISTANCE_WEIGHTS else _static_w
        R, t, rmsd = weighted_kabsch(_heavy_perm(B, perm), R0_heavy, w, allow_reflection)
        if rmsd < best_rmsd:
            best_rmsd = rmsd
            best_perm = perm
            best_R, best_t = R, t

    aligned = B.reorder(best_perm).transformed(best_R, best_t)
    return aligned, best_perm, best_rmsd


# ---------------------------------------------------------------------------
# Fixed-point R₀ computation
# ---------------------------------------------------------------------------

def establish_reference_and_label(
    structures: list[Structure],
    w_type: dict,
    convergence_tol: float = 0.005,
    max_ref_iter: int = 10,
    allow_reflection: bool = True,
    var_floor: float = 1e-3,
) -> tuple[Structure, list[Structure]]:
    """Fixed-point R₀ = dataset medoid in PCA space.

    Returns (R0, aligned_structures).
    Logs a convergence trace to stdout (one line per iteration).
    """
    N = len(structures)
    aligned = list(structures)
    R0 = structures[0]
    prev_matchings: list[list[int] | None] = [None] * N
    prev_centroid: np.ndarray | None = None

    print("R₀ refinement:")
    for iteration in range(1, max_ref_iter + 1):
        new_aligned: list[Structure] = []
        new_matchings: list[list[int]] = []

        for s in tqdm.tqdm(aligned, desc=f"  iter {iteration:02d} aligning", leave=False):
            al, perm, _ = align_to_reference(s, R0, w_type, allow_reflection)
            new_aligned.append(al)
            new_matchings.append(perm)

        # PCA on aligned 39-vectors
        X1 = np.array([s.heavy().ravel() for s in new_aligned])
        means = X1.mean(0)
        stds  = np.maximum(X1.std(0), var_floor)
        Xs = (X1 - means) / stds
        pca_full = PCA()
        pca_full.fit(Xs)
        cumvar = np.cumsum(pca_full.explained_variance_ratio_)
        n_comp = min(max(1, int(np.searchsorted(cumvar, 0.95)) + 1), Xs.shape[1])
        pca = PCA(n_components=n_comp)
        X_pca = pca.fit_transform(Xs)

        centroid = X_pca.mean(0)
        new_r0_idx = int(np.argmin(np.linalg.norm(X_pca - centroid, axis=1)))
        new_R0 = new_aligned[new_r0_idx]

        # Centroid shift (only comparable when dimension is the same)
        if prev_centroid is not None and prev_centroid.shape == centroid.shape:
            centroid_shift = float(np.linalg.norm(centroid - prev_centroid))
        else:
            centroid_shift = float("nan")
        prev_centroid = centroid

        changed = sum(1 for pm, nm in zip(prev_matchings, new_matchings)
                      if pm is not None and pm != nm)
        frac_changed = changed / N

        converged = (iteration > 1 and new_R0.id == R0.id and frac_changed < convergence_tol)
        tag = "  [CONVERGED]" if converged else ""
        print(
            f"  iter={iteration:02d}  R0={new_R0.id}"
            f"  matching_changed={changed:4d}/{N} ({frac_changed*100:5.2f}%)"
            f"  pca_dims={n_comp}  centroid_shift={centroid_shift:.4f}{tag}"
        )
        logging.info(
            "iter=%02d R0=%s changed=%d/%d (%.2f%%) pca_dims=%d shift=%.4f",
            iteration, new_R0.id, changed, N, frac_changed * 100, n_comp, centroid_shift,
        )

        prev_matchings = new_matchings
        R0 = new_R0
        aligned = new_aligned

        if converged:
            break
    else:
        print(f"  [MAX_ITER reached at {max_ref_iter}]")

    return R0, aligned


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    xyz_dir: Path,
    out_dir: Path,
    k_values: list[int],
    w_type: dict = EQUAL_WEIGHTS,
    allow_reflection: bool = True,
    convergence_tol: float = 0.005,
    max_ref_iter: int = 10,
    stats_csv: Path | None = None,
    glob_pat: str = "*.xyz",
) -> dict | None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Logging to file only
    log = out_dir / "approach1_run.log"
    logging.basicConfig(filename=str(log), level=logging.INFO,
                        format="%(asctime)s %(message)s", filemode="w")

    # Parse (reject -extended / .pc / .gzmat, drop off-modal atom counts and
    # off-composition structures)
    print(f"Gathering structures from {xyz_dir} …")
    structures, report = gather_structures(xyz_dir, glob_pat, desc="parsing")
    print_gather_report(report)

    if len(structures) < 4:
        raise SystemExit("Too few structures to cluster.")

    k_values = [k for k in k_values if 2 <= k < len(structures)]
    if not k_values:
        raise SystemExit("No valid k values for this dataset size.")

    # Fixed-point R₀ + alignment
    print("\nEstablishing R₀ …")
    R0, aligned = establish_reference_and_label(
        structures, w_type, convergence_tol, max_ref_iter, allow_reflection
    )
    print(f"R₀ = {R0.id}")

    # Feature matrix: flattened 39-D aligned Cartesian coords
    X = np.array([s.heavy().ravel() for s in aligned])
    print(f"Feature matrix: {X.shape}")

    # k sweep
    best, table = sweep_k(aligned, X, k_values, w_type, allow_reflection, desc="A1 k sweep")
    if best is None:
        print("No valid clustering results.")
        return None

    print(f"\nBest k={best['k']}  intra={best['intra']:.4f}  "
          f"inter={best['inter']:.4f}  ratio={best['ratio']:.4f}  ch_score={best['ch_score']:.4f}")

    # Save standard outputs (includes embeddings.csv + tsne_kmeans.png)
    save_outputs(out_dir, [s.id for s in aligned], table, best)

    # Distribution plots (optional; requires --stats-csv)
    build_cluster_distribution_plots(
        out_dir, [s.id for s in aligned], best["labels"], stats_csv
    )

    # Save R₀ identity
    (out_dir / "r0_id.txt").write_text(R0.id + "\n", encoding="utf-8")

    # Write aligned XYZ (for approach 2 to consume)
    aligned_dir = out_dir / "aligned_xyz"
    aligned_dir.mkdir(exist_ok=True)
    print(f"Writing aligned XYZ to {aligned_dir} …")
    for s in tqdm.tqdm(aligned, desc="writing XYZ", leave=False):
        write_structure_xyz(s, aligned_dir / f"{s.id}.xyz")

    logging.info("Done. Best k=%d ratio=%.4f ch_score=%.4f", best["k"], best["ratio"], best["ch_score"])
    return best


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Approach 1: reference alignment + aligned Cartesian clustering."
    )
    parser.add_argument("--xyz-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--glob", type=str, default="*.xyz",
                        help="Filename pattern within --xyz-dir (default: *.xyz). "
                             "For His datasets whose dir holds both variants, use "
                             "'*_Zn-extended.xyz' to avoid loading each structure twice.")
    parser.add_argument("--k-min", type=int, default=15)
    parser.add_argument("--k-max", type=int, default=35)
    parser.add_argument("--k-step", type=int, default=1)
    parser.add_argument("--no-reflection", action="store_true")
    parser.add_argument("--convergence-tol", type=float, default=0.005)
    parser.add_argument("--max-ref-iter", type=int, default=10)
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
        allow_reflection=not args.no_reflection,
        convergence_tol=args.convergence_tol,
        max_ref_iter=args.max_ref_iter,
        stats_csv=stats_csv,
        glob_pat=args.glob)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
