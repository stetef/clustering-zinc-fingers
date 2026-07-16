#!/usr/bin/env python3
"""Pipeline orchestrator for Zn(Cys/His)₄ structure clustering.

Chains the pipeline steps in order, skipping steps whose key output already
exists (idempotent).  Pass --force to re-run everything.

Works on any 4-coordinate Cys/His site (4cys, 3cys1his, 2cys2his, 1cys3his,
4his).  Composition is auto-detected from the input; residue matching keeps two
classes (Cys, His) so Cys never maps onto His, while His-ND1 and His-NE2 are
matchable.  A dataset must be a single composition (stray odd files are dropped
with a warning).

Stages (each rerunnable on its own via --stage / --from-stage)
--------
  prep      fetch source PDBs from RCSB (any missing from the dataset's
            pdb-files/; skip with --no-fetch) -> step01 annotate (SEC= tags)
            -> step02 stats (metadata + family).  Annotated COPIES are written
            under the output root; INPUTS ARE NEVER MODIFIED.  Runs once and is
            shared by all approaches.
  cluster   the selected approaches:
              step03 approach1  Aligned Cartesian featurization + clustering
              step04 approach2  Z-matrix featurization (optional)
              step05 approach3  PIV featurization + clustering
            Each approach auto-emits cluster_pdb_family.csv.
  validate  step06 per-approach RMSD metrics, plots, PCA→XYZ + cross-approach
            comparison, over whatever approach outputs exist.

Approach 2 uses pure numpy (no chemcoord dependency).  Input XYZ files are read
as-is (no hydrogen-cleanup step); '*-extended.xyz' duplicates, .pc and .gzmat
are rejected automatically, structures with a coordinating non-Cys/His ligand
(GLU/SO4/inhibitor; water tolerated) are rejected as mixed-ligand sites, and
structures whose coordinating-residue atom count is off the mode are dropped.

Artifact placement
------------------
INPUTS live under --base-dir (e.g. data/4cys-large/{xyz-files,pdb-files}) and
are treated as read-only.  All GENERATED artifacts go to a mirror output root,
by default cluster-output/<dataset> (override with --out-dir):

  <out-root>/
    prep/annotated_xyz/    SEC=-annotated copies of the input XYZ
    prep/structure_stats.csv
    approach1/             labels, medoids, k_sweep, kmeans_labels_with_stats,
                           cluster_pdb_family.csv, aligned_xyz/, plots, …
    approach2/  approach3/ same shape (approach2 optional)
    validation/approach{1,3}/   per-approach metrics + plots
    validation/comparison/      cross-approach comparison

Usage
-----
  # Full pipeline on the 4-Cys dataset.  `zch-pipeline` is the console script
  # for this module (== python -m zn_cys_his.clustering.orchestrate):
  uv run zch-pipeline \\
      --base-dir data/4cys-large \\
      [--approaches 1 3]      # which approaches to run (default: 1 3)
      [--k-min 15 --k-max 35] # k sweep range (default step 1)
      [--force]               # re-run all steps

  # Any other dataset — same command; xyz-files/ is the default input subdir:
  uv run zch-pipeline --base-dir data/2cys2his-large

  # Rerun just one stage (e.g. re-cluster approach 3, or redo validation):
  uv run zch-pipeline --base-dir data/4cys-large --stage cluster --approaches 3
  uv run zch-pipeline --base-dir data/4cys-large --stage validate

  # Resume from the cluster stage onward (skip prep):
  uv run zch-pipeline --base-dir data/4cys-large --from-stage cluster
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from zn_cys_his.paths import mirror_output

_PYTHON = sys.executable

# Pipeline stages, in order.  prep = annotate + stats (runs once, shared);
# cluster = the selected approaches; validate = per-approach + comparison.
STAGES = ["prep", "cluster", "validate"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(
    label: str,
    cmd: list[Path | str],
    *,
    skip_if: Path | None = None,
    force: bool = False,
) -> bool:
    """Run cmd.  Skip if skip_if exists and not force.  Return True if ran."""
    if not force and skip_if is not None:
        if skip_if.is_file() or (skip_if.is_dir() and any(skip_if.iterdir())):
            print(f"  skip  {label}")
            return False
    print(f"  run   {label}")
    subprocess.run([str(c) for c in cmd], check=True)
    return True


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def step_fetch_pdbs(xyz_dir: Path, pdb_dir: Path, glob_pat: str, force: bool) -> None:
    """Download any source PDBs missing from pdb_dir (idempotent, once per dataset).

    Non-fatal: individual missing/obsolete ids are reported but do not abort the
    pipeline (annotate/stats warn per-file for anything still absent).
    """
    print("  run   step00 fetch_pdbs")
    cmd = [_PYTHON, "-m", "zn_cys_his.clustering.fetch_pdbs",
           "--xyz-dir", xyz_dir, "--pdb-dir", pdb_dir, "--glob", glob_pat]
    if force:
        cmd.append("--force")
    subprocess.run([str(c) for c in cmd], check=False)


def step_annotate(
    src_xyz_dir: Path,
    pdb_dir: Path,
    out_xyz_dir: Path,
    force: bool,
    glob_pat: str = "*.xyz",
) -> None:
    """Write SEC=-annotated COPIES of src_xyz_dir into out_xyz_dir.

    Inputs are left untouched.  Runs before step02 (stats) so the family string
    picks up secondary structure.  A sentinel in out_xyz_dir marks completion
    for idempotent skipping (cleared by --force).
    """
    marker = out_xyz_dir / ".secstruct_annotated"
    ran = _run(
        "step01 annotate_secstruct",
        [_PYTHON, "-m", "zn_cys_his.clustering.step01_annotate_secstruct",
         src_xyz_dir, "--pdb-dir", pdb_dir, "--glob", glob_pat,
         "--out-dir", out_xyz_dir],
        skip_if=marker,
        force=force,
    )
    if ran:
        marker.write_text("")


def step_compute_stats(
    xyz_dir: Path,
    pdb_dir: Path | None,
    out_csv: Path,
    force: bool,
    glob_pat: str = "*.xyz",
) -> Path:
    _run(
        "step02 compute_structure_stats",
        [_PYTHON, "-m", "zn_cys_his.clustering.step02_compute_stats",
         "--xyz-dir", xyz_dir, "--out-csv", out_csv, "--glob", glob_pat]
        + (["--pdb-dir", pdb_dir] if pdb_dir is not None else [])
        + (["--force"] if force else []),
        skip_if=out_csv,
        force=force,
    )
    return out_csv


def step_approach1(out_root: Path, xyz_dir: Path, k_min: int, k_max: int, k_step: int,
                   force: bool, stats_csv: Path | None = None,
                   weight_scheme: str = "distance", glob_pat: str = "*.xyz") -> Path:
    out = out_root / "approach1"
    cmd = [_PYTHON, "-m", "zn_cys_his.clustering.step03_approach1_cartesian",
           "--xyz-dir", xyz_dir,
           "--out-dir", out,
           "--glob", glob_pat,
           "--k-min", str(k_min), "--k-max", str(k_max), "--k-step", str(k_step),
           "--weight-scheme", weight_scheme]
    if stats_csv is not None:
        cmd += ["--stats-csv", stats_csv]
    _run("step03 approach1_cartesian", cmd, skip_if=out / "labels.csv", force=force)
    return out


def step_approach2(out_root: Path, aligned_dir: Path, k_min: int, k_max: int, k_step: int,
                   force: bool, stats_csv: Path | None = None,
                   weight_scheme: str = "distance") -> Path:
    """Run Approach 2 (pure numpy; no chemcoord dependency)."""
    out = out_root / "approach2"
    cmd = [_PYTHON, "-m", "zn_cys_his.clustering.step04_approach2_zmatrix",
           "--aligned-xyz-dir", aligned_dir,
           "--out-dir", out,
           "--k-min", str(k_min), "--k-max", str(k_max), "--k-step", str(k_step),
           "--weight-scheme", weight_scheme]
    if stats_csv is not None:
        cmd += ["--stats-csv", stats_csv]
    _run("step04 approach2_zmatrix", cmd, skip_if=out / "labels.csv", force=force)
    return out


def step_approach3(out_root: Path, xyz_dir: Path, k_min: int, k_max: int, k_step: int,
                   force: bool, stats_csv: Path | None = None,
                   weight_scheme: str = "distance", glob_pat: str = "*.xyz") -> Path:
    out = out_root / "approach3"
    cmd = [_PYTHON, "-m", "zn_cys_his.clustering.step05_approach3_piv",
           "--xyz-dir", xyz_dir,
           "--out-dir", out,
           "--glob", glob_pat,
           "--k-min", str(k_min), "--k-max", str(k_max), "--k-step", str(k_step),
           "--weight-scheme", weight_scheme]
    if stats_csv is not None:
        cmd += ["--stats-csv", stats_csv]
    _run("step05 approach3_piv", cmd, skip_if=out / "labels.csv", force=force)
    return out


def step_validate(
    out_root: Path,
    approach_dir: Path,
    xyz_dir: Path,
    is_approach1: bool,
    force: bool,
    weight_scheme: str = "distance",
    sampled_val_dir: Path | None = None,
    glob_pat: str = "*.xyz",
) -> Path:
    out = out_root / "validation" / approach_dir.name
    cmd = [_PYTHON, "-m", "zn_cys_his.clustering.step06_validate_clusters",
           "--approach-dir", approach_dir,
           "--xyz-dir", xyz_dir,
           "--glob", glob_pat,
           "--out-dir", out,
           "--weight-scheme", weight_scheme]
    if is_approach1:
        cmd.append("--approach1")
    if sampled_val_dir is not None and sampled_val_dir.is_dir():
        cmd += ["--sampled-val-dir", sampled_val_dir]
    _run(
        f"step06 validate  ({approach_dir.name})",
        cmd,
        skip_if=out / "k_sweep_plot.png",
        force=force,
    )
    return out


def step_compare(out_root: Path, approach_dirs: list[Path], force: bool) -> None:
    out = out_root / "validation" / "comparison"
    _run(
        "step06 validate --compare-dirs",
        [_PYTHON, "-m", "zn_cys_his.clustering.step06_validate_clusters",
         "--compare-dirs"] + [str(d) for d in approach_dirs] + ["--out-dir", out],
        skip_if=out / "comparison_table.csv",
        force=force,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Orchestrate the Zn(Cys)₄ structure clustering pipeline."
    )
    parser.add_argument("--base-dir", type=Path, required=True,
                        help="Base directory for all outputs.")
    parser.add_argument("--xyz-subdir", type=str, default="xyz-files",
                        help="Subdirectory under --base-dir containing raw XYZ files "
                             "(default: xyz-files).")
    parser.add_argument("--xyz-glob", type=str, default="*.xyz",
                        help="Filename pattern within the raw XYZ dir (default: *.xyz). "
                             "'*-extended.xyz' files are auto-rejected (the coordinating-cluster "
                             "'*_Zn.xyz' is kept), as are .pc/.gzmat, leaving one file per structure. "
                             "Structures whose coordinating-residue atom count is off the dataset "
                             "mode are dropped (waters / co-ligands are excluded from the count).")
    parser.add_argument("--approaches", type=int, nargs="+", default=[1, 3],
                        choices=[1, 2, 3],
                        help="Which approaches to run (default: 1 3).")
    parser.add_argument("--k-min",  type=int, default=15)
    parser.add_argument("--k-max",  type=int, default=35)
    parser.add_argument("--k-step", type=int, default=1)
    parser.add_argument("--force", action="store_true",
                        help="Re-run all steps even if outputs exist.")
    parser.add_argument("--weight-scheme", choices=["equal", "shell", "distance"],
                        default="distance",
                        help="RMSD atom weighting for clustering evaluation: equal, "
                             "shell (coordinating atom=1, other arm atoms=0.5), or "
                             "distance (weight = 1/avg_Zn_distance; default).")
    parser.add_argument("--pdb-dir", type=Path, default=None,
                        help="Directory of <pdbid>.pdb files for B-factor and R-factor extraction. "
                             "When provided, structure_stats.csv is generated automatically.")
    parser.add_argument("--stats-csv", type=Path, default=None,
                        help="Explicit per-structure metadata CSV.  If omitted, the prep stage's "
                             "<out-root>/prep/structure_stats.csv is used when present.")
    parser.add_argument("--sampled-val-dir", type=Path, default=None,
                        help="Directory of sampled XYZ files for validation overlays "
                             "(passed to step06_validate_clusters --sampled-val-dir).")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Root for generated artifacts (default: cluster-output/<dataset>, "
                             "mirroring data/; inputs under --base-dir are never modified).")
    parser.add_argument("--stage", choices=STAGES, default=None,
                        help="Run ONLY this stage (prep | cluster | validate). Default: all.")
    parser.add_argument("--from-stage", dest="from_stage", choices=STAGES, default=None,
                        help="Run this stage and every stage after it.")
    parser.add_argument("--no-fetch", action="store_true",
                        help="Skip the prep-stage PDB download from RCSB (use whatever "
                             "PDBs are already present in the pdb dir).")
    args = parser.parse_args()

    if args.stage and args.from_stage:
        raise SystemExit("--stage and --from-stage are mutually exclusive")

    base = args.base_dir.expanduser().resolve()
    xyz_raw = base / args.xyz_subdir
    if not xyz_raw.is_dir():
        raise SystemExit(f"XYZ input directory not found: {xyz_raw}")

    out_root = (args.out_dir.expanduser().resolve() if args.out_dir else mirror_output(base))
    k_min, k_max, k_step = args.k_min, args.k_max, args.k_step
    approaches = sorted(set(args.approaches))
    weight_scheme = args.weight_scheme
    active_glob = args.xyz_glob
    pdb_dir         = args.pdb_dir.expanduser().resolve() if args.pdb_dir else None
    stats_override  = args.stats_csv.expanduser().resolve() if args.stats_csv else None
    sampled_val_dir = args.sampled_val_dir.expanduser().resolve() if args.sampled_val_dir else None

    if args.stage:
        run = {args.stage}
    elif args.from_stage:
        run = set(STAGES[STAGES.index(args.from_stage):])
    else:
        run = set(STAGES)

    # Canonical artifact locations under the mirror output root.
    prep_dir      = out_root / "prep"
    annotated_xyz = prep_dir / "annotated_xyz"
    prep_stats    = prep_dir / "structure_stats.csv"
    annot_pdb_dir = pdb_dir if pdb_dir is not None else (base / "pdb-files")

    print(f"\n=== Pipeline: base={base}")
    print(f"             out={out_root}")
    print(f"             stages={[s for s in STAGES if s in run]} approaches={approaches} "
          f"k={k_min}..{k_max} ===\n")

    # ----- prep: fetch PDBs -> annotate (writes copies) -> stats -----------
    if "prep" in run:
        if not args.no_fetch and pdb_dir is None:
            # Repopulate the dataset's (un-versioned) pdb-files/ from RCSB.
            step_fetch_pdbs(xyz_raw, annot_pdb_dir, active_glob, args.force)
        if annot_pdb_dir.is_dir() and any(annot_pdb_dir.glob("*.pdb")):
            step_annotate(xyz_raw, annot_pdb_dir, annotated_xyz, args.force, glob_pat=active_glob)
        else:
            print(f"  skip  step01 annotate_secstruct (no PDBs found at {annot_pdb_dir})")
        stats_src = annotated_xyz if annotated_xyz.is_dir() else xyz_raw
        if stats_override is None and annot_pdb_dir.is_dir():
            step_compute_stats(stats_src, annot_pdb_dir, prep_stats, args.force, glob_pat=active_glob)
        elif stats_override is None:
            print("  skip  step02 compute_structure_stats (no --pdb-dir/pdb-files or --stats-csv)")

    # Resolve the working XYZ dir + stats CSV for cluster/validate, whether or
    # not prep ran in this invocation (prep artifacts may exist from before).
    xyz_dir = annotated_xyz if annotated_xyz.is_dir() else xyz_raw
    if stats_override is not None:
        stats_csv: Path | None = stats_override
    elif prep_stats.is_file():
        stats_csv = prep_stats
    else:
        stats_csv = None

    # ----- cluster: the selected approaches -------------------------------
    if "cluster" in run:
        if 1 in approaches:
            step_approach1(out_root, xyz_dir, k_min, k_max, k_step, args.force,
                           stats_csv=stats_csv, weight_scheme=weight_scheme, glob_pat=active_glob)
        if 2 in approaches:
            aligned_dir = out_root / "approach1" / "aligned_xyz"
            if aligned_dir.is_dir():
                step_approach2(out_root, aligned_dir, k_min, k_max, k_step, args.force,
                               stats_csv=stats_csv, weight_scheme=weight_scheme)
            else:
                print("  skip  step04 approach2 (aligned_xyz not found; run approach 1 first)")
        if 3 in approaches:
            step_approach3(out_root, xyz_dir, k_min, k_max, k_step, args.force,
                           stats_csv=stats_csv, weight_scheme=weight_scheme, glob_pat=active_glob)

    # ----- validate: every approach output present, + comparison ----------
    if "validate" in run:
        approach_dirs = [out_root / f"approach{n}" for n in approaches
                         if (out_root / f"approach{n}" / "labels.csv").is_file()]
        if not approach_dirs:
            print("  skip  step06 validate (no approach outputs found under out-dir)")
        for app_dir in approach_dirs:
            is_a1 = (app_dir.name == "approach1") and (app_dir / "aligned_xyz").is_dir()
            val_xyz = app_dir / "aligned_xyz" if is_a1 else xyz_dir
            # aligned_xyz is single-variant (*.xyz); a raw dir needs the same glob.
            val_glob = "*.xyz" if is_a1 else active_glob
            # Only pass sampled_val_dir for approach 1 (lives under validation/approach1/).
            svd = sampled_val_dir if (is_a1 and sampled_val_dir is not None) else None
            step_validate(out_root, app_dir, val_xyz, is_approach1=is_a1, force=args.force,
                          weight_scheme=weight_scheme, sampled_val_dir=svd, glob_pat=val_glob)
        if len(approach_dirs) > 1:
            step_compare(out_root, approach_dirs, args.force)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
