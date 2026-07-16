# Refactor notes — clustering pipeline reorganization

Record of the reorganization that turned the ad-hoc `scripts/` tree into the
installable `zn_cys_his` package with a staged, mirror-output pipeline. See
[pipeline.md](pipeline.md) for the day-to-day usage reference; this file is the
"what changed and why" companion.

---

## The new stages

The orchestrator (`zch-pipeline` / `python -m zn_cys_his.clustering.orchestrate`)
runs **three stages**, each independently rerunnable via `--stage` / `--from-stage`:

| Stage | Modules | Does | Writes to |
|---|---|---|---|
| **prep** | fetch_pdbs → step01 annotate → step02 stats | Downloads any missing source PDBs from RCSB (`--no-fetch` to skip), tags CA atoms with `SEC=` from the PDB (writing annotated **copies**), then computes per-structure metadata + `family` string. Runs once, shared by all approaches. | `data/<ds>/pdb-files/` (fetch), `cluster-output/<ds>/prep/{annotated_xyz/, structure_stats.csv}` |
| **cluster** | step03 approach1, step04 approach2, step05 approach3 | The selected featurization + k-means approaches (`--approaches`). Each auto-emits `cluster_pdb_family.csv`. | `cluster-output/<ds>/approach{1,2,3}/` |
| **validate** | step06 validate_clusters | Per-approach RMSD metrics, k-sweep plots, PCA→XYZ, and cross-approach comparison over whatever approach outputs exist. | `cluster-output/<ds>/validation/{approach*,comparison}/` |

Running them:

```bash
# full run (prep -> cluster -> validate)
uv run zch-pipeline --base-dir data/4cys-large

# one stage only
uv run zch-pipeline --base-dir data/4cys-large --stage cluster --approaches 3
uv run zch-pipeline --base-dir data/4cys-large --stage validate

# one stage and everything after it (skip prep)
uv run zch-pipeline --base-dir data/4cys-large --from-stage cluster
```

Each stage is idempotent (skips work whose output exists; `--force` redoes it).

### Step modules (numbered in true execution order)

`stepNN_` prefixes because Python modules can't start with a digit. The old
`00/00a/00b/01–05` numbering (which did **not** match run order) and the
in-place H-cleanup step were removed.

```
step01_annotate_secstruct   step04_approach2_zmatrix
step02_compute_stats        step05_approach3_piv
step03_approach1_cartesian  step06_validate_clusters
```

---

## Artifact placement: `data/` vs `cluster-output/`

**Inputs are read-only; outputs mirror them.**

```
data/<dataset>/            # READ-ONLY inputs (gitignored, kept locally)
  xyz-files/               #   raw pocket XYZ
  pdb-files/               #   source <pdbid>.pdb
  4cys-large/calculated-spectra/   # precomputed FEFF spectra (spectra-viz input)

cluster-output/<dataset>/  # ALL generated artifacts (gitignored; rm -rf to wipe)
  prep/{annotated_xyz/, structure_stats.csv}
  approach{1,2,3}/         #   labels, medoids, k_sweep, kmeans_labels_with_stats,
                           #   cluster_pdb_family.csv, aligned_xyz/, plots, run.log
  validation/{approach*/, comparison/}
```

`paths.py` is the single source of truth: `REPO_ROOT`, `DATA_DIR`,
`CLUSTER_OUTPUT`, and `mirror_output(base_dir)` (maps `data/X` → `cluster-output/X`).

**Why:** the old scheme wrote outputs next to inputs and `step01` *mutated the
input XYZ files in place*, so runs weren't reproducible from pristine inputs.
Now inputs are never touched (verified: byte-identical after a prep run).

---

## Package layout

```
src/zn_cys_his/
  paths.py                 # REPO_ROOT / DATA_DIR / CLUSTER_OUTPUT / mirror_output()
  clustering/              # orchestrate.py + step01..step06 + utils.py
  spectra/                 # sample.py, plot.py
  query_app/               # app.py, build_db.py + prebuilt structures.db / csv / cache
docs/                      # pipeline.md, filtering.md, refactor-notes.md (this file)
tests/                     # test_smoke.py
```

Console scripts (from `pyproject.toml`, each == `python -m <module>`):
`zch-pipeline`, `zch-fetch-pdbs`, `zch-sample-spectra`, `zch-plot-spectra`,
`zch-build-db`.

## Version-control policy for `data/`

Only the small, hard-to-regenerate **text inputs** are committed:
`xyz-files/` + `initial_xyz_files/` (cleaned pocket XYZ, ~19 MB) and
`calculated-spectra/` (FEFF `.dat`, ~11 MB). The large, freely re-downloadable
`pdb-files/` (~2 GB) are **not** committed — the prep stage fetches them from
RCSB via `zch-fetch-pdbs` (idempotent, once). Generated
`analysis-and-visualization/` plots and the whole `cluster-output/` tree are
also excluded. See `.gitignore`.

---

## Notable behavior changes

- **Charge/multiplicity are composition-derived.** `sample.py` tags each sampled
  XYZ with `CHARGE = 2 − n_cys` (Zn²⁺ + one −1 thiolate per cysteine; His
  neutral) and `MULTIPLICITY = 1`:

  | 4cys | 3cys1his | 2cys2his | 1cys3his | 4his |
  |--|--|--|--|--|
  | −2 | −1 | 0 | +1 | +2 |

  Override with `--charge` / `--multiplicity`; skip with `--no-charge-mult`.
- **`cluster_pdb_family.csv` is automatic** — generated for every approach
  (was a standalone helper).
- **Helpers removed.** `add_charge_multiplicity` folded into `sample.py`;
  `extract_cluster_pdb_family` folded into every approach's output. The
  `clustering/helpers/` package is gone.
- **Spectra defaults repointed** at the mirror: `sample.py`/`plot.py` read
  clustering output from `cluster-output/`; `plot.py` reads FEFF spectra from
  `data/4cys-large/calculated-spectra/` and writes to a sibling
  `analysis-and-visualization/`.
- **`build_db.py`** reads `kmeans_labels_with_stats.csv` from `cluster-output/`
  and PDBs from `data/<ds>/pdb-files/`.

---

## What was done (plan / checklist)

### Round 1 — package restructure
- [x] Move `scripts/` → `src/zn_cys_his/` (src layout); add `__init__` + `pyproject` build backend + console scripts.
- [x] Renumber pipeline scripts to true execution order (`stepNN_`).
- [x] Replace `sys.path` hacks + fragile `parents[N]` with `paths.py`.
- [x] Fix stale docstring/README paths; consolidate docs into `docs/`.
- [x] Add `tests/test_smoke.py`.

### Round 2 — data hygiene + drop cleanup
- [x] Delete `.gzmat` / `.pc` / `-extended.xyz` and stray CSVs from input trees.
- [x] Rename `output/xyz_files` → `xyz-files`, `results/validated_structures` → `pdb-files`.
- [x] Remove the H-cleanup step; renumber remaining steps; update orchestrator + docs.

### Round 3 — modular stages + mirror output + artifact design
- [x] Move `4cys-large/calculated-spectra/analysis-and-visualization` up a level.
- [x] Add `CLUSTER_OUTPUT` + `mirror_output()`.
- [x] `step01` writes annotated **copies** (`--out-dir`), never mutating inputs.
- [x] Auto-generate `cluster_pdb_family.csv` per approach; delete that helper.
- [x] Fold composition-derived charge/multiplicity into `sample.py`; delete that helper.
- [x] Refactor `orchestrate.py` into `prep` / `cluster` / `validate` stages with `--stage` / `--from-stage` / `--out-dir`; write to the mirror.
- [x] Migrate existing test-set artifacts to `cluster-output/`; repoint `build_db.py` + spectra defaults.
- [x] Update `.gitignore`, tests, README, `pipeline.md`.
- [x] Verify (uv sync, 18 smoke tests, prep + cluster stages end-to-end) and commit.

---

### Round 4 — data version-control + PDB fetch
- [x] `.gitignore`: version `xyz-files/`, `initial_xyz_files/`, `calculated-spectra/`; exclude `pdb-files/`, `analysis-and-visualization/`, `cluster-output/`.
- [x] `fetch_pdbs.py` + `zch-fetch-pdbs`: download any PDB missing for an XYZ id from RCSB.
- [x] Wire fetch into the prep stage (idempotent; `--no-fetch` to skip; failures non-fatal).
- [x] Remove stale `.secstruct_annotated` sentinels from input dirs.

---

## Possible follow-ups (not done)

- Mirror the `-large` datasets' `calculated-spectra/` + `analysis-and-visualization/`
  into `cluster-output/` too (currently only clustering artifacts are mirrored;
  spectra live under `data/`).
- Wire all five datasets into `build_db.py` (currently 3: 3cys1his, 4cys, 2cys2his).
- A `--stage prep` cache-invalidation check (re-run prep if inputs are newer than
  the annotated copies) — today prep skips purely on the sentinel/output existing.
