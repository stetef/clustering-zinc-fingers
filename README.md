# zn-cys-his

Clustering and spectra analysis of four-coordinate **Zn(Cys/His)** metal-site
structures, extracted and cleaned from the RCSB PDB. Split out of the larger
`pdb-scraper` project so the clustering can be rerun cleanly and in isolation.

## Layout

```
src/zn_cys_his/            # installable package (src layout)
  paths.py                 # REPO_ROOT / DATA_DIR / CLUSTER_OUTPUT + mirror_output()
  clustering/              # the clustering pipeline, driven by orchestrate.py
    orchestrate.py         # stage runner: prep -> cluster -> validate
    utils.py               # shared module imported by every step
    step01_annotate_secstruct.py   # tag CA atoms with SEC= from the PDB
    step02_compute_stats.py        # per-structure metadata + family string
    step03_approach1_cartesian.py  # aligned-Cartesian featurization + clustering
    step04_approach2_zmatrix.py    # Z-matrix featurization (optional)
    step05_approach3_piv.py        # PIV featurization + clustering
    step06_validate_clusters.py    # RMSD metrics, plots, PCA→XYZ, comparison
  spectra/                 # sample cluster representatives + plot XANES/EXAFS/χ(R)
  query_app/               # Streamlit app + its prebuilt structures.db / csv / cache
data/                      # READ-ONLY inputs (gitignored, kept locally)
  <dataset>/xyz-files/     # raw pocket XYZ            (version-controlled)
  <dataset>/pdb-files/     # source .pdb files         (NOT versioned; zch-fetch-pdbs)
  4cys-large/calculated-spectra/   # precomputed FEFF spectra  (version-controlled)
cluster-output/            # ALL generated artifacts, mirroring data/ (gitignored)
  <dataset>/prep/          # annotated XYZ copies + structure_stats.csv
  <dataset>/approach{1,2,3}/  validation/   # clustering outputs + validation
docs/
  pipeline.md              # detailed pipeline writeup + quick command reference
  filtering.md             # gather/clean filtering funnel per dataset
```

The pipeline **step modules are numbered in the order `orchestrate.py` runs them**
(annotate → stats → approaches → validate); module names can't start with a digit,
hence the `stepNN_` prefix.

**Inputs vs. outputs.** `data/` holds only read-only inputs; everything the
pipeline generates — annotated XYZ copies, per-structure stats, clustering
results, validation — lands in `cluster-output/<dataset>/`, mirroring `data/`.
The input XYZ files are never modified. Wipe a run with `rm -rf cluster-output/<dataset>`.

**What's version-controlled.** The small, hard-to-regenerate text inputs are
committed: `xyz-files/` (cleaned pocket XYZ) and `calculated-spectra/` (FEFF
output, ~11 MB). The large, freely re-downloadable `pdb-files/` (~2 GB) are
**not** committed — the prep stage repopulates them from RCSB automatically
(`zch-fetch-pdbs`, run once), so a fresh clone only needs a network connection
to run the full pipeline. `cluster-output/` is regenerated and never committed.

## Setup

```bash
uv sync         # installs deps + this package (editable) into .venv
```

This registers the console scripts below. Dependencies are intentionally minimal —
`numpy`, `scikit-learn`, `matplotlib`, `plotly`, `tqdm`, `pandas`, plus `streamlit` +
`requests` for the query app.

## Console scripts

| Command | Module | Purpose |
|---|---|---|
| `zch-pipeline` | `zn_cys_his.clustering.orchestrate` | run the full clustering pipeline |
| `zch-fetch-pdbs` | `zn_cys_his.clustering.fetch_pdbs` | download source PDBs from RCSB (run automatically by prep) |
| `zch-sample-spectra` | `zn_cys_his.spectra.sample` | sample cluster representatives for spectra |
| `zch-plot-spectra` | `zn_cys_his.spectra.plot` | plot XANES/EXAFS/χ(R) + interactive report |
| `zch-build-db` | `zn_cys_his.query_app.build_db` | rebuild the query app's SQLite DB |

Each is equivalent to `uv run python -m <module>`.

## Run the clustering pipeline

The orchestrator runs three stages — **prep** (annotate + stats) → **cluster**
(the selected approaches) → **validate** — and is idempotent (skips work whose
output already exists; pass `--force` to redo). Point `--base-dir` at any dataset:

```bash
# Full pipeline. Outputs go to cluster-output/4cys-large/ (mirroring data/).
uv run zch-pipeline --base-dir data/4cys-large

# any other composition — same command, composition is auto-detected
uv run zch-pipeline --base-dir data/2cys2his-large
```

Rerun a single stage (or resume from one) without redoing the rest:

```bash
uv run zch-pipeline --base-dir data/4cys-large --stage cluster --approaches 3  # re-cluster only approach 3
uv run zch-pipeline --base-dir data/4cys-large --stage validate               # redo validation/plots
uv run zch-pipeline --base-dir data/4cys-large --from-stage cluster           # cluster + validate, skip prep
```

Artifacts land under `cluster-output/<dataset>/{prep,approach1,approach3,validation}/`
(override the root with `--out-dir`). Each approach auto-emits
`cluster_pdb_family.csv`. See [docs/pipeline.md](docs/pipeline.md) for the full
description of every step, featurization approach, and parameter.

## Sampling + spectra visualization

`zch-sample-spectra` samples cluster representatives from the test-set clustering
(`cluster-output/test-4cys-weighted/approach1`), protonating each and tagging
`CHARGE=/MULTIPLICITY=`. Charge is derived per structure as `2 − n_cys`
(4cys=−2, 3cys1his=−1, 2cys2his=0, 1cys3his=+1, 4his=+2); multiplicity=1.
Override with `--charge`/`--multiplicity`, or skip with `--no-charge-mult`.
`zch-plot-spectra` reads the precomputed FEFF spectra in
`data/4cys-large/calculated-spectra/` and writes the report/overlays to a sibling
`analysis-and-visualization/`:

```bash
uv run zch-sample-spectra
uv run zch-plot-spectra
```

To run against another dataset, pass `--station` / `--approach` / `--out`
explicitly, or generate a station of precomputed spectra for it first.

## Structure-query app

```bash
uv run streamlit run src/zn_cys_his/query_app/app.py
```

Runs off the prebuilt `src/zn_cys_his/query_app/structures.db`. To rebuild it (after
the pipeline has produced fresh `approach1/kmeans_labels_with_stats.csv` files):

```bash
uv run zch-build-db
```

See [src/zn_cys_his/query_app/README.md](src/zn_cys_his/query_app/README.md) for
app-specific details.
