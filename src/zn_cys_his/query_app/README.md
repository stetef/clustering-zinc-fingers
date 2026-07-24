# Structure-query app

Query clustered Zn-site structures across the **3cys1his**, **4cys**, and
**2cys2his** datasets, joined to RCSB publication metadata.

## Files
- `build_db.py` — copies the three source `kmeans_labels_with_stats.csv` files
  into `csv/`, then builds `structures.db`.
- `structures.db` — SQLite database with two tables:
  - `structures` — one row per clustered file. All datasets unified with a
    `dataset` column and a derived `pdb_id` (4-char RCSB code). 4cys rows have
    NULL His-Bfactor columns (that dataset has none).
  - `pdb_metadata` — one row per unique PDB (title, deposition authors,
    deposit/release year, journal, citation title, DOI, PubMed id,
    `is_published`), fetched from the RCSB GraphQL API.
- `metadata_cache.json` — cached RCSB responses so rebuilds are fast/offline.
- `csv/` — copies of the three source CSVs (`3cys1his.csv`, `4cys.csv`,
  `2cys2his.csv`).
- `build_motif_data.py` / `motifs.csv` — distils the per-system BLAST/PROSITE
  workbooks (`blast-output/3cys1his-large/3Cys1His_pdb_motif_and_enzyme.xlsx`,
  `blast-output/4cys-large/4Cys_pdb_motif_and_enzyme.xlsx`, `overview` sheet)
  into a small per-PDB table (consensus enzyme name + top motifs), keyed by
  `(dataset, lowercase pdb_id)` — each system is scanned separately, and the
  366 PDBs shared by the two scans can disagree, so the dataset is part of the
  key. Needs `openpyxl` to read the `.xlsx`; the app itself only reads the CSV.
  Powers the **Consensus name / Motifs** columns in the Unique PDBs tab and the
  **color-by-motif** / **color-by-enzyme** views in Validation. `2cys2his` has
  no scan of its own and borrows another system's row where the PDB is covered
  (~76 of its 284 PDBs). Add a system by listing its workbook in
  `WORKBOOKS` at the top of `build_motif_data.py` and re-running it:

  ```bash
  uv run --with openpyxl python src/zn_cys_his/query_app/build_motif_data.py
  ```

## Rebuild the database
```bash
# from the repo root
uv run zch-build-db       # re-copies CSVs, refreshes DB (uses metadata cache)
```
Delete `metadata_cache.json` (in this directory) first to force a fresh RCSB fetch.

## Run the app
```bash
# from the repo root
uv run streamlit run src/zn_cys_his/query_app/app.py
```

### What you can do
- Choose one dataset or query across all three.
- Filter by `R_free` (e.g. `< 0.1`), resolution, cluster id, and any dihedral
  range (mean or individual χ angles), and restrict to published entries.
- **Files** tab: every matching structure file (download as CSV).
- **Unique PDBs** tab: deduplicated PDB list (files can share a PDB id) with
  clickable `rcsb.org/structure/<code>` links, the BLAST consensus name +
  PROSITE motifs (before the title), plus publication metadata
  (download as CSV).
- **Validation** tab: color the t-SNE by cluster, family, motif, or enzyme
  (BLAST consensus name, falling back to the paper title). Motif coloring has to
  pick one motif per point even though ~46% of these PDBs carry several (up to
  11), so:
  - **hover** any point for its PDB, cluster, the motif it is colored by, and
    its complete motif list;
  - **"All (dominant motif in scope)"** colors each PDB by whichever of *its*
    motifs is commonest within the current scope — a box/lasso selection if you
    make one, else the focus cluster, else the whole dataset. Narrowing the
    scope re-labels multi-motif PDBs by what dominates *that* region (e.g. in
    one PHD-finger region 15 of 57 points flip `ZF_PHD_2` → `ZF_PHD_1`).
    Out-of-scope points stay as faint context; a button resets to the whole
    dataset. Streamlit exposes only plotly *selection* events (no zoom /
    relayout), so box-select stands in for "zoom to here";
  - picking a **single motif** instead highlights every PDB carrying it,
    wherever it sits in that PDB's list;
  - only the N commonest labels get their own color (slider, default 12) — there
    are ~80 distinct first-motifs in 4cys against a 15-color palette — and the
    rest fold into one `other` entry.

  The **Cluster ↔ labels** sub-tab quantifies
  how well the geometry clusters agree with those chemical labels:
  - pairwise **adjusted mutual information** (AMI) among cluster / family / motif
    / enzyme, as a heatmap;
  - **homogeneity / completeness / V-measure** of each label vs the clustering
    (label = ground truth, cluster id = prediction);
  - a per-cluster **composition breakdown** — pick a focus cluster to see the
    enzymes (or family/motif) that make it up, in descending percentage order,
    so the outlier members of a cluster are easy to spot.

Original `.pdb` files live in each config's `pdb-files/`.
