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
  clickable `rcsb.org/structure/<code>` links plus publication metadata
  (download as CSV).

Original `.pdb` files live in each config's `pdb-files/`.
