#!/usr/bin/env python3
"""Build the SQLite database that powers the structure-query Streamlit app.

Steps
-----
1. Copy the three source ``kmeans_labels_with_stats.csv`` files into ``csv/``
   (renamed by cys/his composition).
2. Load them into one unified ``structures`` table, tagging each row with its
   ``dataset`` and a derived ``pdb_id`` (the 4-char RCSB code = token before the
   first underscore in ``id``).
3. Fetch per-PDB metadata (title, authors, publication year, journal, DOI,
   PubMed id, is_published) from the RCSB GraphQL API for every unique PDB and
   store it in a ``pdb_metadata`` table.

Metadata fetches are cached in ``metadata_cache.json`` so re-runs are cheap and
work offline once populated. Run from anywhere:  ``python build_db.py``
"""
from __future__ import annotations

import json
import shutil
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import requests

from zn_cys_his.paths import CLUSTER_OUTPUT, DATA_DIR, REPO_ROOT

HERE = Path(__file__).resolve().parent
REPO = REPO_ROOT
CSV_DIR = HERE / "csv"
DB_PATH = HERE / "structures.db"
CACHE_PATH = HERE / "metadata_cache.json"

# dataset key -> (approach1 clustering output CSV, pdb-files dir for the original .pdb).
# Clustering artifacts live under cluster-output/ (the pipeline's mirror of data/);
# the original PDBs are pristine inputs under data/.
DATASETS = {
    "3cys1his": {
        "csv": CLUSTER_OUTPUT / "3cys1his-large/approach1/kmeans_labels_with_stats.csv",
        "pdb_dir": DATA_DIR / "3cys1his-large/pdb-files",
    },
    "4cys": {
        "csv": CLUSTER_OUTPUT / "test-4cys-weighted/approach1/kmeans_labels_with_stats.csv",
        "pdb_dir": DATA_DIR / "4cys-large/pdb-files",
    },
    "2cys2his": {
        "csv": CLUSTER_OUTPUT / "2cys2his-large/approach1/kmeans_labels_with_stats.csv",
        "pdb_dir": DATA_DIR / "2cys2his-large/pdb-files",
    },
}

RCSB_GRAPHQL = "https://data.rcsb.org/graphql"
GRAPHQL_QUERY = """
query($ids:[String!]!){
  entries(entry_ids:$ids){
    rcsb_id
    struct { title }
    rcsb_accession_info { deposit_date initial_release_date }
    audit_author { name }
    rcsb_primary_citation {
      title
      rcsb_journal_abbrev
      year
      pdbx_database_id_DOI
      pdbx_database_id_PubMed
    }
  }
}
"""


def pdb_code(struct_id: str) -> str:
    """Derive the RCSB 4-char code from a structure id (token before first '_')."""
    return struct_id.split("_")[0].lower()


def copy_and_load() -> pd.DataFrame:
    """Copy source CSVs into csv/ and return one unified DataFrame."""
    CSV_DIR.mkdir(exist_ok=True)
    frames = []
    for key, cfg in DATASETS.items():
        src = cfg["csv"]
        if not src.exists():
            sys.exit(f"ERROR: source CSV not found: {src}")
        dest = CSV_DIR / f"{key}.csv"
        shutil.copyfile(src, dest)
        print(f"copied {src.relative_to(REPO)} -> csv/{key}.csv")
        df = pd.read_csv(dest)
        df.insert(0, "dataset", key)
        df.insert(1, "pdb_id", df["id"].map(pdb_code))
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True, sort=False)
    print(f"loaded {len(combined)} rows across {len(frames)} datasets "
          f"({combined['pdb_id'].nunique()} unique PDBs)")
    return combined


def fetch_metadata(pdb_ids: list[str]) -> dict[str, dict]:
    """Fetch RCSB metadata for the given (lowercase) pdb ids, using a cache."""
    cache: dict[str, dict] = {}
    if CACHE_PATH.exists():
        cache = json.loads(CACHE_PATH.read_text())

    missing = sorted({p for p in pdb_ids if p not in cache})
    print(f"metadata: {len(pdb_ids)} unique PDBs, {len(missing)} to fetch, "
          f"{len(pdb_ids) - len(missing)} cached")

    batch = 150
    for i in range(0, len(missing), batch):
        chunk = missing[i:i + batch]
        try:
            r = requests.post(
                RCSB_GRAPHQL,
                json={"query": GRAPHQL_QUERY, "variables": {"ids": [c.upper() for c in chunk]}},
                timeout=60,
            )
            r.raise_for_status()
            entries = (r.json().get("data") or {}).get("entries") or []
        except Exception as exc:  # noqa: BLE001 - network hiccup, keep going
            print(f"  WARN batch {i // batch}: {exc}")
            entries = []

        got = set()
        for e in entries:
            code = e["rcsb_id"].lower()
            got.add(code)
            cit = e.get("rcsb_primary_citation") or {}
            acc = e.get("rcsb_accession_info") or {}
            journal = cit.get("rcsb_journal_abbrev")
            doi = cit.get("pdbx_database_id_DOI")
            pmid = cit.get("pdbx_database_id_PubMed")
            is_pub = bool(
                journal and journal.strip().lower() not in ("to be published", "")
                and (doi or pmid)
            )
            cache[code] = {
                "pdb_id": code,
                "title": (e.get("struct") or {}).get("title"),
                "deposit_year": (acc.get("deposit_date") or "")[:4] or None,
                "release_year": (acc.get("initial_release_date") or "")[:4] or None,
                "authors": ", ".join(a.get("name", "") for a in (e.get("audit_author") or [])),
                "citation_title": cit.get("title"),
                "journal": journal,
                "citation_year": cit.get("year"),
                "doi": doi,
                "pubmed_id": pmid,
                "is_published": int(is_pub),
            }
        # entries RCSB didn't return (obsolete/withdrawn ids): record a stub
        for code in chunk:
            if code not in got:
                cache[code] = {"pdb_id": code, "title": None, "deposit_year": None,
                               "release_year": None, "authors": None, "citation_title": None,
                               "journal": None, "citation_year": None, "doi": None,
                               "pubmed_id": None, "is_published": 0}
        print(f"  fetched {min(i + batch, len(missing))}/{len(missing)}")
        CACHE_PATH.write_text(json.dumps(cache, indent=0))

    return {p: cache[p] for p in pdb_ids if p in cache}


def main() -> None:
    combined = copy_and_load()
    meta = fetch_metadata(sorted(combined["pdb_id"].unique().tolist()))
    meta_df = pd.DataFrame(list(meta.values()))

    if DB_PATH.exists():
        DB_PATH.unlink()
    with sqlite3.connect(DB_PATH) as con:
        combined.to_sql("structures", con, index=False)
        meta_df.to_sql("pdb_metadata", con, index=False)
        cur = con.cursor()
        cur.execute("CREATE INDEX idx_struct_dataset ON structures(dataset)")
        cur.execute("CREATE INDEX idx_struct_pdb ON structures(pdb_id)")
        cur.execute("CREATE INDEX idx_struct_cluster ON structures(dataset, cluster)")
        cur.execute("CREATE UNIQUE INDEX idx_meta_pdb ON pdb_metadata(pdb_id)")
        con.commit()

    published = int(meta_df["is_published"].sum()) if not meta_df.empty else 0
    print(f"\nwrote {DB_PATH.relative_to(REPO)}")
    print(f"  structures: {len(combined)} rows")
    print(f"  pdb_metadata: {len(meta_df)} PDBs ({published} published)")


if __name__ == "__main__":
    main()
