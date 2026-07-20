#!/usr/bin/env python3
"""Streamlit app to query clustered Zn-site structures across cys/his datasets.

Run:  streamlit run app.py   (from this directory)

Data comes from ``structures.db`` (built by ``build_db.py``):
  - table ``structures``   : one row per clustered structure/file
  - table ``pdb_metadata`` : one row per unique RCSB PDB entry (title, authors,
                             publication year, journal, DOI/PubMed, is_published)

The sidebar builds a parameterized SQL query against ``structures`` joined to
``pdb_metadata``. Every numeric stat gets its own range filter, generated
automatically from the database schema; a range left at its full extent imposes
no filter (so everything, including rows with missing values, is included by
default). Results are shown as (a) every matching file and (b) the deduplicated
set of unique PDBs with clickable links to rcsb.org.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

import validation_tab  # sibling module (streamlit adds the app dir to sys.path)

DB_PATH = Path(__file__).resolve().parent / "structures.db"

# Path segments that mark the repo root, for normalising any legacy absolute path.
_PATH_ANCHORS = ("cluster-output", "data")


def rel_xyz(p: object) -> object:
    """Show xyz_path as a repo-root-relative POSIX path.

    Paths are stored repo-relative by build_db.py; this also normalises any legacy
    absolute value (e.g. a DB built before that change) to the same form.
    """
    if not isinstance(p, str) or not p:
        return p
    path = Path(p)
    if not path.is_absolute():
        return path.as_posix()
    parts = path.parts
    for anchor in _PATH_ANCHORS:
        if anchor in parts:
            return Path(*parts[parts.index(anchor):]).as_posix()
    return p

# Columns that are numeric in the DB but are not "stats" to range-filter on.
NON_STAT_REAL: set[str] = set()  # (all REAL columns here are stats)

# Group stat columns into sidebar sections purely by name, for readability.
def stat_group(col: str) -> str:
    if "dihedral" in col:
        return "Dihedrals (deg)"
    if "bfactor" in col:
        return "B-factors"
    if col in ("resolution_A", "r_work", "r_free"):
        return "Crystallographic quality"
    return "Geometry"


# Sections rendered expanded by default (others collapse to reduce clutter).
OPEN_GROUPS = {"Crystallographic quality", "Geometry"}

# Prettier labels for common columns; anything else is title-cased generically.
LABELS = {
    "resolution_A": "Resolution (Å)",
    "volume_A3": "Volume (Å³)",
    "q_tetra_coord": "q_tetra (coord)",
    "q_tetra_ca": "q_tetra (Cα)",
    "r_work": "R_work",
    "r_free": "R_free",
    "zn_bfactor": "Zn B-factor",
    "cys_dihedral_mean_deg": "Dihedral mean",
    "all_coord_res_bfactor_avg": "All coord-res B-factor",
}


def pretty(col: str) -> str:
    if col in LABELS:
        return LABELS[col]
    label = col
    for suffix in ("_deg", "_avg", "_A3", "_A"):
        label = label.replace(suffix, "")
    return label.replace("_", " ").strip().capitalize()


st.set_page_config(page_title="Zn-site structure query", layout="wide")


@st.cache_resource
def get_conn() -> sqlite3.Connection:
    return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, check_same_thread=False)


@st.cache_data
def stat_columns() -> list[str]:
    """All REAL (float) columns in `structures` = the numeric stats to filter."""
    con = get_conn()
    cols = [r[1] for r in con.execute("PRAGMA table_info(structures)").fetchall()
            if r[2] == "REAL" and r[1] not in NON_STAT_REAL]
    return cols


@st.cache_data
def col_bounds(col: str) -> tuple[float, float]:
    con = get_conn()
    lo, hi = con.execute(
        f"SELECT MIN({col}), MAX({col}) FROM structures WHERE {col} IS NOT NULL"
    ).fetchone()
    return (float(lo) if lo is not None else 0.0,
            float(hi) if hi is not None else 0.0)


@st.cache_data
def datasets() -> list[str]:
    con = get_conn()
    return [r[0] for r in con.execute(
        "SELECT DISTINCT dataset FROM structures ORDER BY dataset"
    ).fetchall()]


@st.cache_data
def clusters_for(selected: tuple[str, ...]) -> list[int]:
    if not selected:
        return []
    con = get_conn()
    ph = ",".join("?" * len(selected))
    return [r[0] for r in con.execute(
        f"SELECT DISTINCT cluster FROM structures WHERE dataset IN ({ph}) ORDER BY cluster",
        selected,
    ).fetchall()]


def run_query(where: str, params: list) -> pd.DataFrame:
    con = get_conn()
    sql = f"""
        SELECT s.*,
               m.title, m.authors, m.release_year, m.deposit_year,
               m.journal, m.citation_title, m.citation_year,
               m.doi, m.pubmed_id, m.is_published
        FROM structures s
        LEFT JOIN pdb_metadata m ON s.pdb_id = m.pdb_id
        {('WHERE ' + where) if where else ''}
    """
    df = pd.read_sql(sql, con, params=params)
    if "citation_year" in df.columns:  # INTEGER + NULLs comes back float; keep it int
        df["citation_year"] = df["citation_year"].astype("Int64")
    return df


# ----------------------------------------------------------------------------- sidebar
st.sidebar.title("Filters")

sel_datasets = st.sidebar.multiselect(
    "Dataset (cys/his composition)", datasets(), default=datasets(),
    help="Query one composition or across all of them.",
)

cluster_opts = clusters_for(tuple(sel_datasets))
sel_clusters = st.sidebar.multiselect(
    "Cluster id", cluster_opts, default=[],
    help="Empty = all clusters. Cluster ids are per-dataset.",
)

pub_only = st.sidebar.checkbox(
    "Published entries only", value=False,
    help="Has a journal citation with a DOI or PubMed id.",
)

st.sidebar.markdown("---")
st.sidebar.caption("Stat ranges — leave a slider at its full width to include "
                   "everything (rows with missing values are kept until you narrow it).")

# Generate one range slider per numeric stat, organised into collapsible groups.
groups: dict[str, list[str]] = {}
for col in stat_columns():
    groups.setdefault(stat_group(col), []).append(col)

# Stable, sensible group ordering.
GROUP_ORDER = ["Crystallographic quality", "Geometry", "Dihedrals (deg)", "B-factors"]
ordered_groups = [g for g in GROUP_ORDER if g in groups] + \
                 [g for g in groups if g not in GROUP_ORDER]

active_ranges: list[tuple[str, float, float]] = []
for group in ordered_groups:
    with st.sidebar.expander(group, expanded=group in OPEN_GROUPS):
        for col in groups[group]:
            lo, hi = col_bounds(col)
            if hi <= lo:  # constant or empty column: nothing to filter
                continue
            pad = (hi - lo) * 0.001  # avoid float-edge clipping of the extremes
            smin, smax = lo - pad, hi + pad
            rng = st.slider(
                pretty(col), min_value=smin, max_value=smax,
                value=(smin, smax), key=f"sl_{col}",
            )
            # Only filter if the user narrowed the range from its full extent.
            if rng[0] > smin or rng[1] < smax:
                active_ranges.append((col, rng[0], rng[1]))

# ----------------------------------------------------------------------------- build WHERE
where_parts: list[str] = []
params: list = []

if sel_datasets:
    where_parts.append(f"s.dataset IN ({','.join('?' * len(sel_datasets))})")
    params += sel_datasets
else:
    where_parts.append("1=0")  # no dataset selected -> no rows

if sel_clusters:
    where_parts.append(f"s.cluster IN ({','.join('?' * len(sel_clusters))})")
    params += sel_clusters

for col, lo, hi in active_ranges:
    where_parts.append(f"s.{col} BETWEEN ? AND ?")
    params += [lo, hi]

if pub_only:
    where_parts.append("m.is_published = 1")

df = run_query(" AND ".join(where_parts), params)
df = df.drop(columns=["has_stats"], errors="ignore")  # constant (always 1)

# ----------------------------------------------------------------------------- main
st.title("Zn-site structure query")
st.caption("Cluster / stat-range / publication filters over 3cys1his, 4cys and "
           "2cys2his datasets, joined to RCSB publication metadata.")

unique_df = df.drop_duplicates(subset="pdb_id")
c1, c2, c3 = st.columns(3)
c1.metric("Matching files", f"{len(df):,}")
c2.metric("Unique PDBs", f"{unique_df['pdb_id'].nunique():,}")
c3.metric("Published PDBs", f"{int(unique_df['is_published'].fillna(0).sum()):,}")

active_summary = (", ".join(f"{pretty(c)} ∈ [{lo:.3g}, {hi:.3g}]"
                            for c, lo, hi in active_ranges) or "none")
st.caption(f"Active stat ranges: {active_summary}")

with st.expander("Show generated SQL"):
    st.code("WHERE " + (" AND ".join(where_parts) if where_parts else "(none)")
            + "\n-- params: " + repr(params), language="sql")

tab_files, tab_pdbs, tab_validation = st.tabs(
    ["📄 Files", "🔗 Unique PDBs", "📊 Validation"])

with tab_files:
    files_view = df.copy()
    if "xyz_path" in files_view.columns:
        files_view["xyz_path"] = files_view["xyz_path"].map(rel_xyz)
    # RCSB link right next to pdb_id.
    link = "https://www.rcsb.org/structure/" + files_view["pdb_id"].str.upper()
    files_view.insert(list(files_view.columns).index("pdb_id") + 1, "rcsb_link", link)

    st.write(f"**{len(files_view):,}** files match the current filters.")

    # Render cluster_color as a solid colour swatch (background == the hex value).
    def _swatch(series: pd.Series) -> list[str]:
        return [f"background-color: {v}; color: {v}"
                if isinstance(v, str) and v.startswith("#") else "" for v in series]

    styled = files_view.style
    if "cluster_color" in files_view.columns:
        styled = styled.apply(_swatch, subset=["cluster_color"])
    # A Styler's display value overrides LinkColumn's display_text, so set the
    # shown text to the arrow here; the raw URL is still used as the href.
    styled = styled.format({"rcsb_link": lambda _v: "open ↗"})
    st.dataframe(
        styled, use_container_width=True, hide_index=True,
        column_config={
            "rcsb_link": st.column_config.LinkColumn("RCSB", display_text="open ↗"),
            "cluster_color": st.column_config.TextColumn("cluster_color", width="small"),
        },
    )
    st.download_button(
        "Download files CSV", files_view.to_csv(index=False).encode(),
        file_name="query_files.csv", mime="text/csv",
    )

with tab_pdbs:
    meta_cols = ["pdb_id", "dataset", "title", "authors", "release_year",
                 "journal", "citation_title", "doi", "pubmed_id", "is_published"]
    cols = [c for c in meta_cols if c in unique_df.columns]
    view = unique_df[cols].copy()
    view.insert(1, "rcsb_link", "https://www.rcsb.org/structure/" + view["pdb_id"].str.upper())
    view["is_published"] = view["is_published"].fillna(0).astype(int).map({1: "✅", 0: ""})
    view = view.sort_values(["dataset", "pdb_id"]).reset_index(drop=True)

    st.write(f"**{len(view):,}** unique PDB entries. Click a link to open it on RCSB.")
    st.dataframe(
        view, use_container_width=True, hide_index=True,
        column_config={
            "rcsb_link": st.column_config.LinkColumn("RCSB", display_text="open ↗"),
            "pubmed_id": st.column_config.TextColumn("PubMed"),
            "is_published": st.column_config.TextColumn("Pub?"),
        },
    )
    st.download_button(
        "Download unique PDBs CSV", view.to_csv(index=False).encode(),
        file_name="query_unique_pdbs.csv", mime="text/csv",
    )

with tab_validation:
    # Native rebuild of the pipeline's cluster-distribution HTML reports; reads its
    # own CSVs under validation_data/ and is independent of the sidebar filters.
    validation_tab.render()
