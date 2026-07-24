#!/usr/bin/env python3
"""Shared loader/parser for the per-PDB PROSITE motif data (``motifs.csv``).

``motifs.csv`` is produced by ``build_motif_data.py`` from the per-system BLAST/
PROSITE workbooks (3Cys1His, 4Cys) — one row per ``(dataset, lowercase pdb_id)``
with a ``consensus_name`` and up to three motif cells. ``motif3`` may itself hold
a ``"; "``-joined overflow of extra motifs, so per-PDB motif *sets* are recovered
by splitting every cell on ``;``. Both the Unique PDBs tab (app.py) and the
Validation tab's motif-coloring view (validation_tab.py) read through here so the
parsing lives in one place.

Rows are dataset-scoped because the scans are run per system and the same PDB can
carry different values in two workbooks (see build_motif_data.py). A dataset with
no scan of its own (2cys2his) falls back to whatever rows other datasets have for
its PDBs — the motifs describe the same protein sequence, just as queried through
another system's entity.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

CSV_PATH = Path(__file__).resolve().parent / "motifs.csv"
# RCSB metadata cache (built by build_db.py); the source of the paper-title
# fallback used by the enzyme label. Absent in a bare validation-only deploy, in
# which case the fallback is simply unavailable.
META_CACHE_PATH = Path(__file__).resolve().parent / "metadata_cache.json"

_MOTIF_COLS = ("motif1", "motif2", "motif3")


def _clean(cell: object) -> str:
    """Normalise a raw cell to a string ('' for NaN/None/blank)."""
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return ""
    return str(cell).strip()


def split_motifs(*cells: object) -> list[str]:
    """Ordered, de-duplicated motifs across the given cells (split on ';')."""
    out: list[str] = []
    for cell in cells:
        for part in _clean(cell).split(";"):
            m = part.strip()
            if m and m not in out:
                out.append(m)
    return out


def first_motif(row: pd.Series) -> str:
    """The most-prevalent (first-listed) motif for a PDB, '' if none."""
    ms = split_motifs(row.get("motif1"))
    return ms[0] if ms else ""


def all_motifs(row: pd.Series) -> list[str]:
    """Every motif a PDB carries, in prevalence order."""
    return split_motifs(*(row.get(c) for c in _MOTIF_COLS))


_COLS = ["dataset", "pdb_id", "consensus_name", "best_hit_name",
         *_MOTIF_COLS, "motifs"]


def _load_raw() -> pd.DataFrame:
    """Every motifs.csv row (all datasets), normalised and with ``motifs`` added."""
    if not CSV_PATH.exists():
        return pd.DataFrame(columns=_COLS)
    df = pd.read_csv(CSV_PATH, dtype=str)
    df["pdb_id"] = df["pdb_id"].astype(str).str.strip().str.lower()
    if "dataset" not in df.columns:  # CSV built before motifs became per-dataset
        df["dataset"] = ""
    df["dataset"] = df["dataset"].fillna("").astype(str).str.strip()
    for c in _MOTIF_COLS + ("consensus_name", "best_hit_name"):
        if c not in df.columns:
            df[c] = ""
    df["motifs"] = df.apply(lambda r: "; ".join(all_motifs(r)), axis=1)
    return df


def load_motifs(dataset: str | None = None) -> pd.DataFrame:
    """Load motifs.csv (empty frame with the right columns if it is absent).

    One row per lowercase pdb_id, with a ``motifs`` column: the full per-PDB motif
    set joined with '; ' for display.

    ``dataset`` selects the rows scanned for that system. A dataset that was never
    scanned (2cys2his) falls back to the cross-dataset view, so its PDBs still get
    labels wherever another system's scan covered them. Pass None for that
    cross-dataset view directly (first dataset in the CSV wins per pdb_id).
    """
    df = _load_raw()
    if dataset is not None:
        own = df[df["dataset"] == dataset]
        if not own.empty:
            return own.reset_index(drop=True)
    return df.drop_duplicates(subset="pdb_id").reset_index(drop=True)


def annotate(view: pd.DataFrame,
             cols: tuple[str, ...] = ("consensus_name", "motifs")) -> pd.DataFrame:
    """Add motif ``cols`` to a frame keyed by ``pdb_id`` (+ optional ``dataset``).

    Each row takes the values scanned for *its own* dataset, falling back to any
    dataset's row for the same PDB — so a table mixing datasets (the Unique PDBs
    tab) labels every PDB from the system it was clustered in. Missing values
    become ''. Returns a new frame; the original is untouched.
    """
    raw = _load_raw()
    if raw.empty:
        return view
    out = view.copy()
    pdb = out["pdb_id"].astype(str).str.strip().str.lower()
    ds = (out["dataset"].astype(str) if "dataset" in out.columns
          else pd.Series("", index=out.index))
    exact = raw.drop_duplicates(subset=["dataset", "pdb_id"]).set_index(["dataset", "pdb_id"])
    anyds = raw.drop_duplicates(subset="pdb_id").set_index("pdb_id")
    key = pd.MultiIndex.from_arrays([ds, pdb])
    for c in cols:
        own = pd.Series(exact[c].reindex(key).to_numpy(), index=out.index)
        fallback = pd.Series(anyds[c].reindex(pdb).to_numpy(), index=out.index)
        out[c] = own.fillna(fallback).fillna("")
    return out


def datasets_with_motifs() -> set[str]:
    """Dataset names that have a motif scan of their own in motifs.csv."""
    return {d for d in _load_raw()["dataset"] if d}


def load_pdb_titles() -> dict[str, str]:
    """pdb_id (lowercase) -> paper (primary-citation) title, from the RCSB cache."""
    if not META_CACHE_PATH.exists():
        return {}
    try:
        data = json.loads(META_CACHE_PATH.read_text())
    except (OSError, ValueError):
        return {}
    titles: dict[str, str] = {}
    for code, rec in (data or {}).items():
        title = str((rec or {}).get("citation_title") or "").strip()
        if title:
            titles[str(code).strip().lower()] = title
    return titles


def enzyme_label_map(dataset: str | None = None) -> dict[str, str]:
    """pdb_id (lowercase) -> enzyme label, for one dataset (or all; see load_motifs).

    The BLAST **consensus name** when the PDB has one, else its **paper title**;
    PDBs with neither are simply absent (callers treat a miss as 'none').
    """
    consensus: dict[str, str] = {}
    motifs = load_motifs(dataset)
    for _, row in motifs.iterrows():
        name = str(row.get("consensus_name") or "").strip()
        if name:
            consensus[row["pdb_id"]] = name
    titles = load_pdb_titles()
    labels: dict[str, str] = {}
    for pid in set(consensus) | set(titles):
        labels[pid] = consensus.get(pid) or titles.get(pid, "")
    return {pid: lab for pid, lab in labels.items() if lab}
