#!/usr/bin/env python3
"""Shared loader/parser for the per-PDB PROSITE motif data (``motifs.csv``).

``motifs.csv`` is produced by ``build_motif_data.py`` from the 3Cys1His BLAST/
PROSITE workbook — one row per lowercase pdb_id with a ``consensus_name`` and up
to three motif cells. ``motif3`` may itself hold a ``"; "``-joined overflow of
extra motifs, so per-PDB motif *sets* are recovered by splitting every cell on
``;``. Both the Unique PDBs tab (app.py) and the Validation tab's motif-coloring
view (validation_tab.py) read through here so the parsing lives in one place.
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


def load_motifs() -> pd.DataFrame:
    """Load motifs.csv (empty frame with the right columns if it is absent).

    Adds a ``motifs`` column: the full per-PDB motif set joined with '; ' for
    display. pdb_id is lowercase to match the rest of the app.
    """
    cols = ["pdb_id", "consensus_name", "best_hit_name", *_MOTIF_COLS, "motifs"]
    if not CSV_PATH.exists():
        return pd.DataFrame(columns=cols)
    df = pd.read_csv(CSV_PATH, dtype=str)
    df["pdb_id"] = df["pdb_id"].astype(str).str.strip().str.lower()
    for c in _MOTIF_COLS + ("consensus_name", "best_hit_name"):
        if c not in df.columns:
            df[c] = ""
    df["motifs"] = df.apply(lambda r: "; ".join(all_motifs(r)), axis=1)
    return df


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


def enzyme_label_map() -> dict[str, str]:
    """pdb_id (lowercase) -> enzyme label.

    The BLAST **consensus name** when the PDB has one, else its **paper title**;
    PDBs with neither are simply absent (callers treat a miss as 'none').
    """
    consensus: dict[str, str] = {}
    motifs = load_motifs()
    for _, row in motifs.iterrows():
        name = str(row.get("consensus_name") or "").strip()
        if name:
            consensus[row["pdb_id"]] = name
    titles = load_pdb_titles()
    labels: dict[str, str] = {}
    for pid in set(consensus) | set(titles):
        labels[pid] = consensus.get(pid) or titles.get(pid, "")
    return {pid: lab for pid, lab in labels.items() if lab}
