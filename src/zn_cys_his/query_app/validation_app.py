#!/usr/bin/env python3
"""Standalone cluster-validation viewer — one system, no database required.

The main app (``app.py``) bundles Zn-specific tabs (Files / Unique PDBs /
Spectra) that need ``structures.db`` and FEFF spectra.  A different system such
as heme has none of that — only the clustering CSVs under ``validation_data/``.
This entry point renders just the Validation tab, so it works for any profile's
output and can be pointed at a single system.

Run a heme-only viewer:

    # after: zch-pipeline --profile heme ... && python build_validation_data.py
    ZCH_DATASETS=heme streamlit run validation_app.py

Or point at a different output tree entirely:

    ZCH_VALIDATION_DIR=/path/to/heme/validation_data streamlit run validation_app.py

Both env vars are read by ``validation_tab``:
  ZCH_VALIDATION_DIR  where the <dataset>/<approach>/ CSVs live
                      (default: ./validation_data next to this file)
  ZCH_DATASETS        comma-separated allowlist of datasets to show
                      (default: all discovered)
"""
from __future__ import annotations

import os

import streamlit as st

import validation_tab  # sibling module (streamlit adds the app dir to sys.path)

_datasets = os.environ.get("ZCH_DATASETS", "").strip()
_title = "Cluster validation" + (f" — {_datasets}" if _datasets else "")

st.set_page_config(page_title=_title, layout="wide")
st.title(_title)
st.caption("Interactive t-SNE + per-cluster metric/ligand distributions. "
           "Reads validation_data/ CSVs; no database required.")

validation_tab.render()
