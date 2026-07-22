#!/usr/bin/env python3
"""Native Streamlit rendering of the cluster-validation reports.

The clustering pipeline emits a large, self-contained ``report_cluster_
distribution.html`` per dataset/approach (hand-built HTML + embedded PNGs). This
module reproduces the same information natively from the two CSVs those PNGs are
themselves rendered from — so the Validation tab is interactive, light, and ships
with a standalone deploy (no 12 MB HTML blobs, no cluster-output/ dependency).

Data lives in ``validation_data/<dataset>/<approach>/`` (populated by
``build_validation_data.py``):

  kmeans_labels_with_stats.csv  per-structure cluster id, cluster_color, family,
                                and the numeric metrics binned by the histograms
  embeddings.csv                frozen t-SNE coords (tsne1/tsne2) per structure

The t-SNE coords are read as-is (never recomputed — t-SNE is nondeterministic and
slow), so the whole tab is zero-compute.

Interaction model (Streamlit-native redesign): a "focus cluster" drives every
panel. It is set either from the sidebar-style selectbox or by box/lasso-selecting
points on the t-SNE scatter; both funnel through one session-state value.
"""
from __future__ import annotations

import math
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

HERE = Path(__file__).resolve().parent
# Data location is overridable so a local copy can point at a different output
# tree (e.g. a heme-only validation_data).  ZCH_DATASETS is an optional
# comma-separated allowlist that restricts which datasets the tab shows.
VALIDATION_DIR = Path(os.environ.get("ZCH_VALIDATION_DIR", HERE / "validation_data"))
_DATASET_ALLOW = {d.strip() for d in os.environ.get("ZCH_DATASETS", "").split(",") if d.strip()}

# Same 8 metrics the pipeline's per-cluster histograms bin (utils._NUMERIC_PLOT_
# METRICS), with plain-text labels (plotly, not matplotlib LaTeX).
# Union of metrics across all profiles; only columns actually present in a given
# dataset's CSV are shown, so Zn and heme datasets each display just their own.
NUMERIC_METRICS: list[tuple[str, str]] = [
    ("volume_A3", "Volume (Å³)"),
    ("q_tetra_coord", "q_tetra (coord)"),
    ("q_tetra_ca", "q_tetra (Cα)"),
    ("r_work", "R_work"),
    ("r_free", "R_free"),
    ("zn_bfactor", "Zn B-factor"),
    ("cys_dihedral_mean_deg", "Dihedral mean (°)"),
    ("all_coord_res_bfactor_avg", "Coord-res B̄"),
    # heme profile metrics
    ("fe_bfactor", "Fe B-factor"),
    ("avg_bfactor", "Avg B-factor (non-Fe)"),
]

# Plain-language definitions of each structure metric, sourced from the pipeline's
# computation code (clustering/step02_compute_stats.py). The first entries match
# NUMERIC_METRICS in order; "Family" (categorical) is appended last. Inline $…$ is
# rendered as LaTeX by st.markdown.
METRIC_DOCS: list[tuple[str, str]] = [
    ("Volume (Å³)",
     "Volume of the tetrahedron formed by the four coordinating residues' **Cα** atoms "
     r"(scalar triple product, $V=\tfrac16\,\lvert\,\vec v_1\cdot(\vec v_2\times\vec v_3)\,\rvert$). "
     "Defined only for 4-coordinate sites."),
    ("q_tetra (coord)",
     "Errington–Debenedetti tetrahedral order parameter for the four **coordinating ligand "
     "atoms** (Cys Sγ, His Nδ1/Nε2), taking the **Zn as the vertex**: "
     r"$q = 1 - \tfrac38\sum_{i<j}\bigl(\cos\theta_{ij}+\tfrac13\bigr)^2$, where "
     r"$\theta_{ij}$ is the ligand–Zn–ligand angle. **1 = perfect tetrahedron, 0 = random.**"),
    ("q_tetra (Cα)",
     "Same order parameter, but over the four **Cα** atoms (angles Cα–Zn–Cα). Reflects the "
     "tetrahedrality of the backbone framework rather than the direct ligands."),
    ("R_work",
     "Crystallographic R-factor (working set) from the deposited structure — agreement between "
     "the refined model and the observed diffraction data. Lower is better (~0.15–0.25 typical). "
     "A data-quality metric, not geometry."),
    ("R_free",
     "Cross-validated R-factor on a held-out set of reflections excluded from refinement; guards "
     "against overfitting. Usually slightly above R_work; lower is better."),
    ("Zn B-factor",
     "Crystallographic B-factor (atomic displacement parameter) of the Zn ion — higher means "
     "more positional uncertainty / thermal motion."),
    ("Dihedral mean (°)",
     "Mean of the per-residue **Zn→ligand→Cβ→Cα** dihedral angles (ligand = Sγ for Cys, "
     "Nδ1/Nε2 for His), averaged over the coordinating residues. Range −180° to +180°; describes "
     "each sidechain's rotation about the metal–ligand bond. *(Includes His ligands despite the "
     "‘cys’ column name.)*"),
    ("Coord-res B̄",
     "Average B-factor across **all coordinating residues**. Each residue contributes the mean "
     "B-factor of its atoms up to Cα (Cys: Cα, Cβ, Sγ; His: Cα, Cβ, Cγ, Nδ1, Cδ2, Cε1, Nε2); "
     "the Zn site itself is excluded."),
    ("Family",
     "A compact code for the **coordinating motif** — the sequence order of the four ligand "
     "residues and the secondary structure they sit in — in two parts split by `-`:\n\n"
     "- **Residue order & spacing** (before the `-`): one letter per ligand along the sequence — "
     "**C** = Cys, **H** = His — with `x`*n* giving the number of residues *between* consecutive "
     "ligands. Ligands on separate chains are split by an extra `-`.\n"
     "- **Secondary structure** (after the `-`): one letter per residue, same order — "
     "**H** = α-helix, **S** = β-sheet, **L** = loop (irregular).\n\n"
     "Example: `Cx5Hx65Cx1C-HHLL` → Cys, 5-residue gap, His, 65-residue gap, Cys, 1-residue gap, "
     "Cys; those four residues sit in helix, helix, loop, loop."),
]

_GRAY = "#cccccc"
_FALLBACK_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
]


# --------------------------------------------------------------------------- data
def available_datasets() -> list[str]:
    if not VALIDATION_DIR.is_dir():
        return []
    names = sorted(p.name for p in VALIDATION_DIR.iterdir() if p.is_dir())
    if _DATASET_ALLOW:
        names = [n for n in names if n in _DATASET_ALLOW]
    return names


def approaches_for(dataset: str) -> list[str]:
    d = VALIDATION_DIR / dataset
    if not d.is_dir():
        return []
    return sorted(p.name for p in d.iterdir()
                  if p.is_dir() and (p / "kmeans_labels_with_stats.csv").exists())


@st.cache_data
def load_labels(dataset: str, approach: str) -> pd.DataFrame:
    df = pd.read_csv(VALIDATION_DIR / dataset / approach / "kmeans_labels_with_stats.csv")
    df["cluster"] = df["cluster"].astype(int)
    return df


@st.cache_data
def load_tsne(dataset: str, approach: str) -> pd.DataFrame:
    """t-SNE points joined to their cluster + color (one row per structure)."""
    emb = pd.read_csv(VALIDATION_DIR / dataset / approach / "embeddings.csv",
                      usecols=lambda c: c in ("id", "tsne1", "tsne2"))
    lab = load_labels(dataset, approach)
    keep = ["id", "cluster"]
    if "cluster_color" in lab.columns:
        keep.append("cluster_color")
    if "family" in lab.columns:
        keep.append("family")
    merged = emb.merge(lab[keep], on="id", how="inner")
    merged["cluster"] = merged["cluster"].astype(int)
    return merged


def color_map_for(labels_df: pd.DataFrame) -> dict[int, str]:
    clusters = sorted(labels_df["cluster"].unique())
    cmap: dict[int, str] = {}
    if "cluster_color" in labels_df.columns:
        for c in clusters:
            vals = labels_df.loc[labels_df["cluster"] == c, "cluster_color"].dropna()
            v = vals.iloc[0] if len(vals) else None
            cmap[c] = v if isinstance(v, str) and v.startswith("#") else None
    for i, c in enumerate(clusters):
        if not cmap.get(c):
            cmap[c] = _FALLBACK_PALETTE[i % len(_FALLBACK_PALETTE)]
    return cmap


# ------------------------------------------------------------------------- figures
def tsne_figure(tsne_df: pd.DataFrame, color_map: dict[int, str], focus) -> go.Figure:
    """Scatter of frozen t-SNE coords, one legend entry per cluster.

    When a cluster is focused, the rest dim to context; customdata carries the
    cluster id so a box/lasso selection can be mapped back to a cluster.
    """
    fig = go.Figure()
    for c in sorted(color_map):
        d = tsne_df[tsne_df["cluster"] == c]
        if d.empty:
            continue
        focused = focus == "All" or focus == c
        fig.add_trace(go.Scattergl(
            x=d["tsne1"], y=d["tsne2"], mode="markers", name=str(c),
            customdata=np.full((len(d), 1), c),
            marker=dict(color=color_map[c], size=7 if focused else 5,
                        opacity=0.85 if focused else 0.12,
                        line=dict(width=0)),
            hovertemplate=f"cluster {c}<extra></extra>",
        ))
    fig.update_layout(
        height=560, margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(title="cluster", itemsizing="constant", font=dict(size=10)),
        xaxis_title="t-SNE 1", yaxis_title="t-SNE 2", dragmode="select",
    )
    return fig


def family_color_map(tsne_df: pd.DataFrame) -> dict[str, str]:
    """Stable color per ligand family, most-common first (blanks -> 'none')."""
    fams = (tsne_df["family"].fillna("").astype(str).str.strip()
            .replace("", "none"))
    order = [f for f, _ in Counter(fams).most_common()]
    return {f: _FALLBACK_PALETTE[i % len(_FALLBACK_PALETTE)] for i, f in enumerate(order)}


def tsne_family_figure(tsne_df: pd.DataFrame, fam_colors: dict[str, str]) -> go.Figure:
    """Scatter of frozen t-SNE coords, one legend entry per ligand family.

    This is the 'color by ligand' view: it shows whether the geometry-driven
    clusters line up with chemical ligand identity (they largely do not for heme —
    which is the point of being able to look).
    """
    fams = (tsne_df["family"].fillna("").astype(str).str.strip().replace("", "none"))
    fig = go.Figure()
    for f in fam_colors:
        d = tsne_df[fams == f]
        if d.empty:
            continue
        fig.add_trace(go.Scattergl(
            x=d["tsne1"], y=d["tsne2"], mode="markers", name=f,
            marker=dict(color=fam_colors[f], size=7, opacity=0.85, line=dict(width=0)),
            hovertemplate=f"{f}<extra></extra>",
        ))
    fig.update_layout(
        height=560, margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(title="ligand", itemsizing="constant", font=dict(size=10)),
        xaxis_title="t-SNE 1", yaxis_title="t-SNE 2",
    )
    return fig


def _hist_bins(values: np.ndarray) -> np.ndarray:
    n = len(values)
    bins = min(30, max(5, n // 10))
    lo, hi = float(np.min(values)), float(np.max(values))
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5
    return np.linspace(lo, hi, bins + 1)


def distribution_figure(labels_df: pd.DataFrame, focus, color: str) -> go.Figure:
    """Per-metric histograms: gray = whole dataset, color = focused cluster.

    Both are normalised to fraction of the *full* dataset (matching the pipeline's
    per-cluster row plots), so the focused overlay reads as its share of the whole.
    """
    metrics = [(c, l) for c, l in NUMERIC_METRICS if c in labels_df.columns]
    has_family = "family" in labels_df.columns and labels_df["family"].astype(str).str.strip().any()
    panels = metrics + ([("family", "Family")] if has_family else [])

    ncols = min(4, len(panels))
    nrows = math.ceil(len(panels) / ncols)
    fig = make_subplots(rows=nrows, cols=ncols,
                        subplot_titles=[lbl for _, lbl in panels],
                        horizontal_spacing=0.06, vertical_spacing=0.14)

    cluster_rows = labels_df if focus == "All" else labels_df[labels_df["cluster"] == focus]

    for i, (col, _lbl) in enumerate(panels):
        r, cpos = i // ncols + 1, i % ncols + 1
        if col == "family":
            counts = Counter(f for f in cluster_rows["family"].astype(str).str.strip() if f)
            fams = [f for f, _ in counts.most_common(20)]
            fig.add_trace(go.Bar(x=fams, y=[counts[f] for f in fams],
                                 marker_color=(color if focus != "All" else _GRAY),
                                 showlegend=False), row=r, col=cpos)
            fig.update_yaxes(title_text="count", row=r, col=cpos)
            continue

        overall = labels_df[col].dropna().to_numpy(dtype=float)
        if overall.size == 0:
            continue
        n_all = overall.size
        edges = _hist_bins(overall)
        centers = (edges[:-1] + edges[1:]) / 2
        width = edges[1] - edges[0]

        h_all, _ = np.histogram(overall, bins=edges)
        fig.add_trace(go.Bar(x=centers, y=h_all / n_all, width=width, marker_color=_GRAY,
                             opacity=0.65, showlegend=False), row=r, col=cpos)

        if focus != "All":
            cvals = cluster_rows[col].dropna().to_numpy(dtype=float)
            if cvals.size:
                h_c, _ = np.histogram(cvals, bins=edges)
                fig.add_trace(go.Bar(x=centers, y=h_c / n_all, width=width, marker_color=color,
                                     opacity=0.85, showlegend=False), row=r, col=cpos)
        fig.update_yaxes(title_text="fraction", row=r, col=cpos)

    # Legend is rendered as HTML swatches above the chart (see render()) to avoid
    # colliding with the first subplot row's titles.
    fig.update_layout(barmode="overlay", showlegend=False, height=300 * nrows,
                      margin=dict(l=10, r=10, t=40, b=10))
    return fig


def summary_table(sub_df: pd.DataFrame) -> pd.DataFrame:
    """Per-structure summary stats (n, mean, median, Q1, Q3, min, max) per metric."""
    rows = []
    for col, label in NUMERIC_METRICS:
        if col not in sub_df.columns:
            continue
        s = sub_df[col].dropna()
        if s.empty:
            continue
        rows.append({
            "Metric": label, "n": int(s.size),
            "mean": s.mean(), "median": s.median(),
            "Q1": s.quantile(0.25), "Q3": s.quantile(0.75),
            "min": s.min(), "max": s.max(),
        })
    return pd.DataFrame(rows)


def overlay_figure(labels_df: pd.DataFrame, col: str, color_map: dict[int, str]) -> go.Figure:
    """One box per cluster for a single metric — the cross-cluster comparison view."""
    fig = go.Figure()
    for c in sorted(color_map):
        vals = labels_df.loc[labels_df["cluster"] == c, col].dropna()
        if len(vals):
            fig.add_trace(go.Box(y=vals, name=str(c), marker_color=color_map[c],
                                 boxpoints="outliers", showlegend=False))
    label = dict(NUMERIC_METRICS).get(col, col)
    fig.update_layout(height=460, margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_title="cluster", yaxis_title=label)
    return fig


# -------------------------------------------------------------------------- render
def _swatch(color: str, label: str) -> str:
    return (f"<span style='display:inline-block;width:12px;height:12px;"
            f"background:{color};border-radius:2px;vertical-align:middle;"
            f"margin:0 4px 2px 0;'></span>{label}")


def _swatch_legend(focus, focus_color: str) -> str:
    """HTML legend for the histogram overlay (gray = all, color = focused cluster)."""
    parts = [_swatch(_GRAY, "all structures")]
    if focus != "All":
        parts.append(_swatch(focus_color, f"cluster {focus}"))
    return "&nbsp;&nbsp;&nbsp;".join(parts)


def _selected_cluster(event):
    """Map a plotly selection event to (focus cluster, selection signature).

    Returns None when the event carries no usable point selection. The focus is
    the most-common cluster among selected points; the signature identifies this
    exact selection so it is acted on only once.
    """
    try:
        pts = event["selection"]["points"]
    except (KeyError, TypeError):
        return None
    clusters = [p["customdata"][0] for p in pts if p.get("customdata")]
    if not clusters:
        return None
    sig = tuple(sorted((p.get("curve_number"), p.get("point_index")) for p in pts))
    return Counter(clusters).most_common(1)[0][0], sig


def render() -> None:
    st.subheader("Cluster validation")
    datasets = available_datasets()
    if not datasets:
        st.info("No validation data found. Run `python build_validation_data.py` "
                "to populate `validation_data/` from cluster-output/.")
        return

    c1, c2, _ = st.columns([1, 1, 2])
    dataset = c1.selectbox("Dataset", datasets, key="val_dataset")
    approaches = approaches_for(dataset)
    approach = c2.selectbox("Approach", approaches, key=f"val_approach_{dataset}")

    labels_df = load_labels(dataset, approach)
    tsne_df = load_tsne(dataset, approach)
    color_map = color_map_for(labels_df)
    clusters = sorted(color_map)

    focus_key = f"val_focus_{dataset}_{approach}"
    sig_key = f"val_sig_{dataset}_{approach}"
    if focus_key not in st.session_state:
        st.session_state[focus_key] = "All"

    options = ["All"] + clusters

    def _on_focus_change() -> None:
        st.session_state[focus_key] = st.session_state[f"sb_{focus_key}"]

    st.caption(f"**{len(labels_df):,}** structures · **{len(clusters)}** clusters "
               f"(k={len(clusters)}). Pick a cluster below or box/lasso-select points "
               "on the t-SNE to focus it; every panel updates.")

    with st.expander("📖 What do these metrics mean?"):
        for label, desc in METRIC_DOCS:
            st.markdown(f"**{label}** — {desc}")

    idx = options.index(st.session_state[focus_key]) if st.session_state[focus_key] in options else 0
    st.selectbox("Focus cluster", options, index=idx,
                 key=f"sb_{focus_key}", on_change=_on_focus_change)

    # "Color by ligand" is offered when the dataset carries a family label
    # (e.g. heme axial/distal ligand). It shows whether the geometry-driven
    # clusters coincide with chemical ligand identity.
    has_family_col = ("family" in tsne_df.columns
                      and tsne_df["family"].astype(str).str.strip().any())
    color_by = "Cluster"
    if has_family_col:
        color_by = st.radio("Color t-SNE by", ["Cluster", "Ligand"], horizontal=True,
                            key=f"val_colorby_{dataset}_{approach}")

    left, right = st.columns([3, 2])
    with left:
        if color_by == "Ligand":
            # Ligand view is a read-only overlay; cluster focus still drives the panels.
            st.plotly_chart(tsne_family_figure(tsne_df, family_color_map(tsne_df)),
                            key=f"tsnefam_{dataset}_{approach}", use_container_width=True)
        else:
            event = st.plotly_chart(
                tsne_figure(tsne_df, color_map, st.session_state[focus_key]),
                key=f"tsne_{dataset}_{approach}", on_select="rerun",
                selection_mode=("points", "box", "lasso"), use_container_width=True,
            )
            # Reconcile a NEW box/lasso selection into focus (guarded by signature so a
            # stale selection doesn't fight the selectbox on unrelated reruns).
            picked = _selected_cluster(event)
            if picked is not None:
                new_focus, sig = picked
                if sig != st.session_state.get(sig_key) and new_focus != st.session_state[focus_key]:
                    st.session_state[sig_key] = sig
                    st.session_state[focus_key] = new_focus
                    st.rerun()

    focus = st.session_state[focus_key]
    sub = labels_df if focus == "All" else labels_df[labels_df["cluster"] == focus]
    focus_color = color_map.get(focus, _GRAY)
    with right:
        st.markdown(f"**{'All clusters' if focus == 'All' else f'Cluster {focus}'}**")
        cnt_cols = st.columns(2)
        cnt_cols[0].metric("Structures", f"{len(sub):,}")
        if focus != "All":
            cnt_cols[1].metric("Share of dataset", f"{len(sub) / len(labels_df):.0%}")
        st.caption("Per-structure summary statistics below.")

    # Per-structure summary stats for the current selection — explicitly a
    # distribution over structures (mean/median/quartiles/range), not single values.
    stats = summary_table(sub)
    if not stats.empty:
        who = "all structures" if focus == "All" else f"cluster {focus}"
        st.markdown(f"**Summary statistics — {who}** (per structure)")
        num_cols = ["mean", "median", "Q1", "Q3", "min", "max"]
        st.dataframe(stats.style.format({c: "{:.3g}" for c in num_cols}),
                     hide_index=True, use_container_width=True)

    # Numeric metrics present in this dataset (empty for metric-free systems, e.g. heme).
    metric_opts = [(c, l) for c, l in NUMERIC_METRICS if c in labels_df.columns]
    has_family = ("family" in labels_df.columns
                  and labels_df["family"].astype(str).str.strip().any())

    dist_tab, overlay_tab = st.tabs(["Per-cluster distributions", "Metric overlays"])
    with dist_tab:
        if not metric_opts and not has_family:
            st.info("This dataset has no per-structure metrics — the t-SNE embedding above "
                    "is the cluster view. (Metric distributions apply to the Zn(Cys/His) "
                    "systems, which ship geometry/quality stats.)")
        else:
            st.markdown(_swatch_legend(focus, focus_color), unsafe_allow_html=True)
            if focus == "All":
                st.caption("Whole-dataset distributions. Focus a cluster to overlay its share.")
            st.plotly_chart(distribution_figure(labels_df, focus, focus_color),
                            use_container_width=True)
    with overlay_tab:
        if not metric_opts:
            st.info("No numeric metrics to compare for this dataset.")
        else:
            sel = st.selectbox("Metric", metric_opts, format_func=lambda t: t[1],
                               key=f"val_overlay_{dataset}_{approach}")
            st.plotly_chart(overlay_figure(labels_df, sel[0], color_map),
                            use_container_width=True)
