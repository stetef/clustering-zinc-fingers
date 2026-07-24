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
from sklearn.metrics import (
    adjusted_mutual_info_score,
    homogeneity_completeness_v_measure,
)

import motif_data  # sibling module (streamlit adds the app dir to sys.path)

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
    ("Fe B-factor",
     "Crystallographic B-factor (atomic displacement parameter) of the heme **Fe** ion — higher "
     "means more positional uncertainty / thermal motion."),
    ("Avg B-factor (non-Fe)",
     "Mean B-factor over **every heavy atom in the extracted cluster except the Fe** — the "
     "porphyrin, the axial/distal ligands, and any pocket residues. A coarse measure of local "
     "disorder / resolution around the site."),
    # NOTE: the "Family" explanation is profile-specific and comes from the
    # structure profile (see family_doc_for), not this static list.
]

# Fallback family description when a dataset has no profile marker / the profile
# defines none.
_FALLBACK_FAMILY_DOC = ("The categorical grouping label for each structure "
                        "(e.g. its ligand/motif composition).")

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


def profile_name_for(dataset: str, approach: str) -> str:
    """Profile that produced this dataset (from the profile.txt marker).

    Defaults to 'zn_cys_his' when absent, matching the original datasets that
    predate the marker.
    """
    marker = VALIDATION_DIR / dataset / approach / "profile.txt"
    if marker.is_file():
        name = marker.read_text().strip()
        if name:
            return name
    return "zn_cys_his"


def family_doc_for(dataset: str, approach: str) -> str:
    """Markdown explaining the `family` label for this dataset's profile."""
    try:
        from zn_cys_his.clustering.profiles import get_profile
        doc = get_profile(profile_name_for(dataset, approach)).family_doc
        if doc:
            return doc
    except Exception:
        pass
    return _FALLBACK_FAMILY_DOC


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
    """Stable color per family, most-common first (blanks -> 'none')."""
    fams = (tsne_df["family"].fillna("").astype(str).str.strip()
            .replace("", "none"))
    order = [f for f, _ in Counter(fams).most_common()]
    return {f: _FALLBACK_PALETTE[i % len(_FALLBACK_PALETTE)] for i, f in enumerate(order)}


def tsne_family_figure(tsne_df: pd.DataFrame, fam_colors: dict[str, str]) -> go.Figure:
    """Scatter of frozen t-SNE coords, one legend entry per family.

    This is the 'color by family' view: it shows whether the geometry-driven
    clusters line up with chemical family identity (they largely do not for heme —
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
        legend=dict(title="family", itemsizing="constant", font=dict(size=10)),
        xaxis_title="t-SNE 1", yaxis_title="t-SNE 2",
    )
    return fig


# --------------------------------------------------------------------------- motifs
def motif_tsne(tsne_df: pd.DataFrame, motifs_df: pd.DataFrame) -> pd.DataFrame:
    """t-SNE points joined to their PDB's PROSITE motif cells (motif1..motif3).

    The structure id's leading token is the 4-char pdb code (e.g. ``3M5L_1`` ->
    ``3m5l``); motifs are keyed by that lowercase code.
    """
    df = tsne_df.copy()
    df["pdb_id"] = df["id"].astype(str).str.split("_").str[0].str.lower()
    df = df.merge(motifs_df[["pdb_id", "motif1", "motif2", "motif3"]],
                  on="pdb_id", how="left")
    return df


def motif_option_list(motif_df: pd.DataFrame) -> list[str]:
    """All distinct motifs present, ordered by how many points carry them."""
    counts: Counter = Counter()
    for _, row in motif_df.iterrows():
        for m in motif_data.all_motifs(row):
            counts[m] += 1
    return [m for m, _ in counts.most_common()]


def tsne_motif_figure(motif_df: pd.DataFrame, selected: str | None) -> go.Figure:
    """Color the t-SNE by PROSITE motif.

    ``selected is None`` -> color every point by its **first** (most-prevalent)
    motif, one legend entry per first-motif. Otherwise highlight only the points
    whose motif set *contains* ``selected`` (even when it is not their first),
    dimming the rest to gray context.
    """
    fig = go.Figure()
    if selected is None:
        firsts = motif_df.apply(motif_data.first_motif, axis=1).replace("", "none")
        order = [m for m, _ in Counter(firsts).most_common()]
        colors = {m: _FALLBACK_PALETTE[i % len(_FALLBACK_PALETTE)]
                  for i, m in enumerate(order)}
        if "none" in colors:
            colors["none"] = _GRAY
        for m in order:
            d = motif_df[firsts == m]
            if d.empty:
                continue
            fig.add_trace(go.Scattergl(
                x=d["tsne1"], y=d["tsne2"], mode="markers", name=m,
                marker=dict(color=colors[m], size=7, opacity=0.85, line=dict(width=0)),
                hovertemplate=f"{m}<extra></extra>",
            ))
    else:
        has = motif_df.apply(lambda r: selected in motif_data.all_motifs(r), axis=1)
        other = motif_df[~has]
        if not other.empty:
            fig.add_trace(go.Scattergl(
                x=other["tsne1"], y=other["tsne2"], mode="markers", name="other",
                marker=dict(color=_GRAY, size=5, opacity=0.15, line=dict(width=0)),
                hovertemplate="other<extra></extra>",
            ))
        hit = motif_df[has]
        if not hit.empty:
            fig.add_trace(go.Scattergl(
                x=hit["tsne1"], y=hit["tsne2"], mode="markers", name=selected,
                marker=dict(color=_FALLBACK_PALETTE[0], size=8, opacity=0.9,
                            line=dict(width=0)),
                hovertemplate=f"{selected}<extra></extra>",
            ))
    fig.update_layout(
        height=560, margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(title="motif", itemsizing="constant", font=dict(size=10)),
        xaxis_title="t-SNE 1", yaxis_title="t-SNE 2",
    )
    return fig


# -------------------------------------------------------------------------- enzyme
def enzyme_tsne(tsne_df: pd.DataFrame, labels: dict[str, str]) -> pd.DataFrame:
    """t-SNE points labelled by enzyme (consensus name > paper title > 'none')."""
    df = tsne_df.copy()
    df["pdb_id"] = df["id"].astype(str).str.split("_").str[0].str.lower()
    df["enzyme"] = df["pdb_id"].map(labels).fillna("").replace("", "none")
    return df


def _short(label: str, width: int = 40) -> str:
    """Truncate a long label (paper titles) for the legend/hover."""
    return label if len(label) <= width else label[: width - 1] + "…"


def tsne_enzyme_figure(enz_df: pd.DataFrame) -> go.Figure:
    """Color the t-SNE by enzyme identity, one legend entry per distinct label."""
    cats = enz_df["enzyme"]
    order = [c for c, _ in Counter(cats).most_common()]
    colors = {c: _FALLBACK_PALETTE[i % len(_FALLBACK_PALETTE)]
              for i, c in enumerate(order)}
    if "none" in colors:
        colors["none"] = _GRAY
    fig = go.Figure()
    for c in order:
        d = enz_df[cats == c]
        if d.empty:
            continue
        name = _short(c)
        fig.add_trace(go.Scattergl(
            x=d["tsne1"], y=d["tsne2"], mode="markers", name=name,
            marker=dict(color=colors[c], size=7, opacity=0.85, line=dict(width=0)),
            hovertemplate=f"{name}<extra></extra>",
        ))
    fig.update_layout(
        height=560, margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(title="enzyme", itemsizing="constant", font=dict(size=10)),
        xaxis_title="t-SNE 1", yaxis_title="t-SNE 2",
    )
    return fig


# -------------------------------------------------------- cluster ↔ label agreement
def label_frame(labels_df: pd.DataFrame, motifs_df: pd.DataFrame,
                enzyme_labels: dict[str, str], *,
                use_family: bool, use_motif: bool, use_enzyme: bool) -> pd.DataFrame:
    """One row per structure with its categorical labels for the agreement metrics.

    Columns: ``cluster_id`` (int, for grouping), ``Cluster`` (str), and whichever
    of ``Family`` / ``Motif`` / ``Enzyme`` the dataset carries. A missing label
    becomes ``"none"`` (its own category) so every structure is scored.
    """
    df = pd.DataFrame({"cluster_id": labels_df["cluster"].astype(int).to_numpy()})
    df["Cluster"] = df["cluster_id"].astype(str)
    pdb = labels_df["id"].astype(str).str.split("_").str[0].str.lower()
    df["pdb_id"] = pdb.to_numpy()

    if use_family and "family" in labels_df.columns:
        df["Family"] = (labels_df["family"].fillna("").astype(str).str.strip()
                        .replace("", "none").to_numpy())
    if use_motif:
        first = {r["pdb_id"]: (motif_data.first_motif(r) or "none")
                 for _, r in motifs_df.iterrows()}
        df["Motif"] = df["pdb_id"].map(first).fillna("none")
    if use_enzyme:
        df["Enzyme"] = df["pdb_id"].map(enzyme_labels).fillna("").replace("", "none")
    return df


def _label_cols(lf: pd.DataFrame) -> list[str]:
    return [c for c in ("Family", "Motif", "Enzyme") if c in lf.columns]


def ami_matrix(lf: pd.DataFrame) -> pd.DataFrame:
    """Pairwise adjusted mutual information among Cluster and every label type."""
    names = ["Cluster"] + _label_cols(lf)
    mat = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            mat.loc[a, b] = (1.0 if a == b
                             else adjusted_mutual_info_score(lf[a], lf[b]))
    return mat


def vmeasure_table(lf: pd.DataFrame) -> pd.DataFrame:
    """Homogeneity / completeness / V-measure of each label vs the clustering.

    Each label type is the ground truth (``labels_true``) and the cluster id is
    the prediction (``labels_pred``), so homogeneity asks "is each cluster pure in
    this label?" and completeness "is each label kept in one cluster?".
    """
    rows = []
    for name in _label_cols(lf):
        h, c, v = homogeneity_completeness_v_measure(lf[name], lf["Cluster"])
        rows.append({"Label": name, "Homogeneity": h, "Completeness": c,
                     "V-measure": v, "AMI vs cluster": mat_ami(lf, name)})
    return pd.DataFrame(rows)


def mat_ami(lf: pd.DataFrame, name: str) -> float:
    return adjusted_mutual_info_score(lf[name], lf["Cluster"])


def ami_heatmap(mat: pd.DataFrame) -> go.Figure:
    names = list(mat.index)
    z = mat.to_numpy(dtype=float)
    fig = go.Figure(go.Heatmap(
        z=z, x=names, y=names, zmin=0, zmax=1, colorscale="Viridis",
        colorbar=dict(title="AMI"),
        text=[[f"{v:.2f}" for v in row] for row in z],
        texttemplate="%{text}", hovertemplate="%{y} vs %{x}: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10),
                      yaxis=dict(autorange="reversed"))
    return fig


def cluster_label_breakdown(lf: pd.DataFrame, cluster_id: int, label_col: str) -> pd.DataFrame:
    """Percentage each label makes up of one cluster, largest first."""
    sub = lf[lf["cluster_id"] == cluster_id]
    n = len(sub)
    vc = sub[label_col].value_counts()
    return pd.DataFrame({
        label_col: vc.index,
        "count": vc.to_numpy(),
        "percent": (100 * vc / n).to_numpy() if n else vc.to_numpy(),
    })


def cluster_top_labels(lf: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """One row per cluster: its dominant label + that label's share and diversity."""
    rows = []
    for cid, sub in lf.groupby("cluster_id"):
        n = len(sub)
        vc = sub[label_col].value_counts()
        rows.append({"cluster": int(cid), "n": n,
                     f"top {label_col.lower()}": vc.index[0],
                     "top %": 100 * vc.iloc[0] / n,
                     "# distinct": int(sub[label_col].nunique())})
    return pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)


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

    # Every family in the dataset, in a fixed order (by overall count desc) so the
    # x-axis is identical across clusters and *all* labels appear — no top-N cap.
    all_family_counts = Counter(f for f in labels_df["family"].astype(str).str.strip() if f)
    all_fams = [f for f, _ in all_family_counts.most_common()]

    for i, (col, _lbl) in enumerate(panels):
        r, cpos = i // ncols + 1, i % ncols + 1
        if col == "family":
            counts = Counter(f for f in cluster_rows["family"].astype(str).str.strip() if f)
            fig.add_trace(go.Bar(x=all_fams, y=[counts.get(f, 0) for f in all_fams],
                                 marker_color=(color if focus != "All" else _GRAY),
                                 showlegend=False), row=r, col=cpos)
            fig.update_yaxes(title_text="count", row=r, col=cpos)
            # Force all category labels to show, angled + small for many families.
            fig.update_xaxes(categoryorder="array", categoryarray=all_fams,
                             tickangle=-60, tickfont=dict(size=8), row=r, col=cpos)
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

    # Only document the metrics this dataset actually has — so heme doesn't show
    # Zn geometry docs and vice versa.  The Family entry is profile-specific and
    # comes from the structure profile.
    present_labels = {lbl for col, lbl in NUMERIC_METRICS if col in labels_df.columns}
    relevant_docs = [(lbl, desc) for lbl, desc in METRIC_DOCS if lbl in present_labels]
    if "family" in labels_df.columns and labels_df["family"].astype(str).str.strip().any():
        relevant_docs.append(("Family", family_doc_for(dataset, approach)))
    if relevant_docs:
        with st.expander("📖 What do these metrics mean?"):
            for label, desc in relevant_docs:
                st.markdown(f"**{label}** — {desc}")

    idx = options.index(st.session_state[focus_key]) if st.session_state[focus_key] in options else 0
    st.selectbox("Focus cluster", options, index=idx,
                 key=f"sb_{focus_key}", on_change=_on_focus_change)

    # "Color by family" is offered when the dataset carries a family label
    # (e.g. heme axial/distal ligand). It shows whether the geometry-driven
    # clusters coincide with chemical family identity. "Color by motif" is offered
    # when the structures have PROSITE motif data (3Cys1His), and can highlight a
    # single motif across the map.
    has_family_col = ("family" in tsne_df.columns
                      and tsne_df["family"].astype(str).str.strip().any())
    motifs_raw = motif_data.load_motifs()
    motif_df = motif_tsne(tsne_df, motifs_raw)
    motif_names = motif_option_list(motif_df)
    has_motif = bool(motif_names)
    enzyme_labels = motif_data.enzyme_label_map()
    enz_df = enzyme_tsne(tsne_df, enzyme_labels)
    has_enzyme = bool((enz_df["enzyme"] != "none").any())

    palette_opts = ["Cluster"]
    if has_family_col:
        palette_opts.append("Family")
    if has_motif:
        palette_opts.append("Motif")
    if has_enzyme:
        palette_opts.append("Enzyme")
    color_by = "Cluster"
    if len(palette_opts) > 1:
        color_by = st.radio("Color t-SNE by", palette_opts, horizontal=True,
                            key=f"val_colorby_{dataset}_{approach}")

    selected_motif: str | None = None
    if color_by == "Motif":
        motif_choices = ["All (first motif per PDB)"] + motif_names
        pick = st.selectbox(
            "Motif", motif_choices, key=f"val_motifpick_{dataset}_{approach}",
            help="PDBs with several motifs show their first motif here; pick a "
                 "specific motif to highlight every PDB that carries it.")
        selected_motif = None if pick.startswith("All") else pick

    left, right = st.columns([3, 2])
    with left:
        if color_by == "Family":
            # Family view is a read-only overlay; cluster focus still drives the panels.
            st.plotly_chart(tsne_family_figure(tsne_df, family_color_map(tsne_df)),
                            key=f"tsnefam_{dataset}_{approach}", use_container_width=True)
        elif color_by == "Motif":
            # Motif view is a read-only overlay too; cluster focus still drives panels.
            st.plotly_chart(tsne_motif_figure(motif_df, selected_motif),
                            key=f"tsnemotif_{dataset}_{approach}", use_container_width=True)
        elif color_by == "Enzyme":
            # Enzyme view is a read-only overlay too; cluster focus still drives panels.
            st.plotly_chart(tsne_enzyme_figure(enz_df),
                            key=f"tsneenz_{dataset}_{approach}", use_container_width=True)
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

    dist_tab, overlay_tab, agree_tab = st.tabs(
        ["Per-cluster distributions", "Metric overlays", "Cluster ↔ labels"])
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

    with agree_tab:
        if not (has_family_col or has_motif or has_enzyme):
            st.info("This dataset has no family/motif/enzyme labels to compare "
                    "against the clustering.")
        else:
            lf = label_frame(labels_df, motifs_raw, enzyme_labels,
                             use_family=has_family_col, use_motif=has_motif,
                             use_enzyme=has_enzyme)

            st.markdown("**Do the geometry-driven clusters agree with the "
                        "chemical labels?**")
            st.caption("Adjusted mutual information (AMI) is 0 for a chance "
                       "labelling and 1 for identical partitions; it is symmetric "
                       "and corrects for the number of groups.")
            m1, m2 = st.columns([1, 1])
            with m1:
                st.plotly_chart(ami_heatmap(ami_matrix(lf)),
                                key=f"ami_{dataset}_{approach}",
                                use_container_width=True)
            with m2:
                st.markdown("**Homogeneity / completeness / V-measure**")
                st.caption("Each label is the ground truth and the cluster id is "
                           "the prediction. **Homogeneity** = each cluster is pure "
                           "in that label; **completeness** = each label stays in "
                           "one cluster; **V-measure** is their harmonic mean.")
                vt_tbl = vmeasure_table(lf)
                st.dataframe(
                    vt_tbl.style.format(
                        {c: "{:.3f}" for c in
                         ("Homogeneity", "Completeness", "V-measure", "AMI vs cluster")}),
                    hide_index=True, use_container_width=True)

            st.markdown("---")
            label_choices = _label_cols(lf)
            default_idx = label_choices.index("Enzyme") if "Enzyme" in label_choices else 0
            bl = st.selectbox("Break each cluster down by", label_choices,
                              index=default_idx, key=f"val_break_{dataset}_{approach}")

            if focus == "All":
                st.caption("Dominant label per cluster (focus a cluster above for "
                           "its full percentage breakdown).")
                top = cluster_top_labels(lf, bl)
                st.dataframe(top.style.format({"top %": "{:.1f}"}),
                             hide_index=True, use_container_width=True)
            else:
                brk = cluster_label_breakdown(lf, int(focus), bl)
                st.markdown(f"**Cluster {focus} — {bl} composition** "
                            f"({len(brk)} distinct, {int(brk['count'].sum())} structures)")
                st.caption("Sorted by share of the cluster, largest first — the "
                           "tail shows the outlier labels in this cluster.")
                st.dataframe(
                    brk.style.format({"percent": "{:.1f}%"}),
                    hide_index=True, use_container_width=True,
                    column_config={
                        "percent": st.column_config.ProgressColumn(
                            "percent", format="%.1f%%", min_value=0.0,
                            max_value=float(brk["percent"].max()) if len(brk) else 100.0),
                    })
                st.download_button(
                    f"Download cluster {focus} {bl} breakdown CSV",
                    brk.to_csv(index=False).encode(),
                    file_name=f"cluster{focus}_{bl.lower()}_breakdown.csv",
                    mime="text/csv", key=f"dl_break_{dataset}_{approach}")
