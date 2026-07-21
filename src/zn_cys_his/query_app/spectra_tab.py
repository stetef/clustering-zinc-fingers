#!/usr/bin/env python3
"""Native Streamlit rendering of the FEFF-calculated EXAFS spectra.

For a representative subset of structures (~4 per cluster) the pipeline computes
theoretical Zn K-edge spectra with FEFF and writes them under
``data/<ds>/calculated-spectra/<id>/`` as ``.dat`` files. Those files are tracked
in git (unlike ``cluster-output/``), so they ship with a Streamlit Cloud deploy
and this tab reads them directly; only a tiny join table (id -> cluster + color)
is precomputed by ``build_spectra_data.py`` into ``spectra_data/``.

Three panels, one per space:
  μ(E)        photon energy (xas col 1) vs absorption μ (col 4) — the raw XANES/EXAFS
  kᵂ·χ(k)     k (col 3) vs χ (col 6), k-weighted by a 0–3 slider; k ≥ 0 only
  χ(R)        r vs |χ(R)| — first two columns of ``chi-R-<id>.dat`` (fixed FT)

Interaction mirrors the Validation tab: a "focus cluster" drives the color.
  • Focus a cluster → every spectrum drawn thin gray, that cluster's on top in color.
  • "All" → clusters stacked into a waterfall, each offset by 5% of the panel's
    range and colored by its cluster color.
The k-weight slider only affects the kᵂ·χ(k) panel; χ(R) is a precomputed transform.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

HERE = Path(__file__).resolve().parent
INDEX_CSV = HERE / "spectra_data" / "spectra_index.csv"


def _repo_root() -> Path:
    """Nearest ancestor of this file that holds the tracked ``data/`` tree.

    The ``.dat`` paths in the index are repo-relative (``data/...``); resolve them
    against whichever ancestor actually contains ``data/`` (falls back to the .git
    root, then a fixed level) so the tab works both in-repo and on a Cloud deploy.
    """
    for anc in HERE.parents:
        if (anc / "data").is_dir() or (anc / ".git").exists():
            return anc
    return HERE.parents[2]


REPO_ROOT = _repo_root()

_GRAY = "#cccccc"
_FALLBACK_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
]

# Fraction of a panel's full data range each cluster is offset by in the waterfall.
_OFFSET_FRAC = 0.05


# --------------------------------------------------------------------------- data
@st.cache_data
def load_index() -> pd.DataFrame:
    if not INDEX_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(INDEX_CSV)
    df["cluster"] = df["cluster"].astype(int)
    return df


@st.cache_data
def load_records(dataset: str) -> list[dict]:
    """Parse every indexed spectrum for a dataset into raw per-space arrays.

    Cached, so the ~100 ``np.loadtxt`` reads happen once per session/dataset.
    """
    idx = load_index()
    idx = idx[idx["dataset"] == dataset]
    records: list[dict] = []
    for row in idx.itertuples(index=False):
        try:
            xas = np.loadtxt(REPO_ROOT / row.xas_rel)
        except (OSError, ValueError):
            continue
        rec = {
            "id": row.id, "pdb_id": row.pdb_id, "cluster": int(row.cluster),
            "photon_E": xas[:, 0], "k": xas[:, 2], "mu": xas[:, 3], "chi": xas[:, 5],
            "r": np.array([]), "chir_mag": np.array([]),
        }
        chir_rel = getattr(row, "chir_rel", "")
        if isinstance(chir_rel, str) and chir_rel:
            try:
                chir = np.loadtxt(REPO_ROOT / chir_rel)
                rec["r"], rec["chir_mag"] = chir[:, 0], chir[:, 1]
            except (OSError, ValueError):
                pass
        records.append(rec)
    return records


def color_map_for(records: list[dict], index: pd.DataFrame) -> dict[int, str]:
    """cluster id -> hex color, taken from the index (fallback palette otherwise)."""
    clusters = sorted({r["cluster"] for r in records})
    cmap: dict[int, str] = {}
    for c in clusters:
        vals = index.loc[index["cluster"] == c, "cluster_color"].dropna()
        v = vals.iloc[0] if len(vals) else None
        cmap[c] = v if isinstance(v, str) and v.startswith("#") else None
    for i, c in enumerate(clusters):
        if not cmap.get(c):
            cmap[c] = _FALLBACK_PALETTE[i % len(_FALLBACK_PALETTE)]
    return cmap


# ------------------------------------------------------------------------- figures
def _spectra_figure(records, xy, focus, color_map, x_title, y_title, height=460,
                    xmax=None):
    """Line overlay of one (x, y) pair per spectrum.

    ``xy(rec) -> (x, y)`` extracts the panel's coordinates from a record. In the
    "All" (waterfall) view each cluster is shifted up by ``rank * 5%`` of the full
    y-range; in a focused view everything is gray except the focused cluster, with
    no offset so the spectra overlay for direct comparison.
    """
    clusters = sorted(color_map)
    rank = {c: i for i, c in enumerate(clusters)}

    # Build (x, y) once so the offset step and traces share the same values.
    built = []
    all_y = []
    for rec in records:
        x, y = xy(rec)
        if x is None or len(x) == 0:
            continue
        built.append((rec, np.asarray(x, float), np.asarray(y, float)))
        all_y.append(y)
    fig = go.Figure()
    if not built:
        fig.update_layout(height=height, xaxis_title=x_title, yaxis_title=y_title)
        return fig

    stacked = np.concatenate(all_y)
    span = float(np.nanmax(stacked) - np.nanmin(stacked)) or 1.0
    step = _OFFSET_FRAC * span

    if focus == "All":
        order = sorted(built, key=lambda b: rank[b[0]["cluster"]])
    else:  # non-focused first, focused last so the color sits on top
        order = sorted(built, key=lambda b: b[0]["cluster"] == focus)

    seen_legend: set[int] = set()
    for rec, x, y in order:
        c = rec["cluster"]
        if focus == "All":
            offset = rank[c] * step
            color, width, opacity = color_map[c], 1.0, 0.9
            show = c not in seen_legend
            seen_legend.add(c)
            name, group = f"cluster {c}", f"c{c}"
        else:
            offset = 0.0
            if c == focus:
                color, width, opacity, group = color_map[c], 1.5, 0.95, "focus"
            else:
                color, width, opacity, group = _GRAY, 0.7, 0.45, "bg"
            show, name = False, rec["id"]
        fig.add_trace(go.Scattergl(
            x=x, y=y + offset, mode="lines",
            line=dict(color=color, width=width), opacity=opacity,
            name=name, legendgroup=group, showlegend=show,
            hovertemplate=f"{rec['id']}<br>cluster {c}<extra></extra>",
        ))

    waterfall = focus == "All"
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title=x_title,
        yaxis_title=(y_title + " (offset by cluster)") if waterfall else y_title,
        showlegend=waterfall,
        legend=dict(title="cluster", itemsizing="constant", font=dict(size=10)),
    )
    if waterfall:  # offsets make the absolute scale arbitrary
        fig.update_yaxes(showticklabels=False)
    if xmax is not None:
        xlo = float(min(np.nanmin(x) for _, x, _ in built))
        fig.update_xaxes(range=[xlo, xmax])
    return fig


def mu_xy(rec):
    return rec["photon_E"], rec["mu"]


def chir_xy(rec):
    return rec["r"], rec["chir_mag"]


def make_kchi_xy(weight: float):
    def xy(rec):
        k, chi = rec["k"], rec["chi"]
        mask = k >= 0  # standard EXAFS: drop the pre-edge negative-k rows
        return k[mask], (k[mask] ** weight) * chi[mask]
    return xy


# -------------------------------------------------------------------------- render
def _swatch(color: str, label: str) -> str:
    return (f"<span style='display:inline-block;width:12px;height:12px;"
            f"background:{color};border-radius:2px;vertical-align:middle;"
            f"margin:0 4px 2px 0;'></span>{label}")


def _swatch_legend(focus, focus_color: str) -> str:
    parts = [_swatch(_GRAY, "all spectra")]
    if focus != "All":
        parts.append(_swatch(focus_color, f"cluster {focus}"))
    return "&nbsp;&nbsp;&nbsp;".join(parts)


def render() -> None:
    st.subheader("Calculated spectra")
    index = load_index()
    if index.empty:
        st.info("No spectra index found. Run `python build_spectra_data.py` to "
                "populate `spectra_data/` from data/*/calculated-spectra/.")
        return

    datasets = sorted(index["dataset"].unique())
    c1, _ = st.columns([1, 3])
    dataset = c1.selectbox("Dataset", datasets, key="spec_dataset")

    records = load_records(dataset)
    if not records:
        st.warning(f"No spectra files could be read for {dataset}.")
        return
    color_map = color_map_for(records, index[index["dataset"] == dataset])
    clusters = sorted(color_map)

    focus_key = f"spec_focus_{dataset}"
    if focus_key not in st.session_state:
        st.session_state[focus_key] = "All"
    options = ["All"] + clusters
    idx = options.index(st.session_state[focus_key]) if st.session_state[focus_key] in options else 0

    st.caption(f"**{len(records)}** FEFF-calculated spectra across **{len(clusters)}** "
               "clusters (~4 representative structures each). Focus a cluster to draw "
               "it in color over every spectrum in gray; **All** stacks the clusters "
               "into a 5%-offset waterfall.")

    focus = st.selectbox("Focus cluster", options, index=idx, key=f"sb_{focus_key}")
    st.session_state[focus_key] = focus
    focus_color = color_map.get(focus, _GRAY)

    if focus != "All":
        st.markdown(_swatch_legend(focus, focus_color), unsafe_allow_html=True)

    # μ(E) — wide, its own row (photon-energy axis spans the XANES + near EXAFS).
    st.markdown("**μ(E) — absorption**")
    st.plotly_chart(
        _spectra_figure(records, mu_xy, focus, color_map,
                        x_title="photon energy (eV)", y_title="μ", height=420,
                        xmax=10000),
        use_container_width=True, key=f"spec_mu_{dataset}",
    )

    # kᵂ·χ(k) and χ(R) side by side — the EXAFS pair.
    left, right = st.columns(2)
    with left:
        weight = st.radio("k-weight (kᵂ)", options=[0, 1, 2, 3], index=3,
                          horizontal=True, key=f"spec_kw_{dataset}",
                          help="Weights χ(k) by kᵂ to emphasise the high-k EXAFS.")
        wlabel = "χ(k)" if weight == 0 else ("k·χ(k)" if weight == 1 else f"k^{weight}·χ(k)")
        st.markdown(f"**{wlabel}**")
        st.plotly_chart(
            _spectra_figure(records, make_kchi_xy(weight), focus, color_map,
                            x_title="k (Å⁻¹)", y_title=wlabel, height=420, xmax=12),
            use_container_width=True, key=f"spec_k_{dataset}",
        )
    with right:
        st.markdown("**|χ(R)| — magnitude**")
        st.caption("Precomputed Fourier transform; not affected by the k-weight slider.")
        st.plotly_chart(
            _spectra_figure(records, chir_xy, focus, color_map,
                            x_title="R (Å)", y_title="|χ(R)|", height=420),
            use_container_width=True, key=f"spec_r_{dataset}",
        )
