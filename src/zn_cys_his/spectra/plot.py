#!/usr/bin/env python3
"""Interactive offline report: cluster stat-distributions + sampled spectra.

This is a close replica of ``report_cluster_distribution_offline.html`` (produced by
``-m zn_cys_his.clustering.step06_validate_clusters``) with EXAFS/XANES
spectra folded into the same layout.  Approach 1 (``kmeans_labels_with_stats.csv`` +
``medoids.csv``) is the reference clustering.

  HERO (top)      Selected point: its id, cluster, whether it is a medoid / has spectra,
                  and the structure's own stats as metric tiles.  A compact XANES / χ(k) /
                  |χ(R)| strip sits below the tiles (secondary, tile-scale).  Updated by
                  clicking any t-SNE point.

  LEFT  (t-SNE)   Interactive 2-D t-SNE of the FULL dataset, colored by k-means cluster.
                  Sampled structures are outlined; each cluster's medoid is a star.  A
                  toggle flips between "All points" and "Sampled only" without rescaling.

  RIGHT (tabs)    Three tabs, matching the reference report's look:
                    • Per Cluster Rows   — one row per cluster: stat histograms (gray =
                      full dataset, cluster color on top) PLUS three spectra subplots
                      (XANES / χ(k) / |χ(R)|; gray = all sampled, cluster color on top,
                      medoid emphasized).  Every cluster gets a row.
                    • Per Metric Overlays — the report's pre-rendered histogram overlays,
                      reused verbatim (one file per metric).
                    • Spectra            — combined overlay + stacked-by-cluster, all on
                      one page.

Plot styling matches the reference report (default matplotlib, fraction-of-total
histograms); the t-SNE is interactive Plotly, inlined so the page is offline.

Run with the project environment:

    uv run zch-plot-spectra          # == python -m zn_cys_his.spectra.plot
"""
from __future__ import annotations

import argparse
import base64
import csv
import html as html_lib
import io
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------------------
# Defaults (relative to repo root); override via CLI flags.
# ----------------------------------------------------------------------------
from zn_cys_his.paths import CLUSTER_OUTPUT, DATA_DIR

# Precomputed FEFF spectra (input) live under data/; clustering output is mirrored
# to cluster-output/.  Override either with --station / --approach.
DEFAULT_STATION = DATA_DIR / "4cys-large/calculated-spectra"
DEFAULT_APPROACH = CLUSTER_OUTPUT / "test-4cys-weighted/approach1"
DEFAULT_LABELS = DEFAULT_APPROACH / "kmeans_labels_with_stats.csv"
DEFAULT_EMBED = DEFAULT_APPROACH / "embeddings.csv"
DEFAULT_MEDOIDS = DEFAULT_APPROACH / "medoids.csv"

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"

PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
    "#5254a3", "#8ca252", "#bd9e39", "#ad494a", "#a55194",
    "#6b6ecf",
]

GRAY = "#cccccc"  # background ("all") color, matching the reference histograms

# Hero stat columns: (csv column, label, decimals)
HERO_STATS = [
    ("family", "family", None),
    ("resolution_A", "resolution (Å)", 2),
    ("volume_A3", "volume (Å³)", 3),
    ("q_tetra_coord", "q tetra coord", 4),
    ("q_tetra_ca", "q tetra Cα", 4),
    ("cys_dihedral_mean_deg", "Cys dihedral mean (°)", 2),
    ("r_work", "R-work", 4),
    ("r_free", "R-free", 4),
    ("zn_bfactor", "Zn B-factor", 2),
]

# Numeric stat histograms — column + mathtext label — matching utils._NUMERIC_PLOT_METRICS.
_NUMERIC_PLOT_METRICS: list[tuple[str, str]] = [
    ("volume_A3",              r"Volume ($\AA^3$)"),
    ("q_tetra_coord",          r"$q_\mathrm{tetra}$ (coord)"),
    ("q_tetra_ca",             r"$q_\mathrm{tetra}$ ($C_\alpha$)"),
    ("r_work",                 r"$R_\mathrm{work}$"),
    ("r_free",                 r"$R_\mathrm{free}$"),
    ("zn_bfactor",             r"Zn $B$-factor"),
    ("cys_dihedral_mean_deg",  r"Dihedral mean ($^\circ$)"),
    ("all_coord_res_bfactor_avg", r"Coord-res $\bar{B}$"),
]

# Font sizes — matching utils cluster-distribution plots.
_FS_LABEL = 13
_FS_TICK = 11
_FS_TITLE = 14
_FS_SUP = 15
_FS_ANNOT = 9

SPEC_META = {
    "xanes": ("Energy (eV)", r"$\mu(E)$", "XANES"),
    "exafs": (r"k (Å$^{-1}$)", r"$\chi(k)$", r"EXAFS $\chi(k)$"),
    "chir": (r"R (Å)", r"$|\chi(R)|$", r"|$\chi$(R)|"),
}

EXAFS_KMIN = 1.0   # EXAFS panels start at k = 1 Å^-1
EXAFS_KMAX = 10.0  # EXAFS panels end at k = 10 Å^-1
EXAFS_XTICKS = [1, 2, 4, 6, 8, 10]  # force a labeled tick at k = 1
XANES_YMIN: float | None = 0.0001   # XANES μ(E) y-axis floor; a labeled tick is added here


# ----------------------------------------------------------------------------
# Matplotlib styling — plain matplotlib defaults so plots match the reference
# report's histograms (DejaVu Sans, default spines); only force a white canvas.
# ----------------------------------------------------------------------------
def setup_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "mathtext.fontset": "dejavusans",
    })


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def read_labels(path: Path) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            out[row["id"]] = row
    return out


def read_embeddings(path: Path) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                out[row["id"]] = (float(row["tsne1"]), float(row["tsne2"]))
            except (KeyError, ValueError):
                continue
    return out


def read_medoids(path: Path) -> dict[str, str]:
    """cluster (as str) -> medoid_id.  Accepts medoid_id / medoid_filename columns."""
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            cid = (row.get("cluster_id") or row.get("cluster") or "").strip()
            mid = (row.get("medoid_id") or row.get("medoid_filename") or "").strip()
            if cid and mid:
                out[cid] = mid
    return out


def load_dat(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.atleast_2d(np.loadtxt(path, comments="#"))
    return data[:, 0], data[:, 1]


def _find(dir_: Path, prefix: str, sid: str) -> Path | None:
    cand = dir_ / f"{prefix}-{sid}.dat"
    if cand.exists():
        return cand
    hits = sorted(dir_.glob(f"{prefix}-*.dat"))
    return hits[0] if hits else None


def cluster_sort_key(c) -> tuple[int, str]:
    s = str(c)
    return (int(s), "") if s.lstrip("-").isdigit() else (10**9, s)


def collect_spectra(station: Path, labels: dict, medoids: dict[str, str]) -> list[dict]:
    """One record per structure directory (those with spectra)."""
    medoid_ids = set(medoids.values())
    records: list[dict] = []
    for d in sorted(p for p in station.iterdir() if p.is_dir()):
        if d.name == "analysis-and-visualization":
            continue
        sid = d.name
        lab = labels.get(sid)
        if lab is None:
            print(f"  ! no cluster label for {sid}; skipping")
            continue
        rec: dict = {"id": sid, "cluster": str(lab.get("cluster", "")),
                     "color": lab.get("cluster_color") or None,
                     "is_medoid": sid in medoid_ids}
        for key, prefix in (("xanes", "xanes"), ("exafs", "exafs"), ("chir", "chi-R")):
            f = _find(d, prefix, sid)
            rec[key] = load_dat(f) if f else None
        records.append(rec)
    return records


def build_color_map(labels: dict, extra_clusters=()) -> dict[str, str]:
    color_by_cluster: dict[str, str] = {}
    for row in labels.values():
        c, col = str(row.get("cluster", "")), row.get("cluster_color")
        if c and col and c not in color_by_cluster:
            color_by_cluster[c] = col
    clusters = sorted({str(row.get("cluster", "")) for row in labels.values()} | set(extra_clusters),
                      key=cluster_sort_key)
    for i, c in enumerate(clusters):
        color_by_cluster.setdefault(c, PALETTE[i % len(PALETTE)])
    return color_by_cluster


# ----------------------------------------------------------------------------
# Figure -> base64 helpers
# ----------------------------------------------------------------------------
def fig_to_b64(fig, dpi: int = 130) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _safe_float(v) -> float | None:
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() == "nan":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _apply_spec_axis(ax, key):
    """Consistent spectra-axis styling: k=1 tick for EXAFS, μ(E) floor for XANES."""
    if key == "exafs":
        ax.set_xlim(EXAFS_KMIN, EXAFS_KMAX)
        ax.set_xticks(EXAFS_XTICKS)
    elif key == "xanes" and XANES_YMIN is not None:
        ax.set_ylim(bottom=XANES_YMIN)
        yt = list(ax.get_yticks())
        if XANES_YMIN not in yt:
            ax.set_yticks(sorted([XANES_YMIN] + [t for t in yt if t > XANES_YMIN]))


def spec_xy(r, key):
    """Return (x, y) for a spectrum, trimming EXAFS to EXAFS_KMIN <= k <= EXAFS_KMAX."""
    xy = r.get(key)
    if xy is None:
        return None
    x, y = xy
    if key == "exafs":
        m = (x >= EXAFS_KMIN) & (x <= EXAFS_KMAX)
        return x[m], y[m]
    return x, y


def _draw_spectra_axis(ax, key, all_records, cluster_records, color):
    """One spectra subplot: gray = all sampled, this cluster's spectra colored on top."""
    xlabel, ylabel, name = SPEC_META[key]
    for r in all_records:                       # gray background (all sampled)
        xy = spec_xy(r, key)
        if xy is not None:
            ax.plot(xy[0], xy[1], color=GRAY, lw=0.9, alpha=0.5, zorder=1)
    n = 0
    for r in cluster_records:                   # this cluster, in color
        xy = spec_xy(r, key)
        if xy is None:
            continue
        ax.plot(xy[0], xy[1], color=color, lw=1.4, alpha=0.8, zorder=3)
        n += 1
    ax.set_xlabel(xlabel, fontsize=_FS_LABEL)
    ax.set_ylabel(ylabel, fontsize=_FS_LABEL)
    ax.set_title(f"{name}  (n={n})", fontsize=_FS_TITLE)
    ax.tick_params(labelsize=_FS_TICK)
    _apply_spec_axis(ax, key)


def render_cluster_row(cluster, all_rows, all_records, color) -> str:
    """One per-cluster figure: stat histograms + 3 spectra subplots, unified grid.

    Mirrors utils._plot_per_cluster_rows (gray = full dataset, cluster color on top),
    then appends XANES / χ(k) / |χ(R)| subplots in the same style.
    """
    cluster_rows = [r for r in all_rows if str(r.get("cluster", "")).strip() == str(cluster)]
    cluster_records = [r for r in all_records if str(r["cluster"]) == str(cluster)]

    # Full-dataset values for the gray background histograms.
    all_vals: dict[str, list[float]] = {}
    for col, _ in _NUMERIC_PLOT_METRICS:
        all_vals[col] = [v for r in all_rows if (v := _safe_float(r.get(col))) is not None]
    has_family = any((r.get("family") or "").strip() for r in all_rows)

    # Panel descriptors: numeric histograms, family bar, then the 3 spectra.
    panel_defs: list[tuple] = [("hist", col, label) for col, label in _NUMERIC_PLOT_METRICS]
    if has_family:
        panel_defs.append(("family",))
    panel_defs += [("spec", k) for k in ("xanes", "exafs", "chir")]

    n = len(panel_defs)
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.5), squeeze=False)
    n_sampled = sum(1 for r in cluster_records if any(r.get(k) is not None for k in ("xanes", "exafs", "chir")))
    fig.suptitle(f"Cluster {cluster}  (n={len(cluster_rows)}, {n_sampled} sampled)",
                 fontsize=_FS_SUP, fontweight="bold")

    for idx, pd in enumerate(panel_defs):
        ax = axes[idx // ncols][idx % ncols]
        kind = pd[0]

        if kind == "hist":
            col, label = pd[1], pd[2]
            overall = all_vals.get(col, [])
            N_col = len(overall)
            if N_col > 0:
                bins = min(30, max(5, N_col // 10))
                lo, hi = float(np.min(overall)), float(np.max(overall))
                if lo == hi:
                    lo, hi = lo - 0.5, hi + 0.5
                bin_edges = np.linspace(lo, hi, bins + 1)
                ax.hist(overall, bins=bin_edges, color=GRAY, alpha=0.65,
                        weights=np.ones(N_col) / N_col, label="all")
                cvals = [v for r in cluster_rows if (v := _safe_float(r.get(col))) is not None]
                if cvals:
                    ax.hist(cvals, bins=bin_edges, color=color, alpha=0.80,
                            weights=np.ones(len(cvals)) / N_col, label=f"c{cluster}")
            ax.set_xlabel(label, fontsize=_FS_LABEL)
            ax.set_ylabel("fraction of total", fontsize=_FS_LABEL)
            ax.set_title(label, fontsize=_FS_TITLE)
            ax.tick_params(labelsize=_FS_TICK)

        elif kind == "family":
            c_fam = [r.get("family", "").strip() for r in cluster_rows
                     if r.get("family", "").strip()]
            counts = Counter(c_fam)
            if counts:
                fams = sorted(counts, key=lambda x: counts[x], reverse=True)[:20]
                xs = range(len(fams))
                ax.bar(xs, [counts[f] for f in fams], color=color, alpha=0.85)
                ax.set_xticks(list(xs))
                ax.set_xticklabels(fams, rotation=90, ha="center", fontsize=7)
                ax.text(0.98, 0.97, fams[0], transform=ax.transAxes,
                        ha="right", va="top", fontsize=_FS_ANNOT,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                  edgecolor="#cccccc", alpha=0.85))
            ax.set_ylabel("count", fontsize=_FS_LABEL)
            ax.set_title("Family", fontsize=_FS_TITLE)
            ax.tick_params(axis="y", labelsize=_FS_TICK)

        else:  # spectra
            _draw_spectra_axis(ax, pd[1], all_records, cluster_records, color)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.tight_layout()
    return fig_to_b64(fig, dpi=110)


def render_combined_panel(records, key, color_by_cluster, figsize=(12, 4.2)) -> str:
    from matplotlib.lines import Line2D

    xlabel, ylabel, name = SPEC_META[key]
    fig, ax = plt.subplots(figsize=figsize)
    used = set()
    for r in records:
        xy = spec_xy(r, key)
        if xy is None:
            continue
        ax.plot(xy[0], xy[1], color=color_by_cluster[r["cluster"]],
                lw=1.3, alpha=0.8, zorder=2)
        used.add(r["cluster"])
    ax.set_xlabel(xlabel, fontsize=_FS_LABEL)
    ax.set_ylabel(ylabel, fontsize=_FS_LABEL)
    ax.set_title(f"{name} — all sampled (colored by cluster)", fontsize=_FS_TITLE)
    ax.tick_params(labelsize=_FS_TICK)
    _apply_spec_axis(ax, key)
    handles = [Line2D([0], [0], color=color_by_cluster[c], lw=3, label=f"cluster {c}")
               for c in sorted(used, key=cluster_sort_key)]
    ax.legend(handles=handles, fontsize=10, ncol=4, frameon=True,
              loc="best", handlelength=1.2)
    fig.tight_layout()
    return fig_to_b64(fig)


def render_stacked_panel(records, key, color_by_cluster, figsize=(12, 7.5)) -> str:
    """All spectra, offset vertically by cluster (~5% of span per cluster)."""
    from matplotlib.lines import Line2D

    xlabel, ylabel, name = SPEC_META[key]
    fig, ax = plt.subplots(figsize=figsize)

    data = {}
    spans = []
    for r in records:
        xy = spec_xy(r, key)
        if xy is None:
            continue
        data[r["id"]] = (xy, r["cluster"])
        spans.append(float(np.ptp(xy[1])))
    span = max(spans) if spans else 1.0
    step = 0.05 * span  # 5% offset per cluster

    clusters = sorted({c for _, c in data.values()}, key=cluster_sort_key)
    for i, c in enumerate(clusters):
        off = i * step
        for sid, (xy, cl) in data.items():
            if cl != c:
                continue
            ax.plot(xy[0], xy[1] + off, color=color_by_cluster[c],
                    lw=1.3, alpha=0.8, zorder=2)

    ax.set_xlabel(xlabel, fontsize=_FS_LABEL)
    ax.set_ylabel(f"{ylabel}  + cluster offset", fontsize=_FS_LABEL)
    ax.set_title(f"{name} — stacked by cluster (5% offset)", fontsize=_FS_TITLE)
    ax.tick_params(labelsize=_FS_TICK)
    _apply_spec_axis(ax, key)
    handles = [Line2D([0], [0], color=color_by_cluster[c], lw=3, label=f"cluster {c}")
               for c in clusters]
    ax.legend(handles=handles, fontsize=10, ncol=4, frameon=True,
              loc="best", handlelength=1.2)
    fig.tight_layout()
    return fig_to_b64(fig)


def render_structure_triplet(r, color) -> str:
    """A compact 1x3 strip (XANES | χ(k) | |χ(R)|) for one structure, tile-scale."""
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 1.9))
    for ax, key in zip(axes, ("xanes", "exafs", "chir")):
        xlabel, ylabel, name = SPEC_META[key]
        xy = spec_xy(r, key)
        if xy is not None:
            ax.plot(xy[0], xy[1], color=color, lw=1.6)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.tick_params(labelsize=6)
        _apply_spec_axis(ax, key)
    fig.tight_layout(pad=0.4)
    return fig_to_b64(fig, dpi=110)


# ----------------------------------------------------------------------------
# Per-metric overlays — reuse the report's pre-rendered histogram PNGs verbatim.
# ----------------------------------------------------------------------------
def discover_overlay_pngs(overlay_dir: Path) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    if not overlay_dir.is_dir():
        return items
    suf = "_all_clusters_overlay.png"
    for p in sorted(overlay_dir.glob("*.png")):
        if p.name.endswith(suf):
            metric = p.name[: -len(suf)]
            items.append({"title": metric.replace("_", " "),
                          "b64": base64.b64encode(p.read_bytes()).decode("ascii")})
    return items


# ----------------------------------------------------------------------------
# Standalone overlay PNGs (kept for quick reference)
# ----------------------------------------------------------------------------
def write_standalone_pngs(records, color_by_cluster, embed, out_dir: Path) -> None:
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    for r in records:
        xy = embed.get(r["id"])
        if xy is None:
            continue
        ax.scatter(*xy, s=90, marker="o", color=color_by_cluster[r["cluster"]],
                   edgecolors="black", linewidths=1.0, zorder=3)
        ax.annotate(r["id"].split("_")[0], xy, fontsize=9, alpha=0.75,
                    xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("Sampled structures in t-SNE space")
    used = sorted({r["cluster"] for r in records}, key=cluster_sort_key)
    ax.legend(handles=[Line2D([0], [0], marker="o", color="w",
                              markerfacecolor=color_by_cluster[c], markersize=10,
                              label=f"cluster {c}") for c in used],
              fontsize=11, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(out_dir / "tsne_scatter.png", dpi=150)
    plt.close(fig)

    for key, fname in (("xanes", "xanes_overlay.png"),
                       ("exafs", "exafs_overlay.png"),
                       ("chir", "chir_overlay.png")):
        b64 = render_combined_panel(records, key, color_by_cluster, figsize=(9, 6.2))
        (out_dir / fname).write_bytes(base64.b64decode(b64))


# ----------------------------------------------------------------------------
# Interactive offline HTML
# ----------------------------------------------------------------------------
def _num(row, col, digits):
    v = row.get(col, "")
    if v in (None, "", "nan"):
        return "n/a"
    if digits is None:
        return html_lib.escape(str(v))
    try:
        return f"{float(v):.{digits}f}"
    except ValueError:
        return html_lib.escape(str(v))


def build_html(*, title, records, labels, embed, color_by_cluster, medoids,
               overlay_images, inline_plotly_js, out: Path) -> None:
    sampled_ids = {r["id"] for r in records}
    medoid_by_cluster = dict(medoids)                 # cluster(str) -> medoid id
    medoid_ids = set(medoids.values())
    all_rows = list(labels.values())

    # All points for the t-SNE (full dataset present in embeddings + labels).
    points = []
    for sid, (x, y) in embed.items():
        lab = labels.get(sid)
        if lab is None:
            continue
        cluster = str(lab.get("cluster", ""))
        stats = {label: _num(lab, col, digits) for col, label, digits in HERO_STATS}
        is_medoid = sid in medoid_ids
        is_sampled = sid in sampled_ids
        role = " ★ medoid" if is_medoid else (" (sampled)" if is_sampled else "")
        points.append({
            "id": sid,
            "cluster": cluster,
            "x": x, "y": y,
            "sampled": is_sampled,
            "medoid": is_medoid,
            "stats": stats,
            "hover": f"<b>{html_lib.escape(sid)}</b><br>cluster {html_lib.escape(cluster)}{role}",
        })

    all_clusters = sorted({str(c) for c in labels_clusters(labels)}, key=cluster_sort_key)

    # --- per-cluster rows: histograms + spectra, every cluster ---
    print("Rendering per-cluster rows (histograms + spectra)...")
    cluster_rows = []
    for c in all_clusters:
        color = color_by_cluster.get(c, "#444444")
        b64 = render_cluster_row(c, all_rows, records, color)
        cluster_rows.append({"cluster": c, "b64": b64})

    print("Rendering combined spectra overlays...")
    combined = {k: render_combined_panel(records, k, color_by_cluster)
                for k in ("xanes", "exafs", "chir")}

    print("Rendering stacked-by-cluster overlays...")
    stacked = {k: render_stacked_panel(records, k, color_by_cluster)
               for k in ("xanes", "exafs", "chir")}

    print("Rendering compact per-structure hero strips...")
    structure_imgs = {r["id"]: render_structure_triplet(r, color_by_cluster.get(r["cluster"], "#444"))
                      for r in records}

    # --- HTML for the right panel ---
    def img_tag(b64, alt):
        return f'<img src="data:image/png;base64,{b64}" alt="{html_lib.escape(alt)}">'

    rows_html = "\n".join(
        f'<div class="cluster-row-card">'
        f'<div class="cluster-row-title">Cluster {html_lib.escape(row["cluster"])}</div>'
        f'{img_tag(row["b64"], "cluster " + row["cluster"] + " row")}'
        f'</div>'
        for row in cluster_rows
    )

    overlay_html = "\n".join(
        f'<div class="metric-row-card">'
        f'<div class="cluster-row-title">{html_lib.escape(it["title"])}</div>'
        f'{img_tag(it["b64"], it["title"])}'
        f'</div>'
        for it in overlay_images
    ) or '<div class="empty-note">No per-metric overlay PNGs found in the clustering dir.</div>'

    spectra_html = "\n".join(
        f'<div class="metric-row-card"><div class="cluster-row-title">{lbl}</div>{img_tag(img, lbl)}</div>'
        for lbl, img in [
            ("XANES — combined", combined["xanes"]),
            ("EXAFS χ(k) — combined", combined["exafs"]),
            ("|χ(R)| — combined", combined["chir"]),
            ("XANES — stacked by cluster", stacked["xanes"]),
            ("EXAFS χ(k) — stacked by cluster", stacked["exafs"]),
            ("|χ(R)| — stacked by cluster", stacked["chir"]),
        ]
    )

    state = {
        "points": points,
        "colors": color_by_cluster,
        "heroFields": [label for _, label, _ in HERO_STATS],
        "medoidByCluster": medoid_by_cluster,
        "nSampled": len(sampled_ids),
        "nTotal": len(points),
        "spectra": structure_imgs,
    }
    state_json = json.dumps(state)

    plotly_html = (f'<script src="{PLOTLY_CDN}"></script>' if inline_plotly_js is None
                   else f"<script>{inline_plotly_js}</script>")

    css = _CSS
    js = "const STATE = " + state_json + ";\n" + _JS

    out.write_text(f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{html_lib.escape(title)}</title>
  {plotly_html}
  <style>{css}</style>
</head>
<body>
  <header>
    <h1>{html_lib.escape(title)}</h1>
    <p>Clickable t-SNE at left. Right panel toggles between per-cluster rows, per-metric overlays, and spectra. Hero shows the selected structure's stats.</p>
  </header>
  <main class="wrap">
    <section class="hero">
      <div class="hero-head">
        <div>
          <h2 id="hero-title" class="hero-title">No point selected</h2>
          <div id="hero-sub" class="hero-sub">Click a point in the t-SNE panel.</div>
        </div>
        <div class="hero-right">
          <div id="hero-cluster-line" class="cluster-line">Cluster &mdash;</div>
          <div id="hero-spectra-line" class="count-line">&mdash;</div>
        </div>
      </div>
      <div id="hero-grid" class="hero-grid"></div>
      <div id="hero-spectra" class="hero-spectra" style="display:none;">
        <div class="hero-spectra-label">sampled spectra</div>
        <img id="hero-spectra-img" src="" alt="structure spectra">
      </div>
    </section>

    <section class="top-layout">
      <div class="panel">
        <div class="panel-head">t-SNE (click points)</div>
        <div class="toggle-bar">
          <button id="btn-all" class="active">All points</button>
          <button id="btn-sampled">Sampled only</button>
        </div>
        <div id="tsne-plot"></div>
      </div>
      <div class="panel">
        <div class="panel-head">Cluster distributions &amp; spectra</div>
        <div class="toggle-bar">
          <button id="btn-rows" class="active">Per Cluster Rows</button>
          <button id="btn-metric">Per Metric Overlays</button>
          <button id="btn-spectra">Spectra</button>
        </div>
        <div id="rows-view" class="rows-scroll">{rows_html}</div>
        <div id="metric-view" class="rows-scroll" style="display:none;">{overlay_html}</div>
        <div id="spectra-view" class="rows-scroll" style="display:none;">{spectra_html}</div>
      </div>
    </section>
  </main>
  <script>{js}</script>
</body>
</html>
""", encoding="utf-8")


def labels_clusters(labels: dict):
    return {str(row.get("cluster", "")) for row in labels.values() if str(row.get("cluster", "")).strip()}


_CSS = """
:root {
  --bg: #f3f1ed; --card: #ffffff; --ink: #1e1d1a; --muted: #6c6a63;
  --border: #d9d4ca; --accent: #0d6b60; --accent-soft: #e4f1ef;
  --hero-accent: #0d6b60; --hero-soft: rgba(13,107,96,0.12); --hero-text: #111111;
}
* { box-sizing: border-box; }
body {
  margin: 0; color: var(--ink);
  background: radial-gradient(circle at 20% -10%, #e7ece7 0%, var(--bg) 40%, #ebe7df 100%);
  font-family: "STIX Two Text", "Iowan Old Style", "Times New Roman", serif;
}
header {
  padding: 30px 28px 18px 28px; border-bottom: 1px solid var(--border);
  background: linear-gradient(135deg, #f8faf7 0%, #ece8df 100%);
}
header h1 { margin: 0; font-size: 30px; letter-spacing: -0.01em; }
header p { margin: 8px 0 0 0; color: var(--muted); font-size: 14px; }
.wrap { max-width: 1700px; margin: 0 auto; padding: 18px 22px 24px 22px; }

.hero {
  margin: 0 0 16px 0; border: 2px solid var(--hero-accent); border-radius: 12px;
  background: linear-gradient(180deg, #ffffff 0%, #f8fbfa 100%);
  padding: 14px; box-shadow: 0 0 0 1px rgba(0,0,0,0.03), 0 4px 16px rgba(0,0,0,0.05);
}
.hero-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 10px; margin-bottom: 10px; }
.hero-title { margin: 0; font-size: 20px; }
.hero-sub { color: var(--muted); font-size: 13px; margin-top: 4px; }
.hero-right { text-align: right; font-size: 13px; color: #43413c; line-height: 1.4; min-width: 220px; }
.hero-right .cluster-line { font-weight: 700; color: var(--hero-accent); }
.hero-right .count-line { color: #5c5952; }
.hero-grid { display: grid; grid-template-columns: repeat(5, minmax(140px, 1fr)); gap: 8px; }
.metric { border: 1px solid #d7e3e0; border-radius: 8px; padding: 8px; background: var(--hero-soft); }
.metric .k { font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em; color: var(--hero-text); opacity: 0.82; }
.metric .v { margin-top: 2px; font-size: 17px; font-weight: 600; color: var(--hero-text); }

/* Compact spectra strip — secondary to the stat tiles. */
.hero-spectra { margin-top: 10px; padding-top: 8px; border-top: 1px dashed var(--hero-accent); }
.hero-spectra-label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em; color: var(--hero-accent); opacity: 0.85; margin-bottom: 4px; }
.hero-spectra img { max-width: 560px; width: 100%; height: auto; display: block; }

.top-layout { display: grid; grid-template-columns: minmax(400px, 560px) 1fr; gap: 16px; align-items: start; }
.panel { background: var(--card); border: 1px solid var(--border); border-radius: 10px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
.panel-head {
  padding: 10px 12px; border-bottom: 1px solid var(--border); background: #fcfbf8;
  font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted);
}
#tsne-plot { width: 100%; height: 620px; }

.toggle-bar { display: inline-flex; margin: 12px; border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }
.toggle-bar button {
  border: none; padding: 9px 14px; font-size: 13px; cursor: pointer; background: white;
  color: #3d3b36; border-right: 1px solid var(--border); font-family: inherit;
}
.toggle-bar button:last-child { border-right: none; }
.toggle-bar button.active { background: var(--accent); color: white; font-weight: 600; }

.rows-scroll { max-height: 760px; overflow: auto; padding: 0 12px 12px 12px; display: grid; gap: 10px; }
.cluster-row-card, .metric-row-card { border: 1px solid var(--border); border-radius: 8px; background: #fff; overflow: hidden; }
.cluster-row-title { padding: 8px 10px; font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; color: #4f4c46; background: #faf8f3; border-bottom: 1px solid var(--border); }
.cluster-row-card img, .metric-row-card img { width: 100%; height: auto; display: block; }
.empty-note { padding: 16px; color: var(--muted); font-size: 13px; }

@media (max-width: 1250px) {
  .top-layout { grid-template-columns: 1fr; }
  #tsne-plot { height: 500px; }
  .rows-scroll { max-height: 620px; }
  .hero-grid { grid-template-columns: repeat(2, minmax(140px, 1fr)); }
}
"""

_JS = r"""
function hexToRgb(hex) {
  const m = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex || "");
  return m ? { r: parseInt(m[1],16), g: parseInt(m[2],16), b: parseInt(m[3],16) } : null;
}

function metricHtml(label, value) {
  return '<div class="metric"><div class="k">' + label + '</div><div class="v">' + value + '</div></div>';
}

function updateHero(p) {
  const cluster = String(p.cluster);
  document.getElementById("hero-title").textContent = p.id;
  let sub;
  if (p.medoid) sub = "★ Medoid of cluster " + cluster + " (spectra computed)";
  else if (p.sampled) sub = "Sampled structure (spectra computed)";
  else sub = "From full dataset (no spectra sampled)";
  document.getElementById("hero-sub").textContent = sub;
  document.getElementById("hero-cluster-line").textContent = "Cluster " + cluster + (p.medoid ? " ★" : "");
  document.getElementById("hero-spectra-line").textContent = p.sampled ? "✓ spectra available" : "no spectra";

  const accent = STATE.colors[cluster] || "#0d6b60";
  const rgb = hexToRgb(accent);
  document.documentElement.style.setProperty("--hero-accent", accent);
  if (rgb) document.documentElement.style.setProperty("--hero-soft",
    "rgba(" + rgb.r + "," + rgb.g + "," + rgb.b + ",0.14)");

  document.getElementById("hero-grid").innerHTML =
    STATE.heroFields.map(f => metricHtml(f, p.stats[f] != null ? p.stats[f] : "n/a")).join("");

  const box = document.getElementById("hero-spectra");
  const img = STATE.spectra[p.id];
  if (p.sampled && img) {
    document.getElementById("hero-spectra-img").src = "data:image/png;base64," + img;
    box.style.display = "block";
  } else {
    box.style.display = "none";
  }
}

let SHOW_ALL = true;
let BG_TRACE_IDX = [];

function buildTsnePlot() {
  const byCluster = {};
  STATE.points.forEach(p => {
    const c = String(p.cluster);
    (byCluster[c] = byCluster[c] || []).push(p);
  });
  const clusters = Object.keys(byCluster).sort((a, b) => Number(a) - Number(b));

  const traces = [];
  BG_TRACE_IDX = [];
  // Background: all non-sampled points, small + faint.
  clusters.forEach(c => {
    const pts = byCluster[c].filter(p => !p.sampled);
    if (!pts.length) return;
    BG_TRACE_IDX.push(traces.length);
    traces.push({
      x: pts.map(p => p.x), y: pts.map(p => p.y), customdata: pts,
      mode: "markers", type: "scattergl", name: "cluster " + c, legendgroup: c,
      marker: { size: 6, color: STATE.colors[c] || "#444", opacity: 0.45, line: { width: 0 } },
      hovertemplate: "%{customdata.hover}<extra></extra>",
    });
  });
  // Sampled (incl. medoids): outlined circles, slightly smaller.
  clusters.forEach(c => {
    const pts = byCluster[c].filter(p => p.sampled);
    if (!pts.length) return;
    traces.push({
      x: pts.map(p => p.x), y: pts.map(p => p.y), customdata: pts,
      mode: "markers", type: "scattergl", name: "cluster " + c + " (sampled)", legendgroup: c,
      showlegend: false,
      marker: { size: 8, color: STATE.colors[c] || "#444", line: { width: 1.5, color: "#111" } },
      hovertemplate: "%{customdata.hover}<extra></extra>",
    });
  });

  const xs = STATE.points.map(p => p.x), ys = STATE.points.map(p => p.y);
  const xmin = Math.min(...xs), xmax = Math.max(...xs);
  const ymin = Math.min(...ys), ymax = Math.max(...ys);
  const px = (xmax - xmin) * 0.05, py = (ymax - ymin) * 0.05;

  const layout = {
    margin: { l: 52, r: 14, t: 16, b: 46 },
    xaxis: { title: "t-SNE 1", range: [xmin - px, xmax + px], zeroline: false },
    yaxis: { title: "t-SNE 2", range: [ymin - py, ymax + py], zeroline: false },
    legend: { orientation: "v", x: 1.01, y: 1, font: { size: 12 } },
    hovermode: "closest", plot_bgcolor: "white", paper_bgcolor: "white",
  };

  Plotly.newPlot("tsne-plot", traces, layout, { displayModeBar: false, responsive: true });

  document.getElementById("tsne-plot").on("plotly_click", ev => {
    if (ev && ev.points && ev.points.length) {
      const p = ev.points[0].customdata;
      if (p) updateHero(p);
    }
  });

  const first = STATE.points.find(p => p.medoid) || STATE.points.find(p => p.sampled) || STATE.points[0];
  if (first) updateHero(first);
}

function setTsneMode(showAll) {
  SHOW_ALL = showAll;
  Plotly.restyle("tsne-plot", { visible: showAll ? true : "legendonly" }, BG_TRACE_IDX);
  document.getElementById("btn-all").classList.toggle("active", showAll);
  document.getElementById("btn-sampled").classList.toggle("active", !showAll);
}

function setSpectraMode(mode) {
  const views = { rows: "rows-view", metric: "metric-view", spectra: "spectra-view" };
  const btns = { rows: "btn-rows", metric: "btn-metric", spectra: "btn-spectra" };
  Object.keys(views).forEach(k => {
    document.getElementById(views[k]).style.display = (k === mode) ? "grid" : "none";
    document.getElementById(btns[k]).classList.toggle("active", k === mode);
  });
}

window.addEventListener("DOMContentLoaded", () => {
  document.getElementById("btn-all").addEventListener("click", () => setTsneMode(true));
  document.getElementById("btn-sampled").addEventListener("click", () => setTsneMode(false));
  document.getElementById("btn-rows").addEventListener("click", () => setSpectraMode("rows"));
  document.getElementById("btn-metric").addEventListener("click", () => setSpectraMode("metric"));
  document.getElementById("btn-spectra").addEventListener("click", () => setSpectraMode("spectra"));
  buildTsnePlot();
});
"""


# ----------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--station", type=Path, default=DEFAULT_STATION)
    ap.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("--embeddings", type=Path, default=DEFAULT_EMBED)
    ap.add_argument("--medoids", type=Path, default=DEFAULT_MEDOIDS)
    ap.add_argument("--clustering-dir", type=Path, default=None,
                    help="Dir holding cluster_distribution_plots/ (default: labels' parent)")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output dir (default: analysis-and-visualization/ beside the station dir)")
    ap.add_argument("--cdn", action="store_true",
                    help="Link Plotly.js from CDN instead of inlining (smaller file, needs internet)")
    args = ap.parse_args()

    setup_style()
    out_dir = args.out or (args.station.parent / "analysis-and-visualization")
    out_dir.mkdir(parents=True, exist_ok=True)
    clustering_dir = args.clustering_dir or args.labels.parent
    overlay_dir = clustering_dir / "cluster_distribution_plots" / "all_cluster_overlays"

    print(f"Reading labels      : {args.labels}")
    print(f"Reading embeddings  : {args.embeddings}")
    print(f"Reading medoids     : {args.medoids}")
    print(f"Overlay PNGs        : {overlay_dir}")
    print(f"Scanning structures : {args.station}")
    labels = read_labels(args.labels)
    embed = read_embeddings(args.embeddings)
    medoids = read_medoids(args.medoids)
    records = collect_spectra(args.station, labels, medoids)
    if not records:
        print("No matched structures found.")
        return 1

    color_by_cluster = build_color_map(labels, extra_clusters=[r["cluster"] for r in records])
    n_clusters_sampled = len({r["cluster"] for r in records})
    print(f"Matched {len(records)} sampled structures "
          f"({n_clusters_sampled} clusters); full dataset = {len(embed)} points.")

    # Report any cluster whose medoid spectrum is absent from the station.
    sampled_ids = {r["id"] for r in records}
    missing = [(c, mid) for c, mid in sorted(medoids.items(), key=lambda kv: cluster_sort_key(kv[0]))
               if mid not in sampled_ids]
    if missing:
        print("\n  ! medoid spectra MISSING from the station (cluster → medoid_id):")
        for c, mid in missing:
            print(f"      cluster {c}: {mid}")
        print("    Those cluster rows show histograms + gray 'all sampled' spectra only.")

    overlay_images = discover_overlay_pngs(overlay_dir)
    print(f"Per-metric overlay PNGs found: {len(overlay_images)}")

    write_standalone_pngs(records, color_by_cluster, embed, out_dir)

    inline_js = None
    if not args.cdn:
        print("Inlining Plotly.js for offline use...")
        import plotly
        plotly_min = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
        inline_js = plotly_min.read_text(encoding="utf-8")

    build_html(
        title="4Cys Cluster Distributions & Sampled Spectra",
        records=records, labels=labels, embed=embed,
        color_by_cluster=color_by_cluster, medoids=medoids,
        overlay_images=overlay_images, inline_plotly_js=inline_js,
        out=out_dir / "spectra_tsne_report_offline.html",
    )

    print(f"\nWrote outputs to: {out_dir}")
    for f in sorted(out_dir.iterdir()):
        size = f.stat().st_size
        unit = f"{size/1e6:.1f} MB" if size > 1e6 else f"{size/1e3:.0f} KB"
        print(f"  {f.name}  ({unit})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
