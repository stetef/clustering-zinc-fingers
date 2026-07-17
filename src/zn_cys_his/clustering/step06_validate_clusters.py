#!/usr/bin/env python3
"""Cluster validation: RMSD metrics, plots, PCA→XYZ reconstruction, comparison,
and interactive HTML distribution reports.

Reads the standard outputs from any approach script (labels.csv or
kmeans_labels_with_stats.csv, medoids.csv, optionally k_sweep.csv) together
with the original XYZ files, and produces:

  k_sweep_plot.png         ratio vs k curve; best k marked  (if k_sweep.csv present)
  cluster_sizes.png        bar chart of cluster sizes at best k
  rmsd_scatter.png         intra vs inter per cluster (scatter)
  rmsd_table.csv           cluster_id, n, intra, inter
  atom_cloud.html          interactive 3D point cloud of all heavy-atom positions
  pca_to_xyz/              (approach 1 only) mean + PC perturbation XYZ files,
                           pc_morph.gif (animated ±PC morph), pc_arrows.html (interactive quiver)
  comparison_table.csv     (multi-approach) best k, intra, inter, ratio
  comparison_plot.png      (multi-approach) bar chart of ratios
  report_cluster_distribution.html       interactive t-SNE + histograms (CDN-backed)
  report_cluster_distribution_offline.html  same, fully self-contained

The HTML report is generated automatically when the approach dir (or
--clustering-dir) contains embeddings.csv (with tsne1/tsne2 columns),
kmeans_cluster_stats_summary.csv, tsne_kmeans.png, and
cluster_distribution_plots/.  Missing inputs produce a skip message, not an
error.

Labels are read from labels.csv (column: structure_id) when present, falling
back to kmeans_labels_with_stats.csv (column: id).  Medoid column names
medoid_id and medoid_filename are both accepted.  k_sweep.csv is optional;
when absent, k is inferred from the number of unique cluster labels.

Usage — single approach (approach-dir with k_sweep, labels, medoids)
---------------------------------------------------------------------
  uv run python -m zn_cys_his.clustering.step06_validate_clusters \\
      --xyz-dir   data/1cys3his-large/xyz-files \\
      --approach-dir cluster-output/1cys3his-large/approach1 \\
      --out-dir   cluster-output/1cys3his-large/validation/approach1

  # Approach 1 PCA→XYZ requires the aligned structures
  uv run python -m zn_cys_his.clustering.step06_validate_clusters \\
      --xyz-dir   cluster-output/1cys3his-large/approach1/aligned_xyz \\
      --approach-dir cluster-output/1cys3his-large/approach1 \\
      --approach1 \\
      --out-dir   cluster-output/1cys3his-large/validation/approach1

Usage — clustering dir (has embeddings/tsne/histograms, may lack k_sweep)
--------------------------------------------------------------------------
  uv run python -m zn_cys_his.clustering.step06_validate_clusters \\
      --approach-dir data/4cys-large/clustering_baseline \\
      --out-dir   data/4cys-large/clustering_baseline

Usage — compare multiple approaches
------------------------------------
  uv run python -m zn_cys_his.clustering.step06_validate_clusters \\
      --compare-dirs cluster-output/1cys3his-large/approach1 cluster-output/1cys3his-large/approach3 \\
      --out-dir  cluster-output/1cys3his-large/validation/comparison
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import csv
import html as html_lib
import json
import math
import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import tqdm

from zn_cys_his.clustering.utils import (
    EQUAL_WEIGHTS, SHELL_WEIGHTS, DISTANCE_WEIGHTS,
    Structure, parse_structure,
    structural_rmsd, cluster_pipeline,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MPL_OK = True
except ImportError:
    MPL_OK = False
    print("matplotlib not available; plots will be skipped.")

try:
    from sklearn.decomposition import PCA
except ImportError:
    pass


# ---------------------------------------------------------------------------
# HTML report helpers
# ---------------------------------------------------------------------------

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"


def _fetch(url: str, cache_dir: Path) -> bytes:
    import hashlib
    import urllib.request

    cache_dir.mkdir(parents=True, exist_ok=True)
    name = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    suffix = Path(url.split("?")[0]).suffix or ".bin"
    cached = cache_dir / f"{name}{suffix}"
    if cached.exists():
        return cached.read_bytes()

    print(f"  downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "cluster_distribution_report/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
    cached.write_bytes(data)
    return data


def _png_to_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def _cluster_sort_key(text: str) -> tuple[int, str]:
    s = text.strip()
    try:
        return (0, f"{int(s):09d}")
    except ValueError:
        return (1, s)


def _float_or_none(value: str | None) -> float | None:
    if value is None:
        return None
    s = value.strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _read_summary_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        cluster = (row.get("cluster") or "").strip()
        if cluster:
            out[cluster] = row
    return out


def _read_label_metadata(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Read id→cluster and cluster→color from kmeans_labels_with_stats.csv."""
    id_to_cluster: dict[str, str] = {}
    color_votes: dict[str, Counter] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sid = (row.get("id") or "").strip()
            cluster = (row.get("cluster") or "").strip()
            if sid and cluster:
                id_to_cluster[sid] = cluster
            color = (row.get("cluster_color") or "").strip().lower()
            if cluster and color and re.match(r"^#[0-9a-f]{6}$", color):
                color_votes.setdefault(cluster, Counter())[color] += 1
    cluster_to_color: dict[str, str] = {}
    for cluster, counter in color_votes.items():
        cluster_to_color[cluster] = counter.most_common(1)[0][0]
    return id_to_cluster, cluster_to_color


_CLUSTER_SUFFIX_RE = re.compile(r"_cluster(\d+)$", flags=re.IGNORECASE)


def _read_embedding_points(path: Path, id_to_cluster: dict[str, str] | None = None) -> list[dict]:
    points: list[dict] = []
    id_to_cluster = id_to_cluster or {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sid = (row.get("id") or "").strip()
            if not sid:
                continue
            cluster = id_to_cluster.get(sid, "")
            if not cluster:
                m = _CLUSTER_SUFFIX_RE.search(sid)
                if not m:
                    continue
                cluster = m.group(1)
            x = _float_or_none(row.get("tsne1"))
            y = _float_or_none(row.get("tsne2"))
            if x is None or y is None:
                continue
            points.append({"id": sid, "cluster": cluster, "x": x, "y": y})
    return points


def _discover_row_images(per_cluster_dir: Path) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    pat = re.compile(r"cluster_(\d+)_metrics_row\.png$", flags=re.IGNORECASE)
    for p in sorted(per_cluster_dir.glob("*.png")):
        m = pat.match(p.name)
        if m:
            items.append({"cluster": m.group(1), "path": str(p), "name": p.name})
    items.sort(key=lambda d: _cluster_sort_key(d["cluster"]))
    return items


def _discover_overlay_images(overlay_dir: Path) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    suf = "_all_clusters_overlay.png"
    for p in sorted(overlay_dir.glob("*.png")):
        if p.name.endswith(suf):
            metric = p.name[: -len(suf)]
            items.append({"metric": metric, "title": metric.replace("_", " "),
                          "path": str(p), "name": p.name})
    return items


def _fmt_stat(v: str | None, digits: int = 4) -> str:
    f = _float_or_none(v)
    return "n/a" if f is None else f"{f:.{digits}f}"


def _build_report_html(
    *,
    title: str,
    summary_rows: dict[str, dict[str, str]],
    points: list[dict],
    color_by_cluster: dict[str, str],
    tsne_b64: str,
    row_images: list[dict[str, str]],
    overlay_images: list[dict[str, str]],
    image_b64: dict[str, str],
    inline_plotly_js: str | None,
) -> str:
    clusters_sorted = sorted(summary_rows.keys(), key=_cluster_sort_key)

    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
    ]
    all_clusters = sorted(
        {str(p["cluster"]) for p in points} | set(clusters_sorted),
        key=_cluster_sort_key,
    )
    for i, c in enumerate(all_clusters):
        color_by_cluster.setdefault(c, palette[i % len(palette)])

    rows_minicards = []
    for c in clusters_sorted:
        r = summary_rows[c]
        rows_minicards.append((c, {
            "cluster": c,
            "n_total": r.get("n_total", ""),
            "n_with_stats": r.get("n_with_stats", ""),
            "volume_mean": _fmt_stat(r.get("volume_A3_mean"), 3),
            "q_coord_mean": _fmt_stat(r.get("q_tetra_coord_mean"), 4),
            "q_ca_mean": _fmt_stat(r.get("q_tetra_ca_mean"), 4),
            "r_work_mean": _fmt_stat(r.get("r_work_mean"), 4),
            "r_free_mean": _fmt_stat(r.get("r_free_mean"), 4),
            "zn_bfactor_mean": _fmt_stat(r.get("zn_bfactor_mean"), 3),
            "dihedral_mean": _fmt_stat(r.get("all_dihedrals_deg_mean"), 2),
            "coord_res_bf_mean": _fmt_stat(r.get("all_coord_res_bfactor_avg_mean"), 3),
        }))

    state = {
        "summary": {c: d for c, d in rows_minicards},
        "clusters": clusters_sorted,
        "points": points,
        "colors": color_by_cluster,
        "initialCluster": clusters_sorted[0] if clusters_sorted else None,
        "tsneImage": f"data:image/png;base64,{tsne_b64}",
    }
    state_json = json.dumps(state)

    row_cards_html = "\n".join(
        f'<div class="cluster-row-card" data-cluster="{html_lib.escape(it["cluster"])}">'
        f'<div class="cluster-row-title">Cluster {html_lib.escape(it["cluster"])}</div>'
        f'<img src="data:image/png;base64,{image_b64[it["path"]]}" alt="{html_lib.escape(it["name"])}">'
        f"</div>"
        for it in row_images
    )

    overlay_cards_html = "\n".join(
        f'<div class="metric-row-card">'
        f'<div class="cluster-row-title">{html_lib.escape(it["title"])}</div>'
        f'<img src="data:image/png;base64,{image_b64[it["path"]]}" alt="{html_lib.escape(it["name"])}">'
        f"</div>"
        for it in overlay_images
    )

    plotly_html = (f'<script src="{PLOTLY_CDN}"></script>' if inline_plotly_js is None
                   else f"<script>{inline_plotly_js}</script>")

    css = """
    :root {
      --bg: #f3f1ed;
      --card: #ffffff;
      --ink: #1e1d1a;
      --muted: #6c6a63;
      --border: #d9d4ca;
      --accent: #0d6b60;
      --accent-soft: #e4f1ef;
      --hero-accent: #0d6b60;
      --hero-soft: rgba(13, 107, 96, 0.12);
      --hero-text: #111111;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: radial-gradient(circle at 20% -10%, #e7ece7 0%, var(--bg) 40%, #ebe7df 100%);
      color: var(--ink);
      font-family: "STIX Two Text", "Iowan Old Style", "Times New Roman", serif;
    }
    header {
      padding: 30px 28px 18px 28px;
      border-bottom: 1px solid var(--border);
      background: linear-gradient(135deg, #f8faf7 0%, #ece8df 100%);
    }
    header h1 { margin: 0; font-size: 30px; letter-spacing: -0.01em; }
    header p { margin: 8px 0 0 0; color: var(--muted); font-size: 14px; }
    .wrap { max-width: 1700px; margin: 0 auto; padding: 18px 22px 24px 22px; }
    .top-layout {
      display: grid;
      grid-template-columns: minmax(400px, 540px) 1fr;
      gap: 16px;
      align-items: start;
    }
    .panel {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 10px;
      overflow: hidden;
      box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .panel-head {
      padding: 10px 12px;
      border-bottom: 1px solid var(--border);
      background: #fcfbf8;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
    }
    #tsne-plot { width: 100%; height: 620px; }
    .toggle-bar {
      display: inline-flex;
      margin: 12px;
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
    }
    .toggle-bar button {
      border: none;
      padding: 9px 14px;
      font-size: 13px;
      cursor: pointer;
      background: white;
      color: #3d3b36;
      border-right: 1px solid var(--border);
    }
    .toggle-bar button:last-child { border-right: none; }
    .toggle-bar button.active { background: var(--accent); color: white; font-weight: 600; }
    .rows-scroll {
      max-height: 620px;
      overflow: auto;
      padding: 0 12px 12px 12px;
      display: grid;
      gap: 10px;
    }
    .cluster-row-card, .metric-row-card {
      border: 1px solid var(--border);
      border-radius: 8px;
      background: #fff;
      overflow: hidden;
    }
    .cluster-row-title {
      padding: 8px 10px;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: #4f4c46;
      background: #faf8f3;
      border-bottom: 1px solid var(--border);
    }
    .cluster-row-card img, .metric-row-card img { width: 100%; height: auto; display: block; }
    .hero {
      margin: 0 0 16px 0;
      border: 2px solid var(--hero-accent);
      border-radius: 12px;
      background: linear-gradient(180deg, #ffffff 0%, #f8fbfa 100%);
      padding: 14px;
      box-shadow: 0 0 0 1px rgba(0,0,0,0.03), 0 4px 16px rgba(0,0,0,0.05);
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
    @media (max-width: 1250px) {
      .top-layout { grid-template-columns: 1fr; }
      #tsne-plot { height: 500px; }
      .rows-scroll { max-height: 560px; }
      .hero-grid { grid-template-columns: repeat(2, minmax(140px, 1fr)); }
    }
    """

    js = """
const STATE = """ + state_json + """;

function showRows(mode) {
  const a = document.getElementById("rows-per-cluster");
  const b = document.getElementById("rows-per-metric");
  const ba = document.getElementById("btn-cluster");
  const bb = document.getElementById("btn-metric");
  const clusterMode = mode === "cluster";
  a.style.display = clusterMode ? "grid" : "none";
  b.style.display = clusterMode ? "none" : "grid";
  ba.classList.toggle("active", clusterMode);
  bb.classList.toggle("active", !clusterMode);
}

function metricHtml(label, value) {
  return '<div class="metric"><div class="k">' + label + '</div><div class="v">' + value + '</div></div>';
}

function hexToRgb(hex) {
  const m = /^#?([a-f\\d]{2})([a-f\\d]{2})([a-f\\d]{2})$/i.exec(hex || "");
  if (!m) return null;
  return { r: parseInt(m[1], 16), g: parseInt(m[2], 16), b: parseInt(m[3], 16) };
}

function updateHero(point) {
  const cluster = String(point.cluster);
  const m = STATE.summary[cluster];
  document.getElementById("hero-title").textContent = point.id;
  document.getElementById("hero-sub").textContent = "Selected from t-SNE";

  const nTotal = m ? (m.n_total || "n/a") : "n/a";
  const nStats = m ? (m.n_with_stats || "n/a") : "n/a";
  document.getElementById("hero-cluster-line").textContent = "Cluster " + cluster;
  document.getElementById("hero-count-line").textContent = "N total: " + nTotal + " | N stats: " + nStats;

  const accent = STATE.colors[cluster] || "#0d6b60";
  const rgb = hexToRgb(accent);
  document.documentElement.style.setProperty("--hero-accent", accent);
  if (rgb) {
    document.documentElement.style.setProperty("--hero-soft", "rgba(" + rgb.r + ", " + rgb.g + ", " + rgb.b + ", 0.14)");
  }
  document.documentElement.style.setProperty("--hero-text", "#111111");

  if (!m) { document.getElementById("hero-grid").innerHTML = ""; return; }

  const html = [
    metricHtml("volume mean (A^3)", m.volume_mean),
    metricHtml("q tetra coord mean", m.q_coord_mean),
    metricHtml("q tetra CA mean", m.q_ca_mean),
    metricHtml("r work mean", m.r_work_mean),
    metricHtml("r free mean", m.r_free_mean),
    metricHtml("Zn B-factor mean", m.zn_bfactor_mean),
    metricHtml("dihedral mean (deg)", m.dihedral_mean),
    metricHtml("coord-res B mean", m.coord_res_bf_mean),
  ].join("");
  document.getElementById("hero-grid").innerHTML = html;
}

function buildTsnePlot() {
  const byCluster = {};
  STATE.points.forEach(p => {
    const c = String(p.cluster);
    if (!byCluster[c]) byCluster[c] = [];
    byCluster[c].push(p);
  });

  const traces = Object.keys(byCluster).sort((a, b) => Number(a) - Number(b)).map(c => {
    const pts = byCluster[c];
    return {
      x: pts.map(p => p.x),
      y: pts.map(p => p.y),
      customdata: pts,
      mode: "markers",
      type: "scattergl",
      name: "cluster " + c,
      marker: { size: 7, color: STATE.colors[c] || "#444", opacity: 0.72, line: { width: 0 } },
      hovertemplate: "<b>%{customdata.id}</b><br>cluster %{customdata.cluster}<br>t-SNE: (%{x:.2f}, %{y:.2f})<extra></extra>",
    };
  });

  const xs = STATE.points.map(p => p.x);
  const ys = STATE.points.map(p => p.y);
  const xmin = Math.min.apply(null, xs), xmax = Math.max.apply(null, xs);
  const ymin = Math.min.apply(null, ys), ymax = Math.max.apply(null, ys);
  const padx = (xmax - xmin) * 0.06, pady = (ymax - ymin) * 0.06;

  const layout = {
    margin: { l: 48, r: 14, t: 24, b: 42 },
    xaxis: { title: "t-SNE 1", range: [xmin - padx, xmax + padx], zeroline: false },
    yaxis: { title: "t-SNE 2", range: [ymin - pady, ymax + pady], zeroline: false },
    legend: { orientation: "h", yanchor: "bottom", y: 1.02, x: 0 },
    hovermode: "closest",
    plot_bgcolor: "white",
    paper_bgcolor: "white",
  };

  Plotly.newPlot("tsne-plot", traces, layout, { displayModeBar: false, responsive: true });

  const plot = document.getElementById("tsne-plot");
  plot.on("plotly_click", ev => {
    if (!ev || !ev.points || !ev.points.length) return;
    const p = ev.points[0].customdata;
    if (p) updateHero(p);
  });

  if (STATE.points.length) updateHero(STATE.points[0]);
}

window.addEventListener("DOMContentLoaded", () => {
  document.getElementById("btn-cluster").addEventListener("click", () => showRows("cluster"));
  document.getElementById("btn-metric").addEventListener("click", () => showRows("metric"));
  showRows("cluster");
  buildTsnePlot();
});
"""

    return f"""<!DOCTYPE html>
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
    <p>Clickable t-SNE at left. Right panel toggles between per-cluster rows and per-metric overlays. Hero metrics from kmeans_cluster_stats_summary.csv.</p>
  </header>
  <main class="wrap">
    <section class="hero">
      <div class="hero-head">
        <div>
          <h2 id="hero-title" class="hero-title">No point selected</h2>
          <div id="hero-sub" class="hero-sub">Click a point in the t-SNE panel.</div>
        </div>
        <div class="hero-right">
          <div id="hero-cluster-line" class="cluster-line">Cluster -</div>
          <div id="hero-count-line" class="count-line">N total: - | N stats: -</div>
        </div>
      </div>
      <div id="hero-grid" class="hero-grid"></div>
    </section>

    <section class="top-layout">
      <div class="panel">
        <div class="panel-head">t-SNE (click points)</div>
        <div id="tsne-plot"></div>
      </div>
      <div class="panel">
        <div class="panel-head">Cluster Distribution Panels</div>
        <div class="toggle-bar">
          <button id="btn-cluster" class="active">Per Cluster Rows</button>
          <button id="btn-metric">Per Metric Overlays</button>
        </div>
        <div id="rows-per-cluster" class="rows-scroll">{row_cards_html}</div>
        <div id="rows-per-metric" class="rows-scroll" style="display:none;">{overlay_cards_html}</div>
      </div>
    </section>
  </main>
  <script>{js}</script>
</body>
</html>
"""


def build_cluster_reports(
    clustering_dir: Path,
    out_dir: Path,
    title: str = "",
    vendor_cache: Path | None = None,
    best_k: int | None = None,
) -> None:
    """Generate online and offline HTML cluster distribution reports.

    Silently skips if any required input files are missing.
    """
    embeddings_csv = clustering_dir / "embeddings.csv"
    summary_csv    = clustering_dir / "kmeans_cluster_stats_summary.csv"
    labels_csv     = clustering_dir / "kmeans_labels_with_stats.csv"
    tsne_png       = clustering_dir / "tsne_kmeans.png"
    per_cluster_dir = clustering_dir / "cluster_distribution_plots" / "per_cluster_rows"
    overlay_dir     = clustering_dir / "cluster_distribution_plots" / "all_cluster_overlays"

    required = [embeddings_csv, summary_csv, tsne_png, per_cluster_dir, overlay_dir]
    missing = [p for p in required if not p.exists()]
    if missing:
        print(f"HTML report skipped (missing: {', '.join(p.name for p in missing)})")
        return

    # Confirm embeddings.csv has t-SNE columns
    with embeddings_csv.open(newline="", encoding="utf-8") as f:
        fieldnames = csv.DictReader(f).fieldnames or []
    if "tsne1" not in fieldnames or "tsne2" not in fieldnames:
        print("HTML report skipped (embeddings.csv has no tsne1/tsne2 columns)")
        return

    print("\nBuilding HTML cluster distribution reports...")

    id_to_cluster: dict[str, str] = {}
    color_by_cluster: dict[str, str] = {}
    if labels_csv.exists():
        id_to_cluster, color_by_cluster = _read_label_metadata(labels_csv)

    summary_rows = _read_summary_csv(summary_csv)
    points = _read_embedding_points(embeddings_csv, id_to_cluster=id_to_cluster)

    if not summary_rows:
        print(f"HTML report skipped: no cluster rows in {summary_csv}")
        return
    if not points:
        print(f"HTML report skipped: no t-SNE points parseable from {embeddings_csv}")
        return

    row_images     = _discover_row_images(per_cluster_dir)
    overlay_images = _discover_overlay_images(overlay_dir)
    if not row_images:
        print(f"HTML report skipped: no per-cluster row images in {per_cluster_dir}")
        return
    if not overlay_images:
        print(f"HTML report skipped: no overlay images in {overlay_dir}")
        return

    image_paths = [tsne_png] + [Path(i["path"]) for i in row_images + overlay_images]
    image_b64 = {str(p): _png_to_b64(p) for p in image_paths}
    tsne_b64 = image_b64[str(tsne_png)]

    report_title = title or f"{clustering_dir.name} Cluster Distribution Report"
    if best_k is not None:
        report_title = report_title.rstrip() + f"  (k={best_k})"

    online_html = _build_report_html(
        title=report_title,
        summary_rows=summary_rows,
        points=points,
        color_by_cluster=dict(color_by_cluster),
        tsne_b64=tsne_b64,
        row_images=row_images,
        overlay_images=overlay_images,
        image_b64=image_b64,
        inline_plotly_js=None,
    )

    print("Building offline variant (inlining Plotly.js)...")
    cache_dir = vendor_cache or (Path.home() / ".cache" / "cluster_distribution_report")
    inline_js = _fetch(PLOTLY_CDN, cache_dir).decode("utf-8")
    offline_html = _build_report_html(
        title=report_title,
        summary_rows=summary_rows,
        points=points,
        color_by_cluster=dict(color_by_cluster),
        tsne_b64=tsne_b64,
        row_images=row_images,
        overlay_images=overlay_images,
        image_b64=image_b64,
        inline_plotly_js=inline_js,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_online  = out_dir / "report_cluster_distribution.html"
    out_offline = out_dir / "report_cluster_distribution_offline.html"
    out_online.write_text(online_html, encoding="utf-8")
    out_offline.write_text(offline_html, encoding="utf-8")

    s1 = out_online.stat().st_size / (1024 * 1024)
    s2 = out_offline.stat().st_size / (1024 * 1024)
    print(f"✓ HTML report → {out_online} ({s1:.2f} MB)")
    print(f"✓ Offline HTML → {out_offline} ({s2:.2f} MB)")


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def _load_labels(labels_csv: Path) -> dict[str, int]:
    """Load structure_id→cluster from labels.csv or kmeans_labels_with_stats.csv.

    Accepts both 'structure_id' (approach dirs) and 'id' (clustering dirs) columns.
    """
    result: dict[str, int] = {}
    with labels_csv.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            sid = (row.get("structure_id") or row.get("id") or "").strip()
            if sid:
                result[sid] = int(row["cluster"])
    return result


def _load_medoids(medoids_csv: Path) -> dict[int, str]:
    """Load cluster_id→medoid_id.  Accepts 'medoid_id' and 'medoid_filename' columns."""
    result: dict[int, str] = {}
    with medoids_csv.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            mid = (row.get("medoid_id") or row.get("medoid_filename") or "").strip()
            result[int(row["cluster_id"])] = mid
    return result


def _load_k_sweep(k_sweep_csv: Path) -> list[dict]:
    rows: list[dict] = []
    with k_sweep_csv.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append({"k": int(row["k"]), "intra": float(row["intra"]),
                         "inter": float(row["inter"]), "ratio": float(row["ratio"]),
                         "ch_score": float(row["ch_score"]) if row.get("ch_score") else 0.0})
    return rows


def _load_structures(xyz_dir: Path, glob_pat: str = "*.xyz") -> tuple[list[Structure], dict[str, Structure]]:
    from zn_cys_his.clustering.utils import gather_structures, print_gather_report
    structs, report = gather_structures(xyz_dir, glob_pat, desc="loading XYZ")
    print_gather_report(report)
    by_id = {s.id: s for s in structs}
    return structs, by_id


# ---------------------------------------------------------------------------
# Intra/inter RMSD metrics
# ---------------------------------------------------------------------------

def compute_rmsd_metrics(
    by_id: dict[str, Structure],
    labels_csv: Path,
    medoids_csv: Path,
    w_type: dict | str = EQUAL_WEIGHTS,
    allow_reflection: bool = True,
) -> list[dict]:
    """Per-cluster RMSD metrics.

    Returns list of dicts: cluster_id, n, medoid_id, mean_intra_rmsd, inter_rmsd.
    """
    labels = _load_labels(labels_csv)
    medoid_ids = _load_medoids(medoids_csv)

    clusters: dict[int, list[str]] = {}
    for sid, cid in labels.items():
        clusters.setdefault(cid, []).append(sid)

    cluster_data: list[dict] = []
    for cid in sorted(clusters.keys()):
        members = clusters[cid]
        mid = medoid_ids.get(cid, members[0])
        if mid not in by_id:
            continue
        med_s = by_id[mid]
        rmsds = []
        for sid in tqdm.tqdm(members, desc=f"cluster {cid} intra", leave=False):
            if sid == mid or sid not in by_id:
                continue
            r, _, _ = structural_rmsd(med_s, by_id[sid], w_type, allow_reflection)
            rmsds.append(r)
        mean_intra = float(np.mean(rmsds)) if rmsds else 0.0
        cluster_data.append({"cluster_id": cid, "n": len(members),
                              "medoid_id": mid, "mean_intra_rmsd": mean_intra})

    med_structs = [(d["cluster_id"], by_id[d["medoid_id"]])
                   for d in cluster_data if d["medoid_id"] in by_id]
    inter_map: dict[int, list[float]] = {d["cluster_id"]: [] for d in cluster_data}
    for (ci, mi), (cj, mj) in combinations(med_structs, 2):
        r, _, _ = structural_rmsd(mi, mj, w_type, allow_reflection)
        inter_map[ci].append(r)
        inter_map[cj].append(r)

    for d in cluster_data:
        vals = inter_map.get(d["cluster_id"], [])
        d["inter_rmsd"] = float(np.mean(vals)) if vals else float("nan")

    return cluster_data


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_k_sweep(k_sweep_table: list[dict], out_path: Path) -> None:
    if not MPL_OK:
        return
    ks       = [r["k"] for r in k_sweep_table]
    ratio    = [r["ratio"] for r in k_sweep_table]
    ch_scores = [r["ch_score"] for r in k_sweep_table]
    best_k = k_sweep_table[max(range(len(ch_scores)), key=lambda i: ch_scores[i])]["k"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

    ax1.plot(ks, ratio, "o-", color="steelblue", linewidth=1.5)
    ax1.axvline(best_k, color="tomato", linestyle="--", label=f"best k={best_k}")
    ax1.set_ylabel("inter / intra RMSD ratio")
    ax1.set_title(f"k sweep  (best k={best_k}  by CH score)")
    ax1.legend()

    ax2.plot(ks, ch_scores, "o-", color="darkorange", linewidth=1.5)
    ax2.axvline(best_k, color="tomato", linestyle="--", label=f"best k={best_k}")
    ax2.set_xlabel("k")
    ax2.set_ylabel("CH-analogue score")
    ax2.set_title("Calinski-Harabasz analogue  (inter²/(k−1)) / (intra²/(N−k))")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_cluster_sizes(labels_csv: Path, out_path: Path, best_k: int | None = None) -> None:
    if not MPL_OK:
        return
    labels = _load_labels(labels_csv)
    counts: dict[int, int] = {}
    for cid in labels.values():
        counts[cid] = counts.get(cid, 0) + 1
    cids = sorted(counts, key=lambda c: counts[c], reverse=True)
    sizes = [counts[c] for c in cids]

    title = f"Cluster sizes  (k={best_k})" if best_k is not None else "Cluster sizes"
    fig, ax = plt.subplots(figsize=(max(6, len(cids) * 0.4), 4))
    ax.bar(range(len(cids)), sizes, color="steelblue")
    ax.set_xlabel("cluster (sorted by size)")
    ax.set_ylabel("n structures")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_rmsd_scatter(cluster_data: list[dict], out_path: Path, title: str = "") -> None:
    if not MPL_OK:
        return
    valid = [d for d in cluster_data
             if not math.isnan(d.get("inter_rmsd", float("nan")))]
    if not valid:
        return
    intra = [d["mean_intra_rmsd"] for d in valid]
    inter = [d["inter_rmsd"] for d in valid]
    cids  = [d["cluster_id"] for d in valid]

    fig, ax = plt.subplots(figsize=(5, 5))
    sc = ax.scatter(intra, inter, c=cids, cmap="viridis", alpha=0.85,
                    edgecolors="k", linewidths=0.5, s=60)
    plt.colorbar(sc, ax=ax, label="cluster id")
    ax.set_xlabel("mean intra-cluster RMSD (Å)")
    ax.set_ylabel("mean inter-cluster RMSD (Å)")
    ax.set_title(title or "Intra vs inter RMSD")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# PCA scree plot
# ---------------------------------------------------------------------------

def plot_pca_scree(approach_dir: Path, out_path: Path, best_k: int | None = None,
                   approach_name: str = "") -> None:
    if not MPL_OK:
        return
    scree_csv = approach_dir / "pca_scree.csv"
    if not scree_csv.is_file():
        print(f"  pca_scree.csv not found in {approach_dir}, skipping scree plot.")
        return

    components, evr, cumvar, retained = [], [], [], []
    with scree_csv.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            components.append(int(row["component"]))
            evr.append(float(row["explained_variance_ratio"]))
            cumvar.append(float(row["cumulative_variance_ratio"]))
            retained.append(row["retained"].strip().lower() == "true")

    if not components:
        return

    n_total    = len(components)
    n_retained = sum(retained)
    var_threshold = cumvar[n_retained - 1] if n_retained > 0 else cumvar[-1]

    colors = ["steelblue" if r else "#cccccc" for r in retained]

    fig, ax1 = plt.subplots(figsize=(max(6, n_total * 0.45 + 1.5), 4))
    ax1.bar(components, [v * 100 for v in evr], color=colors, edgecolor="none")
    ax1.set_xlabel("principal component")
    ax1.set_ylabel("explained variance (%)", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax1.set_xticks(components)

    ax2 = ax1.twinx()
    ax2.plot(components, [v * 100 for v in cumvar], "o-", color="tomato",
             linewidth=1.5, markersize=4)
    ax2.axhline(var_threshold * 100, color="tomato", linestyle=":", linewidth=1, alpha=0.6)
    ax2.set_ylabel("cumulative explained variance (%)", color="tomato")
    ax2.tick_params(axis="y", labelcolor="tomato")
    ax2.set_ylim(0, 105)

    if n_retained < n_total:
        ax1.axvline(n_retained + 0.5, color="dimgray", linestyle="--", linewidth=1)

    label = approach_name or approach_dir.name
    k_str = f"  k={best_k}" if best_k is not None else ""
    ax1.set_title(
        f"PCA scree — {label}{k_str}\n"
        f"{n_retained} / {n_total} components retained  "
        f"({var_threshold * 100:.1f}% variance threshold)"
    )

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [
        Patch(color="steelblue", label="retained dims"),
        Patch(color="#cccccc", label="dropped dims"),
        Line2D([0], [0], color="tomato", marker="o", markersize=4, label="cumulative EVR"),
    ]
    ax1.legend(handles=handles, loc="center right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# PCA → XYZ reconstruction (Approach 1 only)
# ---------------------------------------------------------------------------

def pca_to_xyz(
    aligned_xyz_dir: Path,
    out_dir: Path,
    n_pcs: int = 3,
    epsilon: float = 1.0,
    var_floor: float = 1e-3,
) -> None:
    from zn_cys_his.clustering.utils import write_structure_xyz  # type: ignore

    structs = [s for p in sorted(aligned_xyz_dir.glob("*.xyz"))
               if (s := parse_structure(p)) is not None]
    if not structs:
        print("pca_to_xyz: no structures found.")
        return

    X = np.array([s.heavy().ravel() for s in structs])
    means = X.mean(0)
    stds  = np.maximum(X.std(0), var_floor)
    Xs    = (X - means) / stds

    pca_full = PCA()
    pca_full.fit(Xs)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_comp = min(max(1, int(np.searchsorted(cumvar, 0.95)) + 1), Xs.shape[1])
    pca = PCA(n_components=n_comp)
    pca.fit(Xs)

    out_dir.mkdir(parents=True, exist_ok=True)

    from zn_cys_his.clustering.utils import structure_like
    template = structs[0]
    M = template.n_atoms()

    def _vec_to_structure(feat_vec: np.ndarray, sid: str) -> Structure:
        raw = feat_vec * stds + means
        return structure_like(template, raw.reshape(M, 3), sid)

    write_structure_xyz(_vec_to_structure(np.zeros(M * 3), "mean"), out_dir / "mean.xyz")

    for i in range(min(n_pcs, n_comp)):
        pc_vec = pca.components_[i] * np.sqrt(pca.explained_variance_[i])
        for sign, label in [(+1, "plus"), (-1, "minus")]:
            feat = sign * epsilon * pc_vec
            sid = f"pc{i+1}_{label}"
            write_structure_xyz(_vec_to_structure(feat, sid), out_dir / f"{sid}.xyz")

    explained = pca.explained_variance_ratio_[:n_pcs]
    print(f"pca_to_xyz: wrote mean + PC1..{min(n_pcs,n_comp)} "
          f"(var: {', '.join(f'{v*100:.1f}%' for v in explained)})")

    return explained


# ---------------------------------------------------------------------------
# PCA morph GIF (Approach 1 only)
# ---------------------------------------------------------------------------

# Per-element visual styling — replaces the old fixed 13-atom colour lists so
# any Cys/His composition renders correctly.
_ELEM_COLOR      = {"ZN": "tab:orange", "S": "gold", "N": "steelblue", "C": "silver"}
_ELEM_SIZE       = {"ZN": 300, "S": 150, "N": 150, "C": 70}
_ELEM_COLOR_HTML = {"ZN": "orange", "S": "gold", "N": "#4f7fb8", "C": "#b8b8b8"}
_ELEM_SIZE_HTML  = {"ZN": 7, "S": 5, "N": 5, "C": 4}


def _atom_colors_sizes(template) -> tuple:
    """(colors, sizes) lists (length M) for matplotlib, keyed by element."""
    meta = template.atom_meta()
    colors = [_ELEM_COLOR.get(e, "silver") for _, _, e, _ in meta]
    sizes  = [_ELEM_SIZE.get(e, 70) for _, _, e, _ in meta]
    return colors, sizes


def _html_atom_groups(template) -> list:
    """[(element, [heavy-idx...], html_color, marker_size), ...] for plotly."""
    from collections import OrderedDict
    groups: "OrderedDict[str, list]" = OrderedDict()
    for i, (_, _, e, _) in enumerate(template.atom_meta()):
        groups.setdefault(e, []).append(i)
    return [(e, idxs, _ELEM_COLOR_HTML.get(e, "#b8b8b8"), _ELEM_SIZE_HTML.get(e, 4))
            for e, idxs in groups.items()]


def _draw_structure(ax, coords: np.ndarray, title: str, lims: dict,
                    bonds: list, colors: list, sizes: list) -> None:
    ax.cla()
    ax.set_xlim(*lims["x"]); ax.set_ylim(*lims["y"]); ax.set_zlim(*lims["z"])
    ax.set_title(title, fontsize=9, pad=4)
    ax.set_xlabel("x (Å)", fontsize=7, labelpad=1)
    ax.set_ylabel("y (Å)", fontsize=7, labelpad=1)
    ax.set_zlabel("z (Å)", fontsize=7, labelpad=1)
    ax.tick_params(labelsize=6)
    for a, b in bonds:
        ax.plot([coords[a, 0], coords[b, 0]],
                [coords[a, 1], coords[b, 1]],
                [coords[a, 2], coords[b, 2]],
                color="dimgray", linewidth=1.2, alpha=0.7)
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
               c=colors, s=sizes,
               edgecolors="k", linewidths=0.3, depthshade=True, zorder=5)


def plot_pc_morph_gif(
    pca_xyz_dir: Path,
    out_dir: Path,
    explained: np.ndarray | None = None,
    n_frames: int = 48,
    fps: int = 12,
    elev: float = 20.0,
    azim: float = -60.0,
) -> None:
    if not MPL_OK:
        return
    try:
        from matplotlib.animation import FuncAnimation, PillowWriter
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError:
        print("plot_pc_morph_gif: mpl_toolkits not available, skipping.")
        return

    from zn_cys_his.clustering.utils import structure_bonds
    pc_data: list[tuple[int, np.ndarray, np.ndarray]] = []
    template = None
    for i in range(1, 10):
        minus_p = pca_xyz_dir / f"pc{i}_minus.xyz"
        plus_p  = pca_xyz_dir / f"pc{i}_plus.xyz"
        if not (minus_p.is_file() and plus_p.is_file()):
            break
        m = parse_structure(minus_p)
        p = parse_structure(plus_p)
        if m is not None and p is not None:
            if template is None:
                template = m
            pc_data.append((i, m.heavy(), p.heavy()))

    if not pc_data:
        print("plot_pc_morph_gif: no PC files found.")
        return

    bonds = structure_bonds(template)
    colors, sizes = _atom_colors_sizes(template)

    n_pcs = len(pc_data)
    all_coords = np.vstack([c for _, mn, pl in pc_data for c in [mn, pl]])
    pad = 0.8
    lims = {
        "x": (all_coords[:, 0].min() - pad, all_coords[:, 0].max() + pad),
        "y": (all_coords[:, 1].min() - pad, all_coords[:, 1].max() + pad),
        "z": (all_coords[:, 2].min() - pad, all_coords[:, 2].max() + pad),
    }

    fig = plt.figure(figsize=(4.5 * n_pcs, 4.5))
    axes = [fig.add_subplot(1, n_pcs, j + 1, projection="3d") for j in range(n_pcs)]
    for ax in axes:
        ax.view_init(elev=elev, azim=azim)

    half = n_frames // 2
    ts = np.concatenate([np.linspace(0, 1, half, endpoint=False),
                         np.linspace(1, 0, half, endpoint=False)])

    def update(frame_idx: int):
        t = ts[frame_idx]
        for j, (pc_i, minus, plus) in enumerate(pc_data):
            coords = (1.0 - t) * minus + t * plus
            var_str = f" ({explained[pc_i-1]*100:.1f}%)" if explained is not None and pc_i - 1 < len(explained) else ""
            label = "−" if t < 0.5 else "+"
            _draw_structure(axes[j], coords, f"PC{pc_i}{var_str}  {label}", lims,
                            bonds, colors, sizes)
            axes[j].view_init(elev=elev, azim=azim)
        return []

    anim = FuncAnimation(fig, update, frames=len(ts), interval=1000 // fps, blit=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pc_morph.gif"
    anim.save(str(out_path), writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"PC morph GIF → {out_path}")


# ---------------------------------------------------------------------------
# PCA displacement-arrow plot (Approach 1 only)
# ---------------------------------------------------------------------------

_PC_COLORS_HEX = ["#e05252", "#4caf7d", "#5b8fd4"]


def plot_pc_arrows_html(
    aligned_xyz_dir: Path,
    out_dir: Path,
    n_pcs: int = 3,
    var_floor: float = 1e-3,
    base_scale: float = 2.5,
) -> None:
    try:
        import plotly.graph_objects as go
        from sklearn.decomposition import PCA
    except ImportError:
        print("plot_pc_arrows_html: plotly or sklearn not available, skipping.")
        return

    from zn_cys_his.clustering.utils import structure_bonds
    structs = [s for p in sorted(aligned_xyz_dir.glob("*.xyz"))
               if (s := parse_structure(p)) is not None]
    if not structs:
        return

    template = structs[0]
    M = template.n_atoms()
    X      = np.array([s.heavy().ravel() for s in structs])
    means  = X.mean(0)
    stds   = np.maximum(X.std(0), var_floor)
    Xs     = (X - means) / stds

    pca_full = PCA().fit(Xs)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_comp = min(max(1, int(np.searchsorted(cumvar, 0.95)) + 1), Xs.shape[1])
    pca    = PCA(n_components=n_comp).fit(Xs)

    n_pcs       = min(n_pcs, n_comp)
    mean_coords = means.reshape(M, 3)
    max_var     = pca.explained_variance_ratio_[0]

    pc_vecs: list[np.ndarray] = []
    for i in range(n_pcs):
        raw      = (pca.components_[i] * stds).reshape(M, 3)
        var_pct  = pca.explained_variance_ratio_[i] / max_var
        pc_vecs.append(raw / np.linalg.norm(raw) * base_scale * var_pct)

    bonds = structure_bonds(template)
    traces: list[go.BaseTraceType] = []

    bx, by, bz = [], [], []
    for a, b in bonds:
        bx += [mean_coords[a, 0], mean_coords[b, 0], None]
        by += [mean_coords[a, 1], mean_coords[b, 1], None]
        bz += [mean_coords[a, 2], mean_coords[b, 2], None]
    traces.append(go.Scatter3d(
        x=bx, y=by, z=bz, mode="lines",
        line=dict(color="dimgray", width=4),
        name="bonds", showlegend=False, hoverinfo="skip",
    ))

    for label, idxs, color, size in _html_atom_groups(template):
        pts = mean_coords[idxs]
        traces.append(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode="markers",
            marker=dict(size=size, color=color, line=dict(color="black", width=1)),
            name=label, legendgroup="structure",
        ))

    for i, (pc_v, hex_col) in enumerate(zip(pc_vecs, _PC_COLORS_HEX[:n_pcs])):
        var_pct = pca.explained_variance_ratio_[i] * 100
        legend_label = f"PC{i+1} ({var_pct:.1f}%)"
        for sign in (1, -1):
            disp = sign * pc_v
            tips = mean_coords + disp
            sx, sy, sz = [], [], []
            for j in range(M):
                sx += [mean_coords[j, 0], tips[j, 0], None]
                sy += [mean_coords[j, 1], tips[j, 1], None]
                sz += [mean_coords[j, 2], tips[j, 2], None]
            traces.append(go.Scatter3d(
                x=sx, y=sy, z=sz, mode="lines",
                line=dict(color=hex_col, width=3),
                name=legend_label if sign == 1 else None,
                legendgroup=f"pc{i+1}",
                showlegend=(sign == 1),
                hoverinfo="skip",
            ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        title="Mean structure + PC displacement vectors",
        scene=dict(xaxis_title="x (Å)", yaxis_title="y (Å)", zaxis_title="z (Å)", aspectmode="data"),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pc_arrows.html"
    fig.write_html(str(out_path), include_plotlyjs=True)
    print(f"PC arrows HTML → {out_path}")


# ---------------------------------------------------------------------------
# Heavy-atom position cloud (all approaches)
# ---------------------------------------------------------------------------

def plot_atom_cloud_html(
    structs: list,
    out_dir: Path,
    labels_csv: Path | None = None,
    approach_name: str = "",
    best_k: int | None = None,
) -> None:
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plot_atom_cloud_html: plotly not available, skipping.")
        return

    if not structs:
        return

    labels: dict[str, int] = {}
    if labels_csv is not None and labels_csv.is_file():
        labels = _load_labels(labels_csv)

    from zn_cys_his.clustering.utils import structure_bonds
    template = structs[0]
    all_coords = np.stack([s.heavy() for s in structs])
    mean_coords = all_coords.mean(0)

    # Element groups → (label, idxs, color, marker size, opacity)
    _opacity = {"ZN": 0.85, "S": 0.55, "N": 0.55, "C": 0.45}
    atom_groups = [(e, idxs, color, size, _opacity.get(e, 0.45))
                   for e, idxs, color, size in _html_atom_groups(template)]

    traces: list = []
    for atom_label, idxs, color, size, opacity in atom_groups:
        xs, ys, zs, hover = [], [], [], []
        for s in structs:
            coords = s.heavy()
            cluster_str = f"  cluster {labels[s.id]}" if s.id in labels else ""
            for idx in idxs:
                xs.append(float(coords[idx, 0]))
                ys.append(float(coords[idx, 1]))
                zs.append(float(coords[idx, 2]))
                hover.append(f"{s.id}{cluster_str}")
        traces.append(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="markers",
            marker=dict(size=size, color=color, opacity=opacity, line=dict(width=0)),
            name=atom_label,
            text=hover,
            hovertemplate="%{text}<extra></extra>",
        ))

    bx, by, bz = [], [], []
    for a, b in structure_bonds(template):
        bx += [mean_coords[a, 0], mean_coords[b, 0], None]
        by += [mean_coords[a, 1], mean_coords[b, 1], None]
        bz += [mean_coords[a, 2], mean_coords[b, 2], None]
    traces.insert(0, go.Scatter3d(
        x=bx, y=by, z=bz, mode="lines",
        line=dict(color="dimgray", width=5),
        name="mean bonds", showlegend=True, hoverinfo="skip",
    ))

    title = "Heavy-atom position cloud"
    if approach_name:
        title += f"  —  {approach_name}"
    if best_k is not None:
        title += f"  k={best_k}"
    title += f"  ({len(structs)} structures)"

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="x (Å)", yaxis_title="y (Å)", zaxis_title="z (Å)", aspectmode="data"),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=50, b=0),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"atom_cloud_k{best_k}.html" if best_k is not None else "atom_cloud.html"
    out_path = out_dir / fname
    fig.write_html(str(out_path), include_plotlyjs=True)
    print(f"Atom cloud HTML → {out_path}")


def plot_atom_cloud_by_cluster_html(
    structs: list,
    out_dir: Path,
    labels_csv: Path | None = None,
    approach_name: str = "",
    best_k: int | None = None,
) -> None:
    """Like plot_atom_cloud_html but all atoms colored by cluster rather than atom type."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plot_atom_cloud_by_cluster_html: plotly not available, skipping.")
        return

    if not structs:
        return

    labels: dict[str, int] = {}
    if labels_csv is not None and labels_csv.is_file():
        labels = _load_labels(labels_csv)

    from zn_cys_his.clustering.utils import structure_bonds
    template = structs[0]
    all_coords = np.stack([s.heavy() for s in structs])
    mean_coords = all_coords.mean(0)

    _PALETTE = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
        "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    ]

    # (element label, indices into heavy(), marker size), derived from composition
    atom_meta = [(e, idxs, size) for e, idxs, _c, size in _html_atom_groups(template)]

    unique_clusters = sorted(set(labels.get(s.id, -1) for s in structs))
    cluster_color = {c: _PALETTE[i % len(_PALETTE)] for i, c in enumerate(unique_clusters)}

    traces: list = []
    for cluster_id in unique_clusters:
        cluster_structs = [s for s in structs if labels.get(s.id, -1) == cluster_id]
        xs, ys, zs, sizes, hover = [], [], [], [], []
        for s in cluster_structs:
            coords = s.heavy()
            for atom_name, idxs, sz in atom_meta:
                for idx in idxs:
                    xs.append(float(coords[idx, 0]))
                    ys.append(float(coords[idx, 1]))
                    zs.append(float(coords[idx, 2]))
                    sizes.append(sz)
                    hover.append(f"{s.id}  [{atom_name}]")
        label = f"cluster {cluster_id}" if cluster_id >= 0 else "unlabeled"
        traces.append(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="markers",
            marker=dict(size=sizes, color=cluster_color[cluster_id], opacity=0.55,
                        line=dict(width=0)),
            name=label,
            text=hover,
            hovertemplate="%{text}<extra></extra>",
        ))

    bx, by, bz = [], [], []
    for a, b in structure_bonds(template):
        bx += [mean_coords[a, 0], mean_coords[b, 0], None]
        by += [mean_coords[a, 1], mean_coords[b, 1], None]
        bz += [mean_coords[a, 2], mean_coords[b, 2], None]
    traces.insert(0, go.Scatter3d(
        x=bx, y=by, z=bz, mode="lines",
        line=dict(color="dimgray", width=5),
        name="mean bonds", showlegend=True, hoverinfo="skip",
    ))

    title = "Heavy-atom position cloud (by cluster)"
    if approach_name:
        title += f"  —  {approach_name}"
    if best_k is not None:
        title += f"  k={best_k}"
    title += f"  ({len(structs)} structures)"

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        scene=dict(xaxis_title="x (Å)", yaxis_title="y (Å)", zaxis_title="z (Å)", aspectmode="data"),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=50, b=0),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"atom_cloud_by_cluster_k{best_k}.html" if best_k is not None else "atom_cloud_by_cluster.html"
    out_path = out_dir / fname
    fig.write_html(str(out_path), include_plotlyjs=True)
    print(f"Atom cloud by cluster HTML → {out_path}")


# ---------------------------------------------------------------------------
# Permuted-residue count (requires aligned XYZ with real RESSEQ)
# ---------------------------------------------------------------------------

def _read_resseqs_in_file_order(path: Path) -> list[int]:
    """Return the 4 S-atom RESSEQ values in the order they appear in the file."""
    result = []
    for line in path.read_text().splitlines()[2:]:
        if "#" not in line:
            continue
        comment = line.split("#", 1)[1]
        fields = {}
        for token in comment.split():
            if "=" in token:
                k, _, v = token.partition("=")
                fields[k.upper()] = v
        if fields.get("ATOM", "").upper() in ("SG", "S"):
            try:
                result.append(int(fields["RESSEQ"]))
            except (KeyError, ValueError):
                pass
    return result


def count_permuted_residues(aligned_xyz_dir: Path) -> tuple[int, int]:
    """Count aligned XYZ files where residues were reordered (RESSEQ not ascending).

    Returns (n_permuted, n_total).
    """
    total = permuted = 0
    for p in sorted(aligned_xyz_dir.glob("*.xyz")):
        resseqs = _read_resseqs_in_file_order(p)
        if len(resseqs) == 4:
            total += 1
            if resseqs != sorted(resseqs):
                permuted += 1
    return permuted, total


# ---------------------------------------------------------------------------
# t-SNE overlay plots
# ---------------------------------------------------------------------------

def _load_embeddings(embeddings_csv: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return (ids, tsne_coords, cluster_labels) from embeddings.csv.

    cluster_labels are loaded from the 'cluster' column if present, else zeros.
    """
    ids, tsne, clusters = [], [], []
    with embeddings_csv.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            ids.append(row["id"])
            tsne.append([float(row["tsne1"]), float(row["tsne2"])])
            clusters.append(int(row.get("cluster", 0)))
    return ids, np.array(tsne), np.array(clusters)


def _merge_cluster_labels_into_embeddings(
    ids: list[str],
    clusters_arr: np.ndarray,
    labels_csv: Path,
) -> np.ndarray:
    """Override cluster array with labels from labels.csv / kmeans_labels_with_stats.csv."""
    label_map: dict[str, int] = {}
    with labels_csv.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            sid = (row.get("structure_id") or row.get("id") or "").strip()
            cl = row.get("cluster")
            if sid and cl is not None:
                try:
                    label_map[sid] = int(cl)
                except ValueError:
                    pass
    return np.array([label_map.get(sid, int(clusters_arr[i])) for i, sid in enumerate(ids)])


def plot_tsne_sampled_highlighted(
    embeddings_csv: Path,
    labels_csv: Path,
    sampled_dir: Path,
    out_path: Path,
) -> None:
    """t-SNE scatter colored by cluster; black open circles mark the sampled val files."""
    if not MPL_OK:
        print("plot_tsne_sampled_highlighted: matplotlib not available, skipping.")
        return

    ids, X_tsne, raw_clusters = _load_embeddings(embeddings_csv)
    clusters = _merge_cluster_labels_into_embeddings(ids, raw_clusters, labels_csv)

    sampled_stems = {p.stem for p in sampled_dir.glob("*.xyz")}
    sampled_mask = np.array([sid in sampled_stems for sid in ids])

    unique_labels = sorted(set(int(c) for c in clusters))
    cmap = plt.get_cmap("tab20")

    fig, ax = plt.subplots(figsize=(9, 7))
    for i, c in enumerate(unique_labels):
        mask = clusters == c
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                   c=[cmap(i % 20)], s=15, alpha=0.7, label=str(c))

    if sampled_mask.any():
        ax.scatter(X_tsne[sampled_mask, 0], X_tsne[sampled_mask, 1],
                   facecolors="none", edgecolors="black", s=60, linewidths=1.5,
                   zorder=5, label=f"sampled (n={sampled_mask.sum()})")

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(f"t-SNE  (k={len(unique_labels)})  — sampled val structures circled")
    ncol = max(1, (len(unique_labels) + 1) // 20)
    ax.legend(title="cluster", bbox_to_anchor=(1.02, 1), loc="upper left",
              fontsize=7, ncol=ncol, markerscale=1.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"t-SNE sampled overlay → {out_path}")


def plot_tsne_qtetra_highlighted(
    embeddings_csv: Path,
    labels_stats_csv: Path,
    out_path: Path,
    threshold: float = 0.8,
) -> None:
    """t-SNE scatter colored by cluster; black filled dots mark q_tetra_coord < threshold."""
    if not MPL_OK:
        print("plot_tsne_qtetra_highlighted: matplotlib not available, skipping.")
        return

    ids, X_tsne, raw_clusters = _load_embeddings(embeddings_csv)
    clusters = _merge_cluster_labels_into_embeddings(ids, raw_clusters, labels_stats_csv)

    qtetra_map: dict[str, float] = {}
    with labels_stats_csv.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            sid = (row.get("id") or "").strip()
            v = row.get("q_tetra_coord", "").strip()
            if sid and v:
                try:
                    qtetra_map[sid] = float(v)
                except ValueError:
                    pass

    low_mask = np.array([
        sid in qtetra_map and qtetra_map[sid] < threshold
        for sid in ids
    ])

    unique_labels = sorted(set(int(c) for c in clusters))
    cmap = plt.get_cmap("tab20")

    fig, ax = plt.subplots(figsize=(9, 7))
    for i, c in enumerate(unique_labels):
        mask = clusters == c
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                   c=[cmap(i % 20)], s=15, alpha=0.7, label=str(c))

    if low_mask.any():
        ax.scatter(X_tsne[low_mask, 0], X_tsne[low_mask, 1],
                   c="black", s=20, zorder=5,
                   label=fr"$q_{{tetra}}<{threshold}$ (n={low_mask.sum()})")

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(
        f"t-SNE  (k={len(unique_labels)})"
        fr"  — $q_{{tetra}}<{threshold}$ highlighted"
    )
    ncol = max(1, (len(unique_labels) + 1) // 20)
    ax.legend(title="cluster", bbox_to_anchor=(1.02, 1), loc="upper left",
              fontsize=7, ncol=ncol, markerscale=1.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"t-SNE q_tetra overlay → {out_path}")


# ---------------------------------------------------------------------------
# Cross-approach comparison
# ---------------------------------------------------------------------------

def compare_approaches(
    approach_dirs: list[Path],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for d in approach_dirs:
        ksweep = d / "k_sweep.csv"
        if not ksweep.is_file():
            print(f"  skip {d.name}: no k_sweep.csv")
            continue
        table = _load_k_sweep(ksweep)
        best = max(table, key=lambda r: r["ch_score"])
        rows.append({"approach": d.name, "best_k": best["k"],
                     "intra": best["intra"], "inter": best["inter"],
                     "ratio": best["ratio"], "ch_score": best["ch_score"]})

    if not rows:
        print("No comparison data found.")
        return

    table_path = out_dir / "comparison_table.csv"
    with table_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["approach", "best_k", "intra", "inter", "ratio", "ch_score"])
        w.writeheader()
        w.writerows(rows)
    print(f"Comparison table → {table_path}")

    print("\n## Approach comparison\n")
    print("| Approach | Best k | intra (Å) | inter (Å) | ratio | CH score |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['approach']} | {r['best_k']} | {r['intra']:.4f} | {r['inter']:.4f} | {r['ratio']:.4f} | {r['ch_score']:.4f} |")

    if MPL_OK:
        names = [r["approach"] for r in rows]
        ch_scores = [r["ch_score"] for r in rows]
        ratios = [r["ratio"] for r in rows]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(8, len(names) * 3), 4))
        bars1 = ax1.bar(names, ch_scores, color="darkorange")
        ax1.bar_label(bars1, fmt="%.2f", padding=3)
        ax1.set_ylabel("CH-analogue score")
        ax1.set_title("Approach comparison — CH score")
        bars2 = ax2.bar(names, ratios, color="steelblue")
        ax2.bar_label(bars2, fmt="%.2f", padding=3)
        ax2.set_ylabel("inter / intra RMSD ratio")
        ax2.set_title("Approach comparison — ratio")
        fig.tight_layout()
        plot_path = out_dir / "comparison_plot.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"Comparison plot → {plot_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cluster validation: RMSD metrics, plots, PCA→XYZ, and HTML distribution reports."
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--approach-dir", type=Path,
                      help="Output dir from one approach script.  Accepts approach-style "
                           "dirs (labels.csv, k_sweep.csv) and clustering-style dirs "
                           "(kmeans_labels_with_stats.csv, embeddings.csv with tsne columns).")
    mode.add_argument("--compare-dirs", type=Path, nargs="+",
                      help="Multiple approach output dirs for cross-approach comparison.")

    parser.add_argument("--xyz-dir", type=Path,
                        help="XYZ dir for RMSD computation (aligned_xyz for approach 1).")
    parser.add_argument("--glob", type=str, default="*.xyz",
                        help="Filename pattern within --xyz-dir (default: *.xyz). "
                             "Use '*_Zn-extended.xyz' for raw His dirs holding both variants.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--approach1", action="store_true",
                        help="Enable PCA→XYZ reconstruction (requires --xyz-dir with aligned coords).")
    parser.add_argument("--no-reflection", action="store_true")
    parser.add_argument("--weight-scheme", choices=["equal", "shell", "distance"],
                        default="distance",
                        help="RMSD atom weighting: equal, shell (coord atom=1, other "
                             "arm atoms=0.5), or distance (1/avg_Zn_distance per atom; default).")

    # HTML report options
    parser.add_argument("--clustering-dir", type=Path, default=None,
                        help="Directory containing HTML report inputs (embeddings.csv, "
                             "kmeans_cluster_stats_summary.csv, tsne_kmeans.png, "
                             "cluster_distribution_plots/).  Defaults to --approach-dir.")
    parser.add_argument("--title", default="",
                        help="Title for the HTML report (default: derived from dir name).")
    parser.add_argument("--vendor-cache", type=Path,
                        default=Path.home() / ".cache" / "cluster_distribution_report",
                        help="Cache directory for downloading Plotly.js for offline builds.")
    parser.add_argument("--no-report", action="store_true",
                        help="Skip HTML report generation.")
    parser.add_argument("--sampled-val-dir", type=Path, default=None,
                        help="Directory of sampled XYZ files for validation (used to overlay "
                             "black circles on the t-SNE plot).")

    args = parser.parse_args()

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Cross-approach comparison mode ----
    if args.compare_dirs:
        dirs = [d.expanduser().resolve() for d in args.compare_dirs]
        compare_approaches(dirs, out_dir)
        if not args.no_report:
            for d in dirs:
                build_cluster_reports(
                    clustering_dir=d,
                    out_dir=d,
                    title=args.title or f"{d.name} Cluster Distribution Report",
                    vendor_cache=args.vendor_cache,
                )
        return 0

    # ---- Single approach mode ----
    approach_dir = args.approach_dir.expanduser().resolve()

    # Resolve labels CSV — prefer labels.csv, fall back to kmeans_labels_with_stats.csv
    labels_csv  = approach_dir / "labels.csv"
    if not labels_csv.is_file():
        labels_csv = approach_dir / "kmeans_labels_with_stats.csv"

    medoids_csv = approach_dir / "medoids.csv"
    k_sweep_csv = approach_dir / "k_sweep.csv"

    has_labels  = labels_csv.is_file()
    has_medoids = medoids_csv.is_file()
    best_k: int | None = None

    if has_labels:
        if k_sweep_csv.is_file():
            k_sweep_table = _load_k_sweep(k_sweep_csv)
            best_k = k_sweep_table[max(range(len(k_sweep_table)),
                                        key=lambda i: k_sweep_table[i]["ch_score"])]["k"]
            plot_k_sweep(k_sweep_table, out_dir / "k_sweep_plot.png")
            print(f"k-sweep plot → {out_dir / 'k_sweep_plot.png'}  (best k={best_k})")
        else:
            labels_data = _load_labels(labels_csv)
            best_k = len(set(labels_data.values()))
            print(f"(k_sweep.csv not found; inferred k={best_k} from cluster labels)")

        plot_pca_scree(approach_dir, out_dir / f"pca_scree_k{best_k}.png",
                       best_k=best_k, approach_name=approach_dir.name)
        print(f"PCA scree → {out_dir / f'pca_scree_k{best_k}.png'}")

        plot_cluster_sizes(labels_csv, out_dir / f"cluster_sizes_k{best_k}.png", best_k=best_k)
        print(f"Cluster sizes → {out_dir / f'cluster_sizes_k{best_k}.png'}")
    else:
        print(f"(No labels CSV found in {approach_dir}; skipping k-sweep/size/RMSD validation)")

    # RMSD metrics (requires --xyz-dir and both labels + medoids)
    if args.xyz_dir and has_labels and has_medoids:
        xyz_dir = args.xyz_dir.expanduser().resolve()
        if not xyz_dir.is_dir():
            raise SystemExit(f"--xyz-dir not found: {xyz_dir}")

        print("Loading structures for RMSD computation …")
        structs, by_id = _load_structures(xyz_dir, args.glob)

        print("Rendering atom cloud …")
        plot_atom_cloud_html(structs, out_dir, labels_csv=labels_csv,
                             approach_name=approach_dir.name, best_k=best_k)
        plot_atom_cloud_by_cluster_html(structs, out_dir, labels_csv=labels_csv,
                                        approach_name=approach_dir.name, best_k=best_k)

        print("Computing per-cluster RMSD metrics …")
        _w_map = {"equal": EQUAL_WEIGHTS, "shell": SHELL_WEIGHTS, "distance": DISTANCE_WEIGHTS}
        cluster_data = compute_rmsd_metrics(
            by_id, labels_csv, medoids_csv,
            w_type=_w_map[args.weight_scheme],
            allow_reflection=not args.no_reflection,
        )

        rmsd_table_path = out_dir / f"rmsd_table_k{best_k}.csv"
        with rmsd_table_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["cluster_id", "n", "medoid_id",
                                                "mean_intra_rmsd", "inter_rmsd"])
            w.writeheader()
            w.writerows(cluster_data)
        print(f"RMSD table → {rmsd_table_path}")

        plot_rmsd_scatter(cluster_data, out_dir / f"rmsd_scatter_k{best_k}.png",
                          title=f"Intra vs inter RMSD  (k={best_k})")
        print(f"RMSD scatter → {out_dir / f'rmsd_scatter_k{best_k}.png'}")

        # Count permuted residues (meaningful only when xyz_dir contains aligned files)
        print("Counting permuted residues …")
        n_perm, n_total = count_permuted_residues(xyz_dir)
        perm_line = f"permuted={n_perm}/{n_total} ({100*n_perm/n_total:.1f}% had residues reordered)" if n_total else "permuted=0/0"
        print(f"  {perm_line}")
        (out_dir / "permuted_residues.txt").write_text(perm_line + "\n", encoding="utf-8")

        if args.approach1:
            pca_xyz_dir = out_dir / "pca_to_xyz"
            print(f"PCA → XYZ reconstruction → {pca_xyz_dir}")
            explained = pca_to_xyz(xyz_dir, pca_xyz_dir)
            print("Rendering PC morph GIF …")
            plot_pc_morph_gif(pca_xyz_dir, pca_xyz_dir, explained=explained)
            print("Rendering PC displacement arrows …")
            plot_pc_arrows_html(xyz_dir, pca_xyz_dir)
    elif args.xyz_dir:
        print("(Skipping RMSD metrics: --xyz-dir provided but labels/medoids not found)")
    else:
        print("(Skipping RMSD metrics and PCA→XYZ: no --xyz-dir provided)")

    # t-SNE overlay plots (need embeddings.csv + labels in approach_dir)
    embeddings_csv = approach_dir / "embeddings.csv"
    labels_stats_csv = approach_dir / "kmeans_labels_with_stats.csv"
    labels_for_overlay = labels_csv if labels_csv.is_file() else labels_stats_csv

    if embeddings_csv.is_file() and labels_for_overlay.is_file():
        if args.sampled_val_dir is not None:
            sampled_dir = args.sampled_val_dir.expanduser().resolve()
            if sampled_dir.is_dir():
                plot_tsne_sampled_highlighted(
                    embeddings_csv, labels_for_overlay, sampled_dir,
                    out_dir / "tsne_sampled_highlighted.png",
                )
            else:
                print(f"(Skipping sampled overlay: --sampled-val-dir not found: {sampled_dir})")

        if labels_stats_csv.is_file():
            plot_tsne_qtetra_highlighted(
                embeddings_csv, labels_stats_csv,
                out_dir / "tsne_qtetra_lt0p8.png",
            )
    else:
        print("(Skipping t-SNE overlays: embeddings.csv or labels CSV not found in approach-dir)")

    # HTML distribution report
    if not args.no_report:
        clustering_dir = (args.clustering_dir or approach_dir).expanduser().resolve()
        build_cluster_reports(
            clustering_dir=clustering_dir,
            out_dir=out_dir,
            title=args.title or f"{approach_dir.name} Cluster Distribution Report",
            vendor_cache=args.vendor_cache,
            best_k=best_k,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
