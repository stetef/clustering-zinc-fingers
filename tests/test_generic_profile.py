"""Tests for the generic / heme structure profiles (new clustering path).

Synthesises a tiny 'heme-like' dataset in a tmp dir (a central FE, four pyrrole
nitrogens, ring carbons, two axial ligands) under random rigid motions, with one
file missing an atom, and checks:

  * generic gather canonicalises every file onto one shared atom template and
    flags the missing atom as not-present;
  * the heme matcher discovers a 2-element symmetry group whose non-identity
    member swaps the two axial ligands;
  * approach 1 runs end-to-end on the generic path and labels every structure.

The Zn/Cys/His path is covered by the existing smoke tests and is unchanged
here (its functions take ``matcher=None`` by default).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from zn_cys_his.clustering.profiles import get_profile

_TEMPLATE = (
    [("FE", "Fe", (0.0, 0.0, 0.0))]
    + [(f"N{c}", "N", (x, y, 0.0))
       for c, (x, y) in zip("ABCD", [(2, 0), (0, 2), (-2, 0), (0, -2)])]
    + [(f"C{i}", "C", (x, y, 0.0))
       for i, (x, y) in enumerate([(3, 3), (-3, 3), (-3, -3), (3, -3)])]
)


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    q = rng.normal(size=3)
    q /= np.linalg.norm(q)
    th = rng.uniform(0, 2 * np.pi)
    K = np.array([[0, -q[2], q[1]], [q[2], 0, -q[0]], [-q[1], q[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def _make_dataset(root: Path) -> None:
    rng = np.random.default_rng(0)
    root.mkdir(parents=True, exist_ok=True)
    idx = 0
    for fam, (top, bot) in enumerate([(1.8, 2.0), (2.2, 2.0), (2.0, 3.0)]):
        for _ in range(8):
            atoms = [(n, e, np.array(p, float)) for n, e, p in _TEMPLATE]
            atoms.append(("NAX1", "N", np.array([0.0, 0.0, top])))
            atoms.append(("NAX2", "N", np.array([0.0, 0.0, -bot])))
            coords = np.array([a[2] for a in atoms]) + rng.normal(0, 0.05, (len(atoms), 3))
            coords = coords @ _random_rotation(rng).T + rng.normal(0, 3, 3)
            drop = (idx == 5)  # one file missing an atom
            p = root / f"fam{fam}_{idx:03d}.xyz"
            with p.open("w") as fh:
                rows = [(n, e, c) for (n, e, _), c in zip(atoms, coords)
                        if not (drop and n == "C2")]
                fh.write(f"{len(rows)}\nsynthetic\n")
                for n, e, c in rows:
                    fh.write(f"{e:<2} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}  # ATOM={n}\n")
            idx += 1


def test_generic_gather_canonicalises_and_flags_missing(tmp_path: Path) -> None:
    _make_dataset(tmp_path / "data")
    structs, report = get_profile("generic").gather(tmp_path / "data", "*.xyz")

    assert report["n_kept"] == 24
    # Every structure shares one fixed-length atom template.
    widths = {s.n_atoms() for s in structs}
    assert len(widths) == 1 and widths.pop() == 11  # FE + 4 N + 4 C + 2 axial
    names = {tuple(s.atom_names) for s in structs}
    assert len(names) == 1  # identical template order everywhere
    # The one file missing "C2" has exactly one imputed (not-present) atom.
    missing = [s for s in structs if not s.present.all()]
    assert len(missing) == 1 and int((~missing[0].present).sum()) == 1


def test_heme_matcher_finds_axial_flip(tmp_path: Path) -> None:
    _make_dataset(tmp_path / "data")
    profile = get_profile("heme")
    structs, _ = profile.gather(tmp_path / "data", "*.xyz")
    matcher = profile.build_matcher(structs)

    perms = matcher.perms(structs[0])
    assert len(perms) == 2  # identity + one flip
    names = structs[0].atom_names
    flip = perms[1]
    swapped = {frozenset((names[i], names[flip[i]])) for i in range(len(flip))
               if flip[i] != i}
    assert frozenset(("NAX1", "NAX2")) in swapped  # axial ligands exchange
    # Center (FE) is anchored for distance weighting.
    assert names[matcher.center_index] == "FE"


def test_generic_run_labels_every_structure(tmp_path: Path) -> None:
    from zn_cys_his.clustering.step03_approach1_cartesian import run

    _make_dataset(tmp_path / "data")
    best = run(tmp_path / "data", tmp_path / "out", k_values=[2, 3, 4],
               profile=get_profile("generic"))
    assert best is not None

    import csv
    with (tmp_path / "out" / "labels.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 24
    assert all(r["cluster"] != "" for r in rows)

    # Metric-free profiles still emit a minimal labels-with-stats CSV (the app needs
    # it) — with cluster color + pdb_id, but none of the Zn numeric-metric columns.
    lw = tmp_path / "out" / "kmeans_labels_with_stats.csv"
    assert lw.exists()
    with lw.open() as fh:
        cols = next(csv.reader(fh))
    assert cols == ["id", "cluster", "cluster_color", "pdb_id", "family"]


def test_heme_family_includes_all_pocket_residues(tmp_path: Path) -> None:
    """family = sorted set of ALL non-macrocycle residues present (pocket included)."""
    d = tmp_path / "data"; d.mkdir()

    def write(name: str, extra_lines: list[str]) -> None:
        core = [
            "Fe  0.000 0.000 0.000  # RES=HEM RESSEQ=1 ATOM=FE",
            "N   2.000 0.000 0.000  # RES=HEM RESSEQ=1 ATOM=NA",
            "N   0.000 2.000 0.000  # RES=HEM RESSEQ=1 ATOM=NB",
            "N  -2.000 0.000 0.000  # RES=HEM RESSEQ=1 ATOM=NC",
            "N   0.000 -2.000 0.000  # RES=HEM RESSEQ=1 ATOM=ND",
            "N   0.000 0.000 2.100  # RES=HIS RESSEQ=90 ATOM=NE2",  # proximal, 2.1 Å
        ]
        lines = core + extra_lines
        (d / f"{name}.xyz").write_text(f"{len(lines)}\nheme\n" + "\n".join(lines) + "\n")

    # His only; His + distal NO; His + a far pocket PHE (still counted — it's in the file).
    write("1aaa_h", [])
    write("2bbb_h", ["N  0.000 0.000 -1.900  # RES=NO RESSEQ=200 ATOM=N",
                      "O  0.000 0.600 -2.900  # RES=NO RESSEQ=200 ATOM=O"])
    write("3ccc_h", ["C 8.0 8.0 8.0  # RES=PHE RESSEQ=50 ATOM=CA"])

    structs, _ = get_profile("heme").gather(d, "*.xyz")
    fam = {s.id: s.family for s in structs}
    assert fam["1aaa_h"] == "HIS"
    assert fam["2bbb_h"] == "HIS+NO"
    assert fam["3ccc_h"] == "HIS+PHE"  # pocket PHE included (present in the file)


def test_pdb_back_derive_assigns_names(tmp_path: Path) -> None:
    """A tag-less structure recovers <RES>_<NAME> from a coordinate-matched PDB."""
    from zn_cys_his.clustering.profiles.pdb_tags import (
        back_derive, parse_pdb_atoms, pdb_id_from_stem,
    )
    assert pdb_id_from_stem("1a3n_hem1") == "1a3n"
    assert pdb_id_from_stem("famA_003") is None  # not a PDB id

    pdb = tmp_path / "1abc.pdb"
    pdb.write_text(
        "HETATM    1 FE   HEM A   1      10.000  10.000  10.000  1.00  0.00          FE\n"
        "HETATM    2 NA   HEM A   1      12.000  10.000  10.000  1.00  0.00           N\n"
        "ATOM      3 NE2  HIS A   2      10.000  10.000  12.000  1.00  0.00           N\n"
    )
    atoms = parse_pdb_atoms(pdb)
    assert {a["name"] for a in atoms} == {"FE", "NA", "NE2"}

    raw = {"id": "1abc_h", "elements": ["Fe", "N", "N"],
           "names": ["FE0", "N1", "N2"], "tagged": False,
           "coords": np.array([[10, 10, 10], [12, 10, 10], [10, 10, 12]], float)}
    got = back_derive(raw, atoms)
    assert got is not None
    assert got["names"] == ["HEM_FE", "HEM_NA", "HIS_NE2"]


def test_heme_stats_from_pdb(tmp_path: Path) -> None:
    """compute_stats reads R-factors + B-factors and averages B over non-Fe atoms."""
    from zn_cys_his.clustering.profiles.heme_stats import make_compute_stats

    xyz = tmp_path / "xyz"; xyz.mkdir()
    pdb = tmp_path / "pdb"; pdb.mkdir()
    # PDB with R-factors and three atoms (Fe B=10, two others B=20 and 30).
    (pdb / "1abc.pdb").write_text(
        "REMARK   3   R VALUE            (WORKING SET) : 0.150\n"
        "REMARK   3   FREE R VALUE                     : 0.190\n"
        "HETATM    1 FE   HEM A   1       0.000   0.000   0.000  1.00 10.00          FE\n"
        "HETATM    2 NA   HEM A   1       2.000   0.000   0.000  1.00 20.00           N\n"
        "ATOM      3 NE2  HIS A   2       0.000   0.000   2.100  1.00 30.00           N\n"
    )
    (xyz / "1abc_h.xyz").write_text(
        "3\nheme\n"
        "Fe 0.000 0.000 0.000  # RES=HEM RESSEQ=1 ATOM=FE\n"
        "N  2.000 0.000 0.000  # RES=HEM RESSEQ=1 ATOM=NA\n"
        "N  0.000 0.000 2.100  # RES=HIS RESSEQ=2 ATOM=NE2\n"
    )
    compute = make_compute_stats("FE", macrocycle={"HEM"}, pdb_dir=pdb, fetch=False)
    out = compute(xyz, "*.xyz", tmp_path / "stats.csv")
    assert out is not None

    import csv
    row = next(csv.DictReader(out.open()))
    assert row["r_work"] == "0.150" and row["r_free"] == "0.190"
    assert row["fe_bfactor"] == "10.000"
    assert row["avg_bfactor"] == "25.000"   # mean(20, 30), Fe excluded
    assert row["family"] == "HIS"           # non-macrocycle residue


def test_heme_stats_recentered_xyz(tmp_path: Path) -> None:
    """Coords recentered by a header CENTROID are mapped back to the PDB frame."""
    from zn_cys_his.clustering.profiles.heme_stats import make_compute_stats

    xyz = tmp_path / "xyz"; xyz.mkdir()
    pdb = tmp_path / "pdb"; pdb.mkdir()
    # PDB atoms in the crystal frame (centroid ~ (100, 100, 100)).
    (pdb / "1abc.pdb").write_text(
        "HETATM    1 FE   HEM A   1     100.000 100.000 100.000  1.00 10.00          FE\n"
        "HETATM    2 NA   HEM A   1     102.000 100.000 100.000  1.00 20.00           N\n"
        "ATOM      3 NE2  HIS A   2     100.000 100.000 102.100  1.00 30.00           N\n"
    )
    # XYZ recentered by subtracting CENTROID=(100,100,100) -> coords near origin.
    (xyz / "1abc_h.xyz").write_text(
        "3\nPDB=1abc TARGET=FE CENTROID=(100.000,100.000,100.000)\n"
        "Fe 0.000 0.000 0.000  # RES=HEM RESSEQ=1 ATOM=FE\n"
        "N  2.000 0.000 0.000  # RES=HEM RESSEQ=1 ATOM=NA\n"
        "N  0.000 0.000 2.100  # RES=HIS RESSEQ=2 ATOM=NE2\n"
    )
    compute = make_compute_stats("FE", macrocycle={"HEM"}, pdb_dir=pdb, fetch=False)
    row = next(__import__("csv").DictReader(
        compute(xyz, "*.xyz", tmp_path / "stats.csv").open()))
    # Without the centroid offset these would be blank (atoms ~170 A off).
    assert row["fe_bfactor"] == "10.000"
    assert row["avg_bfactor"] == "25.000"


def test_minimal_report_written(tmp_path: Path) -> None:
    """build_minimal_report emits a self-contained HTML embedding available PNGs."""
    from zn_cys_his.clustering.step06_validate_clusters import build_minimal_report

    # 1x1 PNG so _png_to_b64 has something to embed.
    png = (b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
           b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00"
           b"\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82")
    clu = tmp_path / "clu"; clu.mkdir()
    out = tmp_path / "val"; out.mkdir()
    (clu / "tsne_kmeans.png").write_bytes(png)
    (out / "k_sweep_plot.png").write_bytes(png)
    (out / "rmsd_table_k2.csv").write_text("cluster_id,n\n0,12\n1,12\n")

    build_minimal_report(clu, out, title="Heme test", best_k=2)
    html = (out / "report_cluster_distribution_offline.html").read_text()
    assert "Heme test" in html and "data:image/png;base64," in html
    assert "Per-cluster RMSD" in html  # table embedded
