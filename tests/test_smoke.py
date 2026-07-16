"""Smoke tests: the package imports and every entrypoint is wired correctly.

These do not run the pipeline (that needs the large data/ tree) — they just
guard the restructure: imports resolve, shared paths are sane, and each module
that the orchestrator / console scripts invoke actually exposes a ``main``.
"""
from __future__ import annotations

import importlib

import pytest

STEP_MODULES = [
    "zn_cys_his.clustering.orchestrate",
    "zn_cys_his.clustering.step01_annotate_secstruct",
    "zn_cys_his.clustering.step02_compute_stats",
    "zn_cys_his.clustering.step03_approach1_cartesian",
    "zn_cys_his.clustering.step04_approach2_zmatrix",
    "zn_cys_his.clustering.step05_approach3_piv",
    "zn_cys_his.clustering.step06_validate_clusters",
]

ENTRYPOINT_MODULES = STEP_MODULES + [
    "zn_cys_his.spectra.sample",
    "zn_cys_his.spectra.plot",
    "zn_cys_his.query_app.build_db",
]


def test_paths_point_at_repo_root() -> None:
    from zn_cys_his.paths import DATA_DIR, REPO_ROOT

    # REPO_ROOT should contain the src/ tree and be the parent of data/.
    assert (REPO_ROOT / "src" / "zn_cys_his").is_dir()
    assert DATA_DIR == REPO_ROOT / "data"


def test_mirror_output_maps_data_to_cluster_output() -> None:
    from zn_cys_his.paths import CLUSTER_OUTPUT, DATA_DIR, mirror_output

    assert mirror_output(DATA_DIR / "4cys-large") == CLUSTER_OUTPUT / "4cys-large"


@pytest.mark.parametrize(
    "n_cys, n_his, expected",
    [(4, 0, -2), (3, 1, -1), (2, 2, 0), (1, 3, 1), (0, 4, 2)],
)
def test_composition_charge_is_two_minus_ncys(n_cys: int, n_his: int, expected: int) -> None:
    from zn_cys_his.spectra.sample import composition_charge

    residues = {("CYS", "A", str(i)): {} for i in range(n_cys)}
    residues.update({("HIS", "A", str(100 + i)): {} for i in range(n_his)})
    assert composition_charge(residues) == expected


def test_utils_imports() -> None:
    utils = importlib.import_module("zn_cys_his.clustering.utils")
    for name in ("parse_structure", "Structure", "gather_structures"):
        assert hasattr(utils, name), name


@pytest.mark.parametrize("module", ENTRYPOINT_MODULES)
def test_module_has_main(module: str) -> None:
    mod = importlib.import_module(module)
    assert callable(getattr(mod, "main", None)), f"{module} is missing a main()"
