#!/usr/bin/env python3
"""Download the source PDB files a dataset's XYZ structures came from (RCSB).

For each ``*.xyz`` in ``--xyz-dir`` the 4-char RCSB id is the filename token
before the first ``_`` (e.g. ``1v4p_cluster1_Zn.xyz`` -> ``1v4p``).  Any id whose
``<id>.pdb`` is missing from ``--pdb-dir`` is fetched from
https://files.rcsb.org.  Existing files are left alone, so this is idempotent
and cheap to re-run — it slots into the pipeline's prep stage, which needs the
PDBs for secondary-structure annotation (step01) and B-factor/R-factor stats
(step02).  The large PDB tree is not version-controlled; this repopulates it.

Usage
-----
  uv run zch-fetch-pdbs \\
      --xyz-dir data/3cys1his-large/xyz-files \\
      --pdb-dir data/3cys1his-large/pdb-files
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import requests

RCSB_URL = "https://files.rcsb.org/download/{pid}.pdb"


def pdb_id_from_xyz(xyz_path: Path) -> str:
    """RCSB id = filename stem up to the first underscore (e.g. ``1v4p``)."""
    return xyz_path.stem.split("_", 1)[0]


def pdb_ids_from_dir(xyz_dir: Path, glob_pat: str = "*.xyz") -> list[str]:
    return sorted({pdb_id_from_xyz(p) for p in xyz_dir.glob(glob_pat)})


def download_pdb(pid: str, dest: Path, timeout: float = 30.0) -> bool:
    """Fetch <pid>.pdb into dest.  Returns True on success."""
    try:
        r = requests.get(RCSB_URL.format(pid=pid), timeout=timeout)
    except requests.RequestException as exc:
        print(f"  ! {pid}: request failed ({exc})", file=sys.stderr)
        return False
    if r.status_code != 200:
        print(f"  ! {pid}: HTTP {r.status_code} (obsolete/renamed id?)", file=sys.stderr)
        return False
    if "<html" in r.text[:200].lower():
        print(f"  ! {pid}: response is not a PDB file", file=sys.stderr)
        return False
    dest.write_text(r.text, encoding="utf-8")
    return True


def fetch_missing(
    xyz_dir: Path,
    pdb_dir: Path,
    glob_pat: str = "*.xyz",
    force: bool = False,
    timeout: float = 30.0,
    sleep: float = 0.0,
) -> tuple[int, int, int]:
    """Download every <id>.pdb absent from pdb_dir.  Returns (downloaded, failed, present)."""
    pdb_dir.mkdir(parents=True, exist_ok=True)
    ids = pdb_ids_from_dir(xyz_dir, glob_pat)
    todo = [(pid, pdb_dir / f"{pid}.pdb") for pid in ids
            if force or not (pdb_dir / f"{pid}.pdb").is_file()]
    present = len(ids) - len(todo)

    if not todo:
        print(f"fetch_pdbs: all {len(ids)} PDBs already present in {pdb_dir}")
        return (0, 0, present)

    print(f"fetch_pdbs: {len(todo)}/{len(ids)} PDBs missing → downloading from RCSB …")
    downloaded = failed = 0
    for pid, dest in todo:
        if download_pdb(pid, dest, timeout):
            downloaded += 1
        else:
            failed += 1
        if sleep:
            time.sleep(sleep)
    print(f"fetch_pdbs: downloaded {downloaded}, failed {failed}, already present {present}")
    return (downloaded, failed, present)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--xyz-dir", type=Path, required=True,
                    help="Directory of *.xyz whose ids determine which PDBs to fetch.")
    ap.add_argument("--pdb-dir", type=Path, required=True,
                    help="Destination directory for <id>.pdb files (created if needed).")
    ap.add_argument("--glob", default="*.xyz", help="XYZ filename pattern (default: *.xyz).")
    ap.add_argument("--force", action="store_true", help="re-download even if the file exists.")
    ap.add_argument("--timeout", type=float, default=30.0, help="per-request timeout, seconds.")
    ap.add_argument("--sleep", type=float, default=0.0, help="delay between downloads, seconds.")
    args = ap.parse_args()

    if not args.xyz_dir.is_dir():
        raise SystemExit(f"XYZ dir not found: {args.xyz_dir}")

    _, failed, _ = fetch_missing(args.xyz_dir, args.pdb_dir, args.glob,
                                 args.force, args.timeout, args.sleep)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
