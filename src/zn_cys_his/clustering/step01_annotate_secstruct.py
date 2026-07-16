#!/usr/bin/env python3
"""Annotate XYZ CA atoms with HELIX/SHEET/LOOP secondary structure from a PDB.

Pipeline step 01 (runs before step02 compute_stats).  Each CA atom's comment
gains a ``SEC=LOOP|HELIX|SHEET`` tag, read from the HELIX/SHEET records of the
matching ``<pdbid>.pdb``.  Downstream, step02_compute_stats reads these tags to
build the per-structure ``family`` string, which the validation/clustering
plots then group and label by.

With ``--out-dir`` the annotated files are written as COPIES into that directory
(the input XYZ files are left untouched); without it the edit is made in place.
Either way it is idempotent (any existing SEC= tag is replaced).  If a PDB
cannot be found, its CA atoms are tagged ``SEC=XXXX`` (rendered as ``?`` in the
family string) and a warning is emitted.

Usage
-----
  # Annotate into a separate output dir (how the pipeline runs it — inputs stay pristine):
  uv run python -m zn_cys_his.clustering.step01_annotate_secstruct \\
      data/3cys1his-large/xyz-files \\
      --pdb-dir data/3cys1his-large/pdb-files \\
      --out-dir cluster-output/3cys1his-large/prep/annotated_xyz

  # Or in place, auto-resolving PDBs from <xyz-base>/../pdb-files/<pdbid>.pdb:
  uv run python -m zn_cys_his.clustering.step01_annotate_secstruct \\
      data/3cys1his-large/xyz-files
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class _SecStructRange:
    chain: str
    start: int
    end: int


def _parse_int(value: str) -> int | None:
    try:
        return int(value.strip())
    except (TypeError, ValueError):
        return None


def parse_pdb_secondary_structure(pdb_path: Path) -> Dict[str, List[_SecStructRange]]:
    helix_ranges: List[_SecStructRange] = []
    sheet_ranges: List[_SecStructRange] = []

    with pdb_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("HELIX"):
                # PDB v3.x fixed columns
                start_resseq = _parse_int(line[21:25])
                end_resseq = _parse_int(line[33:37])
                chain = line[19:20].strip()
                if start_resseq is None or end_resseq is None or not chain:
                    continue
                start = min(start_resseq, end_resseq)
                end = max(start_resseq, end_resseq)
                helix_ranges.append(_SecStructRange(chain=chain, start=start, end=end))
            elif line.startswith("SHEET"):
                start_resseq = _parse_int(line[22:26])
                end_resseq = _parse_int(line[33:37])
                chain = line[21:22].strip()
                if start_resseq is None or end_resseq is None or not chain:
                    continue
                start = min(start_resseq, end_resseq)
                end = max(start_resseq, end_resseq)
                sheet_ranges.append(_SecStructRange(chain=chain, start=start, end=end))

    return {"HELIX": helix_ranges, "SHEET": sheet_ranges}


def determine_secondary_structure(
    secstruct: Dict[str, List[_SecStructRange]],
    chain: str,
    resseq: int,
) -> str:
    for entry in secstruct.get("HELIX", []):
        if entry.chain == chain and entry.start <= resseq <= entry.end:
            return "HELIX"
    for entry in secstruct.get("SHEET", []):
        if entry.chain == chain and entry.start <= resseq <= entry.end:
            return "SHEET"
    return "LOOP"


def _parse_meta_tokens(comment: str) -> List[str]:
    return [token for token in comment.strip().split() if token]


def _meta_dict(tokens: List[str]) -> Dict[str, str]:
    meta: Dict[str, str] = {}
    for token in tokens:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key and value:
            meta[key] = value
    return meta


def _update_comment_with_secstruct(comment: str, secstruct: str) -> str:
    tokens = _parse_meta_tokens(comment)
    tokens = [t for t in tokens if not t.startswith("SEC=") and not t.startswith("SECSTRUCT=")]
    tokens.append(f"SEC={secstruct}")
    return " ".join(tokens).strip()


def _resolve_pdb_for_xyz(xyz_path: Path, pdb_dir: Path | None = None) -> Path:
    pdb_id = xyz_path.stem.split("_", 1)[0]
    if pdb_dir is not None:
        return pdb_dir / f"{pdb_id}.pdb"
    # Fallback: sibling pdb-files/ of the xyz dir's parent dataset directory.
    return xyz_path.parents[1] / "pdb-files" / f"{pdb_id}.pdb"


def annotate_xyz_with_secstruct(
    xyz_path: Path, pdb_dir: Path | None = None, out_path: Path | None = None
) -> bool:
    """Annotate xyz_path; write to out_path if given, else in place."""
    if not xyz_path.exists():
        raise SystemExit(f"XYZ file not found: {xyz_path}")

    dest = out_path or xyz_path
    pdb_path = _resolve_pdb_for_xyz(xyz_path, pdb_dir)
    lines = xyz_path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        raise SystemExit(f"XYZ file too short: {xyz_path}")

    header = lines[:2]
    body = lines[2:]
    updated_body: List[str] = []

    if not pdb_path.exists():
        print(f"Warning: PDB not found for {xyz_path.name}: {pdb_path}", file=sys.stderr)
        # Annotate all CA atoms with SEC=XXXX
        for line in body:
            stripped = line.strip()
            if not stripped:
                updated_body.append(line)
                continue
            if "#" in stripped:
                left, right = stripped.split("#", 1)
                comment = right.strip()
            else:
                left, comment = stripped, ""
            parts = left.split()
            if len(parts) < 4:
                updated_body.append(line)
                continue
            meta = _meta_dict(_parse_meta_tokens(comment))
            if meta.get("ATOM") == "CA" and "RESSEQ" in meta and "CHAIN" in meta:
                comment = _update_comment_with_secstruct(comment, "XXXX")
            if comment:
                updated_body.append(f"{left.strip()}  # {comment}")
            else:
                updated_body.append(left.strip())
        dest.write_text("\n".join(header + updated_body) + "\n", encoding="utf-8")
        return True

    secstruct = parse_pdb_secondary_structure(pdb_path)
    for line in body:
        stripped = line.strip()
        if not stripped:
            updated_body.append(line)
            continue
        if "#" in stripped:
            left, right = stripped.split("#", 1)
            comment = right.strip()
        else:
            left, comment = stripped, ""
        parts = left.split()
        if len(parts) < 4:
            updated_body.append(line)
            continue
        meta = _meta_dict(_parse_meta_tokens(comment))
        if meta.get("ATOM") == "CA" and "RESSEQ" in meta and "CHAIN" in meta:
            resseq = _parse_int(meta.get("RESSEQ", ""))
            chain = meta.get("CHAIN", "").strip()
            if resseq is not None and chain:
                sec = determine_secondary_structure(secstruct, chain, resseq)
                comment = _update_comment_with_secstruct(comment, sec)
        if comment:
            updated_body.append(f"{left.strip()}  # {comment}")
        else:
            updated_body.append(left.strip())
    dest.write_text("\n".join(header + updated_body) + "\n", encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate XYZ CA atoms with HELIX/SHEET/LOOP metadata from a PDB file."
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to an input .xyz file or a directory containing .xyz files",
    )
    parser.add_argument(
        "--pdb-dir",
        type=Path,
        default=None,
        help="Directory of <pdbid>.pdb files to read HELIX/SHEET records from. "
             "If omitted, each PDB is resolved to "
             "<xyz-base>/pdb-files/<pdbid>.pdb.",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="*.xyz",
        help="Filename pattern when PATH is a directory (default: *.xyz).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Write annotated COPIES here instead of editing inputs in place. "
             "Files keep their names; the directory is created if needed.",
    )
    args = parser.parse_args()

    pdb_dir = args.pdb_dir.expanduser().resolve() if args.pdb_dir else None
    out_dir = args.out_dir.expanduser().resolve() if args.out_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    target = args.path
    if target.is_dir():
        xyz_files = sorted(target.glob(args.glob))
        if not xyz_files:
            raise SystemExit(f"No files matching {args.glob!r} found in {target}")
        failures = 0
        for xyz_path in xyz_files:
            out_path = (out_dir / xyz_path.name) if out_dir else None
            if not annotate_xyz_with_secstruct(xyz_path, pdb_dir, out_path):
                failures += 1
        if failures:
            raise SystemExit(1)
    else:
        out_path = (out_dir / target.name) if out_dir else None
        ok = annotate_xyz_with_secstruct(target, pdb_dir, out_path)
        if not ok:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
