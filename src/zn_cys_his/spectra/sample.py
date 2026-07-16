#!/usr/bin/env python3
"""Sample additional structures per cluster for a second validation round.

For each k-means cluster in ``approach1/labels.csv`` this samples a number of
*new* structures (default 3, but see ``--cluster-overrides``), copies their
aligned ``.xyz`` files from ``aligned_xyz`` into a fresh "round II" directory,
and guarantees:

  * No overlap with structures already sampled in the round-I directory
    (``sampled-xyz-files-for-val``).
  * The total sampled fraction of any cluster (round I + round II combined)
    never exceeds ``--max-fraction`` (default 0.5).  So a cluster with n=6 and
    2 already sampled can receive at most 1 more (floor(6 * 0.5) = 3 total).

Each copied structure is also protonated (unless ``--no-add-h``): hydrogens are
appended at the end of the file with tetrahedral geometry — 0 H on each SG,
2 H on each CB (methylene, completing its bonds to SG and CA), and 3 H on each
CA (methyl cap, since the backbone is truncated in this model).

Each file's comment line is tagged with ``CHARGE=`` and ``MULTIPLICITY=``
(unless ``--no-charge-mult``).  The charge is derived per structure from its
composition as ``2 − n_cys`` (Zn²⁺ plus one −1 thiolate per cysteine, His
neutral): 4cys=−2, 3cys1his=−1, 2cys2his=0, 1cys3his=+1, 4his=+2.  Multiplicity
is 1.  ``--charge``/``--multiplicity`` override the auto values.

Run with the project environment:

    uv run zch-sample-spectra        # == python -m zn_cys_his.spectra.sample

Use ``--dry-run`` to preview the selection without copying anything.
"""
from __future__ import annotations

import argparse
import csv
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from zn_cys_his.paths import CLUSTER_OUTPUT

DEFAULT_APPROACH = CLUSTER_OUTPUT / "test-4cys-weighted/approach1"
DEFAULT_LABELS = DEFAULT_APPROACH / "labels.csv"
DEFAULT_ALIGNED = DEFAULT_APPROACH / "aligned_xyz"
DEFAULT_EXISTING = (
    CLUSTER_OUTPUT / "test-4cys-weighted/validation/approach1/sampled-xyz-files-for-val"
)
DEFAULT_OUT = (
    CLUSTER_OUTPUT / "test-4cys-weighted/validation/approach1/sampled-xyz-files-for-val-II"
)

CH_BOND = 1.09                       # C-H bond length, angstrom
TET_ANGLE = math.acos(-1.0 / 3.0)    # ideal tetrahedral angle (~109.47 deg)


# ----------------------------------------------------------------------------
# Hydrogen placement (tetrahedral geometry)
# ----------------------------------------------------------------------------
def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _orthonormal_pair(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors spanning the plane perpendicular to unit vector ``u``."""
    ref = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(u, ref))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0])
    v = _unit(np.cross(u, ref))
    w = _unit(np.cross(u, v))
    return v, w


def place_two_h(center: np.ndarray, nbr_a: np.ndarray, nbr_b: np.ndarray,
                bond: float = CH_BOND) -> list[np.ndarray]:
    """Two H completing a tetrahedral (methylene) carbon with two heavy bonds.

    The new bonds are symmetric about the negative bisector of the two existing
    bonds and lie in the plane perpendicular to them.
    """
    n1 = _unit(nbr_a - center)
    n2 = _unit(nbr_b - center)
    bis = _unit(n1 + n2)
    perp = _unit(np.cross(n1, n2))
    half = TET_ANGLE / 2.0
    d1 = -bis * math.cos(half) + perp * math.sin(half)
    d2 = -bis * math.cos(half) - perp * math.sin(half)
    return [center + bond * _unit(d1), center + bond * _unit(d2)]


def place_three_h(center: np.ndarray, nbr: np.ndarray,
                  bond: float = CH_BOND) -> list[np.ndarray]:
    """Three H completing a tetrahedral (methyl) carbon with one heavy bond.

    A staggered tripod: each C-H makes the tetrahedral angle with the single
    existing bond, the three spaced 120 deg apart about that bond axis.
    """
    u = _unit(nbr - center)          # direction toward the one heavy neighbor
    v, w = _orthonormal_pair(u)
    cos_t, sin_t = math.cos(TET_ANGLE), math.sin(TET_ANGLE)
    hs = []
    for k in range(3):
        phi = 2.0 * math.pi * k / 3.0
        d = cos_t * u + sin_t * (math.cos(phi) * v + math.sin(phi) * w)
        hs.append(center + bond * _unit(d))
    return hs


def _parse_meta(comment: str) -> dict[str, str]:
    """Parse ``RES=CYS CHAIN=A RESSEQ=66 ATOM=SG`` from an xyz line comment."""
    meta: dict[str, str] = {}
    for tok in comment.split():
        if "=" in tok:
            key, _, val = tok.partition("=")
            meta[key] = val
    return meta


def _apply_charge_mult(comment: str, charge: int | None, multiplicity: int | None) -> str:
    """Append ``CHARGE=.. MULTIPLICITY=..`` to a comment, replacing any existing tags."""
    if charge is None and multiplicity is None:
        return comment
    tokens = [t for t in comment.split()
              if not t.startswith("CHARGE=") and not t.startswith("MULTIPLICITY=")]
    if charge is not None:
        tokens.append(f"CHARGE={charge}")
    if multiplicity is not None:
        tokens.append(f"MULTIPLICITY={multiplicity}")
    return " ".join(tokens).strip()


def composition_charge(residues: dict) -> int:
    """Formal charge of a 4-coordinate Zn(Cys/His) site: +2 (Zn) − 1 per cysteinate.

    Each coordinating cysteine is a −1 thiolate; histidine is neutral.  So
    charge = 2 − n_cys, giving 4cys=−2, 3cys1his=−1, 2cys2his=0, 1cys3his=+1,
    4his=+2.  ``residues`` is keyed by (RES, CHAIN, RESSEQ).
    """
    n_cys = sum(1 for (res, _, _) in residues if res.upper().startswith("CY"))
    return 2 - n_cys


def protonate_xyz(
    src: Path,
    dst: Path,
    add_h: bool = True,
    tag_charge_mult: bool = True,
    charge: int | None = None,
    multiplicity: int | None = None,
) -> int:
    """Copy ``src`` -> ``dst``; return the number of H atoms added (0 if add_h=False).

    Heavy-atom lines are preserved verbatim.  When ``add_h`` is set, for every CB
    atom 2 H are placed using its SG and CA neighbours within the same residue,
    and for every CA atom 3 H are placed using its CB neighbour (SG atoms get
    none).  When ``tag_charge_mult`` is set, ``CHARGE=/MULTIPLICITY=`` tags are
    written to the comment line: ``charge`` defaults to the composition-derived
    ``2 − n_cys`` (see :func:`composition_charge`) and ``multiplicity`` to 1;
    pass explicit values to override either.
    """
    lines = src.read_text().splitlines()
    comment = lines[1] if len(lines) > 1 else ""
    atom_lines = lines[2:]

    # Parse atoms and group by residue.
    residues: dict[tuple[str, str, str], dict[str, np.ndarray]] = defaultdict(dict)
    for ln in atom_lines:
        if not ln.strip():
            continue
        body, _, cmt = ln.partition("#")
        parts = body.split()
        if len(parts) < 4:
            continue
        xyz = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
        meta = _parse_meta(cmt)
        atom = meta.get("ATOM")
        key = (meta.get("RES", ""), meta.get("CHAIN", ""), meta.get("RESSEQ", ""))
        if atom:
            residues[key][atom] = xyz

    if tag_charge_mult:
        c = charge if charge is not None else composition_charge(residues)
        m = multiplicity if multiplicity is not None else 1
        comment = _apply_charge_mult(comment, c, m)

    def _fmt(elem: str, p: np.ndarray, res: tuple[str, str, str], name: str) -> str:
        r, ch, seq = res
        return (f"{elem:<3} {p[0]:>11.6f} {p[1]:>11.6f} {p[2]:>11.6f}"
                f"  # RES={r} CHAIN={ch} RESSEQ={seq} ATOM={name}")

    h_lines: list[str] = []
    if add_h:
        for key, atoms in residues.items():
            res, _, seq = key
            cb, ca, sg = atoms.get("CB"), atoms.get("CA"), atoms.get("SG")
            if cb is not None and ca is not None and sg is not None:
                for i, h in enumerate(place_two_h(cb, sg, ca), start=1):
                    h_lines.append(_fmt("H", h, key, f"HB{i}"))
            if ca is not None and cb is not None:
                for i, h in enumerate(place_three_h(ca, cb), start=1):
                    h_lines.append(_fmt("H", h, key, f"HA{i}"))

    heavy_lines = [ln for ln in atom_lines if ln.strip()]
    out_lines = [str(len(heavy_lines) + len(h_lines)), comment]
    out_lines.extend(heavy_lines)
    out_lines.extend(h_lines)
    dst.write_text("\n".join(out_lines) + "\n")
    return len(h_lines)


def parse_overrides(spec: str | None) -> dict[int, int]:
    """Parse ``"20:8,4:2"`` into ``{20: 8, 4: 2}``."""
    out: dict[int, int] = {}
    if not spec:
        return out
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        cluster_str, count_str = part.split(":")
        out[int(cluster_str)] = int(count_str)
    return out


def load_clusters(labels_csv: Path) -> dict[int, list[str]]:
    """Return ``{cluster: [structure_id, ...]}`` from labels.csv."""
    members: dict[int, list[str]] = defaultdict(list)
    with labels_csv.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            members[int(row["cluster"])].append(row["structure_id"])
    return dict(members)


def existing_stems(existing_dir: Path) -> set[str]:
    """Structure ids already sampled (by .xyz stem) in the round-I directory."""
    if not existing_dir.is_dir():
        return set()
    return {p.stem for p in existing_dir.glob("*.xyz")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("--aligned-xyz", type=Path, default=DEFAULT_ALIGNED)
    ap.add_argument("--existing", type=Path, default=DEFAULT_EXISTING,
                    help="round-I sampled dir; new picks must not overlap it")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="round-II output dir for the newly sampled .xyz files")
    ap.add_argument("--n-per-cluster", type=int, default=3,
                    help="target number of NEW structures per cluster")
    ap.add_argument("--cluster-overrides", default="20:8",
                    help='per-cluster target overrides, e.g. "20:8,4:2"')
    ap.add_argument("--max-fraction", type=float, default=0.5,
                    help="max fraction of a cluster sampled in total (I + II)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-add-h", action="store_true",
                    help="copy heavy atoms only; skip hydrogen placement")
    ap.add_argument("--charge", type=int, default=None,
                    help="Override the per-structure CHARGE tag. Default: 2 - n_cys, derived "
                         "from each file's residues (4cys=-2, 3cys1his=-1, 2cys2his=0, "
                         "1cys3his=+1, 4his=+2). See --no-charge-mult.")
    ap.add_argument("--multiplicity", type=int, default=None,
                    help="Override the MULTIPLICITY tag (default: 1 for all compositions).")
    ap.add_argument("--no-charge-mult", action="store_true",
                    help="do not write CHARGE=/MULTIPLICITY= tags.")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the selection but copy nothing")
    args = ap.parse_args()

    tag_charge_mult = not args.no_charge_mult

    rng = random.Random(args.seed)
    overrides = parse_overrides(args.cluster_overrides)

    clusters = load_clusters(args.labels)
    already = existing_stems(args.existing)

    print(f"labels:        {args.labels}")
    print(f"aligned xyz:   {args.aligned_xyz}")
    print(f"existing (I):  {args.existing}  ({len(already)} files)")
    print(f"output  (II):  {args.out}")
    print(f"seed={args.seed}  n_per_cluster={args.n_per_cluster}  "
          f"overrides={overrides}  max_fraction={args.max_fraction}")
    print("-" * 88)
    header = (f"{'clust':>5}  {'n':>5}  {'cap':>4}  {'had':>4}  "
              f"{'want':>4}  {'room':>4}  {'avail':>5}  {'add':>4}  note")
    print(header)

    selected: dict[int, list[str]] = {}
    total_add = 0
    warnings: list[str] = []

    for cluster in sorted(clusters):
        members = clusters[cluster]
        n = len(members)
        cap = math.floor(n * args.max_fraction)          # max total sampled
        had = sum(1 for m in members if m in already)     # round-I in this cluster
        want = overrides.get(cluster, args.n_per_cluster)
        room = max(0, cap - had)                           # 50%-cap headroom
        available = [m for m in members if m not in already]
        n_add = min(want, room, len(available))

        note = ""
        if n_add < want:
            reasons = []
            if room < want:
                reasons.append(f"capped@50% (cap={cap}, had={had})")
            if len(available) < want:
                reasons.append(f"only {len(available)} unsampled left")
            note = "; ".join(reasons)
            warnings.append(f"cluster {cluster}: wanted {want}, added {n_add} "
                            f"({note})")

        picks = rng.sample(available, n_add) if n_add else []
        selected[cluster] = picks
        total_add += n_add

        print(f"{cluster:>5}  {n:>5}  {cap:>4}  {had:>4}  {want:>4}  "
              f"{room:>4}  {len(available):>5}  {n_add:>4}  {note}")

    print("-" * 88)
    print(f"total new structures to sample: {total_add}")
    if warnings:
        print("\nnotes:")
        for w in warnings:
            print(f"  - {w}")

    # -- copy --------------------------------------------------------------
    if args.dry_run:
        print("\n[dry-run] no files copied.")
        return

    args.out.mkdir(parents=True, exist_ok=True)
    copied = 0
    total_h = 0
    missing_src: list[str] = []
    for cluster in sorted(selected):
        for sid in selected[cluster]:
            src = args.aligned_xyz / f"{sid}.xyz"
            if not src.is_file():
                missing_src.append(sid)
                continue
            dst = args.out / src.name
            total_h += protonate_xyz(src, dst, add_h=not args.no_add_h,
                                     tag_charge_mult=tag_charge_mult,
                                     charge=args.charge, multiplicity=args.multiplicity)
            copied += 1

    if not tag_charge_mult:
        tag = ""
    elif args.charge is not None:
        tag = f", CHARGE={args.charge} MULTIPLICITY={args.multiplicity if args.multiplicity is not None else 1}"
    else:
        mult = args.multiplicity if args.multiplicity is not None else 1
        tag = f", CHARGE=2-n_cys MULTIPLICITY={mult} (per structure)"
    if args.no_add_h:
        print(f"\ncopied {copied} files (heavy atoms only{tag}) -> {args.out}")
    else:
        print(f"\ncopied {copied} files (+{total_h} H atoms added{tag}) -> {args.out}")
    if missing_src:
        print(f"WARNING: {len(missing_src)} source .xyz files not found in "
              f"{args.aligned_xyz}:")
        for sid in missing_src:
            print(f"  - {sid}")


if __name__ == "__main__":
    main()
