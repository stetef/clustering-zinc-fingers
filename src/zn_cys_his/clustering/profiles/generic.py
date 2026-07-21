"""Generic template-aligned profile for arbitrary rigid structures.

Unlike the Zn(Cys/His) model — a metal plus independent, freely-permutable
residue arms — this profile treats every structure as a single rigid atom cloud
with a fixed, dataset-wide atom template:

  1. read every source XYZ as (element, coords, atom-name) heavy atoms;
  2. build one canonical, ordered atom-name template = the union of names seen
     across the dataset (stable first-appearance order);
  3. canonicalise each structure onto that template, so slot *i* is the same
     atom everywhere and all coordinate vectors are the same length;
  4. atoms missing from a file are imputed (placed at the file's centroid) and
     flagged ``present=False`` so alignment down-weights them to zero.

The default symmetry group is the identity alone (fixed 1:1 correspondence,
plain Kabsch).  Chemistry-specific subclasses (see :mod:`heme`) extend it with a
small permutation set.

Atom names come from the ``ATOM=`` end-of-line tag when present (as in the Zn
extraction pipeline); otherwise a name is synthesised from the element and its
position, which assumes a consistent atom order across files.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .base import GenericStructure, Matcher, StructureProfile


# ---------------------------------------------------------------------------
# Raw XYZ reading
# ---------------------------------------------------------------------------

def _tag(comment: str, key: str) -> Optional[str]:
    for part in comment.split():
        if part.startswith(key):
            return part[len(key):].strip()
    return None


def read_raw_xyz(path: Path, include_h: bool = False) -> Optional[dict]:
    """Read one XYZ file → {names, elements, coords}.  None on failure.

    Heavy atoms only by default (H/D dropped).  ``atom_name`` is the ``ATOM=``
    tag if present, else ``<ELEMENT><ordinal>`` in file order.
    """
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return None
    if len(lines) < 2:
        return None

    names: list[str] = []
    elements: list[str] = []
    coords: list[np.ndarray] = []
    ordinal = 0
    for line in lines[2:]:
        halves = line.split("#", 1)
        parts = halves[0].split()
        comment = halves[1] if len(halves) > 1 else ""
        if len(parts) < 4:
            continue
        elem = parts[0].strip()
        try:
            xyz = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
        except ValueError:
            continue
        if not include_h and elem.upper() in ("H", "D"):
            continue
        name = _tag(comment, "ATOM=")
        if name:
            name = name.upper()
        else:
            name = f"{elem.upper()}{ordinal}"
        ordinal += 1
        names.append(name)
        elements.append(elem)
        coords.append(xyz)

    if not coords:
        return None
    return {"id": path.stem, "names": names,
            "elements": elements, "coords": np.array(coords)}


# ---------------------------------------------------------------------------
# Template building + canonicalisation
# ---------------------------------------------------------------------------

def build_template(raws: list[dict]) -> tuple[list[str], dict[str, str]]:
    """Union of atom names across the dataset, in stable first-appearance order.

    Returns (ordered_names, element_by_name).  Duplicate names within a single
    file are disambiguated with a ``#k`` suffix so they stay distinct slots.
    """
    ordered: list[str] = []
    seen: set[str] = set()
    elem_by_name: dict[str, str] = {}
    for raw in raws:
        local_counts: dict[str, int] = {}
        for name, elem in zip(raw["names"], raw["elements"]):
            k = local_counts.get(name, 0)
            local_counts[name] = k + 1
            uname = name if k == 0 else f"{name}#{k}"
            if uname not in seen:
                seen.add(uname)
                ordered.append(uname)
                elem_by_name[uname] = elem
    return ordered, elem_by_name


def _uniquify(names: list[str]) -> list[str]:
    """Apply the same ``#k`` disambiguation used in build_template to one file."""
    out: list[str] = []
    counts: dict[str, int] = {}
    for name in names:
        k = counts.get(name, 0)
        counts[name] = k + 1
        out.append(name if k == 0 else f"{name}#{k}")
    return out


def canonicalize(
    raw: dict,
    template: list[str],
    elem_by_name: dict[str, str],
    center_name: Optional[str],
) -> GenericStructure:
    """Map one raw structure onto the template, imputing/padding missing atoms."""
    unames = _uniquify(raw["names"])
    coord_by_name = {u: c for u, c in zip(unames, raw["coords"])}

    centroid = raw["coords"].mean(0)
    M = len(template)
    coords = np.zeros((M, 3))
    present = np.zeros(M, dtype=bool)
    for i, tname in enumerate(template):
        if tname in coord_by_name:
            coords[i] = coord_by_name[tname]
            present[i] = True
        else:
            coords[i] = centroid          # neutral placeholder; weight 0 in fit
            present[i] = False

    center_index = None
    if center_name is not None:
        cn = center_name.upper()
        matches = [i for i, t in enumerate(template) if t.split("#")[0] == cn]
        if matches:
            center_index = matches[0]

    return GenericStructure(
        id=raw["id"], coords=coords,
        atom_names=list(template),
        elements=[elem_by_name[t] for t in template],
        present=present, center_index=center_index,
        template_key=("generic", M),
    )


# ---------------------------------------------------------------------------
# Dataset gather (matches the gather_structures report contract)
# ---------------------------------------------------------------------------

def make_gather(center_name: Optional[str] = None,
                include_h: bool = False) -> Callable[[Path, str], tuple]:
    def _gather(xyz_dir: Path, glob_pat: str) -> tuple:
        files = sorted(xyz_dir.glob(glob_pat))
        n_listed = len(files)
        raws = [r for f in files if (r := read_raw_xyz(f, include_h)) is not None]
        n_parse_fail = n_listed - len(raws)
        template, elem_by_name = build_template(raws)
        structures = [canonicalize(r, template, elem_by_name, center_name)
                      for r in raws]
        report = {
            "n_listed": n_listed,
            "modal_atom_count": len(template),
            "dropped_extra_ligand": [],
            "dropped_atomcount": [],
            "n_parse_fail": n_parse_fail,
            "dropped_composition": [],
            "n_kept": len(structures),
            "composition": ("generic", len(template)) if structures else None,
        }
        return structures, report
    return _gather


# ---------------------------------------------------------------------------
# Matcher (identity by default) + XYZ writer
# ---------------------------------------------------------------------------

def make_build_matcher(
    center_name: Optional[str] = None,
    extra_perms: Optional[Callable[[list[str]], list[np.ndarray]]] = None,
) -> Callable[[list], Optional[Matcher]]:
    """Build a matcher whose permutation set = identity + optional extra perms.

    ``extra_perms(template_names) -> list[index arrays]`` lets a subclass add a
    symmetry (e.g. the heme axial flip).  The identity is always first.
    """
    def _build(structures: list) -> Optional[Matcher]:
        if not structures:
            return None
        M = structures[0].n_atoms()
        template = structures[0].atom_names
        perms: list[np.ndarray] = [np.arange(M)]
        if extra_perms is not None:
            for p in extra_perms(template):
                p = np.asarray(p)
                if p.shape == (M,) and not np.array_equal(p, perms[0]):
                    perms.append(p)
        center_index = structures[0].center_index
        return Matcher(perms, center_index)
    return _build


def write_generic_xyz(structure: GenericStructure, path: Path) -> None:
    """Write a heavy-atom XYZ for a GenericStructure (present atoms only)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [(e, xyz, n) for e, xyz, n, pr
            in zip(structure.elements, structure.coords,
                   structure.atom_names, structure.present) if pr]
    with path.open("w", encoding="utf-8") as fh:
        fh.write(f"{len(rows)}\n")
        fh.write(f"id={structure.id}\n")
        for elem, xyz, name in rows:
            fh.write(f"{elem:<2} {xyz[0]:.6f}  {xyz[1]:.6f}  {xyz[2]:.6f}"
                     f"  # ATOM={name}\n")


# ---------------------------------------------------------------------------
# The profile
# ---------------------------------------------------------------------------

PROFILE = StructureProfile(
    name="generic",
    gather=make_gather(center_name=None),
    build_matcher=make_build_matcher(center_name=None),
    write_xyz=write_generic_xyz,
    has_metrics=False,
)
