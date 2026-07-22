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

import re
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .base import GenericStructure, Matcher, StructureProfile

_CENTROID_RE = re.compile(r"CENTROID=\(([^)]+)\)")


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

    # Extraction may recenter coords by subtracting a CENTROID (in the header);
    # capture it so B-factor/name matching can map back to the source PDB frame.
    centroid = None
    if len(lines) >= 2 and (m := _CENTROID_RE.search(lines[1])):
        try:
            centroid = np.array([float(x) for x in m.group(1).split(",")])
        except ValueError:
            centroid = None

    names: list[str] = []
    elements: list[str] = []
    coords: list[np.ndarray] = []
    res_names: list[str] = []
    ordinal = 0
    tagged = False
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
            tagged = True
        else:
            name = f"{elem.upper()}{ordinal}"
        ordinal += 1
        names.append(name)
        elements.append(elem)
        coords.append(xyz)
        res_names.append((_tag(comment, "RES=") or "").upper())

    if not coords:
        return None
    return {"id": path.stem, "names": names, "elements": elements,
            "coords": np.array(coords), "res_names": res_names,
            "tagged": tagged, "centroid": centroid}


def compute_family(raw: dict, center_name: Optional[str],
                   macrocycle: Optional[set] = None) -> str:
    """Family = sorted set of all non-macrocycle residues present in the file.

    Every residue in the extracted environment (the file only contains atoms
    within the extraction cutoff of the center) counts — proximal/axial ligands
    AND pocket residues — so the label reflects the full pocket content, e.g.
    ``HIS`` (5-coordinate), ``HIS+NO``, ``ARG+HIS``, ``GLN+HIS+TYR``.  Only the
    macrocycle residue itself (HEM/HEC/…) is excluded, since it is constant.
    Returns "" when there are no per-atom RES tags to work from.
    """
    res_names = raw.get("res_names") or []
    if not any(res_names):
        return ""
    macro = macrocycle if macrocycle is not None else set()
    present = {r for r in res_names if r and r not in macro}
    return "+".join(sorted(present))


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
        # Match the atom-name token, tolerating a "<RES>_" prefix (from PDB
        # back-derivation) and a "#k" de-duplication suffix.
        def _tok(t: str) -> str:
            return t.split("#")[0].split("_")[-1]
        matches = [i for i, t in enumerate(template) if _tok(t) == cn]
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

def _enrich_from_pdb(raws: list[dict], pdb_dir: Optional[Path], fetch: bool) -> int:
    """Back-derive names for untagged raws whose stem starts with a PDB id.

    Mutates ``raws`` in place; returns the number of structures enriched.  A raw
    that already carried ATOM= tags is left untouched.
    """
    from .pdb_tags import back_derive, parse_pdb_atoms, pdb_id_from_stem, resolve_pdb

    cache: dict[str, list] = {}
    enriched = 0
    for i, raw in enumerate(raws):
        if raw.get("tagged"):
            continue
        pid = pdb_id_from_stem(raw["id"])
        if not pid:
            continue
        if pid not in cache:
            p = resolve_pdb(pid, pdb_dir, fetch)
            cache[pid] = parse_pdb_atoms(p) if p else []
        got = back_derive(raw, cache[pid])
        if got is not None:
            raws[i] = got
            enriched += 1
    return enriched


def make_gather(center_name: Optional[str] = None, include_h: bool = False,
                pdb_dir: Optional[Path] = None, fetch_pdbs: bool = False,
                macrocycle: Optional[set] = None) -> Callable[[Path, str], tuple]:
    def _gather(xyz_dir: Path, glob_pat: str) -> tuple:
        files = sorted(xyz_dir.glob(glob_pat))
        n_listed = len(files)
        raws = [r for f in files if (r := read_raw_xyz(f, include_h)) is not None]
        n_parse_fail = n_listed - len(raws)
        # Recover atom names for tag-less files from their source PDB when possible.
        n_enriched = 0
        if any(not r.get("tagged") for r in raws):
            n_enriched = _enrich_from_pdb(raws, pdb_dir, fetch_pdbs)
            if n_enriched:
                print(f"  back-derived atom tags from PDB for {n_enriched} structure(s)")
        template, elem_by_name = build_template(raws)
        structures = []
        for r in raws:
            s = canonicalize(r, template, elem_by_name, center_name)
            s.family = compute_family(r, center_name, macrocycle=macrocycle)
            structures.append(s)
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

_FAMILY_DOC = (
    "The set of residue names present in each structure's extracted environment "
    "(joined by `+`). Clusters are built from geometry, so a family may span several."
)


def make_profile(pdb_dir: Optional[Path] = None,
                 fetch_pdbs: bool = False) -> StructureProfile:
    return StructureProfile(
        name="generic",
        gather=make_gather(center_name=None, pdb_dir=pdb_dir, fetch_pdbs=fetch_pdbs),
        build_matcher=make_build_matcher(center_name=None),
        write_xyz=write_generic_xyz,
        has_metrics=False,
        family_doc=_FAMILY_DOC,
    )


PROFILE = make_profile()
