# Zn(Cys/His)₄ Structure Clustering Pipeline

Detailed reference for the clustering pipeline. Console scripts: `zch-pipeline`,
`zch-sample-spectra`, `zch-plot-spectra`, `zch-build-db` (each equivalent to
`python -m zn_cys_his...`). All commands run from the repo root via `uv run`.

## Quick command reference


All commands are run from the repo root with `uv run python`.
Scripts live in `src/zn_cys_his/clustering/`.

Works on any 4-coordinate Zn site — **4cys, 3cys1his, 2cys2his, 1cys3his, 4his**.
Composition is auto-detected from the input files; you don't specify it. See the
[Mixed Cys/His datasets](#mixed-cyshis-datasets) section for the one
input-path difference.

---

### One-command orchestrator (recommended)

```bash
# Full dataset — runs approaches 1 and 3 → validation → comparison
uv run python -m zn_cys_his.clustering.orchestrate \
    --base-dir data/4cys-large

# Include approach 2 (pure numpy; no extra dependencies)
uv run python -m zn_cys_his.clustering.orchestrate \
    --base-dir data/4cys-large \
    --approaches 1 2 3

# Small test dataset (k-min/max adjusted for 22 structures)
uv run python -m zn_cys_his.clustering.orchestrate \
    --base-dir data/test-4cys-weighted \
    --xyz-subdir initial_xyz_files \
    --approaches 1 3 \
    --k-min 2 --k-max 8

# Force re-run everything
uv run python -m zn_cys_his.clustering.orchestrate \
    --base-dir data/4cys-large \
    --force
```

Artifacts go to `cluster-output/4cys-large/` (mirroring `data/`); inputs are
never modified.  Override the root with `--out-dir`.

The run is split into three stages — **prep** (annotate + stats) → **cluster**
(the approaches) → **validate** — each independently rerunnable:

```bash
# rerun only one stage
uv run python -m zn_cys_his.clustering.orchestrate --base-dir data/4cys-large --stage cluster --approaches 3
uv run python -m zn_cys_his.clustering.orchestrate --base-dir data/4cys-large --stage validate

# run one stage and everything after it (skip prep)
uv run python -m zn_cys_his.clustering.orchestrate --base-dir data/4cys-large --from-stage cluster
```

---

### Mixed Cys/His datasets

Same scripts, same commands. The His datasets ship raw pocket XYZ
(`COORD=TRUE` tags mark the ligating residues); input is read as-is:

```bash
uv run python -m zn_cys_his.clustering.orchestrate \
    --base-dir data/2cys2his-large
```

**Automatic input filtering** (no flags needed):

1. `.pc` and `.gzmat` are ignored (only `*.xyz` is read).
2. `*-extended.xyz` duplicates are rejected — each His structure ships as both
   `*_Zn-extended.xyz` (full pocket) and `*_Zn.xyz` (coordinating cluster);
   they parse to the same site, so only the non-extended one is kept, leaving
   **one file per structure**.
3. **Mixed-ligand rejection** — a structure is dropped if any *coordinating*
   (`COORD=TRUE`) residue is neither Cys/His nor water. A 4-His site must
   coordinate Zn only through 4 His (SO4/GLU/IMP/inhibitor ligands →
   rejected). Coordinating water is tolerated (ignored). This mainly affects
   4his-large (catalytic zinc-enzyme sites with a 4th exchangeable ligand).
4. **Modal coordinating-atom-count filter** — counts the heavy atoms of the
   coordinating Cys/His residues *only* (by RESSEQ, + Zn). A structure is
   dropped only if its coordinating residues have a non-modal atom count
   (truncated / altloc — "not a true conformation").
5. **Majority-composition filter** — structures whose Cys/His makeup differs
   from the majority are dropped.

Each drop is reported, e.g. `19 structures dropped (extra coordinating ligand: GLU, SO4, …)`.
To override the extended rejection, pass `--xyz-glob '*-extended.xyz'` (a
pattern containing "extended" disables the auto-reject).

What changes automatically per composition:

| Composition | Atoms/site | A1 feature dim | Class-preserving perms |
|---|---|---|---|
| 4cys       | 13 | 39 | 4! = 24 |
| 3cys1his   | 17 | 51 | 3!·1! = 6 |
| 2cys2his   | 21 | 63 | 2!·2! = 4 |
| 1cys3his   | 25 | 75 | 1!·3! = 6 |
| 4his       | 29 | 87 | 4! = 24 |

- **Cys arm** = (SG, Cβ, Cα); **His arm** = full imidazole ring
  (CG, ND1, CD2, CE1, NE2) + (Cβ, Cα). His ligand = whichever of ND1/NE2 is
  closest to Zn.
- **Matching** keeps two classes (Cys, His): Cys never maps onto His; a His
  coordinating via ND1 *can* align onto one via NE2 (their difference shows up
  as RMSD), so a single global R₀ always exists.
- A dataset must be one composition; stray odd-composition files are dropped
  with a warning.

---

### Step-by-step

The orchestrator normally runs step01 (annotate) and step02 (stats) into
`cluster-output/<dataset>/prep/`, then the steps below read the annotated XYZ
from there.  All generated artifacts go under `cluster-output/` (mirroring
`data/`); inputs in `data/` are never modified.

#### step03 — Approach 1: aligned Cartesian

```bash
uv run python -m zn_cys_his.clustering.step03_approach1_cartesian \
    --xyz-dir  cluster-output/4cys-large/prep/annotated_xyz \
    --out-dir  cluster-output/4cys-large/approach1 \
    --k-min 15 --k-max 35
```

Outputs: `labels.csv`, `medoids.csv`, `k_sweep.csv`, `aligned_xyz/`, `r0_id.txt`

#### step04 — Approach 2: Z-matrix (pure numpy; no extra dependencies)

```bash
# From approach 1 aligned structures (recommended — avoids recomputing R₀)
uv run python -m zn_cys_his.clustering.step04_approach2_zmatrix \
    --aligned-xyz-dir cluster-output/4cys-large/approach1/aligned_xyz \
    --out-dir         cluster-output/4cys-large/approach2 \
    --k-min 15 --k-max 35

# Standalone (re-computes R₀ from raw XYZ)
uv run python -m zn_cys_his.clustering.step04_approach2_zmatrix \
    --xyz-dir  cluster-output/4cys-large/prep/annotated_xyz \
    --out-dir  cluster-output/4cys-large/approach2
```

#### step05 — Approach 3: PIV

```bash
uv run python -m zn_cys_his.clustering.step05_approach3_piv \
    --xyz-dir  cluster-output/4cys-large/prep/annotated_xyz \
    --out-dir  cluster-output/4cys-large/approach3 \
    --k-min 15 --k-max 35
```

#### step06 — Validate one approach

```bash
OUT=cluster-output/4cys-large

# Approach 1 — RMSD metrics + PCA→XYZ reconstruction
uv run python -m zn_cys_his.clustering.step06_validate_clusters \
    --approach-dir $OUT/approach1 \
    --xyz-dir      $OUT/approach1/aligned_xyz \
    --approach1 \
    --out-dir      $OUT/validation/approach1

# Approach 3 — RMSD metrics
uv run python -m zn_cys_his.clustering.step06_validate_clusters \
    --approach-dir $OUT/approach3 \
    --xyz-dir      $OUT/prep/annotated_xyz \
    --out-dir      $OUT/validation/approach3
```

#### step06 — Cross-approach comparison

```bash
uv run python -m zn_cys_his.clustering.step06_validate_clusters \
    --compare-dirs \
        cluster-output/4cys-large/approach1 \
        cluster-output/4cys-large/approach3 \
    --out-dir cluster-output/4cys-large/validation/comparison
```

---

### Standard output format (all approach scripts)

```
{out_dir}/
  labels.csv                    structure_id, cluster
  medoids.csv                   cluster_id, medoid_id
  k_sweep.csv                   k, intra, inter, ratio, ch_score
  per_cluster_intra.csv         cluster_id, mean_intra_rmsd
  kmeans_labels_with_stats.csv  labels joined with per-structure stats (needs stats CSV)
  cluster_pdb_family.csv        cluster, pdb_id, family (auto-emitted per approach)
  *_run.log                     detailed run log

approach1 additionally:
  aligned_xyz/            one 13-atom XYZ per structure (R₀ residue order)
  r0_id.txt               stem of the R₀ reference structure
```

**k selection criterion:** best k is the one that maximizes the CH-analogue score:
`ch_score = (inter² / (k−1)) / (intra² / (N−k))`.
This penalizes overly fine-grained k; the raw `ratio = inter/intra` is also recorded but
trends upward with k and is not used for selection.

---

### Key parameters

| Parameter | Default | Where |
|---|---|---|
| `--k-min` | 15 | 02, 03, 04 |
| `--k-max` | 35 | 02, 03, 04 |
| `--k-step` | 1 | 02, 03, 04 |
| `--convergence-tol` | 0.005 | 02 (R₀ iteration) |
| `--max-ref-iter` | 10 | 02 (R₀ iteration) |
| `--no-reflection` | off | 02, 03, 04 (off = merge enantiomers) |
| `--approaches` | 1 3 | 00 |
| `--force` | off | 00 |

---

## Pipeline details



---

### 0. Composition generality

The pipeline handles any 4-coordinate Zn site — **4cys, 3cys1his, 2cys2his, 1cys3his, 4his** — and
auto-detects the composition from the input files. The geometric machinery (Kabsch, PCA, k-means,
k-sweep, evaluation metric) is composition-agnostic; only two things vary with chemistry:

1. **Per-residue arm and ligand atom.** Cys arm = (SG, Cβ, Cα), ligand = SG. His arm = the full
   imidazole ring (CG, ND1, CD2, CE1, NE2) + (Cβ, Cα), ligand = whichever of ND1/NE2 is closest to
   Zn. This makes the number of heavy atoms per site composition-dependent (13 for 4cys … 29 for
   4his).
2. **Residue matching classes.** Two classes — Cys and His. Cys never matches His. His-ND1 and
   His-NE2 are the *same* class, so a His coordinating via ND1 can align onto one via NE2; their
   geometric difference surfaces as RMSD rather than forbidding the match. This keeps a single global
   R₀ possible even when a dataset mixes ND1/NE2 coordination (which they all do).

A pure-Cys(4) dataset reduces exactly to the original behaviour (13 atoms, 39-D, 4! = 24 matchings).
A run must be a single composition; stray odd files are dropped with a warning.

---

### 1. Goal

Cluster ~thousands of DFT-relaxed Zn coordination-site structures so that structures likely to
produce similar EXAFS spectra are grouped together. Clusters are then sampled to build an ML training
dataset with an even distribution of structurally distinct configurations, avoiding overrepresentation
of near-duplicates.

Each structure is Zn + its coordinating-residue arms. For a **Zn(Cys)₄** site this is **13 heavy
atoms**: Zn + 4 × (S, Cβ, Cα); His-containing sites are larger (the imidazole ring is kept). Each arm
terminates at Cα, fixed as the DFT boundary. Hydrogens are excluded throughout.

**Physical sensitivity hierarchy** (what drives spectral variation):
1. Zn–S distances — first-shell single scattering, dominant EXAFS signal
2. S–S distances and Zn–S–S angles — second-shell multiple scattering, tetrahedral distortion
3. Zn–Cβ, Zn–Cα — outer-shell contributions
4. Inter-residue cross-distances (S–Cα) — relative residue orientation

Key EXAFS physics note: even identical Zn–S distances can produce different spectra if the angular
geometry differs (e.g., "3+1" vs. ideal tetrahedral). Multiple-scattering sensitivity to angles is
established in Rehr & Albers (2000) and Trigub et al. (2014).

**Chirality:** EXAFS is achiral — a structure and its mirror image produce the same spectrum.
Enantiomers are merged everywhere (RMSD allows reflection; featurizations use chirality-insensitive
encodings). This is a deliberate design choice, not a limitation.

---

### 2. The Two Core Symmetry Problems

Every featurization approach must address these before clustering produces meaningful results.

#### Problem 1: Coordinate Frame

Structures from different DFT runs arrive in arbitrary lab-frame XYZ. Two physically identical
structures can have completely different coordinate arrays if one is rotated or translated.

**The near-tetrahedral complication:** Ideal T_d symmetry has λ₁ = λ₂ = λ₃ for the inertia tensor
(proportional to identity). For nearly-tetrahedral Zn(Cys)₄, eigenvalues are close but not exactly
equal, making principal-axis alignment numerically unstable — small structural perturbations can
cause large eigenvector rotations. This rules out inertia-tensor-based alignment.

#### Problem 2: Residue Ordering

Chemically equivalent residues can be listed in any order across input files. Even after perfect
rotational alignment, inconsistent listing order scrambles feature vectors. The joint solve
enumerates only **class-preserving** permutations — residues are permuted within their matching class
(Cys↔Cys, His↔His), never across. For 4cys this is all 4! = 24 orderings; for 2cys2his it is
2!·2! = 4; for 3cys1his, 3!·1! = 6. Cys and His are never interchanged.

**Why Zn–S distance sort fails:** All four Zn–S bonds are typically within ~0.05–0.1 Å of each
other — near-ties are the norm. A sort-based canonical order can pair wrong sulfurs between otherwise
identical structures, inflating RMSD by up to an S–S distance. This was the weak point of earlier
pipeline versions and is why the current pipeline uses a 24-permutation joint solve.

**Why 12-atom scaffold (S + Cβ + Cα) rather than 4 S atoms alone:** The four S atoms form an
approximately regular tetrahedron around Zn, which has 12 rotational symmetries — up to 12 of the
24 permutations yield near-identical RMSD values when aligning S only, making the choice arbitrary.
Including Cβ and Cα breaks this symmetry: the Cα–Cβ–S vector points in a unique direction for each
residue, giving each Cys a geometric fingerprint that makes the correct permutation unambiguous.

---

### 3. Script Inventory

| Script | Role |
|---|---|
| `orchestrate.py` | End-to-end driver; skips steps whose output exists unless `--force` |
| `step01_annotate_secstruct.py` | Tag CA atoms with SEC= (HELIX/SHEET/LOOP) from the matching PDB |
| `step02_compute_stats.py` | Per-structure metadata + family string (uses SEC= from step01) |
| `utils.py` | Shared library: `Structure`, parsing, Kabsch, RMSD, clustering, k sweep |
| `step03_approach1_cartesian.py` | Approach 1: fixed-point R₀ alignment → aligned Cartesian clustering |
| `step04_approach2_zmatrix.py` | Approach 2: Z-matrix via chemcoord (requires `uv add chemcoord`) |
| `step05_approach3_piv.py` | Approach 3: PIV (sorted pairwise distances; fully invariant) |
| `step06_validate_clusters.py` | Validation: RMSD metrics, k-sweep plots, PCA→XYZ, cross-approach comparison |

#### Pipeline flow

```
raw XYZ files (xyz-files/)
    |
    v
step01_annotate_secstruct.py → step02_compute_stats.py → structure_stats.csv
    |
    +------------------------------------------+
    |                                          |
    v                                          v
step03_approach1_cartesian.py          step05_approach3_piv.py
(fixed-point R₀ → aligned Cartesian)  (PIV; fully invariant)
→ approach1/                       → approach3/
  labels.csv, medoids.csv,            labels.csv, medoids.csv,
  k_sweep.csv, aligned_xyz/           k_sweep.csv
    |
    v
step04_approach2_zmatrix.py  [optional; needs chemcoord]
(Z-matrix on R₀-labeled aligned structures)
→ approach2/
  labels.csv, medoids.csv, k_sweep.csv
    |
    +------------------------------------------+
    |                                          |
    v                                          v
step06_validate_clusters.py            step06_validate_clusters.py
(per-approach RMSD metrics + plots) (--compare-dirs; cross-approach)
→ validation/approach{1,2,3}/      → validation/comparison/
```

---

### 4. Script Details

#### `orchestrate.py`

Stage runner (prep → cluster → validate). Skips work whose output exists unless
`--force`.

Key arguments:
```
--base-dir        read-only input dataset dir under data/ (e.g. data/4cys-large)
--out-dir         artifact root (default: cluster-output/<dataset>, mirrors data/)
--xyz-subdir      raw XYZ subdir under --base-dir (default: xyz-files)
--stage           run ONLY this stage: prep | cluster | validate
--from-stage      run this stage and every stage after it
--approaches      which to run: 1 2 3 or any subset (default: 1 3)
--k-min/max/step  k sweep range (default: 15..35 step 1)
--weight-scheme   RMSD atom weighting: distance (default), equal, shell
--force           re-run all steps
```

#### `step01_annotate_secstruct.py`

Tags each CA atom in the XYZ comment line with `SEC=HELIX|SHEET|LOOP`, read from
the HELIX/SHEET records of the matching `<pdbid>.pdb` in `pdb-files/`. Runs in
place; a `.secstruct_annotated` sentinel marks completion for idempotent skips.

#### `step02_compute_stats.py`

Computes per-structure metadata (geometry, B-factors, resolution/R-factors from
the PDB, tetrahedricity, dihedrals) plus a `family` string that folds in the
secondary-structure tags from step01. Writes `structure_stats.csv` next to the
XYZ files.

#### `utils.py`

Shared library, not a standalone script. Key exports:

- `Structure` dataclass — `id`, `zn (3,)`, `residues: list[Residue]` (canonical order: Cys before His, then by RESSEQ). Each `Residue` carries its `res_type`, `coord_atom` (the Zn ligand), and its arm `coords`. `structure.heavy()` returns the (M,3) array [Zn | arm₀ | arm₁ | …]. `composition()` gives the class signature; `class_preserving_perms(res_types)` enumerates the legal matchings.
- `parse_structure(path)` — reads XYZ using `ATOM=` EOL tags (authoritative; connectivity not used)
- `write_structure_xyz(structure, path)` — writes 13-atom XYZ
- `weighted_kabsch(P, Q, w, allow_reflection)` — SVD-based Kabsch superposition with optional reflection
- `structural_rmsd(A, B, w_type, allow_reflection)` — matching-minimized RMSD over all 24 residue permutations (see §5)
- `cluster_pipeline(X, k, ...)` — z-score with variance floor → PCA 95% → k-means
- `evaluate_clustering(structures, labels, ...)` — computes inter/intra RMSD ratio + CH-analogue score
- `sweep_k(structures, X, k_values, ...)` — k sweep; best result selected by max `ch_score`
- `save_outputs(out_dir, id_list, k_sweep_table, best)` — writes standard CSVs

#### `step03_approach1_cartesian.py`

Implements Approach 1 (Aligned Cartesian):
- Parses XYZ using `ATOM=` tags
- Runs fixed-point R₀ iteration: align all structures to current R₀ (joint class-preserving permutation search), recompute R₀ as dataset PCA medoid, repeat until convergence
- Features: 39-D flattened aligned Cartesian `[Zn | S0..3 | Cb0..3 | Ca0..3]`
- Writes aligned XYZ to `aligned_xyz/` (consumed by Approach 2)

#### `step04_approach2_zmatrix.py`

Implements Approach 2 (Z-matrix):
- Reads aligned XYZ from Approach 1 (`--aligned-xyz-dir`) or raw XYZ and recomputes R₀
- Z-matrix via chemcoord; one construction table built from R₀, reused for all structures (mandatory for column consistency — see §6)
- Features: bond lengths r (raw Å), bond angles θ (raw °), cos(φ) for dihedrals (chirality-insensitive)
- Requires `uv add chemcoord`

#### `step05_approach3_piv.py`

Implements Approach 3 (PIV):
- Fully independent of R₀; reads raw XYZ
- 78-D PIV: sorted pairwise distances in 9 atom-pair-type blocks
- Near-constant covalent-bond columns handled by variance floor in z-scoring (not explicit exclusion)

#### `step06_validate_clusters.py`

**Single approach** (`--approach-dir`):
- k-sweep ratio curve + CH-analogue score plot
- Cluster sizes bar chart
- Per-cluster RMSD scatter (requires `--xyz-dir`)
- RMSD table CSV
- PCA→XYZ reconstruction (`--approach1` flag; Approach 1 only — Cartesian features map directly to atom motions)

**Cross-approach comparison** (`--compare-dirs`):
- Reads `k_sweep.csv` from each approach directory
- Comparison table CSV + bar chart of inter/intra ratios

---

### 5. Shared Evaluation Metric

All three approaches use the same RMSD-based metrics, making the comparison fair.

#### Structural RMSD (the distance function)

`structural_rmsd(A, B)` — minimum best-fit RMSD over all **class-preserving** residue matchings (Zn
fixed; atoms within a residue never permuted), with reflection allowed. The matching count is 24 for
4cys, 4 for 2cys2his, 6 for 3cys1his, etc. Cost per pair: (n_perms)×2 tiny 3×3 SVDs. If A and B have
different composition the RMSD is infinite (they cannot be matched).

**Why not shortcut to a canonical ordering here:** even after the featurization-level ordering, the
four Zn–S distances are near-degenerate, so a sort-based canonical order can pair wrong sulfurs,
inflating RMSD by up to an S–S distance. The 24-permutation enumeration makes the metric exact and
independent of any approach's ordering — required for a fair cross-approach comparison.

**Atom weights (`--weight-scheme`):** Three presets are available, selected via `--weight-scheme`:

| Scheme | Description |
|---|---|
| `distance` (default) | weight = 1 / avg_Zn_distance per atom pair — atoms closer to Zn are weighted higher; Zn itself gets weight 1. |
| `equal` | all atoms weighted uniformly (Zn=S=Cβ=Cα=1). |
| `shell` | coordinating atom=1, all other arm atoms=0.5 — shell-distance proxy without actual geometry. |

For `distance` weighting, each atom slot's weight is computed per structure pair *and* per residue permutation: D = (dist(A_atom, A.Zn) + dist(B_atom_perm, B.Zn)) / 2, weight = 1/D. Since permuting B's residues changes which atoms occupy each slot, weights are recomputed for each class-preserving permutation. Atoms close to Zn (ligand at ~2 Å) get the highest weight; Cα at ~4 Å gets the lowest.

#### Medoid, Intra, Inter

- **Medoid:** member nearest the cluster centroid in PCA space (not the raw coordinate centroid)
- **Intra:** mean `structural_rmsd(medoid, member)` per cluster, then unweighted arithmetic mean across clusters
- **Inter:** mean `structural_rmsd(medoid_i, medoid_j)` over all medoid pairs
- **Ratio:** `inter / intra` (higher = better)

#### Calinski–Harabasz Analogue Score (primary k-selection metric)

Raw `inter/intra` increases monotonically with k because intra shrinks as clusters get smaller,
regardless of cluster quality. The CH-analogue normalizes for this:

```
ch_score = (inter² / (k − 1)) / (intra² / (N − k))
```

A high score means clusters are simultaneously well-separated (large inter) and compact (small intra)
relative to what k alone predicts. The score peaks at a genuinely good k rather than simply at the
maximum tested k. The raw ratio is still recorded in `k_sweep.csv` for reference.

#### k grid

`k ∈ {15, 16, 17, …, 35}` (step 1). Each approach selects its own optimal k.

---

### 6. The Three Featurization Approaches

#### Approach 1 — Aligned Cartesian

**Features:** 39-D flattened Cartesian coordinates `[Zn | S0..3 | Cβ0..3 | Cα0..3]` after alignment
to a global reference R₀.

**How R₀ is established (fixed-point iteration):**
1. Initialize R₀ as any structure.
2. `align_to_reference`: for each of the 24 residue permutations π of each structure B, compute
   weighted Kabsch onto R₀ and record RMSD. Keep (π\*, R\*, t\*) with minimum RMSD. Apply the
   winning transform — if a reflection wins, it is applied (collapses enantiomers onto R₀'s handedness).
3. Z-score → PCA 95% on the aligned 39-vectors.
4. Set R₀ ← the structure nearest the PCA-space centroid (the dataset medoid).
5. Stop when R₀ is unchanged **and** < 0.5% of structures change their matching. Cap at 10 iterations.

**Why iterate rather than one-shot:** R₀ shifts as alignment improves — the PCA space and its medoid
change each iteration. Typically converges in 2–5 iterations.

**Why joint rotation+matching enumeration rather than rotate-then-match:** Kabsch needs a correspondence
to produce a rotation; they are mutually dependent. With only 24 matchings the exact joint optimum is
found by trying all of them. ICP-style alternation (rotate from a guess, assign nearest, re-rotate)
risks local minima and is unnecessary at n=4.

**DoF diagnostic:** log PCA component count at 95%; expect ~25. Substantially more indicates labeling
scrambling (35 = 3×13 − 6 internal DoF; minus ~8 near-constant covalent bonds ≈ 25 variable DoF).

**Pros/cons:**

| | |
|---|---|
| ✅ | Invertible — PCA components map directly to atom positions |
| ✅ | Compact (39-D) |
| ✅ | Global R₀ is data-derived, not external |
| ✅ | Joint matching robust to near-tetrahedral Zn–S degeneracy |
| ✅ | Reflection applied — enantiomers merged |
| ⚠️ | Fixed-point iteration (small overhead) |
| ⚠️ | Features are relative to R₀; different datasets produce different R₀ |

---

#### Approach 2 — Z-Matrix (Internal Coordinates)

**Features:** Bond lengths r (Å, raw), bond angles θ (°, raw), and cos(φ) for dihedrals — using
chemcoord's Z-matrix representation with a single construction table built from R₀.

**Procedure:**
1. Reorder each structure's residues by the global R₀ labeling from Approach 1.
2. Build a single `construction_table` once from R₀ (`chemcoord.Cartesian.get_zmat()`). Reuse the same
   table for every structure — this is mandatory. chemcoord can infer different connectivity tables
   per structure, breaking column correspondence.
3. Extract r, θ, and cos(φ) per atom.

**Why cos(φ) only for dihedrals:** `cos` is even — invariant under φ → −φ (reflection). Dropping
`sin(φ)` deliberately discards handedness, merging enantiomers as required.

**Dependency on Approach 1:** needs only the global residue labeling, not the rotational alignment.
Internal coordinates are rotation/translation invariant by construction. If chemcoord is not installed,
Approach 2 is silently skipped.

**Pros/cons:**

| | |
|---|---|
| ✅ | Rotation/translation invariant by construction |
| ✅ | Directly encodes bonds, angles, dihedrals — physically interpretable |
| ✅ | cos(φ) encoding merges enantiomers |
| ⚠️ | Requires `chemcoord` |
| ⚠️ | Construction table must be built once from R₀ and reused |

---

#### Approach 3 — Permutation Invariant Vector (PIV)

**Features:** 78-D vector of all pairwise Euclidean distances, grouped into 9 atom-pair-type blocks
and sorted ascending within each block.

| Block | Count |
|---|---|
| Zn–S | 4 |
| Zn–Cβ | 4 |
| Zn–Cα | 4 |
| S–S | 6 |
| S–Cβ | 16 |
| S–Cα | 16 |
| Cβ–Cβ | 6 |
| Cβ–Cα | 16 |
| Cα–Cα | 6 |

**Key decisions:**
- **Block partitioning** by atom-pair type (not a single global sort) — preserves physical structure.
- **Raw distances**, no switching function — z-scoring handles dynamic range; switching functions
  saturate large distances and harm reconstruction.
- **Retain covalent-bond distances** in the full 78-vector — needed for reconstruction.
- **Near-constant covalent-bond columns** (S–Cβ ≈ 1.82 Å, Cβ–Cα ≈ 1.52 Å, sub-0.01 Å variation):
  handled by variance floor (std clamped to ≥ 1e-3 Å) in z-scoring, not explicit exclusion.
  Explicitly excluding them would break the reconstruction; the floor is the minimal intervention.
- **Fully independent of R₀** — no reference structure, no residue ordering needed.

**Angular information** is encoded implicitly via cross-distances (S–Cα, S–S capture the tetrahedral
geometry; different S–S patterns diagnose geometry type):

| Geometry | S–S pattern |
|---|---|
| Ideal tetrahedral | All 6 equal (~3.8 Å) |
| 3+1 (one outlier S) | 3 short + 3 long |
| Square planar | 4 short + 2 long |
| Irregular | All 6 different |

**Reconstruction (optional):** PIV sorting destroys per-atom labels, so back-conversion requires
re-deriving atom types via chemistry (covalent bond lengths anchor the assignment), assembling a
candidate distance matrix, and running classical MDS. ~80% useful, not required — medoid structures
can always be visualized directly since cluster membership is tracked by structure ID.

**Pros/cons:**

| | |
|---|---|
| ✅ | Fully invariant to rotation, translation, residue permutation by construction |
| ✅ | No reference structure needed |
| ✅ | Numerically stable — no eigenvalue degeneracy |
| ✅ | Preserves all geometric information except chirality |
| ⚠️ | 78-D (PCA reduces this in practice) |
| ⚠️ | PCA components cannot be mapped directly back to atom positions |
| ⚠️ | Reconstruction requires a dedicated inverse routine |

---

#### Approach 4 — SOAP/ACE (Deprioritized)

Systematic spherical-harmonic × radial-basis expansion. Mathematically guaranteed invariance.
Deprioritized for this system: (a) high capability relative to a small fixed-topology 13-atom system;
(b) not invertible; (c) requires additional hyperparameter tuning.
References: Bartók et al. (2013), Drautz (2019), Himanen et al. (2020, DScribe).

---

### 7. Approach Comparison

| Property | Approach 1: Aligned Cartesian | Approach 2: Z-Matrix | Approach 3: PIV |
|---|---|---|---|
| Removes translation | ✅ | ✅ | ✅ |
| Removes rotation | ✅ (aligned to global R₀) | ✅ (by construction) | ✅ (by construction) |
| Resolves residue ordering | ✅ (joint matching to R₀) | ✅ (via R₀ labeling) | ✅ (by construction) |
| Chirality insensitive | ✅ (reflection applied) | ✅ (cos φ only) | ✅ |
| Stable for near-T_d | ✅ (24-permutation robust) | N/A | ✅ |
| Invertible / interpretable | ✅ full | ✅ full | ❌ not directly |
| PCA → real structures | ✅ | ✅ (with care) | ❌ |
| External reference needed | ❌ (data-derived R₀) | ⚠️ (R₀ labeling required) | ❌ |
| Feature dimensionality | 39 (3n) | 33 (3n−6) | 78 (n(n−1)/2) |
| Encodes angles explicitly | ✅ | ✅ | ⚠️ implicit via cross-dist |

---

### 8. Shared Pipeline Details

#### XYZ file format

Input files must have `ATOM=` tags in end-of-line comments:
```
ZN   0.000  0.000  0.000  # ATOM=ZN ...
S   -1.918  1.153 -1.048  # ATOM=SG ...
C   -2.604  1.790  0.524  # ATOM=CB ...
C   -3.887  2.627  0.400  # ATOM=CA ...
```
Tags `ATOM=ZN`, `ATOM=SG` (or `S`), `ATOM=CB`, `ATOM=CA` are authoritative. Hydrogen atoms (any
other tag) are ignored. Connectivity is not inferred from geometry except in PIV reconstruction.

#### Clustering pipeline (all approaches)

1. **Z-score** each column. Variance floor: `std = max(std, 1e-3 Å)` — avoids amplifying noise in
   near-constant covalent-bond columns.
2. **PCA**, retain 95% cumulative variance. Expected ~25 components (see DoF diagnostic).
3. **k-means** in PCA space: fixed `random_state=0`, `n_init ≥ 10`.

#### Output directory layout

Inputs are read-only under `data/<dataset>/`; everything below is generated
under the mirror output root `cluster-output/<dataset>/` (override: `--out-dir`).

```
data/<dataset>/               (read-only inputs)
  xyz-files/                  raw pocket XYZ
  pdb-files/                  source <pdbid>.pdb

cluster-output/<dataset>/     (generated; wipe with rm -rf)
  prep/
    annotated_xyz/            SEC=-annotated COPIES of the input XYZ (step01)
    structure_stats.csv       per-structure metadata + family (step02)
  approach1/
    labels.csv                structure_id → cluster
    medoids.csv               cluster_id → medoid_id
    k_sweep.csv               k, intra, inter, ratio, ch_score
    per_cluster_intra.csv
    kmeans_labels_with_stats.csv
    cluster_pdb_family.csv    cluster, pdb_id, family (auto)
    aligned_xyz/              one 13-atom XYZ per structure (R₀ order)
    r0_id.txt                 R₀ structure stem
    approach1_run.log
  approach2/                  same format; optional
  approach3/                  same format; no aligned_xyz
  validation/
    approach{1,2,3}/
      k_sweep_plot.png
      cluster_sizes.png
      rmsd_scatter.png
      rmsd_table.csv
      pca_to_xyz/             mean.xyz, pc1_plus.xyz, pc1_minus.xyz, …
    comparison/
      comparison_table.csv
      comparison_plot.png
```

#### Degrees of freedom reference

- 3 × 13 = 39 Cartesian DoF
- −3 translations, −3 rotations → **33 internal DoF**
- −8 near-constant covalent bonds (S–Cβ and Cβ–Cα × 4 residues) → **~25 variable DoF**
- PCA at 95% is expected to retain ~25 components. Substantially more → labeling scrambled.

#### Logging and terminal output

- Terminal text is fine before/after a progress bar — never during one (no per-line prints inside
  active tqdm loops; use `tqdm.write(...)` if something must surface mid-loop).
- Per-item warnings are collected and summarized after the bar, not streamed per structure.
- Detailed status goes to a log file (Python `logging` with `FileHandler`; no console `StreamHandler`
  inside loops). One readable text log + machine-parseable JSONL mirror.
- R₀ convergence trace format per iteration:
  ```
  iter=01  R0=struct_0427  matching_changed= 312/3000 (10.40%)  refl_flipped= 88  pca_dims=27  centroid_shift=0.0421
  iter=02  R0=struct_0613  matching_changed=  41/3000 ( 1.37%)  refl_flipped= 11  pca_dims=25  centroid_shift=0.0068
  iter=03  R0=struct_0613  matching_changed=   9/3000 ( 0.30%)  refl_flipped=  2  pca_dims=25  centroid_shift=0.0011  [CONVERGED]
  ```

---

### 9. Parameters (all set unless noted)

| Component | Parameter | Value | Notes |
|---|---|---|---|
| Pipeline | normalization | z-score | variance floor 1e-3 Å |
| Pipeline | PCA variance | 0.95 | |
| Pipeline | clustering | k-means | n_init ≥ 10, fixed seed |
| Metric | medoid | PCA-centroid-nearest | |
| Metric | intra aggregation | unweighted arithmetic mean | |
| Metric | k-selection objective | max ch_score | raw ratio also saved |
| Metric | RMSD correspondence | matching-minimized (24) + reflection | |
| Metric | atom weights | `distance` (1/avg_Zn_distance; default) | `equal` and `shell` also available via `--weight-scheme` |
| Sweep | k grid | 15–35 step 1 | |
| A1 | alignment | joint rotation+matching to global R₀ (24-enumeration) | |
| A1 | R₀ method | dataset medoid in PCA space, fixed-point | fallback: Zn–S sort |
| A1 | convergence_tol | 0.005 | < 0.5% structures change matching |
| A1 | max_ref_iter | 10 | |
| A1 | reflection | allowed and applied | merges enantiomers |
| A1 | per-cluster tightening | off (open) | rotation-only, labels fixed |
| A2 | construction table | built once from R₀, reused | |
| A2 | dihedral encoding | cos(φ) only | chirality-insensitive |
| A3 | distances | raw, block-sorted ascending | |
| A3 | covalent bonds | retained; variance-floored for clustering | |
| A3 | reconstruction | optional | ~80% useful, not required |
| Global | chirality | enantiomers merged | reflection allowed everywhere |
| Global | atom typing | from `ATOM=` EOL tags | connectivity only in PIV reconstruction |

---

### 10. Tests and Validation

1. **Invariance unit tests** — for each featurization, assert feature vector is unchanged under:
   random rotation, translation, residue relabeling, and reflection (for A3 and the metric; A1/A2
   after their chirality-insensitive encodings).
2. **Identity RMSD** — `structural_rmsd(A, A) ≈ 0`; `structural_rmsd(A, rotate(A)) ≈ 0`;
   `structural_rmsd(A, mirror(A)) ≈ 0`; and `structural_rmsd` of two relabeled-but-identical
   structures ≈ 0 (guards the residue-ordering degeneracy bug).
3. **DoF diagnostic** — PCA component count at 95% logged each R₀ iteration; expect ~25.
4. **Reconstruction (A3, if built)** — round-trip `piv_reconstruct(approach3_piv(s))` on known
   structures; report reconstruction RMSD.
5. **Determinism** — fixed seeds reproduce labels.

---

### 11. Dependencies

`numpy`, `scipy` (SVD, eigh), `scikit-learn` (PCA, KMeans), `chemcoord` (Approach 2 only; silently
skipped if absent), `tqdm` (progress bars). Standard library `logging` for file-based logging.
No GPU required; full run is CPU-bound and tractable for N ≈ 3000.

---

### 12. References

| Reference | DOI | Relevance |
|---|---|---|
| Kabsch (1976), *Acta Cryst.* A32, 922–923 | https://doi.org/10.1107/S0567739476001873 | Optimal rotation |
| Kabsch (1978), *Acta Cryst.* A34, 827–828 | https://doi.org/10.1107/S0567739478001680 | Kabsch extension |
| Weser, Hein-Janke & Mata (2023), *J. Comput. Chem.* 44, 710–726 | https://doi.org/10.1002/jcc.27029 | chemcoord (Approach 2) |
| Parsons et al. (2005), *J. Comput. Chem.* 26, 1063–1068 | https://doi.org/10.1002/jcc.20237 | SN-NeRF Z-matrix → Cartesian |
| Gallet & Pietrucci (2013), *J. Chem. Phys.* 139, 074101 | https://doi.org/10.1063/1.4818005 | PIV — primary reference for Approach 3 |
| Hoffmann & Noé (2019), arXiv:1910.03131 | https://arxiv.org/abs/1910.03131 | Distance-matrix representations |
| Bartók, Kondor & Csányi (2013), *Phys. Rev. B* 87, 184115 | https://doi.org/10.1103/PhysRevB.87.184115 | SOAP descriptor |
| Drautz (2019), *Phys. Rev. B* 99, 014104 | https://doi.org/10.1103/PhysRevB.99.014104 | ACE framework |
| Himanen et al. (2020), *Comput. Phys. Commun.* 247, 106949 | https://doi.org/10.1016/j.cpc.2019.106949 | DScribe (SOAP/ACSF) |
| Trigub et al. (2014), *J. Surf. Investig.* 8, 20–27 | https://doi.org/10.1134/S1027451014010170 | EXAFS angular sensitivity |
| Rehr & Albers (2000), *Rev. Mod. Phys.* 72, 621 | https://doi.org/10.1103/RevModPhys.72.621 | EXAFS multiple-scattering theory |
| Herringer et al. (2024), *J. Chem. Theory Comput.* 20, 178–198 | https://doi.org/10.1021/acs.jctc.3c00923 | PINES: PIV + neural networks |
