# Phase-1 Implementation Plan: Sidechain-Aware Cross-Residue Distance Features

(Returned by Plan agent on 2026-04-28. Does NOT include Phase 0 — UniProt /
90%-identity-cluster fold dedup + UCBShift-200 exclusion — which was raised
after the agent was launched. That is layered in at the bottom of this doc.)

## 0. Findings from code reading that change the previous plan

- `DistanceAttentionPerPosition` (`model.py:53–117`) processes a tensor of shape `(B, W, M)` where `W` is the window size. There is **no notion of "offset"** — every pair is treated as belonging to one residue position. The model has no offset embedding to extend.
- The spatial-neighbor pathway already implements a **joint cross-residue attention scaffold** (`model.py:633–681`): it concatenates the query's intra-residue distances + a single CA–CA distance + the neighbor's intra-residue distances into one set of pairs and runs them through the *same* `DistanceAttentionPerPosition` module. This is the existing pattern we should extend rather than rebuild from scratch.
- The atom vocabulary (`config.py:61–76`) is hard-coded to the 88 PDB atom names. There are no synthetic atom slots, so adding ring/centroid pseudo-atoms is purely additive.
- The current dist-array storage uses **`int16` for atom indices** (vocab is 88 < 32k) and **`float16` for distances**. Adding an `int8` offset column is cheap.
- Per-fold structural storage in v2 is `~1.2 GB` (3 × 408 MB for hybrid 534k residues, M=400). The AlphaFold dedup cache has **2.05M residues per fold** — 3 × 1.6 GB = ~4.8 GB intra-only, currently. Cross-features add proportionally.
- `data/cache` symlinks to `/home/brooks/1TB/data_main_cache/cache` (a *legacy* hybrid cache dir, NOT v2). The v2 caches at `data/struct_retrieval_v2/cache/fold_*/` are addressed via the explicit `data/struct_retrieval_v2/` path. **The Phase-1 cache rebuild target is `data/struct_retrieval_v2/cache/fold_*/`** and the AF struct-only equivalent at `/home/brooks/1TB/Wynn/data_alphafold_cache_dedup/fold_*/`.
- `inference.py` currently parses each `dist_*` key from a per-residue dict (`inference.py:243–255`). It has no concept of cross-residue pairs. The hook point is `extract_features_from_pdb`.
- `parse_distance_columns` in `dataset.py:54–65` has a regex that only matches `dist_{A1}_{A2}` (no offsets). The cache builder reads pre-computed `dist_*` columns from the CSV. We will NOT add cross-distance columns to the 7 GB CSV — we will compute them at cache-build time from the parsed PDB, preserving the existing CSV.
- The CSV-based pipeline is already a memory bottleneck (1500+ distance columns). We must avoid expanding it. **Cross-residue features are computed from PDB coords directly during cache build, not from CSV columns.** This requires re-parsing PDBs in `05_build_training_cache.py` — a new step.

The previous plan's "Option A vs Option B" framing assumed offsets attached to existing intra arrays. Given (a) the cross-pair count varies wildly (small loop residue ≈ 0 cross, dense core residue ≈ 200+) and (b) we need to query both window-cross AND spatial-neighbor-cross, **a separate `cross_*` array set is the correct choice**.

---

## 1. Architecture decision: **Option B (separate cross arrays)** — recommendation

### Why Option B beats Option A

- **Sparsity differs by 5–10×.** Most residues have ~10–40 intra-pairs but 0–200 cross-pairs depending on solvent exposure. Packing them together forces M to grow to ~600 to safely cover the long tail, wasting `400 × n_residues × ~6 bytes` for the residues that don't need it. Separate arrays let us cap cross-distances at `M_CROSS = 200` independently.
- **Inference simpler.** `extract_features_from_pdb` already iterates per-residue. Separate arrays mean the cross-pair computation is a separate pass, easy to write, easy to unit-test for parity.
- **No collision with existing `flat_dist_count`.** `dist_count` already drives masking in `__getitem__`. A `flat_cross_count` keeps that contract intact for any code that still reads only intra (e.g. retrieval-side neighbor encoder which uses `flat_query_struct`, not raw distances).
- **The model already concatenates pair sets** at `model.py:660–663`. Adding cross-pairs to that joint concat costs one extra `torch.cat` and one extra `int8`-embed lookup for offset.

### What concrete code changes for Option B

**New cache arrays (per residue, fixed shapes):**

| array | shape | dtype | bytes/row |
|---|---|---|---|
| `cross_atom1.npy` | `(N, M_CROSS)` | `int16` | 400 |
| `cross_atom2.npy` | `(N, M_CROSS)` | `int16` | 400 |
| `cross_offset.npy` | `(N, M_CROSS)` | `int8` (signed: −5..+5 for window, 100+k for spatial-k) | 200 |
| `cross_values.npy` | `(N, M_CROSS)` | `float16` | 400 |
| `cross_count.npy` | `(N,)` | `int16` | 2 |

`M_CROSS = 200`. Offset encoding: window neighbors use signed offset −5..+5 (excluding 0). Spatial-neighbor cross pairs use a separate value space `100, 101, 102, 103, 104` (one per spatial slot k=0..4). This keeps a single `int8` column and a single embedding lookup distinguishing all 15 contexts.

**`config.py`:**
```
MAX_CROSS_DISTANCES = 200
CROSS_DIST_CUTOFF = 8.0  # Å — only store cross-pairs at distance ≤ 8 Å
N_CROSS_OFFSET_TYPES = 16  # 11 window (offset −5..+5) + 5 spatial slots; padding idx = 16
```

**`distance_features.py`:** add `calc_cross_residue_distances(atoms_self, atoms_other, atom_to_idx, cutoff)` returning a list of `(a1_idx, a2_idx, distance)` tuples. Iterates heavy×heavy plus H→heavy, drops pairs `> cutoff`, returns sorted ascending by distance (priority pruning).

**`05_build_training_cache.py`:** in the per-protein loop, after parsing per-residue atom dicts, compute cross-pairs per (residue, ±5 window neighbor) and (residue, k=0..4 spatial neighbor). Pack into `flat_cross_*` arrays. This is the new heavy step.

**PDB re-parsing requirement.** The current cache builder reads `dist_*` columns from the CSV — it does *not* re-open PDB files. To compute cross-pairs we now must re-parse PDBs. Pick **(B-clean)**: modify `05_build_training_cache.py` to load PDBs alongside the CSV, using `pdb_utils.parse_pdb`. Runtime cost: `~2453 proteins × 0.5 s = ~20 minutes` per fold (PDB parsing is fast).

**`dataset.py`:**
- `_load_structural_data` adds 5 new mmaps for `cross_*` arrays.
- `__getitem__` returns `cross_atom1_idx`, `cross_atom2_idx`, `cross_offset`, `cross_distances`, `cross_dist_mask` for the center residue (W=11 window-cross + spatial-cross both attach to the center). Shape `(M_CROSS,)` per residue.
- Window neighbors (`safe_idx[w]`) do **not** get their own cross arrays in Phase 1. The center residue's cross-pairs already capture both directions of `i↔i±k` because we store both endpoints per residue.

**`model.py` — `DistanceAttentionPerPosition`:**
- Add `nn.Embedding(N_CROSS_OFFSET_TYPES + 1, embed_dim, padding_idx=N_CROSS_OFFSET_TYPES)` as `self.offset_embed`.
- Accept an optional `offset_idx` argument (shape matching `atom1_idx`). For intra-pairs, pass a tensor filled with a special "intra" offset id (reserve offset id 0 for intra; shift window offsets to 1..11 = signed −5..+5; spatial slots = 12..16).
- Adjust `input_dim` from `embed_dim*2 + 1` to `embed_dim*2 + embed_dim_offset + 1` where `embed_dim_offset = 8` (small).
- All existing call sites pass an `intra` filled tensor; the new cross call passes the actual offsets.

**`model.py` — joint attention block (lines 633–681):**
- Compute the center distance attention with both intra AND cross pairs concatenated as one `(B, 1, M + M_CROSS)` set.
- Leave the spatial-neighbor joint attention block unchanged (it still uses intra×2 + CA-CA).

This is the minimal-touch path: cross-features feed the center residue only, where the prediction target lives. Window non-center positions still get only intra-pairs (their CNN feature aggregation already mixes context spatially).

### Concrete code locations to edit (Option B)

1. `config.py` — add `MAX_CROSS_DISTANCES`, `CROSS_DIST_CUTOFF`, `N_CROSS_OFFSET_TYPES`. Lines around 179.
2. `distance_features.py` — new function `calc_cross_residue_distances`. End of file.
3. `05_build_training_cache.py` — new PDB-loading helper + new arrays inside `build_cache_for_fold` (around line 247 alloc and line 336 dist-loop) and saving (line 459).
4. `dataset.py` — extend `_load_structural_data` (line 223), `__getitem__` (line 325) to return cross tensors.
5. `model.py` — extend `DistanceAttentionPerPosition.__init__` and `.forward` (line 53–117); use cross tensors at center position in `ShiftPredictor.forward` (around line 605, just before/after `dist_emb = ...`).
6. `inference.py` — in `extract_features_from_pdb` (line 123): after computing intra `all_dist_features`, compute cross-pairs per-residue using parsed atom dicts and pack identically to the cache.
7. `train.py` — collation just `**batch` works; no change needed because the cross tensors are added to the dict by `__getitem__`.

---

## 2. Atom vocabulary changes (Phase 2 — defer to after Phase 1 lands)

Phase 1 uses **only existing 88 atom types**. Cross-pairs are real heavy×heavy and H×heavy pairs between the center and neighbors. No vocabulary change is required for Phase 1 alone.

For Phase 2 (centroids), reserve the additions ahead of time so a future cache rebuild can include them without atom-id renumbering. Append to `ATOM_TYPES` after index 87 (do not insert):

```
'PHE_RING', 'TYR_RING', 'TRP_RING5', 'TRP_RING6', 'HIS_RING',  # ring centroids
'ASP_COO', 'GLU_COO',                                            # carboxylate centroids
'ASN_AMIDE', 'GLN_AMIDE',                                        # amide N–C–O centroids
'ARG_GUAN',                                                       # guanidinium
```

**Decision: bump `N_ATOM_TYPES` to 98 in Phase 1** even though Phase-1 caches will never emit IDs 88–97. This avoids two checkpoint-compatibility breaks.

The offset embedding integrates trivially: it's a separate embedding concatenated alongside `atom1_emb` and `atom2_emb`. Padding-idx handling is identical to atom embedding.

---

## 3. Cache schema delta

### New files per fold (added to `cache/fold_k/structural/`)

```
cross_atom1.npy    int16  (N, 200)
cross_atom2.npy    int16  (N, 200)
cross_offset.npy   int8   (N, 200)    # offset codes: 0=intra (unused here), 1..11=window, 12..16=spatial slot, 17=padding
cross_values.npy   float16 (N, 200)   # already-divided-by-10, clipped to cutoff/10
cross_count.npy    int16  (N,)
```

Sizes per fold:

| dataset | N_residues | bytes/row | per-fold delta |
|---|---|---|---|
| hybrid v2 | 534,569 | 1402 | **0.75 GB** |
| AF dedup | 2,054,000 | 1402 | **2.88 GB** |

### 5-fold totals

- Hybrid v2: **3.7 GB** new cache data
- AF dedup: **14.4 GB** new cache data

### Cutoff justification

8 Å heavy-heavy cutoff captures:
- All H-bond pairs (≤ 3.5 Å donor-acceptor)
- All ring–CA proximity for ring-current shifts (typically 4–6 Å)
- All disulfide S–S (2.0–2.1 Å) and CYS–CYS context (≤ 7 Å)

H-to-heavy uses 6 Å cutoff. Cross-residue H–H still excluded (matching intra rule).

Empirical sanity check: expect mean ~50–80 cross-pairs per residue, p99 ≈ 180. **M_CROSS = 200 with priority pruning by distance ascending = adequate.** If `count == 200` for >0.5% of residues, raise M_CROSS to 256.

---

## 4. Dataset/loader update plan

### `dataset.py` changes

1. Add to `_load_structural_data` (after line 252):
   ```python
   self.flat_cross_atom1 = np.load(sd / 'cross_atom1.npy', mmap_mode='r')
   self.flat_cross_atom2 = np.load(sd / 'cross_atom2.npy', mmap_mode='r')
   self.flat_cross_offset = np.load(sd / 'cross_offset.npy', mmap_mode='r')
   self.flat_cross_values = np.load(sd / 'cross_values.npy', mmap_mode='r')
   self.flat_cross_count = torch.from_numpy(np.load(sd / 'cross_count.npy'))
   ```

2. In `__getitem__` (after the per-window dist loop ~line 358), add center-residue cross extraction:
   ```python
   M_CR = self.max_cross_distances
   cross_a1 = torch.full((M_CR,), self.n_atom_types, dtype=torch.long)
   cross_a2 = torch.full((M_CR,), self.n_atom_types, dtype=torch.long)
   cross_off = torch.full((M_CR,), N_CROSS_OFFSET_TYPES, dtype=torch.long)
   cross_d = torch.zeros(M_CR, dtype=torch.float32)
   cross_m = torch.zeros(M_CR, dtype=torch.bool)
   nc = int(self.flat_cross_count[global_idx].item())
   if nc > 0:
       cross_a1[:nc] = torch.from_numpy(self.flat_cross_atom1[global_idx, :nc].astype(np.int64))
       cross_a2[:nc] = torch.from_numpy(self.flat_cross_atom2[global_idx, :nc].astype(np.int64))
       cross_off[:nc] = torch.from_numpy(self.flat_cross_offset[global_idx, :nc].astype(np.int64))
       cross_d[:nc] = torch.from_numpy(self.flat_cross_values[global_idx, :nc].astype(np.float32))
       cross_m[:nc] = True
   result['cross_atom1_idx'] = cross_a1
   result['cross_atom2_idx'] = cross_a2
   result['cross_offset_idx'] = cross_off
   result['cross_distances'] = cross_d
   result['cross_dist_mask'] = cross_m
   ```

3. Add `max_cross_distances` to the loaded `config.json` (read on construction).

### Backward compatibility

If `cross_atom1.npy` does not exist (old caches), set the attributes to `None` and `__getitem__` returns zero-filled cross tensors with `cross_dist_mask = False`.

### Normalization

Cross distances use the **same** scaling as intra: `value / 10.0`, clipped to `[-5, 10]`. The cutoff at 8 Å means cross values are always in `[0, 0.8]` range pre-clip, well within the existing distribution.

### Per-AA stats — confirmed unaffected.

---

## 5. Inference parity

### Hook point

`inference.py:extract_features_from_pdb` is the *only* place inference computes per-residue features.

1. **Compute cross-pairs after intra-features are computed** (insert at line 165):
   ```python
   from distance_features import calc_cross_residue_distances
   from config import MAX_CROSS_DISTANCES, CROSS_DIST_CUTOFF, CROSS_H_CUTOFF
   cross_by_rid = {}
   for i, rid in enumerate(res_ids):
       atoms_i = aa_data[rid]['atoms']
       cross_pairs = []
       # Window: ±5
       for off in range(-CONTEXT_WINDOW, CONTEXT_WINDOW + 1):
           if off == 0:
               continue
           j = i + off
           if 0 <= j < len(res_ids) and (res_ids[j] - rid) == off:
               other_atoms = aa_data[res_ids[j]]['atoms']
               offset_code = (off + CONTEXT_WINDOW) + 1
               for (a1, a2, d) in calc_cross_residue_distances(atoms_i, other_atoms, atom_to_idx, CROSS_DIST_CUTOFF, CROSS_H_CUTOFF):
                   cross_pairs.append((a1, a2, offset_code, d))
       # Spatial neighbors
       nb_info = neighbors.get(rid, {'ids': [-1]*K_SPATIAL_NEIGHBORS})
       for k, nrid in enumerate(nb_info['ids']):
           if nrid < 0 or nrid not in aa_data:
               continue
           offset_code = 12 + k
           other_atoms = aa_data[nrid]['atoms']
           for (a1, a2, d) in calc_cross_residue_distances(atoms_i, other_atoms, atom_to_idx, CROSS_DIST_CUTOFF, CROSS_H_CUTOFF):
               cross_pairs.append((a1, a2, offset_code, d))
       cross_pairs.sort(key=lambda t: t[3])
       cross_pairs = cross_pairs[:MAX_CROSS_DISTANCES]
       cross_by_rid[rid] = cross_pairs
   ```

2. **Pack into per-residue tensors** in the per-residue loop. Apply the same `/10.0` clip as intra.

3. The model's `forward` receives them via `**batch` — no other plumbing.

### Drift test

Add `tests/test_inference_cache_parity.py`:
- Pick 1 protein from fold 1.
- Build its cache row via `05_build_training_cache.py`'s helper.
- Run `extract_features_from_pdb` on the same PDB.
- Assert cross arrays match within fp16 tolerance after sorting both sides by `(offset_code, distance)`.

This must be run after every change to either `extract_features_from_pdb` or `calc_cross_residue_distances`. Run it as the smoke test gate before kicking off a full cache rebuild.

---

## 6. Disk-cleanup plan for /home/brooks/1TB/ — needed to free ≥80 GB

Current free: 9.8 GB. Target: 80+ GB free.

| Path | Size | Verdict |
|---|---:|---|
| `data_alphafold_cache_retrieval/` | 112 GB | **SAFE TO DELETE** — old retrieval-AF cache, current AF uses `data_alphafold_cache_dedup` |
| `struct_retrieval/` (v1) | 103 GB | **SAFE TO DELETE** after archiving best ckpt — superseded by v2 |
| `alphafold_retrieval_indices/` | 58 GB | **DELETE** — verify no live retrieval-AF jobs |
| `moved_from_main/` | 55 GB | **SAFE TO DELETE** — pure backup |
| `data_archive/` | 38 GB | **KEEP MOST**, partial dedup of structure_data_hybrid.csv |
| `data_alphafold_cache_noret/` | 27 GB | **SAFE TO DELETE** — superseded by `_dedup` |
| `retrieval_indices/` | 26 GB | **MAYBE → KEEP for now** |
| `struct_retrieval_v2/` | 72 GB | **KEEP — current best** |
| `data_alphafold_cache_dedup/` | 27 GB | **KEEP — current AF cache** |

**Estimated freed space (recommended deletions): ~355 GB.**

### Verification commands to run before deleting (USER MUST EXECUTE)

```bash
# Confirm v1 vs v2 checkpoint provenance
sha256sum /home/brooks/1TB/Wynn/struct_retrieval/checkpoints/best_retrieval_fold1.pt
sha256sum /home/brooks/1TB/Wynn/struct_retrieval_v2/checkpoints_v3/best_retrieval_fold1.pt

# Confirm no active jobs reference the deletion candidates
ps aux | grep -E "(data_alphafold_cache_retrieval|struct_retrieval/cache|alphafold_retrieval_indices)"

# Archive v1 best checkpoint before deletion
mkdir -p /home/brooks/Work/Wynn/checkpoint_archive
cp /home/brooks/1TB/Wynn/struct_retrieval/checkpoints/best_retrieval_fold1.pt \
   /home/brooks/Work/Wynn/checkpoint_archive/v1_best_retrieval_fold1.pt
```

### Recommended deletion order

```bash
# 1. Lowest-risk first
rm -rf /home/brooks/1TB/Wynn/moved_from_main          # 55 GB
rm -rf /home/brooks/1TB/Wynn/data_alphafold_cache_noret  # 27 GB

# 2. After archiving v1 ckpt
rm -rf /home/brooks/1TB/Wynn/struct_retrieval         # 103 GB

# 3. After confirming no active retrieval-AF jobs
rm -rf /home/brooks/1TB/Wynn/data_alphafold_cache_retrieval  # 112 GB
rm -rf /home/brooks/1TB/Wynn/alphafold_retrieval_indices     # 58 GB

# 4. Remove broken symlinks
find /home/brooks/Work/Wynn/shift_prediction -xtype l -ls
```

Steps 1+2 alone → ~185 GB free, safely sufficient for Phase 1.

---

## 7. Sequencing

1. **Code skeleton (no rebuild yet).** config.py + distance_features.py + dataset.py backward-compat + model.py with offset embedding + `tests/test_inference_cache_parity.py`.
2. **Smoke test.** Wire `extract_features_from_pdb`, build a 5-protein toy cache, confirm parity test passes.
3. **Disk cleanup** (steps 1+2 from §6).
4. **Build fold 1 cross cache** (~30–45 min).
5. **Fold 1 retraining** (~15 h). Gate criterion: TYR_CB MAE ≤ baseline × 0.95.
6. **Build folds 2–5 caches** in parallel (~40–60 min).
7. **Train folds 2–5** (~15 h each).
8. **AlphaFold cache rebuild** (Phase 1.5). Defer until hybrid Phase 1 confirmed.

### Validation

- Hold-out: `claude/plddt_vs_error/200_hybrid_holdout.json` — 37-protein curated subset.
- Per-atom MAE focus list: TYR-CA/CB/N, PHE-CA/CB/N, TRP-CA/CB/N, MET-CB/CE, CYS-CA/CB/N, HIS-CA/CB.
- Compare against: v2 struct-only baseline + UCBShift X on the same 37 proteins.

**Critical path to "is this working?": ~16 hours.**

---

## 8. Risk register

| Risk | Mitigation |
|---|---|
| Cache size blowup if M_CROSS=200 truncates real signal | Pre-flight: dump cross-pair counts on fold-1 first 50 proteins, confirm p99 ≤ 200. Distance-ascending priority pruning. |
| Over-fitting from new high-dim input | 0.2 dropout on offset embedding output. Smoke-test by training fold 1 with offset embedding frozen at 0 — verify still benefits. |
| Disulfide pair quality | Unit test: parse 1UBQ for CYS, verify S-S cross-distance matches PDBe values. |
| Inference vs training drift | Parity test in §5 is the hard gate. Run after every change to extract_features_from_pdb or calc_cross_residue_distances. |
| Existing checkpoint compatibility | Bumping N_ATOM_TYPES from 88 to 98 → atom_embed shape change. Old ckpts load with strict=False, new rows zero-initialized. Sanity check on-fold MAE preserves within ε. |
| PDB re-parsing failures | Use pdb_utils.resolve_pdb_path. Log failures to build_provenance.json. |
| fp16 distance precision for [0.0, 0.8] | ~3 decimal digits — sufficient. |

---

## 9. Estimated wall-clock

| Step | Wall-clock |
|---|---:|
| Code skeleton + tests | 4–6 h |
| Smoke test | 5 min |
| Disk cleanup | 3 min |
| Cache build fold 1 (hybrid) | 30–45 min |
| Train fold 1 struct-only with cross | 15 h |
| Eval on 37-protein hold-out | 2 min |
| Cache build folds 2–5 (parallel) | 40–60 min |
| Train folds 2–5 (sequential per GPU) | 15 h × 4 |
| AlphaFold cache rebuild (5 folds) | 3 h × 5 = 15 h |
| AlphaFold retraining | 15 h × 5 |

---

## 10. Final checklist

### Files modified

- [ ] `config.py` — add `MAX_CROSS_DISTANCES = 200`, `CROSS_DIST_CUTOFF = 8.0`, `CROSS_H_CUTOFF = 6.0`, `N_CROSS_OFFSET_TYPES = 16`, bump `N_ATOM_TYPES` from 88 to 98 (reserve Phase-2 atoms).
- [ ] `distance_features.py` — add `calc_cross_residue_distances`.
- [ ] `pdb_utils.py` — no change (already has `parse_pdb`).
- [ ] `05_build_training_cache.py` — new PDB lookup loader; new arrays in `build_cache_for_fold`; new save lines.
- [ ] `dataset.py` — load `cross_*` mmaps; emit cross tensors; backward-compat with missing files.
- [ ] `model.py` — `DistanceAttentionPerPosition` adds offset embedding; concatenate center cross-pairs.
- [ ] `inference.py` — extend `extract_features_from_pdb` with cross-pair computation.
- [ ] `tests/test_inference_cache_parity.py` (new).

### Data rebuilt

- [ ] `data/struct_retrieval_v2/cache/fold_{1..5}/structural/cross_*.npy` (5 × ~750 MB = ~3.7 GB)
- [ ] `data_alphafold_cache_dedup/fold_{1..5}/structural/cross_*.npy` (5 × ~2.9 GB = ~14.4 GB) — Phase 1.5

### Checkpoints invalidated

All current checkpoints (hybrid v2 + AF dedup) become **partially compatible**: they load with `strict=False`, missing keys are the new offset-embedding rows and the 10 new atom-vocab embedding rows (zero-initialized). Predictions differ by ≤ ε on intra-only inputs. **However, you cannot publish numbers from these patched-up checkpoints — they must be retrained.**

Still usable as **frozen-base initialization** for Phase 1 retraining via `--freeze_base`.

---

## Phase 0 (NEW — added after Plan agent finished)

The Plan agent didn't have these requirements. They must run BEFORE the
sidechain-features cache rebuild because both require the same expensive
operation (full cache rebuild from PDBs).

### 0.A — Use 90%-identity-cluster fold dedup (NOT just UniProt)

Found:
- `data/identity_clusters_90.json` exists (7,131 BMRB→cluster entries).
- `experiments/dedupe_folds_by_uniprot.py` exists but only produces AF outputs and only uses UniProt.

Proposed policy (extends the existing deduper):

```python
# fold = MD5(cluster_or_uniprot_or_bmrb) % 5 + 1
def fold_for_bmrb(bmrb):
    cluster = identity_clusters_90.get(bmrb)  # 90% identity cluster id
    if cluster:
        return MD5(f"cluster_{cluster}") % 5 + 1
    uniprot = bmrb_uniprot_mapping.get(bmrb)
    if uniprot:
        return MD5(f"uniprot_{uniprot}") % 5 + 1
    return MD5(f"bmrb_{bmrb}") % 5 + 1  # fallback — unique sequences
```

Output a single canonical `data/fold_assignments_identity90.json` covering
both the hybrid (12,427 BMRBs) and AF (11,225 BMRBs) supersets. Same fold ID
for any BMRB present in both pipelines.

Verification (must pass before rebuilding caches):
- For every fold f, every (BMRB_a in fold f train, BMRB_b in fold f test)
  pair with `cluster_a == cluster_b`: count must be 0.
- For every fold f, every pair with `uniprot_a == uniprot_b`: count must be 0.

### 0.B — Exclude all 197 UCBShift-200 test proteins from training

Find every BMRB in our training set whose paired PDB ID is one of the 197
test proteins (or 200, depending on what's in the test directory) from
`/home/brooks/Work/Wynn/UCBShift_testing/CSpred/train_model/pdbs/test/`.
Add a hard exclusion in the dedup script: those BMRBs go into a special
fold 0 = "always test, never train", or are dropped entirely from the
training pool.

This makes UCBShift-200 a **legitimately unseen** test set for our model —
right now, even with fold-based dedup, we can leak through retrieval and
the evaluation 37/43 subsets are derived from cache membership rather than
from the UCBShift test set itself.

### 0.C — Sequencing relative to Phase 1

Phase 0 lives BEFORE step 4 of §7 above. Concretely:

1. ... (as above through step 3 disk cleanup) ...
3.5. **Phase 0:** Generate `data/fold_assignments_identity90.json` and
     `data/excluded_ucbshift200.json`. Re-run `dedupe_folds_*` to produce
     deduped per-fold CSVs. Verify no within-fold leakage.
4. Build fold 1 cache **using the new fold assignments** (Phase 0 + Phase 1
   in one rebuild). Same wall-clock as before.
5. (and so on)
