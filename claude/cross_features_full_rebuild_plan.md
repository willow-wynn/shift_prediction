# Option A: Full cache rebuild with cross features + identity-90 fold split

This document describes how to integrate cross-residue distance features
into `05_build_training_cache.py` for a clean-slate rebuild that ALSO
adopts the new identity-90 fold assignment from Phase 0.

We are currently shipping option B (`04c_compute_cross_arrays.py`, the
incremental augmentation script) for early validation. Option A is the
production path once we want canonical training caches.

---

## Why option A eventually

Option B (incremental augmentation) is cheap but has two limitations:

1. **PDB drift.** The intra-distances in the existing v2 cache came from a
   specific PDB chosen at `01_build_datasets` time. Option B picks the
   first resolvable PDB for each BMRB, which may differ. Intra and cross
   then refer to slightly different conformations — usable for validation,
   not ideal for production.

2. **Wrong fold split.** The v2 cache uses BMRB-hash splits (~50% UniProt
   leakage). Phase 0 produced the canonical
   `data/fold_assignments_identity90.json` with no leakage and a fold-0
   exclusion set of 731 BMRBs (UCBShift-200 paired + identity/UniProt
   twins propagated). Option B doesn't change which BMRBs go in which fold.

Option A fixes both by rebuilding from the source CSVs.

---

## Critical files to modify

| File | Change |
|---|---|
| `05_build_training_cache.py` | Adopt new fold-assignment file. Add per-protein PDB re-parsing step. Add cross-pair computation per residue via `distance_features.build_cross_arrays_for_residue`. Save 5 new `.npy` files alongside existing structural arrays. |
| `03b_extract_struct_embeddings.py` | (No code change needed — but re-run on the new fold split before re-running 04 + 05.) |
| `04_build_retrieval_index.py` | Pass the new fold-assignment file via `--fold_assignments`. |

The model-side and inference-side changes already landed in commit
`fed0ff6`. The shared helper landed in `1afd6ed`. The deduper landed in
`f24aa06`. So Option A is purely a rebuild script change.

---

## Concrete code changes in `05_build_training_cache.py`

### 1. New fold-assignment loader (replaces the per-fold-CSV reading)

Currently the cache builder reads pre-split per-fold CSVs:
```python
fold_csv = os.path.join(args.data_dir, f'structure_data_hybrid_fold_{fold}.csv')
fold_df = pd.read_csv(fold_csv, ...)
```

Replace with a single source-of-truth read:
```python
# New: read canonical full hybrid CSV + JSON fold map
all_csv = os.path.join(args.data_dir, 'structure_data_hybrid.csv')
fold_map = json.load(open(args.fold_assignments))   # bmrb_id -> 0..5
df = pd.read_csv(all_csv, dtype={'bmrb_id': str})
df['split'] = df['bmrb_id'].astype(str).map(fold_map).fillna(-1).astype(int)
fold_df = df[df['split'] == fold]   # for the requested fold
```

This means we no longer maintain 5 per-fold CSVs at all. The deduper writes
just `fold_assignments_identity90.json` (already done) and the cache
builder reads it.

CLI change:
```python
parser.add_argument('--fold_assignments',
    default='data/fold_assignments_identity90.json',
    help='JSON mapping bmrb_id -> fold (0..5). Fold 0 = excluded.')
```

### 2. PDB re-parsing inside the per-protein loop

The cache builder currently builds all distance arrays from CSV `dist_*`
columns. To compute cross-pairs we need the raw PDB. Two sub-options:

**Sub-option A1 — re-parse inline.** In the per-protein loop, look up
the PDB path for the BMRB and call `parse_pdb`. Cost: ~0.25 s/protein for
~2,500 proteins/fold = ~10 min added to fold build time.

**Sub-option A2 — pre-compute a cross_pairs.npz per BMRB once, consume it.**
Run a separate stage `04b_extract_cross_pairs.py` once per dataset that
walks all BMRBs and writes `<data_dir>/cross_pairs/<bmrb>.npz`. Then the
cache builder mmap's those files. Slightly more code, slightly faster
overall if you rebuild caches more than once.

Recommend **A1** for first iteration (less code, easier reasoning). Move
to A2 if cache rebuilds become a hot path.

### 3. PDB resolver

Reuse the `resolve_bmrb_pdb` function from `04c_compute_cross_arrays.py`
(promoted to a shared utility, e.g. `pdb_utils.resolve_bmrb_pdb`):

```python
def resolve_bmrb_pdb(bmrb_id, pairs, uniprot_map, search_dirs):
    # 1. Experimental from pairs.csv (first listed PDB)
    # 2. AlphaFold AF-<uniprot>-F1-model_v6{,_fix}.pdb
    # Returns path or None
```

Search dirs default:
```python
search_dirs = [
    'data/pdbs',                                          # experimental
    '/home/brooks/1TB/Wynn/data_archive/alphafold',       # AF v6
    'data/esmfold',                                       # ESM-fold fallback
]
```

CRITICAL: the cache builder must record which PDB it picked per BMRB so
intra and cross stay consistent. Add a new file written alongside the
cache:
```
cache/fold_K/structural/bmrb_pdb_used.json    {"<bmrb_id>": "<pdb_path>"}
```

### 4. Cross-pair computation per residue

Inside the existing per-residue loop where intra arrays are filled, add:

```python
from distance_features import build_cross_arrays_for_residue
from config import (MAX_CROSS_DISTANCES, N_CROSS_OFFSET_TYPES,
                    CROSS_DIST_CUTOFF, CROSS_H_CUTOFF, CONTEXT_WINDOW)

# Before the per-residue loop, parse the PDB once per BMRB:
aa_data = parse_pdb_aa(pdb_path)
res_ids_in_order = sorted(aa_data.keys())

# For each residue with global_idx g and residue_id rid:
sp_ids = flat_spatial_ids[g].tolist()  # (K_SPATIAL,)
a1, a2, off, vals, n = build_cross_arrays_for_residue(
    center_rid=rid, aa_data=aa_data,
    res_ids_in_order=res_ids_in_order,
    spatial_neighbor_ids=sp_ids,
    atom_to_idx=ATOM_TO_IDX,
    context_window=CONTEXT_WINDOW,
    max_cross_distances=MAX_CROSS_DISTANCES,
    n_cross_offset_types=N_CROSS_OFFSET_TYPES,
    heavy_cutoff=CROSS_DIST_CUTOFF,
    h_cutoff=CROSS_H_CUTOFF,
)
flat_cross_atom1[g] = a1
flat_cross_atom2[g] = a2
flat_cross_offset[g] = off
flat_cross_values[g] = vals
flat_cross_count[g] = n
```

Allocations near the start of `build_cache_for_fold`:
```python
flat_cross_atom1 = np.full((n_residues, MAX_CROSS_DISTANCES),
                            len(atom_to_idx), dtype=np.int16)
flat_cross_atom2 = np.full_like(flat_cross_atom1, len(atom_to_idx))
flat_cross_offset = np.full((n_residues, MAX_CROSS_DISTANCES),
                             N_CROSS_OFFSET_TYPES, dtype=np.int8)
flat_cross_values = np.zeros((n_residues, MAX_CROSS_DISTANCES),
                              dtype=np.float16)
flat_cross_count = np.zeros((n_residues,), dtype=np.int16)
```

### 5. Save to disk

Add to the existing structural-save block:
```python
np.save(sd / 'cross_atom1.npy', flat_cross_atom1)
np.save(sd / 'cross_atom2.npy', flat_cross_atom2)
np.save(sd / 'cross_offset.npy', flat_cross_offset)
np.save(sd / 'cross_values.npy', flat_cross_values)
np.save(sd / 'cross_count.npy', flat_cross_count)
```

### 6. Config provenance

In `config.json` written next to the cache:
```python
config['max_cross_distances'] = MAX_CROSS_DISTANCES
config['n_cross_offset_types'] = N_CROSS_OFFSET_TYPES
config['cross_dist_cutoff'] = CROSS_DIST_CUTOFF
config['cross_h_cutoff'] = CROSS_H_CUTOFF
```

`dataset.py` already reads these via `config.get('max_cross_distances', ...)`
with a fallback to the global `config.py` constants.

---

## Sequencing for a clean Phase-1 rebuild

```
1. Verify Phase 0 outputs exist
   - data/fold_assignments_identity90.json
   - data/excluded_ucbshift200.json

2. Disk cleanup (1TB drive currently 9.8 GB free)
   - rm -rf /home/brooks/1TB/Wynn/moved_from_main         (55 GB)
   - rm -rf /home/brooks/1TB/Wynn/data_alphafold_cache_noret (27 GB)
   → ~82 GB free, sufficient for one fold's intermediate work

3. Modify 05_build_training_cache.py per §1-6 above

4. Smoke-build fold-1 hybrid:
   python 05_build_training_cache.py --data hybrid --folds 1 --no_retrieval
   Expected: ~30-45 min wall-clock. ~2.2 GB new cross arrays.

5. Run parity test against the freshly-built cache:
   python tests/test_inference_cache_parity.py
   (Must pass; if it fails, hard stop.)

6. Train fold-1 struct-only with cross features:
   python train.py --data hybrid --fold 1 --epochs 150 --structure_only
   Expected: ~15 hours.

7. Evaluate on:
   - The new deduped fold-1 test residues
   - The fold-0 (UCBShift-200) hold-out
   Gate: TYR_CB / PHE_CB / TRP_CB MAE ≤ 0.95 × baseline.

8. If gate passes:
   - Disk cleanup steps 3+4 from claude/sidechain_features_plan.md (~265 GB more free)
   - Build folds 2-5 hybrid (parallel, ~40-60 min)
   - Train folds 2-5 (~15 h each, sequential per GPU, OR run on Easley H100)
   - Phase 1.5: AF cache rebuild (15 h cache + 75 h training)
```

---

## Easley H100 deployment

For the larger AF rebuild (15 h cache build, 75 h training across 5 folds),
run on the Easley HPC cluster:

```bash
ssh wynnwillow@easley.alliance.unm.edu
# Sync code (data should already be on cluster, since previous AF runs
# happened there per /home/brooks/1TB/Wynn/runs/alphafold_struct_fold1_dedup_easley/)
rsync -av /home/brooks/Work/Wynn/shift_prediction/ \
  wynnwillow@easley.alliance.unm.edu:~/shift_prediction/
# On cluster:
sbatch run_phase1_rebuild_alphafold.sh
```

Sbatch template (write at `claude/easley_phase1.sh`, NOT yet committed):
```bash
#!/bin/bash
#SBATCH --partition=h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=phase1_%j.out

module load cuda
source ~/miniconda3/bin/activate shift_faiss
cd ~/shift_prediction
python 05_build_training_cache.py --data alphafold --folds 1 --no_retrieval
python train.py --data alphafold --fold 1 --epochs 150 --structure_only \
    --run_name af_struct_phase1_fold1
```

---

## Backwards compatibility / rollback

- `dataset.py` and `model.py` already work without cross arrays present
  (verified backward-compat in commit `fed0ff6`).
- Existing v2 caches continue to load and train normally.
- Existing checkpoints load with `strict=False` against the new model;
  the new `cross_distance_attention.*` weights stay at random init but
  produce zero output when cross_dist_mask is all-False (legacy path).
- To roll back Phase 1 entirely: revert commits `fed0ff6`, `1afd6ed`,
  and any cache-builder change. Phase 0 (deduper) can stay independent.

---

## Acceptance criteria for shipping option A

1. Phase-0 verification still passes after the rebuild.
2. Inference-cache parity test passes against the freshly-built cache.
3. Held-out fold-1 test set MAE on aromatic CB shifts (TYR_CB, PHE_CB,
   TRP_CB) is ≤ 95% of the baseline v2 struct-only checkpoint when both
   are evaluated on the IDENTITY-90 deduped fold-1 test residues (NOT
   the leaky v2 split — comparing apples to apples).
4. UCBShift-200 fold-0 evaluation completes and shows comparable or
   better numbers than the existing AF struct-only checkpoint.
