# CLAUDE.md

Project-level guidance for Claude Code in this repo. Kept concise — tool-level
harness rules are elsewhere; this file is just domain context and common commands.

## What this is

Retrieval-augmented deep learning for predicting NMR chemical shifts from
protein 3D structure. Predicts 49 shift types (6 backbone + 43 sidechain) per
residue. Wynn runs this. "brooks" is the local workstation; "easley" is the
HPC cluster (login: `ssh wynnwillow@easley.alliance.unm.edu`, key auth).

## Common commands

### Training
```bash
python train.py --data hybrid --fold 1 --epochs 150                     # full retrieval model
python train.py --data hybrid --fold 1 --epochs 200 --structure_only    # no retrieval
python train.py --data hybrid --fold 1 --freeze_base --base_checkpoint runs/.../best.pt
```

### Inference
```bash
python inference.py --model <ckpt> --pdb <file_or_PDB_ID> -o out.csv
```

### Data pipeline (in order)
```bash
python 00_fetch_bmrb_shifts.py
python 01_build_datasets.py --online
python 03_extract_esm_embeddings.py       # or 03b for structure embeddings
python 04_build_retrieval_index.py
python 05_build_training_cache.py
```

### Benchmarks
```bash
python run_benchmark.py                   # ours + UCBShift
python run_ucbshift_testset.py            # UCBShift on our full test set
```

UCBShift source: `/home/brooks/Work/Wynn/Claude_stuff/CSpred/`
UCBShift env:    `/home/brooks/Programs/localcolabfold/conda/envs/ucbshift/`

## Architecture

### Model (`model.py`)

`ShiftPredictor` (`ShiftPredictorWithRetrieval` alias kept for back-compat)
with a `use_retrieval` flag.

Base encoder (always active):
- `DistanceAttentionPerPosition`: attention over intramolecular atom-pair
  distances (88 atom types) → 256-dim per position
- 5-layer residual CNN `[256→512→768→1024→1280]` over ±5 residue window
- `SpatialNeighborAttention`: multi-head cross-attention over K=5 spatial
  neighbors, CNN center as query → 192-dim
- Bond geometry projection → 16-dim
- Total base encoding: 1280 + 192 = 1472-dim
- `struct_head`: 4-layer MLP → `n_shifts`

Retrieval pathway (when enabled):
- `RetrievalNeighborEncoder`: per-neighbor features (AA type, rank, cosine
  sim, shifts, 49-dim structural summary)
- 2x `SelfAttentionLayer` (neighbors attend to each other)
- 3x `ShiftSpecificCrossAttention` (per-shift queries attend to neighbors)
- `DirectTransferHead` → weighted shift transfer
- `retrieval_gate`: sigmoid blend of retrieval vs struct-only
- Same-AA mask: only neighbors matching query AA type are used

### Dataset (`dataset.py`)

`CachedRetrievalDataset` reads memory-mapped numpy arrays built by
`05_build_training_cache.py`. Structural features live in RAM; retrieval
data (shifts, masks, codes) is on disk via mmap. Per-AA normalization
converts global z-scores → per-AA z-scores on the fly when per_aa stats
are available.

### Key data paths

- `data/` — main dataset (hybrid experimental PDB + AlphaFold)
- `data/cache/fold_{1-5}/` — training caches
- `data/struct_retrieval_v2/` — structure-bootstrapped retrieval (current best)
  - `checkpoints_v3/best_retrieval_fold1.pt`
  - `retrieval_indices/` — FAISS indices
  - `cache/` — training caches for this variant
- `data_refdb/` — UCBShift RefDB in our format (2,386 proteins)
- `results/benchmark/`, `results/testset_ucbshift/` — per-protein JSONs
- `runs/` — training outputs

### Normalization

Training applies:
- Distances / 10.0, clipped to [-5, 10]
- DSSP features z-normalized with training-set mean/std (`dssp_stats.json`)
- Shifts z-normalized globally, optionally per-AA in the dataset

`inference.py` must apply the identical normalization — this has been a
recurring bug source (distance /10 and DSSP z-norm were missing at one point).

### Retrieval exclusion

- Training: same-protein exclusion by BMRB ID.
- Inference: pass actual BMRB ID to `retriever.retrieve()` for exclusion,
  or `__inference__` for a brand-new protein.

## Ongoing issues

- **Retrieval gate is conservative** (values 0.003–0.08 even for
  self-retrieval at cosine > 0.8). Struct-only dominates predictions.
- **UCBShift-Y gap** on the 197-set traces to mTM-align finding better
  structural transfers than our FAISS embedding retrieval. Current work
  (`ucbshifty_repl/`) replicates UCBShift-Y over our larger dataset with
  same-BMRB/same-UniProt exclusions, to probe how much of their advantage
  survives fair comparison.
- **Neighbor encoder is underinformed** — sees only a 49-dim summary per
  neighbor, not the neighbor's local window / CNN features / spatial
  neighbors.

## Output conventions

- Benchmark per-protein results → `results/<subdir>/` as JSON per protein
  so reruns can resume.
- Training output → `runs/<run_name>/`.
- Don't write output files at the repo root.
- Large artifacts (caches, embeddings, indices) go on `/home/brooks/1TB/`
  with a symlink back.
