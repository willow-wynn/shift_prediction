# Chemical Shift Prediction

Retrieval-augmented deep learning for predicting NMR chemical shifts from protein 3D structure. Predicts 49 shift types (6 backbone + 43 sidechain) per residue.

## Setup

```bash
pip install -r requirements.txt
```

External tools needed:
- [DSSP](https://github.com/PDB-REDO/dssp) (secondary structure assignment)
- [MMseqs2](https://github.com/soedinglab/MMseqs2) (sequence clustering, step 02)

## Pipeline

Run in order. Each step produces inputs for the next.

### Data preparation

```bash
# 0. Download chemical shifts from BMRB
python 00_fetch_bmrb_shifts.py

# 1. Build datasets: find PDB/AlphaFold structures, align, compute features
python 01_build_datasets.py --online

# 2. Cluster sequences at 90% identity (for retrieval exclusion)
python cluster_sequences.py

# 3. Extract ESM-2 embeddings (2560-dim per residue)
python 03_extract_esm_embeddings.py
```

### Training

```bash
# Build caches + train (hybrid dataset, fold 1 held out)
python main.py --data hybrid --fold 1 --epochs 150

# Or step by step:
python 04_build_retrieval_index.py     # FAISS indices
python 05_build_training_cache.py      # Memory-mapped caches
python 06_train.py --fold 1            # Train
python 07_evaluate.py --model data/checkpoints/best_fold1.pt --fold 1
```

### Structure-bootstrapped retrieval (no ESM needed)

Train a pure structure model, extract its embeddings, use those for retrieval:

```bash
# Phase 1: Train structure-only model
python train_structure_only.py --fold 1 --epochs 200

# Phase 2: Extract structure embeddings
python 03b_extract_struct_embeddings.py \
    --checkpoint data/struct_only/checkpoints/best_struct_fold1.pt \
    --output data/struct_retrieval/struct_embeddings.h5

# Phase 3: Build FAISS indices + caches from structure embeddings
python 04_build_retrieval_index.py --embeddings data/struct_retrieval/struct_embeddings.h5 \
    --output_dir data/struct_retrieval/retrieval_indices
python 05_build_training_cache.py --embeddings data/struct_retrieval/struct_embeddings.h5 \
    --index_dir data/struct_retrieval/retrieval_indices \
    --output_dir data/struct_retrieval/cache

# Phase 4: Train retrieval with frozen base encoder
python train_retrieval_frozen.py \
    --struct_checkpoint data/struct_only/checkpoints/best_struct_fold1.pt \
    --cache_dir data/struct_retrieval/cache --fold 1 --epochs 150
```

### Inference

```bash
python inference.py --model data/checkpoints/best_fold1.pt --pdb protein.pdb
python inference.py --model data/checkpoints/best_fold1.pt --pdb protein.pdb --chain A -o shifts.csv
```

## Architecture

**Base encoder:** Per-residue distance attention over all intramolecular atom pairs → 5-layer residual CNN over ±5 residue window → spatial neighbor attention (K=5, min 4 residue separation) → 1472-dim encoding

**Retrieval pathway:** FAISS nearest-neighbor retrieval (K=32) → neighbor encoder → self-attention (2 layers) → shift-specific cross-attention (3 layers) → direct transfer head → learned gating blends retrieval with structure-only prediction

**Features per residue:** 88 atom-type distance pairs, residue/SS/mismatch embeddings, DSSP hydrogen bond geometry, phi/psi angles, inter-residue peptide bond lengths and CA-CA distances

## Datasets

Three structure sources via `--data`:

| Flag | Description |
|------|-------------|
| `hybrid` | Experimental PDB where available, AlphaFold otherwise (default) |
| `experimental` | Experimental PDB structures only |
| `alphafold` | AlphaFold predicted structures only |

## Files

### Pipeline scripts
| Script | Description |
|--------|-------------|
| `00_fetch_bmrb_shifts.py` | Download chemical shifts from BMRB |
| `01_build_datasets.py` | Build structure datasets from PDB/AlphaFold |
| `cluster_sequences.py` | MMseqs2 sequence clustering |
| `03_extract_esm_embeddings.py` | ESM-2 embedding extraction |
| `03b_extract_struct_embeddings.py` | Structure model embedding extraction |
| `04_build_retrieval_index.py` | FAISS index building |
| `05_build_training_cache.py` | Memory-mapped training cache |
| `06_train.py` | Model training |
| `07_evaluate.py` | Model evaluation with baselines and plots |
| `main.py` | Unified cache + train pipeline |
| `inference.py` | Single-PDB prediction |
| `train_structure_only.py` | Structure-only model (no retrieval) |
| `train_retrieval_frozen.py` | Retrieval training with frozen base encoder |
| `08_train_imputation.py` | Shift imputation model training |
| `09_eval_imputation.py` | Imputation evaluation |

### Libraries
| Module | Description |
|--------|-------------|
| `config.py` | All constants, paths, hyperparameters |
| `model.py` | Neural network architecture |
| `dataset.py` | Memory-mapped cached dataset |
| `retrieval.py` | FAISS retrieval with identity exclusion |
| `pdb_utils.py` | PDB parsing, DSSP |
| `distance_features.py` | Intramolecular distance computation |
| `spatial_neighbors.py` | KD-tree spatial neighbor finder |
| `alignment.py` | Sequence alignment |
| `structure_selection.py` | Best chain/model selection |
| `alphafold_utils.py` | AlphaFold DB download |
| `data_quality.py` | Data filtering with provenance |
| `random_coil.py` | Random coil shift tables |
| `imputation_model.py` | Imputation network |
| `imputation_dataset.py` | Imputation dataset |

## Configuration

All in `config.py`. Key settings:

- Model: CNN [256, 512, 768, 1024, 1280], K=5 spatial neighbors, K=32 retrieved
- Training: lr=2e-4, batch=1024, Huber δ=0.5, cosine annealing (T₀=50, T_mult=2)
- ESM-2: esm2_t36_3B_UR50D, layer 36, 2560-dim
- 88 canonical atom types, 49 shift types
