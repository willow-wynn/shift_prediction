# Chemical Shift Prediction Pipeline

Retrieval-augmented deep learning pipeline for predicting NMR chemical shifts (49 shift types including backbone and sidechain) from protein 3D structure.

## Overview

Given a protein's PDB structure, the model:

1. Extracts structural features (intramolecular atom-pair distances, backbone angles, DSSP secondary structure, spatial neighbors)
2. Uses ESM-2 embeddings to retrieve similar residues from a training database via FAISS
3. Combines structural encoding with retrieved neighbor shifts through shift-specific cross-attention and direct transfer
4. Blends retrieval-augmented and structure-only predictions via learned dual gating
5. Predicts per-residue chemical shifts for 49 atom types (6 backbone + 43 sidechain)

## Quick Start

```bash
# Train on experimental structures (builds cache automatically if needed)
python main.py --data experimental --fold 1 --epochs 150

# Train on hybrid (experimental + AlphaFold where no experimental exists)
python main.py --data hybrid --fold 1 --epochs 150 --skip_cache

# Train on AlphaFold structures only
python main.py --data alphafold --fold 1 --epochs 150

# Predict shifts for a PDB file
python inference.py --model data/checkpoints/best_fold1.pt --pdb protein.pdb
python inference.py --model data/checkpoints/best_fold1.pt --pdb protein.pdb --chain A --backbone_only
python inference.py --model data/checkpoints/best_fold1.pt --pdb protein.pdb -o predictions.csv

# Evaluate a trained model
python 07_evaluate.py --data_dir data --model data/checkpoints/best_fold1.pt --fold 1 --plots
```

## Datasets

Three structure datasets are supported via `--data`:

| Dataset | Flag | Description |
|---------|------|-------------|
| Hybrid | `--data hybrid` | Experimental structures where available, AlphaFold otherwise. Primary training set. |
| Experimental | `--data experimental` | Experimental PDB structures only |
| AlphaFold | `--data alphafold` | AlphaFold predicted structures only |

All datasets share the same canonical atom vocabulary (88 atom types), ESM embeddings, and FAISS retrieval indices. Models trained on one dataset can be evaluated on data cached from any other.

Each dataset is stored in its own directory (`data/`, `data_alphafold/`, `data_experimental/`) with per-fold caches under `cache/` and model checkpoints under `checkpoints/`.

## Pipeline Steps

### Data Preparation (run once)

| Step | Script | Description |
|------|--------|-------------|
| 00 | `00_fetch_bmrb_shifts.py` | Download and pivot chemical shift data from BMRB |
| 01 | `01_build_datasets.py` | Find PDB/AlphaFold structures, align sequences, compute features, produce 3 CSV datasets |
| 02 | `cluster_sequences.py` | Cluster sequences at 90% identity using MMseqs2 for retrieval exclusion |
| 03 | `03_extract_esm_embeddings.py` | Extract ESM-2 embeddings (esm2_t36_3B_UR50D, layer 36, 2560-dim) |
| 04 | `04_build_retrieval_index.py` | Build fold-aware FAISS IVF indices with sequence identity exclusion |

### Training and Evaluation

| Script | Description |
|--------|-------------|
| `main.py` | Unified pipeline: builds cache if needed, then trains. Use `--data` to select dataset. |
| `05_build_training_cache.py` | Pre-compute retrieval results and save memory-mapped cache (called by `main.py`) |
| `06_train.py` | Training script (reads CSV for stats — use `main.py` instead for cache-only training) |
| `07_evaluate.py` | Evaluation: per-shift MAE/RMSE/R², baselines, retrieval analysis, plots |
| `08_train_imputation.py` | Train shift imputation model (structure + observed shifts + retrieval) |
| `09_eval_imputation.py` | Evaluate imputation model |
| `inference.py` | Predict shifts from a single PDB file |

## Model Architecture

### Base Structural Encoder
- **Distance attention** (`DistanceAttentionPerPosition`): Attention over all intramolecular atom-pair distances per residue position (88 atom types, learned embeddings)
- **1D residual CNN**: Processes a context window of ±5 residues (11 positions) through 5 residual blocks → 1280-dim
- **Spatial neighbor attention** (`SpatialNeighborAttention`): Attends to K=5 nearest spatial neighbors (min 4 residues sequence separation). Each neighbor contributes residue type, secondary structure, backbone angles, CA distance, and its own distance attention embedding.

### Retrieval Pathway
- **ESM-2 embeddings** (2560-dim per residue) are used as query vectors for FAISS nearest-neighbor retrieval (K=32 neighbors)
- **RetrievalNeighborEncoder**: Encodes each retrieved neighbor using amino acid type, rank, cosine similarity, same-AA indicator, shift values, deviation from consensus, structural features
- **Self-attention** (2 layers): Neighbors attend to each other
- **Shift-specific cross-attention** (3 layers): Per-shift-type query embeddings attend to neighbor encodings, allowing different shifts to focus on different neighbors
- **Direct transfer head**: Learned per-neighbor per-shift scoring for direct shift copying
- **Dual gating**: `direct_gate` blends direct transfer vs attention prediction; `retrieval_gate` blends retrieval vs structure-only. Both are learned functions that adapt per-sample.

### Prediction Flow
```
struct_pred = structure_head(base_encoding)
attn_pred = attention_head(shift_context, base_encoding)
direct_pred = direct_transfer(neighbor_encodings, retrieved_shifts)

retrieval_pred = direct_gate * direct_pred + (1 - direct_gate) * attn_pred
final_pred = retrieval_gate * retrieval_pred + (1 - retrieval_gate) * struct_pred
```

When no retrieval neighbors are available (e.g., during single-PDB inference), the model gracefully falls back to structure-only prediction via the gating mechanism.

### Shift Imputation Model
A second-stage model (`imputation_model.py`) leverages partially observed chemical shifts to predict missing ones, combining structural encoding, observed shift context, and shift-aware retrieval.

## Dataset Construction (Step 01)

### Structure Sources

For each BMRB protein entry the pipeline finds structures from two sources:

- **Experimental PDB**: Looks up PDB IDs via `pairs.csv` and the BMRB API. Selects the chain with highest sequence identity.
- **AlphaFold**: Maps BMRB → UniProt → AlphaFold DB for predicted structures.

### Sequence Alignment

Each PDB chain is aligned to the BMRB sequence using Biopython pairwise alignment. A minimum **80% sequence identity** cutoff is enforced. The alignment classifies each position as `match`, `mismatch`, `protein_edge`, `gap_in_cs`, or `gap_in_structure`.

### NMR Model Selection

For multi-model NMR PDB files: superimpose all models onto Model 1 (Kabsch, CA atoms), compute median coordinates, select the model with lowest RMSD from the median.

## Configuration

All hyperparameters, paths, and constants are in `config.py`:

- **Canonical atom vocabulary**: 88 atom types shared across all datasets (`ATOM_TYPES`, `ATOM_TO_IDX`)
- **Dataset directories**: `DATASET_DIRS` maps `hybrid`/`alphafold`/`experimental` to directory paths
- **Training**: learning rate (2e-4), batch size (1024), Huber delta (0.5), weight decay (0.05), cosine annealing with warm restarts (T_0=50, T_mult=2)
- **Model**: CNN channels [256, 512, 768, 1024, 1280], K=5 spatial neighbors, K=32 retrieved, 8-head retrieval attention
- **ESM-2**: esm2_t36_3B_UR50D, layer 36, 2560-dim embeddings

## Utility Modules

| Module | Description |
|--------|-------------|
| `config.py` | Central configuration: atom vocabulary, dataset paths, hyperparameters, model architecture |
| `model.py` | Neural network: distance attention, CNN, spatial attention, retrieval cross-attention, dual gating |
| `dataset.py` | Memory-efficient cached dataset with disk-mapped retrieval data |
| `data_quality.py` | `FilterLog` class for provenance-tracked data filtering |
| `analyze_data_quality.py` | Comprehensive data cleaning: outlier detection, physical range checks, shift referencing errors |
| `random_coil.py` | Random coil shift tables (Schwarzinger 2001, Wishart 1995) |
| `distance_features.py` | Intra-residue atom-pair distance computation from PDB coordinates |
| `spatial_neighbors.py` | K-nearest spatial neighbor finder with minimum sequence separation |
| `alignment.py` | Biopython-based sequence alignment for BMRB-to-PDB residue mapping |
| `pdb_utils.py` | PDB file parsing, DSSP wrapper, coordinate extraction |
| `structure_selection.py` | Best-chain selection by alignment identity; Kabsch superposition |
| `rcsb_search.py` | RCSB PDB API search |
| `alphafold_utils.py` | AlphaFold DB structure download via UniProt mapping |
| `retrieval.py` | FAISS-based retrieval with identity exclusion |
| `imputation_model.py` | Shift imputation network |
| `imputation_dataset.py` | Extends cached dataset with observed shift context |

## Requirements

See `requirements.txt`. Core dependencies:

- Python >= 3.9
- PyTorch >= 2.0
- NumPy, Pandas, SciPy
- Biopython >= 1.80
- fair-esm (ESM-2 protein language model)
- faiss-cpu or faiss-gpu
- h5py (ESM embedding storage)
- matplotlib (evaluation plots)
- wandb (optional training logging)
- tqdm

External tools:
- [MMseqs2](https://github.com/soedinglab/MMseqs2) — sequence clustering (Step 02)
- [DSSP](https://github.com/PDB-REDO/dssp) — secondary structure assignment (Step 01)

## Directory Structure

```
shift_prediction/
├── main.py                           # Unified pipeline: cache + train
├── inference.py                      # Single-PDB shift prediction
├── 00_fetch_bmrb_shifts.py           # Step 0: BMRB data download
├── 01_build_datasets.py              # Step 1: Structure selection + dataset compilation
├── cluster_sequences.py              # Step 2: Sequence clustering
├── 03_extract_esm_embeddings.py      # Step 3: ESM-2 extraction
├── 04_build_retrieval_index.py       # Step 4: FAISS index building
├── 05_build_training_cache.py        # Step 5: Memory-mapped cache
├── 06_train.py                       # Step 6: Model training
├── 07_evaluate.py                    # Step 7: Model evaluation
├── 08_train_imputation.py            # Step 8: Imputation training
├── 09_eval_imputation.py             # Step 9: Imputation evaluation
├── config.py                         # Central configuration
├── model.py                          # Model architecture
├── dataset.py                        # Cached dataset
├── retrieval.py                      # FAISS retrieval
├── pdb_utils.py                      # PDB parsing + DSSP
├── distance_features.py              # Distance feature computation
├── spatial_neighbors.py              # Spatial neighbor finder
├── alignment.py                      # Sequence alignment
├── structure_selection.py            # Chain/model selection
├── data_quality.py                   # Data filtering with provenance
├── analyze_data_quality.py           # Comprehensive data cleaning
├── random_coil.py                    # Random coil shift tables
├── alphafold_utils.py                # AlphaFold DB download
├── rcsb_search.py                    # RCSB PDB API
├── imputation_model.py               # Imputation model
├── imputation_dataset.py             # Imputation dataset
├── requirements.txt
├── data/                             # Hybrid dataset (primary)
│   ├── structure_data_hybrid.csv
│   ├── chemical_shifts.csv
│   ├── pairs.csv
│   ├── esm_embeddings.h5
│   ├── retrieval_indices/            # Per-fold FAISS indices
│   ├── cache/                        # Per-fold training caches
│   ├── checkpoints/                  # Model checkpoints
│   ├── pdbs/                         # Experimental PDB structures
│   ├── alphafold/                    # AlphaFold structures
│   └── build_log.csv
├── data_alphafold/                   # AlphaFold-only dataset
│   ├── structure_data_hybrid.csv -> structure_data_alphafold.csv
│   ├── cache/
│   └── checkpoints/
└── data_experimental/                # Experimental-only dataset
    ├── structure_data_hybrid.csv -> structure_data_experimental.csv
    ├── cache/
    └── checkpoints/
```
