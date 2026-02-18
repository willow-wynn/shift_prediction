# Chemical Shift Prediction Pipeline

Retrieval-augmented deep learning pipeline for predicting backbone chemical shifts (CA, CB, C, N, H, HA) from protein 3D structure.

## Overview

This pipeline predicts NMR backbone chemical shifts from protein structures using a retrieval-augmented architecture. Given a protein's PDB structure and sequence, it:

1. Extracts structural features (distances, angles, DSSP, spatial neighbors)
2. Computes physics-based features (ring currents, half-sphere exposure, hydrogen bond geometry)
3. Uses ESM-2 embeddings to retrieve similar residues from a training database via FAISS
4. Combines structural encoding with retrieved neighbor shifts through cross-attention and query-conditioned transfer
5. Predicts per-residue backbone chemical shifts with calibrated confidence estimates

## Features

### Outlier Filtering with Provenance Tracking
Physical range filters (e.g., CA: 40-70 ppm, N: 100-140 ppm) remove impossible shift values. Every removed value is logged with its protein ID, residue ID, column, original value, and the rule that triggered removal. The `FilterLog` class in `data_quality.py` provides full audit trails saved to CSV.

### Dense Distance Features
Uses 7 backbone/near-backbone atoms (C, CA, CB, H, HA, N, O) producing approximately 21 dense atom pair distances per residue. An attention mechanism over these distance features learns which pairwise distances matter most.

### PDB Quality Selection
`01_select_pdb_structures.py` evaluates multiple PDB structures per BMRB entry and selects the best based on resolution, R-factor, sequence identity to the BMRB sequence, and completeness. Every selection decision is logged.

### Physics-Based Features
Computed in `physics_features.py`:
- **Ring current effects**: Estimated ring current shifts for H and HA from nearby aromatic residues (PHE, TYR, TRP, HIS)
- **Half-sphere exposure (HSE)**: Upper and lower hemisphere CA neighbor counts, providing a continuous measure of burial
- **Hydrogen bond geometry**: Distances and energies from DSSP H-bond assignments

These are encoded by a 2-layer MLP (`PhysicsFeatureEncoder` in `model.py`) producing a 64-dimensional vector concatenated with the structural encoding.

### Random Coil Correction in Retrieval Transfer
When transferring chemical shifts from retrieved neighbors to the query residue, the `QueryConditionedTransfer` module applies random coil correction before weighted averaging:

```
corrected = RC[query_aa] + (retrieved_shift - RC[retrieved_aa])
```

This preserves the structural (secondary) shift contribution while adjusting for the intrinsic chemical shift difference between amino acid types. Random coil values are from Schwarzinger et al. (2001) and Wishart et al. (1995).

### Shift Imputation Model
`08_train_imputation.py` trains a second-stage model (`imputation_model.py`) that leverages partially observed chemical shifts to predict missing ones. It combines three information sources:
- **Structural encoding**: Distance attention + CNN + spatial attention
- **Observed shift context**: A 1D CNN over the residue's known shifts (with masking), so available measurements inform the prediction
- **Shift-aware retrieval**: Retrieved neighbor shifts are re-weighted based on observed shift similarity

Each sample is a (residue, shift_type) pair predicting a single scalar.

## Pipeline Steps

| Step | Script | Description |
|------|--------|-------------|
| 00 | `00_fetch_bmrb_shifts.py` | Download chemical shift data from BMRB for all entries |
| 01 | `01_select_pdb_structures.py` | Select best PDB structure per BMRB entry with quality scoring |
| 02 | `02_compile_dataset.py` | Compile final dataset: align sequences, compute distances, DSSP, spatial neighbors, physics features |
| 03 | `03_extract_esm_embeddings.py` | Extract ESM-2 embeddings (esm2_t36_3B_UR50D, layer 36, 2560-dim) for all proteins; compressed fp16 HDF5 output |
| 04 | `04_build_retrieval_index.py` | Build fold-aware FAISS IVF indices with same-fold exclusion for retrieval |
| 05 | `05_build_training_cache.py` | Pre-compute retrieval results and save memory-mapped cache for fast training I/O |
| 06 | `06_train.py` | Train the retrieval-augmented model with 5-fold cross-validation |
| 07 | `07_evaluate.py` | Comprehensive evaluation: per-shift metrics, baselines, retrieval analysis, plots |
| 08 | `08_train_imputation.py` | Train shift imputation model (structure + observed shifts + retrieval) |
| 09 | `09_eval_imputation.py` | Evaluate imputation model against structure-only, mean, and random coil baselines |

## Utility Modules

| Module | Description |
|--------|-------------|
| `config.py` | Central configuration: paths, amino acid mappings, shift ranges, hyperparameters |
| `model.py` | Neural network: distance attention, CNN encoder, spatial attention, physics encoder, retrieval cross-attention, query-conditioned transfer with random coil correction |
| `dataset.py` | Memory-efficient cached dataset with disk-mapped retrieval data and physics features |
| `data_quality.py` | `FilterLog` class and filtering functions (outliers, duplicates, non-standard residues) with full provenance |
| `random_coil.py` | Random coil shift tables (Schwarzinger 2001, Wishart 1995) and correction functions |
| `physics_features.py` | Ring current estimation, HSE computation, H-bond features |
| `distance_features.py` | Dense backbone distance computation from PDB coordinates |
| `spatial_neighbors.py` | K-nearest spatial neighbor finder with minimum sequence separation |
| `alignment.py` | Biopython-based sequence alignment for BMRB-to-PDB residue mapping |
| `pdb_utils.py` | PDB file parsing, chain extraction, coordinate retrieval |
| `retrieval.py` | FAISS-based retrieval module with embedding lookup and random coil correction |
| `imputation_model.py` | Shift imputation neural network: structural encoding + observed shift context CNN + shift-aware retrieval, conditioned on shift type |
| `imputation_dataset.py` | Extends cached dataset with observed shift context; per-(residue, shift_type) sampling with target masking |

## Quick Start

```bash
# Step 0: Fetch BMRB chemical shift data
python 00_fetch_bmrb_shifts.py --output_dir data

# Step 1: Select PDB structures
python 01_select_pdb_structures.py --shifts_dir data --output_dir data --pdb_dir data/pdbs

# Step 2: Compile the dataset
python 02_compile_dataset.py --shifts_dir data --pdb_dir data/pdbs --output data/structure_data.csv

# Step 3: Extract ESM-2 embeddings (GPU recommended)
python 03_extract_esm_embeddings.py --data_dir data --output data/esm_embeddings.h5

# Step 4: Build FAISS retrieval indices
python 04_build_retrieval_index.py --data_dir data --embeddings data/esm_embeddings.h5 --output_dir data/retrieval_indices

# Step 5: Build training cache (memory-mapped)
python 05_build_training_cache.py --data_dir data --output_dir cache

# Step 6: Train structure-only model
python 06_train.py --data_dir data --cache_dir cache --fold 1 --epochs 200 --output_dir checkpoints

# Step 7: Evaluate structure-only model
python 07_evaluate.py --data_dir data --cache_dir cache --model checkpoints/best_retrieval_fold1.pt --fold 1 --output_dir eval_results --plots

# Step 8: Train imputation model (uses existing cache + adds shift context)
python 08_train_imputation.py --data_dir data --fold 1 --epochs 350 --output_dir checkpoints

# Step 9: Evaluate imputation model
python 09_eval_imputation.py --model checkpoints/best_imputation_fold1.pt --data_dir data --fold 1 --output_dir eval_imputation_results
```

## Data Provenance

The `FilterLog` system tracks every data modification with full context:

- **Entry-level logging**: Each removed/modified value records the step name, reason, protein ID, residue ID, column, original value, and action taken
- **Summary logging**: Each filtering step records aggregate counts and human-readable descriptions
- **Export**: `FilterLog.save(path)` writes both a detailed CSV and a summary CSV
- **In-pipeline**: `02_compile_dataset.py` generates a provenance log covering alignment, distance computation, DSSP, and quality filtering
- **Training**: `06_train.py` saves a `training_provenance_fold{N}.json` with data counts, outlier statistics, per-epoch metrics, and learning rate schedule

## Requirements

See `requirements.txt`. Core dependencies:

- PyTorch >= 2.0
- NumPy, Pandas, SciPy
- Biopython >= 1.80 (sequence alignment, PDB parsing)
- fair-esm (ESM-2 protein language model)
- faiss-cpu or faiss-gpu (approximate nearest neighbor search)
- h5py (ESM embedding storage)
- scikit-learn (metrics)
- matplotlib (evaluation plots)
- wandb (optional training logging)
- tqdm (progress bars)

## Directory Structure

After running the full pipeline:

```
shift_prediction/
├── 00_fetch_bmrb_shifts.py        # Step 0: BMRB data download
├── 01_select_pdb_structures.py    # Step 1: PDB quality selection
├── 02_compile_dataset.py          # Step 2: Dataset compilation
├── 03_extract_esm_embeddings.py   # Step 3: ESM-2 extraction
├── 04_build_retrieval_index.py    # Step 4: FAISS index building
├── 05_build_training_cache.py     # Step 5: Memory-mapped cache
├── 06_train.py                    # Step 6: Structure model training
├── 07_evaluate.py                 # Step 7: Structure model evaluation
├── 08_train_imputation.py         # Step 8: Imputation model training
├── 09_eval_imputation.py          # Step 9: Imputation model evaluation
├── config.py
├── model.py
├── dataset.py
├── imputation_model.py
├── imputation_dataset.py
├── data_quality.py
├── random_coil.py
├── physics_features.py
├── distance_features.py
├── spatial_neighbors.py
├── alignment.py
├── pdb_utils.py
├── retrieval.py
├── requirements.txt
├── README.md
├── data/
│   ├── structure_data.csv          # Compiled dataset
│   ├── esm_embeddings.h5           # ESM-2 embeddings
│   ├── retrieval_indices/          # Per-fold FAISS indices
│   ├── pdbs/                       # Downloaded PDB files
│   ├── provenance_log.csv          # Data compilation provenance
│   └── provenance_log_summary.csv  # Provenance summary
├── cache/
│   ├── fold_{k}/                   # Per-fold cached tensors
│   │   ├── structural/             # Residue types, distances, DSSP, physics
│   │   ├── retrieval/              # FAISS retrieval results
│   │   └── imputation/             # Shift context arrays
│   └── ...
├── checkpoints/
│   ├── best_retrieval_fold1.pt     # Best structure model (fold 1)
│   ├── best_imputation_fold1.pt    # Best imputation model (fold 1)
│   ├── checkpoint_fold1_epoch50.pt # Periodic checkpoint
│   └── training_provenance_fold1.json
└── eval_results/
    ├── evaluation_fold1.json       # Full evaluation metrics
    ├── per_shift_metrics_fold1.csv # Per-shift breakdown
    ├── per_protein_metrics_fold1.csv
    └── plots_fold1/
```
