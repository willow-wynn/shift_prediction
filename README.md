# Chemical Shift Prediction Pipeline

Retrieval-augmented deep learning pipeline for predicting backbone chemical shifts (CA, CB, C, N, H, HA) from protein 3D structure.

## Overview

This pipeline predicts NMR backbone chemical shifts from protein structures using a retrieval-augmented architecture. Given a protein's PDB structure and sequence, it:

1. Extracts structural features (intramolecular distances, backbone angles, DSSP secondary structure, spatial neighbors)
2. Computes physics-based features (ring currents, half-sphere exposure, hydrogen bond geometry)
3. Uses ESM-2 embeddings to retrieve similar residues from a training database via FAISS
4. Combines structural encoding with retrieved neighbor shifts through cross-attention and query-conditioned transfer
5. Predicts per-residue backbone chemical shifts with calibrated confidence estimates

## Pipeline Steps

| Step | Script | Description |
|------|--------|-------------|
| 00 | `00_fetch_bmrb_shifts.py` | Download and pivot chemical shift data from BMRB |
| 01 | `01_build_datasets.py` | Find PDB/AlphaFold structures, align sequences, compute all features, produce 3 output datasets |
| 02 | `cluster_sequences.py` | Cluster sequences at 90% identity using MMseqs2 for retrieval exclusion |
| 03 | `03_extract_esm_embeddings.py` | Extract ESM-2 embeddings (esm2_t36_3B_UR50D, layer 36, 2560-dim) |
| 04 | `04_build_retrieval_index.py` | Build fold-aware FAISS IVF indices with sequence identity exclusion |
| 05 | `05_build_training_cache.py` | Pre-compute retrieval results and save memory-mapped cache for fast training I/O |
| 06 | `06_train.py` | Train the retrieval-augmented model with 5-fold cross-validation |
| 07 | `07_evaluate.py` | Comprehensive evaluation: per-shift metrics, baselines, retrieval analysis, plots |
| 08 | `08_train_imputation.py` | Train shift imputation model (structure + observed shifts + retrieval) |
| 09 | `09_eval_imputation.py` | Evaluate imputation model against structure-only, mean, and random coil baselines |

### Quick Start

```bash
# Step 0: Fetch BMRB chemical shift data
python 00_fetch_bmrb_shifts.py --download --output-dir data

# Step 1: Build datasets (online mode fetches PDB/AlphaFold structures)
python 01_build_datasets.py --online --output-dir data --pdb-dir data/pdbs

# Step 2: Cluster sequences for retrieval exclusion (requires MMseqs2)
python cluster_sequences.py --data_dir data

# Step 3: Extract ESM-2 embeddings (GPU recommended)
python 03_extract_esm_embeddings.py --data_dir data

# Step 4: Build FAISS retrieval indices
python 04_build_retrieval_index.py --data_dir data

# Step 5: Build training cache (memory-mapped)
python 05_build_training_cache.py --data_dir data

# Step 6: Train model
python 06_train.py --data_dir data --fold 1 --epochs 200

# Step 7: Evaluate model
python 07_evaluate.py --data_dir data --model checkpoints/best_retrieval_fold1.pt --fold 1 --plots

# Step 8: Train imputation model
python 08_train_imputation.py --data_dir data --fold 1 --epochs 350

# Step 9: Evaluate imputation model
python 09_eval_imputation.py --model checkpoints/best_imputation_fold1.pt --data_dir data --fold 1
```

### Inference

```bash
# CLI prediction on a PDB file
python predict.py --checkpoint checkpoints/best_retrieval_fold1.pt --pdb my_protein.pdb --chain A

# Web UI (requires gradio)
python app.py --checkpoint checkpoints/best_retrieval_fold1.pt
```

## Dataset Construction (Step 01)

`01_build_datasets.py` is the main data pipeline. It replaces the earlier separate structure selection and compilation scripts.

### Structure Sources

For each BMRB protein entry the pipeline finds structures from two sources:

- **Experimental PDB**: Looks up PDB IDs via `pairs.csv` and the BMRB API (in `--online` mode). For each candidate PDB, selects the chain with the highest sequence identity to the BMRB sequence.
- **AlphaFold**: Maps BMRB -> UniProt -> AlphaFold DB to retrieve predicted structures.

Proteins with no PDB mapping go straight to AlphaFold-only (there is no BLAST fallback).

### Sequence Alignment and Identity

Each PDB chain is aligned to the BMRB sequence using Biopython pairwise alignment. A minimum **80% sequence identity** cutoff (`MIN_SEQUENCE_IDENTITY` in `config.py`) is enforced — proteins below this threshold are rejected.

The alignment produces a residue mapping that classifies each position:

| Mismatch Type | Meaning |
|---------------|---------|
| `match` | Identical residue in both structure and shift data |
| `mismatch` | Aligned but different residue types |
| `protein_edge` | Within 2 residues of chain termini |
| `gap_in_cs` | Structure residue has no chemical shift counterpart (gap in shift sequence) |
| `gap_in_structure` | Chemical shift residue has no structure counterpart (gap in structure) |

### NMR Model Selection

For multi-model NMR PDB files, the pipeline selects a single representative model:

1. Parse all models from the PDB file
2. Superimpose all models onto Model 1 using Kabsch superposition (CA atoms)
3. Compute median CA coordinates across all models
4. Select the model with the lowest RMSD from the median coordinates

### Three Output Datasets

| Dataset | File | Description |
|---------|------|-------------|
| Hybrid | `structure_data_hybrid.csv` | Experimental structures where available, AlphaFold otherwise. Primary training set. |
| Experimental | `structure_data_experimental.csv` | Experimental PDB structures only |
| AlphaFold | `structure_data_alphafold.csv` | All AlphaFold structures |

### Quality Filtering

- Physical range filters remove impossible shift values (e.g., CA: 40-70 ppm)
- Statistical outlier detection grouped by **(residue_code, shift_column)**, removing values beyond 3 IQRs
- Non-standard residues and nucleotide entries are discarded
- All filtering decisions are logged to `build_log.csv` with full provenance

## Sequence Clustering for Retrieval Exclusion

`cluster_sequences.py` uses MMseqs2 to cluster all protein sequences at 90% identity. The output (`identity_clusters_90.json`) maps each BMRB ID to the list of other BMRB IDs with >90% sequence identity. This exclusion map is used during FAISS retrieval to prevent data leakage from homologous proteins.

**Requires**: [MMseqs2](https://github.com/soedinglab/MMseqs2) installed and on `$PATH`.

## Model Architecture

### Structural Encoding
- **Distance attention** (`DistanceAttentionPerPosition`): Learns over 7 backbone atom pair distances per residue via attention
- **1D CNN encoder**: Processes the context window (i +/- 5 residues) of structural features
- **Spatial neighbor attention** (`SpatialNeighborAttention`): Attends to the k-nearest spatial neighbors. Each spatial neighbor contributes residue type, secondary structure, backbone angles, CA distance, and its own intramolecular distance embeddings (computed by the shared distance attention module)

### Physics Features
- Ring current effects on H and HA from nearby aromatics
- Half-sphere exposure (upper/lower CA neighbor counts)
- Hydrogen bond geometry from DSSP
- Encoded by a 2-layer MLP (`PhysicsFeatureEncoder`)

### Retrieval-Augmented Prediction
- **ESM-2 embeddings**: Used as query vectors for FAISS nearest-neighbor retrieval
- **Cross-attention**: Attends to retrieved neighbor embeddings
- **Query-conditioned transfer**: Applies random coil correction when transferring shifts from retrieved neighbors:
  ```
  corrected = RC[query_aa] + (retrieved_shift - RC[retrieved_aa])
  ```
- **>90% identity exclusion**: During retrieval, residues from proteins with >90% sequence identity to the query are excluded (in addition to same-protein exclusion)

### Shift Imputation Model
A second-stage model (`imputation_model.py`) leverages partially observed chemical shifts to predict missing ones, combining structural encoding, observed shift context, and shift-aware retrieval.

## Utility Modules

| Module | Description |
|--------|-------------|
| `config.py` | Central configuration: paths, amino acid mappings, shift ranges, hyperparameters |
| `model.py` | Neural network: distance attention, CNN encoder, spatial attention, physics encoder, retrieval cross-attention, query-conditioned transfer |
| `dataset.py` | Memory-efficient cached dataset with disk-mapped retrieval data, spatial neighbor distances, and physics features |
| `data_quality.py` | `FilterLog` class and filtering functions (outliers, duplicates, non-standard residues) with full provenance |
| `random_coil.py` | Random coil shift tables (Schwarzinger 2001, Wishart 1995) and correction functions |
| `physics_features.py` | Ring current estimation, HSE computation, H-bond features |
| `distance_features.py` | Dense backbone distance computation from PDB coordinates |
| `spatial_neighbors.py` | K-nearest spatial neighbor finder with minimum sequence separation |
| `alignment.py` | Biopython-based sequence alignment for BMRB-to-PDB residue mapping |
| `pdb_utils.py` | PDB file parsing (single and multi-model), chain extraction, coordinate retrieval |
| `structure_selection.py` | Best-chain selection by alignment identity; Kabsch superposition |
| `rcsb_search.py` | RCSB PDB API search for structure discovery |
| `alphafold_utils.py` | AlphaFold DB structure download via UniProt mapping |
| `cluster_sequences.py` | MMseqs2-based sequence clustering for retrieval exclusion |
| `retrieval.py` | FAISS-based retrieval with identity exclusion and random coil correction |
| `predict.py` | Inference engine: load a trained model and predict shifts from a PDB file |
| `app.py` | Gradio web UI for interactive prediction |
| `imputation_model.py` | Shift imputation network: structural encoding + observed shift context + shift-aware retrieval |
| `imputation_dataset.py` | Extends cached dataset with observed shift context; per-(residue, shift_type) sampling |

## Requirements

See `requirements.txt`. Core dependencies:

- Python >= 3.9
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

External tools (not pip-installable):
- [MMseqs2](https://github.com/soedinglab/MMseqs2) — required for sequence clustering (Step 02)
- [DSSP](https://github.com/PDB-REDO/dssp) — required for secondary structure assignment (used in Step 01)

Optional:
- gradio — for the web UI (`app.py`)

## Directory Structure

```
homologies_better_data/
├── 00_fetch_bmrb_shifts.py        # Step 0: BMRB data download
├── 01_build_datasets.py           # Step 1: Structure selection + dataset compilation
├── cluster_sequences.py           # Step 2: Sequence clustering (MMseqs2)
├── 03_extract_esm_embeddings.py   # Step 3: ESM-2 extraction
├── 04_build_retrieval_index.py    # Step 4: FAISS index building
├── 05_build_training_cache.py     # Step 5: Memory-mapped cache
├── 06_train.py                    # Step 6: Model training
├── 07_evaluate.py                 # Step 7: Model evaluation
├── 08_train_imputation.py         # Step 8: Imputation model training
├── 09_eval_imputation.py          # Step 9: Imputation model evaluation
├── predict.py                     # Inference engine
├── app.py                         # Gradio web UI
├── config.py
├── model.py
├── dataset.py
├── structure_selection.py
├── pdb_utils.py
├── rcsb_search.py
├── alphafold_utils.py
├── retrieval.py
├── cluster_sequences.py
├── alignment.py
├── data_quality.py
├── random_coil.py
├── physics_features.py
├── distance_features.py
├── spatial_neighbors.py
├── imputation_model.py
├── imputation_dataset.py
├── requirements.txt
├── README.md
├── data/
│   ├── chemical_shifts.csv          # BMRB shift data
│   ├── pairs.csv                    # BMRB -> PDB ID mappings
│   ├── structure_data_hybrid.csv    # Primary dataset (experimental + AlphaFold)
│   ├── structure_data_experimental.csv
│   ├── structure_data_alphafold.csv
│   ├── identity_clusters_90.json    # Sequence identity exclusion map
│   ├── esm_embeddings.h5            # ESM-2 embeddings
│   ├── retrieval_indices/           # Per-fold FAISS indices
│   ├── cache/                       # Pre-computed retrieval caches
│   ├── pdbs/                        # Downloaded PDB files
│   ├── alphafold/                   # Downloaded AlphaFold structures
│   └── build_log.csv               # Data provenance log
├── checkpoints/
│   ├── best_retrieval_fold1.pt
│   ├── best_imputation_fold1.pt
│   └── training_provenance_fold1.json
└── eval_results/
    ├── evaluation_fold1.json
    ├── per_shift_metrics_fold1.csv
    └── plots_fold1/
```
