# Night 2 Report — March 19-20, 2026

## What Happened Today (Day Session)

### Mechanistic Interpretability Findings

1. **The attention head was dead on ALL 49 shifts.** It produced predictions worse than structure-only on every single shift type. The direct gate learned a near-constant ~0.8 to suppress it. Both components (attn_head, direct_gate) were removed from the model.

2. **The direct transfer + retrieval gate ensemble works by bias cancellation**, not by retrieval providing better information. On backbone shifts, the neighbor mean is closer to truth than struct_pred only 38-46% of the time (worse than coin flip). Yet the gated blend improves predictions because struct systematically overshoots one direction and direct undershoots the other.

3. **SER CB has a +4.5 ppm systematic bias in struct_head** — 98.9% of predictions are too high. This is because global CB normalization (mean=37.8, std=12.7) forces the model to bridge ALA CB at 19 and THR CB at 70 through one output neuron. Per-AA normalization implemented to fix this.

4. **N-shift is dominated by amino acid identity and DSSP, NOT distances.** Ablating all 1413 intra-residue distances changes N MAE by 0.0004 ppm. The model uses residue type (+0.89 ppm) and DSSP (+0.39 ppm) for N prediction.

5. **i-1 window position matters +0.24 ppm for N** when completely ablated. The CNN does extract some preceding residue information, but given the 5+ ppm range of N values across preceding residue types, there's room to improve.

6. **The model has ZERO inter-residue distance features.** All 1413 distance columns are intra-residue. There are ~28 atoms from neighboring residues within 5A of each N, and the model can't see any of them. The spatial neighbor module gives CA-CA distances but not atom-level cross-residue geometry.

### Architecture Changes Made

1. **Removed attn_head and direct_gate** — dead components confirmed by analysis
2. **Joint distance attention for spatial neighbors** — query + neighbor intra-residue distances + CA-CA bridge are now processed together in one attention computation, enabling inter-residue geometry learning
3. **Canonical 88-atom vocabulary** in config.py — all datasets use same atoms, models are interchangeable
4. **Per-AA normalization** — per-(amino_acid, shift_type) mean/std computed and patched into all cache configs
5. **Bigger struct_head** — 1472→1024→512→256→49 (pending, will activate for next training run)
6. **`main.py`** — unified pipeline with `--data {hybrid,alphafold,experimental}`
7. **`inference.py`** — single-PDB shift prediction

### Data Pipeline

- Added DSSP columns to experimental CSV (was missing all 9 DSSP cols — caused N-shift to be disproportionately bad)
- Added pLDDT column to AlphaFold CSV (for future loss weighting)
- Set up data_experimental/ directory with caches
- All files moved to 1TB with symlinks (retrieval indices, CSVs, PDBs, embeddings)

## What's Running Tonight

1. **Hybrid training** (main.py --data hybrid --fold 1 --epochs 150) — at epoch ~80, ETA ~5 AM
2. **Experimental cache rebuild** — rebuilding all 5 folds with DSSP columns
3. **Physics-only prototype** — atom-level shift prediction using only local 3D geometry (CPU)
4. **Morning alarm** — 7:30 AM

## Plan After Training Finishes

1. Put bigger struct_head back
2. Start experimental training with DSSP + per-AA normalization
3. Eventually: rebuild alphafold cache with canonical atoms and pLDDT

## Key Insight for Future Direction

The physics-only approach (predict per-atom from local 3D neighborhood) would eliminate:
- The bimodal sidechain carbon problem (same atom in different AA = different neighbors)
- The need for residue boundaries
- The missing inter-residue distance problem

But anisotropic effects (ring currents, C=O magnetic anisotropy) depend on orientation, not just distance. Distances to enough non-coplanar neighbors implicitly determine orientation, so a sufficiently expressive model could learn this — but it's an open question whether it actually would.
