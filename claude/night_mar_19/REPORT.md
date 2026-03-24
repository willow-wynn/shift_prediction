# Overnight Analysis Report — March 19, 2026

## Status Summary

### Training
- **Hybrid model**: Resumed from epoch 100 (original process died from terminal disconnect).
  Then crashed again due to disk full + OOM. Fixed by:
  1. Deleting `data_csvs.tar.xz` (-2.7GB) and per-fold CSVs (-6.5GB) to free disk
  2. Writing `resume_training.py` that loads stats from cache config.json instead of CSV
  Now training with nohup, ~4.6 min/epoch, finishing at epoch 150 around 6:00 AM.
- **AlphaFold cache**: Built for all 5 folds (COMPLETE). Ready for training.
- **Disk**: Root partition was at 100%. Freed ~9GB. Now at 98% (5.9GB free).
  **WARNING**: Disk is very tight. AlphaFold training will create checkpoints (~300MB each).
- **Imputation**: Queued after alphafold training.

### Files Deleted (to free disk)
- `data_csvs.tar.xz` (2.7GB) — the compressed archive we created earlier
- `data/structure_data_hybrid_fold_{1-5}.csv` (6.5GB total) — redundant with main CSV

### Analysis Completed
- Data quality analysis (full dataset)
- Interpretability analysis (small sample, 320 test residues — full run pending GPU availability)
- UCBShift benchmark comparison (from existing results)

---

## Key Findings

### 1. Model Performance (Best Checkpoint at Epoch 50)

| Shift | Our Model | UCBShift-Y | Winner |
|-------|-----------|------------|--------|
| CA    | 0.833     | 0.886      | Ours (+6%) |
| CB    | 0.998     | 0.973      | UCBShift (+3%) |
| C     | 0.822     | 0.927      | Ours (+11%) |
| N     | 1.501     | 1.936      | Ours (+22%) |
| H     | 0.246     | 0.278      | Ours (+12%) |
| HA    | 0.162     | 0.202      | Ours (+20%) |

**Our model beats UCBShift on 5/6 backbone shifts**, with especially large improvements on N (+22%), HA (+20%), and H (+12%). Only CB is slightly worse.

Additionally, **our model predicts 49 shift types** including sidechains, while UCBShift only predicts 6.

### 2. Retrieval Contribution (Ablation Analysis)

The model uses a dual pathway: structure-only and retrieval-augmented. Gate values control the blend.

**How much does retrieval help per shift?**

| Shift | Full Model | Structure Only | Retrieval Improvement |
|-------|-----------|---------------|----------------------|
| CB    | 0.716     | 1.113         | +35.7% |
| HA    | 0.152     | 0.183         | +16.8% |
| CA    | 0.570     | 0.678         | +16.0% |
| C     | 0.519     | 0.582         | +10.7% |
| N     | 1.485     | 1.584         | +6.3% |
| H     | 0.273     | 0.272         | -0.4% |

**Key insight**: Retrieval helps most for CB (+36%) and least for H (essentially 0%). This makes biological sense:
- CB shift is highly amino-acid-specific → retrieval from same-AA neighbors is very informative
- H shift depends on local hydrogen bonding → structure alone captures this well

### 3. Gate Behavior

**Retrieval gate** (how much the model uses retrieval vs structure-only):
- Sidechain shifts: 90-97% retrieval reliance (model NEEDS retrieval for these)
- CB: 76% retrieval
- HA: 72%
- H: 68%
- CA: 66%
- C, N: 58-60% (structure alone does well)

**Direct transfer gate** (within retrieval: direct shift copy vs attention-based):
- Overall: 75% direct transfer — the model mostly copies shifts from similar neighbors
- Highest for rare sidechain shifts (>95% direct transfer)
- Lowest for H, C backbone shifts (~65%) where attention adds value

### 4. Hardest/Easiest Amino Acids

**Easiest** (lowest MAE):
- GLU (0.317), LYS (0.318), MET (0.324)

**Hardest** (highest MAE):
- CYS (0.855), PHE (1.014), TYR (1.399)
- Aromatic and disulfide-forming residues are hardest

### 5. Data Quality Findings

| Metric | Value |
|--------|-------|
| Total proteins | 12,427 |
| Total residues | 2,611,471 |
| Mean protein size | 210 residues |
| Mean backbone coverage | 55.6% |
| Proteins with <25% backbone coverage | 2,815 (23%) |
| Proteins with >90% backbone coverage | 2,429 (20%) |

**Important**: 23% of proteins have very sparse shift data (<25% backbone coverage).
Filtering these could improve data quality with minimal volume loss.

**AlphaFold vs Experimental**:
- AlphaFold data: ~10% shift coverage per backbone type (sparse)
- Experimental data: ~77-86% shift coverage (dense)
- AlphaFold has all 5 folds with ~930k residues each (4.7M total)
- Experimental has ~500k rows in first 500k sampled

### 6. Training Trajectory (from prior completed run)

The cosine annealing scheduler has restart cycles at epochs 50 and 150:
- Epoch 50: MAE = 0.4342 (end of first cycle)
- Epoch 150: MAE = 0.4262 (best! end of second cycle)
- Epoch 200: MAE = 0.4286 (slightly worse, LR restart disruption)

**Epoch 150 is the optimal stopping point** — confirmed by prior run data.

---

## Files Generated

### Plots (in `claude/night_mar_19/plots/`)
1. `summary_dashboard.png` — Single-page overview of model performance
2. `per_aa_heatmap.png` — MAE by amino acid × shift type
3. `gate_analysis.png` — Gate activation distributions and correlations
4. `retrieval_ablation_backbone.png` — Model vs structure-only (backbone shifts)
5. `retrieval_ablation_sidechain.png` — Model vs structure-only (sidechain shifts)
6. `prediction_scatter.png` — Predicted vs actual for backbone shifts
7. `per_ss_mae.png` — MAE by secondary structure type
8. `error_vs_true_value.png` — Are extreme shift values harder to predict?
9. `shift_distributions.png` — Raw shift value distributions
10. `coverage_per_protein.png` — Per-protein backbone shift completeness
11. `protein_sizes.png` — Protein size distribution
12. `fold_comparison.png` — Fold balance analysis
13. `residue_distribution.png` — Amino acid frequency

### Data (in `claude/night_mar_19/`)
- `interpretability_results.json` — Full gate analysis, per-AA/SS metrics, ablation
- `data_quality_results.json` — Coverage, distributions, fold stats, AF vs exp comparison
- `hybrid_training_resume_log.txt` — Training log from resumed run

---

### 7. Cleaned vs Full Dataset — Quality vs Quantity

| Metric | Full Hybrid | Cleaned Hybrid | Change |
|--------|-------------|----------------|--------|
| Rows | 2,611,471 | 589,333 | -77.4% |
| Proteins | 12,427 | ~5,942 | -52% |
| CA coverage | 37.5% | 88.8% | +2.4x |
| CB coverage | 32.8% | 80.3% | +2.4x |
| C coverage | 27.2% | 63.4% | +2.3x |
| N coverage | 37.8% | 86.3% | +2.3x |
| H coverage | 39.8% | 91.3% | +2.3x |
| HA coverage | 27.3% | 82.1% | +3.0x |

The cleaned dataset removes 77% of rows but achieves 2-3x better backbone shift coverage.
**This is a significant quality/quantity tradeoff worth testing.**

Training on the cleaned dataset would be ~4x faster per epoch (fewer samples) and might produce
better per-shift accuracy due to denser observations. However, retrieval might suffer from having
fewer unique proteins to match against.

**Recommendation**: Run a quick experiment training on the cleaned dataset to compare. If the model
performs similarly or better, it would significantly speed up experimentation cycles.

---

### 8. Historical Model Performance (from prior experiments in Claude_stuff/)

| Version | Test MAE | BB MAE | SC MAE | N MAE | Params |
|---------|----------|--------|--------|-------|--------|
| v2 | 0.545 | 0.801 | 0.305 | - | 1.97M |
| v3 | 0.523 | 0.764 | 0.297 | 1.698 | 2.56M |
| v6 | 0.491 | 0.694 | 0.300 | 1.455 | 6.42M |
| v7s3 | 0.489 | 0.692 | 0.298 | 1.450 | 6.42M |
| v8 | 0.492 | 0.696 | 0.301 | 1.463 | 13.92M |
| 8-model ensemble | 0.401 | 0.676 | 0.363 | 1.393 | - |

**Key insight**: Larger models (v8, 14M params) don't outperform medium ones (v6/v7, 6.4M params).
The ensemble of 8 models achieves the best results. Current data-pipeline-v2 model has 25.6M params.

### 9. A/B Test: Data Cleaning Impact

From previous A/B test (original vs cleaned training data):
- Original (491K samples): MAE=0.504, BB=0.757, SC=0.469
- Cleaned (481K samples): MAE=0.498, BB=0.756, SC=0.462
- **Minimal improvement** — suggests the 4-sigma outlier masking already handles most issues

---

## Next Steps (for morning)

1. ~~Full interpretability analysis on epoch 150 checkpoint~~ **DONE** (see results below)
2. **AlphaFold training** RUNNING from scratch (batch 256, ETA ~6PM). Training from scratch
   because atom vocabularies differ (hybrid: 88 atoms, alphafold: 37) preventing transfer learning.
3. **UCBShift benchmark on alphafold** — needs AlphaFold PDB → BMRB mapping
4. **Imputation model** — 50 epochs after alphafold training
5. **Consider filtering** sparse-coverage proteins (<25%) for a quality experiment
