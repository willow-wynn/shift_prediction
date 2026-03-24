# Morning Report — March 20, 2026

## What Happened Overnight

### Hybrid Training (COMPLETE)
- Trained with new architecture (removed dead attn_head/direct_gate, joint distance attention)
- **Best: Epoch 135, MAE 0.4671** (previous best was 0.5135)
- Checkpoints at epochs 75, 100, 125, 150 in `data/checkpoints/`

### Experimental Training (RUNNING)
- Started fresh with DSSP columns + bigger struct_head + per-AA normalization
- Currently at **epoch 10, MAE 0.4738**
- **N-shift: 1.776** — already better than previous experimental run's best of 1.820 at epoch 40
- DSSP is clearly helping the N-shift as predicted by the mechanistic analysis
- Running in background: `nohup python main.py --data experimental --fold 1 --epochs 150 --batch_size 512 --skip_cache`

### Experimental Cache Rebuilt
- All 5 folds with DSSP columns (n_dssp=9)
- Per-AA stats patched in
- Canonical 88-atom vocabulary

### Physics Prototype
- Kept getting OOM-killed competing with training for RAM
- Feature extraction (parsing PDBs + computing neighbor distances) was also painfully slow
- Fixed cdist approach (0.08s vs hours per protein)
- Still not trained — needs to run when RAM is free
- Will work once experimental training finishes or can be run separately

### Architecture Changes Active
1. **Removed attn_head + direct_gate** — confirmed dead by analysis
2. **Joint distance attention** — query + neighbor distances processed together
3. **Bigger struct_head** — 1472→1024→512→256→49 (4 layers, was 3)
4. **Per-AA normalization** — targets normalized per (amino_acid, shift_type)

## Key Numbers

| Model | Overall MAE | N-shift | CA | CB | Architecture |
|-------|-----------|---------|----|----|-------------|
| Old hybrid (ep 150) | 0.514 | 1.431 | 0.812 | 0.981 | Old (with dead attn_head) |
| New hybrid (ep 135) | **0.467** | ? | ? | ? | New (joint dist, no attn_head) |
| New experimental (ep 10) | 0.474 | 1.776 | 0.862 | 0.927 | New + DSSP + bigger head |
| Old experimental (ep 40) | 0.471 | 1.820 | 0.886 | 0.924 | Old, no DSSP |

## What's Running
- Experimental training: epoch 10/150, ~4.5 min/epoch, ETA ~5 PM today
- GPU: 97%, 25GB VRAM

## What Needs Doing
1. Let experimental training finish (or check progress when convenient)
2. Physics prototype — run when GPU/RAM is available
3. Rebuild alphafold cache with canonical atoms + pLDDT (low priority)
4. The per-AA normalization changes haven't been validated end-to-end yet on the eval side

## Files to Look At
- `claude/night_mar_19/REPORT_night2.md` — detailed findings from yesterday
- `claude/night_mar_19/plots/` — all interpretability charts
- `claude/night_mar_19/worst_predictions.json` — worst prediction analysis
- `claude/night_mar_19/n_shift_mechanistic.json` — N-shift mechanistic findings
