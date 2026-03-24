#!/usr/bin/env python3
"""
Find the worst predictions from the hybrid model and diagnose what went wrong.

Analyzes:
1. Top N worst absolute errors across all shifts
2. Per-shift worst predictions
3. Patterns: amino acid, secondary structure, retrieval quality, neighbor count
4. Attempts to explain WHY each prediction was bad
"""

import gc
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
from torch.amp import autocast
from contextlib import nullcontext
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict

from config import (
    STANDARD_RESIDUES, RESIDUE_TO_IDX, AA_3_TO_1,
    SS_TYPES, N_RESIDUE_TYPES, K_RETRIEVED,
)
from dataset import CachedRetrievalDataset
from model import ShiftPredictorWithRetrieval

BACKBONE_SHIFTS = {'ca_shift', 'cb_shift', 'c_shift', 'n_shift', 'h_shift', 'ha_shift'}
IDX_TO_AA = {v: k for k, v in RESIDUE_TO_IDX.items()}


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    clean_sd = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    n_atom_types = clean_sd['distance_attention.atom_embed.weight'].shape[0] - 1
    struct_keys = sorted([k for k in clean_sd if k.startswith('struct_head.') and k.endswith('.weight')])
    n_shifts = clean_sd[struct_keys[-1]].shape[0]
    n_dssp = clean_sd['dssp_proj.weight'].shape[1] if 'dssp_proj.weight' in clean_sd else 0
    cnn_channels = []
    for i in range(0, 10, 2):
        key = f'cnn.{i}.conv1.weight'
        if key in clean_sd:
            cnn_channels.append(clean_sd[key].shape[0])
    spatial_hidden = clean_sd.get('spatial_attention.fallback_embed', torch.zeros(192)).shape[0]
    retrieval_hidden = clean_sd.get('cross_attn_layers.0.shift_embed.weight', torch.zeros(1, 320)).shape[-1]

    from config import N_SS_TYPES, N_MISMATCH_TYPES
    model = ShiftPredictorWithRetrieval(
        n_atom_types=n_atom_types, n_residue_types=N_RESIDUE_TYPES,
        n_ss_types=N_SS_TYPES, n_mismatch_types=N_MISMATCH_TYPES,
        n_dssp=n_dssp, n_shifts=n_shifts, cnn_channels=cnn_channels,
        spatial_hidden=spatial_hidden, retrieval_hidden=retrieval_hidden,
    ).to(device)

    filtered = {k: v for k, v in clean_sd.items() if not k.startswith('physics_encoder.')}
    model.load_state_dict(filtered, strict=False)
    model.eval()
    return model, checkpoint


def collect_all_predictions(model, loader, device, stats, shift_cols):
    """Collect predictions with full metadata for error analysis."""
    model.eval()

    # Hook gates
    direct_gate_vals = []
    retrieval_gate_vals = []
    struct_pred_vals = []

    def hook_dg(m, inp, out): direct_gate_vals.append(out.detach().cpu())
    def hook_rg(m, inp, out): retrieval_gate_vals.append(out.detach().cpu())
    def hook_sp(m, inp, out): struct_pred_vals.append(out.detach().cpu())

    h1 = model.direct_gate.register_forward_hook(hook_dg)
    h2 = model.retrieval_gate.register_forward_hook(hook_rg)
    h3 = model.struct_head.register_forward_hook(hook_sp)

    all_data = {
        'pred_norm': [], 'target_norm': [], 'mask': [],
        'residue_code': [], 'ss_code': [],
        'ret_valid': [], 'ret_distances': [],
        'direct_gate': [], 'retrieval_gate': [], 'struct_pred_norm': [],
        'global_idx': [], 'prot_idx': [],
    }

    sample_offset = 0
    try:
        for batch in tqdm(loader, desc="Collecting predictions"):
            batch_dev = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            target = batch_dev.pop('shift_target')
            mask = batch_dev.pop('shift_mask')

            direct_gate_vals.clear()
            retrieval_gate_vals.clear()
            struct_pred_vals.clear()

            with torch.no_grad():
                ctx = autocast('cuda') if device == 'cuda' else nullcontext()
                with ctx:
                    pred = model(**batch_dev)

            bs = pred.shape[0]
            all_data['pred_norm'].append(pred.cpu().numpy())
            all_data['target_norm'].append(target.cpu().numpy())
            all_data['mask'].append(mask.cpu().numpy())
            all_data['residue_code'].append(batch_dev['query_residue_code'].cpu().numpy())
            all_data['ss_code'].append(batch_dev['ss_idx'][:, batch_dev['ss_idx'].shape[1]//2].cpu().numpy())
            all_data['ret_valid'].append(batch_dev['retrieved_valid'].cpu().numpy())
            all_data['ret_distances'].append(batch_dev['retrieved_distances'].cpu().numpy())

            if direct_gate_vals:
                all_data['direct_gate'].append(direct_gate_vals[0].squeeze(-1).numpy())
            if retrieval_gate_vals:
                all_data['retrieval_gate'].append(retrieval_gate_vals[0].squeeze(-1).numpy())
            if struct_pred_vals:
                all_data['struct_pred_norm'].append(struct_pred_vals[0].numpy())

            sample_offset += bs

    finally:
        h1.remove(); h2.remove(); h3.remove()

    # Concatenate
    for k in all_data:
        if all_data[k]:
            all_data[k] = np.concatenate(all_data[k], axis=0)
        else:
            all_data[k] = np.array([])

    # Denormalize
    pred = all_data['pred_norm'].copy()
    target = all_data['target_norm'].copy()
    struct_pred = all_data['struct_pred_norm'].copy() if len(all_data['struct_pred_norm']) > 0 else None
    for si, col in enumerate(shift_cols):
        if col in stats:
            pred[:, si] = pred[:, si] * stats[col]['std'] + stats[col]['mean']
            target[:, si] = target[:, si] * stats[col]['std'] + stats[col]['mean']
            if struct_pred is not None:
                struct_pred[:, si] = struct_pred[:, si] * stats[col]['std'] + stats[col]['mean']

    all_data['pred'] = pred
    all_data['target'] = target
    all_data['errors'] = np.abs(pred - target)
    if struct_pred is not None:
        all_data['struct_pred'] = struct_pred
        all_data['struct_errors'] = np.abs(struct_pred - target)

    return all_data


def analyze_worst(data, shift_cols, stats, dataset, top_n=50):
    """Find and analyze the worst predictions."""
    mask = data['mask'].astype(bool)
    errors = data['errors']

    print(f"\n{'='*80}")
    print(f"WORST PREDICTION ANALYSIS ({len(data['pred']):,} samples)")
    print(f"{'='*80}")

    # ========== Overall worst across all shifts ==========
    print(f"\n--- TOP {top_n} WORST PREDICTIONS (all shifts) ---")
    print(f"{'Rank':>4} {'Shift':>15} {'Pred':>8} {'True':>8} {'Error':>8} "
          f"{'AA':>4} {'SS':>3} {'#Nbr':>5} {'RGate':>6} {'DGate':>6} "
          f"{'StructE':>8} {'SampleIdx':>10}")

    # Flatten to find worst
    worst_list = []
    for si, col in enumerate(shift_cols):
        col_mask = mask[:, si]
        if col_mask.sum() == 0:
            continue
        sample_indices = np.where(col_mask)[0]
        col_errors = errors[sample_indices, si]
        for rank_in_col, idx_in_masked in enumerate(np.argsort(-col_errors)):
            sample_idx = sample_indices[idx_in_masked]
            worst_list.append({
                'sample_idx': int(sample_idx),
                'shift_idx': si,
                'shift_col': col,
                'error': float(col_errors[idx_in_masked]),
                'pred': float(data['pred'][sample_idx, si]),
                'true': float(data['target'][sample_idx, si]),
                'aa_idx': int(data['residue_code'][sample_idx]),
                'ss_idx': int(data['ss_code'][sample_idx]),
                'n_valid_neighbors': int(data['ret_valid'][sample_idx].sum()),
                'rgate': float(data['retrieval_gate'][sample_idx, si]) if len(data['retrieval_gate']) > 0 else 0,
                'dgate': float(data['direct_gate'][sample_idx, si]) if len(data['direct_gate']) > 0 else 0,
                'struct_error': float(data['struct_errors'][sample_idx, si]) if 'struct_errors' in data else 0,
            })

    worst_list.sort(key=lambda x: -x['error'])

    for rank, w in enumerate(worst_list[:top_n]):
        aa = IDX_TO_AA.get(w['aa_idx'], '?')
        ss = SS_TYPES[w['ss_idx']] if w['ss_idx'] < len(SS_TYPES) else '?'
        is_bb = '*' if w['shift_col'] in BACKBONE_SHIFTS else ' '
        print(f"{rank+1:>4} {is_bb}{w['shift_col']:>14} {w['pred']:>8.2f} {w['true']:>8.2f} "
              f"{w['error']:>8.2f} {aa:>4} {ss:>3} {w['n_valid_neighbors']:>5} "
              f"{w['rgate']:>6.3f} {w['dgate']:>6.3f} {w['struct_error']:>8.2f} {w['sample_idx']:>10}")

    # ========== Per-backbone-shift worst ==========
    print(f"\n--- TOP 10 WORST PER BACKBONE SHIFT ---")
    bb_cols = sorted([c for c in shift_cols if c in BACKBONE_SHIFTS])
    for col in bb_cols:
        si = shift_cols.index(col)
        col_mask = mask[:, si]
        if col_mask.sum() == 0:
            continue
        sample_indices = np.where(col_mask)[0]
        col_errors = errors[sample_indices, si]
        top10_idx = np.argsort(-col_errors)[:10]

        name = col.replace('_shift', '').upper()
        print(f"\n  {name} (n={col_mask.sum():,}, mean MAE={col_errors.mean():.3f}, median={np.median(col_errors):.3f}):")
        print(f"  {'Rank':>4} {'Pred':>8} {'True':>8} {'Error':>8} {'AA':>5} {'SS':>3} "
              f"{'#Nbr':>5} {'RGate':>6} {'StructErr':>9}")

        for rank, idx in enumerate(top10_idx):
            si_idx = sample_indices[idx]
            aa = IDX_TO_AA.get(int(data['residue_code'][si_idx]), '?')
            ss = SS_TYPES[int(data['ss_code'][si_idx])] if int(data['ss_code'][si_idx]) < len(SS_TYPES) else '?'
            rg = data['retrieval_gate'][si_idx, si] if len(data['retrieval_gate']) > 0 else 0
            se = data['struct_errors'][si_idx, si] if 'struct_errors' in data else 0
            n_nbr = int(data['ret_valid'][si_idx].sum())
            print(f"  {rank+1:>4} {data['pred'][si_idx, si]:>8.2f} {data['target'][si_idx, si]:>8.2f} "
                  f"{col_errors[idx]:>8.2f} {aa:>5} {ss:>3} {n_nbr:>5} {rg:>6.3f} {se:>9.2f}")

    # ========== Pattern analysis ==========
    print(f"\n--- ERROR PATTERN ANALYSIS ---")

    # By amino acid
    print(f"\n  Per-AA error breakdown (backbone shifts only):")
    bb_indices = [shift_cols.index(c) for c in shift_cols if c in BACKBONE_SHIFTS]
    aa_errors = defaultdict(list)
    for i in range(len(data['pred'])):
        aa = IDX_TO_AA.get(int(data['residue_code'][i]), 'UNK')
        for si in bb_indices:
            if mask[i, si]:
                aa_errors[aa].append(errors[i, si])

    print(f"  {'AA':>5} {'Mean':>8} {'Median':>8} {'P95':>8} {'P99':>8} {'Max':>8} {'Count':>8}")
    for aa in sorted(aa_errors.keys(), key=lambda x: -np.mean(aa_errors[x])):
        errs = np.array(aa_errors[aa])
        print(f"  {aa:>5} {np.mean(errs):>8.3f} {np.median(errs):>8.3f} "
              f"{np.percentile(errs, 95):>8.3f} {np.percentile(errs, 99):>8.3f} "
              f"{np.max(errs):>8.3f} {len(errs):>8}")

    # By retrieval quality
    print(f"\n  Error by number of valid retrieval neighbors:")
    n_valid = data['ret_valid'].sum(axis=1)
    for lo, hi in [(0, 0), (1, 5), (6, 15), (16, 25), (26, 32)]:
        bin_mask = (n_valid >= lo) & (n_valid <= hi)
        if bin_mask.sum() == 0:
            continue
        bin_errors = []
        for i in np.where(bin_mask)[0]:
            for si in bb_indices:
                if mask[i, si]:
                    bin_errors.append(errors[i, si])
        if bin_errors:
            be = np.array(bin_errors)
            print(f"  {lo:>2}-{hi:>2} neighbors: mean={np.mean(be):.3f} median={np.median(be):.3f} "
                  f"p99={np.percentile(be, 99):.3f} max={np.max(be):.3f} (n={len(be):,})")

    # Retrieval gate vs error
    if len(data['retrieval_gate']) > 0:
        print(f"\n  Error by retrieval gate value (backbone):")
        for lo, hi in [(0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]:
            gate_errors = []
            for si in bb_indices:
                gv = data['retrieval_gate'][:, si]
                m = mask[:, si] & (gv >= lo) & (gv < hi + 0.001)
                if m.sum() > 0:
                    gate_errors.extend(errors[:, si][m].tolist())
            if gate_errors:
                ge = np.array(gate_errors)
                print(f"  gate [{lo:.1f}-{hi:.1f}): mean={np.mean(ge):.3f} "
                      f"p99={np.percentile(ge, 99):.3f} max={np.max(ge):.3f} (n={len(ge):,})")

    # Structure-only vs full model on worst cases
    if 'struct_errors' in data:
        print(f"\n  On the 1000 worst predictions:")
        worst_indices = []
        for w in worst_list[:1000]:
            worst_indices.append((w['sample_idx'], w['shift_idx']))

        model_worse = 0
        struct_worse = 0
        for si_idx, si in worst_indices:
            me = errors[si_idx, si]
            se = data['struct_errors'][si_idx, si]
            if me > se:
                model_worse += 1
            else:
                struct_worse += 1

        print(f"    Full model worse than struct-only: {model_worse} / {len(worst_indices)}")
        print(f"    Full model better than struct-only: {struct_worse} / {len(worst_indices)}")
        print(f"    -> Retrieval HURTS on {model_worse/len(worst_indices)*100:.1f}% of worst predictions")

    # Look at proteins with worst performance
    print(f"\n  Attempting per-protein analysis...")
    if hasattr(dataset, 'samples'):
        prot_errors = defaultdict(list)
        for i in range(min(len(data['pred']), len(dataset.samples))):
            _, prot_idx = dataset.samples[i]
            for si in bb_indices:
                if mask[i, si]:
                    prot_errors[int(prot_idx)].append(errors[i, si])

        prot_mae = {p: np.mean(e) for p, e in prot_errors.items() if len(e) > 5}
        worst_prots = sorted(prot_mae.items(), key=lambda x: -x[1])[:10]

        print(f"\n  Top 10 worst proteins:")
        print(f"  {'ProtIdx':>8} {'MAE':>8} {'MaxErr':>8} {'nRes':>6}")
        for pidx, mae in worst_prots:
            errs = prot_errors[pidx]
            print(f"  {pidx:>8} {mae:>8.3f} {max(errs):>8.3f} {len(errs):>6}")

    return worst_list


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--cache_dir', default='data/cache')
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--device', default=None)
    parser.add_argument('--top_n', type=int, default=50)
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    if args.checkpoint is None:
        for candidate in ['data/checkpoints/best_retrieval_fold1.pt',
                          'data/checkpoints/best_fold1.pt']:
            if os.path.exists(candidate):
                args.checkpoint = candidate
                break
        if args.checkpoint is None:
            print("ERROR: No checkpoint found")
            sys.exit(1)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")

    model, ckpt = load_model(args.checkpoint, device)
    stats = ckpt.get('stats', {})
    shift_cols = ckpt.get('shift_cols', [])
    k_retrieved = ckpt.get('k_retrieved', K_RETRIEVED)

    test_cache = os.path.join(args.cache_dir, f'fold_{args.fold}')
    dataset = CachedRetrievalDataset.load(test_cache, len(shift_cols), k_retrieved,
                                          stats=stats, shift_cols=shift_cols)
    print(f"Test samples: {len(dataset):,}")

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=(device == 'cuda'))

    data = collect_all_predictions(model, loader, device, stats, shift_cols)

    del model
    if device == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()

    worst = analyze_worst(data, shift_cols, stats, dataset, top_n=args.top_n)

    # Save results
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'worst_predictions.json')
    save_data = {
        'top_50_worst': worst[:50],
        'checkpoint': args.checkpoint,
        'fold': args.fold,
        'n_samples': len(data['pred']),
    }
    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
