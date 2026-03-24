#!/usr/bin/env python3
"""
Mechanistic interpretability for N-shift predictions.

1. Feature ablation: zero out each input type, measure N-shift MAE change
2. Gradient attribution: which inputs drive N predictions most
3. Good-vs-bad comparison: what's different about inputs when N is wrong
4. Attention probing: what do the spatial/retrieval attention weights look like
"""

import gc
import os
import sys
import json
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast
from contextlib import nullcontext
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import (
    STANDARD_RESIDUES, RESIDUE_TO_IDX, AA_3_TO_1,
    SS_TYPES, N_RESIDUE_TYPES, K_RETRIEVED, DSSP_COLS,
)
from dataset import CachedRetrievalDataset
from model import ShiftPredictorWithRetrieval

PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
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


# ============================================================================
# 1. Feature Ablation
# ============================================================================

def run_ablation(model, loader, device, stats, shift_cols, n_shift_idx, max_batches=50):
    """Zero out each input type and measure impact on N-shift predictions."""

    ablation_configs = {
        'baseline': {},  # no ablation
        'no_distances': {'distances': True, 'neighbor_distances': True},
        'no_dssp': {'dssp_features': True},
        'no_spatial_neighbors': {'neighbor_valid': True},
        'no_retrieval': {'retrieved_valid': True},
        'no_residue_type': {'residue_idx': True},
        'no_ss_type': {'ss_idx': True},
        'no_query_struct': {'query_struct': True},
        'no_neighbor_struct': {'neighbor_struct': True},
    }

    n_mean = stats[shift_cols[n_shift_idx]]['mean']
    n_std = stats[shift_cols[n_shift_idx]]['std']

    results = {}
    for config_name, ablations in ablation_configs.items():
        all_errors = []
        model.eval()

        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            target = batch.pop('shift_target')
            mask = batch.pop('shift_mask')

            # Apply ablations
            for key in ablations:
                if key in batch:
                    if key == 'neighbor_valid':
                        batch[key] = torch.zeros_like(batch[key])
                    elif key == 'retrieved_valid':
                        batch[key] = torch.zeros_like(batch[key])
                    elif key == 'residue_idx':
                        # Set to UNK token
                        batch[key] = torch.full_like(batch[key], N_RESIDUE_TYPES)
                    elif key == 'ss_idx':
                        from config import N_SS_TYPES
                        batch[key] = torch.full_like(batch[key], N_SS_TYPES)
                    else:
                        batch[key] = torch.zeros_like(batch[key])

            with torch.no_grad():
                ctx = autocast('cuda') if device == 'cuda' else nullcontext()
                with ctx:
                    pred = model(**batch)

            n_mask = mask[:, n_shift_idx]
            if n_mask.sum() > 0:
                pred_n = pred[:, n_shift_idx][n_mask] * n_std + n_mean
                true_n = target[:, n_shift_idx][n_mask] * n_std + n_mean
                all_errors.extend((pred_n - true_n).abs().cpu().numpy().tolist())

        mae = np.mean(all_errors) if all_errors else 0
        results[config_name] = {
            'mae': mae,
            'n_samples': len(all_errors),
        }
        print(f"  {config_name:25s}: N MAE = {mae:.4f} ({len(all_errors):,} samples)")

    return results


# ============================================================================
# 2. Gradient Attribution
# ============================================================================

def gradient_attribution(model, loader, device, stats, shift_cols, n_shift_idx, max_batches=20):
    """Compute gradient of N-shift prediction w.r.t. input features."""

    n_std = stats[shift_cols[n_shift_idx]]['std']

    gradient_norms = defaultdict(list)
    continuous_inputs = ['distances', 'dssp_features', 'query_struct', 'neighbor_struct',
                         'neighbor_dist', 'neighbor_angles']

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break

        batch = {k: v.to(device) for k, v in batch.items()}
        target = batch.pop('shift_target')
        mask = batch.pop('shift_mask')

        n_mask = mask[:, n_shift_idx]
        if n_mask.sum() == 0:
            continue

        # Enable gradients on continuous inputs
        for key in continuous_inputs:
            if key in batch:
                batch[key] = batch[key].detach().requires_grad_(True)

        pred = model(**batch)
        # Sum N predictions for masked samples
        n_preds = pred[:, n_shift_idx][n_mask]
        loss = n_preds.sum()
        loss.backward()

        # Collect gradient norms
        for key in continuous_inputs:
            if key in batch and batch[key].grad is not None:
                grad = batch[key].grad[n_mask]
                # Per-sample L2 norm of gradient
                if grad.dim() == 1:
                    gradient_norms[key].extend(grad.abs().cpu().numpy().tolist())
                else:
                    flat = grad.reshape(grad.shape[0], -1)
                    norms = flat.norm(dim=1).cpu().numpy()
                    gradient_norms[key].extend(norms.tolist())

        model.zero_grad()

    print("\n  Gradient attribution (mean |dN/d_input|):")
    results = {}
    for key in sorted(gradient_norms.keys(), key=lambda k: -np.mean(gradient_norms[k])):
        vals = np.array(gradient_norms[key])
        mean_norm = np.mean(vals)
        results[key] = {'mean_grad_norm': float(mean_norm), 'std': float(np.std(vals))}
        print(f"    {key:25s}: {mean_norm:.6f} (± {np.std(vals):.6f})")

    return results


# ============================================================================
# 3. Good vs Bad Comparison
# ============================================================================

def good_vs_bad_analysis(model, loader, device, stats, shift_cols, n_shift_idx, max_batches=100):
    """Compare input features for good vs bad N predictions."""
    model.eval()
    n_mean = stats[shift_cols[n_shift_idx]]['mean']
    n_std = stats[shift_cols[n_shift_idx]]['std']

    good_data = defaultdict(list)
    bad_data = defaultdict(list)
    threshold = 3.0  # ppm - errors above this are "bad"

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        target = batch.pop('shift_target')
        mask = batch.pop('shift_mask')

        with torch.no_grad():
            ctx = autocast('cuda') if device == 'cuda' else nullcontext()
            with ctx:
                pred = model(**batch)

        n_mask = mask[:, n_shift_idx].cpu().numpy().astype(bool)
        if n_mask.sum() == 0:
            continue

        pred_n = (pred[:, n_shift_idx].cpu().numpy() * n_std + n_mean)
        true_n = (target[:, n_shift_idx].cpu().numpy() * n_std + n_mean)
        errors = np.abs(pred_n - true_n)

        for i in range(len(errors)):
            if not n_mask[i]:
                continue

            bucket = bad_data if errors[i] > threshold else good_data

            # Collect features
            bucket['error'].append(errors[i])
            bucket['true_n'].append(true_n[i])
            bucket['pred_n'].append(pred_n[i])
            bucket['aa'].append(int(batch['query_residue_code'][i].cpu()))
            bucket['n_valid_nbr'].append(int(batch['retrieved_valid'][i].cpu().sum()))

            # DSSP features
            center = batch['dssp_features'].shape[1] // 2
            dssp = batch['dssp_features'][i, center].cpu().numpy()
            for di, dcol in enumerate(DSSP_COLS):
                bucket[f'dssp_{dcol}'].append(float(dssp[di]))

            # Mean retrieval cosine similarity
            ret_valid = batch['retrieved_valid'][i].cpu().numpy()
            ret_dist = batch['retrieved_distances'][i].cpu().numpy()
            if ret_valid.sum() > 0:
                bucket['mean_ret_sim'].append(float(ret_dist[ret_valid].mean()))
            else:
                bucket['mean_ret_sim'].append(0.0)

            # Distance feature coverage
            dist_mask = batch['dist_mask'][i, center].cpu().numpy()
            bucket['n_distances'].append(int(dist_mask.sum()))

            # Spatial neighbor count
            nb_valid = batch['neighbor_valid'][i].cpu().numpy()
            bucket['n_spatial_nbr'].append(int(nb_valid.sum()))

    print(f"\n  Good predictions (<{threshold} ppm error): {len(good_data['error']):,}")
    print(f"  Bad predictions  (>{threshold} ppm error): {len(bad_data['error']):,}")

    if not bad_data['error']:
        return {}

    # Compare distributions
    results = {}
    compare_keys = ['true_n', 'n_valid_nbr', 'mean_ret_sim', 'n_distances', 'n_spatial_nbr']
    compare_keys += [f'dssp_{c}' for c in DSSP_COLS]

    print(f"\n  {'Feature':>25s} {'Good(mean)':>12} {'Bad(mean)':>12} {'Ratio':>8}")
    print(f"  {'-'*60}")

    for key in compare_keys:
        if key not in good_data or key not in bad_data:
            continue
        g = np.array(good_data[key])
        b = np.array(bad_data[key])
        gm, bm = np.mean(g), np.mean(b)
        ratio = bm / gm if abs(gm) > 1e-6 else float('inf')
        results[key] = {'good_mean': float(gm), 'bad_mean': float(bm), 'ratio': float(ratio)}
        print(f"  {key:>25s} {gm:>12.4f} {bm:>12.4f} {ratio:>8.3f}")

    # AA distribution comparison
    print(f"\n  AA distribution (good vs bad):")
    good_aa = defaultdict(int)
    bad_aa = defaultdict(int)
    for a in good_data['aa']:
        good_aa[IDX_TO_AA.get(a, '?')] += 1
    for a in bad_data['aa']:
        bad_aa[IDX_TO_AA.get(a, '?')] += 1
    total_good = len(good_data['aa'])
    total_bad = len(bad_data['aa'])

    print(f"  {'AA':>5} {'Good%':>8} {'Bad%':>8} {'Enrichment':>12}")
    all_aas = sorted(set(list(good_aa.keys()) + list(bad_aa.keys())))
    enrichments = {}
    for aa in all_aas:
        gp = good_aa.get(aa, 0) / total_good * 100
        bp = bad_aa.get(aa, 0) / total_bad * 100
        enrich = bp / gp if gp > 0 else float('inf')
        enrichments[aa] = enrich
        if bp > 0.5:  # only show AAs with >0.5% of bad
            print(f"  {aa:>5} {gp:>8.1f} {bp:>8.1f} {enrich:>12.2f}x")

    # True N value distribution for bad predictions
    print(f"\n  True N-shift distribution for BAD predictions:")
    bad_true = np.array(bad_data['true_n'])
    for lo, hi in [(100, 110), (110, 115), (115, 120), (120, 125), (125, 130), (130, 140)]:
        n = ((bad_true >= lo) & (bad_true < hi)).sum()
        pct = n / len(bad_true) * 100
        print(f"    [{lo:>3}-{hi:>3}) ppm: {n:>6} ({pct:.1f}%)")

    results['aa_enrichments'] = enrichments
    return results


# ============================================================================
# 4. Attention Probing
# ============================================================================

def probe_attention(model, loader, device, stats, shift_cols, n_shift_idx, max_batches=30):
    """Probe spatial and retrieval attention patterns for N predictions."""
    model.eval()
    n_mean = stats[shift_cols[n_shift_idx]]['mean']
    n_std = stats[shift_cols[n_shift_idx]]['std']

    # Hook into spatial attention scores
    spatial_scores = []
    def hook_spatial(module, input, output):
        # Get attention weights from the spatial attention module
        # The module computes scores internally - we'll capture them
        pass

    # Instead: hook the retrieval cross-attention to get per-shift attention patterns
    cross_attn_weights = []
    def hook_cross_attn(module, input, output):
        cross_attn_weights.append(output.detach().cpu())

    # Hook the retrieval gate and direct gate for N specifically
    rgate_vals = []
    dgate_vals = []
    def hook_rg(m, i, o): rgate_vals.append(o.detach().cpu())
    def hook_dg(m, i, o): dgate_vals.append(o.detach().cpu())

    h_rg = model.retrieval_gate.register_forward_hook(hook_rg)
    h_dg = model.direct_gate.register_forward_hook(hook_dg)

    # Also capture struct_head output
    struct_preds = []
    def hook_sh(m, i, o): struct_preds.append(o.detach().cpu())
    h_sh = model.struct_head.register_forward_hook(hook_sh)

    good_gates = {'rg': [], 'dg': [], 'struct_error': [], 'full_error': []}
    bad_gates = {'rg': [], 'dg': [], 'struct_error': [], 'full_error': []}

    try:
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            target = batch.pop('shift_target')
            mask = batch.pop('shift_mask')

            rgate_vals.clear(); dgate_vals.clear(); struct_preds.clear()

            with torch.no_grad():
                ctx = autocast('cuda') if device == 'cuda' else nullcontext()
                with ctx:
                    pred = model(**batch)

            n_mask = mask[:, n_shift_idx].cpu().numpy().astype(bool)
            if n_mask.sum() == 0:
                continue

            pred_n = pred[:, n_shift_idx].cpu().numpy() * n_std + n_mean
            true_n = target[:, n_shift_idx].cpu().numpy() * n_std + n_mean
            errors = np.abs(pred_n - true_n)

            if struct_preds:
                sp = struct_preds[0][:, n_shift_idx].numpy() * n_std + n_mean
                struct_errors = np.abs(sp - true_n)

            if rgate_vals and dgate_vals:
                rg = rgate_vals[0].squeeze(-1)[:, n_shift_idx].numpy()
                dg = dgate_vals[0].squeeze(-1)[:, n_shift_idx].numpy()

                for i in range(len(errors)):
                    if not n_mask[i]:
                        continue
                    bucket = bad_gates if errors[i] > 3.0 else good_gates
                    bucket['rg'].append(rg[i])
                    bucket['dg'].append(dg[i])
                    bucket['full_error'].append(errors[i])
                    if struct_preds:
                        bucket['struct_error'].append(struct_errors[i])

    finally:
        h_rg.remove(); h_dg.remove(); h_sh.remove()

    print(f"\n  Gate behavior on N-shift (good <3ppm vs bad >3ppm):")
    for name, label in [('rg', 'Retrieval gate'), ('dg', 'Direct gate')]:
        if good_gates[name] and bad_gates[name]:
            gm = np.mean(good_gates[name])
            bm = np.mean(bad_gates[name])
            print(f"    {label:20s}: good={gm:.4f}  bad={bm:.4f}")

    if good_gates['struct_error'] and bad_gates['struct_error']:
        g_se = np.mean(good_gates['struct_error'])
        b_se = np.mean(bad_gates['struct_error'])
        g_fe = np.mean(good_gates['full_error'])
        b_fe = np.mean(bad_gates['full_error'])
        print(f"\n    On BAD N predictions:")
        print(f"      Struct-only MAE: {b_se:.3f}")
        print(f"      Full model MAE:  {b_fe:.3f}")
        print(f"      Retrieval helps: {b_se - b_fe:.3f} ppm ({(b_se-b_fe)/b_se*100:.1f}%)")
        print(f"\n    On GOOD N predictions:")
        print(f"      Struct-only MAE: {g_se:.3f}")
        print(f"      Full model MAE:  {g_fe:.3f}")
        print(f"      Retrieval helps: {g_se - g_fe:.3f} ppm ({(g_se-g_fe)/g_se*100:.1f}%)")

    return {'good_gates': {k: float(np.mean(v)) for k, v in good_gates.items() if v},
            'bad_gates': {k: float(np.mean(v)) for k, v in bad_gates.items() if v}}


# ============================================================================
# Plotting
# ============================================================================

def plot_ablation(ablation_results, plot_dir):
    baseline = ablation_results['baseline']['mae']
    configs = sorted(
        [(k, v) for k, v in ablation_results.items() if k != 'baseline'],
        key=lambda x: -(x[1]['mae'] - baseline)
    )

    names = [c[0].replace('no_', '').replace('_', ' ').title() for c in configs]
    impacts = [(c[1]['mae'] - baseline) for c in configs]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['red' if x > 0 else 'green' for x in impacts]
    bars = ax.barh(range(len(names)), impacts, color=colors, alpha=0.7, edgecolor='black')
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=11)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Change in N-shift MAE (ppm)', fontsize=12)
    ax.set_title(f'Feature Ablation: Impact on N-shift (baseline MAE={baseline:.3f})', fontsize=13)

    for i, (impact, bar) in enumerate(zip(impacts, bars)):
        ax.text(impact + 0.01 if impact > 0 else impact - 0.01,
                i, f'{impact:+.3f}', va='center',
                ha='left' if impact > 0 else 'right', fontsize=9)

    ax.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'n_shift_ablation.png'), dpi=200, bbox_inches='tight')
    plt.close()


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--cache_dir', default='data/cache')
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    if args.checkpoint is None:
        for c in ['data/checkpoints/best_retrieval_fold1.pt', 'data/checkpoints/best_fold1.pt']:
            if os.path.exists(c):
                args.checkpoint = c
                break

    print("=" * 70)
    print("  N-SHIFT MECHANISTIC INTERPRETABILITY")
    print("=" * 70)

    model, ckpt = load_model(args.checkpoint, device)
    stats = ckpt.get('stats', {})
    shift_cols = ckpt.get('shift_cols', [])
    k_retrieved = ckpt.get('k_retrieved', K_RETRIEVED)

    n_shift_idx = shift_cols.index('n_shift')
    print(f"  N-shift is column index {n_shift_idx}")
    print(f"  N-shift stats: mean={stats['n_shift']['mean']:.2f}, std={stats['n_shift']['std']:.2f}")

    test_cache = os.path.join(args.cache_dir, f'fold_{args.fold}')
    dataset = CachedRetrievalDataset.load(test_cache, len(shift_cols), k_retrieved,
                                          stats=stats, shift_cols=shift_cols)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=(device == 'cuda'))

    # 1. Feature ablation
    print(f"\n{'='*70}")
    print("1. FEATURE ABLATION")
    print(f"{'='*70}")
    ablation = run_ablation(model, loader, device, stats, shift_cols, n_shift_idx)

    # 2. Gradient attribution
    print(f"\n{'='*70}")
    print("2. GRADIENT ATTRIBUTION")
    print(f"{'='*70}")
    gradients = gradient_attribution(model, loader, device, stats, shift_cols, n_shift_idx)

    # 3. Good vs bad comparison
    print(f"\n{'='*70}")
    print("3. GOOD vs BAD N PREDICTIONS")
    print(f"{'='*70}")
    comparison = good_vs_bad_analysis(model, loader, device, stats, shift_cols, n_shift_idx)

    # 4. Attention/gate probing
    print(f"\n{'='*70}")
    print("4. GATE PROBING")
    print(f"{'='*70}")
    gate_analysis = probe_attention(model, loader, device, stats, shift_cols, n_shift_idx)

    # Plot
    plot_ablation(ablation, PLOT_DIR)
    print(f"\n  Saved n_shift_ablation.png")

    # Save all results
    all_results = {
        'ablation': ablation,
        'gradients': gradients,
        'good_vs_bad': comparison,
        'gate_analysis': gate_analysis,
    }
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'n_shift_mechanistic.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Saved {out_path}")


if __name__ == '__main__':
    main()
