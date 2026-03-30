#!/usr/bin/env python3
"""
Compare retrieval quality: ESM embeddings vs structure model embeddings.

For fold 1 test residues:
1. Extract base_encoding from trained structure model
2. Build a temporary FAISS index from structure embeddings (excluding fold 1)
3. Retrieve neighbors using structure embeddings
4. Compare neighbor shift quality against existing ESM-based neighbors

Metric: for each query residue with known shifts, how close are the
retrieved neighbors' shifts to the true shifts? Lower MAE = better neighbors.
"""

import argparse
import json
import os
import sys
import time

import faiss
import h5py
import numpy as np
import pickle
import torch
from collections import defaultdict
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    K_RETRIEVED, FAISS_NPROBE, N_BOND_GEOM,
    STANDARD_RESIDUES, RESIDUE_TO_IDX,
)
from dataset import CachedRetrievalDataset
from inference import load_model


def extract_embeddings_from_cache(model, cache_dir, device, batch_size=512):
    """Extract base_encoding for every residue in a fold cache.

    Returns:
        embeddings: (total_residues, embed_dim) float32
        bmrb_ids: list of bmrb_id strings per residue
        residue_ids: list of residue_id ints per residue
    """
    # Load dataset
    config_path = os.path.join(cache_dir, 'config.json')
    with open(config_path) as f:
        cache_config = json.load(f)

    n_shifts = cache_config['n_shifts']
    total_residues = cache_config['total_residues']
    stats = cache_config.get('stats', {})
    shift_cols = cache_config.get('shift_cols', [])

    ds = CachedRetrievalDataset.load(
        cache_dir, n_shifts, K_RETRIEVED,
        stats=stats, shift_cols=shift_cols,
    )

    # Override samples to cover ALL residues (not just those with shifts)
    protein_offsets = ds.protein_offsets.numpy()
    n_proteins = len(protein_offsets)
    all_samples = []
    for prot_idx in range(n_proteins):
        start = int(protein_offsets[prot_idx])
        end = int(protein_offsets[prot_idx + 1]) if prot_idx + 1 < n_proteins else total_residues
        for gidx in range(start, end):
            all_samples.append((gidx, prot_idx))
    ds.samples = np.array(all_samples, dtype=np.int32)

    # Hook to capture base_encoding
    captured = {}

    def hook_fn(module, inp, out):
        captured['base_encoding'] = inp[0].detach().cpu()

    handle = model.struct_head[0].register_forward_hook(hook_fn)

    # Extract
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

    all_embeddings = []
    all_bmrb = []
    all_resid = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"  Extracting from {cache_dir}"):
            # Move to device
            batch_dev = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            _ = batch_dev.pop('shift_target', None)
            _ = batch_dev.pop('shift_mask', None)

            # Forward pass
            model(**batch_dev)

            # Capture base_encoding
            be = captured['base_encoding']  # (B, 1472)
            all_embeddings.append(be.numpy())

    handle.remove()

    embeddings = np.concatenate(all_embeddings, axis=0)  # (total_residues, 1472)

    # Map indices to (bmrb_id, residue_id)
    for gidx in range(total_residues):
        bmrb_id = ds.idx_to_bmrb.get(str(gidx), 'unknown')
        rid = ds.global_to_resid.get(str(gidx), -1)
        all_bmrb.append(str(bmrb_id))
        all_resid.append(int(rid))

    return embeddings, all_bmrb, all_resid


def build_index(embeddings, n_list=None):
    """Build a FAISS IVF index from embeddings."""
    n, d = embeddings.shape
    embeddings = embeddings.copy().astype(np.float32)
    faiss.normalize_L2(embeddings)

    if n_list is None:
        n_list = min(4096, max(100, n // 100))

    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, n_list, faiss.METRIC_INNER_PRODUCT)

    # Train
    train_size = min(500000, n)
    if train_size < n:
        train_idx = np.random.choice(n, train_size, replace=False)
        train_data = embeddings[train_idx]
    else:
        train_data = embeddings
    index.train(train_data)
    index.add(embeddings)
    index.nprobe = FAISS_NPROBE

    return index


def compare_neighbors(
    query_shifts, query_masks, query_aa_codes,
    esm_neighbor_shifts, esm_neighbor_masks, esm_neighbor_valid, esm_neighbor_codes,
    struct_neighbor_shifts, struct_neighbor_masks, struct_neighbor_valid, struct_neighbor_codes,
    shift_cols, stats,
):
    """Compare neighbor quality between ESM and structure retrieval.

    For each query residue, compute the MAE of the best same-AA neighbor's shifts
    vs the true shifts. Lower = better neighbors.
    """
    n_queries = len(query_shifts)
    n_shifts = len(shift_cols)

    esm_maes = []
    struct_maes = []
    both_valid = 0

    for i in range(n_queries):
        q_mask = query_masks[i]
        if not q_mask.any():
            continue
        q_aa = query_aa_codes[i]

        # ESM: find best same-AA neighbor
        esm_mae = _best_neighbor_mae(
            query_shifts[i], q_mask, q_aa,
            esm_neighbor_shifts[i], esm_neighbor_masks[i],
            esm_neighbor_valid[i], esm_neighbor_codes[i],
            shift_cols, stats,
        )

        # Struct: find best same-AA neighbor
        struct_mae = _best_neighbor_mae(
            query_shifts[i], q_mask, q_aa,
            struct_neighbor_shifts[i], struct_neighbor_masks[i],
            struct_neighbor_valid[i], struct_neighbor_codes[i],
            shift_cols, stats,
        )

        if esm_mae is not None and struct_mae is not None:
            esm_maes.append(esm_mae)
            struct_maes.append(struct_mae)
            both_valid += 1

    return np.array(esm_maes), np.array(struct_maes), both_valid


def _best_neighbor_mae(q_shifts, q_mask, q_aa,
                        nb_shifts, nb_masks, nb_valid, nb_codes,
                        shift_cols, stats):
    """MAE of closest same-AA neighbor's shifts vs query."""
    best_mae = None

    for k in range(len(nb_valid)):
        if not nb_valid[k]:
            continue
        if nb_codes[k] != q_aa:
            continue

        # Compute MAE for overlapping shifts
        overlap = q_mask & nb_masks[k]
        if overlap.sum() == 0:
            continue

        # Denormalize both to ppm
        q_ppm = np.zeros(len(shift_cols))
        nb_ppm = np.zeros(len(shift_cols))
        for si, col in enumerate(shift_cols):
            if overlap[si] and col in stats:
                q_ppm[si] = q_shifts[si] * stats[col]['std'] + stats[col]['mean']
                nb_ppm[si] = nb_shifts[k, si] * stats[col]['std'] + stats[col]['mean']

        mae = np.abs(q_ppm[overlap] - nb_ppm[overlap]).mean()
        if best_mae is None or mae < best_mae:
            best_mae = mae

    return best_mae


def main():
    parser = argparse.ArgumentParser(
        description='Compare ESM vs structure-based retrieval quality')
    parser.add_argument('--checkpoint', default='data/checkpoints/best_fold1.pt')
    parser.add_argument('--cache_dir', default='data/cache')
    parser.add_argument('--fold', type=int, default=1,
                        help='Test fold to compare on')
    parser.add_argument('--esm_index_dir', default='data/retrieval_indices')
    parser.add_argument('--esm_embeddings', default='data/esm_embeddings.h5')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--n_compare', type=int, default=50000,
                        help='Number of test residues to compare (default: 50000)')
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print("=" * 70)
    print("RETRIEVAL COMPARISON: ESM vs Structure Embeddings")
    print("=" * 70)
    t0 = time.time()

    # Load model
    print("\nLoading model...")
    model, info = load_model(args.checkpoint, device)
    stats = info['stats']
    shift_cols = info['shift_cols']
    embed_dim = None

    # Capture embedding dim
    captured = {}
    def hook_fn(module, inp, out):
        captured['base_encoding'] = inp[0].detach().cpu()
    handle = model.struct_head[0].register_forward_hook(hook_fn)

    # ========== Extract structure embeddings for training folds ==========
    print("\n--- Extracting structure embeddings (training folds) ---")
    train_embeddings = []
    train_bmrb_ids = []
    train_residue_ids = []
    train_shifts_all = []
    train_shift_masks_all = []
    train_residue_codes_all = []

    for fold_k in range(1, 6):
        if fold_k == args.fold:
            continue
        fold_cache = os.path.join(args.cache_dir, f'fold_{fold_k}')
        if not os.path.exists(os.path.join(fold_cache, 'config.json')):
            print(f"  SKIP fold_{fold_k}: no config.json")
            continue

        with open(os.path.join(fold_cache, 'config.json')) as f:
            fc = json.load(f)
        total_res = fc['total_residues']

        embs, bmrbs, rids = extract_embeddings_from_cache(
            model, fold_cache, device, batch_size=args.batch_size)

        # Also load shifts for metadata
        sd = os.path.join(fold_cache, 'structural')
        flat_shifts = np.load(os.path.join(sd, 'shifts.npy'))
        flat_shift_mask = np.load(os.path.join(sd, 'shift_mask.npy'))
        flat_residue_idx = np.load(os.path.join(sd, 'residue_idx.npy'))

        offset = len(train_embeddings)
        train_embeddings.append(embs)
        train_bmrb_ids.extend(bmrbs)
        train_residue_ids.extend(rids)
        train_shifts_all.append(flat_shifts)
        train_shift_masks_all.append(flat_shift_mask)
        train_residue_codes_all.append(flat_residue_idx)

        print(f"  Fold {fold_k}: {total_res} residues, embed shape {embs.shape}")

    train_embeddings = np.concatenate(train_embeddings, axis=0)
    train_shifts_all = np.concatenate(train_shifts_all, axis=0)
    train_shift_masks_all = np.concatenate(train_shift_masks_all, axis=0)
    train_residue_codes_all = np.concatenate(train_residue_codes_all, axis=0)

    embed_dim = train_embeddings.shape[1]
    print(f"\nTotal training pool: {len(train_embeddings)} residues, {embed_dim}-dim")

    # ========== Extract structure embeddings for test fold ==========
    print(f"\n--- Extracting structure embeddings (test fold {args.fold}) ---")
    test_cache = os.path.join(args.cache_dir, f'fold_{args.fold}')
    test_embs, test_bmrbs, test_rids = extract_embeddings_from_cache(
        model, test_cache, device, batch_size=args.batch_size)

    handle.remove()

    # Load test shifts
    test_sd = os.path.join(test_cache, 'structural')
    test_shifts = np.load(os.path.join(test_sd, 'shifts.npy'))
    test_shift_masks = np.load(os.path.join(test_sd, 'shift_mask.npy'))
    test_residue_codes = np.load(os.path.join(test_sd, 'residue_idx.npy'))

    print(f"Test fold: {len(test_embs)} residues")

    # ========== Build FAISS index from structure embeddings ==========
    print("\n--- Building FAISS index from structure embeddings ---")
    struct_index = build_index(train_embeddings)
    print(f"  Index: {struct_index.ntotal} vectors, dim={embed_dim}")

    # Build metadata for structure retrieval (for same-protein exclusion)
    train_bmrb_arr = np.array(train_bmrb_ids)

    # ========== Retrieve neighbors for test set ==========
    # Subsample for speed
    n_test = len(test_embs)
    n_compare = min(args.n_compare, n_test)
    if n_compare < n_test:
        compare_idx = np.random.choice(n_test, n_compare, replace=False)
        compare_idx.sort()
    else:
        compare_idx = np.arange(n_test)

    print(f"\n--- Retrieving neighbors for {n_compare} test residues ---")

    K = K_RETRIEVED

    # Structure retrieval
    print("  Structure retrieval...")
    query_embs = test_embs[compare_idx].copy().astype(np.float32)
    faiss.normalize_L2(query_embs)
    k_extra = K + 50
    s_dists, s_indices = struct_index.search(query_embs, k_extra)

    # Filter same-protein
    struct_nb_shifts = np.zeros((n_compare, K, len(shift_cols)), dtype=np.float32)
    struct_nb_masks = np.zeros((n_compare, K, len(shift_cols)), dtype=bool)
    struct_nb_valid = np.zeros((n_compare, K), dtype=bool)
    struct_nb_codes = np.zeros((n_compare, K), dtype=np.int32)
    struct_nb_dists = np.zeros((n_compare, K), dtype=np.float32)

    for q in range(n_compare):
        query_bmrb = test_bmrbs[compare_idx[q]]
        count = 0
        for i in range(k_extra):
            idx = s_indices[q, i]
            if idx < 0:
                continue
            if train_bmrb_arr[idx] == query_bmrb:
                continue
            struct_nb_shifts[q, count] = train_shifts_all[idx]
            struct_nb_masks[q, count] = train_shift_masks_all[idx]
            struct_nb_valid[q, count] = True
            struct_nb_codes[q, count] = train_residue_codes_all[idx]
            struct_nb_dists[q, count] = s_dists[q, i]
            count += 1
            if count >= K:
                break

    # ESM retrieval - load from existing cache
    print("  Loading ESM retrieval from cache...")
    test_rd = os.path.join(test_cache, 'retrieval')
    esm_nb_shifts_raw = np.load(os.path.join(test_rd, 'shifts.npy'), mmap_mode='r')
    esm_nb_masks_raw = np.load(os.path.join(test_rd, 'shift_masks.npy'), mmap_mode='r')
    esm_nb_valid_raw = np.load(os.path.join(test_rd, 'valid.npy'), mmap_mode='r')
    esm_nb_codes_raw = np.load(os.path.join(test_rd, 'residue_codes.npy'), mmap_mode='r')
    esm_nb_dists_raw = np.load(os.path.join(test_rd, 'distances.npy'), mmap_mode='r')

    # Normalize ESM retrieved shifts (they're stored raw in the cache)
    esm_nb_shifts = np.zeros((n_compare, K, len(shift_cols)), dtype=np.float32)
    esm_nb_masks = np.zeros((n_compare, K, len(shift_cols)), dtype=bool)
    esm_nb_valid = np.zeros((n_compare, K), dtype=bool)
    esm_nb_codes = np.zeros((n_compare, K), dtype=np.int32)
    esm_nb_dists = np.zeros((n_compare, K), dtype=np.float32)

    for q in range(n_compare):
        gidx = compare_idx[q]
        esm_nb_shifts[q] = esm_nb_shifts_raw[gidx].astype(np.float32)
        esm_nb_masks[q] = esm_nb_masks_raw[gidx]
        esm_nb_valid[q] = esm_nb_valid_raw[gidx]
        esm_nb_codes[q] = esm_nb_codes_raw[gidx].astype(np.int32)
        esm_nb_dists[q] = esm_nb_dists_raw[gidx].astype(np.float32)

    # Normalize retrieved shifts to match test shift normalization
    for si, col in enumerate(shift_cols):
        if col in stats:
            mean = stats[col]['mean']
            std = stats[col]['std']
            if std > 1e-6:
                valid_mask = esm_nb_masks[:, :, si]
                esm_nb_shifts[:, :, si] = np.where(
                    valid_mask,
                    (esm_nb_shifts[:, :, si] - mean) / std,
                    0.0
                )

    # ========== Compare neighbor quality ==========
    print("\n--- Comparing neighbor quality ---")

    query_shifts_sub = test_shifts[compare_idx]
    query_masks_sub = test_shift_masks[compare_idx]
    query_aa_sub = test_residue_codes[compare_idx]

    esm_maes, struct_maes, n_valid = compare_neighbors(
        query_shifts_sub, query_masks_sub, query_aa_sub,
        esm_nb_shifts, esm_nb_masks, esm_nb_valid, esm_nb_codes,
        struct_nb_shifts, struct_nb_masks, struct_nb_valid, struct_nb_codes,
        shift_cols, stats,
    )

    # ========== Report ==========
    print(f"\n{'='*70}")
    print(f"RESULTS ({n_valid} residues with valid neighbors from both methods)")
    print(f"{'='*70}")

    print(f"\n  Best same-AA neighbor shift MAE (ppm):")
    print(f"    ESM retrieval:       mean={np.mean(esm_maes):.4f}  median={np.median(esm_maes):.4f}")
    print(f"    Structure retrieval: mean={np.mean(struct_maes):.4f}  median={np.median(struct_maes):.4f}")

    diff = esm_maes - struct_maes
    struct_wins = (diff > 0).sum()
    esm_wins = (diff < 0).sum()
    ties = (diff == 0).sum()
    print(f"\n  Head-to-head (per residue):")
    print(f"    Structure better: {struct_wins} ({100*struct_wins/n_valid:.1f}%)")
    print(f"    ESM better:      {esm_wins} ({100*esm_wins/n_valid:.1f}%)")
    print(f"    Tied:            {ties} ({100*ties/n_valid:.1f}%)")
    print(f"    Mean improvement: {np.mean(diff):.4f} ppm (positive = struct better)")

    # Per-percentile comparison
    print(f"\n  Percentile comparison (MAE in ppm):")
    print(f"    {'Percentile':>12s}  {'ESM':>8s}  {'Struct':>8s}  {'Diff':>8s}")
    for p in [10, 25, 50, 75, 90, 95]:
        e = np.percentile(esm_maes, p)
        s = np.percentile(struct_maes, p)
        print(f"    {p:>10d}th  {e:>8.3f}  {s:>8.3f}  {e-s:>+8.3f}")

    # Per-shift comparison
    print(f"\n  Per-shift best-neighbor MAE:")
    backbone = {'ca_shift', 'cb_shift', 'c_shift', 'n_shift', 'h_shift', 'ha_shift'}
    for si, col in enumerate(shift_cols):
        if col not in backbone:
            continue
        esm_shift_maes = []
        struct_shift_maes = []

        for q in range(n_valid):
            q_mask = query_masks_sub[q]
            if not q_mask[si]:
                continue
            q_aa = query_aa_sub[q]

            # Best ESM neighbor for this shift
            best_e = _shift_best_nb(
                query_shifts_sub[q, si], q_aa, si,
                esm_nb_shifts[q], esm_nb_masks[q], esm_nb_valid[q], esm_nb_codes[q],
                col, stats)
            best_s = _shift_best_nb(
                query_shifts_sub[q, si], q_aa, si,
                struct_nb_shifts[q], struct_nb_masks[q], struct_nb_valid[q], struct_nb_codes[q],
                col, stats)

            if best_e is not None and best_s is not None:
                esm_shift_maes.append(best_e)
                struct_shift_maes.append(best_s)

        if esm_shift_maes:
            name = col.replace('_shift', '').upper()
            e_mean = np.mean(esm_shift_maes)
            s_mean = np.mean(struct_shift_maes)
            print(f"    {name:>4s}: ESM={e_mean:.3f}  Struct={s_mean:.3f}  Diff={e_mean-s_mean:+.3f}  (n={len(esm_shift_maes)})")

    # Similarity distribution
    esm_valid_sims = esm_nb_dists[esm_nb_valid]
    struct_valid_sims = struct_nb_dists[struct_nb_valid]
    print(f"\n  Cosine similarity distribution:")
    print(f"    ESM:    mean={np.mean(esm_valid_sims):.4f}  std={np.std(esm_valid_sims):.4f}")
    print(f"    Struct: mean={np.mean(struct_valid_sims):.4f}  std={np.std(struct_valid_sims):.4f}")

    # Count valid neighbors
    esm_n_valid_per = esm_nb_valid.sum(axis=1)
    struct_n_valid_per = struct_nb_valid.sum(axis=1)
    print(f"\n  Valid neighbors per query:")
    print(f"    ESM:    mean={np.mean(esm_n_valid_per):.1f}  (0 neighbors: {(esm_n_valid_per==0).sum()})")
    print(f"    Struct: mean={np.mean(struct_n_valid_per):.1f}  (0 neighbors: {(struct_n_valid_per==0).sum()})")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s")
    print("=" * 70)


def _shift_best_nb(q_val, q_aa, si, nb_shifts, nb_masks, nb_valid, nb_codes, col, stats):
    """Find MAE of best same-AA neighbor for a single shift."""
    best = None
    for k in range(len(nb_valid)):
        if not nb_valid[k]:
            continue
        if nb_codes[k] != q_aa:
            continue
        if not nb_masks[k, si]:
            continue
        if col in stats:
            q_ppm = q_val * stats[col]['std'] + stats[col]['mean']
            nb_ppm = nb_shifts[k, si] * stats[col]['std'] + stats[col]['mean']
        else:
            q_ppm = q_val
            nb_ppm = nb_shifts[k, si]
        mae = abs(q_ppm - nb_ppm)
        if best is None or mae < best:
            best = mae
    return best


if __name__ == '__main__':
    main()
