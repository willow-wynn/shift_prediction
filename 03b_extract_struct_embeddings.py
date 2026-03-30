#!/usr/bin/env python3
"""
Extract structure model embeddings (base_encoding) for all residues.

Produces an HDF5 file in the same format as ESM embeddings, allowing
the existing retrieval pipeline (04, 05) to work unchanged.

Usage:
    python 03b_extract_struct_embeddings.py \
        --checkpoint data/checkpoints/best_fold1.pt \
        --cache_dir data/cache \
        --output data/struct_retrieval/struct_embeddings.h5
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import K_RETRIEVED
from dataset import CachedRetrievalDataset
from inference import load_model


def extract_fold(model, cache_dir, device, batch_size=512):
    """Extract base_encoding for every residue in a fold cache.

    Returns dict: bmrb_id -> [(residue_id, embedding), ...]
    """
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

    # Override samples to cover ALL residues
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

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4,
                        pin_memory=(device == 'cuda'), prefetch_factor=2,
                        persistent_workers=True)

    # Accumulate by protein
    protein_data = defaultdict(list)  # bmrb_id -> [(rid, embedding), ...]
    sample_idx = 0

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"  {os.path.basename(cache_dir)}"):
            batch_dev = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            _ = batch_dev.pop('shift_target', None)
            _ = batch_dev.pop('shift_mask', None)

            model(**batch_dev)
            be = captured['base_encoding'].numpy()  # (B, embed_dim)

            B = be.shape[0]
            for i in range(B):
                gidx = all_samples[sample_idx + i][0]
                bmrb_id = ds.idx_to_bmrb.get(str(gidx), None)
                rid = ds.global_to_resid.get(str(gidx), None)
                if bmrb_id is not None and rid is not None:
                    protein_data[str(bmrb_id)].append((int(rid), be[i]))

            sample_idx += B

    handle.remove()
    return protein_data, be.shape[1]


def main():
    parser = argparse.ArgumentParser(
        description='Extract structure model embeddings for retrieval')
    parser.add_argument('--checkpoint', default='data/checkpoints/best_fold1.pt')
    parser.add_argument('--cache_dir', default='data/cache')
    parser.add_argument('--output', default='data/struct_retrieval/struct_embeddings.h5')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--device', default=None)
    parser.add_argument('--folds', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    args = parser.parse_args()

    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print("=" * 70)
    print("EXTRACT STRUCTURE MODEL EMBEDDINGS")
    print("=" * 70)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Cache dir:  {args.cache_dir}")
    print(f"  Output:     {args.output}")
    print(f"  Device:     {device}")
    t0 = time.time()

    # Load model
    print("\nLoading model...")
    model, info = load_model(args.checkpoint, device)

    # Extract from all folds
    all_protein_data = {}
    embed_dim = None

    for fold_k in args.folds:
        fold_cache = os.path.join(args.cache_dir, f'fold_{fold_k}')
        if not os.path.exists(os.path.join(fold_cache, 'config.json')):
            print(f"  SKIP fold_{fold_k}: not found")
            continue

        print(f"\nExtracting fold {fold_k}...")
        pdata, edim = extract_fold(model, fold_cache, device, args.batch_size)
        embed_dim = edim

        # Merge (no duplicates across folds)
        for bmrb_id, entries in pdata.items():
            if bmrb_id in all_protein_data:
                all_protein_data[bmrb_id].extend(entries)
            else:
                all_protein_data[bmrb_id] = entries

        n_res = sum(len(v) for v in pdata.values())
        print(f"  Fold {fold_k}: {len(pdata)} proteins, {n_res} residues")

    total_proteins = len(all_protein_data)
    total_residues = sum(len(v) for v in all_protein_data.values())
    print(f"\nTotal: {total_proteins} proteins, {total_residues} residues, {embed_dim}-dim")

    # Write HDF5
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print(f"\nWriting HDF5: {args.output}")

    with h5py.File(args.output, 'w') as f:
        emb_group = f.create_group('embeddings')

        for bmrb_id in tqdm(sorted(all_protein_data.keys()), desc="  Writing"):
            entries = all_protein_data[bmrb_id]
            # Sort by residue_id
            entries.sort(key=lambda x: x[0])
            rids = np.array([e[0] for e in entries], dtype=np.int32)
            embs = np.array([e[1] for e in entries], dtype=np.float16)

            grp = emb_group.create_group(bmrb_id)
            grp.create_dataset('residue_ids', data=rids)
            grp.create_dataset('embeddings', data=embs, compression='gzip',
                               compression_opts=4)

        # Metadata
        meta = f.create_group('metadata')
        meta.attrs['embed_dim'] = embed_dim
        meta.attrs['model'] = 'struct_base_encoding'
        meta.attrs['repr_layer'] = 0
        meta.attrs['n_proteins'] = total_proteins
        meta.attrs['source_checkpoint'] = args.checkpoint

    elapsed = time.time() - t0
    file_size = os.path.getsize(args.output) / (1024**3)
    print(f"\nDone in {elapsed:.0f}s")
    print(f"  Output: {args.output} ({file_size:.2f} GB)")
    print(f"  Proteins: {total_proteins}")
    print(f"  Residues: {total_residues}")
    print(f"  Embed dim: {embed_dim}")


if __name__ == '__main__':
    main()
