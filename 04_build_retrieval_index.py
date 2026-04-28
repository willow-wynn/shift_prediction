#!/usr/bin/env python
"""
04 - FAISS Retrieval Index Builder (Better Data Pipeline)

Builds fold-aware FAISS IVF indices for retrieval-augmented chemical shift
prediction.  For each fold k (1-5), builds an index from all proteins NOT in
fold k so that training on fold k never sees its own data in the retrieval pool.

Adapted from homologies/retrieval_index.py with:
- Constants imported from config.py
- Provenance logging: exact counts of proteins/residues per fold, exclusions,
  any discarded data
- Saves: FAISS index, metadata pickle, shift_cols.json per fold
- Uses inner product metric (cosine similarity after L2 normalization)

Output layout (in --output_dir):
    index_exclude_fold_{k}.faiss
    metadata_exclude_fold_{k}.pkl
    shift_cols.json
    config.json

Usage:
    python 04_build_retrieval_index.py \\
        --data_dir ./data \\
        --embeddings ./data/esm_embeddings.h5 \\
        --output_dir ./data/retrieval_indices
"""

import argparse
import json
import os
import pickle
import sys
import time

# Local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import faiss
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (
    FAISS_NPROBE,
    K_RETRIEVED,
    RESIDUE_TO_IDX,
    ESM_EMBED_DIM,
)


# ============================================================================
# Data Loading
# ============================================================================

def load_embeddings_and_shifts(
    h5_path: str,
    csv_path: str,
    shift_cols: list,
) -> tuple:
    """Load ESM embeddings and corresponding chemical shifts.

    Returns:
        data: dict mapping bmrb_id -> {embeddings, residue_ids, shifts,
              shift_mask, residue_codes, fold}
        provenance: dict with counts of skipped/missing data
    """
    provenance = {
        'proteins_in_h5': 0,
        'proteins_in_csv': 0,
        'proteins_matched': 0,
        'proteins_no_fold': 0,
        'proteins_no_shifts': 0,
        'proteins_not_in_h5': 0,
    }

    print("Loading embeddings from HDF5...")
    embeddings_by_protein = {}

    with h5py.File(h5_path, 'r') as h5f:
        emb_group = h5f['embeddings']
        provenance['proteins_in_h5'] = len(emb_group)
        for bmrb_id in tqdm(emb_group.keys(), desc="  Loading embeddings"):
            prot_group = emb_group[bmrb_id]
            embeddings_by_protein[bmrb_id] = {
                'residue_ids': prot_group['residue_ids'][:],
                'embeddings': prot_group['embeddings'][:].astype(np.float32),
            }

    print(f"  Loaded {len(embeddings_by_protein)} proteins from HDF5")

    # Load shifts from CSV — only the columns we actually need (saves GB)
    print("Loading shifts from CSV...")
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    keep = ['bmrb_id', 'residue_id', 'residue_code']
    for c in ('split', 'fold'):
        if c in header:
            keep.append(c)
    keep.extend([c for c in header if c in shift_cols])
    df = pd.read_csv(csv_path, usecols=keep, dtype={'bmrb_id': str}, low_memory=False)
    csv_proteins = df['bmrb_id'].nunique()
    provenance['proteins_in_csv'] = csv_proteins

    not_in_h5 = set()

    for bmrb_id, protein_df in tqdm(df.groupby('bmrb_id'), desc="  Matching shifts"):
        bmrb_str = str(bmrb_id)
        if bmrb_str not in embeddings_by_protein:
            not_in_h5.add(bmrb_str)
            continue

        protein_df = protein_df.sort_values('residue_id')
        stored_rids = embeddings_by_protein[bmrb_str]['residue_ids']

        # Fold assignment
        if 'split' in protein_df.columns:
            fold = protein_df['split'].iloc[0]
        elif 'fold' in protein_df.columns:
            fold = protein_df['fold'].iloc[0]
        else:
            provenance['proteins_no_fold'] += 1
            continue

        embeddings_by_protein[bmrb_str]['fold'] = int(fold)

        # Build shift data aligned to stored residue IDs
        rid_to_idx = {int(rid): i for i, rid in enumerate(stored_rids)}

        n_residues = len(stored_rids)
        n_shifts = len(shift_cols)

        shifts = np.zeros((n_residues, n_shifts), dtype=np.float32)
        shift_mask = np.zeros((n_residues, n_shifts), dtype=bool)
        residue_codes = np.zeros(n_residues, dtype=np.int32)

        for _, row in protein_df.iterrows():
            rid = int(row['residue_id'])
            if rid not in rid_to_idx:
                continue

            idx = rid_to_idx[rid]

            # Residue code
            code = str(row.get('residue_code', 'UNK')).upper()
            residue_codes[idx] = RESIDUE_TO_IDX.get(code, RESIDUE_TO_IDX['UNK'])

            # Shifts
            for si, col in enumerate(shift_cols):
                if col in row.index and pd.notna(row[col]):
                    shifts[idx, si] = float(row[col])
                    shift_mask[idx, si] = True

        embeddings_by_protein[bmrb_str]['shifts'] = shifts
        embeddings_by_protein[bmrb_str]['shift_mask'] = shift_mask
        embeddings_by_protein[bmrb_str]['residue_codes'] = residue_codes

    provenance['proteins_not_in_h5'] = len(not_in_h5)

    # Filter out incomplete proteins
    before = len(embeddings_by_protein)
    no_fold = []
    no_shifts = []
    for k, v in list(embeddings_by_protein.items()):
        if 'fold' not in v:
            no_fold.append(k)
            del embeddings_by_protein[k]
        elif 'shifts' not in v:
            no_shifts.append(k)
            del embeddings_by_protein[k]

    provenance['proteins_no_fold'] += len(no_fold)
    provenance['proteins_no_shifts'] = len(no_shifts)
    provenance['proteins_matched'] = len(embeddings_by_protein)

    print(f"  Matched {len(embeddings_by_protein)} proteins with embeddings + shifts + folds")

    return embeddings_by_protein, provenance


# ============================================================================
# Index Building
# ============================================================================

def build_fold_indices_streaming(
    h5_path: str,
    fold_assignments: dict,
    shift_data: dict,
    output_dir: str,
    folds: list,
    shift_cols: list,
    use_gpu: bool = True,
    mode: str = 'only',
):
    """Memory-efficient fold index builder: loads embeddings per-fold only.

    Never holds the full embeddings dataset in RAM (~55 GB for AF dataset).
    For each target_fold, opens the HDF5 file and loads only the proteins
    that belong to (or excluding) that fold, then builds the index.
    """
    import gc
    os.makedirs(output_dir, exist_ok=True)

    # Determine embedding dimension from h5
    with h5py.File(h5_path, 'r') as h5f:
        sample_bmrb = next(iter(h5f['embeddings'].keys()))
        embed_dim = h5f['embeddings'][sample_bmrb]['embeddings'].shape[1]
    print(f"Embedding dimension: {embed_dim}")
    print(f"Building indices (mode={mode}) for folds: {folds}")

    fold_provenance = {}
    prefix = 'only' if mode == 'only' else 'exclude'

    for target_fold in folds:
        index_path = os.path.join(output_dir, f'index_{prefix}_fold_{target_fold}.faiss')
        metadata_path = os.path.join(output_dir, f'metadata_{prefix}_fold_{target_fold}.pkl')
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            print(f"\n  Fold {target_fold}: index exists, skipping")
            continue

        print(f"\n{'=' * 50}")
        tag = 'ONLY' if mode == 'only' else 'excluding'
        print(f"Building index {tag} fold {target_fold}")
        print(f"{'=' * 50}")

        # Select bmrb_ids to include in this index
        if mode == 'only':
            target_bmrbs = [b for b, f in fold_assignments.items() if f == target_fold]
        else:
            target_bmrbs = [b for b, f in fold_assignments.items() if f != target_fold]
        print(f"  Target proteins: {len(target_bmrbs)}")

        all_embeddings = []
        all_metadata = []
        residues_by_fold = {}

        with h5py.File(h5_path, 'r') as h5f:
            emb_group = h5f['embeddings']
            for bmrb_str in tqdm(target_bmrbs, desc="  Loading embeddings"):
                if bmrb_str not in emb_group:
                    continue
                if bmrb_str not in shift_data:
                    continue
                prot_emb = emb_group[bmrb_str]['embeddings'][:].astype(np.float32)
                prot_rids = emb_group[bmrb_str]['residue_ids'][:]
                sd = shift_data[bmrb_str]
                shifts = sd['shifts']
                shift_mask = sd['shift_mask']
                residue_codes = sd['residue_codes']
                fold_val = fold_assignments[bmrb_str]
                residues_by_fold[fold_val] = residues_by_fold.get(fold_val, 0) + len(prot_rids)

                for i in range(len(prot_rids)):
                    all_embeddings.append(prot_emb[i])
                    all_metadata.append({
                        'bmrb_id': bmrb_str,
                        'residue_id': int(prot_rids[i]),
                        'residue_code': int(residue_codes[i]),
                        'shifts': shifts[i].tolist(),
                        'shift_mask': shift_mask[i].tolist(),
                    })

        if not all_embeddings:
            print(f"  Skip: no data matched for fold {target_fold}")
            continue

        all_embeddings = np.array(all_embeddings, dtype=np.float32)
        total_residues = len(all_embeddings)
        print(f"  Total residues in index: {total_residues:,}")
        for f_val in sorted(residues_by_fold):
            print(f"    Fold {f_val}: {residues_by_fold[f_val]:,} residues")

        faiss.normalize_L2(all_embeddings)
        n_list = max(100, min(4096, total_residues // 100))
        print(f"  Building IVF index with {n_list} clusters...")

        quantizer = faiss.IndexFlatIP(embed_dim)
        index = faiss.IndexIVFFlat(quantizer, embed_dim, n_list, faiss.METRIC_INNER_PRODUCT)

        if total_residues > 500_000:
            train_indices = np.random.choice(total_residues, 500_000, replace=False)
            train_subset = all_embeddings[train_indices]
        else:
            train_subset = all_embeddings

        if use_gpu and faiss.get_num_gpus() > 0:
            print("  Using GPU for training...")
            gpu_res = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(gpu_res, 0, index)
            gpu_index.train(train_subset)
            index = faiss.index_gpu_to_cpu(gpu_index)
        else:
            index.train(train_subset)
        index.add(all_embeddings)
        index.nprobe = FAISS_NPROBE

        faiss.write_index(index, index_path)
        print(f"  Saved index to {index_path}")
        with open(metadata_path, 'wb') as f:
            pickle.dump(all_metadata, f)
        print(f"  Saved metadata ({len(all_metadata):,} entries) to {metadata_path}")

        fold_provenance[target_fold] = {
            'mode': mode,
            'total_residues': total_residues,
            'residues_by_fold': residues_by_fold,
            'n_ivf_clusters': n_list,
            'nprobe': FAISS_NPROBE,
        }

        # Free memory before next fold
        del all_embeddings, all_metadata, index
        if 'gpu_res' in locals(): del gpu_res
        gc.collect()

    return fold_provenance


def build_fold_indices(
    data: dict,
    output_dir: str,
    folds: list,
    use_gpu: bool = True,
    mode: str = 'only',
):
    """Build FAISS indices for specified folds, with full provenance logging.

    Two modes:
      - 'only'    (default, correct): for each fold k, index contains ONLY fold k
        residues. Used with the per-fold cache design where fold_k/ retrieves
        from fold k's residues. train.py loads other-fold caches for training,
        so training samples never retrieve test-fold residues.
      - 'exclude' (legacy): for each fold k, index contains all residues EXCEPT
        fold k. Had subtle train/test leakage — kept for back-compat only.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Determine embedding dimension
    sample_protein = list(data.values())[0]
    embed_dim = sample_protein['embeddings'].shape[1]
    print(f"Embedding dimension: {embed_dim}")

    all_folds = sorted(set(v['fold'] for v in data.values()))
    print(f"All folds found in data: {all_folds}")
    print(f"Building indices (mode={mode}) for folds: {folds}")

    fold_provenance = {}
    prefix = 'only' if mode == 'only' else 'exclude'

    for target_fold in folds:
        # Skip if index already exists
        index_path = os.path.join(output_dir, f'index_{prefix}_fold_{target_fold}.faiss')
        metadata_path = os.path.join(output_dir, f'metadata_{prefix}_fold_{target_fold}.pkl')
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            print(f"\n  Fold {target_fold}: index already exists, skipping")
            continue

        print(f"\n{'=' * 50}")
        if mode == 'only':
            print(f"Building index containing ONLY fold {target_fold}")
        else:
            print(f"Building index excluding fold {target_fold}")
        print(f"{'=' * 50}")

        all_embeddings = []
        all_metadata = []
        proteins_included = 0
        proteins_excluded = 0
        residues_by_fold = {}

        for bmrb_id, prot_data in data.items():
            if mode == 'only':
                include = (prot_data['fold'] == target_fold)
            else:
                include = (prot_data['fold'] != target_fold)
            if not include:
                proteins_excluded += 1
                continue

            proteins_included += 1
            fold_val = prot_data['fold']
            n_res = len(prot_data['residue_ids'])
            residues_by_fold[fold_val] = residues_by_fold.get(fold_val, 0) + n_res

            for i in range(n_res):
                all_embeddings.append(prot_data['embeddings'][i])
                all_metadata.append({
                    'bmrb_id': bmrb_id,
                    'residue_id': int(prot_data['residue_ids'][i]),
                    'residue_code': int(prot_data['residue_codes'][i]),
                    'shifts': prot_data['shifts'][i].tolist(),
                    'shift_mask': prot_data['shift_mask'][i].tolist(),
                })

        all_embeddings = np.array(all_embeddings, dtype=np.float32)
        total_residues = len(all_embeddings)

        print(f"  Proteins included: {proteins_included}")
        print(f"  Proteins excluded: {proteins_excluded}")
        print(f"  Total residues in index: {total_residues:,}")
        for f_val in sorted(residues_by_fold):
            print(f"    Fold {f_val}: {residues_by_fold[f_val]:,} residues")

        # Normalize for cosine similarity
        faiss.normalize_L2(all_embeddings)

        # IVF cluster count
        n_list = min(4096, total_residues // 100)
        n_list = max(100, n_list)

        print(f"  Building IVF index with {n_list} clusters...")

        quantizer = faiss.IndexFlatIP(embed_dim)
        index = faiss.IndexIVFFlat(
            quantizer, embed_dim, n_list, faiss.METRIC_INNER_PRODUCT,
        )

        # Training subset
        if total_residues > 500_000:
            train_indices = np.random.choice(total_residues, 500_000, replace=False)
            train_subset = all_embeddings[train_indices]
            print(f"  Training on 500,000 / {total_residues:,} residues")
        else:
            train_subset = all_embeddings
            print(f"  Training on all {total_residues:,} residues")

        # GPU-accelerated training if available
        if use_gpu and faiss.get_num_gpus() > 0:
            print("  Using GPU for index training...")
            gpu_res = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(gpu_res, 0, index)
            gpu_index.train(train_subset)
            index = faiss.index_gpu_to_cpu(gpu_index)
        else:
            index.train(train_subset)

        index.add(all_embeddings)
        index.nprobe = FAISS_NPROBE

        # Save index + metadata using the mode-specific prefix
        faiss.write_index(index, index_path)
        print(f"  Saved index to {index_path}")
        with open(metadata_path, 'wb') as f:
            pickle.dump(all_metadata, f)
        print(f"  Saved metadata ({len(all_metadata):,} entries) to {metadata_path}")

        fold_provenance[target_fold] = {
            'mode': mode,
            'proteins_included': proteins_included,
            'proteins_excluded': proteins_excluded,
            'total_residues': total_residues,
            'residues_by_fold': residues_by_fold,
            'n_ivf_clusters': n_list,
            'nprobe': FAISS_NPROBE,
        }

    return fold_provenance


def get_shift_columns(df_columns: list) -> list:
    """Get chemical shift column names."""
    return sorted([c for c in df_columns if c.endswith('_shift')])


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Build fold-aware FAISS retrieval indices.',
    )
    parser.add_argument(
        '--data_dir', default='./data',
        help='Directory containing the compiled CSV (default: ./data)',
    )
    parser.add_argument(
        '--embeddings', default=None,
        help='Path to ESM embeddings HDF5 (default: <data_dir>/esm_embeddings.h5)',
    )
    parser.add_argument(
        '--output_dir', default=None,
        help='Output directory for indices (default: /home/brooks/1TB/Wynn/'
             '<data_dir_basename>/retrieval_indices — 1TB drive, since FAISS '
             'indices are ~12 GB each × 5 folds)',
    )
    parser.add_argument(
        '--folds', type=int, nargs='+', default=[1, 2, 3, 4, 5],
        help='Which folds to build indices for (default: 1 2 3 4 5)',
    )
    parser.add_argument(
        '--no_gpu', action='store_true',
        help='Disable GPU for index building',
    )
    parser.add_argument(
        '--mode', choices=['only', 'exclude'], default='only',
        help='Index semantics. "only" (default, correct): each fold_k index '
             'contains only fold k residues, matching the per-fold cache '
             'design where fold_k/ retrieves from fold k. "exclude" (legacy, '
             'leaky): each fold_k index contains everything except fold k.',
    )
    args = parser.parse_args()

    if args.embeddings is None:
        args.embeddings = os.path.join(args.data_dir, 'esm_embeddings.h5')
    if args.output_dir is None:
        # Default to 1TB drive to avoid filling main disk (~60 GB per dataset)
        ds_name = os.path.basename(os.path.abspath(args.data_dir))
        args.output_dir = f'/home/brooks/1TB/Wynn/{ds_name}_retrieval_indices'

    print("=" * 60)
    print("FAISS Retrieval Index Builder (Better Data Pipeline)")
    print("=" * 60)
    print(f"  Data directory:  {args.data_dir}")
    print(f"  Embeddings:      {args.embeddings}")
    print(f"  Output dir:      {args.output_dir}")
    print(f"  Folds:           {args.folds}")
    print(f"  K_RETRIEVED:     {K_RETRIEVED}")
    print(f"  FAISS_NPROBE:    {FAISS_NPROBE}")
    print(f"  GPU:             {'disabled' if args.no_gpu else 'enabled'}")
    print()

    # Locate compiled CSV — try full names first, then per-fold files if present
    csv_path = None
    for name in ['structure_data_hybrid.csv', 'structure_data.csv',
                 'compiled_dataset.csv', 'sidechain_structure_data.csv',
                 'small_structure_data.csv']:
        candidate = os.path.join(args.data_dir, name)
        if os.path.exists(candidate):
            csv_path = candidate
            break
    # Fallback: if only per-fold CSVs exist, virtually concatenate them.
    per_fold_csvs = [os.path.join(args.data_dir, f'structure_data_hybrid_fold_{f}.csv')
                     for f in range(1, 6)]
    if csv_path is None and all(os.path.exists(p) for p in per_fold_csvs):
        print("  No single CSV found; concatenating per-fold files in memory")
        parts = []
        for p in per_fold_csvs:
            # Use only the columns we actually need for index building
            hdr = pd.read_csv(p, nrows=0).columns
            keep = [c for c in hdr if c == 'bmrb_id' or c == 'residue_id'
                    or c == 'residue_code' or c == 'split'
                    or c.endswith('_shift') or c.endswith('_ambiguity_code')]
            parts.append(pd.read_csv(p, usecols=keep, dtype={'bmrb_id': str}, low_memory=False))
        df_full = pd.concat(parts, ignore_index=True)
        del parts
        # Write a temp combined csv for downstream consumers (or pass directly)
        tmp_csv = os.path.join(args.output_dir, '_tmp_combined.csv')
        os.makedirs(args.output_dir, exist_ok=True)
        df_full.to_csv(tmp_csv, index=False)
        del df_full
        csv_path = tmp_csv
    if csv_path is None:
        candidates = [f for f in os.listdir(args.data_dir) if f.endswith('.csv')]
        raise FileNotFoundError(
            f"Cannot find dataset CSV in {args.data_dir}. "
            f"Available CSVs: {candidates}"
        )

    # Get shift columns
    df_sample = pd.read_csv(csv_path, nrows=1, dtype={'bmrb_id': str})
    shift_cols = get_shift_columns(df_sample.columns.tolist())
    print(f"Shift columns ({len(shift_cols)}): {shift_cols}")

    # Memory-efficient path: do NOT load all embeddings upfront. Instead
    # read fold assignments + shifts from CSV (small), then stream embeddings
    # per-fold from HDF5 inside build_fold_indices_streaming().
    print("Loading fold + shift data from CSV (memory-efficient)...")
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    keep = ['bmrb_id', 'residue_id', 'residue_code']
    fold_col = None
    for c in ('split', 'fold'):
        if c in header:
            keep.append(c); fold_col = c
    keep.extend([c for c in header if c in shift_cols])
    df = pd.read_csv(csv_path, usecols=keep, dtype={'bmrb_id': str}, low_memory=False)
    print(f"  CSV: {len(df):,} rows, {df['bmrb_id'].nunique()} proteins")

    # Fold assignments (one per bmrb_id)
    fold_assignments = df.groupby('bmrb_id')[fold_col].first().astype(int).to_dict()

    # For each bmrb_id, open its h5 group once to get stored residue_ids,
    # then fill shifts/mask/residue_codes arrays aligned to those rids.
    print("Building per-protein shift arrays (aligned to h5 residue_ids)...")
    shift_data = {}
    with h5py.File(args.embeddings, 'r') as h5f:
        emb_group = h5f['embeddings']
        for bmrb_id, protein_df in tqdm(df.groupby('bmrb_id'), desc='  proteins'):
            bmrb_str = str(bmrb_id)
            if bmrb_str not in emb_group:
                continue
            stored_rids = emb_group[bmrb_str]['residue_ids'][:]
            rid_to_idx = {int(rid): i for i, rid in enumerate(stored_rids)}
            n_res = len(stored_rids)
            shifts = np.zeros((n_res, len(shift_cols)), dtype=np.float32)
            shift_mask = np.zeros((n_res, len(shift_cols)), dtype=bool)
            residue_codes = np.zeros(n_res, dtype=np.int32)
            protein_df = protein_df.sort_values('residue_id')
            for _, row in protein_df.iterrows():
                rid = int(row['residue_id'])
                if rid not in rid_to_idx: continue
                idx = rid_to_idx[rid]
                code = str(row.get('residue_code', 'UNK')).upper()
                residue_codes[idx] = RESIDUE_TO_IDX.get(code, RESIDUE_TO_IDX['UNK'])
                for si, col in enumerate(shift_cols):
                    v = row.get(col)
                    if v is not None and pd.notna(v):
                        shifts[idx, si] = float(v)
                        shift_mask[idx, si] = True
            shift_data[bmrb_str] = {'shifts': shifts, 'shift_mask': shift_mask,
                                    'residue_codes': residue_codes}
    del df
    print(f"  Shift data built for {len(shift_data)} proteins")

    # Save shift columns
    os.makedirs(args.output_dir, exist_ok=True)
    shift_cols_path = os.path.join(args.output_dir, 'shift_cols.json')
    with open(shift_cols_path, 'w') as f:
        json.dump(shift_cols, f)
    print(f"\nSaved shift_cols.json ({len(shift_cols)} columns)")

    # Build indices (streaming — loads embeddings per fold only)
    t0 = time.time()
    fold_provenance = build_fold_indices_streaming(
        h5_path=args.embeddings,
        fold_assignments=fold_assignments,
        shift_data=shift_data,
        output_dir=args.output_dir,
        folds=args.folds,
        shift_cols=shift_cols,
        use_gpu=not args.no_gpu,
        mode=args.mode,
    )
    elapsed = time.time() - t0

    # Save config + full provenance
    config = {
        'embed_dim': ESM_EMBED_DIM,
        'folds': args.folds,
        'shift_cols': shift_cols,
        'faiss_nprobe': FAISS_NPROBE,
        'k_retrieved': K_RETRIEVED,
    }
    config_path = os.path.join(args.output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    # ---- Final Provenance Report ----
    print("\n" + "=" * 60)
    print("PROVENANCE REPORT")
    print("=" * 60)
    print(f"  Proteins with fold assignments: {len(fold_assignments)}")
    print(f"  Proteins with shift data:       {len(shift_data)}")
    print()
    print(f"  Index building ({elapsed:.1f}s):")
    for fold_k, fp in sorted(fold_provenance.items()):
        print(f"    Fold {fold_k} ({fp.get('mode', 'only')}):")
        print(f"      Residues in index:  {fp['total_residues']:,}")
        print(f"      IVF clusters:       {fp['n_ivf_clusters']}")
    print("=" * 60)

    print("\nDone!")


if __name__ == "__main__":
    main()
