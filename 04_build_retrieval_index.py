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

    # Load shifts from CSV
    print("Loading shifts from CSV...")
    df = pd.read_csv(csv_path, dtype={'bmrb_id': str})
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

def build_fold_indices(
    data: dict,
    output_dir: str,
    folds: list,
    use_gpu: bool = True,
):
    """Build FAISS indices for specified folds, with full provenance logging."""
    os.makedirs(output_dir, exist_ok=True)

    # Determine embedding dimension
    sample_protein = list(data.values())[0]
    embed_dim = sample_protein['embeddings'].shape[1]
    print(f"Embedding dimension: {embed_dim}")

    all_folds = sorted(set(v['fold'] for v in data.values()))
    print(f"All folds found in data: {all_folds}")
    print(f"Building indices for excluded folds: {folds}")

    fold_provenance = {}

    for exclude_fold in folds:
        print(f"\n{'=' * 50}")
        print(f"Building index excluding fold {exclude_fold}")
        print(f"{'=' * 50}")

        # Collect embeddings from all other folds
        all_embeddings = []
        all_metadata = []

        proteins_included = 0
        proteins_excluded = 0
        residues_by_fold = {}

        for bmrb_id, prot_data in data.items():
            if prot_data['fold'] == exclude_fold:
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
        print(f"  Proteins excluded (fold {exclude_fold}): {proteins_excluded}")
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

        # Save index
        index_path = os.path.join(output_dir, f'index_exclude_fold_{exclude_fold}.faiss')
        faiss.write_index(index, index_path)
        print(f"  Saved index to {index_path}")

        # Save metadata
        metadata_path = os.path.join(
            output_dir, f'metadata_exclude_fold_{exclude_fold}.pkl',
        )
        with open(metadata_path, 'wb') as f:
            pickle.dump(all_metadata, f)
        print(f"  Saved metadata ({len(all_metadata):,} entries) to {metadata_path}")

        fold_provenance[exclude_fold] = {
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
        help='Output directory for indices (default: <data_dir>/retrieval_indices)',
    )
    parser.add_argument(
        '--folds', type=int, nargs='+', default=[1, 2, 3, 4, 5],
        help='Which folds to build indices for (default: 1 2 3 4 5)',
    )
    parser.add_argument(
        '--no_gpu', action='store_true',
        help='Disable GPU for index building',
    )
    args = parser.parse_args()

    if args.embeddings is None:
        args.embeddings = os.path.join(args.data_dir, 'esm_embeddings.h5')
    if args.output_dir is None:
        args.output_dir = os.path.join(args.data_dir, 'retrieval_indices')

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

    # Locate compiled CSV
    # Try common dataset names in order of preference
    csv_path = None
    for name in ['structure_data.csv', 'compiled_dataset.csv', 'sidechain_structure_data.csv', 'small_structure_data.csv']:
        candidate = os.path.join(args.data_dir, name)
        if os.path.exists(candidate):
            csv_path = candidate
            break
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

    # Load data
    data, load_provenance = load_embeddings_and_shifts(
        h5_path=args.embeddings,
        csv_path=csv_path,
        shift_cols=shift_cols,
    )

    # Save shift columns
    os.makedirs(args.output_dir, exist_ok=True)
    shift_cols_path = os.path.join(args.output_dir, 'shift_cols.json')
    with open(shift_cols_path, 'w') as f:
        json.dump(shift_cols, f)
    print(f"\nSaved shift_cols.json ({len(shift_cols)} columns)")

    # Build indices
    t0 = time.time()
    fold_provenance = build_fold_indices(
        data=data,
        output_dir=args.output_dir,
        folds=args.folds,
        use_gpu=not args.no_gpu,
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
    print(f"  Data loading:")
    print(f"    Proteins in HDF5:         {load_provenance['proteins_in_h5']}")
    print(f"    Proteins in CSV:          {load_provenance['proteins_in_csv']}")
    print(f"    Proteins matched:         {load_provenance['proteins_matched']}")
    print(f"    Removed (no fold):        {load_provenance['proteins_no_fold']}")
    print(f"    Removed (no shifts):      {load_provenance['proteins_no_shifts']}")
    print(f"    Removed (not in HDF5):    {load_provenance['proteins_not_in_h5']}")
    print()
    print(f"  Index building ({elapsed:.1f}s):")
    for fold_k, fp in sorted(fold_provenance.items()):
        print(f"    Fold {fold_k} excluded:")
        print(f"      Proteins in index:  {fp['proteins_included']}")
        print(f"      Proteins excluded:  {fp['proteins_excluded']}")
        print(f"      Residues in index:  {fp['total_residues']:,}")
        print(f"      IVF clusters:       {fp['n_ivf_clusters']}")
    print("=" * 60)

    print("\nDone!")


if __name__ == "__main__":
    main()
