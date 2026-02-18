#!/usr/bin/env python
"""
03 - ESM-2 Embedding Extraction (Better Data Pipeline)

Extracts per-residue ESM-2 embeddings (esm2_t36_3B_UR50D, layer 36, 2560-dim)
for all proteins in the compiled dataset.

Adapted from homologies/esm_extraction.py with:
- Constants imported from config.py
- Provenance logging: every filtering/skip/failure is counted and reported
- Compressed fp16 HDF5 output: one dataset per protein keyed by bmrb_id
- Sequences reconstructed by filling alignment gaps with 'X'
- BOS/EOS tokens removed from ESM output

Output HDF5 layout:
    embeddings/<bmrb_id>/residue_ids   -> (n_residues,)  int32
    embeddings/<bmrb_id>/embeddings    -> (n_residues, 2560)  float16, gzip
    metadata/ attrs: embed_dim, model, repr_layer, n_proteins

Usage:
    python 03_extract_esm_embeddings.py --data_dir ./data --output ./data/esm_embeddings.h5
"""

import argparse
import os
import sys
import time

# Local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config import (
    ESM_MODEL_NAME,
    ESM_EMBED_DIM,
    ESM_REPR_LAYER,
    AA_3_TO_1,
    NONSTANDARD_MAP,
)

# ESM imports
try:
    import esm
except ImportError:
    raise ImportError("Please install fair-esm: pip install fair-esm")


# ============================================================================
# Amino Acid Helpers
# ============================================================================

def three_to_one(residue_code: str) -> str:
    """Convert three-letter amino acid code to single letter.

    Handles standard residues, known non-standard residues, and unknowns.
    """
    code = str(residue_code).upper().strip()
    # Direct standard lookup
    if code in AA_3_TO_1:
        return AA_3_TO_1[code]
    # Non-standard mapping (e.g. MSE -> MET -> M)
    mapped = NONSTANDARD_MAP.get(code)
    if mapped and mapped in AA_3_TO_1:
        return AA_3_TO_1[mapped]
    return 'X'


# ============================================================================
# Sequence Reconstruction
# ============================================================================

def reconstruct_sequence(protein_df: pd.DataFrame) -> tuple:
    """Reconstruct a protein sequence from residue data, filling gaps with X.

    Args:
        protein_df: DataFrame for a single protein with 'residue_id' and 'residue_code'.

    Returns:
        sequence: Single-letter amino acid string.
        residue_to_seqpos: dict mapping residue_id -> 0-based position in *sequence*.
    """
    protein_df = protein_df.sort_values('residue_id')

    residue_ids = protein_df['residue_id'].values
    residue_codes = protein_df['residue_code'].values

    min_res = int(residue_ids.min())
    max_res = int(residue_ids.max())

    existing = {int(rid): code for rid, code in zip(residue_ids, residue_codes)}

    sequence_chars = []
    residue_to_seqpos = {}

    for rid in range(min_res, max_res + 1):
        if rid in existing:
            aa = three_to_one(existing[rid])
            residue_to_seqpos[rid] = len(sequence_chars)
            sequence_chars.append(aa)
        else:
            sequence_chars.append('X')

    return ''.join(sequence_chars), residue_to_seqpos


# ============================================================================
# Batch Extraction
# ============================================================================

def extract_embeddings_batch(
    model,
    batch_converter,
    sequences: list,
    device: torch.device,
    repr_layer: int = ESM_REPR_LAYER,
) -> dict:
    """Extract ESM embeddings for a batch of sequences.

    Args:
        model: ESM model.
        batch_converter: ESM batch converter.
        sequences: List of (name, sequence) tuples.
        device: Torch device.
        repr_layer: Layer to extract representations from.

    Returns:
        dict mapping sequence name -> numpy array (seq_len, embed_dim).
    """
    batch_labels, batch_strs, batch_tokens = batch_converter(sequences)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[repr_layer], return_contacts=False)

    token_reps = results["representations"][repr_layer]

    embeddings = {}
    for i, (name, seq) in enumerate(sequences):
        seq_len = len(seq)
        # Remove BOS token at position 0; take positions 1 through seq_len (inclusive)
        emb = token_reps[i, 1:seq_len + 1, :].cpu().numpy()
        embeddings[name] = emb

    return embeddings


# ============================================================================
# Main Processing
# ============================================================================

def process_proteins(
    df: pd.DataFrame,
    model,
    batch_converter,
    device: torch.device,
    output_path: str,
    batch_size: int = 4,
    repr_layer: int = ESM_REPR_LAYER,
):
    """Process all proteins and save embeddings to HDF5.

    Logs provenance for every skip/failure.
    """
    proteins = list(df.groupby('bmrb_id'))
    n_total = len(proteins)
    print(f"Processing {n_total} proteins...")

    # Provenance counters
    skipped_empty = 0
    failed = []
    processed = 0
    total_residues_stored = 0

    with h5py.File(output_path, 'w') as h5f:
        emb_group = h5f.create_group('embeddings')
        meta_group = h5f.create_group('metadata')

        meta_group.attrs['embed_dim'] = ESM_EMBED_DIM
        meta_group.attrs['model'] = ESM_MODEL_NAME
        meta_group.attrs['repr_layer'] = repr_layer

        batch = []
        batch_metadata = []  # (bmrb_id, residue_to_seqpos, protein_df)

        for bmrb_id, protein_df in tqdm(proteins, desc="Extracting embeddings"):
            sequence, residue_to_seqpos = reconstruct_sequence(protein_df)

            if len(sequence) == 0:
                skipped_empty += 1
                continue

            batch.append((str(bmrb_id), sequence))
            batch_metadata.append((bmrb_id, residue_to_seqpos, protein_df))

            if len(batch) >= batch_size:
                try:
                    _process_batch(
                        batch, batch_metadata, model, batch_converter,
                        device, emb_group, repr_layer,
                    )
                    processed += len(batch)
                    total_residues_stored += sum(
                        len(md[1]) for md in batch_metadata
                    )
                except Exception as e:
                    for name, _ in batch:
                        failed.append((name, str(e)))
                batch = []
                batch_metadata = []

        # Process remaining
        if batch:
            try:
                _process_batch(
                    batch, batch_metadata, model, batch_converter,
                    device, emb_group, repr_layer,
                )
                processed += len(batch)
                total_residues_stored += sum(
                    len(md[1]) for md in batch_metadata
                )
            except Exception as e:
                for name, _ in batch:
                    failed.append((name, str(e)))

        meta_group.attrs['n_proteins'] = processed

    # ---- Provenance Report ----
    print("\n" + "=" * 60)
    print("PROVENANCE REPORT")
    print("=" * 60)
    print(f"  Total proteins in dataset:  {n_total}")
    print(f"  Successfully processed:     {processed}")
    print(f"  Total residues stored:      {total_residues_stored:,}")
    print(f"  Skipped (empty sequence):   {skipped_empty}")
    print(f"  Failed:                     {len(failed)}")
    if failed:
        for fid, reason in failed[:10]:
            print(f"    {fid}: {reason}")
        if len(failed) > 10:
            print(f"    ... and {len(failed) - 10} more")
    print("=" * 60)


def _process_batch(
    batch: list,
    batch_metadata: list,
    model,
    batch_converter,
    device: torch.device,
    emb_group: h5py.Group,
    repr_layer: int,
):
    """Process a single batch and save to HDF5."""
    embeddings = extract_embeddings_batch(
        model, batch_converter, batch, device, repr_layer,
    )

    for (bmrb_id_str, sequence), (_, residue_to_seqpos, protein_df) in zip(batch, batch_metadata):
        emb = embeddings[bmrb_id_str]  # (seq_len, embed_dim)

        valid_residue_ids = protein_df['residue_id'].values

        residue_ids_stored = []
        embeddings_stored = []

        for rid in valid_residue_ids:
            rid = int(rid)
            if rid in residue_to_seqpos:
                seq_pos = residue_to_seqpos[rid]
                if seq_pos < len(emb):
                    residue_ids_stored.append(rid)
                    embeddings_stored.append(emb[seq_pos])

        if embeddings_stored:
            prot_group = emb_group.create_group(bmrb_id_str)
            prot_group.create_dataset(
                'residue_ids',
                data=np.array(residue_ids_stored, dtype=np.int32),
            )
            prot_group.create_dataset(
                'embeddings',
                data=np.array(embeddings_stored, dtype=np.float16),
                compression='gzip',
                compression_opts=4,
            )
            prot_group.attrs['sequence'] = sequence
            prot_group.attrs['n_residues'] = len(residue_ids_stored)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Extract ESM-2 embeddings for all proteins in the compiled dataset.',
    )
    parser.add_argument(
        '--data_dir', default='./data',
        help='Directory containing the compiled CSV (default: ./data)',
    )
    parser.add_argument(
        '--output', default=None,
        help='Output HDF5 path (default: <data_dir>/esm_embeddings.h5)',
    )
    parser.add_argument(
        '--batch_size', type=int, default=4,
        help='Number of sequences per ESM batch (default: 4)',
    )
    parser.add_argument(
        '--max_seq_len', type=int, default=None,
        help='Maximum sequence length; longer sequences are truncated (default: no limit)',
    )
    parser.add_argument(
        '--device', default=None,
        help='Torch device (default: cuda if available, else cpu)',
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(args.data_dir, 'esm_embeddings.h5')

    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 60)
    print("ESM-2 Embedding Extraction (Better Data Pipeline)")
    print("=" * 60)
    print(f"  Model:           {ESM_MODEL_NAME}")
    print(f"  Repr layer:      {ESM_REPR_LAYER}")
    print(f"  Embed dim:       {ESM_EMBED_DIM}")
    print(f"  Device:          {args.device}")
    print(f"  Batch size:      {args.batch_size}")
    print(f"  Data directory:  {args.data_dir}")
    print(f"  Output file:     {args.output}")
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

    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path, dtype={'bmrb_id': str})
    n_residues = len(df)
    n_proteins = df['bmrb_id'].nunique()
    print(f"  Loaded {n_residues:,} residues from {n_proteins:,} proteins")

    # Load ESM model
    print(f"\nLoading ESM-2 model ({ESM_MODEL_NAME})...")
    print("  This may take a few minutes and requires ~12GB GPU memory")
    t0 = time.time()
    model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model = model.eval()

    device = torch.device(args.device)
    model = model.to(device)
    print(f"  Model loaded on {device} in {time.time() - t0:.1f}s")

    # Extract embeddings
    print("\nExtracting embeddings...")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    process_proteins(
        df=df,
        model=model,
        batch_converter=batch_converter,
        device=device,
        output_path=args.output,
        batch_size=args.batch_size,
        repr_layer=ESM_REPR_LAYER,
    )

    # Verify
    print(f"\nVerifying output file {args.output}...")
    with h5py.File(args.output, 'r') as h5f:
        n_stored = len(h5f['embeddings'])
        embed_dim = h5f['metadata'].attrs['embed_dim']
        print(f"  Stored embeddings for {n_stored} proteins")
        print(f"  Embedding dimension: {embed_dim}")

        if n_stored > 0:
            sample_id = list(h5f['embeddings'].keys())[0]
            sample_emb = h5f['embeddings'][sample_id]['embeddings'][:]
            print(f"  Sample protein {sample_id}: shape {sample_emb.shape}, dtype {sample_emb.dtype}")

    print("\nDone!")


if __name__ == "__main__":
    main()
