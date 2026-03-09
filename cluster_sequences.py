#!/usr/bin/env python3
"""
Cluster protein sequences at 90% identity using MMseqs2.

Produces an exclusion map: for each protein, the set of proteins with >90%
sequence identity. Used by the retrieval system to prevent data leakage.

Usage:
    python cluster_sequences.py --data_dir ./data

Output:
    data/identity_clusters_90.json — maps bmrb_id -> [list of bmrb_ids with >90% identity]
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from config import AA_3_TO_1, DATA_DIR


def extract_sequences(csv_path):
    """Extract one sequence per protein from the compiled dataset CSV.

    Returns:
        dict: bmrb_id -> single-letter amino acid sequence
    """
    df = pd.read_csv(csv_path, usecols=['bmrb_id', 'residue_id', 'residue_code'],
                      dtype={'bmrb_id': str})

    sequences = {}
    for bmrb_id, group in df.groupby('bmrb_id'):
        group = group.sort_values('residue_id').drop_duplicates(subset='residue_id')
        seq_chars = []
        for code in group['residue_code'].values:
            aa = AA_3_TO_1.get(str(code).upper())
            if aa:
                seq_chars.append(aa)
            else:
                seq_chars.append('X')
        if len(seq_chars) >= 10:
            sequences[str(bmrb_id)] = ''.join(seq_chars)

    return sequences


def run_mmseqs2_clustering(sequences, identity_threshold=0.90, coverage=0.80):
    """Run MMseqs2 easy-cluster and return cluster membership.

    Args:
        sequences: dict bmrb_id -> sequence string
        identity_threshold: minimum sequence identity (0-1)
        coverage: minimum alignment coverage (0-1)

    Returns:
        dict: bmrb_id -> set of bmrb_ids in the same cluster (including self)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write FASTA
        fasta_path = os.path.join(tmpdir, 'sequences.fasta')
        with open(fasta_path, 'w') as f:
            for bmrb_id, seq in sequences.items():
                f.write(f'>{bmrb_id}\n{seq}\n')

        out_prefix = os.path.join(tmpdir, 'cluster')
        tmp_dir = os.path.join(tmpdir, 'tmp')
        os.makedirs(tmp_dir, exist_ok=True)

        cmd = [
            'mmseqs', 'easy-cluster',
            fasta_path, out_prefix, tmp_dir,
            '--min-seq-id', str(identity_threshold),
            '-c', str(coverage),
            '--cov-mode', '0',  # bidirectional coverage
            '--threads', '4',
        ]

        print(f"  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"  MMseqs2 stderr: {result.stderr}")
            raise RuntimeError(f"MMseqs2 failed with return code {result.returncode}")

        # Parse cluster TSV: representative\tmember
        cluster_tsv = out_prefix + '_cluster.tsv'
        clusters = {}  # representative -> set of members

        with open(cluster_tsv) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    rep, member = parts[0], parts[1]
                    if rep not in clusters:
                        clusters[rep] = set()
                    clusters[rep].add(member)

        # Build exclusion map: each protein maps to all others in its cluster
        exclusion_map = {}
        for rep, members in clusters.items():
            for m in members:
                # Exclude all other cluster members (not self)
                exclusion_map[m] = sorted(members - {m})

        return exclusion_map


def main():
    parser = argparse.ArgumentParser(
        description='Cluster sequences at 90%% identity for retrieval exclusion.'
    )
    parser.add_argument(
        '--data_dir', default=DATA_DIR,
        help='Directory containing compiled CSV(s)'
    )
    parser.add_argument(
        '--identity', type=float, default=0.90,
        help='Sequence identity threshold (default: 0.90)'
    )
    parser.add_argument(
        '--output', default=None,
        help='Output JSON path (default: <data_dir>/identity_clusters_90.json)'
    )
    args = parser.parse_args()

    if args.output is None:
        pct = int(args.identity * 100)
        args.output = os.path.join(args.data_dir, f'identity_clusters_{pct}.json')

    # Find the dataset CSV
    csv_path = None
    for name in ['structure_data_hybrid.csv', 'structure_data.csv',
                  'compiled_dataset.csv', 'structure_data_experimental.csv']:
        candidate = os.path.join(args.data_dir, name)
        if os.path.exists(candidate):
            csv_path = candidate
            break

    if csv_path is None:
        print(f"ERROR: No dataset CSV found in {args.data_dir}")
        sys.exit(1)

    print(f"Extracting sequences from {csv_path}...")
    sequences = extract_sequences(csv_path)
    print(f"  {len(sequences)} proteins with sequences >= 10 residues")

    print(f"\nClustering at {args.identity:.0%} identity...")
    exclusion_map = run_mmseqs2_clustering(sequences, identity_threshold=args.identity)

    # Count stats
    n_with_neighbors = sum(1 for v in exclusion_map.values() if v)
    total_exclusions = sum(len(v) for v in exclusion_map.values())

    print(f"\n  Exclusion map:")
    print(f"    Proteins with >90% identity neighbors: {n_with_neighbors}")
    print(f"    Total exclusion pairs: {total_exclusions}")

    with open(args.output, 'w') as f:
        json.dump(exclusion_map, f, indent=2)
    print(f"\n  Saved to {args.output}")


if __name__ == '__main__':
    main()
