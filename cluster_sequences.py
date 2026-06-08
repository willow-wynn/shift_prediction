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


def _run_mmseqs_easy_cluster(fasta_path, out_prefix, tmp_dir,
                             identity_threshold, coverage, cov_mode):
    """Run a single `mmseqs easy-cluster` and parse its cluster TSV.

    Returns dict: representative -> set of members (members include the rep).
    """
    cmd = [
        'mmseqs', 'easy-cluster',
        fasta_path, out_prefix, tmp_dir,
        '--min-seq-id', str(identity_threshold),
        '-c', str(coverage),
        '--cov-mode', str(cov_mode),
        '--threads', '4',
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"  MMseqs2 stderr: {result.stderr}")
        raise RuntimeError(f"MMseqs2 failed with return code {result.returncode}")

    cluster_tsv = out_prefix + '_cluster.tsv'
    clusters = {}  # representative -> set of members
    with open(cluster_tsv) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                rep, member = parts[0], parts[1]
                clusters.setdefault(rep, set()).add(member)
    return clusters


def _clusters_to_exclusion_map(clusters):
    """rep->{members} dict to bmrb_id -> sorted [other members] exclusion map."""
    exclusion_map = {}
    for rep, members in clusters.items():
        for m in members:
            # Exclude all other cluster members (not self)
            exclusion_map[m] = sorted(members - {m})
    return exclusion_map


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

        clusters = _run_mmseqs_easy_cluster(
            fasta_path, out_prefix, tmp_dir,
            identity_threshold, coverage, cov_mode=0)  # bidirectional coverage
        return _clusters_to_exclusion_map(clusters)


def run_mmseqs2_containment_clustering(sequences, identity_threshold=0.90,
                                       coverage=0.80):
    """Containment clustering: catch length-mismatched same-protein twins.

    The default bidirectional-coverage clustering (`--cov-mode 0`) never groups
    two sequences that are ~identical over the SHORTER one but differ in length
    (tag/construct/peptide-vs-full). Those twins are the residual fold-split leak
    (finding #1). Here we re-cluster with `--cov-mode 1` (coverage of the
    TARGET, i.e. the shorter representative) at the same `--min-seq-id`, so a
    short sequence contained in a longer one joins the longer one's cluster.

    Kept SEPARATE from run_mmseqs2_clustering so the canonical 80%-bidirectional
    identity_clusters_90.json stays byte-for-byte back-compatible; 02 unions
    these containment edges in addition to the existing ones.

    Returns:
        dict: bmrb_id -> sorted [other members of its containment cluster]
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        fasta_path = os.path.join(tmpdir, 'sequences.fasta')
        with open(fasta_path, 'w') as f:
            for bmrb_id, seq in sequences.items():
                f.write(f'>{bmrb_id}\n{seq}\n')

        out_prefix = os.path.join(tmpdir, 'cluster_contain')
        tmp_dir = os.path.join(tmpdir, 'tmp')
        os.makedirs(tmp_dir, exist_ok=True)

        clusters = _run_mmseqs_easy_cluster(
            fasta_path, out_prefix, tmp_dir,
            identity_threshold, coverage, cov_mode=1)  # coverage of target
        return _clusters_to_exclusion_map(clusters)


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

    pct = int(args.identity * 100)
    if args.output is None:
        args.output = os.path.join(args.data_dir, f'identity_clusters_{pct}.json')
    # Containment-cluster output sits next to the main clusters file. 02 unions
    # these edges so length-mismatched same-protein twins (substring twins)
    # don't get split across folds (leak finding #1).
    containment_output = os.path.join(
        args.data_dir, f'containment_clusters_{pct}.json')

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

    # --- Containment clustering (finding #1: substring twins) ---
    print(f"\nContainment clustering at {args.identity:.0%} identity "
          f"(--cov-mode 1, coverage of shorter target)...")
    containment_map = run_mmseqs2_containment_clustering(
        sequences, identity_threshold=args.identity)
    c_with_neighbors = sum(1 for v in containment_map.values() if v)
    c_total = sum(len(v) for v in containment_map.values())
    print(f"  Containment map:")
    print(f"    Proteins with containment neighbors: {c_with_neighbors}")
    print(f"    Total containment pairs: {c_total}")
    with open(containment_output, 'w') as f:
        json.dump(containment_map, f, indent=2)
    print(f"  Saved to {containment_output}")


if __name__ == '__main__':
    main()
