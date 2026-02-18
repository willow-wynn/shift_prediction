"""
Sequence alignment utilities for structure-shift matching.

Uses Biopython PairwiseAligner with the same parameters as the existing pipeline:
  match=2, mismatch=-1, gap_open=-2, gap_extend=-0.5
"""

import pandas as pd
import numpy as np
from Bio import Align
from tqdm import tqdm
from config import AA_3_TO_1, NUCLEOTIDES


def is_amino_acid(residue_name):
    """Check if residue is an amino acid (not nucleotide)."""
    res = str(residue_name).upper().strip()
    if res in NUCLEOTIDES:
        return False
    if res.startswith('D') and len(res) == 2:
        return False
    return res in AA_3_TO_1


def to_single_letter(codes):
    """Convert list of 3-letter codes to single-letter sequence."""
    return ''.join(AA_3_TO_1.get(str(c).upper(), 'X') for c in codes)


def extract_sequences(df, id_col, res_id_col, res_name_col):
    """Extract sequences from a dataframe.

    Args:
        df: DataFrame with protein data
        id_col: Column name for protein ID
        res_id_col: Column name for residue ID
        res_name_col: Column name for residue name (3-letter code)

    Returns:
        Dict mapping protein_id -> {residue_ids, residue_names, sequence}
    """
    sequences = {}
    for prot_id in df[id_col].unique():
        prot_df = df[df[id_col] == prot_id]
        residues = prot_df[[res_id_col, res_name_col]].drop_duplicates().sort_values(res_id_col)
        residues = residues[residues[res_name_col].apply(is_amino_acid)]
        if len(residues) > 0:
            sequences[prot_id] = {
                'residue_ids': residues[res_id_col].tolist(),
                'residue_names': residues[res_name_col].tolist(),
                'sequence': to_single_letter(residues[res_name_col].tolist()),
            }
    return sequences


def align_sequences(seq1, seq2):
    """Perform global pairwise alignment.

    Returns:
        Best alignment object, or None on failure
    """
    aligner = Align.PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = 2
    aligner.mismatch_score = -1
    aligner.open_gap_score = -2
    aligner.extend_gap_score = -0.5

    alignments = aligner.align(seq1, seq2)
    return alignments[0] if alignments else None


def align_proteins(struct_seqs, shift_seqs):
    """Align structure and shift sequences, return residue-level mapping.

    Args:
        struct_seqs: Dict from extract_sequences (structure data)
        shift_seqs: Dict from extract_sequences (shift data)

    Returns:
        DataFrame with columns: protein_id, struct_res_id, shift_res_id, match
    """
    common = sorted(set(struct_seqs.keys()) & set(shift_seqs.keys()))
    print(f"Proteins with both structure and shift data: {len(common)}")

    all_mappings = []
    failed = 0

    for prot_id in tqdm(common, desc="Aligning sequences"):
        s1 = struct_seqs[prot_id]
        s2 = shift_seqs[prot_id]

        if not s1['sequence'] or not s2['sequence']:
            failed += 1
            continue

        try:
            alignment = align_sequences(s1['sequence'], s2['sequence'])
            if alignment is None:
                failed += 1
                continue

            a1, a2 = str(alignment[0]), str(alignment[1])
            i1, i2 = 0, 0

            for c1, c2 in zip(a1, a2):
                if c1 != '-' and c2 != '-':
                    all_mappings.append({
                        'protein_id': prot_id,
                        'struct_res_id': s1['residue_ids'][i1],
                        'shift_res_id': s2['residue_ids'][i2],
                        'match': c1 == c2,
                    })
                if c1 != '-':
                    i1 += 1
                if c2 != '-':
                    i2 += 1
        except Exception:
            failed += 1
            continue

    if failed > 0:
        print(f"  Failed alignments: {failed}")
    print(f"  Total aligned residues: {len(all_mappings)}")

    return pd.DataFrame(all_mappings)
