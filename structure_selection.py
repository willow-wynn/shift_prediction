"""
Alignment-identity-based structure/chain selection with Kabsch superposition.

For proteins with multiple candidate PDB structures or chains, selects the
single best chain by:
  1. Aligning each candidate chain's sequence to the shift sequence
  2. Computing sequence identity (n_match / n_aligned)
  3. Picking the chain with highest identity

Also provides Kabsch superposition utilities for NMR model selection.
"""

import numpy as np
from alignment import align_sequences
from config import AA_3_TO_1
from pdb_utils import parse_pdb


def kabsch_superimpose(coords_mobile, coords_target):
    """Superimpose mobile coordinates onto target using the Kabsch algorithm.

    Finds the optimal rotation and translation that minimizes RMSD between
    two sets of corresponding 3D points.

    Args:
        coords_mobile: np.ndarray of shape (N, 3) — points to be moved
        coords_target: np.ndarray of shape (N, 3) — reference points

    Returns:
        (rotation, translation) where:
            rotation: (3, 3) rotation matrix
            translation: (3,) translation vector
        Apply as: aligned = (coords_mobile - centroid_mobile) @ rotation + centroid_target
    """
    coords_mobile = np.asarray(coords_mobile, dtype=np.float64)
    coords_target = np.asarray(coords_target, dtype=np.float64)

    centroid_m = coords_mobile.mean(axis=0)
    centroid_t = coords_target.mean(axis=0)

    centered_m = coords_mobile - centroid_m
    centered_t = coords_target - centroid_t

    H = centered_m.T @ centered_t
    U, S, Vt = np.linalg.svd(H)

    # Correct for reflection
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1.0, 1.0, np.sign(d)])

    rotation = Vt.T @ sign_matrix @ U.T
    translation = centroid_t - centroid_m @ rotation

    return rotation, translation


def _extract_chain_sequence(residues, chain_id):
    """Extract single-letter sequence and CA coords for a specific chain.

    Args:
        residues: dict from parse_pdb — {(chain, res_id): {residue_name, atoms}}
        chain_id: chain letter to extract

    Returns:
        (sequence_str, res_ids, ca_coords_dict) or (None, None, None) if empty
        ca_coords_dict maps res_id -> np.array([x,y,z])
    """
    items = sorted(
        ((rid, rdata) for (ch, rid), rdata in residues.items() if ch == chain_id),
        key=lambda x: x[0],
    )
    if not items:
        return None, None, None

    seq_chars = []
    res_ids = []
    ca_coords = {}

    for rid, rdata in items:
        rname = rdata['residue_name']
        aa = AA_3_TO_1.get(rname)
        if aa is None:
            continue
        seq_chars.append(aa)
        res_ids.append(rid)
        ca = rdata['atoms'].get('CA')
        if ca is not None and np.all(np.isfinite(ca)):
            ca_coords[rid] = ca

    if not seq_chars:
        return None, None, None

    return ''.join(seq_chars), res_ids, ca_coords


def _compute_alignment_identity(alignment):
    """Compute sequence identity from an alignment.

    Returns:
        (n_match, n_aligned, identity) where identity = n_match / n_aligned
    """
    a1, a2 = str(alignment[0]), str(alignment[1])
    n_match = 0
    n_aligned = 0
    for c1, c2 in zip(a1, a2):
        if c1 != '-' and c2 != '-':
            n_aligned += 1
            if c1 == c2:
                n_match += 1
    if n_aligned == 0:
        return 0, 0, 0.0
    return n_match, n_aligned, n_match / n_aligned


def select_best_chain(candidates, shift_sequence):
    """Select the best PDB chain from multiple candidates via alignment identity.

    For each candidate chain:
      1. Align its sequence to shift_sequence
      2. Compute sequence identity

    Pick the chain with the highest sequence identity.

    Args:
        candidates: list of (pdb_path, chain_id) tuples
        shift_sequence: single-letter amino acid sequence from shift data

    Returns:
        (pdb_path, chain_id, alignment) for the best chain,
        or (None, None, None) if no valid candidates
    """
    if not candidates or not shift_sequence:
        return None, None, None

    # Collect alignment data for each candidate
    chain_data = []  # list of (pdb_path, chain_id, alignment, identity, n_aligned)

    for pdb_path, chain_id in candidates:
        try:
            residues = parse_pdb(pdb_path, chain_id=chain_id)
        except Exception:
            continue

        seq, rids, ca_coords = _extract_chain_sequence(residues, chain_id)
        if seq is None:
            # Try without chain filter if chain_id didn't match
            all_chains = set(ch for ch, _ in residues.keys())
            if chain_id not in all_chains and all_chains:
                chain_id = sorted(all_chains)[0]
                seq, rids, ca_coords = _extract_chain_sequence(residues, chain_id)
                if seq is None:
                    continue
            else:
                continue

        alignment = align_sequences(seq, shift_sequence)
        if alignment is None:
            continue

        n_match, n_aligned, identity = _compute_alignment_identity(alignment)
        if n_aligned < 10:
            continue

        chain_data.append((pdb_path, chain_id, alignment, identity, n_aligned))

    if not chain_data:
        return None, None, None

    if len(chain_data) == 1:
        return chain_data[0][0], chain_data[0][1], chain_data[0][2]

    # Pick chain with highest identity, breaking ties by n_aligned
    best = max(chain_data, key=lambda x: (x[3], x[4]))
    return best[0], best[1], best[2]
