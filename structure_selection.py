"""
RMSD-based structure/chain selection using Kabsch superposition.

For proteins with multiple candidate PDB structures or chains, selects the
single best chain by:
  1. Aligning each candidate chain's sequence to the shift sequence
  2. Superimposing all chains onto a common reference using Kabsch on shared CA atoms
  3. Computing median coordinates and per-chain RMSD from median
  4. Selecting the chain with lowest RMSD
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


def _get_common_ca(alignment, struct_rids, struct_ca, shift_rids):
    """From an alignment, get the mapping of shift residue positions to CA coords.

    Returns list of (shift_pos_index, ca_coord) for matched positions that have CA.
    """
    a1, a2 = str(alignment[0]), str(alignment[1])
    i1, i2 = 0, 0
    matched = []

    for c1, c2 in zip(a1, a2):
        if c1 != '-' and c2 != '-':
            srid = struct_rids[i1]
            if srid in struct_ca:
                matched.append((i2, struct_ca[srid]))
        if c1 != '-':
            i1 += 1
        if c2 != '-':
            i2 += 1

    return matched


def select_best_chain(candidates, shift_sequence):
    """Select the best PDB chain from multiple candidates via RMSD analysis.

    For each candidate chain:
      1. Align its sequence to shift_sequence
      2. Extract CA coordinates at aligned positions

    Then across all chains:
      3. Find positions present in ALL chains
      4. Superimpose all chains onto the first using Kabsch
      5. Compute median coordinates per position
      6. Compute per-chain RMSD from median
      7. Return the chain with lowest RMSD

    If only one candidate, returns it directly.

    Args:
        candidates: list of (pdb_path, chain_id) tuples
        shift_sequence: single-letter amino acid sequence from shift data

    Returns:
        (pdb_path, chain_id, alignment) for the best chain,
        or (None, None, None) if no valid candidates
    """
    if not candidates or not shift_sequence:
        return None, None, None

    # Collect aligned CA coords for each candidate
    chain_data = []  # list of (pdb_path, chain_id, alignment, {shift_pos: ca_coord})

    for pdb_path, chain_id in candidates:
        try:
            residues = parse_pdb(pdb_path, chain_id=chain_id)
        except Exception:
            continue

        seq, rids, ca_coords = _extract_chain_sequence(residues, chain_id)
        if seq is None or not ca_coords:
            # Try without chain filter if chain_id didn't match
            all_chains = set(ch for ch, _ in residues.keys())
            if chain_id not in all_chains and all_chains:
                chain_id = sorted(all_chains)[0]
                seq, rids, ca_coords = _extract_chain_sequence(residues, chain_id)
                if seq is None or not ca_coords:
                    continue
            else:
                continue

        alignment = align_sequences(seq, shift_sequence)
        if alignment is None:
            continue

        matched = _get_common_ca(alignment, rids, ca_coords, list(range(len(shift_sequence))))
        if len(matched) < 10:
            continue

        pos_to_ca = {pos: coord for pos, coord in matched}
        chain_data.append((pdb_path, chain_id, alignment, pos_to_ca))

    if not chain_data:
        return None, None, None

    if len(chain_data) == 1:
        return chain_data[0][0], chain_data[0][1], chain_data[0][2]

    # Find positions common to ALL chains
    common_positions = set(chain_data[0][3].keys())
    for _, _, _, pos_map in chain_data[1:]:
        common_positions &= set(pos_map.keys())

    common_positions = sorted(common_positions)
    if len(common_positions) < 10:
        # Not enough overlap; return chain with most aligned positions
        best = max(chain_data, key=lambda x: len(x[3]))
        return best[0], best[1], best[2]

    # Extract coordinate matrices (N_chains x N_positions x 3)
    n_chains = len(chain_data)
    n_pos = len(common_positions)
    all_coords = np.zeros((n_chains, n_pos, 3))

    for ci, (_, _, _, pos_map) in enumerate(chain_data):
        for pi, pos in enumerate(common_positions):
            all_coords[ci, pi, :] = pos_map[pos]

    # Superimpose all chains onto the first
    ref_coords = all_coords[0]
    aligned_coords = np.zeros_like(all_coords)
    aligned_coords[0] = ref_coords

    for ci in range(1, n_chains):
        rot, trans = kabsch_superimpose(all_coords[ci], ref_coords)
        aligned_coords[ci] = all_coords[ci] @ rot + trans

    # Compute median coordinates per position
    median_coords = np.median(aligned_coords, axis=0)

    # Compute per-chain RMSD from median
    rmsds = np.zeros(n_chains)
    for ci in range(n_chains):
        diff = aligned_coords[ci] - median_coords
        rmsds[ci] = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))

    best_idx = int(np.argmin(rmsds))
    return chain_data[best_idx][0], chain_data[best_idx][1], chain_data[best_idx][2]
