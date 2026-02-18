"""
Dense distance feature computation.

Instead of computing all 863 pairwise distances (88% NaN), we compute only:
1. ~21 backbone atom pair distances (C, CA, CB, H, HA, N, O -- all pairs = 21)
2. 5 sidechain summary statistics per residue

This gives ~26 features with much lower NaN rates.
"""

import numpy as np
from config import DISTANCE_ATOMS


def get_distance_column_names():
    """Get the list of dense distance column names.

    Returns:
        List of column name strings like 'dist_C_CA', 'dist_C_CB', etc.
    """
    atoms = sorted(DISTANCE_ATOMS)
    cols = []
    for i, a1 in enumerate(atoms):
        for a2 in atoms[i+1:]:
            cols.append(f'dist_{a1}_{a2}')
    return cols


def get_sidechain_summary_names():
    """Get column names for sidechain summary statistics."""
    return [
        'sc_n_resolved',      # Number of resolved sidechain atoms
        'sc_mean_dist_ca',    # Mean distance of SC atoms from CA
        'sc_compactness',     # Std of SC atom distances from CA (how compact)
        'sc_max_extent',      # Max distance of any SC atom from CA
        'sc_centroid_dist',   # Distance from CA to SC centroid
    ]


def calc_dense_distances(atoms):
    """Calculate distances between dense backbone/near-backbone atom pairs.

    Args:
        atoms: Dict mapping atom_name -> np.array([x, y, z])

    Returns:
        Dict mapping 'dist_A1_A2' -> float distance
    """
    distance_atoms = sorted(DISTANCE_ATOMS)
    dists = {}

    for i, a1 in enumerate(distance_atoms):
        if a1 not in atoms:
            continue
        for a2 in distance_atoms[i+1:]:
            if a2 not in atoms:
                continue
            try:
                d = np.linalg.norm(atoms[a1] - atoms[a2])
                if np.isfinite(d):
                    dists[f'dist_{a1}_{a2}'] = float(d)
            except Exception:
                pass

    return dists


def calc_sidechain_summary(atoms, backbone_set=None):
    """Compute sidechain summary statistics for a residue.

    Args:
        atoms: Dict mapping atom_name -> np.array([x, y, z])
        backbone_set: Set of backbone atom names (default: standard set)

    Returns:
        Dict with sidechain summary features
    """
    if backbone_set is None:
        backbone_set = {'N', 'H', 'HN', 'CA', 'HA', 'HA2', 'HA3', 'CB', 'C', 'O'}

    ca = atoms.get('CA')
    if ca is None or not np.all(np.isfinite(ca)):
        return {}

    # Collect sidechain atom coordinates
    sc_coords = []
    for name, coord in atoms.items():
        if name not in backbone_set and np.all(np.isfinite(coord)):
            sc_coords.append(coord)

    result = {'sc_n_resolved': len(sc_coords)}

    if len(sc_coords) == 0:
        return result

    sc_coords = np.array(sc_coords)

    # Distances from CA to each sidechain atom
    dists_from_ca = np.linalg.norm(sc_coords - ca, axis=1)

    result['sc_mean_dist_ca'] = float(np.mean(dists_from_ca))
    result['sc_compactness'] = float(np.std(dists_from_ca)) if len(dists_from_ca) > 1 else 0.0
    result['sc_max_extent'] = float(np.max(dists_from_ca))

    # Centroid distance
    centroid = np.mean(sc_coords, axis=0)
    result['sc_centroid_dist'] = float(np.linalg.norm(centroid - ca))

    return result


def compute_all_distance_features(atoms):
    """Compute both dense distances and sidechain summaries.

    Args:
        atoms: Dict mapping atom_name -> np.array([x, y, z])

    Returns:
        Dict of all distance features
    """
    features = {}
    features.update(calc_dense_distances(atoms))
    features.update(calc_sidechain_summary(atoms))
    return features
