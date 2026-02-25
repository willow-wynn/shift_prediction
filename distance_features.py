"""
Distance feature computation using all resolved atoms.

Computes pairwise intra-residue distances with the following rules:
  - Heavy-heavy: all pairwise distances between non-hydrogen atoms
  - H-to-heavy: distances from each hydrogen to each heavy atom
  - No H-H distances

Also retains sidechain summary statistics from the previous approach.
"""

import numpy as np
from pdb_utils import classify_atom


def calc_all_distances(atoms):
    """Calculate distances between all atom pairs obeying H-heavy rules.

    Rules:
      - heavy-heavy: include
      - hydrogen-heavy: include
      - hydrogen-hydrogen: exclude

    Args:
        atoms: Dict mapping atom_name -> np.array([x, y, z])

    Returns:
        Dict mapping 'dist_{A1}_{A2}' -> float distance
        (A1 < A2 lexicographically to avoid duplicates)
    """
    names = sorted(atoms.keys())
    heavy = [n for n in names if classify_atom(n) == 'heavy']
    hydrogen = [n for n in names if classify_atom(n) == 'hydrogen']

    dists = {}

    # Heavy-heavy pairs
    for i, a1 in enumerate(heavy):
        coord1 = atoms[a1]
        if not np.all(np.isfinite(coord1)):
            continue
        for a2 in heavy[i + 1:]:
            coord2 = atoms[a2]
            if not np.all(np.isfinite(coord2)):
                continue
            d = float(np.linalg.norm(coord1 - coord2))
            if np.isfinite(d):
                dists[f'dist_{a1}_{a2}'] = d

    # Hydrogen-to-heavy pairs
    for h in hydrogen:
        coord_h = atoms[h]
        if not np.all(np.isfinite(coord_h)):
            continue
        for hv in heavy:
            coord_hv = atoms[hv]
            if not np.all(np.isfinite(coord_hv)):
                continue
            d = float(np.linalg.norm(coord_h - coord_hv))
            if np.isfinite(d):
                # Ensure consistent key ordering
                key_a, key_b = sorted([h, hv])
                dists[f'dist_{key_a}_{key_b}'] = d

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

    sc_coords = []
    for name, coord in atoms.items():
        if name not in backbone_set and np.all(np.isfinite(coord)):
            sc_coords.append(coord)

    result = {'sc_n_resolved': len(sc_coords)}

    if len(sc_coords) == 0:
        return result

    sc_coords = np.array(sc_coords)
    dists_from_ca = np.linalg.norm(sc_coords - ca, axis=1)

    result['sc_mean_dist_ca'] = float(np.mean(dists_from_ca))
    result['sc_compactness'] = float(np.std(dists_from_ca)) if len(dists_from_ca) > 1 else 0.0
    result['sc_max_extent'] = float(np.max(dists_from_ca))

    centroid = np.mean(sc_coords, axis=0)
    result['sc_centroid_dist'] = float(np.linalg.norm(centroid - ca))

    return result


def get_sidechain_summary_names():
    """Get column names for sidechain summary statistics."""
    return [
        'sc_n_resolved',
        'sc_mean_dist_ca',
        'sc_compactness',
        'sc_max_extent',
        'sc_centroid_dist',
    ]


def compute_all_distance_features(atoms):
    """Compute all distance features + sidechain summaries.

    Args:
        atoms: Dict mapping atom_name -> np.array([x, y, z])

    Returns:
        Dict of all distance features
    """
    features = {}
    features.update(calc_all_distances(atoms))
    features.update(calc_sidechain_summary(atoms))
    return features
