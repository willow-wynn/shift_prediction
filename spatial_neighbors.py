"""
Spatial neighbor computation using CA atom positions.

Finds the K nearest residues in 3D space with minimum sequence separation,
exactly matching the existing pipeline's approach.
"""

import numpy as np
from scipy.spatial import cKDTree
from config import K_SPATIAL_NEIGHBORS, MIN_SEQ_SEPARATION


def find_neighbors(struct_data, k=None, min_sep=None):
    """Find K nearest spatial neighbors for each residue.

    Uses CA atom positions and enforces minimum sequence separation.

    Args:
        struct_data: Dict mapping residue_id -> {atoms: {name: coord}, ...}
        k: Number of neighbors (default from config)
        min_sep: Minimum sequence separation (default from config)

    Returns:
        Dict mapping residue_id -> {
            ids: [k neighbor residue IDs],
            dists: [k distances in Angstroms],
            seps: [k sequence separations],
        }
        Padded with -1 if fewer than k valid neighbors.
    """
    if k is None:
        k = K_SPATIAL_NEIGHBORS
    if min_sep is None:
        min_sep = MIN_SEQ_SEPARATION

    # Collect CA coordinates
    res_ids = []
    coords = []
    for rid in sorted(struct_data.keys()):
        atoms = struct_data[rid].get('atoms', {})
        if 'CA' in atoms and np.all(np.isfinite(atoms['CA'])):
            res_ids.append(rid)
            coords.append(atoms['CA'])

    if len(res_ids) < 2:
        return {}

    res_ids = np.array(res_ids)
    coords = np.array(coords)

    # Build KD-tree
    tree = cKDTree(coords)

    # Query more than k to allow for sequence separation filtering
    query_k = min(len(res_ids), k + min_sep * 2 + 5)
    dists, idxs = tree.query(coords, k=query_k)

    neighbors = {}
    for i, rid in enumerate(res_ids):
        nids, ndists, nseps = [], [], []

        for j in range(query_k):
            if len(nids) >= k:
                break
            nrid = res_ids[idxs[i, j]]
            if nrid == rid:
                continue
            sep = abs(int(nrid) - int(rid))
            if sep < min_sep:
                continue
            nids.append(int(nrid))
            ndists.append(float(dists[i, j]))
            nseps.append(sep)

        # Pad to k
        while len(nids) < k:
            nids.append(-1)
            ndists.append(-1.0)
            nseps.append(-1)

        neighbors[rid] = {
            'ids': nids[:k],
            'dists': ndists[:k],
            'seps': nseps[:k],
        }

    return neighbors
