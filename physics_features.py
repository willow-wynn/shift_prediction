"""
Physics-based features for chemical shift prediction.

Retained feature:
    - Hydrogen bond geometry from DSSP

Removed (previously present):
    - Aromatic ring current effects
    - Half-sphere exposure (HSE)
    - Order parameter from B-factors
"""

import numpy as np


def compute_hbond_geometry(dssp_data, all_coords):
    """Compute hydrogen bond donor-acceptor geometry from DSSP data.

    DSSP assigns up to four hydrogen bonds per residue (two N-H...O=C and
    two O=C...H-N patterns) as relative residue index offsets and estimated
    energies. This function resolves those relative indices to actual
    donor-acceptor distances using the backbone N and O coordinates.

    Parameters
    ----------
    dssp_data : dict
        DSSP feature dictionary for the target residue. Expected keys:
            'nh_o_1_relidx', 'nh_o_1_energy',
            'o_nh_1_relidx', 'o_nh_1_energy',
            'residue_idx' : int
    all_coords : dict or list
        Per-residue coordinate data.

    Returns
    -------
    dict
        'hbond_dist_1', 'hbond_energy_1', 'hbond_dist_2', 'hbond_energy_2'
    """
    result = {
        'hbond_dist_1': np.nan,
        'hbond_energy_1': np.nan,
        'hbond_dist_2': np.nan,
        'hbond_energy_2': np.nan,
    }

    if dssp_data is None:
        return result

    res_idx = dssp_data.get('residue_idx', None)
    if res_idx is None:
        return result

    rel1 = dssp_data.get('nh_o_1_relidx', 0)
    energy1 = dssp_data.get('nh_o_1_energy', np.nan)

    if rel1 != 0 and not np.isnan(energy1):
        partner_idx = res_idx + int(rel1)
        dist1 = _get_no_distance(res_idx, partner_idx, all_coords, 'N', 'O')
        result['hbond_dist_1'] = dist1
        result['hbond_energy_1'] = float(energy1)

    rel2 = dssp_data.get('o_nh_1_relidx', 0)
    energy2 = dssp_data.get('o_nh_1_energy', np.nan)

    if rel2 != 0 and not np.isnan(energy2):
        partner_idx = res_idx + int(rel2)
        dist2 = _get_no_distance(partner_idx, res_idx, all_coords, 'N', 'O')
        result['hbond_dist_2'] = dist2
        result['hbond_energy_2'] = float(energy2)

    return result


def _get_no_distance(donor_idx, acceptor_idx, all_coords, donor_atom='N', acceptor_atom='O'):
    """Compute the N...O distance between a donor and acceptor residue."""
    try:
        if isinstance(all_coords, dict):
            donor_coord = all_coords[donor_idx].get(donor_atom)
            acceptor_coord = all_coords[acceptor_idx].get(acceptor_atom)
        else:
            if donor_idx < 0 or donor_idx >= len(all_coords):
                return np.nan
            if acceptor_idx < 0 or acceptor_idx >= len(all_coords):
                return np.nan
            donor_coord = all_coords[donor_idx].get(donor_atom)
            acceptor_coord = all_coords[acceptor_idx].get(acceptor_atom)

        if donor_coord is None or acceptor_coord is None:
            return np.nan

        donor_coord = np.asarray(donor_coord, dtype=np.float64)
        acceptor_coord = np.asarray(acceptor_coord, dtype=np.float64)

        return float(np.linalg.norm(donor_coord - acceptor_coord))

    except (KeyError, IndexError, TypeError):
        return np.nan


def compute_all_physics_features(dssp_data=None, all_coords=None):
    """Compute all physics-based features for a single residue.

    Parameters
    ----------
    dssp_data : dict or None
        DSSP features for the residue.
    all_coords : dict or list or None
        Per-residue atom coordinate dicts.

    Returns
    -------
    dict
        H-bond geometry features.
    """
    if dssp_data is not None and all_coords is not None:
        return compute_hbond_geometry(dssp_data, all_coords)
    return {
        'hbond_dist_1': np.nan,
        'hbond_energy_1': np.nan,
        'hbond_dist_2': np.nan,
        'hbond_energy_2': np.nan,
    }


def get_physics_feature_names():
    """Return the complete list of physics feature column names."""
    return ['hbond_dist_1', 'hbond_energy_1', 'hbond_dist_2', 'hbond_energy_2']
