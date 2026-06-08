"""Fixed, citation-grounded chemical-shift normalization constants.

PURPOSE: replace per-fold, data-estimated z-norm with FIXED published constants so
normalization introduces ZERO cross-fold / held-out leakage and is identical for every
fold. target = (shift - center[AA,atom]) / scale[AA,atom].

SOURCES (no invented numbers):
  * Backbone centers (CA,CB,C,N,H,HA): random-coil shifts, Schwarzinger et al. 2001
    (JACS 123:2970) + Wishart et al. 1995 (J.Biomol.NMR 5:67)  -- via random_coil.RC_SHIFTS.
  * Sidechain centers: BMRB filtered per-(AA,atom) MEAN shift.
  * ALL scales (sigma): BMRB filtered per-(AA,atom) STANDARD DEVIATION
    (bmrb.io/ref_info/csstats.php?restype=aa&set=filt, full BMRB database, 3-sigma outlier
    filtered). Filtered (not raw 'full') chosen because the unfiltered set is inflated by
    mis-referenced deposits (e.g. PRO N sigma 19.7 -> 5.1 filtered).
  * Individual methyl protons (HD11/HG21/...) map to the BMRB methyl pseudo-atom (MD1/MG2/MD/MG/MB).

This table is a frozen snapshot fetched 2026-06-08; regenerate via experiments/.../gen_fixed_norm.py.
"""

BMRB_FILT = {
    'ALA': {'C': (177.813, 2.012), 'CA': (53.128, 1.916), 'CB': (18.958, 1.742), 'H': (8.196, 0.57), 'HA': (4.237, 0.419), 'MB': (1.356, 0.244), 'N': (123.404, 3.403)},
    'ARG': {'C': (176.487, 1.956), 'CA': (56.755, 2.254), 'CB': (30.622, 1.767), 'CD': (43.163, 0.919), 'CG': (27.212, 1.191), 'CZ': (159.874, 3.525), 'H': (8.234, 0.595), 'HA': (4.285, 0.443), 'HB2': (1.795, 0.254), 'HB3': (1.765, 0.267), 'HD2': (3.118, 0.227), 'HD3': (3.103, 0.254), 'HE': (7.357, 0.567), 'HG2': (1.566, 0.26), 'HG3': (1.547, 0.277), 'HH11': (6.893, 0.428), 'HH12': (6.844, 0.356), 'HH21': (6.805, 0.395), 'HH22': (6.795, 0.348), 'N': (120.921, 3.569), 'NE': (84.594, 1.605), 'NH1': (74.135, 4.857), 'NH2': (72.769, 2.916)},
    'ASN': {'C': (175.28, 1.739), 'CA': (53.51, 1.836), 'CB': (38.679, 1.633), 'CG': (176.717, 2.352), 'H': (8.322, 0.607), 'HA': (4.658, 0.348), 'HB2': (2.802, 0.306), 'HB3': (2.748, 0.325), 'HD21': (7.327, 0.474), 'HD22': (7.148, 0.49), 'N': (118.986, 3.821), 'ND2': (112.753, 2.233)},
    'ASP': {'C': (176.421, 1.667), 'CA': (54.659, 1.984), 'CB': (40.866, 1.572), 'CG': (179.237, 1.85), 'H': (8.294, 0.551), 'HA': (4.581, 0.306), 'HB2': (2.711, 0.259), 'HB3': (2.66, 0.266), 'HD2': (6.567, 3.358), 'N': (120.754, 3.698)},
    'CYS': {'C': (174.805, 2.031), 'CA': (58.021, 3.447), 'CB': (33.412, 6.562), 'H': (8.377, 0.672), 'HA': (4.655, 0.534), 'HB2': (2.955, 0.427), 'HB3': (2.892, 0.438), 'HG': (2.069, 1.277), 'N': (120.092, 4.352)},
    'GLN': {'C': (176.329, 1.864), 'CA': (56.517, 2.062), 'CB': (29.15, 1.745), 'CD': (179.697, 1.257), 'CG': (33.784, 1.09), 'H': (8.219, 0.556), 'HA': (4.257, 0.416), 'HB2': (2.044, 0.247), 'HB3': (2.015, 0.26), 'HE21': (7.227, 0.439), 'HE22': (7.035, 0.434), 'HG2': (2.315, 0.258), 'HG3': (2.295, 0.275), 'N': (120.114, 3.424), 'NE2': (111.87, 1.691)},
    'GLU': {'C': (176.921, 1.868), 'CA': (57.292, 2.048), 'CB': (29.947, 1.663), 'CD': (182.193, 2.278), 'CG': (36.107, 1.181), 'H': (8.331, 0.569), 'HA': (4.237, 0.397), 'HB2': (2.021, 0.203), 'HB3': (1.997, 0.208), 'HE2': (4.132, 1.447), 'HG2': (2.268, 0.204), 'HG3': (2.25, 0.208), 'N': (120.814, 3.383)},
    'GLY': {'C': (173.898, 1.77), 'CA': (45.345, 1.275), 'H': (8.328, 0.608), 'H1': (8.525, 0.215), 'HA2': (3.959, 0.362), 'HA3': (3.894, 0.36), 'N': (109.551, 3.504)},
    'HIS': {'C': (175.241, 1.932), 'CA': (56.459, 2.284), 'CB': (30.256, 2.07), 'CD2': (120.288, 3.246), 'CE1': (137.583, 2.313), 'CG': (132.256, 2.948), 'H': (8.245, 0.662), 'HA': (4.599, 0.422), 'HB2': (3.104, 0.348), 'HB3': (3.048, 0.37), 'HD1': (8.813, 4.931), 'HD2': (7.0, 0.476), 'HE1': (7.949, 0.572), 'HE2': (9.603, 2.249), 'N': (119.745, 4.019), 'ND1': (192.694, 18.688), 'NE2': (183.316, 16.475)},
    'ILE': {'C': (175.961, 1.87), 'CA': (61.665, 2.671), 'CB': (38.533, 1.976), 'CD1': (13.385, 1.64), 'CG1': (27.752, 1.702), 'CG2': (17.516, 1.308), 'H': (8.255, 0.67), 'HA': (4.153, 0.543), 'HB': (1.788, 0.286), 'HG12': (1.274, 0.391), 'HG13': (1.203, 0.405), 'MD': (0.683, 0.279), 'MG': (0.777, 0.266), 'N': (121.407, 4.146)},
    'LEU': {'C': (177.101, 1.896), 'CA': (55.668, 2.089), 'CB': (42.204, 1.816), 'CD1': (24.614, 1.588), 'CD2': (24.098, 1.668), 'CG': (26.782, 1.074), 'H': (8.215, 0.626), 'H1': (8.204, 0.446), 'HA': (4.293, 0.449), 'HB2': (1.61, 0.336), 'HB3': (1.529, 0.358), 'HG': (1.511, 0.33), 'MD1': (0.754, 0.273), 'MD2': (0.735, 0.279), 'N': (121.863, 3.805)},
    'LYS': {'C': (176.721, 1.89), 'CA': (56.944, 2.143), 'CB': (32.737, 1.736), 'CD': (28.967, 1.091), 'CE': (41.894, 0.863), 'CG': (24.895, 1.103), 'H': (8.175, 0.584), 'HA': (4.253, 0.419), 'HB2': (1.779, 0.237), 'HB3': (1.753, 0.256), 'HD2': (1.607, 0.205), 'HD3': (1.6, 0.209), 'HE2': (2.914, 0.193), 'HE3': (2.908, 0.197), 'HG2': (1.369, 0.247), 'HG3': (1.355, 0.261), 'N': (121.132, 3.659), 'NZ': (33.174, 2.642), 'QZ': (7.417, 0.603)},
    'MET': {'C': (176.248, 2.022), 'CA': (56.123, 2.177), 'CB': (32.911, 2.129), 'CE': (17.113, 1.688), 'CG': (32.031, 1.302), 'H': (8.252, 0.568), 'HA': (4.388, 0.453), 'HB2': (2.029, 0.32), 'HB3': (1.995, 0.332), 'HG2': (2.42, 0.355), 'HG3': (2.396, 0.37), 'ME': (1.889, 0.374), 'N': (120.171, 3.413)},
    'PHE': {'C': (175.481, 1.938), 'CA': (58.107, 2.532), 'CB': (39.863, 2.025), 'CD1': (131.574, 1.257), 'CD2': (131.568, 1.221), 'CE1': (130.717, 1.334), 'CE2': (130.755, 1.229), 'CG': (138.244, 2.787), 'CZ': (129.238, 1.531), 'H': (8.329, 0.709), 'HA': (4.603, 0.553), 'HB2': (3.001, 0.359), 'HB3': (2.942, 0.38), 'HD1': (7.056, 0.309), 'HD2': (7.06, 0.31), 'HE1': (7.086, 0.306), 'HE2': (7.085, 0.31), 'HZ': (6.997, 0.413), 'N': (120.371, 4.069)},
    'PRO': {'C': (176.775, 1.433), 'CA': (63.331, 1.476), 'CB': (31.835, 1.142), 'CD': (50.34, 1.045), 'CG': (27.198, 1.109), 'H': (8.519, 0.473), 'HA': (4.386, 0.32), 'HB2': (2.079, 0.339), 'HB3': (2.003, 0.348), 'HD2': (3.651, 0.344), 'HD3': (3.619, 0.369), 'HG2': (1.927, 0.301), 'HG3': (1.906, 0.312), 'N': (135.63, 5.146)},
    'SER': {'C': (174.627, 1.669), 'CA': (58.67, 2.017), 'CB': (63.789, 1.503), 'H': (8.277, 0.554), 'HA': (4.466, 0.392), 'HB2': (3.87, 0.249), 'HB3': (3.847, 0.267), 'HG': (5.336, 1.049), 'N': (116.336, 3.369)},
    'THR': {'C': (174.546, 1.686), 'CA': (62.201, 2.537), 'CB': (69.698, 1.67), 'CG2': (21.543, 1.102), 'H': (8.225, 0.603), 'HA': (4.446, 0.466), 'HB': (4.166, 0.32), 'HG1': (5.048, 1.34), 'MG': (1.138, 0.21), 'N': (115.348, 4.662)},
    'TRP': {'C': (176.234, 1.975), 'CA': (57.735, 2.486), 'CB': (29.892, 1.981), 'CD1': (126.543, 1.905), 'CD2': (127.329, 8.942), 'CE2': (137.658, 6.58), 'CE3': (120.481, 1.868), 'CG': (110.981, 1.894), 'CH2': (123.789, 1.579), 'CZ2': (114.271, 1.42), 'CZ3': (121.364, 1.612), 'H': (8.258, 0.768), 'HA': (4.651, 0.51), 'HB2': (3.186, 0.334), 'HB3': (3.122, 0.362), 'HD1': (7.137, 0.334), 'HE1': (10.09, 0.632), 'HE3': (7.322, 0.414), 'HH2': (6.985, 0.349), 'HZ2': (7.289, 0.312), 'HZ3': (6.881, 0.364), 'N': (121.543, 4.062), 'NE1': (129.271, 2.115)},
    'TYR': {'C': (175.538, 1.932), 'CA': (58.159, 2.469), 'CB': (39.223, 2.112), 'CD1': (132.719, 1.395), 'CD2': (132.713, 1.518), 'CE1': (117.943, 1.322), 'CE2': (117.922, 1.238), 'CG': (129.656, 3.275), 'CZ': (156.899, 2.407), 'H': (8.274, 0.714), 'HA': (4.597, 0.542), 'HB2': (2.908, 0.36), 'HB3': (2.846, 0.38), 'HD1': (6.939, 0.289), 'HD2': (6.936, 0.289), 'HE1': (6.701, 0.222), 'HE2': (6.703, 0.223), 'HH': (9.088, 1.571), 'N': (120.432, 4.046)},
    'VAL': {'C': (175.713, 1.838), 'CA': (62.521, 2.813), 'CB': (32.669, 1.755), 'CG1': (21.483, 1.361), 'CG2': (21.299, 1.518), 'H': (8.266, 0.653), 'HA': (4.157, 0.564), 'HB': (1.985, 0.309), 'MG1': (0.822, 0.261), 'MG2': (0.805, 0.277), 'N': (121.099, 4.325)},
}

RC_BACKBONE = {  # random-coil centers, 3-letter AA -> {atom: ppm}
    'ALA': {'CA': 52.5, 'CB': 19.1, 'C': 177.8, 'N': 123.8, 'H': 8.24, 'HA': 4.35},
    'ARG': {'CA': 56.0, 'CB': 30.7, 'C': 176.3, 'N': 120.5, 'H': 8.27, 'HA': 4.38},
    'ASN': {'CA': 53.1, 'CB': 38.7, 'C': 175.2, 'N': 118.7, 'H': 8.4, 'HA': 4.75},
    'ASP': {'CA': 54.2, 'CB': 41.1, 'C': 176.3, 'N': 120.4, 'H': 8.41, 'HA': 4.64},
    'CYS': {'CA': 58.2, 'CB': 28.0, 'C': 174.6, 'N': 118.8, 'H': 8.32, 'HA': 4.56},
    'GLN': {'CA': 55.7, 'CB': 29.4, 'C': 176.0, 'N': 119.8, 'H': 8.32, 'HA': 4.37},
    'GLU': {'CA': 56.6, 'CB': 30.3, 'C': 176.6, 'N': 120.2, 'H': 8.36, 'HA': 4.29},
    'GLY': {'CA': 45.1, 'C': 174.9, 'N': 108.8, 'H': 8.33, 'HA': 3.97},
    'HIS': {'CA': 55.0, 'CB': 29.0, 'C': 174.1, 'N': 118.2, 'H': 8.42, 'HA': 4.63},
    'ILE': {'CA': 61.1, 'CB': 38.8, 'C': 176.4, 'N': 119.9, 'H': 8.0, 'HA': 4.17},
    'LEU': {'CA': 55.1, 'CB': 42.4, 'C': 177.6, 'N': 121.8, 'H': 8.16, 'HA': 4.38},
    'LYS': {'CA': 56.5, 'CB': 33.1, 'C': 176.6, 'N': 120.4, 'H': 8.29, 'HA': 4.36},
    'MET': {'CA': 55.4, 'CB': 32.9, 'C': 176.3, 'N': 119.6, 'H': 8.28, 'HA': 4.52},
    'PHE': {'CA': 57.7, 'CB': 39.6, 'C': 175.8, 'N': 120.3, 'H': 8.3, 'HA': 4.66},
    'PRO': {'CA': 63.3, 'CB': 32.1, 'C': 177.3, 'N': 137.4, 'HA': 4.44},
    'SER': {'CA': 58.3, 'CB': 63.8, 'C': 174.6, 'N': 115.7, 'H': 8.31, 'HA': 4.5},
    'THR': {'CA': 61.8, 'CB': 69.8, 'C': 174.7, 'N': 113.6, 'H': 8.15, 'HA': 4.35},
    'TRP': {'CA': 57.5, 'CB': 29.6, 'C': 176.1, 'N': 121.3, 'H': 8.25, 'HA': 4.7},
    'TYR': {'CA': 57.9, 'CB': 38.8, 'C': 175.9, 'N': 120.3, 'H': 8.12, 'HA': 4.6},
    'VAL': {'CA': 62.2, 'CB': 32.9, 'C': 176.3, 'N': 119.2, 'H': 8.03, 'HA': 4.18},
}


# individual methyl proton -> ordered candidate BMRB methyl pseudo-atoms (per-AA resolver)
_METHYL = {
    'HB1':['MB'],'HB2':['MB'],'HB3':['MB'],            # ALA only (ALA HB* == MB)
    'HD11':['MD1','MD'],'HD12':['MD1','MD'],'HD13':['MD1','MD'],
    'HD21':['MD2','MD'],'HD22':['MD2','MD'],'HD23':['MD2','MD'],
    'HG11':['MG1','MG'],'HG12':['MG1','MG'],'HG13':['MG1','MG'],
    'HG21':['MG2','MG'],'HG22':['MG2','MG'],'HG23':['MG2','MG'],
    'HE1':['ME'],'HE2':['ME'],'HE3':['ME'],   # MET epsilon-methyl (direct HE1/HE2 win for aromatics)
}

def _atom_from_col(shift_col):
    return shift_col.replace('_shift','').upper()

def get_scale(aa3, shift_col):
    """Fixed per-(AA,atom) sigma (ppm). Returns None if no grounded value exists."""
    atom=_atom_from_col(shift_col); d=BMRB_FILT.get(aa3,{})
    if atom in d: return d[atom][1]
    for m in _METHYL.get(atom,[]):
        if m in d: return d[m][1]
    return None

def get_center(aa3, shift_col):
    """Fixed per-(AA,atom) center (ppm): random-coil for backbone, BMRB mean otherwise."""
    atom=_atom_from_col(shift_col)
    if aa3 in RC_BACKBONE and atom in RC_BACKBONE[aa3]:
        return RC_BACKBONE[aa3][atom]
    d=BMRB_FILT.get(aa3,{})
    if atom in d: return d[atom][0]
    for m in _METHYL.get(atom,[]):
        if m in d: return d[m][0]
    return None


# ---------------------------------------------------------------------------
# Integration helpers: feed FIXED constants through the existing per-AA machinery
# ---------------------------------------------------------------------------
import copy as _copy


def build_fixed_per_aa(stats, shift_cols):
    """Return a stats dict whose 'per_aa' block is the FIXED (center, scale)
    constants instead of data-derived mean/std. Global per-column entries and
    'dssp' are kept verbatim (used only to recover raw ppm from the cache's
    global-z storage -- exact, no leakage). Chemically-absent (AA,col) combos
    (masked out in training) fall back to the global column stat."""
    new = _copy.deepcopy(stats) if stats else {}
    per_aa = {}
    for aa3 in BMRB_FILT:  # the 20 standard residues
        d = {}
        for col in shift_cols:
            c = get_center(aa3, col)
            s = get_scale(aa3, col)
            if c is None or s is None:
                g = (stats or {}).get(col)
                if g:
                    c = g['mean'] if c is None else c
                    s = g['std'] if s is None else s
                else:
                    c = 0.0 if c is None else c
                    s = 1.0 if s is None else s
            d[col] = {'mean': float(c), 'std': float(max(s, 0.1))}
        per_aa[aa3] = d
    new['per_aa'] = per_aa
    new['normalization'] = 'fixed_rc_bmrb_2026-06-08'  # provenance marker
    return new


def build_loss_tensors(shift_cols, standard_residues):
    """(group_weight, partner) torch tensors of shape (n_aa, n_shifts), indexed by
    residue-type code. group_weight = 1/N inside an equivalence group else 1.0;
    partner[a,i] = column index of i's prochiral partner for residue a, else -1."""
    import torch
    import equiv_groups as eg
    gw_map, pairs_map = eg.build_loss_structure(shift_cols)
    n_aa = len(standard_residues); S = len(shift_cols)
    group_weight = torch.ones(n_aa, S, dtype=torch.float32)
    partner = torch.full((n_aa, S), -1, dtype=torch.long)
    for a, aa3 in enumerate(standard_residues):
        if aa3 in gw_map:
            group_weight[a] = torch.from_numpy(gw_map[aa3])
        for (i, j) in pairs_map.get(aa3, []):
            partner[a, i] = j
            partner[a, j] = i
    return group_weight, partner
