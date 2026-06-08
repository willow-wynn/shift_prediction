"""Per-residue magnetic-equivalence groups and prochiral pairs over the 49 shift
columns, for an equivalence-aware, swap-invariant loss.

- EQUIV_GROUPS[res3] = list of column-tuples that are MAGNETICALLY EQUIVALENT
  (one observable). Members are weighted 1/len in the loss so the group counts once.
  Covers methyls (3 H) and fast-ring-flip aromatic pairs (Phe/Tyr ortho & meta H and C).
- PROCHIRAL_PAIRS[res3] = list of column 2-tuples whose label assignment is arbitrary
  (diastereotopic methylenes, the two Val/Leu methyls, NH2 protons). Scored
  swap-invariant: min over the two assignments. Applied unconditionally.

Column names use the model's '<atom>_shift' convention (lowercase).
Chemistry is standard NMR; validated against shift_cols + cache population.
"""

def _s(*atoms):
    return tuple(f'{a.lower()}_shift' for a in atoms)

# Methyl (3 equivalent H) groups, symmetric-aromatic groups
EQUIV_GROUPS = {
    'ALA': [_s('HB1', 'HB2', 'HB3')],
    'VAL': [_s('HG11', 'HG12', 'HG13'), _s('HG21', 'HG22', 'HG23')],
    'LEU': [_s('HD11', 'HD12', 'HD13'), _s('HD21', 'HD22', 'HD23')],
    'ILE': [_s('HG21', 'HG22', 'HG23'), _s('HD11', 'HD12', 'HD13')],
    'THR': [_s('HG21', 'HG22', 'HG23')],
    'MET': [_s('HE1', 'HE2', 'HE3')],
    # fast ring flip -> ortho (D) and meta (E) protons & carbons equivalent
    'PHE': [_s('HD1', 'HD2'), _s('HE1', 'HE2'), _s('CD1', 'CD2'), _s('CE1', 'CE2')],
    'TYR': [_s('HD1', 'HD2'), _s('HE1', 'HE2'), _s('CD1', 'CD2'), _s('CE1', 'CE2')],
}

# Diastereotopic / arbitrarily-labelled pairs -> swap-invariant
PROCHIRAL_PAIRS = {
    'GLY': [_s('HA2', 'HA3')],
    'SER': [_s('HB2', 'HB3')],
    'CYS': [_s('HB2', 'HB3')],
    'ASP': [_s('HB2', 'HB3')],
    'ASN': [_s('HB2', 'HB3'), _s('HD21', 'HD22')],          # side-chain NH2
    'PHE': [_s('HB2', 'HB3')],
    'TYR': [_s('HB2', 'HB3')],
    'HIS': [_s('HB2', 'HB3')],
    'TRP': [_s('HB2', 'HB3')],
    'LEU': [_s('HB2', 'HB3'), _s('CD1', 'CD2')],  # CD1/CD2 (two delta methyls) swap-invariant
    'VAL': [_s('CG1', 'CG2')],                    # CG1/CG2 (two gamma methyls) swap-invariant
    'ILE': [_s('HG12', 'HG13')],                  # gamma1 CH2
    'ARG': [_s('HB2', 'HB3'), _s('HG2', 'HG3'), _s('HD2', 'HD3')],
    'LYS': [_s('HB2', 'HB3'), _s('HG2', 'HG3'), _s('HD2', 'HD3'), _s('HE2', 'HE3')],
    'PRO': [_s('HB2', 'HB3'), _s('HG2', 'HG3'), _s('HD2', 'HD3')],
    'GLN': [_s('HB2', 'HB3'), _s('HG2', 'HG3'), _s('HE21', 'HE22')],  # NE2 H2
    'GLU': [_s('HB2', 'HB3'), _s('HG2', 'HG3')],
    'MET': [_s('HB2', 'HB3'), _s('HG2', 'HG3')],
}


def build_loss_structure(shift_cols):
    """Return (group_weight, pairs) given the model's ordered shift_cols.

    group_weight[res3] -> np.array(len(shift_cols)) of per-column loss weight (1/N
      inside an equivalence group, else 1.0). Columns absent for a residue keep 1.0
      (they're masked out anyway).
    pairs[res3] -> list of (i,j) column-index pairs to score swap-invariant.
    """
    import numpy as np
    idx = {c: i for i, c in enumerate(shift_cols)}
    aas = set(list(EQUIV_GROUPS) + list(PROCHIRAL_PAIRS))
    group_weight, pairs = {}, {}
    for aa in aas:
        w = np.ones(len(shift_cols), dtype=np.float32)
        for grp in EQUIV_GROUPS.get(aa, []):
            present = [idx[c] for c in grp if c in idx]
            if len(present) >= 2:
                for i in present:
                    w[i] = 1.0 / len(present)
        group_weight[aa] = w
        pj = []
        for a, b in PROCHIRAL_PAIRS.get(aa, []):
            if a in idx and b in idx:
                pj.append((idx[a], idx[b]))
        pairs[aa] = pj
    return group_weight, pairs
