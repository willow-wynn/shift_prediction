#!/usr/bin/env python3
"""
Physics-only chemical shift predictor prototype.

Predicts shifts per-ATOM (not per-residue) using only:
- Distances to all atoms within 5A
- Distances to all electronegative atoms (O, N, S) within 10A
- Distances to aromatic ring centroids within 10A
- Element type of each neighbor atom

No residue boundaries, no sequence, no retrieval.
Pure local 3D electronic environment → shift.

This is a proof of concept to see if local atomic geometry alone
can predict chemical shifts, especially N-shifts.
"""

import gc
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from pdb_utils import parse_pdb, run_dssp
from config import AA_3_TO_1, RESIDUE_TO_IDX, STANDARD_RESIDUES

# ============================================================================
# Element/atom type classification
# ============================================================================

ELEMENT_TYPES = ['C', 'N', 'O', 'S', 'H', 'OTHER']
ELEMENT_TO_IDX = {e: i for i, e in enumerate(ELEMENT_TYPES)}

# Aromatic residues and their ring atoms
AROMATIC_RINGS = {
    'PHE': [['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ']],
    'TYR': [['CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ']],
    'TRP': [['CG', 'CD1', 'CD2', 'NE1', 'CE2'], ['CE2', 'CE3', 'CZ2', 'CZ3', 'CH2', 'CD2']],
    'HIS': [['CG', 'ND1', 'CD2', 'CE1', 'NE2']],
}

# Backbone atoms whose shifts we want to predict
TARGET_ATOMS = {'N': 'n_shift', 'CA': 'ca_shift', 'CB': 'cb_shift',
                'C': 'c_shift', 'H': 'h_shift', 'HA': 'ha_shift'}

# Electronegative atoms
ELECTRONEG_ATOMS = {'O', 'OD1', 'OD2', 'OE1', 'OE2', 'OG', 'OG1', 'OH', 'OXT',
                    'N', 'ND1', 'ND2', 'NE', 'NE1', 'NE2', 'NH1', 'NH2', 'NZ',
                    'SD', 'SG'}


def get_element(atom_name):
    """Get element type from PDB atom name."""
    name = atom_name.strip()
    if not name:
        return 'OTHER'
    if name[0] == 'C':
        return 'C'
    if name[0] == 'N':
        return 'N'
    if name[0] == 'O':
        return 'O'
    if name[0] == 'S':
        return 'S'
    if name[0] == 'H' or (len(name) >= 2 and name[0].isdigit() and name[1] == 'H'):
        return 'H'
    return 'OTHER'


def compute_ring_centroid(atoms, ring_atom_names):
    """Compute centroid of an aromatic ring."""
    coords = []
    for name in ring_atom_names:
        if name in atoms and np.all(np.isfinite(atoms[name])):
            coords.append(atoms[name])
    if len(coords) < 3:
        return None
    return np.mean(coords, axis=0)


# ============================================================================
# Dataset: per-atom features from PDB
# ============================================================================

class AtomShiftDataset(Dataset):
    """Dataset of (atom_environment, shift) pairs extracted from PDB + CSV."""

    def __init__(self, samples, max_neighbors=64, max_electroneg=32, max_rings=8):
        self.samples = samples
        self.max_neighbors = max_neighbors
        self.max_electroneg = max_electroneg
        self.max_rings = max_rings

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        MN = self.max_neighbors
        ME = self.max_electroneg
        MR = self.max_rings

        # Nearby atoms within 5A: (element_idx, distance)
        nbr_elements = torch.full((MN,), len(ELEMENT_TYPES), dtype=torch.long)  # padding
        nbr_distances = torch.zeros(MN)
        nbr_valid = torch.zeros(MN, dtype=torch.bool)
        for i, (elem, dist) in enumerate(s['neighbors'][:MN]):
            nbr_elements[i] = ELEMENT_TO_IDX.get(elem, ELEMENT_TO_IDX['OTHER'])
            nbr_distances[i] = dist
            nbr_valid[i] = True

        # Electronegative atoms within 10A
        elec_elements = torch.full((ME,), len(ELEMENT_TYPES), dtype=torch.long)
        elec_distances = torch.zeros(ME)
        elec_valid = torch.zeros(ME, dtype=torch.bool)
        for i, (elem, dist) in enumerate(s['electroneg'][:ME]):
            elec_elements[i] = ELEMENT_TO_IDX.get(elem, ELEMENT_TO_IDX['OTHER'])
            elec_distances[i] = dist
            elec_valid[i] = True

        # Ring centroids within 10A
        ring_distances = torch.zeros(MR)
        ring_valid = torch.zeros(MR, dtype=torch.bool)
        for i, dist in enumerate(s['rings'][:MR]):
            ring_distances[i] = dist
            ring_valid[i] = True

        # Query atom element
        query_element = torch.tensor(ELEMENT_TO_IDX.get(s['element'], 0), dtype=torch.long)

        # Target shift (raw ppm)
        target = torch.tensor(s['shift_ppm'], dtype=torch.float32)

        return {
            'nbr_elements': nbr_elements,
            'nbr_distances': nbr_distances,
            'nbr_valid': nbr_valid,
            'elec_elements': elec_elements,
            'elec_distances': elec_distances,
            'elec_valid': elec_valid,
            'ring_distances': ring_distances,
            'ring_valid': ring_valid,
            'query_element': query_element,
            'target': target,
        }


# ============================================================================
# Model
# ============================================================================

class PhysicsShiftPredictor(nn.Module):
    """Predict chemical shift from local atomic environment."""

    def __init__(self, n_elements=6, embed_dim=32, hidden_dim=128, n_heads=4):
        super().__init__()

        self.element_embed = nn.Embedding(n_elements + 1, embed_dim, padding_idx=n_elements)

        # Encode each neighbor: element embedding + distance features
        # Use RBF distance encoding
        self.n_rbf = 16
        self.rbf_centers = nn.Parameter(torch.linspace(0.5, 10.0, self.n_rbf), requires_grad=False)
        self.rbf_width = 0.5

        neighbor_input = embed_dim + self.n_rbf
        self.neighbor_encoder = nn.Sequential(
            nn.Linear(neighbor_input, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Attention over neighbors
        self.query_proj = nn.Linear(embed_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_scale = hidden_dim ** -0.5

        # Electronegative attention (separate)
        self.elec_encoder = nn.Sequential(
            nn.Linear(neighbor_input, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.elec_attn_proj = nn.Linear(hidden_dim, hidden_dim)

        # Ring current features
        self.ring_encoder = nn.Sequential(
            nn.Linear(self.n_rbf, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 3 + embed_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def rbf_encode(self, distances):
        """Radial basis function encoding of distances."""
        d = distances.unsqueeze(-1)  # (..., 1)
        return torch.exp(-((d - self.rbf_centers) ** 2) / (2 * self.rbf_width ** 2))

    def forward(self, nbr_elements, nbr_distances, nbr_valid,
                elec_elements, elec_distances, elec_valid,
                ring_distances, ring_valid, query_element):
        B = query_element.shape[0]

        # Query
        q_emb = self.element_embed(query_element)  # (B, embed)

        # Encode nearby neighbors
        nbr_emb = self.element_embed(nbr_elements)  # (B, MN, embed)
        nbr_rbf = self.rbf_encode(nbr_distances)  # (B, MN, n_rbf)
        nbr_feat = self.neighbor_encoder(torch.cat([nbr_emb, nbr_rbf], dim=-1))  # (B, MN, hidden)

        # Attention: query attends to neighbors
        Q = self.query_proj(q_emb).unsqueeze(1)  # (B, 1, hidden)
        K = self.key_proj(nbr_feat)  # (B, MN, hidden)
        V = self.value_proj(nbr_feat)  # (B, MN, hidden)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.attn_scale  # (B, 1, MN)
        scores = scores.masked_fill(~nbr_valid.unsqueeze(1), -1e4)
        attn = F.softmax(scores, dim=-1)
        nbr_ctx = torch.matmul(attn, V).squeeze(1)  # (B, hidden)

        # Zero out if no valid neighbors
        any_nbr = nbr_valid.any(dim=1, keepdim=True).float()
        nbr_ctx = nbr_ctx * any_nbr

        # Encode electronegative atoms
        elec_emb = self.element_embed(elec_elements)
        elec_rbf = self.rbf_encode(elec_distances)
        elec_feat = self.elec_encoder(torch.cat([elec_emb, elec_rbf], dim=-1))

        # Simple attention pooling for electroneg
        elec_scores = self.elec_attn_proj(elec_feat).sum(dim=-1)  # (B, ME)
        elec_scores = elec_scores.masked_fill(~elec_valid, -1e4)
        elec_attn = F.softmax(elec_scores, dim=-1).unsqueeze(-1)
        elec_ctx = (elec_feat * elec_attn).sum(dim=1)  # (B, hidden)
        any_elec = elec_valid.any(dim=1, keepdim=True).float()
        elec_ctx = elec_ctx * any_elec

        # Ring current features
        ring_rbf = self.rbf_encode(ring_distances)  # (B, MR, n_rbf)
        ring_feat = self.ring_encoder(ring_rbf)  # (B, MR, hidden)
        ring_mask = ring_valid.unsqueeze(-1).float()
        ring_ctx = (ring_feat * ring_mask).sum(dim=1) / (ring_mask.sum(dim=1) + 1e-6)
        any_ring = ring_valid.any(dim=1, keepdim=True).float()
        ring_ctx = ring_ctx * any_ring

        # Combine and predict
        combined = torch.cat([q_emb, nbr_ctx, elec_ctx, ring_ctx], dim=-1)
        shift = self.head(combined).squeeze(-1)

        return shift


# ============================================================================
# Data extraction from PDB + ground truth
# ============================================================================

def extract_atom_samples(pdb_path, shifts_dict, chain_id=None):
    """Extract per-atom samples from a PDB file.

    Args:
        pdb_path: path to PDB
        shifts_dict: {residue_id: {atom_name: shift_ppm}} ground truth

    Returns:
        list of sample dicts
    """
    residues = parse_pdb(pdb_path, chain_id=chain_id)
    if not residues:
        return []

    # Build global atom coordinate list
    all_atoms = []  # (chain, resid, atom_name, coord, element, residue_name)
    ring_centroids = []  # (coord,)

    for (chain, rid), info in residues.items():
        res_name = info['residue_name']
        for aname, coord in info['atoms'].items():
            if np.all(np.isfinite(coord)):
                elem = get_element(aname)
                all_atoms.append((chain, rid, aname, coord, elem, res_name))

        # Compute ring centroids
        if res_name in AROMATIC_RINGS:
            for ring_atoms in AROMATIC_RINGS[res_name]:
                centroid = compute_ring_centroid(info['atoms'], ring_atoms)
                if centroid is not None:
                    ring_centroids.append(centroid)

    if not all_atoms:
        return []

    # Vectorized distance computation
    from scipy.spatial.distance import cdist

    coords = np.array([a[3] for a in all_atoms])
    elements = np.array([a[4] for a in all_atoms])
    atom_names_arr = np.array([a[2] for a in all_atoms])
    atom_resids = np.array([a[1] for a in all_atoms])
    electroneg_mask = np.array([a[2] in ELECTRONEG_ATOMS for a in all_atoms])

    ring_coords = np.array(ring_centroids) if ring_centroids else np.zeros((0, 3))

    # Collect query atoms
    query_positions = []
    query_meta = []

    for (chain, rid), info in residues.items():
        res_name = info['residue_name']
        if rid not in shifts_dict:
            continue
        for target_atom, shift_col in TARGET_ATOMS.items():
            if target_atom not in info['atoms']:
                continue
            if shift_col not in shifts_dict[rid]:
                continue
            query_coord = info['atoms'][target_atom]
            if not np.all(np.isfinite(query_coord)):
                continue
            shift_ppm = shifts_dict[rid][shift_col]
            if np.isnan(shift_ppm):
                continue
            query_positions.append(query_coord)
            query_meta.append((chain, rid, target_atom, get_element(target_atom),
                              shift_ppm, shift_col, res_name))

    if not query_positions:
        return []

    query_coords = np.array(query_positions)

    # Full distance matrix: queries × all_atoms (vectorized, fast)
    D = cdist(query_coords, coords)  # (n_queries, n_atoms)

    # Ring distances if any
    ring_D = cdist(query_coords, ring_coords) if len(ring_coords) > 0 else None

    samples = []
    for qi, (chain, rid, target_atom, element, shift_ppm, shift_col, res_name) in enumerate(query_meta):
        dists = D[qi]

        # Self-exclusion mask
        self_mask = (atom_resids == rid) & (atom_names_arr == target_atom)

        # 5A neighbors (excluding self)
        mask5 = (dists < 5.0) & ~self_mask
        idx5 = np.where(mask5)[0]
        if len(idx5) > 0:
            order = np.argsort(dists[idx5])
            neighbors = [(elements[idx5[j]], float(dists[idx5[j]])) for j in order]
        else:
            neighbors = []

        # 10A electronegative (excluding self)
        mask10e = (dists < 10.0) & electroneg_mask & ~self_mask
        idx10e = np.where(mask10e)[0]
        if len(idx10e) > 0:
            order = np.argsort(dists[idx10e])
            electroneg = [(elements[idx10e[j]], float(dists[idx10e[j]])) for j in order]
        else:
            electroneg = []

        # Ring centroids within 10A
        if ring_D is not None:
            ring_dists = ring_D[qi]
            ring_mask = ring_dists < 10.0
            rings = sorted(ring_dists[ring_mask].tolist())
        else:
            rings = []

        samples.append({
            'element': element,
            'shift_ppm': shift_ppm,
            'shift_col': shift_col,
            'neighbors': neighbors,
            'electroneg': electroneg,
            'rings': rings,
            'residue_name': res_name,
            'residue_id': rid,
            'atom_name': target_atom,
        })

    return samples


def build_dataset_from_csv(csv_path, pdb_dir, max_proteins=200):
    """Build atom-level dataset from a structure CSV and PDB files."""
    import pandas as pd

    shift_cols = list(TARGET_ATOMS.values())
    usecols = ['bmrb_id', 'residue_id'] + [c for c in shift_cols]
    df = pd.read_csv(csv_path, usecols=usecols,
                      dtype={'bmrb_id': str}, nrows=100000, low_memory=False)

    # Build BMRB -> PDB mapping from pairs.csv
    pairs_path = os.path.join(os.path.dirname(csv_path), 'pairs.csv')
    bmrb_to_pdb = {}
    if os.path.exists(pairs_path):
        pairs = pd.read_csv(pairs_path, dtype={'Entry_ID': str})
        for _, row in pairs.iterrows():
            bid = str(row['Entry_ID'])
            pdbs = str(row.get('pdb_ids', '')).split(',')
            bmrb_to_pdb[bid] = [p.strip().upper() for p in pdbs if p.strip()]
    print(f"  BMRB->PDB mappings: {len(bmrb_to_pdb)}")

    all_samples = []
    proteins_done = 0

    for bmrb_id, grp in tqdm(df.groupby('bmrb_id'), desc="Processing proteins"):
        if proteins_done >= max_proteins:
            break

        # Find PDB file via pairs mapping
        pdb_path = None
        pdb_ids = bmrb_to_pdb.get(str(bmrb_id), [])
        for pid in pdb_ids:
            candidate = os.path.join(pdb_dir, f'{pid}.pdb')
            if os.path.exists(candidate):
                pdb_path = candidate
                break

        if pdb_path is None:
            continue

        # Build shifts dict
        shifts = {}
        for _, row in grp.iterrows():
            rid = int(row['residue_id'])
            shifts[rid] = {}
            for col in shift_cols:
                if pd.notna(row.get(col)):
                    shifts[rid][col] = float(row[col])

        chain_id = None  # let parse_pdb take all chains

        try:
            samples = extract_atom_samples(pdb_path, shifts, chain_id=chain_id)
            all_samples.extend(samples)
            proteins_done += 1
        except Exception as e:
            continue

    return all_samples


# ============================================================================
# Training
# ============================================================================

def train_physics_model(samples, n_epochs=50, batch_size=256, lr=1e-3, device='cpu'):
    """Train the physics-only model."""

    # Split train/test (80/20 by sample index, shuffled)
    np.random.seed(42)
    indices = np.arange(len(samples))
    np.random.shuffle(indices)
    split = int(0.8 * len(indices))
    train_indices = set(indices[:split].tolist())

    train_samples = [samples[i] for i in range(len(samples)) if i in train_indices]
    test_samples = [samples[i] for i in range(len(samples)) if i not in train_indices]

    print(f"Train: {len(train_samples):,}, Test: {len(test_samples):,}")

    # Per-atom-type normalization
    atom_stats = {}
    for atom_name, shift_col in TARGET_ATOMS.items():
        vals = [s['shift_ppm'] for s in train_samples if s['shift_col'] == shift_col]
        if vals:
            atom_stats[shift_col] = {'mean': np.mean(vals), 'std': max(np.std(vals), 0.1)}

    # Normalize targets
    for s in train_samples + test_samples:
        col = s['shift_col']
        if col in atom_stats:
            s['shift_ppm'] = (s['shift_ppm'] - atom_stats[col]['mean']) / atom_stats[col]['std']

    train_ds = AtomShiftDataset(train_samples)
    test_ds = AtomShiftDataset(test_samples)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = PhysicsShiftPredictor(hidden_dim=128).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=lr*0.01)

    best_mae = float('inf')
    for epoch in range(1, n_epochs + 1):
        # Train
        model.train()
        train_loss = 0
        train_n = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            target = batch.pop('target')
            pred = model(**batch)
            loss = F.huber_loss(pred, target, delta=0.5)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(target)
            train_n += len(target)
        scheduler.step()

        # Eval every 5 epochs
        if epoch % 5 == 0 or epoch == n_epochs:
            model.eval()
            per_shift = defaultdict(list)
            with torch.no_grad():
                for batch in test_loader:
                    batch_dev = {k: v.to(device) for k, v in batch.items()}
                    target = batch_dev.pop('target')
                    pred = model(**batch_dev)

                    # Denormalize per sample
                    for i in range(len(pred)):
                        si = test_samples[sum(1 for _ in range(i))]  # approximate
                        # Just collect normalized errors — denormalize at summary
                        pass

                    # Simpler: denormalize all at once per shift type
                    pred_np = pred.cpu().numpy()
                    target_np = target.cpu().numpy()

            # Recompute with denormalization
            all_preds = []
            all_targets = []
            all_cols = []
            with torch.no_grad():
                offset = 0
                for batch in test_loader:
                    batch_dev = {k: v.to(device) for k, v in batch.items()}
                    target = batch_dev.pop('target')
                    pred = model(**batch_dev)
                    bs = len(pred)
                    for i in range(bs):
                        col = test_samples[offset + i]['shift_col']
                        if col in atom_stats:
                            p = pred[i].item() * atom_stats[col]['std'] + atom_stats[col]['mean']
                            t = target[i].item() * atom_stats[col]['std'] + atom_stats[col]['mean']
                            per_shift[col].append(abs(p - t))
                    offset += bs

            print(f"Epoch {epoch:>3}/{n_epochs}  loss={train_loss/train_n:.4f}", end="  ")
            for col in ['n_shift', 'ca_shift', 'cb_shift', 'c_shift', 'h_shift', 'ha_shift']:
                if col in per_shift:
                    mae = np.mean(per_shift[col])
                    name = col.replace('_shift', '').upper()
                    print(f"{name}={mae:.3f}", end="  ")
            overall = np.mean([np.mean(v) for v in per_shift.values()])
            print(f"ALL={overall:.3f}")

            if overall < best_mae:
                best_mae = overall

    print(f"\nBest overall MAE: {best_mae:.3f}")
    return model, atom_stats


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='data/structure_data_hybrid.csv')
    parser.add_argument('--pdb_dir', default='data/pdbs')
    parser.add_argument('--max_proteins', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()

    print("=" * 60)
    print("  PHYSICS-ONLY SHIFT PREDICTOR PROTOTYPE")
    print("=" * 60)
    print(f"  CSV: {args.csv}")
    print(f"  PDB dir: {args.pdb_dir}")
    print(f"  Max proteins: {args.max_proteins}")
    print(f"  Device: {args.device}")

    print("\nExtracting atom-level features from PDBs...")
    samples = build_dataset_from_csv(args.csv, args.pdb_dir, max_proteins=args.max_proteins)
    print(f"Total samples: {len(samples):,}")

    # Count per shift type
    from collections import Counter
    type_counts = Counter(s['shift_col'] for s in samples)
    for col, n in sorted(type_counts.items()):
        print(f"  {col}: {n:,}")

    print(f"\nTraining physics model...")
    model, stats = train_physics_model(
        samples, n_epochs=args.epochs, batch_size=args.batch_size,
        lr=1e-3, device=args.device)

    # Save
    out_dir = os.path.dirname(os.path.abspath(__file__))
    torch.save({
        'model_state_dict': model.state_dict(),
        'atom_stats': stats,
    }, os.path.join(out_dir, 'physics_prototype_model.pt'))
    print(f"Saved to {out_dir}/physics_prototype_model.pt")


if __name__ == '__main__':
    main()
