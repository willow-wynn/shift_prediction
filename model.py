#!/usr/bin/env python3
"""
Structure-only Chemical Shift Predictor.

Architecture (single structure-only pathway; the retrieval pathway was removed —
see legacy/ for the retired ESM+FAISS retrieval code):
  - DistanceAttentionPerPosition (intra-residue atom-pair distances → per-position)
  - CrossDistanceAttention: cross-residue (sidechain-aware) distances at the center,
    added as a per-AA-gated residual (cross_gate)
  - 5-layer residual CNN over the residue window → center extraction
  - SpatialNeighborAttention: K=5 spatial neighbors → 192-dim
  - struct_head: MLP → n_shifts
"""

import os
import sys

# Local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    N_RESIDUE_TYPES, N_SS_TYPES, N_MISMATCH_TYPES,
    STANDARD_RESIDUES, AA_3_TO_1, DSSP_COLS,
    DIST_ATTN_EMBED, DIST_ATTN_HIDDEN,
    CNN_CHANNELS, KERNEL_SIZE,
    INPUT_DROPOUT, LAYER_DROPOUTS, HEAD_DROPOUT,
    SPATIAL_ATTN_HIDDEN,
    MAX_VALID_DISTANCES,
    N_BOND_GEOM,
    MAX_CROSS_DISTANCES, N_CROSS_OFFSET_TYPES, CROSS_OFFSET_EMBED_DIM,
)


# ============================================================================
# Original Components (unchanged from base model)
# ============================================================================

class DistanceAttentionPerPosition(nn.Module):
    """Attention over intramolecular distances for a single residue position."""

    def __init__(self, n_atom_types, embed_dim=32, hidden_dim=256, dropout=0.25):
        super().__init__()

        self.n_atom_types = n_atom_types
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.atom_embed = nn.Embedding(n_atom_types + 1, embed_dim, padding_idx=n_atom_types)

        input_dim = embed_dim * 2 + 1
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.attn_score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1)
        )

        self.value_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.fallback_embed = nn.Parameter(torch.zeros(hidden_dim))
        self.out_dim = hidden_dim

    def forward(self, atom1_idx, atom2_idx, distances, mask):
        B, W, D = distances.shape

        atom1_emb = self.atom_embed(atom1_idx)
        atom2_emb = self.atom_embed(atom2_idx)

        distances = distances * mask.float()

        combined = torch.cat([
            atom1_emb,
            atom2_emb,
            distances.unsqueeze(-1)
        ], dim=-1)

        hidden = self.input_proj(combined)

        scores = self.attn_score(hidden).squeeze(-1)

        any_valid = mask.any(dim=2)

        scores = scores.masked_fill(~mask, -1e4)
        attn_weights = F.softmax(scores, dim=-1)

        values = self.value_proj(hidden)
        output = (values * attn_weights.unsqueeze(-1)).sum(dim=2)

        fallback = self.fallback_embed.view(1, 1, -1).expand(B, W, -1)
        output = torch.where(any_valid.unsqueeze(-1), output, fallback)

        return output


class CrossDistanceAttention(nn.Module):
    """Attention over CROSS-residue atom-pair distances at the center position.

    Identical structure to DistanceAttentionPerPosition except:
      - Each pair carries an offset code (which neighbor residue the partner
        atom belongs to: ±5 sequence window or one of 5 spatial neighbors).
        Encoded as a small embedding concatenated alongside atom embeddings.
      - Optionally shares the atom_embed weights with the intra module so
        the meaning of an atom type is consistent across both pathways.
      - Operates on a single position per sample (B, 1, M_CR), since only
        the center residue gets cross-features in Phase 1.
    """

    def __init__(self, n_atom_types, n_offset_types,
                 atom_embed_dim=32, offset_embed_dim=8,
                 hidden_dim=256, dropout=0.25,
                 shared_atom_embed: nn.Embedding = None):
        super().__init__()

        self.n_atom_types = n_atom_types
        self.n_offset_types = n_offset_types
        self.atom_embed_dim = atom_embed_dim
        self.offset_embed_dim = offset_embed_dim
        self.hidden_dim = hidden_dim

        if shared_atom_embed is not None:
            self.atom_embed = shared_atom_embed
        else:
            self.atom_embed = nn.Embedding(
                n_atom_types + 1, atom_embed_dim, padding_idx=n_atom_types)
        self.offset_embed = nn.Embedding(
            n_offset_types + 1, offset_embed_dim, padding_idx=n_offset_types)
        # Initialize offset embedding to zero so a freshly-introduced cross
        # pathway makes no contribution at start of training (lets the model
        # rediscover the cross signal incrementally).
        with torch.no_grad():
            self.offset_embed.weight.zero_()

        input_dim = atom_embed_dim * 2 + offset_embed_dim + 1
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attn_score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
        )
        self.value_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.fallback_embed = nn.Parameter(torch.zeros(hidden_dim))
        self.out_dim = hidden_dim

    def forward(self, atom1_idx, atom2_idx, offset_idx, distances, mask):
        """Args all shaped (B, 1, M_CR). Returns (B, 1, hidden_dim)."""
        B, P, D = distances.shape

        atom1_emb = self.atom_embed(atom1_idx)        # (B, 1, M_CR, atom_dim)
        atom2_emb = self.atom_embed(atom2_idx)
        off_emb = self.offset_embed(offset_idx)        # (B, 1, M_CR, off_dim)

        distances = distances * mask.float()
        combined = torch.cat([
            atom1_emb, atom2_emb, off_emb, distances.unsqueeze(-1)
        ], dim=-1)

        hidden = self.input_proj(combined)
        scores = self.attn_score(hidden).squeeze(-1)
        any_valid = mask.any(dim=2)
        scores = scores.masked_fill(~mask, -1e4)
        attn_weights = F.softmax(scores, dim=-1)
        values = self.value_proj(hidden)
        output = (values * attn_weights.unsqueeze(-1)).sum(dim=2)
        fallback = self.fallback_embed.view(1, 1, -1).expand(B, P, -1)
        output = torch.where(any_valid.unsqueeze(-1), output, fallback)
        return output


class SpatialNeighborAttention(nn.Module):
    """Attention over k spatially proximate residues.

    Each neighbor contributes:
    - Residue type embedding
    - Secondary structure embedding
    - Continuous features (CA distance, sequence separation, phi/psi angles)
    - Structural embedding from distance features (via a shared DistanceAttentionPerPosition)
    """

    def __init__(self, n_residue_types, n_ss_types, k_neighbors=5,
                 embed_dim=64, hidden_dim=256, dropout=0.30,
                 dist_attn_hidden=None, query_dim=None, n_heads=4):
        super().__init__()

        self.k = k_neighbors
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.residue_embed = nn.Embedding(n_residue_types + 1, embed_dim)
        self.ss_embed = nn.Embedding(n_ss_types + 1, embed_dim // 2)

        self.continuous_proj = nn.Linear(6, embed_dim // 2)

        # dist_attn_hidden is the output dim of the distance attention module
        # It will be projected down to embed_dim for combining
        self.dist_attn_hidden = dist_attn_hidden
        if dist_attn_hidden is not None and dist_attn_hidden > 0:
            self.dist_proj = nn.Linear(dist_attn_hidden, embed_dim // 2)
            combined_dim = embed_dim + embed_dim // 2 + embed_dim // 2 + embed_dim // 2
        else:
            self.dist_proj = None
            combined_dim = embed_dim + embed_dim // 2 + embed_dim // 2

        self.q_proj = nn.Linear(query_dim, hidden_dim)
        self.k_proj = nn.Linear(combined_dim, hidden_dim)
        self.v_proj = nn.Linear(combined_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.ablate = set()  # feature-ablation flags (set externally before compile)
        self.attn_drop = nn.Dropout(dropout)

        self.fallback_embed = nn.Parameter(torch.zeros(hidden_dim))
        self.out_dim = hidden_dim

    def forward(self, neighbor_res_idx, neighbor_ss_idx,
                neighbor_dist, neighbor_seq_sep, neighbor_angles,
                neighbor_valid,
                query_encoding=None,
                neighbor_dist_embeddings=None):
        """
        Args:
            neighbor_res_idx: (B, K) residue type indices
            neighbor_ss_idx: (B, K) secondary structure indices
            neighbor_dist: (B, K) CA distances
            neighbor_seq_sep: (B, K) sequence separations
            neighbor_angles: (B, K, 4) sin/cos of phi/psi
            neighbor_valid: (B, K) validity mask
            query_encoding: (B, query_dim) center residue encoding from CNN
            neighbor_dist_embeddings: (B, K, dist_attn_hidden) structural embeddings from distance features
        """
        B = neighbor_res_idx.size(0)

        res_emb = self.residue_embed(neighbor_res_idx)
        ss_emb = self.ss_embed(neighbor_ss_idx)
        # Feature ablation (zero a feature group's contribution at train/eval time)
        if 'residue_type' in self.ablate:
            res_emb = res_emb * 0.0
        if 'ss' in self.ablate:
            ss_emb = ss_emb * 0.0

        neighbor_dist = torch.where(torch.isnan(neighbor_dist), torch.zeros_like(neighbor_dist), neighbor_dist)
        neighbor_angles = torch.where(torch.isnan(neighbor_angles), torch.zeros_like(neighbor_angles), neighbor_angles)

        dist_norm = neighbor_dist / 15.0
        seq_sep_norm = torch.log1p(neighbor_seq_sep.abs().float()) / 5.0

        continuous = torch.cat([
            dist_norm.unsqueeze(-1),
            seq_sep_norm.unsqueeze(-1),
            neighbor_angles
        ], dim=-1)
        cont_emb = self.continuous_proj(continuous)

        if self.dist_proj is not None and neighbor_dist_embeddings is not None:
            dist_emb_proj = self.dist_proj(neighbor_dist_embeddings)
            combined = torch.cat([res_emb, ss_emb, cont_emb, dist_emb_proj], dim=-1)
        else:
            combined = torch.cat([res_emb, ss_emb, cont_emb], dim=-1)

        any_valid = neighbor_valid.any(dim=1)

        # Multi-head cross-attention: center queries against neighbor keys/values
        H, HD = self.n_heads, self.head_dim
        K = combined.size(1)

        Q = self.q_proj(query_encoding).view(B, 1, H, HD).transpose(1, 2)  # (B, H, 1, HD)
        Kp = self.k_proj(combined).view(B, K, H, HD).transpose(1, 2)       # (B, H, K, HD)
        V = self.v_proj(combined).view(B, K, H, HD).transpose(1, 2)        # (B, H, K, HD)

        scores = torch.matmul(Q, Kp.transpose(-2, -1)) / (HD ** 0.5)      # (B, H, 1, K)
        scores = scores.masked_fill(~neighbor_valid.unsqueeze(1).unsqueeze(2), -1e4)
        attn_weights = self.attn_drop(F.softmax(scores, dim=-1))

        out = torch.matmul(attn_weights, V)                                # (B, H, 1, HD)
        out = out.transpose(1, 2).contiguous().view(B, H * HD)             # (B, hidden_dim)
        output = self.out_proj(out)

        fallback = self.fallback_embed.unsqueeze(0).expand(B, -1)
        output = torch.where(any_valid.unsqueeze(-1), output, fallback)

        return output


class ResidualBlock1D(nn.Module):
    """Residual block for 1D convolution."""

    def __init__(self, in_ch, out_ch, kernel=3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2)
        self.gn1 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=kernel // 2)
        self.gn2 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = F.gelu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return F.gelu(out + identity)


# ============================================================================
# Full Model (structure-only)
# ============================================================================


class ShiftPredictor(nn.Module):
    """Structure-only chemical shift prediction.

    Encoder: intra + cross-residue distance attention, CNN over the residue
    window, spatial-neighbor cross-attention; struct_head MLP → n_shifts.
    """

    def __init__(
        self,
        n_atom_types: int,
        n_residue_types: int = N_RESIDUE_TYPES,
        n_ss_types: int = N_SS_TYPES,
        n_mismatch_types: int = N_MISMATCH_TYPES,
        n_dssp: int = len(DSSP_COLS),
        n_shifts: int = 49,
        dist_attn_embed: int = DIST_ATTN_EMBED,
        dist_attn_hidden: int = DIST_ATTN_HIDDEN,
        cnn_channels: list = None,
        kernel: int = KERNEL_SIZE,
        input_dropout: float = INPUT_DROPOUT,
        layer_dropouts: list = None,
        head_dropout: float = HEAD_DROPOUT,
        spatial_hidden: int = SPATIAL_ATTN_HIDDEN,
        k_spatial: int = 5,
    ):
        super().__init__()

        cnn_channels = cnn_channels or list(CNN_CHANNELS)
        layer_dropouts = layer_dropouts or list(LAYER_DROPOUTS)

        self.n_shifts = n_shifts
        self.n_residue_types = n_residue_types

        # ========== Base encoder ==========
        self.distance_attention = DistanceAttentionPerPosition(
            n_atom_types=n_atom_types,
            embed_dim=dist_attn_embed,
            hidden_dim=dist_attn_hidden,
            dropout=0.25
        )

        # Cross-residue distance attention (Phase 1 sidechain-aware features).
        # Shares atom_embed weights with the intra distance attention so atom
        # type semantics stay consistent across both pathways. Output is
        # added as a residual at the center window position before the CNN.
        # Output is zero at init (offset_embed.weight starts at zero, see
        # CrossDistanceAttention.__init__) so old checkpoints behave
        # identically until cross-features are wired in via training.
        self.cross_distance_attention = CrossDistanceAttention(
            n_atom_types=n_atom_types,
            n_offset_types=N_CROSS_OFFSET_TYPES,
            atom_embed_dim=dist_attn_embed,
            offset_embed_dim=CROSS_OFFSET_EMBED_DIM,
            hidden_dim=dist_attn_hidden,
            dropout=0.25,
            shared_atom_embed=self.distance_attention.atom_embed,
        )
        # Per-amino-acid sigmoid gate for the cross-residue residual.
        # The model can downweight cross when it's noise (loop residues)
        # and upweight it when it matters (aromatic-rich, disulfide,
        # H-bond contexts). Init to logit=0 → gate=0.5 at start, so cross
        # contributes at half strength and the model tunes per AA.
        self.cross_gate = nn.Embedding(n_residue_types + 1, 1)
        with torch.no_grad():
            self.cross_gate.weight.zero_()

        self.residue_embed = nn.Embedding(n_residue_types + 1, 64)
        self.ss_embed = nn.Embedding(n_ss_types + 1, 32)
        self.mismatch_embed = nn.Embedding(n_mismatch_types + 1, 16)
        self.valid_embed = nn.Linear(1, 16)

        if n_dssp > 0:
            self.dssp_proj = nn.Linear(n_dssp, 32)
            dssp_dim = 32
        else:
            self.dssp_proj = None
            dssp_dim = 0

        # Inter-residue bond geometry (CA-CA distances, peptide bond lengths)
        self.bond_proj = nn.Linear(N_BOND_GEOM, 16)
        self.ablate = set()  # feature-ablation flags (set externally before compile)
        bond_dim = 16

        cnn_input_dim = dist_attn_hidden + 64 + 32 + 16 + 16 + dssp_dim + bond_dim

        self.input_norm = nn.LayerNorm(cnn_input_dim)
        self.input_dropout_layer = nn.Dropout(input_dropout)

        cnn_layers = []
        in_ch = cnn_input_dim
        for out_ch, drop_p in zip(cnn_channels, layer_dropouts):
            cnn_layers.append(ResidualBlock1D(in_ch, out_ch, kernel))
            cnn_layers.append(nn.Dropout(drop_p))
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)
        cnn_out_dim = cnn_channels[-1]


        self.spatial_attention = SpatialNeighborAttention(
            n_residue_types=n_residue_types,
            n_ss_types=n_ss_types,
            k_neighbors=k_spatial,
            embed_dim=64,
            hidden_dim=spatial_hidden,
            dropout=0.30,
            dist_attn_hidden=dist_attn_hidden,
            query_dim=cnn_out_dim,
        )

        base_encoder_dim = cnn_out_dim + spatial_hidden
        self._base_encoder_dim = base_encoder_dim

        # ========== Prediction head ==========
        self.struct_head = nn.Sequential(
            nn.Linear(base_encoder_dim, 1024), nn.GELU(), nn.Dropout(head_dropout),
            nn.Linear(1024, 512), nn.GELU(), nn.Dropout(head_dropout),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(head_dropout * 0.5),
            nn.Linear(256, n_shifts))

        # Initialize output layers to zero
        with torch.no_grad():
            self.struct_head[-1].weight.zero_()
            self.struct_head[-1].bias.zero_()

    def forward(
        self,
        # Structural inputs
        atom1_idx, atom2_idx, distances, dist_mask,
        residue_idx, ss_idx, mismatch_idx, is_valid,
        dssp_features,
        neighbor_res_idx, neighbor_ss_idx,
        neighbor_dist, neighbor_seq_sep, neighbor_angles,
        neighbor_valid,
        # Spatial neighbor distance features
        neighbor_atom1_idx=None, neighbor_atom2_idx=None,
        neighbor_distances=None, neighbor_dist_mask=None,
        # Cross-residue distance features (Phase 1: center residue only)
        cross_atom1_idx=None, cross_atom2_idx=None, cross_offset_idx=None,
        cross_distances=None, cross_dist_mask=None,
        # Inter-residue bond geometry
        bond_geom=None,
        # Deprecated / legacy keys accepted but ignored (old dataset compatibility)
        physics_features=None,
        **kwargs,
    ):
        B = distances.size(0)
        _abl = self.ablate  # feature-ablation flags (empty set = no ablation)

        # ========== Base encoder ==========
        dist_emb = self.distance_attention(atom1_idx, atom2_idx, distances, dist_mask)
        if 'intra' in _abl:
            dist_emb = dist_emb * 0.0

        # Cross-residue features at the center position (Phase 1).
        # When cross_* tensors are absent (old caches) or all-masked, the
        # CrossDistanceAttention falls back to its (zero-initialized) fallback
        # embedding so this is identity for legacy checkpoints/caches.
        if cross_atom1_idx is not None and cross_distances is not None and 'cross' not in _abl:
            cross_a1 = cross_atom1_idx.unsqueeze(1)         # (B, 1, M_CR)
            cross_a2 = cross_atom2_idx.unsqueeze(1)
            cross_off = cross_offset_idx.unsqueeze(1)
            cross_d = cross_distances.unsqueeze(1)
            cross_m = cross_dist_mask.unsqueeze(1)
            cross_emb = self.cross_distance_attention(
                cross_a1, cross_a2, cross_off, cross_d, cross_m
            ).squeeze(1)                                     # (B, hidden_dim)
            W = dist_emb.size(1)
            center_w = W // 2
            # Per-AA sigmoid gate on the cross residual
            center_aa = residue_idx[:, center_w]              # (B,) long
            gate = torch.sigmoid(self.cross_gate(center_aa)).squeeze(-1)  # (B,)
            mask_w = torch.zeros(W, device=dist_emb.device)
            mask_w[center_w] = 1.0
            dist_emb = dist_emb + (
                gate.view(B, 1, 1)
                * cross_emb.unsqueeze(1)
                * mask_w.view(1, -1, 1)
            )

        res_emb = self.residue_embed(residue_idx)
        ss_emb = self.ss_embed(ss_idx)
        mismatch_emb = self.mismatch_embed(mismatch_idx)
        valid_emb = self.valid_embed(is_valid.unsqueeze(-1))
        if 'residue_type' in _abl:
            res_emb = res_emb * 0.0
        if 'ss' in _abl:
            ss_emb = ss_emb * 0.0

        # Bond geometry projection
        if bond_geom is None:
            bond_geom = torch.zeros(B, distances.size(1), N_BOND_GEOM, device=distances.device)
        bond_emb = self.bond_proj(bond_geom)
        if 'bond_geom' in _abl:
            bond_emb = bond_emb * 0.0

        if self.dssp_proj is not None:
            dssp_emb = self.dssp_proj(dssp_features)
            if 'dssp' in _abl:
                dssp_emb = dssp_emb * 0.0
            window_feat = torch.cat([dist_emb, res_emb, ss_emb, mismatch_emb, valid_emb, dssp_emb, bond_emb], dim=-1)
        else:
            window_feat = torch.cat([dist_emb, res_emb, ss_emb, mismatch_emb, valid_emb, bond_emb], dim=-1)

        x = self.input_dropout_layer(self.input_norm(window_feat))

        # CNN over window
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)

        center_idx = x.size(1) // 2
        x_center = x[:, center_idx, :]  # (B, cnn_out_dim)

        # Spatial neighbor distance embeddings
        # For each neighbor, combine query's intra-residue distances + neighbor's
        # intra-residue distances + the CA-CA cross-residue distance into one
        # attention set. This lets the model learn inter-residue geometry by
        # jointly attending over both residues' distance features.
        neighbor_dist_embeddings = None
        if neighbor_atom1_idx is not None and neighbor_distances is not None:
            K_sp = neighbor_atom1_idx.size(1)
            M_sp = neighbor_atom1_idx.size(2)

            # Query center distances: (B, M) -> expand to (B, K_sp, M)
            center_w = distances.size(1) // 2
            q_a1 = atom1_idx[:, center_w, :].unsqueeze(1).expand(-1, K_sp, -1)
            q_a2 = atom2_idx[:, center_w, :].unsqueeze(1).expand(-1, K_sp, -1)
            q_d = distances[:, center_w, :].unsqueeze(1).expand(-1, K_sp, -1)
            q_m = dist_mask[:, center_w, :].unsqueeze(1).expand(-1, K_sp, -1)

            # CA-CA distance as a single "distance pair" with special atom indices
            # Use CA atom index (index 1 in canonical vocab) for both
            ca_idx = 1  # CA in ATOM_TYPES
            ca_a1 = torch.full((B, K_sp, 1), ca_idx, dtype=torch.long, device=distances.device)
            ca_a2 = torch.full((B, K_sp, 1), ca_idx, dtype=torch.long, device=distances.device)
            ca_d = neighbor_dist.unsqueeze(-1)  # (B, K_sp, 1)
            ca_m = neighbor_valid.unsqueeze(-1)  # (B, K_sp, 1)

            # Concatenate: query distances + CA-CA + neighbor distances
            # Total distance pairs per neighbor: M + 1 + M
            joint_a1 = torch.cat([q_a1, ca_a1, neighbor_atom1_idx], dim=-1)  # (B, K_sp, 2M+1)
            joint_a2 = torch.cat([q_a2, ca_a2, neighbor_atom2_idx], dim=-1)
            joint_d = torch.cat([q_d, ca_d, neighbor_distances], dim=-1)
            joint_m = torch.cat([q_m, ca_m, neighbor_dist_mask], dim=-1)

            # Reshape for distance attention: (B*K_sp, 1, 2M+1)
            D_joint = joint_a1.size(-1)
            nb_a1 = joint_a1.reshape(B * K_sp, 1, D_joint)
            nb_a2 = joint_a2.reshape(B * K_sp, 1, D_joint)
            nb_d = joint_d.reshape(B * K_sp, 1, D_joint)
            nb_m = joint_m.reshape(B * K_sp, 1, D_joint)

            nb_dist_emb = self.distance_attention(nb_a1, nb_a2, nb_d, nb_m)
            neighbor_dist_embeddings = nb_dist_emb.squeeze(1).view(B, K_sp, -1)

        x_spatial = self.spatial_attention(
            neighbor_res_idx, neighbor_ss_idx,
            neighbor_dist, neighbor_seq_sep, neighbor_angles,
            neighbor_valid,
            query_encoding=x_center,
            neighbor_dist_embeddings=neighbor_dist_embeddings,
        )
        if 'spatial' in _abl:
            x_spatial = x_spatial * 0.0

        base_encoding = torch.cat([x_center, x_spatial], dim=-1)

        # ========== Structure-only prediction ==========
        struct_pred = self.struct_head(base_encoding)  # (B, n_shifts)
        return struct_pred


# ============================================================================
# Factory function
# ============================================================================

def create_model(
    n_atom_types: int,
    n_shifts: int = 49,
    **kwargs,
) -> ShiftPredictor:
    """Create model with sensible defaults from config."""
    # Retrieval pathway removed; silently drop any legacy retrieval kwargs so
    # old callers/checkpoint-metadata don't break.
    for _k in ("use_retrieval", "retrieval_hidden", "retrieval_heads",
               "retrieval_dropout", "n_self_attn", "n_cross_attn",
               "k_retrieved", "n_struct"):
        kwargs.pop(_k, None)
    return ShiftPredictor(
        n_atom_types=n_atom_types,
        n_shifts=n_shifts,
        **kwargs,
    )


# Backward compatibility alias
ShiftPredictorWithRetrieval = ShiftPredictor
