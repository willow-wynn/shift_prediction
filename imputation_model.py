1#!/usr/bin/env python3
"""
Chemical Shift Imputation Model with Structure + Observed Shifts + Retrieval.

Combines three information sources:
1. Structural encoding (distances, spatial neighbors, DSSP)
2. Observed shift context (neighboring residues' measured chemical shifts)
3. Retrieval transfer (FAISS-retrieved similar residues, shift-aware weighting)

Key design differences from model.py:
- Single unified retrieval mechanism (no twin cross-attention + transfer)
- Shift context as a first-class input stream via 1D CNN
- Shift-type conditioning: predicts one shift at a time (like src/train_imputation.py)
- Observed shifts fed into retrieval attention for shift-aware re-ranking
- No per-residue evolutionary features (redundant with residue embedding)
- No per-shift prediction heads (single output head conditioned on shift_type)

Reuses from model.py:
- DistanceAttentionPerPosition
- ResidualBlock1D
- SpatialNeighborAttention
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    N_RESIDUE_TYPES, N_SS_TYPES, N_MISMATCH_TYPES,
    STANDARD_RESIDUES, DSSP_COLS,
    DIST_ATTN_EMBED, DIST_ATTN_HIDDEN,
    KERNEL_SIZE,
    INPUT_DROPOUT, HEAD_DROPOUT,
    SPATIAL_ATTN_HIDDEN,
    MAX_VALID_DISTANCES,
    N_BOND_GEOM,
    MAX_CROSS_DISTANCES, N_CROSS_OFFSET_TYPES, CROSS_OFFSET_EMBED_DIM,
)
from model import (
    DistanceAttentionPerPosition,
    CrossDistanceAttention,
    ResidualBlock1D,
    SpatialNeighborAttention,
)


# ============================================================================
# Shift Context Encoder
# ============================================================================

class ShiftContextEncoder(nn.Module):
    """Encode observed chemical shifts from a local context window.

    Per-position features: [residue_embed, observed_shifts * mask, mask, position_encoding]
    Processed by a 1D CNN, then center-extracted.

    This is conceptually similar to the ShiftPredictor in src/train_imputation.py
    but smaller (serves as one input stream, not the whole model).
    """

    def __init__(
        self,
        n_shifts: int,
        n_residue_types: int = N_RESIDUE_TYPES,
        embed_dim: int = 48,
        channels: list = None,
        kernel: int = 3,
        dropout: float = 0.15,
    ):
        super().__init__()

        channels = channels or [128, 256, 384]

        self.n_shifts = n_shifts
        self.residue_embed = nn.Embedding(n_residue_types + 1, embed_dim)

        # Per-position input: residue_embed + shifts + mask + is_valid + is_center
        pos_input_dim = embed_dim + n_shifts + n_shifts + 1 + 1
        self.input_norm = nn.LayerNorm(pos_input_dim)
        self.input_dropout = nn.Dropout(dropout)

        layers = []
        in_ch = pos_input_dim
        for out_ch in channels:
            layers.append(ResidualBlock1D(in_ch, out_ch, kernel))
            layers.append(nn.Dropout(dropout))
            in_ch = out_ch
        self.cnn = nn.Sequential(*layers)

        self.out_dim = channels[-1]

    def forward(
        self,
        residue_idx: torch.Tensor,        # (B, W) residue type indices
        observed_shifts: torch.Tensor,     # (B, W, n_shifts) z-normalized
        shift_masks: torch.Tensor,         # (B, W, n_shifts) availability
        is_valid: torch.Tensor,            # (B, W) whether position exists
    ) -> torch.Tensor:
        """Returns (B, out_dim) shift context encoding."""
        B, W = residue_idx.shape

        res_emb = self.residue_embed(residue_idx)  # (B, W, embed_dim)

        # Build is_center indicator
        center = W // 2
        is_center = torch.zeros(B, W, 1, device=residue_idx.device)
        is_center[:, center, :] = 1.0

        x = torch.cat([
            res_emb,
            observed_shifts * shift_masks,  # zero out unavailable
            shift_masks,
            is_valid.unsqueeze(-1),
            is_center,
        ], dim=-1)  # (B, W, pos_input_dim)

        x = self.input_dropout(self.input_norm(x))
        x = x.transpose(1, 2)  # (B, C, W)
        x = self.cnn(x)
        x = x.transpose(1, 2)  # (B, W, C)

        # Extract center position
        x_center = x[:, center, :]  # (B, out_dim)
        return x_center


# ============================================================================
# Spatial Neighbor Shift Attention
# ============================================================================

class SpatialNeighborShiftAttention(nn.Module):
    """Multi-head cross-attention from center query to spatial neighbors' observed shifts.

    Complements ShiftContextEncoder. ShiftContextEncoder sees the ±W sequence
    window; this module sees the K spatially-closest residues' chemical shifts.
    Beta-sheet pairing partners, ring stacks, and disulfide contacts can be
    distant in sequence but close in space — their shifts are highly
    informative for imputing a target shift.

    Per-neighbor feature: residue_emb + observed_shifts*mask + mask + CA_dist
    Query: structural+context encoding at center (plus optional shift-type emb
    appended upstream by the caller).
    """

    def __init__(self, n_shifts, n_residue_types, query_dim,
                 hidden_dim=192, n_heads=4, dropout=0.25):
        super().__init__()
        assert hidden_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.residue_embed = nn.Embedding(n_residue_types + 1, 32)
        per_nbr_dim = 32 + n_shifts + n_shifts + 1
        self.neighbor_proj = nn.Sequential(
            nn.Linear(per_nbr_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.q_proj = nn.Linear(query_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.fallback = nn.Parameter(torch.zeros(hidden_dim))
        self.out_dim = hidden_dim

    def forward(self, query, neighbor_res_idx, neighbor_shifts,
                neighbor_shift_masks, neighbor_dist, neighbor_valid):
        """
        Args:
            query: (B, query_dim)
            neighbor_res_idx: (B, K) residue type indices at spatial neighbors
            neighbor_shifts: (B, K, n_shifts) z-normalized observed shifts
            neighbor_shift_masks: (B, K, n_shifts) 1.0 where shift is observed
            neighbor_dist: (B, K) CA-CA distance
            neighbor_valid: (B, K) which neighbor slots are filled
        Returns:
            (B, hidden_dim)
        """
        B, K = neighbor_res_idx.shape
        H, HD = self.n_heads, self.head_dim

        res_emb = self.residue_embed(neighbor_res_idx)
        shifts_z = neighbor_shifts * neighbor_shift_masks
        d_norm = (neighbor_dist / 15.0).unsqueeze(-1)
        feat = torch.cat([res_emb, shifts_z, neighbor_shift_masks, d_norm], dim=-1)
        kv = self.neighbor_proj(feat)

        Q = self.q_proj(query).view(B, 1, H, HD).transpose(1, 2)
        Kp = self.k_proj(kv).view(B, K, H, HD).transpose(1, 2)
        Vp = self.v_proj(kv).view(B, K, H, HD).transpose(1, 2)

        scores = torch.matmul(Q, Kp.transpose(-2, -1)) * self.scale
        scores = scores.masked_fill(~neighbor_valid.unsqueeze(1).unsqueeze(2), -1e4)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, Vp).transpose(1, 2).contiguous().view(B, H * HD)
        out = self.out_proj(out)

        any_valid = neighbor_valid.any(dim=1)
        fb = self.fallback.unsqueeze(0).expand(B, -1)
        return torch.where(any_valid.unsqueeze(-1), out, fb)


# ============================================================================
# Unified Retrieval Transfer
# ============================================================================

class ShiftImputationModel(nn.Module):
    """Chemical shift imputation using structure + observed shifts + retrieval.

    Architecture:
    1. Structural Encoder: DistanceAttention -> CNN -> center -> SpatialAttention
    2. Shift Context Encoder: 1D CNN over observed shifts in local window
    3. Unified Retrieval Transfer: cross-attention with shift-aware re-ranking
    4. Fusion -> shift-type-conditioned prediction head -> scalar output
    """

    def __init__(
        self,
        n_atom_types: int,
        n_shifts: int = 6,
        n_residue_types: int = N_RESIDUE_TYPES,
        n_ss_types: int = N_SS_TYPES,
        n_mismatch_types: int = N_MISMATCH_TYPES,
        n_dssp: int = len(DSSP_COLS),
        # Structural encoder
        dist_attn_embed: int = DIST_ATTN_EMBED,
        dist_attn_hidden: int = DIST_ATTN_HIDDEN,
        struct_cnn_channels: list = None,
        kernel: int = KERNEL_SIZE,
        input_dropout: float = INPUT_DROPOUT,
        struct_layer_dropouts: list = None,
        head_dropout: float = HEAD_DROPOUT,
        spatial_hidden: int = SPATIAL_ATTN_HIDDEN,
        k_spatial: int = 5,
        # Shift context encoder
        shift_context_channels: list = None,
        shift_context_dropout: float = 0.15,
        # Dropout for the spatial-neighbor-shift attention
        attn_dropout: float = 0.3,
    ):
        super().__init__()

        struct_cnn_channels = struct_cnn_channels or [256, 512, 768, 1024, 1280]
        struct_layer_dropouts = struct_layer_dropouts or [0.40, 0.40, 0.40, 0.40, 0.40]
        shift_context_channels = shift_context_channels or [128, 256, 384]

        self.n_shifts = n_shifts
        self.n_residue_types = n_residue_types

        # ========== Structural Encoder ==========
        self.distance_attention = DistanceAttentionPerPosition(
            n_atom_types=n_atom_types,
            embed_dim=dist_attn_embed,
            hidden_dim=dist_attn_hidden,
            dropout=0.25,
        )

        # Cross-residue distance attention (Phase 1 sidechain-aware features).
        # Shares atom_embed weights with the intra module. Output is added
        # as a per-AA-gated residual to the center window position before CNN.
        self.cross_distance_attention = CrossDistanceAttention(
            n_atom_types=n_atom_types,
            n_offset_types=N_CROSS_OFFSET_TYPES,
            atom_embed_dim=dist_attn_embed,
            offset_embed_dim=CROSS_OFFSET_EMBED_DIM,
            hidden_dim=dist_attn_hidden,
            dropout=0.25,
            shared_atom_embed=self.distance_attention.atom_embed,
        )
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

        # Inter-residue bond geometry projection (CA-CA prev/next, peptide bonds)
        self.bond_proj = nn.Linear(N_BOND_GEOM, 16)
        bond_dim = 16

        struct_cnn_input = dist_attn_hidden + 64 + 32 + 16 + 16 + dssp_dim + bond_dim
        self.struct_input_norm = nn.LayerNorm(struct_cnn_input)
        self.struct_input_dropout = nn.Dropout(input_dropout)

        struct_layers = []
        in_ch = struct_cnn_input
        for out_ch, drop_p in zip(struct_cnn_channels, struct_layer_dropouts):
            struct_layers.append(ResidualBlock1D(in_ch, out_ch, kernel))
            struct_layers.append(nn.Dropout(drop_p))
            in_ch = out_ch
        self.struct_cnn = nn.Sequential(*struct_layers)
        struct_cnn_out = struct_cnn_channels[-1]

        # Spatial neighbor attention with joint distance attention pathway
        # (mirrors model.ShiftPredictor: the per-neighbor distance embedding
        # combines query+CA-CA+neighbor atom-pair distances).
        self.spatial_attention = SpatialNeighborAttention(
            n_residue_types=n_residue_types,
            n_ss_types=n_ss_types,
            k_neighbors=k_spatial,
            embed_dim=64,
            hidden_dim=spatial_hidden,
            dropout=0.30,
            dist_attn_hidden=dist_attn_hidden,
            query_dim=struct_cnn_out,
        )

        base_encoder_dim = struct_cnn_out + spatial_hidden

        # ========== Shift Context Encoder ==========
        self.shift_context_encoder = ShiftContextEncoder(
            n_shifts=n_shifts,
            n_residue_types=n_residue_types,
            embed_dim=48,
            channels=shift_context_channels,
            kernel=kernel,
            dropout=shift_context_dropout,
        )
        shift_context_dim = self.shift_context_encoder.out_dim

        # ========== Spatial Neighbor Shift Attention (NEW) ==========
        # Attends to observed shifts at the K spatially-closest residues.
        # Complements ShiftContextEncoder (sequence window): catches shift
        # signal from beta-sheet partners and other long-range contacts.
        # Query carries the shift-type one-hot so attention can specialize
        # per target shift type.
        spatial_shift_query_dim = base_encoder_dim + shift_context_dim + n_shifts
        self.spatial_shift_attention = SpatialNeighborShiftAttention(
            n_shifts=n_shifts,
            n_residue_types=n_residue_types,
            query_dim=spatial_shift_query_dim,
            hidden_dim=spatial_hidden,
            n_heads=4,
            dropout=attn_dropout,
        )
        spatial_shift_dim = self.spatial_shift_attention.out_dim

        # ========== Fusion + Prediction ==========
        # fused = base + shift_ctx + spatial_shift_ctx + retrieval_ctx + transferred + trust + shift_type_emb
        self.shift_type_proj = nn.Sequential(
            nn.Linear(n_shifts, 128),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(128, 128),
        )

        fused_dim = (base_encoder_dim + shift_context_dim + spatial_shift_dim + 128)

        self.prediction_head = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(256, 1),
        )

        # Initialize output to zero
        with torch.no_grad():
            self.prediction_head[-1].weight.zero_()
            self.prediction_head[-1].bias.zero_()

    def forward(
        self,
        # Structural inputs (same as model.py ShiftPredictor)
        atom1_idx, atom2_idx, distances, dist_mask,
        residue_idx, ss_idx, mismatch_idx, is_valid,
        dssp_features,
        neighbor_res_idx, neighbor_ss_idx,
        neighbor_dist, neighbor_seq_sep, neighbor_angles,
        neighbor_valid,
        # Spatial neighbor distance features (for joint distance attention)
        neighbor_atom1_idx=None, neighbor_atom2_idx=None,
        neighbor_distances=None, neighbor_dist_mask=None,
        # Cross-residue distance features (center residue only)
        cross_atom1_idx=None, cross_atom2_idx=None, cross_offset_idx=None,
        cross_distances=None, cross_dist_mask=None,
        # Inter-residue bond geometry
        bond_geom=None,
        # Shift context inputs (sequence window)
        context_residue_idx=None,
        context_observed_shifts=None,
        context_shift_masks=None,
        context_is_valid=None,
        # Spatial-neighbor observed shifts (NEW: nearby in space, possibly far in sequence)
        spatial_neighbor_shifts=None,       # (B, K, n_shifts) z-normalized
        spatial_neighbor_shift_masks=None,  # (B, K, n_shifts) availability
        # Shift type conditioning
        shift_type=None,           # (B, n_shifts) one-hot shift type
        # Observed shifts at center (for retrieval conditioning)
        center_observed_shifts=None,
        center_shift_masks=None,
        # Deprecated: accepted but ignored
        physics_features=None,
        **kwargs,
    ):
        """Forward pass. Returns (B,) scalar prediction."""
        B = distances.size(0)
        W = distances.size(1)
        center_w = W // 2

        # ========== 1. Structural Encoder ==========
        dist_emb = self.distance_attention(atom1_idx, atom2_idx, distances, dist_mask)

        # Cross-residue features at the center position. When cross_* tensors
        # are absent (legacy caches), CrossDistanceAttention is skipped and
        # behavior matches the pre-cross checkpoint.
        if cross_atom1_idx is not None and cross_distances is not None:
            cross_a1 = cross_atom1_idx.unsqueeze(1)
            cross_a2 = cross_atom2_idx.unsqueeze(1)
            cross_off = cross_offset_idx.unsqueeze(1)
            cross_d = cross_distances.unsqueeze(1)
            cross_m = cross_dist_mask.unsqueeze(1)
            cross_emb = self.cross_distance_attention(
                cross_a1, cross_a2, cross_off, cross_d, cross_m
            ).squeeze(1)
            center_aa = residue_idx[:, center_w]
            gate = torch.sigmoid(self.cross_gate(center_aa)).squeeze(-1)
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

        # Bond geometry projection
        if bond_geom is None:
            bond_geom = torch.zeros(B, W, N_BOND_GEOM, device=distances.device)
        bond_emb = self.bond_proj(bond_geom)

        if self.dssp_proj is not None:
            dssp_emb = self.dssp_proj(dssp_features)
            x = torch.cat([dist_emb, res_emb, ss_emb, mismatch_emb, valid_emb, dssp_emb, bond_emb], dim=-1)
        else:
            x = torch.cat([dist_emb, res_emb, ss_emb, mismatch_emb, valid_emb, bond_emb], dim=-1)

        x = self.struct_input_dropout(self.struct_input_norm(x))
        x = x.transpose(1, 2)
        x = self.struct_cnn(x)
        x = x.transpose(1, 2)

        center_idx = x.size(1) // 2
        x_center = x[:, center_idx, :]

        # Joint distance attention for spatial neighbors (mirrors model.py).
        # Combines query intra-pairs + CA-CA pair + neighbor intra-pairs into
        # one attention set so the per-neighbor 256-d embedding encodes joint
        # geometry. Falls back to None when neighbor distance arrays absent.
        neighbor_dist_embeddings = None
        if neighbor_atom1_idx is not None and neighbor_distances is not None:
            K_sp = neighbor_atom1_idx.size(1)
            M_sp = neighbor_atom1_idx.size(2)

            q_a1 = atom1_idx[:, center_w, :].unsqueeze(1).expand(-1, K_sp, -1)
            q_a2 = atom2_idx[:, center_w, :].unsqueeze(1).expand(-1, K_sp, -1)
            q_d = distances[:, center_w, :].unsqueeze(1).expand(-1, K_sp, -1)
            q_m = dist_mask[:, center_w, :].unsqueeze(1).expand(-1, K_sp, -1)

            ca_idx = 1  # CA in ATOM_TYPES
            ca_a1 = torch.full((B, K_sp, 1), ca_idx, dtype=torch.long, device=distances.device)
            ca_a2 = torch.full((B, K_sp, 1), ca_idx, dtype=torch.long, device=distances.device)
            ca_d = neighbor_dist.unsqueeze(-1)
            ca_m = neighbor_valid.unsqueeze(-1)

            joint_a1 = torch.cat([q_a1, ca_a1, neighbor_atom1_idx], dim=-1)
            joint_a2 = torch.cat([q_a2, ca_a2, neighbor_atom2_idx], dim=-1)
            joint_d = torch.cat([q_d, ca_d, neighbor_distances], dim=-1)
            joint_m = torch.cat([q_m, ca_m, neighbor_dist_mask], dim=-1)

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

        base_encoding = torch.cat([x_center, x_spatial], dim=-1)

        # ========== 2. Shift Context Encoder ==========
        shift_context = self.shift_context_encoder(
            context_residue_idx, context_observed_shifts,
            context_shift_masks, context_is_valid,
        )

        # ========== 3. Spatial Neighbor Shift Attention (NEW) ==========
        # Attends to observed shifts at the K spatial neighbors. Falls back
        # to zero-filled tensors if not provided by the dataset.
        if spatial_neighbor_shifts is None:
            K_sp = neighbor_res_idx.size(1)
            spatial_neighbor_shifts = torch.zeros(
                B, K_sp, self.n_shifts, device=base_encoding.device)
            spatial_neighbor_shift_masks = torch.zeros(
                B, K_sp, self.n_shifts, device=base_encoding.device)

        spatial_shift_query = torch.cat([
            base_encoding, shift_context, shift_type,
        ], dim=-1)
        spatial_shift_context = self.spatial_shift_attention(
            query=spatial_shift_query,
            neighbor_res_idx=neighbor_res_idx,
            neighbor_shifts=spatial_neighbor_shifts,
            neighbor_shift_masks=spatial_neighbor_shift_masks,
            neighbor_dist=neighbor_dist,
            neighbor_valid=neighbor_valid,
        )

        # ========== 5. Fusion + Prediction ==========
        shift_type_emb = self.shift_type_proj(shift_type)

        fused = torch.cat([
            base_encoding,
            shift_context,
            spatial_shift_context,
            shift_type_emb,
        ], dim=-1)

        prediction = self.prediction_head(fused).squeeze(-1)  # (B,)

        # Safety: clamp NaN
        prediction = torch.where(torch.isnan(prediction), torch.zeros_like(prediction), prediction)

        return prediction


# ============================================================================
# Factory
# ============================================================================

def create_imputation_model(
    n_atom_types: int,
    n_shifts: int = 6,
    shift_cols: list = None,  # kept for caller-API compat; not used in the model
    **kwargs,
) -> ShiftImputationModel:
    """Create imputation model with sensible defaults."""
    for _k in ('retrieval_hidden', 'retrieval_heads', 'retrieval_dropout'):
        kwargs.pop(_k, None)
    return ShiftImputationModel(
        n_atom_types=n_atom_types,
        n_shifts=n_shifts,
        **kwargs,
    )
