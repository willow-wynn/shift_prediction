1#!/usr/bin/env python3
"""
Chemical Shift Imputation Model with Structure + Observed Shifts + Retrieval.

Combines three information sources:
1. Structural encoding (distances, spatial neighbors, DSSP, physics)
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
- PhysicsFeatureEncoder
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
    RETRIEVAL_HIDDEN, RETRIEVAL_HEADS, RETRIEVAL_DROPOUT,
    MAX_VALID_DISTANCES,
)
from random_coil import build_rc_tensor
from model import (
    DistanceAttentionPerPosition,
    ResidualBlock1D,
    SpatialNeighborAttention,
    PhysicsFeatureEncoder,
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
# Unified Retrieval Transfer
# ============================================================================

class UnifiedRetrievalTransfer(nn.Module):
    """Single retrieval mechanism producing both latent context and transferred shifts.

    Replaces the twin RetrievalCrossAttention + QueryConditionedTransfer from model.py.

    Query = concat(base_encoding, shift_context, observed_shifts_at_center)
    attends to K retrieved neighbors via multi-head cross-attention.

    Outputs:
    - retrieval_context: (B, hidden_dim) latent retrieval features
    - transferred_shifts: (B, n_shifts) RC-corrected weighted shift transfer
    - trust_scores: (B, n_shifts) per-shift confidence in transfer

    Shift-aware re-ranking: when observed shifts are available for the query,
    neighbors with similar shifts get upweighted.
    """

    def __init__(
        self,
        query_dim: int,
        n_shifts: int,
        n_residue_types: int = N_RESIDUE_TYPES,
        hidden_dim: int = 192,
        n_heads: int = 4,
        dropout: float = 0.25,
        use_random_coil: bool = True,
        shift_cols: list = None,
    ):
        super().__init__()

        self.n_shifts = n_shifts
        self.n_residue_types = n_residue_types
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.use_random_coil = use_random_coil

        # Random coil lookup table
        if use_random_coil:
            if shift_cols is None:
                shift_cols = ['ca_shift', 'cb_shift', 'c_shift',
                              'n_shift', 'h_shift', 'ha_shift']
            rc_np = build_rc_tensor(STANDARD_RESIDUES, shift_cols)
            rc_padded = np.full((n_residue_types + 1, n_shifts), np.nan, dtype=np.float32)
            rc_padded[:rc_np.shape[0], :rc_np.shape[1]] = rc_np
            self.register_buffer('rc_table', torch.from_numpy(rc_padded))
        else:
            self.rc_table = None

        # Query projection: takes structural encoding + shift context + observed shifts
        self.query_proj = nn.Sequential(
            nn.Linear(query_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Neighbor representation
        self.residue_embed = nn.Embedding(n_residue_types + 1, 16)
        # neighbor input: shifts (n_shifts) + shift_masks (n_shifts) + res_embed (16) + cosine_sim (1) + same_type (1)
        neighbor_input_dim = n_shifts + n_shifts + 16 + 1 + 1
        self.neighbor_proj = nn.Sequential(
            nn.Linear(neighbor_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Cross-attention Q/K/V
        self.attn_q = nn.Linear(hidden_dim, hidden_dim)
        self.attn_k = nn.Linear(hidden_dim, hidden_dim)
        self.attn_v = nn.Linear(hidden_dim, hidden_dim)

        # Shift-aware re-ranking: project observed shifts to attention bias
        # When query has observed shifts, bias attention toward neighbors with similar shifts
        self.shift_similarity_proj = nn.Sequential(
            nn.Linear(n_shifts * 2, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, n_heads),
        )

        # Output projections
        self.context_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Per-shift transfer weights from attention output
        self.to_shift_weights = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_shifts),
        )

        # Trust gate
        self.trust_residue_embed = nn.Embedding(n_residue_types + 1, 32)
        self.trust_query_proj = nn.Sequential(
            nn.Linear(query_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Per-shift stats: coverage, variance, mean_dist, same_type_coverage
        per_shift_stats_dim = 4
        global_stats_dim = 4

        trust_context_dim = hidden_dim + 32 + global_stats_dim
        self.trust_context_proj = nn.Sequential(
            nn.Linear(trust_context_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.trust_per_shift = nn.Sequential(
            nn.Linear(hidden_dim + per_shift_stats_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Per-shift scaling
        self.shift_scale = nn.Parameter(torch.ones(n_shifts))
        self.fallback_shift = nn.Parameter(torch.zeros(n_shifts))
        self.fallback_context = nn.Parameter(torch.zeros(hidden_dim))

        self.out_dim = hidden_dim

    def _apply_random_coil_correction(self, retrieved_shifts, query_residue_code,
                                       retrieved_residue_codes, retrieved_shift_masks):
        """Apply RC correction: corrected = RC[query] + (shift - RC[retrieved])."""
        if self.rc_table is None:
            return retrieved_shifts

        query_idx = query_residue_code.clamp(0, self.rc_table.size(0) - 1)
        retrieved_idx = retrieved_residue_codes.clamp(0, self.rc_table.size(0) - 1)

        rc_query = self.rc_table[query_idx].unsqueeze(1)        # (B, 1, S)
        rc_retrieved = self.rc_table[retrieved_idx]              # (B, K, S)

        corrected = rc_query + (retrieved_shifts - rc_retrieved)

        rc_valid = ~torch.isnan(rc_query) & ~torch.isnan(rc_retrieved)
        corrected = torch.where(rc_valid, corrected, retrieved_shifts)
        corrected = torch.where(retrieved_shift_masks, corrected, retrieved_shifts)
        return corrected

    def forward(
        self,
        query_encoding: torch.Tensor,          # (B, query_dim)
        query_residue_code: torch.Tensor,      # (B,)
        query_observed_shifts: torch.Tensor,   # (B, n_shifts) observed shifts at center
        query_shift_masks: torch.Tensor,       # (B, n_shifts) which are available
        retrieved_shifts: torch.Tensor,        # (B, K, n_shifts)
        retrieved_shift_masks: torch.Tensor,   # (B, K, n_shifts)
        retrieved_residue_codes: torch.Tensor, # (B, K)
        retrieved_distances: torch.Tensor,     # (B, K) cosine similarities
        retrieved_valid: torch.Tensor,         # (B, K)
    ):
        """
        Returns:
            retrieval_context: (B, hidden_dim)
            transferred_shifts: (B, n_shifts)
            trust_scores: (B, n_shifts)
        """
        B, K, S = retrieved_shifts.shape

        # RC correction
        if self.use_random_coil:
            retrieved_shifts = self._apply_random_coil_correction(
                retrieved_shifts, query_residue_code,
                retrieved_residue_codes, retrieved_shift_masks,
            )

        any_valid = retrieved_valid.any(dim=1)  # (B,)
        same_type = (retrieved_residue_codes == query_residue_code.unsqueeze(1)).float()

        # Build query
        q = self.query_proj(query_encoding)  # (B, hidden_dim)

        # Build neighbor representations
        res_emb = self.residue_embed(retrieved_residue_codes)  # (B, K, 16)
        neighbor_input = torch.cat([
            retrieved_shifts,
            retrieved_shift_masks.float(),
            res_emb,
            retrieved_distances.unsqueeze(-1),
            same_type.unsqueeze(-1),
        ], dim=-1)
        kv = self.neighbor_proj(neighbor_input)  # (B, K, hidden_dim)

        # Multi-head cross-attention
        Q = self.attn_q(q).unsqueeze(1).view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)
        K_attn = self.attn_k(kv).view(B, K, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.attn_v(kv).view(B, K, self.n_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(Q, K_attn.transpose(-2, -1)) * self.scale  # (B, H, 1, K)

        # Shift-aware re-ranking bias
        # Compare query observed shifts with each neighbor's shifts
        query_shifts_expanded = query_observed_shifts.unsqueeze(1).expand(-1, K, -1)  # (B, K, S)
        shift_diff = torch.cat([
            (query_shifts_expanded - retrieved_shifts) * query_shift_masks.unsqueeze(1),
            query_shift_masks.unsqueeze(1).expand(-1, K, -1),
        ], dim=-1)  # (B, K, 2*S)
        shift_bias = self.shift_similarity_proj(shift_diff)  # (B, K, n_heads)
        shift_bias = shift_bias.permute(0, 2, 1).unsqueeze(2)  # (B, H, 1, K)
        attn_scores = attn_scores + shift_bias

        # Mask invalid neighbors
        attn_mask = ~retrieved_valid.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, K)
        # Handle all-masked case
        all_masked = ~any_valid
        if all_masked.any():
            attn_mask = attn_mask.clone()
            attn_mask[all_masked, :, :, 0] = False

        attn_scores = attn_scores.masked_fill(attn_mask, -1e4)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, H, 1, K)

        attn_out = torch.matmul(attn_weights, V)  # (B, H, 1, head_dim)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, self.hidden_dim)

        # Retrieval context output
        retrieval_context = self.context_proj(attn_out)
        fallback_ctx = self.fallback_context.unsqueeze(0).expand(B, -1)
        retrieval_context = torch.where(any_valid.unsqueeze(-1), retrieval_context, fallback_ctx)

        # Transferred shifts via learned per-shift weights
        shift_weight_bias = self.to_shift_weights(attn_out)  # (B, S)
        base_weights = retrieved_distances.unsqueeze(-1).expand(-1, -1, S)
        same_type_bonus = same_type.unsqueeze(-1) * 0.5
        weights = base_weights + same_type_bonus + shift_weight_bias.unsqueeze(1) * 0.1

        valid_mask = retrieved_valid.unsqueeze(-1) & retrieved_shift_masks
        weights = weights.masked_fill(~valid_mask, -1e4)
        weights = F.softmax(weights, dim=1)  # (B, K, S)

        transferred = (weights * retrieved_shifts).sum(dim=1) * self.shift_scale
        fallback_s = self.fallback_shift.unsqueeze(0).expand(B, -1)
        transferred = torch.where(any_valid.unsqueeze(-1), transferred, fallback_s)

        # Trust gate with per-shift statistics
        with torch.no_grad():
            valid_float = retrieved_valid.float()
            valid_mask_float = valid_mask.float()

            per_shift_count = valid_mask_float.sum(dim=1)
            per_shift_coverage = per_shift_count / (K + 1e-8)

            masked_shifts = retrieved_shifts * valid_mask_float
            per_shift_mean = masked_shifts.sum(dim=1) / (per_shift_count + 1e-8)
            shift_diff_sq = (retrieved_shifts - per_shift_mean.unsqueeze(1)) ** 2
            per_shift_var = (shift_diff_sq * valid_mask_float).sum(dim=1) / (per_shift_count + 1e-8)
            per_shift_var_norm = torch.log1p(per_shift_var).clamp(0, 5) / 5

            dist_expanded = retrieved_distances.unsqueeze(-1).expand(-1, -1, S)
            per_shift_mean_dist = (dist_expanded * valid_mask_float).sum(dim=1) / (per_shift_count + 1e-8)

            same_type_with_shift = same_type.unsqueeze(-1) * valid_mask_float
            per_shift_same_type = same_type_with_shift.sum(dim=1) / (per_shift_count + 1e-8)

            per_shift_stats = torch.stack([
                per_shift_coverage, per_shift_var_norm,
                per_shift_mean_dist, per_shift_same_type,
            ], dim=-1)  # (B, S, 4)

            n_valid = valid_float.sum(dim=1, keepdim=True) / K
            n_same_type = (same_type.squeeze(-1) if same_type.dim() > 2 else same_type).sum(dim=1, keepdim=True) * valid_float.sum(dim=1, keepdim=True) / (K * K + 1e-8)
            # Fix: same_type is (B, K), valid_float is (B, K)
            n_same_type = (same_type * valid_float).sum(dim=1, keepdim=True) / K

            masked_dist_global = retrieved_distances.masked_fill(~retrieved_valid, 0.0)
            mean_dist = masked_dist_global.sum(dim=1, keepdim=True) / (valid_float.sum(dim=1, keepdim=True) + 1e-8)
            max_dist = masked_dist_global.max(dim=1, keepdim=True)[0]

            global_stats = torch.cat([mean_dist, max_dist, n_valid, n_same_type], dim=-1)

        query_res_embed = self.trust_residue_embed(query_residue_code)
        trust_query = self.trust_query_proj(query_encoding)

        trust_context = self.trust_context_proj(
            torch.cat([trust_query, query_res_embed, global_stats], dim=-1)
        )

        trust_context_expanded = trust_context.unsqueeze(1).expand(-1, S, -1)
        trust_input = torch.cat([trust_context_expanded, per_shift_stats], dim=-1)
        trust = self.trust_per_shift(trust_input).squeeze(-1)  # (B, S)

        trust = torch.where(any_valid.unsqueeze(-1), trust, torch.zeros_like(trust))
        has_any_neighbor = per_shift_count > 0
        trust = torch.where(has_any_neighbor, trust, torch.zeros_like(trust))

        return retrieval_context, transferred, trust


# ============================================================================
# Shift Imputation Model
# ============================================================================

class ShiftImputationModel(nn.Module):
    """Chemical shift imputation using structure + observed shifts + retrieval.

    Architecture:
    1. Structural Encoder: DistanceAttention -> CNN -> center -> SpatialAttention -> Physics
    2. Shift Context Encoder: 1D CNN over observed shifts in local window
    3. Unified Retrieval Transfer: cross-attention with shift-aware re-ranking
    4. Fusion -> shift-type-conditioned prediction head -> scalar output
    """

    def __init__(
        self,
        n_atom_types: int,
        n_shifts: int = 6,
        n_physics: int = 28,
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
        # Retrieval
        retrieval_hidden: int = RETRIEVAL_HIDDEN,
        retrieval_heads: int = RETRIEVAL_HEADS,
        retrieval_dropout: float = RETRIEVAL_DROPOUT,
        use_random_coil: bool = True,
        shift_cols: list = None,
    ):
        super().__init__()

        struct_cnn_channels = struct_cnn_channels or [256, 512, 768, 1024, 1280]
        struct_layer_dropouts = struct_layer_dropouts or [0.40, 0.40, 0.40, 0.40, 0.40]
        shift_context_channels = shift_context_channels or [128, 256, 384]

        self.n_shifts = n_shifts
        self.n_residue_types = n_residue_types
        self.retrieval_dropout_rate = retrieval_dropout

        # ========== Structural Encoder ==========
        self.distance_attention = DistanceAttentionPerPosition(
            n_atom_types=n_atom_types,
            embed_dim=dist_attn_embed,
            hidden_dim=dist_attn_hidden,
            dropout=0.25,
        )

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

        struct_cnn_input = dist_attn_hidden + 64 + 32 + 16 + 16 + dssp_dim
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

        self.spatial_attention = SpatialNeighborAttention(
            n_residue_types=n_residue_types,
            n_ss_types=n_ss_types,
            k_neighbors=k_spatial,
            embed_dim=64,
            hidden_dim=spatial_hidden,
            dropout=0.30,
        )

        self.physics_encoder = PhysicsFeatureEncoder(
            n_physics=n_physics, hidden_dim=64, dropout=0.2,
        )
        physics_dim = self.physics_encoder.out_dim

        base_encoder_dim = struct_cnn_out + spatial_hidden + physics_dim

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

        # ========== Unified Retrieval Transfer ==========
        # Query dim = base_encoder + shift_context + observed_shifts_at_center
        retrieval_query_dim = base_encoder_dim + shift_context_dim + n_shifts
        self.retrieval = UnifiedRetrievalTransfer(
            query_dim=retrieval_query_dim,
            n_shifts=n_shifts,
            n_residue_types=n_residue_types,
            hidden_dim=retrieval_hidden,
            n_heads=retrieval_heads,
            dropout=retrieval_dropout,
            use_random_coil=use_random_coil,
            shift_cols=shift_cols,
        )

        # ========== Fusion + Prediction ==========
        # fused = base_encoding + shift_context + retrieval_context + transferred_shifts + trust + shift_type_embed
        self.shift_type_proj = nn.Sequential(
            nn.Linear(n_shifts, 128),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(128, 128),
        )

        fused_dim = (base_encoder_dim + shift_context_dim +
                     retrieval_hidden + n_shifts + n_shifts + 128)

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
        # Structural inputs (same as model.py)
        atom1_idx, atom2_idx, distances, dist_mask,
        residue_idx, ss_idx, mismatch_idx, is_valid,
        dssp_features,
        neighbor_res_idx, neighbor_ss_idx,
        neighbor_dist, neighbor_seq_sep, neighbor_angles,
        neighbor_valid,
        # Retrieval inputs
        query_residue_code,
        retrieved_shifts, retrieved_shift_masks,
        retrieved_residue_codes, retrieved_distances, retrieved_valid,
        # Physics features
        physics_features,
        # Shift context inputs (NEW)
        context_residue_idx,       # (B, W) residue types in context window
        context_observed_shifts,   # (B, W, n_shifts) z-normalized shifts
        context_shift_masks,       # (B, W, n_shifts) availability masks
        context_is_valid,          # (B, W) position validity
        # Shift type conditioning
        shift_type,                # (B, n_shifts) one-hot shift type
        # Observed shifts at center (for retrieval conditioning)
        center_observed_shifts,    # (B, n_shifts) shifts at center residue
        center_shift_masks,        # (B, n_shifts) availability at center
        **kwargs,
    ):
        """Forward pass. Returns (B, 1) scalar prediction."""
        B = distances.size(0)

        # ========== 1. Structural Encoder ==========
        dist_emb = self.distance_attention(atom1_idx, atom2_idx, distances, dist_mask)
        res_emb = self.residue_embed(residue_idx)
        ss_emb = self.ss_embed(ss_idx)
        mismatch_emb = self.mismatch_embed(mismatch_idx)
        valid_emb = self.valid_embed(is_valid.unsqueeze(-1))

        if self.dssp_proj is not None:
            dssp_emb = self.dssp_proj(dssp_features)
            x = torch.cat([dist_emb, res_emb, ss_emb, mismatch_emb, valid_emb, dssp_emb], dim=-1)
        else:
            x = torch.cat([dist_emb, res_emb, ss_emb, mismatch_emb, valid_emb], dim=-1)

        x = self.struct_input_dropout(self.struct_input_norm(x))
        x = x.transpose(1, 2)
        x = self.struct_cnn(x)
        x = x.transpose(1, 2)

        center_idx = x.size(1) // 2
        x_center = x[:, center_idx, :]

        x_spatial = self.spatial_attention(
            neighbor_res_idx, neighbor_ss_idx,
            neighbor_dist, neighbor_seq_sep, neighbor_angles,
            neighbor_valid,
        )

        if physics_features is None:
            physics_features = torch.zeros(
                B, self.physics_encoder.mlp[0].in_features,
                device=x_center.device, dtype=x_center.dtype,
            )
        x_physics = self.physics_encoder(physics_features)

        base_encoding = torch.cat([x_center, x_spatial, x_physics], dim=-1)

        # ========== 2. Shift Context Encoder ==========
        shift_context = self.shift_context_encoder(
            context_residue_idx, context_observed_shifts,
            context_shift_masks, context_is_valid,
        )

        # ========== 3. Retrieval Transfer ==========
        # Build retrieval query from structural + shift context + observed
        center_shifts_clean = center_observed_shifts * center_shift_masks
        retrieval_query = torch.cat([
            base_encoding, shift_context, center_shifts_clean,
        ], dim=-1)

        # Apply retrieval dropout during training
        if self.training and self.retrieval_dropout_rate > 0:
            drop_mask = torch.rand(B, device=base_encoding.device) < self.retrieval_dropout_rate
            retrieved_valid_use = retrieved_valid.clone()
            retrieved_valid_use[drop_mask] = False
        else:
            retrieved_valid_use = retrieved_valid

        retrieval_context, transferred_shifts, trust_scores = self.retrieval(
            query_encoding=retrieval_query,
            query_residue_code=query_residue_code,
            query_observed_shifts=center_shifts_clean,
            query_shift_masks=center_shift_masks,
            retrieved_shifts=retrieved_shifts,
            retrieved_shift_masks=retrieved_shift_masks,
            retrieved_residue_codes=retrieved_residue_codes,
            retrieved_distances=retrieved_distances,
            retrieved_valid=retrieved_valid_use,
        )

        # ========== 4. Fusion + Prediction ==========
        shift_type_emb = self.shift_type_proj(shift_type)

        fused = torch.cat([
            base_encoding,
            shift_context,
            retrieval_context,
            transferred_shifts,
            trust_scores,
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
    n_physics: int = 28,
    shift_cols: list = None,
    use_random_coil: bool = True,
    **kwargs,
) -> ShiftImputationModel:
    """Create imputation model with sensible defaults."""
    return ShiftImputationModel(
        n_atom_types=n_atom_types,
        n_shifts=n_shifts,
        n_physics=n_physics,
        shift_cols=shift_cols,
        use_random_coil=use_random_coil,
        **kwargs,
    )
