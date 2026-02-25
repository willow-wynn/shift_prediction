#!/usr/bin/env python3
"""
Chemical Shift Predictor with Retrieval Augmentation (Better Data Pipeline)

Adapted from homologies/model_with_retrieval.py with the following modifications:

1. NEW PhysicsFeatureEncoder module:
   - Encodes physics features (ring currents, HSE, H-bonds, order param)
   - Simple MLP: Linear -> GELU -> Dropout -> Linear producing 64-dim output
   - Output concatenated with base encoder, expanding base_encoder_dim by 64

2. MODIFIED QueryConditionedTransfer:
   - Supports random coil correction on transferred shifts
   - When use_random_coil=True: corrected = RC[query_aa] + (shift - RC[retrieved_aa])
   - Correction applied BEFORE weighted averaging in the transfer
   - Uses RC_SHIFTS from random_coil module via a registered buffer lookup table

3. MODIFIED ShiftPredictorWithRetrieval:
   - Accepts physics_features input in forward()
   - Integrates PhysicsFeatureEncoder into the base encoding
   - base_encoder_dim = cnn_out_dim + spatial_hidden + 64 (physics)
   - Imports constants from config instead of hardcoding
   - Includes create_model() factory function

4. UNCHANGED components (identical to original):
   - DistanceAttentionPerPosition
   - SpatialNeighborAttention
   - ResidualBlock1D
   - RetrievalCrossAttention
   - RetrievalShiftTransfer
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
    RETRIEVAL_HIDDEN, RETRIEVAL_HEADS, RETRIEVAL_DROPOUT,
    MAX_VALID_DISTANCES,
)
from random_coil import RC_SHIFTS, build_rc_tensor


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


class SpatialNeighborAttention(nn.Module):
    """Attention over k spatially proximate residues."""

    def __init__(self, n_residue_types, n_ss_types, k_neighbors=5,
                 embed_dim=64, hidden_dim=256, dropout=0.30):
        super().__init__()

        self.k = k_neighbors

        self.residue_embed = nn.Embedding(n_residue_types + 1, embed_dim)
        self.ss_embed = nn.Embedding(n_ss_types + 1, embed_dim // 2)

        self.continuous_proj = nn.Linear(6, embed_dim // 2)

        combined_dim = embed_dim + embed_dim // 2 + embed_dim // 2

        self.attn_score = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.value_net = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.fallback_embed = nn.Parameter(torch.zeros(hidden_dim))
        self.out_dim = hidden_dim

    def forward(self, neighbor_res_idx, neighbor_ss_idx,
                neighbor_dist, neighbor_seq_sep, neighbor_angles,
                neighbor_valid):
        B = neighbor_res_idx.size(0)

        res_emb = self.residue_embed(neighbor_res_idx)
        ss_emb = self.ss_embed(neighbor_ss_idx)

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

        combined = torch.cat([res_emb, ss_emb, cont_emb], dim=-1)

        any_valid = neighbor_valid.any(dim=1)

        scores = self.attn_score(combined).squeeze(-1)
        scores = scores.masked_fill(~neighbor_valid, -1e4)
        attn_weights = F.softmax(scores, dim=-1)

        values = self.value_net(combined)
        output = (values * attn_weights.unsqueeze(-1)).sum(dim=1)

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
# NEW: Physics Feature Encoder
# ============================================================================

class PhysicsFeatureEncoder(nn.Module):
    """
    Encode physics-based features into a fixed-dim representation.

    Input features (~8 dimensions):
        - ring_current_h, ring_current_ha (2)
        - hse_up, hse_down, hse_ratio (3)
        - hbond_dist_1, hbond_energy_1 (2)
        - hbond_dist_2, hbond_energy_2 (2) [currently unused placeholders -- kept for compat]
        - order_parameter (1)
        Total: ~8 (flexible, determined at init)

    Output: 64-dim vector concatenated with the base encoder.
    """

    def __init__(self, n_physics: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_physics, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.out_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, n_physics) physics feature vector

        Returns:
            (B, hidden_dim) encoded features
        """
        # Replace NaN with 0 for safety
        x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        return self.mlp(x)


# ============================================================================
# Retrieval Cross-Attention (unchanged from original)
# ============================================================================

class RetrievalCrossAttention(nn.Module):
    """
    Cross-attention to retrieved similar residues.

    The model learns to:
    1. Weight retrieved examples by their relevance (cosine similarity already provides a prior)
    2. Weight by same-residue-type (chemical shifts are residue-type-specific)
    3. Extract useful information from retrieved shifts

    Key insight: retrieved shifts are already normalized, so the model learns to
    use them as a strong prior, weighted by relevance.
    """

    def __init__(
        self,
        query_dim: int,
        n_shifts: int,
        n_residue_types: int = N_RESIDUE_TYPES,
        hidden_dim: int = 256,
        n_heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.query_dim = query_dim
        self.n_shifts = n_shifts
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads

        # Project query (structural encoding) to attention space
        self.query_proj = nn.Linear(query_dim, hidden_dim)

        # Project retrieved information to key/value space
        # Retrieved info: shifts (n_shifts) + shift_masks (n_shifts) + residue_code (embedded)
        self.residue_embed = nn.Embedding(n_residue_types + 1, 32)

        # Key projection: shift info + residue type + cosine similarity
        key_input_dim = n_shifts + n_shifts + 32 + 1  # shifts + masks + res_embed + cosine
        self.key_proj = nn.Linear(key_input_dim, hidden_dim)

        # Value projection: primarily the shifts themselves
        self.value_proj = nn.Linear(key_input_dim, hidden_dim)

        # Multi-head attention
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Learnable fallback when no retrieval available
        self.fallback = nn.Parameter(torch.zeros(hidden_dim))

        self.out_dim = hidden_dim

    def forward(
        self,
        query: torch.Tensor,           # (B, query_dim) - structural encoding
        query_residue_code: torch.Tensor,  # (B,) - query residue type
        retrieved_shifts: torch.Tensor,     # (B, K, n_shifts)
        retrieved_shift_masks: torch.Tensor,  # (B, K, n_shifts)
        retrieved_residue_codes: torch.Tensor,  # (B, K)
        retrieved_distances: torch.Tensor,  # (B, K) - cosine similarities
        retrieved_valid: torch.Tensor,      # (B, K) - validity mask
        retrieval_dropout: float = 0.0,     # Dropout retrieval during training
    ) -> torch.Tensor:
        """
        Args:
            query: Structural encoding of query residue
            query_residue_code: Residue type of query
            retrieved_shifts: Chemical shifts of retrieved residues (normalized)
            retrieved_shift_masks: Which shifts are valid for each retrieved residue
            retrieved_residue_codes: Residue types of retrieved residues
            retrieved_distances: Cosine similarities (higher = more similar)
            retrieved_valid: Which retrieved positions are valid
            retrieval_dropout: Probability of dropping ALL retrieval (forces model to use structure)

        Returns:
            output: (B, hidden_dim) retrieval-augmented features
        """
        B, K, _ = retrieved_shifts.shape

        # Apply retrieval dropout (drop ALL retrieval for some samples)
        if self.training and retrieval_dropout > 0:
            drop_mask = torch.rand(B, device=query.device) < retrieval_dropout
            retrieved_valid = retrieved_valid.clone()
            retrieved_valid[drop_mask] = False

        # Check if any sample has valid retrieval
        any_valid = retrieved_valid.any(dim=1)  # (B,)

        # Project query
        query_proj = self.query_proj(query)  # (B, hidden_dim)
        query_proj = query_proj.unsqueeze(1)  # (B, 1, hidden_dim)

        # Build key/value representations for retrieved
        res_emb = self.residue_embed(retrieved_residue_codes)  # (B, K, 32)

        # Same-type indicator (query residue type matches retrieved)
        same_type = (retrieved_residue_codes == query_residue_code.unsqueeze(1)).float()  # (B, K)

        # Key/value input: shifts + masks + residue embedding + cosine similarity
        kv_input = torch.cat([
            retrieved_shifts,  # (B, K, n_shifts)
            retrieved_shift_masks.float(),  # (B, K, n_shifts)
            res_emb,  # (B, K, 32)
            retrieved_distances.unsqueeze(-1),  # (B, K, 1)
        ], dim=-1)

        keys = self.key_proj(kv_input)  # (B, K, hidden_dim)
        values = self.value_proj(kv_input)  # (B, K, hidden_dim)

        # Create attention mask (True = ignore)
        attn_mask = ~retrieved_valid  # (B, K)

        # Handle case where ALL positions are masked (avoid NaN)
        # Add a dummy position that will be masked in output
        all_masked = ~any_valid  # (B,)
        if all_masked.any():
            # For samples with no valid retrieval, allow attention to first position
            # but we'll replace output with fallback anyway
            attn_mask = attn_mask.clone()
            attn_mask[all_masked, 0] = False

        # Multi-head cross-attention
        # Query: (B, 1, hidden_dim), Key/Value: (B, K, hidden_dim)
        attn_out, _ = self.attn(
            query_proj, keys, values,
            key_padding_mask=attn_mask,
            need_weights=False,
        )  # (B, 1, hidden_dim)

        attn_out = attn_out.squeeze(1)  # (B, hidden_dim)

        # Output projection
        output = self.output_proj(attn_out)

        # Replace with fallback for samples with no valid retrieval
        fallback = self.fallback.unsqueeze(0).expand(B, -1)
        output = torch.where(any_valid.unsqueeze(-1), output, fallback)

        return output


class RetrievalShiftTransfer(nn.Module):
    """
    Direct shift transfer from retrieved examples (simple baseline).

    This module computes a weighted average of retrieved shifts,
    providing a strong baseline prediction that the model can refine.

    Weighting is based on:
    1. Cosine similarity (higher = more weight)
    2. Same residue type (bonus weight)
    3. Learned residue-type-specific weights
    """

    def __init__(
        self,
        n_shifts: int,
        n_residue_types: int = N_RESIDUE_TYPES,
        temperature: float = 0.1,
    ):
        super().__init__()

        self.n_shifts = n_shifts
        self.temperature = temperature

        # Learnable bonus for same-type matches (per shift type)
        self.same_type_bonus = nn.Parameter(torch.ones(n_shifts) * 0.5)

        # Learnable per-shift scaling
        self.shift_scale = nn.Parameter(torch.ones(n_shifts))

    def forward(
        self,
        query_residue_code: torch.Tensor,     # (B,)
        retrieved_shifts: torch.Tensor,        # (B, K, n_shifts)
        retrieved_shift_masks: torch.Tensor,   # (B, K, n_shifts)
        retrieved_residue_codes: torch.Tensor, # (B, K)
        retrieved_distances: torch.Tensor,     # (B, K) cosine similarities
        retrieved_valid: torch.Tensor,         # (B, K)
        query_encoding: torch.Tensor = None,   # unused in simple version
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute weighted transfer of shifts from retrieved examples.

        Returns:
            transferred_shifts: (B, n_shifts) weighted average of retrieved shifts
            transfer_confidence: (B, n_shifts) confidence in transfer (0-1)
        """
        B, K, S = retrieved_shifts.shape

        # Same-type indicator
        same_type = (retrieved_residue_codes == query_residue_code.unsqueeze(1))  # (B, K)

        # Compute weights: cosine similarity + same-type bonus
        # Shape: (B, K, n_shifts)
        base_weights = retrieved_distances.unsqueeze(-1).expand(-1, -1, S)  # (B, K, S)
        same_type_bonus = same_type.unsqueeze(-1) * self.same_type_bonus.unsqueeze(0).unsqueeze(0)

        weights = base_weights + same_type_bonus  # (B, K, S)

        # Apply temperature and mask invalid
        weights = weights / self.temperature

        # Mask: must be valid retrieval AND have valid shift for this type
        valid_mask = retrieved_valid.unsqueeze(-1) & retrieved_shift_masks  # (B, K, S)

        weights = weights.masked_fill(~valid_mask, -1e4)
        weights = F.softmax(weights, dim=1)  # (B, K, S)

        # Weighted sum
        transferred_shifts = (weights * retrieved_shifts).sum(dim=1)  # (B, S)
        transferred_shifts = transferred_shifts * self.shift_scale

        # Confidence: fraction of valid matches, weighted by similarity
        confidence = valid_mask.float().sum(dim=1) / K  # (B, S)
        confidence = confidence.clamp(0, 1)

        return transferred_shifts, confidence


# ============================================================================
# MODIFIED: Query-Conditioned Transfer with Random Coil Correction
# ============================================================================

class QueryConditionedTransfer(nn.Module):
    """
    Query-conditioned shift transfer from retrieved examples.

    MODIFIED from original: Supports optional random coil correction.
    When use_random_coil=True, retrieved shifts are corrected BEFORE weighted
    averaging:
        corrected = RC[query_aa] + (retrieved_shift - RC[retrieved_aa])
    This preserves the secondary (structural) shift contribution while adjusting
    for intrinsic chemical shift differences between amino acid types.

    Unlike simple weighted averaging, this module:
    1. Uses the structural encoding to determine which retrieved neighbors to trust
    2. Learns per-shift attention patterns over the K retrieved neighbors
    3. Computes a "trust gate" with PER-SHIFT statistics for calibrated uncertainty

    The trust gate receives:
    - Per-shift coverage, variance, mean_dist, same_type_coverage
    - Query residue type embedding
    """

    def __init__(
        self,
        query_dim: int,
        n_shifts: int,
        n_residue_types: int = N_RESIDUE_TYPES,
        hidden_dim: int = 128,
        n_heads: int = 4,
        dropout: float = 0.2,
        use_random_coil: bool = True,
        shift_cols: list = None,
    ):
        super().__init__()

        self.n_shifts = n_shifts
        self.n_residue_types = n_residue_types
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.use_random_coil = use_random_coil

        # Build random coil lookup table as a buffer (not a parameter)
        # Shape: (n_residue_types, n_shifts) -- NaN where unavailable
        if use_random_coil:
            if shift_cols is None:
                # Default shift column names matching the 6 backbone shifts
                shift_cols = ['ca_shift', 'cb_shift', 'c_shift', 'n_shift', 'h_shift', 'ha_shift']
            rc_np = build_rc_tensor(STANDARD_RESIDUES, shift_cols)  # (n_res, n_shifts)
            # Pad with NaN row for UNK / out-of-range indices
            rc_padded = np.full((n_residue_types + 1, n_shifts), np.nan, dtype=np.float32)
            rc_padded[:rc_np.shape[0], :rc_np.shape[1]] = rc_np
            self.register_buffer('rc_table', torch.from_numpy(rc_padded))
        else:
            self.rc_table = None

        # Project query structural encoding
        self.query_proj = nn.Sequential(
            nn.Linear(query_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Project retrieval context for each neighbor
        # Input: cosine_dist (1) + same_type (1) + residue_type_embed (16)
        self.residue_embed = nn.Embedding(n_residue_types + 1, 16)
        context_input_dim = 1 + 1 + 16  # distance + same_type + residue_embed

        self.context_proj = nn.Sequential(
            nn.Linear(context_input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Multi-head attention: query attends to retrieved contexts
        # to produce per-shift attention weights
        self.attn_q = nn.Linear(hidden_dim, hidden_dim)
        self.attn_k = nn.Linear(hidden_dim, hidden_dim)
        self.attn_v = nn.Linear(hidden_dim, hidden_dim)

        # Project attention output to per-shift weights
        self.to_shift_weights = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_shifts),
        )

        # ========== Enhanced Trust Gate with Per-Shift Statistics ==========
        # Query residue embedding for trust (some residues have tighter distributions)
        self.trust_residue_embed = nn.Embedding(n_residue_types + 1, 32)

        # Project query encoding for trust
        self.trust_query_proj = nn.Sequential(
            nn.Linear(query_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Per-shift statistics input:
        # - coverage: (n_shifts,) fraction of neighbors with each shift
        # - variance: (n_shifts,) variance of retrieved shifts
        # - mean_dist: (n_shifts,) mean distance of neighbors with each shift
        # - same_type_coverage: (n_shifts,) fraction of same-type neighbors with each shift
        per_shift_stats_dim = 4  # coverage, variance, mean_dist, same_type_coverage

        # Global statistics: mean_dist, max_dist, n_valid, n_same_type
        global_stats_dim = 4

        # Trust gate processes: query_proj (hidden) + residue_embed (32) + global_stats (4)
        # Then combines with per-shift stats
        trust_context_dim = hidden_dim + 32 + global_stats_dim

        self.trust_context_proj = nn.Sequential(
            nn.Linear(trust_context_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Per-shift trust computation
        # Input: trust_context (hidden) + per_shift_stats (4) for each shift
        self.trust_per_shift = nn.Sequential(
            nn.Linear(hidden_dim + per_shift_stats_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        # Per-shift output scaling (learnable)
        self.shift_scale = nn.Parameter(torch.ones(n_shifts))

        # Fallback for when no valid retrieval
        self.fallback_shift = nn.Parameter(torch.zeros(n_shifts))

        self.scale = self.head_dim ** -0.5

    def _apply_random_coil_correction(
        self,
        retrieved_shifts: torch.Tensor,       # (B, K, S)
        query_residue_code: torch.Tensor,      # (B,)
        retrieved_residue_codes: torch.Tensor,  # (B, K)
        retrieved_shift_masks: torch.Tensor,   # (B, K, S)
    ) -> torch.Tensor:
        """Apply random coil correction to retrieved shifts BEFORE averaging.

        corrected[b,k,s] = RC[query_aa[b], s] + (shift[b,k,s] - RC[retrieved_aa[b,k], s])

        Where RC values are unavailable (NaN in table), the shift is left unchanged.
        """
        if self.rc_table is None:
            return retrieved_shifts

        # Lookup RC values: clamp indices for safety
        query_idx = query_residue_code.clamp(0, self.rc_table.size(0) - 1)
        retrieved_idx = retrieved_residue_codes.clamp(0, self.rc_table.size(0) - 1)

        rc_query = self.rc_table[query_idx]              # (B, S)
        rc_retrieved = self.rc_table[retrieved_idx]       # (B, K, S)

        # Expand query RC for broadcasting
        rc_query_expanded = rc_query.unsqueeze(1)         # (B, 1, S)

        # Compute correction: RC_query + (shift - RC_retrieved)
        corrected = rc_query_expanded + (retrieved_shifts - rc_retrieved)

        # Where either RC is NaN, fall back to original shift
        rc_valid = ~torch.isnan(rc_query_expanded) & ~torch.isnan(rc_retrieved)
        corrected = torch.where(rc_valid, corrected, retrieved_shifts)

        # Also respect shift masks (don't correct invalid shifts)
        corrected = torch.where(retrieved_shift_masks, corrected, retrieved_shifts)

        return corrected

    def forward(
        self,
        query_residue_code: torch.Tensor,     # (B,)
        retrieved_shifts: torch.Tensor,        # (B, K, n_shifts)
        retrieved_shift_masks: torch.Tensor,   # (B, K, n_shifts)
        retrieved_residue_codes: torch.Tensor, # (B, K)
        retrieved_distances: torch.Tensor,     # (B, K) cosine similarities
        retrieved_valid: torch.Tensor,         # (B, K)
        query_encoding: torch.Tensor,          # (B, query_dim) structural encoding
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute query-conditioned transfer of shifts.

        Returns:
            transferred_shifts: (B, n_shifts) weighted shifts from retrieval
            trust_scores: (B, n_shifts) how much to trust this transfer (0-1)
        """
        B, K, S = retrieved_shifts.shape

        # ========== Apply random coil correction BEFORE averaging ==========
        if self.use_random_coil:
            retrieved_shifts = self._apply_random_coil_correction(
                retrieved_shifts, query_residue_code,
                retrieved_residue_codes, retrieved_shift_masks,
            )

        # Same-type indicator
        same_type = (retrieved_residue_codes == query_residue_code.unsqueeze(1)).float()  # (B, K)

        # Check if any valid retrieval exists
        any_valid = retrieved_valid.any(dim=1)  # (B,)

        # ========== Build query representation ==========
        q = self.query_proj(query_encoding)  # (B, hidden)

        # ========== Build context for each retrieved neighbor ==========
        res_embed = self.residue_embed(retrieved_residue_codes)  # (B, K, 16)

        context_input = torch.cat([
            retrieved_distances.unsqueeze(-1),  # (B, K, 1)
            same_type.unsqueeze(-1),            # (B, K, 1)
            res_embed,                          # (B, K, 16)
        ], dim=-1)  # (B, K, 18)

        ctx = self.context_proj(context_input)  # (B, K, hidden)

        # ========== Multi-head attention over retrieved neighbors ==========
        Q = self.attn_q(q).unsqueeze(1)  # (B, 1, hidden)
        K_attn = self.attn_k(ctx)        # (B, K, hidden)
        V = self.attn_v(ctx)             # (B, K, hidden)

        # Reshape for multi-head attention
        Q = Q.view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)
        K_attn = K_attn.view(B, K, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, K, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        attn_scores = torch.matmul(Q, K_attn.transpose(-2, -1)) * self.scale

        # Mask invalid neighbors
        attn_mask = ~retrieved_valid.unsqueeze(1).unsqueeze(2)
        attn_scores = attn_scores.masked_fill(attn_mask, -1e4)

        attn_weights = F.softmax(attn_scores, dim=-1)

        # Apply attention to values
        attn_out = torch.matmul(attn_weights, V)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, self.hidden_dim)

        # ========== Compute per-neighbor, per-shift weights ==========
        shift_weight_bias = self.to_shift_weights(attn_out)

        base_weights = retrieved_distances.unsqueeze(-1).expand(-1, -1, S)
        same_type_bonus = same_type.unsqueeze(-1) * 0.5

        weights = base_weights + same_type_bonus
        weights = weights + shift_weight_bias.unsqueeze(1) * 0.1

        # Mask: must be valid retrieval AND have valid shift for this type
        valid_mask = retrieved_valid.unsqueeze(-1) & retrieved_shift_masks  # (B, K, S)

        weights = weights.masked_fill(~valid_mask, -1e4)
        weights = F.softmax(weights, dim=1)

        # ========== Compute transferred shifts ==========
        transferred = (weights * retrieved_shifts).sum(dim=1)
        transferred = transferred * self.shift_scale

        # Handle case where no valid retrieval
        fallback = self.fallback_shift.unsqueeze(0).expand(B, -1)
        transferred = torch.where(any_valid.unsqueeze(-1), transferred, fallback)

        # ========== Compute PER-SHIFT trust statistics ==========
        # These are just statistics - no gradients needed (saves massive autograd overhead)
        with torch.no_grad():
            valid_float = retrieved_valid.float()  # (B, K)
            valid_mask_float = valid_mask.float()  # (B, K, S)

            # Per-shift coverage: fraction of valid neighbors that have each shift
            per_shift_count = valid_mask_float.sum(dim=1)  # (B, S)
            per_shift_coverage = per_shift_count / (K + 1e-8)  # (B, S)

            # Per-shift variance: variance of retrieved shifts (masked)
            masked_shifts = retrieved_shifts * valid_mask_float
            per_shift_mean = masked_shifts.sum(dim=1) / (per_shift_count + 1e-8)  # (B, S)
            shift_diff_sq = (retrieved_shifts - per_shift_mean.unsqueeze(1)) ** 2
            shift_diff_sq = shift_diff_sq * valid_mask_float
            per_shift_var = shift_diff_sq.sum(dim=1) / (per_shift_count + 1e-8)  # (B, S)
            per_shift_var_norm = torch.log1p(per_shift_var).clamp(0, 5) / 5  # (B, S)

            # Per-shift mean distance
            dist_expanded = retrieved_distances.unsqueeze(-1).expand(-1, -1, S)  # (B, K, S)
            masked_dist = dist_expanded * valid_mask_float
            per_shift_mean_dist = masked_dist.sum(dim=1) / (per_shift_count + 1e-8)  # (B, S)

            # Per-shift same-type coverage
            same_type_expanded = same_type.unsqueeze(-1).expand(-1, -1, S)  # (B, K, S)
            same_type_with_shift = same_type_expanded * valid_mask_float
            per_shift_same_type = same_type_with_shift.sum(dim=1) / (per_shift_count + 1e-8)  # (B, S)

            # Stack per-shift stats: (B, S, 4)
            per_shift_stats = torch.stack([
                per_shift_coverage,
                per_shift_var_norm,
                per_shift_mean_dist,
                per_shift_same_type,
            ], dim=-1)  # (B, S, 4)

            # ========== Compute global statistics ==========
            n_valid = valid_float.sum(dim=1, keepdim=True) / K  # (B, 1)
            n_same_type = (same_type * valid_float).sum(dim=1, keepdim=True) / K  # (B, 1)

            masked_dist_global = retrieved_distances.masked_fill(~retrieved_valid, 0.0)
            mean_dist = masked_dist_global.sum(dim=1, keepdim=True) / (valid_float.sum(dim=1, keepdim=True) + 1e-8)
            max_dist = masked_dist_global.max(dim=1, keepdim=True)[0]

            global_stats = torch.cat([mean_dist, max_dist, n_valid, n_same_type], dim=-1)  # (B, 4)

        # ========== Compute trust with per-shift awareness ==========
        # Query residue embedding
        query_res_embed = self.trust_residue_embed(query_residue_code)  # (B, 32)

        # Project query encoding for trust
        trust_query = self.trust_query_proj(query_encoding)  # (B, hidden)

        # Build trust context
        trust_context = torch.cat([
            trust_query,        # (B, hidden)
            query_res_embed,    # (B, 32)
            global_stats,       # (B, 4)
        ], dim=-1)  # (B, hidden+32+4)

        trust_context = self.trust_context_proj(trust_context)  # (B, hidden)

        # Compute per-shift trust by combining context with per-shift stats
        trust_context_expanded = trust_context.unsqueeze(1).expand(-1, S, -1)  # (B, S, hidden)
        trust_input = torch.cat([trust_context_expanded, per_shift_stats], dim=-1)  # (B, S, hidden+4)

        trust = self.trust_per_shift(trust_input).squeeze(-1)  # (B, S)

        # Zero trust if no valid retrieval
        trust = torch.where(any_valid.unsqueeze(-1), trust, torch.zeros_like(trust))

        # Zero trust for shifts with no coverage
        has_any_neighbor = per_shift_count > 0
        trust = torch.where(has_any_neighbor, trust, torch.zeros_like(trust))

        return transferred, trust


# ============================================================================
# MODIFIED: Full Model with Retrieval + Physics Features
# ============================================================================

class ShiftPredictorWithRetrieval(nn.Module):
    """
    Chemical shift prediction with retrieval augmentation and physics features.

    MODIFIED from original:
    - Integrates PhysicsFeatureEncoder (64-dim) into the base encoder
    - base_encoder_dim = cnn_out_dim + spatial_hidden + 64
    - forward() accepts physics_features tensor
    - Uses config constants for defaults

    Architecture:
    1. Base encoder: Distance attention + CNN + Spatial attention
    2. Physics encoder: MLP over ring currents, HSE, H-bonds
    3. Retrieval branch: Cross-attention to retrieved neighbors + direct transfer
    4. Fusion: Combine structural, physics, and retrieval features
    5. Per-shift prediction heads
    """

    def __init__(
        self,
        # Original model params
        n_atom_types: int,
        n_residue_types: int = N_RESIDUE_TYPES,
        n_ss_types: int = N_SS_TYPES,
        n_mismatch_types: int = N_MISMATCH_TYPES,
        n_dssp: int = len(DSSP_COLS),
        n_shifts: int = 6,
        n_physics: int = 28,
        dist_attn_embed: int = DIST_ATTN_EMBED,
        dist_attn_hidden: int = DIST_ATTN_HIDDEN,
        cnn_channels: list = None,
        kernel: int = KERNEL_SIZE,
        input_dropout: float = INPUT_DROPOUT,
        layer_dropouts: list = None,
        head_dropout: float = HEAD_DROPOUT,
        spatial_hidden: int = SPATIAL_ATTN_HIDDEN,
        k_spatial: int = 5,
        # Retrieval params
        retrieval_hidden: int = RETRIEVAL_HIDDEN,
        retrieval_heads: int = RETRIEVAL_HEADS,
        retrieval_dropout: float = RETRIEVAL_DROPOUT,
        use_direct_transfer: bool = True,
        use_query_conditioned_transfer: bool = True,
        use_random_coil: bool = True,
        shift_cols: list = None,
    ):
        super().__init__()

        cnn_channels = cnn_channels or list(CNN_CHANNELS)
        layer_dropouts = layer_dropouts or list(LAYER_DROPOUTS)

        self.n_shifts = n_shifts
        self.n_residue_types = n_residue_types
        self.use_direct_transfer = use_direct_transfer
        self.retrieval_dropout_rate = retrieval_dropout

        # ========== Base encoder (unchanged) ==========
        self.distance_attention = DistanceAttentionPerPosition(
            n_atom_types=n_atom_types,
            embed_dim=dist_attn_embed,
            hidden_dim=dist_attn_hidden,
            dropout=0.25
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

        cnn_input_dim = dist_attn_hidden + 64 + 32 + 16 + 16 + dssp_dim

        self.input_norm = nn.LayerNorm(cnn_input_dim)
        self.input_dropout = nn.Dropout(input_dropout)

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
            dropout=0.30
        )

        # ========== NEW: Physics feature encoder ==========
        self.physics_encoder = PhysicsFeatureEncoder(
            n_physics=n_physics,
            hidden_dim=64,
            dropout=0.2,
        )
        physics_dim = self.physics_encoder.out_dim  # 64

        # base_encoder_dim now includes physics
        base_encoder_dim = cnn_out_dim + spatial_hidden + physics_dim

        # ========== Retrieval components ==========
        self.retrieval_cross_attn = RetrievalCrossAttention(
            query_dim=base_encoder_dim,
            n_shifts=n_shifts,
            n_residue_types=n_residue_types,
            hidden_dim=retrieval_hidden,
            n_heads=retrieval_heads,
            dropout=retrieval_dropout,
        )

        if use_direct_transfer:
            if use_query_conditioned_transfer:
                # Use learned query-conditioned transfer with RC correction
                self.shift_transfer = QueryConditionedTransfer(
                    query_dim=base_encoder_dim,
                    n_shifts=n_shifts,
                    n_residue_types=n_residue_types,
                    hidden_dim=retrieval_hidden,
                    n_heads=retrieval_heads,
                    dropout=retrieval_dropout,
                    use_random_coil=use_random_coil,
                    shift_cols=shift_cols,
                )
            else:
                # Fall back to simple weighted average
                self.shift_transfer = RetrievalShiftTransfer(
                    n_shifts=n_shifts,
                    n_residue_types=n_residue_types,
                )

        # ========== Fusion and prediction ==========
        # Fused dimension: base_encoder + retrieval_cross_attn
        fused_dim = base_encoder_dim + retrieval_hidden

        if use_direct_transfer:
            # Also include transferred shifts and confidence
            fused_dim += n_shifts * 2  # transferred + confidence

        # Prediction heads (one per shift type)
        self.shift_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fused_dim, 256),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(256, 128),
                nn.GELU(),
                nn.Dropout(head_dropout),
                nn.Linear(128, 1)
            ) for _ in range(n_shifts)
        ])

        # Initialize output layers
        for head in self.shift_heads:
            with torch.no_grad():
                head[-1].weight.zero_()
                head[-1].bias.zero_()

    def forward(
        self,
        # Original structural inputs
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
        # NEW: Physics features
        physics_features=None,
        # Allow forwards/backwards compatibility with extra fields
        **kwargs,
    ):
        """Forward pass with retrieval augmentation and physics features."""
        B = distances.size(0)

        # ========== Base encoder (unchanged) ==========
        dist_emb = self.distance_attention(
            atom1_idx, atom2_idx, distances, dist_mask
        )

        res_emb = self.residue_embed(residue_idx)
        ss_emb = self.ss_embed(ss_idx)
        mismatch_emb = self.mismatch_embed(mismatch_idx)
        valid_emb = self.valid_embed(is_valid.unsqueeze(-1))

        if self.dssp_proj is not None:
            dssp_emb = self.dssp_proj(dssp_features)
            x = torch.cat([dist_emb, res_emb, ss_emb, mismatch_emb, valid_emb, dssp_emb], dim=-1)
        else:
            x = torch.cat([dist_emb, res_emb, ss_emb, mismatch_emb, valid_emb], dim=-1)

        x = self.input_dropout(self.input_norm(x))
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)

        center_idx = x.size(1) // 2
        x_center = x[:, center_idx, :]

        x_spatial = self.spatial_attention(
            neighbor_res_idx, neighbor_ss_idx,
            neighbor_dist, neighbor_seq_sep, neighbor_angles,
            neighbor_valid
        )

        # ========== NEW: Physics encoding ==========
        if physics_features is None:
            # Backward compatibility: zeros if not provided
            physics_features = torch.zeros(
                B, self.physics_encoder.mlp[0].in_features,
                device=x_center.device, dtype=x_center.dtype,
            )
        x_physics = self.physics_encoder(physics_features)  # (B, 64)

        base_encoding = torch.cat([x_center, x_spatial, x_physics], dim=-1)  # (B, base_encoder_dim)

        # ========== Retrieval branch ==========
        # Determine retrieval dropout rate (only during training)
        ret_dropout = self.retrieval_dropout_rate if self.training else 0.0

        # Cross-attention to retrieved examples
        retrieval_features = self.retrieval_cross_attn(
            query=base_encoding,
            query_residue_code=query_residue_code,
            retrieved_shifts=retrieved_shifts,
            retrieved_shift_masks=retrieved_shift_masks,
            retrieved_residue_codes=retrieved_residue_codes,
            retrieved_distances=retrieved_distances,
            retrieved_valid=retrieved_valid,
            retrieval_dropout=ret_dropout,
        )  # (B, retrieval_hidden)

        # Direct transfer of shifts (query-conditioned if enabled)
        if self.use_direct_transfer:
            transferred_shifts, transfer_confidence = self.shift_transfer(
                query_residue_code=query_residue_code,
                retrieved_shifts=retrieved_shifts,
                retrieved_shift_masks=retrieved_shift_masks,
                retrieved_residue_codes=retrieved_residue_codes,
                retrieved_distances=retrieved_distances,
                retrieved_valid=retrieved_valid,
                query_encoding=base_encoding,
            )  # (B, n_shifts), (B, n_shifts)

            # Fuse everything
            fused = torch.cat([
                base_encoding,
                retrieval_features,
                transferred_shifts,
                transfer_confidence,
            ], dim=-1)
        else:
            fused = torch.cat([base_encoding, retrieval_features], dim=-1)

        # ========== Prediction ==========
        predictions = torch.stack(
            [head(fused).squeeze(-1) for head in self.shift_heads],
            dim=-1
        )

        predictions = torch.where(torch.isnan(predictions), torch.zeros_like(predictions), predictions)

        return predictions


# ============================================================================
# Factory function
# ============================================================================

def create_model(
    n_atom_types: int,
    n_shifts: int = 6,
    n_physics: int = 28,
    shift_cols: list = None,
    use_random_coil: bool = True,
    **kwargs,
) -> ShiftPredictorWithRetrieval:
    """Create model with sensible defaults from config.

    Args:
        n_atom_types: Number of unique atom types in distance columns
        n_shifts: Number of chemical shift types to predict
        n_physics: Number of physics feature dimensions
        shift_cols: List of shift column names (for RC correction lookup)
        use_random_coil: Whether to apply random coil correction in transfer
        **kwargs: Override any ShiftPredictorWithRetrieval parameter

    Returns:
        Configured ShiftPredictorWithRetrieval instance
    """
    return ShiftPredictorWithRetrieval(
        n_atom_types=n_atom_types,
        n_shifts=n_shifts,
        n_physics=n_physics,
        shift_cols=shift_cols,
        use_random_coil=use_random_coil,
        **kwargs,
    )
