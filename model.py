#!/usr/bin/env python3
"""
Chemical Shift Predictor with Retrieval Augmentation.

Architecture:
  Base encoder:
    - DistanceAttentionPerPosition (atom-pair distances → 256-dim per position)
    - 5-layer residual CNN on 11-position window → 1280-dim center extraction
    - PrevNextEncoder: explicit i-1/i+1 features from window
    - SpatialNeighborAttention: K=5 spatial neighbors → 192-dim
    - base_encoder_dim = 1280 + 128 + 192 = 1600

  Retrieval pathway (adapted from V9 reranker):
    - RetrievalNeighborEncoder: per-neighbor features → (B, K, retrieval_hidden)
    - 2x SelfAttentionLayer: neighbors attend to each other
    - 3x ShiftSpecificCrossAttention: per-shift query embeddings attend to neighbors
    - DirectTransferHead: learned scoring → weighted shift average
    - Dual gating: direct_gate + retrieval_gate for graceful fallback

  Prediction:
    - struct_pred: structure-only MLP head
    - attn_pred: attention-based retrieval head
    - direct_pred: direct shift transfer
    - retrieval_pred = dg * direct_pred + (1-dg) * attn_pred
    - final_pred = rg * retrieval_pred + (1-rg) * struct_pred
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
    """Attention over k spatially proximate residues.

    Each neighbor contributes:
    - Residue type embedding
    - Secondary structure embedding
    - Continuous features (CA distance, sequence separation, phi/psi angles)
    - Structural embedding from distance features (via a shared DistanceAttentionPerPosition)
    """

    def __init__(self, n_residue_types, n_ss_types, k_neighbors=5,
                 embed_dim=64, hidden_dim=256, dropout=0.30,
                 dist_attn_hidden=None):
        super().__init__()

        self.k = k_neighbors

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
                neighbor_valid,
                neighbor_dist_embeddings=None):
        """
        Args:
            neighbor_res_idx: (B, K) residue type indices
            neighbor_ss_idx: (B, K) secondary structure indices
            neighbor_dist: (B, K) CA distances
            neighbor_seq_sep: (B, K) sequence separations
            neighbor_angles: (B, K, 4) sin/cos of phi/psi
            neighbor_valid: (B, K) validity mask
            neighbor_dist_embeddings: (B, K, dist_attn_hidden) structural embeddings from distance features
        """
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

        if self.dist_proj is not None and neighbor_dist_embeddings is not None:
            dist_emb_proj = self.dist_proj(neighbor_dist_embeddings)
            combined = torch.cat([res_emb, ss_emb, cont_emb, dist_emb_proj], dim=-1)
        else:
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
# V9 Retrieval Components: Neighbor Encoder + Self-Attention + Cross-Attention
# ============================================================================

class SafeMultiHeadAttention(nn.Module):
    """Multi-head attention with validity masking."""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)

    def forward(self, query, key, value, valid_mask=None):
        B, Lq, D = query.shape
        Lk = key.shape[1]
        H, HD = self.n_heads, self.head_dim
        Q = self.q_proj(query).view(B, Lq, H, HD).transpose(1, 2)
        K = self.k_proj(key).view(B, Lk, H, HD).transpose(1, 2)
        V = self.v_proj(value).view(B, Lk, H, HD).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        if valid_mask is not None:
            scores = scores.masked_fill(~valid_mask.unsqueeze(1).unsqueeze(2), -1e4)
        weights = self.attn_drop(F.softmax(scores, dim=-1))
        out = torch.matmul(weights, V)
        out = out.transpose(1, 2).contiguous().view(B, Lq, D)
        out = self.out_proj(out)
        if valid_mask is not None:
            out = out * valid_mask.any(dim=1).float().unsqueeze(1).unsqueeze(2)
        return out


class SelfAttentionLayer(nn.Module):
    """Pre-norm transformer self-attention layer with validity masking."""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attn = SafeMultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model * 2, d_model))
        self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, valid_mask):
        xn = self.norm1(x)
        x = x + self.drop(self.attn(xn, xn, xn, valid_mask))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class RetrievalNeighborEncoder(nn.Module):
    """Encode each retrieved neighbor into a d-dim vector.

    Uses available features: AA type, rank position, cosine similarity,
    same-AA indicator, shift values, deviation from neighbor consensus,
    and mask coverage fraction.
    """

    def __init__(self, n_residue_types, n_shifts, n_struct, d_model, k=32, dropout=0.1):
        super().__init__()
        self.aa_embed = nn.Embedding(n_residue_types + 1, 48, padding_idx=n_residue_types)
        self.rank_embed = nn.Embedding(k, 24)
        # input: aa_emb(48) + rank(24) + cos_sim(1) + same_aa(1) + shifts(S) + deviation(S) + mask_frac(1)
        #        + nbr_struct(n_struct) + struct_diff(n_struct) + struct_l2(1)
        input_dim = 48 + 24 + 1 + 1 + n_shifts + n_shifts + 1 + n_struct + n_struct + 1
        self.net = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, ret_codes, ret_distances, ret_shifts, ret_masks,
                ret_valid, query_aa, nbr_mean, ret_nbr_struct, query_struct):
        B, K, S = ret_shifts.shape
        codes = ret_codes.clamp(0, self.aa_embed.num_embeddings - 2) * ret_valid.long()
        aa_emb = self.aa_embed(codes)
        ranks = torch.arange(K, device=ret_codes.device).unsqueeze(0).expand(B, -1)
        rank_emb = self.rank_embed(ranks)
        same_aa = (ret_codes == query_aa.unsqueeze(1)).float().unsqueeze(-1)
        cos_sim = ret_distances.unsqueeze(-1)
        mf = ret_masks.float()
        masked_shifts = ret_shifts * mf
        deviation = ((ret_shifts - nbr_mean.unsqueeze(1)) * mf).clamp(-10, 10)
        mask_frac = mf.sum(dim=-1, keepdim=True) / max(S, 1)
        query_struct_exp = query_struct.unsqueeze(1).expand(-1, K, -1)
        struct_diff = (query_struct_exp - ret_nbr_struct).clamp(-10, 10)
        struct_l2 = torch.norm(struct_diff, dim=-1, keepdim=True) / 10.0
        features = torch.cat([
            aa_emb, rank_emb, cos_sim, same_aa,
            masked_shifts, deviation, mask_frac,
            ret_nbr_struct, struct_diff, struct_l2,
        ], dim=-1)
        enc = self.net(features)
        return enc * ret_valid.unsqueeze(-1).float()


class ShiftSpecificCrossAttention(nn.Module):
    """Per-shift query embeddings attend to neighbor encodings.

    Each shift type has its own learned query, allowing different shifts
    to attend to different neighbors (e.g., CA cares about different
    neighbors than N).
    """

    def __init__(self, d_model, n_shifts, n_heads=4, dropout=0.1):
        super().__init__()
        self.n_shifts = n_shifts
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.shift_embed = nn.Embedding(n_shifts, d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.cos_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, neighbor_enc, ret_valid, ret_distances):
        B, K, D = neighbor_enc.shape
        S, H, HD = self.n_shifts, self.n_heads, self.head_dim
        shift_ids = torch.arange(S, device=neighbor_enc.device)
        shift_emb = self.shift_embed(shift_ids)
        Q = self.q_proj(shift_emb).unsqueeze(0).expand(B, -1, -1)
        Kp = self.k_proj(neighbor_enc)
        Vp = self.v_proj(neighbor_enc)
        Q = Q.view(B, S, H, HD).transpose(1, 2)
        Kp = Kp.view(B, K, H, HD).transpose(1, 2)
        Vp = Vp.view(B, K, H, HD).transpose(1, 2)
        attn = torch.matmul(Q, Kp.transpose(-2, -1)) * self.scale
        attn = attn + ret_distances.unsqueeze(1).unsqueeze(2) * self.cos_scale
        attn = attn.masked_fill(~ret_valid.unsqueeze(1).unsqueeze(2), -1e4)
        attn = self.dropout(F.softmax(attn, dim=-1))
        out = torch.matmul(attn, Vp)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        out = self.out_proj(out)
        out = self.norm(out + shift_emb.unsqueeze(0))
        return out * ret_valid.any(dim=1).float().unsqueeze(1).unsqueeze(2)


class DirectTransferHead(nn.Module):
    """Learned per-neighbor per-shift scoring for direct shift transfer."""

    def __init__(self, d_model, n_shifts, dropout=0.1):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, n_shifts))
        self.temperature = nn.Parameter(torch.ones(n_shifts) * 2.0)
        self.shift_scale = nn.Parameter(torch.ones(n_shifts))
        self.shift_bias = nn.Parameter(torch.zeros(n_shifts))

    def forward(self, neighbor_enc, ret_shifts, ret_masks, ret_valid):
        B, K, S = ret_shifts.shape
        scores = self.scorer(neighbor_enc) * self.temperature
        valid_3d = ret_valid.unsqueeze(-1) & ret_masks
        scores = scores.masked_fill(~valid_3d, -1e4)
        weights = F.softmax(scores, dim=1) * valid_3d.float()
        transferred = (weights * ret_shifts).sum(dim=1)
        transferred = transferred * self.shift_scale + self.shift_bias
        return transferred * valid_3d.any(dim=1).float()


class RandomCoilCorrector(nn.Module):
    """Apply random coil correction to retrieved shifts.

    corrected[b,k,s] = RC[query_aa, s] + (shift[b,k,s] - RC[retrieved_aa, s])
    RC values are in z-score space (normalized using dataset stats).
    """

    def __init__(self, n_residue_types, n_shifts, shift_cols=None, stats=None):
        super().__init__()
        if shift_cols is None:
            shift_cols = ['ca_shift', 'cb_shift', 'c_shift', 'n_shift', 'h_shift', 'ha_shift']
        rc_np = build_rc_tensor(STANDARD_RESIDUES, shift_cols)
        if stats is not None:
            for si, col in enumerate(shift_cols):
                if col in stats and si < rc_np.shape[1]:
                    mean, std = stats[col]['mean'], stats[col]['std']
                    valid = ~np.isnan(rc_np[:, si])
                    rc_np[valid, si] = (rc_np[valid, si] - mean) / std
        rc_padded = np.full((n_residue_types + 1, n_shifts), np.nan, dtype=np.float32)
        rc_padded[:rc_np.shape[0], :rc_np.shape[1]] = rc_np
        self.register_buffer('rc_table', torch.from_numpy(rc_padded))

    def forward(self, retrieved_shifts, query_residue_code, retrieved_residue_codes,
                retrieved_shift_masks):
        query_idx = query_residue_code.clamp(0, self.rc_table.size(0) - 1)
        retrieved_idx = retrieved_residue_codes.clamp(0, self.rc_table.size(0) - 1)
        rc_query = self.rc_table[query_idx].unsqueeze(1)
        rc_retrieved = self.rc_table[retrieved_idx]
        corrected = rc_query + (retrieved_shifts - rc_retrieved)
        rc_valid = ~torch.isnan(rc_query) & ~torch.isnan(rc_retrieved)
        corrected = torch.where(rc_valid, corrected, retrieved_shifts)
        corrected = torch.where(retrieved_shift_masks, corrected, retrieved_shifts)
        return corrected


class PrevNextEncoder(nn.Module):
    """Explicitly encode previous and next residue features from the window.

    Key insight from SHIFTX2: psi(i-1) alone accounts for 18.7% of 15N
    prediction power. Instead of relying on the CNN to discover this,
    we extract i-1 and i+1 features directly.
    """

    def __init__(self, cnn_input_dim, d_out=128, dropout=0.2):
        super().__init__()
        self.prev_net = nn.Sequential(
            nn.Linear(cnn_input_dim, d_out), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_out, d_out), nn.LayerNorm(d_out))
        self.next_net = nn.Sequential(
            nn.Linear(cnn_input_dim, d_out), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_out, d_out), nn.LayerNorm(d_out))
        self.combine = nn.Sequential(
            nn.Linear(d_out * 2, d_out), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_out, d_out), nn.LayerNorm(d_out))
        self.out_dim = d_out

    def forward(self, window_features, is_valid):
        """
        Args:
            window_features: (B, W, cnn_input_dim) per-position features before CNN
            is_valid: (B, W) validity mask
        """
        center = window_features.size(1) // 2
        prev_pos = center - 1
        next_pos = center + 1

        prev_feat = self.prev_net(window_features[:, prev_pos])
        prev_feat = prev_feat * is_valid[:, prev_pos].unsqueeze(-1).float()

        next_feat = self.next_net(window_features[:, next_pos])
        next_feat = next_feat * is_valid[:, next_pos].unsqueeze(-1).float()

        return self.combine(torch.cat([prev_feat, next_feat], dim=-1))


# ============================================================================
# Full Model with Retrieval
# ============================================================================

class ShiftPredictorWithRetrieval(nn.Module):
    """Chemical shift prediction with V9-style retrieval augmentation.

    Architecture:
      Base encoder: Distance attention + CNN + PrevNext + Spatial
      Retrieval: Neighbor encoder + self-attn + shift-specific cross-attn
                 + direct transfer + dual gating
      Prediction: struct_pred, retrieval_pred blended by learned gates
    """

    def __init__(
        self,
        n_atom_types: int,
        n_residue_types: int = N_RESIDUE_TYPES,
        n_ss_types: int = N_SS_TYPES,
        n_mismatch_types: int = N_MISMATCH_TYPES,
        n_dssp: int = len(DSSP_COLS),
        n_shifts: int = 6,
        n_physics: int = 0,  # Deprecated, accepted for backward compat but ignored
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
        n_self_attn: int = 2,
        n_cross_attn: int = 3,
        k_retrieved: int = 32,
        n_struct: int = 49,
        # RC correction
        use_random_coil: bool = True,
        shift_cols: list = None,
        stats: dict = None,
        # Legacy params (accepted but ignored for backward compat)
        use_direct_transfer: bool = True,
        use_query_conditioned_transfer: bool = True,
    ):
        super().__init__()

        cnn_channels = cnn_channels or list(CNN_CHANNELS)
        layer_dropouts = layer_dropouts or list(LAYER_DROPOUTS)

        self.n_shifts = n_shifts
        self.n_residue_types = n_residue_types
        self.retrieval_dropout_rate = retrieval_dropout

        # ========== Base encoder ==========
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
        self.input_dropout_layer = nn.Dropout(input_dropout)

        cnn_layers = []
        in_ch = cnn_input_dim
        for out_ch, drop_p in zip(cnn_channels, layer_dropouts):
            cnn_layers.append(ResidualBlock1D(in_ch, out_ch, kernel))
            cnn_layers.append(nn.Dropout(drop_p))
            in_ch = out_ch
        self.cnn = nn.Sequential(*cnn_layers)
        cnn_out_dim = cnn_channels[-1]

        # PrevNextEncoder: explicit i-1/i+1 features
        self.prevnext_encoder = PrevNextEncoder(
            cnn_input_dim=cnn_input_dim,
            d_out=128,
            dropout=0.2,
        )
        prevnext_dim = self.prevnext_encoder.out_dim

        self.spatial_attention = SpatialNeighborAttention(
            n_residue_types=n_residue_types,
            n_ss_types=n_ss_types,
            k_neighbors=k_spatial,
            embed_dim=64,
            hidden_dim=spatial_hidden,
            dropout=0.30,
            dist_attn_hidden=dist_attn_hidden,
        )

        base_encoder_dim = cnn_out_dim + prevnext_dim + spatial_hidden

        # ========== Retrieval pathway (V9-style) ==========

        # RC correction (applied to retrieved shifts before transfer)
        self.rc_corrector = None
        if use_random_coil:
            self.rc_corrector = RandomCoilCorrector(
                n_residue_types, n_shifts, shift_cols, stats)

        # Neighbor encoder + self-attention
        self.neighbor_encoder = RetrievalNeighborEncoder(
            n_residue_types, n_shifts, n_struct, retrieval_hidden,
            k=k_retrieved, dropout=retrieval_dropout)

        self.self_attn_layers = nn.ModuleList([
            SelfAttentionLayer(retrieval_hidden, retrieval_heads, retrieval_dropout)
            for _ in range(n_self_attn)])

        # Shift-specific cross-attention (3 layers with residual FFN)
        self.cross_attn_layers = nn.ModuleList([
            ShiftSpecificCrossAttention(
                retrieval_hidden, n_shifts, retrieval_heads, retrieval_dropout)
            for _ in range(n_cross_attn)])
        self.cross_ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(retrieval_hidden, retrieval_hidden * 2), nn.GELU(),
                nn.Dropout(retrieval_dropout),
                nn.Linear(retrieval_hidden * 2, retrieval_hidden),
                nn.LayerNorm(retrieval_hidden))
            for _ in range(n_cross_attn)])

        # Direct transfer head
        self.direct_transfer = DirectTransferHead(
            retrieval_hidden, n_shifts, retrieval_dropout)

        # ========== Prediction heads ==========

        # Structure-only prediction
        self.struct_head = nn.Sequential(
            nn.Linear(base_encoder_dim, 512), nn.GELU(), nn.Dropout(head_dropout),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(head_dropout),
            nn.Linear(256, n_shifts))

        # Attention-based retrieval prediction (per-shift)
        # Input: shift_ctx (retrieval_hidden) + base_encoding (base_encoder_dim)
        self.attn_head = nn.Sequential(
            nn.Linear(retrieval_hidden + base_encoder_dim, retrieval_hidden),
            nn.GELU(), nn.Dropout(head_dropout),
            nn.Linear(retrieval_hidden, 1))

        # Dual gates
        # direct_gate: blend direct transfer vs attention prediction
        self.direct_gate = nn.Sequential(
            nn.Linear(retrieval_hidden + 1, 96), nn.GELU(),
            nn.Linear(96, 1), nn.Sigmoid())

        # retrieval_gate: blend retrieval vs structure-only
        self.retrieval_gate = nn.Sequential(
            nn.Linear(retrieval_hidden + base_encoder_dim + 1, 192), nn.GELU(),
            nn.Linear(192, 1), nn.Sigmoid())

        self._base_encoder_dim = base_encoder_dim

        # Initialize output layers to zero
        with torch.no_grad():
            self.struct_head[-1].weight.zero_()
            self.struct_head[-1].bias.zero_()

    def _compute_nbr_mean(self, retrieved_shifts, retrieved_shift_masks, retrieved_valid):
        """Precompute neighbor consensus (masked mean of retrieved shifts)."""
        valid_3d = retrieved_valid.unsqueeze(-1) & retrieved_shift_masks
        valid_count = valid_3d.float().sum(dim=1).clamp(min=1)
        nbr_mean = (retrieved_shifts * valid_3d.float()).sum(dim=1) / valid_count
        return nbr_mean

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
        # Retrieval inputs
        query_residue_code=None,
        retrieved_shifts=None, retrieved_shift_masks=None,
        retrieved_residue_codes=None, retrieved_distances=None, retrieved_valid=None,
        # Deprecated: accepted but ignored (for old dataset compatibility)
        physics_features=None,
        # Structural feature vectors for retrieval
        query_struct=None,
        neighbor_struct=None,
        # Extra fields for compatibility
        **kwargs,
    ):
        B = distances.size(0)

        # ========== Base encoder ==========
        dist_emb = self.distance_attention(atom1_idx, atom2_idx, distances, dist_mask)

        res_emb = self.residue_embed(residue_idx)
        ss_emb = self.ss_embed(ss_idx)
        mismatch_emb = self.mismatch_embed(mismatch_idx)
        valid_emb = self.valid_embed(is_valid.unsqueeze(-1))

        if self.dssp_proj is not None:
            dssp_emb = self.dssp_proj(dssp_features)
            window_feat = torch.cat([dist_emb, res_emb, ss_emb, mismatch_emb, valid_emb, dssp_emb], dim=-1)
        else:
            window_feat = torch.cat([dist_emb, res_emb, ss_emb, mismatch_emb, valid_emb], dim=-1)

        x = self.input_dropout_layer(self.input_norm(window_feat))

        # PrevNext encoding (from pre-CNN features at positions i-1, i+1)
        x_prevnext = self.prevnext_encoder(x, is_valid)  # (B, 128)

        # CNN over window
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)

        center_idx = x.size(1) // 2
        x_center = x[:, center_idx, :]  # (B, cnn_out_dim)

        # Spatial neighbor distance embeddings
        neighbor_dist_embeddings = None
        if neighbor_atom1_idx is not None and neighbor_distances is not None:
            K_sp = neighbor_atom1_idx.size(1)
            M_sp = neighbor_atom1_idx.size(2)
            nb_a1 = neighbor_atom1_idx.view(B * K_sp, 1, M_sp)
            nb_a2 = neighbor_atom2_idx.view(B * K_sp, 1, M_sp)
            nb_d = neighbor_distances.view(B * K_sp, 1, M_sp)
            nb_m = neighbor_dist_mask.view(B * K_sp, 1, M_sp)
            nb_dist_emb = self.distance_attention(nb_a1, nb_a2, nb_d, nb_m)
            neighbor_dist_embeddings = nb_dist_emb.squeeze(1).view(B, K_sp, -1)

        x_spatial = self.spatial_attention(
            neighbor_res_idx, neighbor_ss_idx,
            neighbor_dist, neighbor_seq_sep, neighbor_angles,
            neighbor_valid,
            neighbor_dist_embeddings=neighbor_dist_embeddings,
        )

        base_encoding = torch.cat([x_center, x_prevnext, x_spatial], dim=-1)

        # ========== Structure-only prediction ==========
        struct_pred = self.struct_head(base_encoding)  # (B, n_shifts)

        # ========== Retrieval pathway ==========
        # Apply retrieval dropout (drop ALL retrieval for some training samples)
        if self.training and self.retrieval_dropout_rate > 0:
            drop_mask = torch.rand(B, device=base_encoding.device) < self.retrieval_dropout_rate
            retrieved_valid = retrieved_valid.clone()
            retrieved_valid[drop_mask] = False

        # Apply random coil correction
        if self.rc_corrector is not None:
            retrieved_shifts = self.rc_corrector(
                retrieved_shifts, query_residue_code,
                retrieved_residue_codes, retrieved_shift_masks)

        # Neighbor consensus
        nbr_mean = self._compute_nbr_mean(
            retrieved_shifts, retrieved_shift_masks, retrieved_valid)

        # Encode neighbors (with structural context)
        K = retrieved_shifts.size(1)
        if query_struct is None:
            query_struct = torch.zeros(B, self.neighbor_encoder.net[0].in_features,
                                       device=base_encoding.device)
        if neighbor_struct is None:
            neighbor_struct = torch.zeros(B, K, query_struct.size(-1),
                                          device=base_encoding.device)
        nbr_enc = self.neighbor_encoder(
            retrieved_residue_codes, retrieved_distances,
            retrieved_shifts, retrieved_shift_masks,
            retrieved_valid, query_residue_code, nbr_mean,
            neighbor_struct, query_struct)

        # Self-attention over neighbors
        for sa_layer in self.self_attn_layers:
            nbr_enc = sa_layer(nbr_enc, retrieved_valid)

        # Shift-specific cross-attention (accumulate across layers)
        shift_ctx = None
        for cross_layer, ffn_layer in zip(self.cross_attn_layers, self.cross_ffn_layers):
            c = cross_layer(nbr_enc, retrieved_valid, retrieved_distances)
            c = ffn_layer(c) + c
            shift_ctx = c if shift_ctx is None else shift_ctx + c
        # shift_ctx: (B, n_shifts, retrieval_hidden)

        # Attention-based prediction: combine shift context with structural encoding
        S = self.n_shifts
        ctx_exp = base_encoding.unsqueeze(1).expand(-1, S, -1)
        attn_pred = self.attn_head(
            torch.cat([shift_ctx, ctx_exp], dim=-1)).squeeze(-1)  # (B, n_shifts)

        # Direct transfer prediction
        direct_pred = self.direct_transfer(
            nbr_enc, retrieved_shifts, retrieved_shift_masks, retrieved_valid)

        # ========== Dual gating ==========
        valid_3d = retrieved_valid.unsqueeze(-1) & retrieved_shift_masks
        n_valid_norm = (valid_3d.float().sum(dim=1) / max(K, 1)).unsqueeze(-1)  # (B, n_shifts, 1)

        # Direct gate: blend direct transfer vs attention prediction
        dg = self.direct_gate(
            torch.cat([shift_ctx, n_valid_norm], dim=-1)).squeeze(-1)  # (B, n_shifts)
        retrieval_pred = dg * direct_pred + (1 - dg) * attn_pred
        retrieval_pred = retrieval_pred * retrieved_valid.any(dim=1, keepdim=True).float()

        # Retrieval gate: blend retrieval vs structure-only
        rg = self.retrieval_gate(
            torch.cat([shift_ctx, ctx_exp, n_valid_norm], dim=-1)).squeeze(-1)  # (B, n_shifts)
        predictions = rg * retrieval_pred + (1 - rg) * struct_pred

        predictions = torch.where(
            torch.isnan(predictions), torch.zeros_like(predictions), predictions)

        return predictions


# ============================================================================
# Factory function
# ============================================================================

def create_model(
    n_atom_types: int,
    n_shifts: int = 6,
    shift_cols: list = None,
    use_random_coil: bool = True,
    stats: dict = None,
    n_physics: int = 0,  # Deprecated, accepted for backward compat but ignored
    **kwargs,
) -> ShiftPredictorWithRetrieval:
    """Create model with sensible defaults from config.

    Args:
        n_atom_types: Number of unique atom types in distance columns
        n_shifts: Number of chemical shift types to predict
        shift_cols: List of shift column names (for RC correction lookup)
        use_random_coil: Whether to apply random coil correction in transfer
        stats: Per-shift normalization stats (mean/std) for RC z-score conversion
        n_physics: Deprecated, ignored. Accepted for backward compatibility.
        **kwargs: Override any ShiftPredictorWithRetrieval parameter

    Returns:
        Configured ShiftPredictorWithRetrieval instance
    """
    # Remove n_physics from kwargs if caller passed it, to avoid double-passing
    kwargs.pop('n_physics', None)
    return ShiftPredictorWithRetrieval(
        n_atom_types=n_atom_types,
        n_shifts=n_shifts,
        shift_cols=shift_cols,
        use_random_coil=use_random_coil,
        stats=stats,
        **kwargs,
    )
