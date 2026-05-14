"""
Industrial AI — Remaining Useful Life (RUL) Prediction Models

Implements production-grade deep learning architectures for RUL estimation:
1. AttentionLSTM — Bidirectional LSTM with multi-head attention
2. TemporalConvNetwork (TCN) — Dilated causal convolutions
3. TemporalFusionTransformer (TFT) — Interpretable multi-horizon forecasting
4. BayesianLSTM — MC-Dropout uncertainty estimation

All models trained on NASA C-MAPSS dataset conventions.
Piece-wise linear RUL target: min(actual_RUL, RUL_CAP=125)

Mathematical Notes
------------------
NASA Scoring Function:
    s_i = exp(-RUL_i/13) - 1   if RUL_i < 0  (late predictions)
    s_i = exp(RUL_i/10) - 1    if RUL_i >= 0 (early predictions)
    Score = Σ exp(s_i)

Asymmetric — penalizes late predictions more than early ones.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ─────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────

@dataclass
class RULPrediction:
    """Output of RUL prediction model."""
    predicted_rul: float            # Cycles remaining
    uncertainty_lower: float        # Lower confidence bound (5th percentile)
    uncertainty_upper: float        # Upper confidence bound (95th percentile)
    epistemic_uncertainty: float    # Model uncertainty
    aleatoric_uncertainty: float    # Data noise uncertainty
    health_index: float             # 0=new, 1=failed
    degradation_rate: float         # Rate of health deterioration
    failure_probability_30d: float  # P(failure within 30 cycles)
    attention_weights: np.ndarray   # Temporal attention heatmap


# ─────────────────────────────────────────────────────────────────
# Building Blocks
# ─────────────────────────────────────────────────────────────────

class MultiHeadTemporalAttention(nn.Module):
    """
    Multi-head attention for temporal sequences.
    
    A(Q,K,V) = softmax(QKᵀ/√d_k) V
    
    Multi-head: concat(head_1, ..., head_h) W_O
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        causal: bool = True,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, f"d_model {d_model} not divisible by n_heads {n_heads}"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.causal = causal
        self.scale = math.sqrt(self.d_k)
        
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        B, T, _ = x.shape
        
        Q = self.W_Q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if self.causal:
            causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), -1e9)
        
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2), -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(B, T, self.d_model)
        output = self.W_O(context)
        
        # Average attention across heads for interpretability
        avg_attn = attn_weights.mean(dim=1)
        
        return output, avg_attn


class TemporalConvBlock(nn.Module):
    """
    Dilated causal convolution block (TCN building block).
    
    Receptive field: 1 + 2*(kernel_size-1)*2^level
    Handles long-range temporal dependencies efficiently.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        
        # Causal padding to maintain sequence length
        self.padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=self.padding,
        ))
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(
            out_channels, out_channels, kernel_size,
            dilation=dilation, padding=self.padding,
        ))
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        
        # Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self._init_weights()
    
    def _init_weights(self) -> None:
        nn.init.normal_(self.conv1.weight, 0, 0.01)
        nn.init.normal_(self.conv2.weight, 0, 0.01)
    
    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, T)
        residual = x if self.downsample is None else self.downsample(x)
        
        out = self.conv1(x)
        out = out[:, :, :-self.padding]  # Remove causal padding
        out = out.transpose(1, 2)       # (B, T, C) for LayerNorm
        out = self.norm1(out).transpose(1, 2)
        out = self.activation(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = out[:, :, :-self.padding]
        out = out.transpose(1, 2)
        out = self.norm2(out).transpose(1, 2)
        out = self.activation(out)
        out = self.dropout(out)
        
        return self.activation(out + residual)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer models."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer("pe", pe)
    
    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ─────────────────────────────────────────────────────────────────
# Model 1: Attention LSTM
# ─────────────────────────────────────────────────────────────────

class AttentionLSTMRUL(nn.Module):
    """
    Bidirectional LSTM with multi-head temporal attention for RUL prediction.
    
    Architecture:
        Input → LayerNorm → BiLSTM × n_layers → MultiHead Attention
              → Dense Head → RUL (scalar)
    
    Parameters
    ----------
    n_features : int
        Number of sensor/feature inputs
    seq_len : int
        Sequence length (time window)
    hidden_size : int
        LSTM hidden state dimension
    n_layers : int
        Number of LSTM layers
    n_heads : int
        Number of attention heads
    dropout : float
        Dropout probability
    """
    
    def __init__(
        self,
        n_features: int = 14,
        seq_len: int = 30,
        hidden_size: int = 128,
        n_layers: int = 3,
        n_heads: int = 8,
        dropout: float = 0.2,
        use_bidirectional: bool = True,
    ) -> None:
        super().__init__()
        
        self.n_features = n_features
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        
        # Input normalization
        self.input_norm = nn.LayerNorm(n_features)
        
        # Feature projection
        self.input_proj = nn.Linear(n_features, hidden_size)
        
        # Bidirectional LSTM stack
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            bidirectional=use_bidirectional,
            batch_first=True,
        )
        lstm_out_dim = hidden_size * 2 if use_bidirectional else hidden_size
        
        # Projection to attention dimension
        self.lstm_proj = nn.Linear(lstm_out_dim, hidden_size)
        self.lstm_norm = nn.LayerNorm(hidden_size)
        
        # Temporal attention
        self.attention = MultiHeadTemporalAttention(
            d_model=hidden_size,
            n_heads=n_heads,
            dropout=dropout,
            causal=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_size)
        
        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.ffn_norm = nn.LayerNorm(hidden_size)
        
        # RUL prediction head
        self.rul_head = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softplus(),  # RUL ≥ 0
        )
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.orthogonal_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
            elif "weight" in name and param.dim() == 2:
                nn.init.xavier_uniform_(param)
    
    def forward(
        self,
        x: Tensor,
        return_attention: bool = False,
    ) -> tuple[Tensor, Tensor] | Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        x : Tensor, shape (B, T, F)
            Input sequence: B=batch, T=time, F=features
        return_attention : bool
            If True, return (rul, attention_weights)
            
        Returns
        -------
        rul : Tensor, shape (B, 1)
        attention_weights : Tensor, shape (B, T, T)  [if return_attention]
        """
        B, T, F = x.shape
        
        # Normalize and project input
        x = self.input_norm(x)
        x = self.input_proj(x)                         # (B, T, H)
        
        # LSTM
        lstm_out, _ = self.lstm(x)                     # (B, T, H*2 or H)
        lstm_out = self.lstm_proj(lstm_out)             # (B, T, H)
        x = self.lstm_norm(lstm_out + x)               # Residual
        
        # Temporal attention (Pre-LN style)
        attn_out, attn_weights = self.attention(x)     # (B, T, H), (B, T, T)
        x = self.attn_norm(x + attn_out)
        
        # Feed-forward
        x = self.ffn_norm(x + self.ffn(x))
        
        # Pool: use last time step (causal)
        context = x[:, -1, :]                         # (B, H)
        
        rul = self.rul_head(context)                   # (B, 1)
        
        if return_attention:
            return rul, attn_weights
        return rul
    
    def predict_with_uncertainty(
        self,
        x: Tensor,
        n_samples: int = 50,
    ) -> dict[str, Tensor]:
        """
        MC-Dropout uncertainty estimation.
        
        Runs forward pass N times with dropout enabled.
        Returns mean, std, and confidence intervals.
        """
        self.train()  # Enable dropout
        
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x)
                predictions.append(pred)
        
        self.eval()
        
        preds = torch.stack(predictions, dim=0)  # (N, B, 1)
        mean = preds.mean(dim=0)
        std = preds.std(dim=0)
        
        return {
            "mean": mean,
            "std": std,
            "lower_5": preds.quantile(0.05, dim=0),
            "upper_95": preds.quantile(0.95, dim=0),
            "epistemic_uncertainty": std,
        }


# ─────────────────────────────────────────────────────────────────
# Model 2: Temporal Convolutional Network
# ─────────────────────────────────────────────────────────────────

class TCNRULPredictor(nn.Module):
    """
    Temporal Convolutional Network for RUL prediction.
    
    Uses exponentially dilated causal convolutions:
    Receptive field = 1 + 2·(K-1)·(1 + 2 + 4 + ... + 2^{L-1})
                    = 1 + 2·(K-1)·(2^L - 1)
    
    For K=3, L=8: RF = 1 + 4·255 = 1021 steps
    """
    
    def __init__(
        self,
        n_features: int = 14,
        n_channels: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        
        if n_channels is None:
            n_channels = [64, 64, 128, 128, 256, 256, 256, 128]
        
        self.n_features = n_features
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, n_channels[0]),
            nn.LayerNorm(n_channels[0]),
        )
        
        # TCN blocks with exponential dilation
        self.tcn_blocks = nn.ModuleList()
        in_ch = n_channels[0]
        for i, out_ch in enumerate(n_channels):
            dilation = 2 ** i
            self.tcn_blocks.append(
                TemporalConvBlock(in_ch, out_ch, kernel_size, dilation, dropout)
            )
            in_ch = out_ch
        
        # Global context via attention pooling
        self.attn_pool = nn.MultiheadAttention(
            embed_dim=n_channels[-1],
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )
        
        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(n_channels[-1], 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Softplus(),
        )
    
    def forward(self, x: Tensor) -> Tensor:
        B, T, F = x.shape
        
        # Project features
        x_proj = self.input_proj(x)         # (B, T, C)
        
        # TCN requires (B, C, T)
        x_tcn = x_proj.transpose(1, 2)
        for block in self.tcn_blocks:
            x_tcn = block(x_tcn)
        
        x_out = x_tcn.transpose(1, 2)      # (B, T, C)
        
        # Attention pooling
        query = x_out[:, -1:, :]           # Last timestep as query
        context, _ = self.attn_pool(query, x_out, x_out)
        context = context.squeeze(1)       # (B, C)
        
        return self.head(context)


# ─────────────────────────────────────────────────────────────────
# Model 3: Simplified Temporal Fusion Transformer
# ─────────────────────────────────────────────────────────────────

class TFTRULPredictor(nn.Module):
    """
    Temporal Fusion Transformer for interpretable RUL prediction.
    
    Based on Lim et al. (2021) "Temporal Fusion Transformers for 
    Interpretable Multi-horizon Time Series Forecasting"
    
    Key components:
    - Variable Selection Networks (VSN) for feature importance
    - Gated Residual Networks (GRN) for non-linear processing
    - LSTM encoder for temporal state propagation
    - Multi-head attention for long-range dependencies
    - Quantile outputs for uncertainty
    """
    
    def __init__(
        self,
        n_features: int = 14,
        hidden_size: int = 128,
        n_heads: int = 4,
        n_lstm_layers: int = 2,
        dropout: float = 0.1,
        seq_len: int = 30,
        quantiles: list[float] | None = None,
    ) -> None:
        super().__init__()
        
        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]
        
        self.hidden_size = hidden_size
        self.n_features = n_features
        self.quantiles = quantiles
        
        # Variable Selection Network
        self.vsn = VariableSelectionNetwork(n_features, hidden_size, dropout)
        
        # LSTM encoder
        self.lstm_encoder = nn.LSTM(
            hidden_size, hidden_size, n_lstm_layers,
            dropout=dropout, batch_first=True,
        )
        
        # Gated residual connections
        self.pre_attn_grn = GatedResidualNetwork(hidden_size, hidden_size, dropout)
        
        # Temporal self-attention
        self.attention = MultiHeadTemporalAttention(hidden_size, n_heads, dropout)
        self.attn_grn = GatedResidualNetwork(hidden_size, hidden_size, dropout)
        
        # Position-wise feed-forward
        self.ff_grn = GatedResidualNetwork(hidden_size, hidden_size, dropout)
        
        # Quantile output heads
        self.quantile_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, 32),
                nn.GELU(),
                nn.Linear(32, 1),
                nn.Softplus(),
            )
            for _ in quantiles
        ])
        
        self.norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """
        Returns dict: {'q10': ..., 'q50': ..., 'q90': ..., 'vsn_weights': ...}
        """
        # Variable selection
        x_selected, vsn_weights = self.vsn(x)      # (B, T, H)
        
        # LSTM encoding
        x_enc, _ = self.lstm_encoder(x_selected)   # (B, T, H)
        
        # Pre-attention GRN
        x_pre = self.pre_attn_grn(x_enc)
        
        # Temporal attention
        x_attn, attn_weights = self.attention(x_pre)
        x_post = self.attn_grn(x_pre + x_attn)
        
        # Feed-forward
        x_ff = self.ff_grn(x_post)
        x_final = self.norm(x_ff)
        
        # Use last timestep
        context = x_final[:, -1, :]                # (B, H)
        
        outputs = {
            f"q{int(q*100)}": head(context)
            for q, head in zip(self.quantiles, self.quantile_heads)
        }
        outputs["vsn_weights"] = vsn_weights
        outputs["attention_weights"] = attn_weights
        
        return outputs
    
    def predict_rul(self, x: Tensor) -> RULPrediction:
        """High-level prediction with full uncertainty output."""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
        
        q50 = outputs.get("q50", outputs.get("q50", torch.tensor([0.0])))
        q10 = outputs.get("q10", q50 * 0.8)
        q90 = outputs.get("q90", q50 * 1.2)
        
        rul = float(q50.squeeze())
        
        return RULPrediction(
            predicted_rul=rul,
            uncertainty_lower=float(q10.squeeze()),
            uncertainty_upper=float(q90.squeeze()),
            epistemic_uncertainty=float((q90 - q10).squeeze() / 4),
            aleatoric_uncertainty=float((q90 - q10).squeeze() / 4),
            health_index=float(1.0 / (1.0 + rul / 125.0)),
            degradation_rate=float(1.0 / max(rul, 1.0)),
            failure_probability_30d=float(1.0 - np.exp(-30.0 / max(rul, 1.0))),
            attention_weights=outputs.get("attention_weights", torch.zeros(1, 1, 1)).squeeze().numpy(),
        )


# ─────────────────────────────────────────────────────────────────
# TFT Sub-components
# ─────────────────────────────────────────────────────────────────

class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network — core TFT building block.
    
    GRN(a, c) = LayerNorm(a + GLU(η₁, η₂))
    where η₁, η₂ = ELU(W₁a + b₁ + W_ca·c + b₂), W₂a + b₂
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float = 0.1,
        output_size: int | None = None,
    ) -> None:
        super().__init__()
        
        if output_size is None:
            output_size = hidden_size
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size * 2)  # GLU requires 2x
        self.fc_skip = nn.Linear(input_size, output_size) if input_size != output_size else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_size)
        self.output_size = output_size
    
    def forward(self, x: Tensor, context: Tensor | None = None) -> Tensor:
        # ELU activation path
        eta = F.elu(self.fc1(x))
        if context is not None:
            eta = eta + context
        
        # Gated Linear Unit
        gate_input = self.fc2(eta)
        value, gate = gate_input.chunk(2, dim=-1)
        gated = value * torch.sigmoid(gate)
        gated = self.dropout(gated)
        
        # Residual + LayerNorm
        skip = self.fc_skip(x)
        if gated.shape[-1] != skip.shape[-1]:
            gated = gated[..., :skip.shape[-1]]
        
        return self.norm(skip + gated)


class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network — learns soft variable weights per timestep.
    
    Allows the model to select which sensor inputs are most relevant
    for RUL prediction at each time point.
    """
    
    def __init__(
        self,
        n_variables: int,
        hidden_size: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.n_variables = n_variables
        self.hidden_size = hidden_size
        
        # Per-variable GRN
        self.variable_grns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for _ in range(n_variables)
        ])
        
        # Softmax selection weights
        self.selection_grn = GatedResidualNetwork(
            input_size=n_variables * hidden_size,
            hidden_size=hidden_size,
            dropout=dropout,
            output_size=n_variables,
        )
    
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        x: (B, T, F)
        Returns: (selected, weights) where selected=(B,T,H), weights=(B,T,F)
        """
        B, T, F = x.shape
        
        # Process each variable
        var_embeddings = []
        for i in range(F):
            v = x[:, :, i:i+1]                     # (B, T, 1)
            v = v.view(B * T, 1)
            emb = self.variable_grns[i](v)         # (B*T, H)
            emb = emb.view(B, T, self.hidden_size)
            var_embeddings.append(emb)
        
        # Flatten for selection
        stacked = torch.stack(var_embeddings, dim=-1)   # (B, T, H, F)
        flat = stacked.permute(0, 1, 3, 2).reshape(B * T, F * self.hidden_size)
        
        # Compute selection weights
        weights = torch.softmax(
            self.selection_grn(flat).view(B, T, F),
            dim=-1,
        )
        
        # Weighted combination
        weights_exp = weights.unsqueeze(2)              # (B, T, 1, F)
        selected = (stacked * weights_exp.permute(0, 1, 3, 2).unsqueeze(-2)).sum(dim=-1)
        # Simpler: weighted sum
        selected = torch.stack(var_embeddings, dim=-1)  # (B, T, H, F)
        selected = (selected * weights.unsqueeze(2)).sum(dim=-1)  # (B, T, H)
        
        return selected, weights


# ─────────────────────────────────────────────────────────────────
# Loss Functions
# ─────────────────────────────────────────────────────────────────

class NASAScoringLoss(nn.Module):
    """
    NASA C-MAPSS asymmetric scoring loss.
    
    Penalizes late predictions (positive error = too optimistic) more
    than early predictions (negative error = too pessimistic):
    
    s_i = exp(-d_i/13) - 1   if d_i < 0  (late)
    s_i = exp(d_i/10)  - 1   if d_i >= 0 (early)
    
    Score = Σ exp(s_i)  (want to minimize)
    """
    
    def __init__(self, alpha_late: float = 13.0, alpha_early: float = 10.0) -> None:
        super().__init__()
        self.alpha_late = alpha_late
        self.alpha_early = alpha_early
    
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        d = pred.squeeze() - target.squeeze()  # predicted - actual RUL
        # d < 0: early prediction (predicted too high)
        # d > 0: late prediction (predicted too low = dangerous)
        
        s = torch.where(
            d < 0,
            torch.exp(-d / self.alpha_late) - 1,
            torch.exp(d / self.alpha_early) - 1,
        )
        return torch.sum(torch.exp(s))


class QuantileLoss(nn.Module):
    """
    Pinball loss for quantile regression.
    
    L_q(y, ŷ) = q·(y - ŷ)   if y >= ŷ
                (q-1)·(y-ŷ) if y < ŷ
    """
    
    def __init__(self, quantiles: list[float] | None = None) -> None:
        super().__init__()
        self.quantiles = quantiles or [0.1, 0.5, 0.9]
    
    def forward(
        self,
        predictions: dict[str, Tensor],
        target: Tensor,
    ) -> Tensor:
        total_loss = torch.tensor(0.0, device=target.device)
        
        for q, key in zip(self.quantiles, [f"q{int(q*100)}" for q in self.quantiles]):
            if key not in predictions:
                continue
            pred = predictions[key].squeeze()
            y = target.squeeze()
            
            error = y - pred
            loss = torch.where(
                error >= 0,
                q * error,
                (q - 1) * error,
            )
            total_loss = total_loss + loss.mean()
        
        return total_loss


class CombinedRULLoss(nn.Module):
    """
    Combined loss: MSE + NASA Score + Monotonicity penalty.
    
    Monotonicity penalty encourages RUL predictions to be non-increasing
    over time for a given unit (reflecting physical degradation).
    """
    
    def __init__(
        self,
        mse_weight: float = 1.0,
        nasa_weight: float = 0.01,
        monotonicity_weight: float = 0.1,
    ) -> None:
        super().__init__()
        self.mse_weight = mse_weight
        self.nasa_weight = nasa_weight
        self.monotonicity_weight = monotonicity_weight
        self.nasa_loss = NASAScoringLoss()
    
    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        sequence_preds: Tensor | None = None,
    ) -> dict[str, Tensor]:
        mse = F.mse_loss(pred.squeeze(), target.squeeze())
        nasa = self.nasa_loss(pred, target) / (len(pred) + 1e-6)
        
        mono_penalty = torch.tensor(0.0, device=pred.device)
        if sequence_preds is not None and len(sequence_preds) > 1:
            # Penalize increasing RUL predictions over time
            diffs = sequence_preds[1:] - sequence_preds[:-1]
            mono_penalty = torch.mean(F.relu(diffs))
        
        total = (
            self.mse_weight * mse +
            self.nasa_weight * nasa +
            self.monotonicity_weight * mono_penalty
        )
        
        return {
            "total": total,
            "mse": mse,
            "nasa_score": nasa,
            "monotonicity": mono_penalty,
        }


# ─────────────────────────────────────────────────────────────────
# Model Registry & Factory
# ─────────────────────────────────────────────────────────────────

RUL_MODEL_REGISTRY = {
    "lstm_attention": AttentionLSTMRUL,
    "tcn": TCNRULPredictor,
    "tft": TFTRULPredictor,
}


def create_rul_model(
    model_type: str,
    n_features: int = 14,
    seq_len: int = 30,
    **kwargs,
) -> nn.Module:
    """
    Factory function for RUL prediction models.
    
    Parameters
    ----------
    model_type : str
        One of: 'lstm_attention', 'tcn', 'tft'
    n_features : int
        Number of input features
    seq_len : int
        Input sequence length
    """
    if model_type not in RUL_MODEL_REGISTRY:
        raise ValueError(f"Unknown model type '{model_type}'. Choose from: {list(RUL_MODEL_REGISTRY)}")
    
    model_cls = RUL_MODEL_REGISTRY[model_type]
    return model_cls(n_features=n_features, seq_len=seq_len, **kwargs)


def compute_rul_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> dict[str, float]:
    """
    Compute comprehensive RUL evaluation metrics.
    
    Metrics:
    - RMSE, MAE, MAPE
    - NASA Score
    - PHM08 Score
    """
    pred = np.asarray(predictions).ravel()
    true = np.asarray(targets).ravel()
    
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    mae = float(np.mean(np.abs(pred - true)))
    mape = float(np.mean(np.abs((pred - true) / (true + 1e-6))) * 100)
    
    # NASA asymmetric score
    d = pred - true
    s = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    nasa_score = float(np.sum(np.exp(s)))
    
    # R² score
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    r2 = float(1 - ss_res / (ss_tot + 1e-12))
    
    # PHM08 exponential score (alternative)
    phm_score = float(np.sum(np.exp(s)))
    
    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "nasa_score": nasa_score,
        "phm08_score": phm_score,
        "r2": r2,
        "max_error": float(np.max(np.abs(pred - true))),
        "mean_error": float(np.mean(pred - true)),
    }
