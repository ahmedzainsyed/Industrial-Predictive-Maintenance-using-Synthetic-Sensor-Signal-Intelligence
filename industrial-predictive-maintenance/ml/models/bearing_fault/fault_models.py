"""
Industrial AI — Bearing Fault Diagnosis Engine

Implements three complementary architectures for bearing fault classification:
1. SpectralCNN        — 2D CNN on FFT/STFT spectrograms
2. HybridCNNLSTM      — Spatial + temporal feature fusion
3. TransformerClassifier — Self-attention based classifier

Classes: healthy | inner_race_fault | outer_race_fault | ball_fault

Mathematical Foundation
-----------------------
Convolutional feature extraction:
  h_l = σ(W_l * h_{l-1} + b_l)   (* = convolution)

Class Activation Mapping (CAM):
  CAM_c(x,y) = Σ_k w_k^c · A_k(x,y)

Grad-CAM:
  α_k^c = (1/Z) Σ_i Σ_j ∂y^c/∂A_{ij}^k
  L^c_{Grad-CAM} = ReLU(Σ_k α_k^c · A^k)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


FAULT_CLASSES = ["healthy", "inner_race_fault", "outer_race_fault", "ball_fault"]
N_CLASSES = len(FAULT_CLASSES)


@dataclass
class FaultDiagnosisResult:
    """Complete bearing fault diagnosis output."""
    predicted_class: str
    predicted_index: int
    class_probabilities: dict[str, float]
    confidence: float
    cam_heatmap: np.ndarray          # Class activation map
    grad_cam: np.ndarray             # Grad-CAM visualization
    shap_values: np.ndarray | None   # SHAP feature importance
    feature_embeddings: np.ndarray   # Penultimate layer features (for UMAP)
    severity_score: float            # 0 (healthy) to 1 (critical)


# ─────────────────────────────────────────────────────────────────
# Model 1: Spectral CNN (2D spectrogram input)
# ─────────────────────────────────────────────────────────────────

class ResidualConvBlock(nn.Module):
    """ResNet-style conv block for spectral feature extraction."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, pad, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout)
        self.activation = nn.GELU()

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
            nn.BatchNorm2d(out_channels),
        ) if in_channels != out_channels or stride != 1 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        residual = self.shortcut(x)
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        return self.activation(out + residual)


class SpectralCNNClassifier(nn.Module):
    """
    2D CNN bearing fault classifier operating on STFT spectrograms.

    Input: (B, 1, freq_bins, time_frames) — STFT magnitude spectrogram
    Output: (B, n_classes) logits
    """

    def __init__(
        self,
        n_classes: int = N_CLASSES,
        freq_bins: int = 257,
        time_frames: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.n_classes = n_classes

        # Encoder backbone
        self.encoder = nn.Sequential(
            # Block 1: 1 → 32
            ResidualConvBlock(1, 32, kernel_size=3, stride=1),
            nn.MaxPool2d(2, 2),                         # /2
            # Block 2: 32 → 64
            ResidualConvBlock(32, 64, kernel_size=3, stride=1),
            nn.MaxPool2d(2, 2),                         # /4
            # Block 3: 64 → 128
            ResidualConvBlock(64, 128, kernel_size=3, stride=1),
            nn.MaxPool2d(2, 2),                         # /8
            # Block 4: 128 → 256
            ResidualConvBlock(128, 256, kernel_size=3, stride=1),
            nn.MaxPool2d(2, 2),                         # /16
        )

        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, n_classes),
        )

        # CAM weight extractor — hooks for last conv layer
        self._feature_maps: Tensor | None = None
        self._register_cam_hook()

    def _register_cam_hook(self) -> None:
        """Register forward hook to capture last conv feature maps for CAM."""
        def hook(module, input, output):
            self._feature_maps = output.detach()

        # Hook on the last ResidualConvBlock
        last_block = list(self.encoder.children())[-2]
        last_block.register_forward_hook(hook)

    def forward(self, x: Tensor) -> Tensor:
        features = self.encoder(x)
        pooled = self.gap(features)
        return self.classifier(pooled)

    def get_cam(self, x: Tensor, class_idx: int | None = None) -> np.ndarray:
        """
        Class Activation Map — identifies discriminative spectrogram regions.
        CAM_c(f,t) = Σ_k w_k^c · A_k(f,t)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            if class_idx is None:
                class_idx = int(logits.argmax(dim=-1).item())

        # Get final FC weights for target class
        final_fc = list(self.classifier.children())[-1]
        cam_weights = final_fc.weight[class_idx].detach()  # (256,)

        if self._feature_maps is None:
            return np.zeros((16, 16))

        # GAP over feature maps, then weight
        feat = self._feature_maps[0]            # (256, H, W)
        cam = torch.einsum("c,chw->hw", cam_weights, feat)
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)

        return cam.cpu().numpy()

    def get_grad_cam(self, x: Tensor, class_idx: int | None = None) -> np.ndarray:
        """
        Grad-CAM: α_k^c = (1/Z) Σ_ij ∂y^c/∂A^k_ij
        L^c = ReLU(Σ_k α_k^c · A^k)
        """
        self.train()  # Enable gradients
        x = x.requires_grad_(True)

        # Forward
        features = self.encoder(x)
        pooled = self.gap(features)
        logits = self.classifier(pooled)

        if class_idx is None:
            class_idx = int(logits.argmax(dim=-1).item())

        # Backward on target class score
        self.zero_grad()
        logits[0, class_idx].backward()

        # Gradient of target class w.r.t. last conv feature maps
        gradients = x.grad
        if self._feature_maps is None:
            return np.zeros((16, 16))

        # Pool gradients
        feat = self._feature_maps[0]        # (C, H, W)
        alpha = feat.mean(dim=(1, 2))       # (C,) — global average pooling of gradients

        grad_cam = torch.einsum("c,chw->hw", alpha, feat)
        grad_cam = F.relu(grad_cam)

        cam_min, cam_max = grad_cam.min(), grad_cam.max()
        if cam_max > cam_min:
            grad_cam = (grad_cam - cam_min) / (cam_max - cam_min)

        self.eval()
        return grad_cam.detach().cpu().numpy()


# ─────────────────────────────────────────────────────────────────
# Model 2: Hybrid CNN-LSTM
# ─────────────────────────────────────────────────────────────────

class HybridCNNLSTMClassifier(nn.Module):
    """
    Hybrid architecture: 1D CNN for local spectral features
    followed by BiLSTM for temporal context.

    Input: (B, T, F) — time-series of feature vectors
    Output: (B, n_classes) logits
    """

    def __init__(
        self,
        n_features: int = 64,
        seq_len: int = 128,
        n_classes: int = N_CLASSES,
        cnn_channels: list[int] | None = None,
        lstm_hidden: int = 128,
        n_lstm_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        if cnn_channels is None:
            cnn_channels = [64, 128, 256]

        # 1D CNN feature extractor (operates on feature dimension)
        cnn_layers = []
        in_ch = n_features
        for out_ch in cnn_channels:
            cnn_layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
            ])
            in_ch = out_ch

        self.cnn = nn.Sequential(*cnn_layers)

        # BiLSTM temporal encoder
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=n_lstm_layers,
            bidirectional=True,
            dropout=dropout if n_lstm_layers > 1 else 0.0,
            batch_first=True,
        )

        # Attention over LSTM outputs
        lstm_dim = lstm_hidden * 2
        self.attention_q = nn.Linear(lstm_dim, lstm_dim)
        self.attention_k = nn.Linear(lstm_dim, lstm_dim)
        self.attention_v = nn.Linear(lstm_dim, lstm_dim)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(lstm_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

        self._embedding: Tensor | None = None

    def forward(self, x: Tensor, return_embedding: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        """
        x: (B, T, F) — batch × time × features
        """
        B, T, F = x.shape

        # CNN expects (B, C, T) — treat features as channels
        x_cnn = x.permute(0, 2, 1)             # (B, F, T)
        x_cnn = self.cnn(x_cnn)                # (B, C_out, T)
        x_cnn = x_cnn.permute(0, 2, 1)        # (B, T, C_out)

        # BiLSTM
        lstm_out, _ = self.lstm(x_cnn)        # (B, T, H*2)

        # Self-attention pooling
        Q = self.attention_q(lstm_out)
        K = self.attention_k(lstm_out)
        V = self.attention_v(lstm_out)
        scale = lstm_out.size(-1) ** 0.5
        attn = torch.softmax(
            torch.bmm(Q, K.transpose(1, 2)) / scale, dim=-1
        )
        context = torch.bmm(attn, V).mean(dim=1)   # (B, H*2)

        self._embedding = context.detach()

        logits = self.classifier(context)

        if return_embedding:
            return logits, context
        return logits


# ─────────────────────────────────────────────────────────────────
# Model 3: Transformer Classifier
# ─────────────────────────────────────────────────────────────────

class TransformerFaultClassifier(nn.Module):
    """
    Transformer encoder for bearing fault classification.

    Uses [CLS] token for classification (BERT-style).
    Supports both raw time-series and spectral feature inputs.
    """

    def __init__(
        self,
        n_features: int = 64,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 512,
        n_classes: int = N_CLASSES,
        max_seq_len: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.d_model = d_model

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Learnable positional encoding
        self.pos_emb = nn.Embedding(max_seq_len + 1, d_model)  # +1 for CLS

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

        # Attention weight storage for explainability
        self._attention_weights: list[Tensor] = []

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, T, F)
        Returns: (B, n_classes) logits
        """
        B, T, _ = x.shape

        # Project to model dimension
        x = self.input_proj(x)              # (B, T, D)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)     # (B, T+1, D)

        # Positional encoding
        positions = torch.arange(T + 1, device=x.device).unsqueeze(0)
        x = x + self.pos_emb(positions)

        # Transformer
        x = self.transformer(x)
        x = self.norm(x)

        # CLS token output → classify
        cls_output = x[:, 0, :]            # (B, D)
        return self.classifier(cls_output)


# ─────────────────────────────────────────────────────────────────
# Ensemble Fault Classifier
# ─────────────────────────────────────────────────────────────────

class EnsembleFaultClassifier(nn.Module):
    """
    Ensemble of CNN + CNN-LSTM + Transformer with learned weight fusion.

    Uses temperature-calibrated softmax outputs for reliable confidence.
    """

    def __init__(
        self,
        spectrogram_cnn: SpectralCNNClassifier,
        cnn_lstm: HybridCNNLSTMClassifier,
        transformer: TransformerFaultClassifier,
        temperature: float = 1.5,
    ) -> None:
        super().__init__()

        self.spectrogram_cnn = spectrogram_cnn
        self.cnn_lstm = cnn_lstm
        self.transformer = transformer
        self.temperature = nn.Parameter(torch.tensor(temperature))

        # Learnable ensemble weights
        self.ensemble_weights = nn.Parameter(torch.ones(3) / 3.0)

    def forward(
        self,
        spectrogram: Tensor,
        features: Tensor,
    ) -> dict[str, Tensor]:
        """
        spectrogram: (B, 1, F, T) for SpectralCNN
        features: (B, T, F) for CNN-LSTM and Transformer

        Returns dict with logits, probabilities, weights
        """
        logits_cnn = self.spectrogram_cnn(spectrogram)
        logits_lstm = self.cnn_lstm(features)
        logits_trans = self.transformer(features)

        # Softmax weights (ensure they sum to 1)
        w = torch.softmax(self.ensemble_weights, dim=0)

        # Temperature-scaled ensemble
        ensemble_logits = (
            w[0] * logits_cnn +
            w[1] * logits_lstm +
            w[2] * logits_trans
        ) / self.temperature

        probs = torch.softmax(ensemble_logits, dim=-1)

        return {
            "logits": ensemble_logits,
            "probabilities": probs,
            "individual_logits": {
                "cnn": logits_cnn,
                "cnn_lstm": logits_lstm,
                "transformer": logits_trans,
            },
            "ensemble_weights": w,
        }


# ─────────────────────────────────────────────────────────────────
# Anomaly Detection: LSTM-VAE
# ─────────────────────────────────────────────────────────────────

class LSTMVariationalAutoencoder(nn.Module):
    """
    LSTM-VAE for unsupervised anomaly detection.

    Reconstruction error → anomaly score.
    Higher reconstruction error = more anomalous.

    ELBO = E[log p(x|z)] - β·KL(q(z|x) || p(z))
    KL(q||p) = -0.5·Σ(1 + log σ² - μ² - σ²)
    """

    def __init__(
        self,
        n_features: int = 14,
        seq_len: int = 30,
        hidden_size: int = 64,
        latent_dim: int = 16,
        n_layers: int = 2,
        dropout: float = 0.2,
        beta: float = 1.0,
    ) -> None:
        super().__init__()

        self.n_features = n_features
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.beta = beta

        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(
            n_features, hidden_size, n_layers,
            dropout=dropout, batch_first=True,
        )

        # Latent space projections
        self.mu_proj = nn.Linear(hidden_size, latent_dim)
        self.logvar_proj = nn.Linear(hidden_size, latent_dim)

        # Decoder
        self.decoder_input = nn.Linear(latent_dim, hidden_size)
        self.decoder_lstm = nn.LSTM(
            hidden_size, hidden_size, n_layers,
            dropout=dropout, batch_first=True,
        )
        self.decoder_output = nn.Linear(hidden_size, n_features)

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Encode to latent distribution parameters."""
        _, (h_n, _) = self.encoder_lstm(x)
        h = h_n[-1]                    # Last layer hidden state
        return self.mu_proj(h), self.logvar_proj(h)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Reparameterization trick: z = μ + ε·σ"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: Tensor, seq_len: int) -> Tensor:
        """Decode latent vector to reconstructed sequence."""
        h = self.decoder_input(z)
        h = h.unsqueeze(1).expand(-1, seq_len, -1)    # (B, T, H)
        out, _ = self.decoder_lstm(h)
        return self.decoder_output(out)                # (B, T, F)

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        B, T, F = x.shape
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z, T)

        return {
            "reconstruction": x_recon,
            "mu": mu,
            "logvar": logvar,
            "z": z,
        }

    def compute_loss(self, x: Tensor, output: dict[str, Tensor]) -> dict[str, Tensor]:
        """Compute VAE ELBO loss."""
        recon_loss = F.mse_loss(output["reconstruction"], x, reduction="sum") / x.size(0)

        # KL divergence: -0.5·Σ(1 + log σ² - μ² - σ²)
        kl = -0.5 * torch.sum(
            1 + output["logvar"] - output["mu"].pow(2) - output["logvar"].exp(),
            dim=-1,
        ).mean()

        total = recon_loss + self.beta * kl

        return {"total": total, "reconstruction": recon_loss, "kl": kl}

    def anomaly_score(self, x: Tensor, n_samples: int = 20) -> Tensor:
        """
        Compute anomaly score using reconstruction error.
        Higher score = more anomalous.

        Uses MC sampling for robust score estimation.
        """
        self.train()
        scores = []
        with torch.no_grad():
            for _ in range(n_samples):
                out = self.forward(x)
                recon_error = F.mse_loss(out["reconstruction"], x, reduction="none")
                scores.append(recon_error.mean(dim=(1, 2)))

        self.eval()
        score_tensor = torch.stack(scores, dim=0)
        return score_tensor.mean(dim=0)    # (B,)


# ─────────────────────────────────────────────────────────────────
# Conformal Prediction Wrapper
# ─────────────────────────────────────────────────────────────────

class ConformalPredictionWrapper:
    """
    Conformal prediction for distribution-free uncertainty quantification.

    Provides coverage guarantees:
    P(Y ∈ C(X)) ≥ 1 - α

    where C(X) is the prediction set and α is the miscoverage rate.

    Calibration procedure:
    1. Run model on calibration set
    2. Compute nonconformity scores: s_i = 1 - ŷ_i[y_i]
    3. Find q = ⌈(n+1)(1-α)⌉/n quantile of calibration scores
    4. At test time: C(x) = {y : s(x,y) ≤ q}
    """

    def __init__(
        self,
        model: nn.Module,
        alpha: float = 0.05,
    ) -> None:
        self.model = model
        self.alpha = alpha
        self._calibration_scores: list[float] = []
        self._quantile: float | None = None
        self._calibrated = False

    def calibrate(
        self,
        calibration_loader,
        device: str = "cpu",
    ) -> float:
        """
        Calibrate on held-out calibration set.
        Returns the computed quantile threshold.
        """
        self.model.eval()
        scores = []

        with torch.no_grad():
            for batch in calibration_loader:
                x, y = batch
                x, y = x.to(device), y.to(device)
                logits = self.model(x)
                probs = torch.softmax(logits, dim=-1)
                # Nonconformity score: 1 - predicted probability of true class
                true_probs = probs[torch.arange(len(y)), y]
                nc_scores = 1 - true_probs.cpu().numpy()
                scores.extend(nc_scores.tolist())

        self._calibration_scores = scores
        n = len(scores)
        level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self._quantile = float(np.quantile(scores, min(level, 1.0)))
        self._calibrated = True

        return self._quantile

    def predict_set(
        self,
        x: Tensor,
        device: str = "cpu",
    ) -> list[list[str]]:
        """
        Return conformal prediction sets — guaranteed coverage.

        Returns list of valid class sets for each sample.
        """
        if not self._calibrated or self._quantile is None:
            raise RuntimeError("Must calibrate before predicting")

        self.model.eval()
        with torch.no_grad():
            x = x.to(device)
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        sets = []
        for prob_row in probs:
            # Include all classes where 1-p ≤ quantile (i.e., p ≥ 1-quantile)
            valid_indices = np.where(1 - prob_row <= self._quantile)[0]
            sets.append([FAULT_CLASSES[i] for i in valid_indices])

        return sets

    def predict_with_confidence(
        self,
        x: Tensor,
        device: str = "cpu",
    ) -> list[FaultDiagnosisResult]:
        """Full fault diagnosis with conformal confidence."""
        self.model.eval()
        with torch.no_grad():
            x = x.to(device)
            if hasattr(self.model, 'get_grad_cam'):
                logits = self.model(x)
                cam = self.model.get_cam(x)
                grad_cam = self.model.get_grad_cam(x.clone().requires_grad_(True))
            else:
                logits = self.model(x)
                cam = np.zeros((8, 8))
                grad_cam = np.zeros((8, 8))

            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        results = []
        for i, prob in enumerate(probs):
            pred_idx = int(np.argmax(prob))
            # Severity: 0=healthy, increases with fault confidence
            severity = 0.0 if pred_idx == 0 else float(prob[pred_idx])

            results.append(FaultDiagnosisResult(
                predicted_class=FAULT_CLASSES[pred_idx],
                predicted_index=pred_idx,
                class_probabilities={cls: float(p) for cls, p in zip(FAULT_CLASSES, prob)},
                confidence=float(np.max(prob)),
                cam_heatmap=cam,
                grad_cam=grad_cam,
                shap_values=None,
                feature_embeddings=np.zeros(128),
                severity_score=severity,
            ))

        return results


# ─────────────────────────────────────────────────────────────────
# Loss Functions
# ─────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss for class-imbalanced bearing fault data.
    FL(p_t) = -α_t·(1-p_t)^γ·log(p_t)

    Down-weights well-classified examples, focuses on hard negatives.
    """

    def __init__(
        self,
        alpha: Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction="none")
        p_t = torch.exp(-ce_loss)
        focal_loss = (1 - p_t) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing CE — improves calibration for bearing fault classifier."""

    def __init__(self, smoothing: float = 0.1) -> None:
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        n_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # Smooth target distribution
        with torch.no_grad():
            smooth_targets = torch.full_like(log_probs, self.smoothing / (n_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)

        return (-smooth_targets * log_probs).sum(dim=-1).mean()


def compute_classification_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    probabilities: np.ndarray | None = None,
) -> dict[str, float]:
    """Comprehensive classification metrics for bearing fault diagnosis."""
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        roc_auc_score, matthews_corrcoef, cohen_kappa_score,
    )

    accuracy = float(accuracy_score(targets, predictions))
    f1_macro = float(f1_score(targets, predictions, average="macro", zero_division=0))
    f1_weighted = float(f1_score(targets, predictions, average="weighted", zero_division=0))
    precision = float(precision_score(targets, predictions, average="macro", zero_division=0))
    recall = float(recall_score(targets, predictions, average="macro", zero_division=0))
    mcc = float(matthews_corrcoef(targets, predictions))
    kappa = float(cohen_kappa_score(targets, predictions))

    metrics = {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "precision_macro": precision,
        "recall_macro": recall,
        "mcc": mcc,
        "cohen_kappa": kappa,
    }

    if probabilities is not None:
        try:
            auc = float(roc_auc_score(targets, probabilities, multi_class="ovr", average="macro"))
            metrics["roc_auc_macro"] = auc
        except Exception:
            pass

    return metrics
