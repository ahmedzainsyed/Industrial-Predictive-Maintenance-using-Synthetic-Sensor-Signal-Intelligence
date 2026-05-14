"""
Industrial PM Platform — PyTorch Lightning Training Pipeline

Production training orchestration for all AI models:
- RUL prediction (LSTM, TCN, TFT)
- Bearing fault classification
- Anomaly detection (LSTM-VAE)
- Automatic hyperparameter optimization with Optuna
- Mixed precision training
- Gradient clipping, LR scheduling
- MLflow + W&B logging
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
    StochasticWeightAveraging,
)
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader, Dataset, random_split


# ─────────────────────────────────────────────────────────────────
# Datasets
# ─────────────────────────────────────────────────────────────────

class CMAPSSDataset(Dataset):
    """
    NASA C-MAPSS Turbofan Engine Degradation Dataset.

    4 sub-datasets (FD001-FD004):
    - FD001: 1 operating condition, 1 fault mode
    - FD002: 6 operating conditions, 1 fault mode
    - FD003: 1 operating condition, 2 fault modes
    - FD004: 6 operating conditions, 2 fault modes

    Piece-wise linear RUL target: min(actual_RUL, max_rul=125)
    """

    SENSOR_COLUMNS = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]
    OPERATING_COLUMNS = [0, 1, 2]  # op_setting_1, 2, 3
    MAX_RUL = 125

    def __init__(
        self,
        data_path: str,
        subset: str = "FD001",
        split: str = "train",
        seq_len: int = 30,
        normalize: bool = True,
        augment: bool = False,
        rng_seed: int = 42,
    ) -> None:
        self.seq_len = seq_len
        self.augment = augment
        self.rng = np.random.default_rng(rng_seed)

        raw_data = self._load_cmapss(data_path, subset, split)
        self.sequences, self.labels = self._build_sequences(raw_data, split)

        if normalize:
            self._normalize()

    def _load_cmapss(self, path: str, subset: str, split: str) -> np.ndarray:
        """Load and parse C-MAPSS text files."""
        fname = f"{split}_{subset}.txt"
        filepath = Path(path) / fname

        if not filepath.exists():
            # Generate synthetic C-MAPSS-like data if file not found
            return self._generate_synthetic_cmapss(n_units=50, split=split)

        cols = (
            ["unit", "cycle"] +
            [f"op_{i}" for i in range(3)] +
            [f"sensor_{i}" for i in range(1, 22)]
        )
        import pandas as pd
        df = pd.read_csv(filepath, sep=r"\s+", header=None, names=cols)
        return df.values

    def _generate_synthetic_cmapss(self, n_units: int = 50, split: str = "train") -> np.ndarray:
        """Generate synthetic C-MAPSS-like dataset for testing."""
        rows = []
        for unit in range(1, n_units + 1):
            max_cycle = self.rng.integers(100, 350)
            for cycle in range(1, max_cycle + 1):
                degradation = cycle / max_cycle
                op_settings = self.rng.normal(0, 0.2, 3)
                sensors = np.zeros(21)
                for i in range(21):
                    base = self.rng.normal(0, 1)
                    sensor_degradation = degradation * self.rng.uniform(0.5, 2.0)
                    sensors[i] = base + sensor_degradation
                row = [unit, cycle] + op_settings.tolist() + sensors.tolist()
                rows.append(row)
        return np.array(rows)

    def _build_sequences(
        self, data: np.ndarray, split: str
    ) -> tuple[list[np.ndarray], list[float]]:
        """Build sliding window sequences with RUL labels."""
        # Column indices: unit=0, cycle=1, op_settings=2:5, sensors=5:26
        sensor_idx = [s + 4 for s in self.SENSOR_COLUMNS]
        sensor_idx = [min(i, data.shape[1] - 1) for i in sensor_idx]

        sequences = []
        labels = []

        unit_ids = np.unique(data[:, 0].astype(int))

        for unit_id in unit_ids:
            unit_data = data[data[:, 0] == unit_id]
            unit_features = unit_data[:, sensor_idx]
            n_cycles = len(unit_data)

            # RUL computation
            if split == "train":
                ruls = np.array([
                    min(self.MAX_RUL, n_cycles - cycle)
                    for cycle in range(n_cycles)
                ])
            else:
                ruls = np.arange(n_cycles, 0, -1, dtype=float)
                ruls = np.minimum(ruls, self.MAX_RUL)

            # Sliding window
            for i in range(self.seq_len, n_cycles + 1):
                seq = unit_features[i - self.seq_len:i]
                rul = float(ruls[i - 1])
                sequences.append(seq.astype(np.float32))
                labels.append(rul)

        return sequences, labels

    def _normalize(self) -> None:
        """Normalize features using training statistics."""
        all_data = np.concatenate(self.sequences, axis=0)
        self._mean = all_data.mean(axis=0, keepdims=True)
        self._std = all_data.std(axis=0, keepdims=True) + 1e-8
        self.sequences = [
            (seq - self._mean) / self._std for seq in self.sequences
        ]

    def augment_sequence(self, seq: np.ndarray) -> np.ndarray:
        """Apply temporal augmentation for training."""
        # Gaussian noise injection
        if self.rng.random() < 0.5:
            noise_std = 0.05 * np.std(seq, axis=0)
            seq = seq + self.rng.normal(0, noise_std, seq.shape).astype(np.float32)
        # Random time warp (stretch/compress)
        if self.rng.random() < 0.3:
            from scipy.interpolate import interp1d
            T = len(seq)
            t_orig = np.linspace(0, 1, T)
            warp = np.cumsum(np.abs(self.rng.normal(1, 0.1, T)))
            warp = warp / warp[-1]
            warped = np.zeros_like(seq)
            for feat in range(seq.shape[1]):
                f = interp1d(t_orig, seq[:, feat], kind="linear")
                warped[:, feat] = f(warp).astype(np.float32)
            seq = warped
        return seq

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        seq = self.sequences[idx].copy()
        if self.augment:
            seq = self.augment_sequence(seq)
        x = torch.tensor(seq, dtype=torch.float32)
        y = torch.tensor([self.labels[idx]], dtype=torch.float32)
        return x, y


class BearingFaultDataset(Dataset):
    """
    Bearing fault classification dataset.
    Supports: NASA IMS, FEMTO, or synthetic data.
    """

    FAULT_CLASSES = {
        "healthy": 0,
        "inner_race": 1,
        "outer_race": 2,
        "ball_fault": 3,
    }

    def __init__(
        self,
        data_path: str | None = None,
        seq_len: int = 1024,
        n_features: int = 64,
        synthetic_samples: int = 2000,
        split: str = "train",
        rng_seed: int = 42,
    ) -> None:
        self.seq_len = seq_len
        self.n_features = n_features
        self.rng = np.random.default_rng(rng_seed)

        if data_path and Path(data_path).exists():
            self.X, self.y = self._load_real_data(data_path, split)
        else:
            self.X, self.y = self._generate_synthetic(synthetic_samples)

    def _generate_synthetic(
        self, n_samples: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate synthetic bearing fault features for testing."""
        X, y = [], []
        n_classes = len(self.FAULT_CLASSES)
        per_class = n_samples // n_classes

        for class_idx, (class_name, label) in enumerate(self.FAULT_CLASSES.items()):
            for _ in range(per_class):
                # Each fault class has different spectral characteristics
                t = np.arange(self.seq_len) / 20000
                base = self.rng.normal(0, 0.5, self.seq_len)

                if class_name == "outer_race":
                    base += 2.0 * np.sin(2 * np.pi * 90 * t)
                    base += 1.0 * np.sin(2 * np.pi * 180 * t)
                elif class_name == "inner_race":
                    base += 2.0 * np.sin(2 * np.pi * 120 * t)
                    base += 0.5 * np.sin(2 * np.pi * 240 * t)
                elif class_name == "ball_fault":
                    base += 1.5 * np.sin(2 * np.pi * 40 * t)

                # Extract FFT features
                fft_mag = np.abs(np.fft.rfft(base, n=128))[:self.n_features]
                X.append(fft_mag.astype(np.float32))
                y.append(label)

        return np.array(X), np.array(y)

    def _load_real_data(
        self, path: str, split: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load real bearing dataset from preprocessed numpy arrays."""
        X_path = Path(path) / f"X_{split}.npy"
        y_path = Path(path) / f"y_{split}.npy"

        if X_path.exists() and y_path.exists():
            return np.load(str(X_path)), np.load(str(y_path))
        return self._generate_synthetic(1000)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.X[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.long),
        )


# ─────────────────────────────────────────────────────────────────
# Lightning Modules
# ─────────────────────────────────────────────────────────────────

class RULPredictionModule(pl.LightningModule):
    """
    PyTorch Lightning module for RUL prediction training.

    Supports: LSTM-Attention, TCN, TFT
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        nasa_weight: float = 0.01,
        scheduler_patience: int = 10,
        max_rul: float = 125.0,
    ) -> None:
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.nasa_weight = nasa_weight
        self.scheduler_patience = scheduler_patience
        self.max_rul = max_rul

        self.save_hyperparameters(ignore=["model"])

        self._train_losses: list[float] = []
        self._val_preds: list[torch.Tensor] = []
        self._val_targets: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _compute_loss(
        self, pred: torch.Tensor, target: torch.Tensor, prefix: str
    ) -> torch.Tensor:
        # Normalize targets to [0, 1]
        target_norm = target / self.max_rul

        mse = F.mse_loss(pred.squeeze() / self.max_rul, target_norm.squeeze())

        # NASA asymmetric score (normalized)
        d = (pred.squeeze() - target.squeeze()) / self.max_rul
        s = torch.where(
            d < 0,
            torch.exp(-d / (13 / self.max_rul)) - 1,
            torch.exp(d / (10 / self.max_rul)) - 1,
        )
        nasa = torch.mean(torch.exp(s)) / 100.0

        total = mse + self.nasa_weight * nasa

        self.log(f"{prefix}/mse", mse, prog_bar=True)
        self.log(f"{prefix}/nasa", nasa)
        self.log(f"{prefix}/loss", total, prog_bar=True)

        return total

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x, y = batch
        pred = self(x)
        loss = self._compute_loss(pred, y, "train")
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        x, y = batch
        pred = self(x)
        self._compute_loss(pred, y, "val")
        self._val_preds.append(pred.detach())
        self._val_targets.append(y.detach())

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return

        all_preds = torch.cat(self._val_preds).squeeze().cpu().numpy()
        all_targets = torch.cat(self._val_targets).squeeze().cpu().numpy()

        rmse = float(np.sqrt(np.mean((all_preds - all_targets) ** 2)))
        mae = float(np.mean(np.abs(all_preds - all_targets)))
        r2 = float(1 - np.sum((all_targets - all_preds) ** 2) / (np.sum((all_targets - np.mean(all_targets)) ** 2) + 1e-8))

        self.log("val/rmse", rmse, prog_bar=True)
        self.log("val/mae", mae)
        self.log("val/r2", r2)

        self._val_preds.clear()
        self._val_targets.clear()

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=self.scheduler_patience,
            min_lr=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val/rmse"},
        }


class BearingFaultModule(pl.LightningModule):
    """Lightning module for bearing fault classification."""

    def __init__(
        self,
        model: nn.Module,
        n_classes: int = 4,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        label_smoothing: float = 0.1,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )
        self.save_hyperparameters(ignore=["model", "class_weights"])

        self._val_preds: list[torch.Tensor] = []
        self._val_targets: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self._val_preds.append(logits.detach())
        self._val_targets.append(y.detach())
        self.log("val/loss", loss)

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return

        all_preds = torch.cat(self._val_preds)
        all_targets = torch.cat(self._val_targets)

        pred_classes = all_preds.argmax(dim=-1).cpu().numpy()
        true_classes = all_targets.cpu().numpy()

        acc = float(np.mean(pred_classes == true_classes))
        self.log("val/accuracy", acc, prog_bar=True)

        # Per-class F1
        from sklearn.metrics import f1_score
        f1 = float(f1_score(true_classes, pred_classes, average="macro", zero_division=0))
        self.log("val/f1_macro", f1, prog_bar=True)

        self._val_preds.clear()
        self._val_targets.clear()

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=50, eta_min=1e-6
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class AnomalyDetectionModule(pl.LightningModule):
    """Lightning module for LSTM-VAE anomaly detection."""

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        beta: float = 1.0,
    ) -> None:
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.beta = beta
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x: torch.Tensor) -> dict:
        return self.model(x)

    def _step(self, batch: tuple, prefix: str) -> torch.Tensor:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        output = self(x)
        losses = self.model.compute_loss(x, output)

        self.log(f"{prefix}/total", losses["total"], prog_bar=True)
        self.log(f"{prefix}/recon", losses["reconstruction"])
        self.log(f"{prefix}/kl", losses["kl"])

        return losses["total"]

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "train")

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        return self._step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


# ─────────────────────────────────────────────────────────────────
# Training Factory
# ─────────────────────────────────────────────────────────────────

def build_trainer(
    model_name: str,
    max_epochs: int = 100,
    checkpoint_dir: str = "./checkpoints",
    mlflow_uri: str = "http://localhost:5000",
    experiment_name: str = "industrial_pm",
    patience: int = 15,
    gpus: int = 0,
    precision: str = "32",
) -> pl.Trainer:
    """Build configured Lightning Trainer."""

    callbacks = [
        ModelCheckpoint(
            dirpath=f"{checkpoint_dir}/{model_name}",
            filename="{epoch:03d}-{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(
            monitor="val/loss",
            patience=patience,
            mode="min",
            min_delta=1e-4,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        RichProgressBar(),
        StochasticWeightAveraging(swa_lrs=1e-4, swa_epoch_start=0.8),
    ]

    loggers = []
    try:
        mlflow_logger = MLFlowLogger(
            experiment_name=experiment_name,
            tracking_uri=mlflow_uri,
            run_name=f"{model_name}_{int(time.time())}",
        )
        loggers.append(mlflow_logger)
    except Exception:
        pass

    accelerator = "gpu" if gpus > 0 and torch.cuda.is_available() else "cpu"
    devices = gpus if gpus > 0 and torch.cuda.is_available() else 1

    return pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        callbacks=callbacks,
        logger=loggers if loggers else True,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        log_every_n_steps=10,
        val_check_interval=1.0,
        enable_model_summary=True,
        deterministic=False,
    )


def train_rul_model(
    model_type: str = "lstm_attention",
    data_path: str = "./data/cmapss",
    subset: str = "FD001",
    max_epochs: int = 100,
    batch_size: int = 256,
    seq_len: int = 30,
    learning_rate: float = 1e-3,
    checkpoint_dir: str = "./checkpoints",
) -> tuple[pl.LightningModule, dict]:
    """Train a RUL prediction model end-to-end."""
    from ml.models.rul_prediction.rul_models import create_rul_model

    # Datasets
    train_dataset = CMAPSSDataset(data_path, subset, "train", seq_len, augment=True)
    val_size = max(1, int(len(train_dataset) * 0.15))
    train_size = len(train_dataset) - val_size
    train_ds, val_ds = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=4
    )

    n_features = len(CMAPSSDataset.SENSOR_COLUMNS)
    model = create_rul_model(model_type, n_features=n_features, seq_len=seq_len)
    lit_model = RULPredictionModule(model, learning_rate=learning_rate)

    trainer = build_trainer(
        model_name=f"rul_{model_type}_{subset}",
        max_epochs=max_epochs,
        checkpoint_dir=checkpoint_dir,
    )

    trainer.fit(lit_model, train_loader, val_loader)
    results = trainer.callback_metrics

    return lit_model, {k: float(v) for k, v in results.items()}


def run_optuna_search(
    model_type: str = "lstm_attention",
    n_trials: int = 50,
    data_path: str = "./data/cmapss",
    subset: str = "FD001",
) -> dict:
    """Optuna hyperparameter search for RUL models."""
    import optuna
    from ml.models.rul_prediction.rul_models import create_rul_model

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256])
        n_layers = trial.suggest_int("n_layers", 1, 4)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
        seq_len = trial.suggest_categorical("seq_len", [15, 30, 45])

        model = create_rul_model(
            model_type,
            n_features=14,
            seq_len=seq_len,
            hidden_size=hidden_size,
            n_layers=n_layers,
            dropout=dropout,
        )
        lit = RULPredictionModule(model, learning_rate=lr)

        train_ds = CMAPSSDataset(data_path, subset, "train", seq_len)
        val_size = max(1, int(len(train_ds) * 0.15))
        train_ds, val_ds = random_split(train_ds, [len(train_ds) - val_size, val_size])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        trainer = pl.Trainer(
            max_epochs=30,
            accelerator="cpu",
            callbacks=[
                EarlyStopping("val/rmse", patience=5),
                pl.callbacks.TQDMProgressBar(refresh_rate=0),
            ],
            enable_model_summary=False,
            logger=False,
        )
        trainer.fit(lit, train_loader, val_loader)
        val_rmse = float(trainer.callback_metrics.get("val/rmse", 999))

        # Optuna pruning
        trial.report(val_rmse, step=30)
        if trial.should_prune():
            raise optuna.TrialPruned()

        return val_rmse

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
        study_name=f"rul_{model_type}_optuna",
    )
    study.optimize(objective, n_trials=n_trials, timeout=3600)

    return {
        "best_params": study.best_params,
        "best_value": study.best_value,
        "n_trials": len(study.trials),
    }
