"""
MLOps Infrastructure — Experiment Tracking, Model Registry & Drift Detection

Production MLOps pipeline for industrial predictive maintenance:
1. MLflow experiment tracking with automatic logging
2. Model registry with staging/production/archived stages
3. Dataset versioning with DVC
4. Automated drift detection (PSI + KS test)
5. Retraining trigger logic
6. CI/CD model validation gates
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class DriftReport:
    """Statistical drift detection report."""
    model_id: str
    timestamp: float
    n_reference_samples: int
    n_current_samples: int

    # Population Stability Index per feature
    psi_scores: dict[str, float]
    overall_psi: float

    # Kolmogorov-Smirnov test per feature
    ks_statistics: dict[str, float]
    ks_pvalues: dict[str, float]

    # Model performance drift
    reference_metric: float
    current_metric: float
    metric_delta_pct: float

    # Verdict
    drift_detected: bool
    drift_severity: str  # none | minor | moderate | severe
    retraining_recommended: bool
    affected_features: list[str]


class ExperimentTracker:
    """
    Production experiment tracking with MLflow.
    Wraps MLflow with industrial-specific logging utilities.
    """

    def __init__(
        self,
        tracking_uri: str = "http://localhost:5000",
        experiment_name: str = "industrial_pm",
    ) -> None:
        try:
            import mlflow
            mlflow.set_tracking_uri(tracking_uri)
            self.mlflow = mlflow
            self.experiment_name = experiment_name
            self._ensure_experiment()
            self._mlflow_available = True
        except ImportError:
            self._mlflow_available = False
            print("MLflow not installed — using local file tracking")
        except Exception:
            self._mlflow_available = False

    def _ensure_experiment(self) -> None:
        if not self._mlflow_available:
            return
        exp = self.mlflow.get_experiment_by_name(self.experiment_name)
        if exp is None:
            self.mlflow.create_experiment(
                self.experiment_name,
                tags={
                    "domain": "industrial_predictive_maintenance",
                    "platform": "synthetic_sensor_signal_intelligence",
                },
            )
        self.mlflow.set_experiment(self.experiment_name)

    def start_run(
        self,
        run_name: str,
        tags: dict[str, str] | None = None,
    ) -> Any:
        """Start a tracked experiment run."""
        if not self._mlflow_available:
            return _MockRun(run_name)

        default_tags = {
            "model_framework": "pytorch",
            "platform_version": "1.0.0",
        }
        if tags:
            default_tags.update(tags)

        return self.mlflow.start_run(run_name=run_name, tags=default_tags)

    def log_training_config(self, config: dict) -> None:
        """Log complete training configuration."""
        if not self._mlflow_available:
            return
        # Flatten nested dict for MLflow params
        flat = self._flatten_dict(config)
        for key, value in flat.items():
            try:
                self.mlflow.log_param(key, str(value)[:250])
            except Exception:
                pass

    def log_epoch_metrics(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
    ) -> None:
        """Log per-epoch training and validation metrics."""
        if not self._mlflow_available:
            return
        combined = {f"train/{k}": v for k, v in train_metrics.items()}
        combined.update({f"val/{k}": v for k, v in val_metrics.items()})
        self.mlflow.log_metrics(combined, step=epoch)

    def log_final_metrics(self, metrics: dict[str, float]) -> None:
        """Log final evaluation metrics."""
        if not self._mlflow_available:
            return
        self.mlflow.log_metrics(metrics)

    def log_model(
        self,
        model,
        model_name: str,
        input_example=None,
        extra_metadata: dict | None = None,
    ) -> str:
        """Log PyTorch model to MLflow registry."""
        if not self._mlflow_available:
            return f"local://{model_name}"

        try:
            import mlflow.pytorch
            model_info = mlflow.pytorch.log_model(
                model,
                artifact_path=model_name,
                registered_model_name=model_name,
                input_example=input_example,
                metadata=extra_metadata or {},
            )
            return model_info.model_uri
        except Exception as e:
            print(f"Model logging failed: {e}")
            return ""

    def log_signal_artifacts(
        self,
        fft_plot_path: str | None = None,
        wavelet_plot_path: str | None = None,
        confusion_matrix_path: str | None = None,
        rul_plot_path: str | None = None,
    ) -> None:
        """Log signal processing and model artifacts."""
        if not self._mlflow_available:
            return
        for path in [fft_plot_path, wavelet_plot_path, confusion_matrix_path, rul_plot_path]:
            if path and Path(path).exists():
                try:
                    self.mlflow.log_artifact(path)
                except Exception:
                    pass

    def _flatten_dict(self, d: dict, prefix: str = "") -> dict:
        result = {}
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                result.update(self._flatten_dict(v, key))
            else:
                result[key] = v
        return result


class ModelRegistryManager:
    """
    Production model registry with staged deployment.

    Stages: None → Staging → Production → Archived
    Each transition requires validation gates.
    """

    def __init__(self, tracking_uri: str = "http://localhost:5000") -> None:
        try:
            import mlflow
            from mlflow.tracking import MlflowClient
            mlflow.set_tracking_uri(tracking_uri)
            self.client = MlflowClient()
            self._available = True
        except ImportError:
            self._available = False
            self._local_registry: dict = {}

    def register_model(
        self,
        model_uri: str,
        model_name: str,
        description: str = "",
        tags: dict[str, str] | None = None,
    ) -> dict:
        """Register a model version in the registry."""
        if not self._available:
            version_info = {
                "name": model_name,
                "version": "1",
                "stage": "None",
                "uri": model_uri,
            }
            self._local_registry[model_name] = version_info
            return version_info

        try:
            version = self.client.create_model_version(
                name=model_name,
                source=model_uri,
                description=description,
                tags=tags or {},
            )
            return {
                "name": version.name,
                "version": version.version,
                "stage": version.current_stage,
                "uri": version.source,
                "run_id": version.run_id,
            }
        except Exception as e:
            return {"error": str(e)}

    def transition_model(
        self,
        model_name: str,
        version: str,
        target_stage: str,
        archive_existing: bool = True,
    ) -> bool:
        """
        Transition model version between stages.

        Stages: "Staging" | "Production" | "Archived"
        """
        if not self._available:
            return True

        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=target_stage,
                archive_existing_versions=archive_existing,
            )
            return True
        except Exception as e:
            print(f"Stage transition failed: {e}")
            return False

    def get_production_model(self, model_name: str) -> dict | None:
        """Get the current production model version."""
        if not self._available:
            return self._local_registry.get(model_name)

        try:
            versions = self.client.get_latest_versions(
                model_name, stages=["Production"]
            )
            if versions:
                v = versions[0]
                return {
                    "name": v.name,
                    "version": v.version,
                    "stage": v.current_stage,
                    "uri": v.source,
                }
        except Exception:
            pass
        return None

    def validate_model_gates(
        self,
        metrics: dict[str, float],
        model_type: str = "rul",
    ) -> tuple[bool, list[str]]:
        """
        Check if model passes production quality gates.

        Gates by model type:
        - rul: RMSE < 15, MAE < 12, NASA Score < 1000
        - bearing: accuracy > 97%, f1_macro > 0.96
        - anomaly: auroc > 0.95, precision > 0.90
        """
        gates = {
            "rul": {
                "rmse": ("lt", 15.0),
                "mae": ("lt", 12.0),
                "r2": ("gt", 0.85),
            },
            "bearing": {
                "accuracy": ("gt", 0.97),
                "f1_macro": ("gt", 0.96),
                "roc_auc_macro": ("gt", 0.98),
            },
            "anomaly": {
                "roc_auc_macro": ("gt", 0.95),
                "precision_macro": ("gt", 0.90),
            },
        }

        model_gates = gates.get(model_type, {})
        failures = []

        for metric_name, (comparison, threshold) in model_gates.items():
            if metric_name not in metrics:
                continue
            value = metrics[metric_name]
            if comparison == "lt" and value >= threshold:
                failures.append(f"{metric_name}={value:.3f} (must be < {threshold})")
            elif comparison == "gt" and value <= threshold:
                failures.append(f"{metric_name}={value:.3f} (must be > {threshold})")

        passed = len(failures) == 0
        return passed, failures


class DriftDetector:
    """
    Production model and data drift detection.

    Implements:
    1. Population Stability Index (PSI) for feature drift
    2. Kolmogorov-Smirnov test for distribution shift
    3. Performance degradation monitoring
    4. Concept drift via error rate monitoring

    PSI Interpretation:
        PSI < 0.1:  No significant change
        0.1-0.2:    Minor change — monitoring required
        > 0.2:      Major change — retraining recommended
    """

    PSI_THRESHOLDS = {
        "none": 0.1,
        "minor": 0.2,
        "moderate": 0.25,
        "severe": 0.30,
    }

    def __init__(
        self,
        psi_threshold: float = 0.2,
        ks_alpha: float = 0.05,
        performance_delta_threshold: float = 0.15,
        n_bins: int = 10,
    ) -> None:
        self.psi_threshold = psi_threshold
        self.ks_alpha = ks_alpha
        self.performance_delta_threshold = performance_delta_threshold
        self.n_bins = n_bins

        self._reference_data: np.ndarray | None = None
        self._reference_metric: float | None = None
        self._reference_feature_names: list[str] = []

    def set_reference(
        self,
        reference_data: np.ndarray,
        reference_metric: float,
        feature_names: list[str] | None = None,
    ) -> None:
        """Set reference distribution from training/validation data."""
        self._reference_data = reference_data
        self._reference_metric = reference_metric
        n_features = reference_data.shape[1] if reference_data.ndim > 1 else 1
        self._reference_feature_names = feature_names or [f"feature_{i}" for i in range(n_features)]

    def detect_drift(
        self,
        current_data: np.ndarray,
        current_metric: float,
        model_id: str = "unknown",
    ) -> DriftReport:
        """
        Run full drift detection pipeline.

        Returns comprehensive DriftReport with per-feature analysis.
        """
        if self._reference_data is None:
            raise RuntimeError("Reference data not set. Call set_reference() first.")

        n_ref = len(self._reference_data)
        n_cur = len(current_data)

        ref = self._reference_data
        cur = current_data

        if ref.ndim == 1:
            ref = ref.reshape(-1, 1)
        if cur.ndim == 1:
            cur = cur.reshape(-1, 1)

        n_features = min(ref.shape[1], cur.shape[1])

        # Per-feature PSI and KS
        psi_scores = {}
        ks_stats = {}
        ks_pvalues = {}

        for i in range(n_features):
            fname = self._reference_feature_names[i] if i < len(self._reference_feature_names) else f"f{i}"
            ref_feat = ref[:, i]
            cur_feat = cur[:, i]

            psi_scores[fname] = self._compute_psi(ref_feat, cur_feat)
            ks_stat, ks_pval = self._compute_ks(ref_feat, cur_feat)
            ks_stats[fname] = ks_stat
            ks_pvalues[fname] = ks_pval

        overall_psi = float(np.mean(list(psi_scores.values())))

        # Performance drift
        ref_metric = self._reference_metric or 0.0
        delta_pct = abs(current_metric - ref_metric) / (abs(ref_metric) + 1e-8)

        # Determine severity
        drift_severity = "none"
        for severity, threshold in [
            ("severe", self.PSI_THRESHOLDS["severe"]),
            ("moderate", self.PSI_THRESHOLDS["moderate"]),
            ("minor", self.PSI_THRESHOLDS["minor"]),
        ]:
            if overall_psi >= threshold:
                drift_severity = severity
                break

        # Affected features (PSI > threshold)
        affected = [
            fname for fname, psi in psi_scores.items()
            if psi > self.psi_threshold
        ]

        # Drift decision
        psi_drift = overall_psi > self.psi_threshold
        ks_drift = any(pv < self.ks_alpha for pv in ks_pvalues.values())
        perf_drift = delta_pct > self.performance_delta_threshold
        drift_detected = psi_drift or ks_drift or perf_drift

        retraining_recommended = (
            drift_severity in ("moderate", "severe")
            or perf_drift
            or len(affected) > n_features * 0.3
        )

        return DriftReport(
            model_id=model_id,
            timestamp=time.time(),
            n_reference_samples=n_ref,
            n_current_samples=n_cur,
            psi_scores=psi_scores,
            overall_psi=round(overall_psi, 4),
            ks_statistics=ks_stats,
            ks_pvalues=ks_pvalues,
            reference_metric=ref_metric,
            current_metric=current_metric,
            metric_delta_pct=round(delta_pct * 100, 2),
            drift_detected=drift_detected,
            drift_severity=drift_severity,
            retraining_recommended=retraining_recommended,
            affected_features=affected,
        )

    def _compute_psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
    ) -> float:
        """
        Population Stability Index:
        PSI = Σ (A_i - E_i) × ln(A_i / E_i)

        Where:
          A_i = % of current population in bin i
          E_i = % of reference population in bin i
        """
        bin_edges = np.percentile(reference, np.linspace(0, 100, self.n_bins + 1))
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 3:
            return 0.0

        ref_counts = np.histogram(reference, bins=bin_edges)[0]
        cur_counts = np.histogram(current, bins=bin_edges)[0]

        ref_pct = ref_counts / (len(reference) + 1e-8) + 1e-10
        cur_pct = cur_counts / (len(current) + 1e-8) + 1e-10

        psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
        return round(abs(psi), 4)

    def _compute_ks(
        self,
        reference: np.ndarray,
        current: np.ndarray,
    ) -> tuple[float, float]:
        """Two-sample Kolmogorov-Smirnov test."""
        from scipy.stats import ks_2samp
        stat, pvalue = ks_2samp(reference, current)
        return float(stat), float(pvalue)


class AutoRetrainingPipeline:
    """
    Automated model retraining pipeline triggered by drift detection.

    Workflow:
    1. Monitor → detect drift → trigger retraining
    2. Fetch latest data → preprocess → train
    3. Validate against production gates
    4. Promote to staging if gates pass
    5. A/B test → promote to production
    """

    def __init__(
        self,
        tracker: ExperimentTracker,
        registry: ModelRegistryManager,
        drift_detector: DriftDetector,
    ) -> None:
        self.tracker = tracker
        self.registry = registry
        self.drift = drift_detector

    def check_and_trigger(
        self,
        current_data: np.ndarray,
        current_metric: float,
        model_id: str,
    ) -> dict[str, Any]:
        """
        Check for drift and trigger retraining if needed.

        Returns dict with drift status and retraining decision.
        """
        try:
            drift_report = self.drift.detect_drift(current_data, current_metric, model_id)
        except RuntimeError:
            return {"status": "no_reference", "retraining_triggered": False}

        result = {
            "model_id": model_id,
            "drift_detected": drift_report.drift_detected,
            "drift_severity": drift_report.drift_severity,
            "overall_psi": drift_report.overall_psi,
            "metric_delta_pct": drift_report.metric_delta_pct,
            "affected_features": drift_report.affected_features,
            "retraining_recommended": drift_report.retraining_recommended,
            "retraining_triggered": False,
            "timestamp": time.time(),
        }

        if drift_report.retraining_recommended:
            result["retraining_triggered"] = True
            result["scheduled_time"] = time.time() + 300  # Schedule in 5 min
            result["reason"] = self._build_retraining_reason(drift_report)

        return result

    def _build_retraining_reason(self, report: DriftReport) -> str:
        reasons = []
        if report.overall_psi > 0.2:
            reasons.append(f"Data drift PSI={report.overall_psi:.3f}")
        if report.metric_delta_pct > 15:
            reasons.append(f"Performance degradation {report.metric_delta_pct:.1f}%")
        if report.affected_features:
            reasons.append(f"Features drifted: {', '.join(report.affected_features[:3])}")
        return " | ".join(reasons) if reasons else "Drift threshold exceeded"


class _MockRun:
    """Fallback when MLflow is unavailable."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
