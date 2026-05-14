"""
Industrial Predictive Maintenance Platform — Configuration

Pydantic Settings v2 configuration with environment variable support,
type validation, and sensible industrial defaults.
"""

from __future__ import annotations

import secrets
from functools import lru_cache
from typing import Any, Literal

from pydantic import AnyHttpUrl, Field, PostgresDsn, RedisDsn, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Platform-wide configuration — loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Core ──────────────────────────────────────────────────────
    VERSION: str = "1.0.0"
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"
    LOG_LEVEL: str = "INFO"
    DEBUG: bool = False
    SECRET_KEY: str = Field(default_factory=lambda: secrets.token_hex(32))

    # ── API ────────────────────────────────────────────────────────
    API_V1_PREFIX: str = "/api/v1"
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:5173"]
    ALLOWED_HOSTS: list[str] = ["*"]

    # ── Database ───────────────────────────────────────────────────
    DATABASE_URL: str = "postgresql+asyncpg://pmuser:pmpassword@localhost:5432/predictive_maintenance"
    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 40
    DB_POOL_TIMEOUT: int = 30
    DB_POOL_RECYCLE: int = 1800
    DB_ECHO: bool = False

    # ── Redis ──────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_CACHE_TTL: int = 300          # seconds
    REDIS_STREAM_MAXLEN: int = 10_000   # max stream entries

    # ── Celery ─────────────────────────────────────────────────────
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"
    CELERY_TASK_SERIALIZER: str = "json"
    CELERY_RESULT_SERIALIZER: str = "json"
    CELERY_TASK_SOFT_TIME_LIMIT: int = 300
    CELERY_TASK_HARD_TIME_LIMIT: int = 600

    # ── MQTT ───────────────────────────────────────────────────────
    MQTT_BROKER_HOST: str = "localhost"
    MQTT_BROKER_PORT: int = 1883
    MQTT_KEEPALIVE: int = 60
    MQTT_QOS: int = 1
    MQTT_TOPIC_PREFIX: str = "industrial/pm"

    # ── AI / ML ────────────────────────────────────────────────────
    MODEL_DEVICE: str = "cpu"           # "cuda", "cpu", "mps"
    MODEL_ARTIFACTS_DIR: str = "/app/artifacts"
    INFERENCE_BATCH_SIZE: int = 32
    INFERENCE_MAX_WORKERS: int = 4
    MODEL_CACHE_SIZE: int = 10          # max models in LRU cache

    # Bayesian / Uncertainty
    MC_DROPOUT_SAMPLES: int = 50
    CONFORMAL_ALPHA: float = 0.05       # 95% coverage

    # ── Signal Processing ──────────────────────────────────────────
    SAMPLING_RATE_HZ: int = 20_000      # IMS bearing default
    FFT_WINDOW_SIZE: int = 1024
    FFT_OVERLAP: float = 0.5
    FFT_WINDOW_FUNC: str = "hann"       # hann, hamming, blackman

    WAVELET_FAMILY: str = "db8"         # Daubechies-8
    WAVELET_LEVELS: int = 6
    WAVELET_THRESHOLD_MODE: str = "soft"

    SEGMENT_LENGTH: int = 2048          # samples per segment
    SEGMENT_OVERLAP: float = 0.25

    SPECTRAL_ENTROPY_BINS: int = 256
    CEPSTRAL_LIFTER: int = 22

    # ── Streaming ──────────────────────────────────────────────────
    TELEMETRY_PUBLISH_INTERVAL_MS: int = 100   # 10 Hz
    STREAM_BUFFER_SIZE: int = 10_000
    ANOMALY_DETECTION_WINDOW: int = 50
    WEBSOCKET_HEARTBEAT_INTERVAL: int = 30

    # ── Digital Twin ───────────────────────────────────────────────
    TWIN_UPDATE_INTERVAL_MS: int = 100
    TWIN_DEGRADATION_SEED: int = 42
    SIMULATION_SPEED: float = 1.0       # 1.0 = real-time
    MAX_TWIN_INSTANCES: int = 50

    # ── MLOps ──────────────────────────────────────────────────────
    MLFLOW_TRACKING_URI: str = "http://localhost:5000"
    MLFLOW_EXPERIMENT_NAME: str = "industrial_pm"
    DVC_REMOTE_URL: str = "s3://pm-dvc-store"
    WANDB_PROJECT: str = "industrial-pm"
    WANDB_ENTITY: str | None = None

    # Model drift thresholds
    DRIFT_PSI_THRESHOLD: float = 0.2    # Population Stability Index
    DRIFT_KS_THRESHOLD: float = 0.05    # Kolmogorov-Smirnov p-value
    RETRAINING_TRIGGER_MAE_DELTA: float = 0.15  # 15% degradation

    # ── Industrial Parameters ──────────────────────────────────────
    # Bearing fault characteristic frequencies (normalized)
    BPFI_MULT: float = 7.29    # Ball Pass Freq Inner race
    BPFO_MULT: float = 5.42    # Ball Pass Freq Outer race
    BSF_MULT: float = 2.36     # Ball Spin Frequency
    FTF_MULT: float = 0.38     # Fundamental Train Frequency

    # C-MAPSS dataset configuration
    CMAPSS_SENSORS: list[int] = Field(
        default=[2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]
    )  # 14 selected sensors
    CMAPSS_MAX_RUL: int = 125   # Piece-wise linear RUL cap

    # Weibull reliability parameters (defaults)
    WEIBULL_SHAPE_BETA: float = 2.5
    WEIBULL_SCALE_ETA: float = 1000.0   # hours

    @field_validator("MODEL_DEVICE")
    @classmethod
    def validate_device(cls, v: str) -> str:
        allowed = {"cpu", "cuda", "mps", "cuda:0", "cuda:1"}
        if v not in allowed:
            raise ValueError(f"MODEL_DEVICE must be one of {allowed}")
        return v

    @field_validator("ENVIRONMENT")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        return v.lower()

    @model_validator(mode="after")
    def validate_production_settings(self) -> "Settings":
        if self.ENVIRONMENT == "production":
            if self.SECRET_KEY == secrets.token_hex(32):
                raise ValueError("Production requires an explicit SECRET_KEY")
            if self.DEBUG:
                raise ValueError("DEBUG must be False in production")
        return self

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"

    @property
    def redis_stream_key(self) -> str:
        return f"{self.MQTT_TOPIC_PREFIX}:stream"

    @property
    def mqtt_telemetry_topic(self) -> str:
        return f"{self.MQTT_TOPIC_PREFIX}/telemetry/+"

    @property
    def mqtt_anomaly_topic(self) -> str:
        return f"{self.MQTT_TOPIC_PREFIX}/anomaly/+"

    @property
    def nyquist_frequency(self) -> float:
        return self.SAMPLING_RATE_HZ / 2.0

    @property
    def frequency_resolution(self) -> float:
        return self.SAMPLING_RATE_HZ / self.FFT_WINDOW_SIZE

    def get_bearing_fault_frequencies(self, shaft_rpm: float) -> dict[str, float]:
        """Compute bearing fault characteristic frequencies for given shaft speed."""
        shaft_hz = shaft_rpm / 60.0
        return {
            "BPFI": self.BPFI_MULT * shaft_hz,
            "BPFO": self.BPFO_MULT * shaft_hz,
            "BSF": self.BSF_MULT * shaft_hz,
            "FTF": self.FTF_MULT * shaft_hz,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings instance — singleton pattern."""
    return Settings()


settings = get_settings()
