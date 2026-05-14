"""
Industrial Predictive Maintenance Platform — FastAPI Application Entry Point

Production-grade async FastAPI application with:
- Lifespan context management
- Database connection pooling
- Redis connection management
- WebSocket telemetry hub
- Prometheus metrics
- Structured logging
- Exception handling middleware
- Rate limiting
- CORS configuration
"""

from __future__ import annotations

import asyncio
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from prometheus_fastapi_instrumentator import Instrumentator

from app.api.v1.router import api_router
from app.core.config import settings
from app.core.events import startup_handler, shutdown_handler
from app.core.exceptions import (
    IndustrialPlatformError,
    SignalProcessingError,
    ModelInferenceError,
    DataIngestionError,
)
from app.core.logging import configure_logging
from app.db.session import engine, AsyncSessionLocal
from app.services.streaming.websocket_manager import WebSocketManager
from app.services.streaming.telemetry_hub import TelemetryHub

# ─────────────────────────────────────────────────────────────────
# Logging Configuration
# ─────────────────────────────────────────────────────────────────
configure_logging(log_level=settings.LOG_LEVEL)
logger = structlog.get_logger(__name__)

# ─────────────────────────────────────────────────────────────────
# Prometheus Metrics
# ─────────────────────────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "pm_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
)
REQUEST_LATENCY = Histogram(
    "pm_http_request_duration_seconds",
    "HTTP request latency",
    ["method", "endpoint"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)
INFERENCE_LATENCY = Histogram(
    "pm_model_inference_duration_seconds",
    "Model inference latency",
    ["model_name", "model_type"],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)
ANOMALY_DETECTIONS = Counter(
    "pm_anomaly_detections_total",
    "Total anomaly detections",
    ["asset_id", "severity"],
)
ACTIVE_WEBSOCKETS = Counter(
    "pm_websocket_connections_total",
    "Total WebSocket connections established",
    ["stream_type"],
)

# ─────────────────────────────────────────────────────────────────
# Global State
# ─────────────────────────────────────────────────────────────────
websocket_manager: WebSocketManager | None = None
telemetry_hub: TelemetryHub | None = None


# ─────────────────────────────────────────────────────────────────
# Application Lifespan
# ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application lifecycle — startup and graceful shutdown."""
    global websocket_manager, telemetry_hub

    logger.info(
        "Starting Industrial Predictive Maintenance Platform",
        version=settings.VERSION,
        environment=settings.ENVIRONMENT,
    )

    # Initialize WebSocket manager
    websocket_manager = WebSocketManager()
    app.state.websocket_manager = websocket_manager

    # Initialize telemetry hub (MQTT + streaming)
    telemetry_hub = TelemetryHub(
        mqtt_host=settings.MQTT_BROKER_HOST,
        mqtt_port=settings.MQTT_BROKER_PORT,
        websocket_manager=websocket_manager,
    )
    app.state.telemetry_hub = telemetry_hub

    # Run startup handlers
    await startup_handler(app)

    # Start background telemetry streaming
    telemetry_task = asyncio.create_task(
        telemetry_hub.start_streaming(),
        name="telemetry_hub",
    )

    logger.info("Platform startup complete — ready to serve requests")

    try:
        yield
    finally:
        logger.info("Platform shutdown initiated")

        # Cancel background tasks
        telemetry_task.cancel()
        try:
            await telemetry_task
        except asyncio.CancelledError:
            pass

        # Run shutdown handlers
        await shutdown_handler(app)

        # Close database engine
        await engine.dispose()

        logger.info("Platform shutdown complete")


# ─────────────────────────────────────────────────────────────────
# Application Factory
# ─────────────────────────────────────────────────────────────────
def create_application() -> FastAPI:
    """Factory function creating the production FastAPI application."""

    app = FastAPI(
        title="Industrial Predictive Maintenance Platform",
        description="""
## 🏭 Industrial Predictive Maintenance Platform

Production-grade industrial AI system for:

- **Real-time anomaly detection** — streaming sensor intelligence
- **Bearing fault diagnosis** — CNN/LSTM spectral classifiers
- **RUL prediction** — LSTM/TCN/Transformer degradation models
- **Signal processing** — FFT, CWT, spectral entropy, cepstral analysis
- **Digital twin** — rotating machinery physics simulation
- **Edge AI** — quantized ONNX inference benchmarking
- **Uncertainty quantification** — Bayesian neural networks, MC-Dropout

### Supported Datasets
- NASA C-MAPSS Turbofan (FD001–FD004)
- NASA IMS Bearing Dataset
- FEMTO Bearing Dataset
- Synthetic telemetry streams

### Signal Processing
- FFT spectral analysis with harmonic detection
- Continuous/Discrete Wavelet Transform (Morlet, Daubechies)
- Spectral entropy, kurtosis, cepstral coefficients
- Adaptive denoising with Wiener/Kalman filters
        """,
        version=settings.VERSION,
        docs_url="/docs" if settings.ENVIRONMENT != "production" else None,
        redoc_url="/redoc" if settings.ENVIRONMENT != "production" else None,
        lifespan=lifespan,
        openapi_tags=[
            {"name": "health", "description": "Platform health & readiness"},
            {"name": "assets", "description": "Industrial asset management"},
            {"name": "sensors", "description": "Sensor data ingestion & retrieval"},
            {"name": "inference", "description": "AI model inference endpoints"},
            {"name": "signal", "description": "Signal processing & analysis"},
            {"name": "rul", "description": "Remaining Useful Life prediction"},
            {"name": "bearing", "description": "Bearing fault diagnosis"},
            {"name": "anomaly", "description": "Anomaly detection & localization"},
            {"name": "twin", "description": "Digital twin simulation"},
            {"name": "streaming", "description": "Real-time telemetry streaming"},
            {"name": "edge", "description": "Edge AI benchmarking"},
            {"name": "mlops", "description": "MLOps & model management"},
            {"name": "analytics", "description": "Reliability analytics"},
        ],
    )

    # ── Middleware Stack ────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    if settings.ENVIRONMENT == "production":
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=settings.ALLOWED_HOSTS,
        )

    # ── Prometheus Instrumentation ──────────────────────────────────
    Instrumentator(
        should_group_status_codes=True,
        should_ignore_untemplated=True,
        should_respect_env_var=True,
        env_var_name="ENABLE_METRICS",
        body_handlers=[],
    ).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

    # ── Include API Routers ─────────────────────────────────────────
    app.include_router(api_router, prefix=settings.API_V1_PREFIX)

    # ── Custom Exception Handlers ───────────────────────────────────
    _register_exception_handlers(app)

    # ── Request Middleware ──────────────────────────────────────────
    _register_request_middleware(app)

    return app


def _register_exception_handlers(app: FastAPI) -> None:
    """Register domain-specific exception handlers."""

    @app.exception_handler(IndustrialPlatformError)
    async def platform_error_handler(
        request: Request, exc: IndustrialPlatformError
    ) -> JSONResponse:
        logger.error(
            "Platform error",
            error_code=exc.error_code,
            message=exc.message,
            path=request.url.path,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.error_code,
                "message": exc.message,
                "details": exc.details,
                "request_id": request.state.request_id,
            },
        )

    @app.exception_handler(SignalProcessingError)
    async def signal_error_handler(
        request: Request, exc: SignalProcessingError
    ) -> JSONResponse:
        logger.error("Signal processing error", detail=str(exc))
        return JSONResponse(
            status_code=422,
            content={
                "error": "SIGNAL_PROCESSING_FAILED",
                "message": str(exc),
                "request_id": request.state.request_id,
            },
        )

    @app.exception_handler(ModelInferenceError)
    async def inference_error_handler(
        request: Request, exc: ModelInferenceError
    ) -> JSONResponse:
        logger.error("Model inference error", model=exc.model_name, detail=str(exc))
        return JSONResponse(
            status_code=503,
            content={
                "error": "INFERENCE_FAILED",
                "message": str(exc),
                "model": exc.model_name,
                "request_id": request.state.request_id,
            },
        )

    @app.exception_handler(500)
    async def internal_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled internal error", path=request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "error": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred",
                "request_id": getattr(request.state, "request_id", "unknown"),
            },
        )


def _register_request_middleware(app: FastAPI) -> None:
    """Register request tracking and timing middleware."""

    @app.middleware("http")
    async def request_tracking_middleware(
        request: Request, call_next
    ) -> Response:
        # Assign request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id

        # Time the request
        start_time = time.perf_counter()

        # Add request ID to response headers
        response = await call_next(request)
        duration = time.perf_counter() - start_time

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{duration:.4f}s"

        # Record Prometheus metrics
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status_code=response.status_code,
        ).inc()
        REQUEST_LATENCY.labels(
            method=request.method,
            endpoint=request.url.path,
        ).observe(duration)

        # Structured access log
        logger.info(
            "HTTP request",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round(duration * 1000, 2),
            request_id=request_id,
        )

        return response


# ─────────────────────────────────────────────────────────────────
# WebSocket Endpoints (defined at app level for hub access)
# ─────────────────────────────────────────────────────────────────
app = create_application()


@app.websocket("/ws/telemetry/{asset_id}")
async def websocket_telemetry(websocket: WebSocket, asset_id: str) -> None:
    """Real-time telemetry stream for a specific industrial asset."""
    manager: WebSocketManager = app.state.websocket_manager
    await manager.connect(websocket, channel=f"telemetry:{asset_id}")
    ACTIVE_WEBSOCKETS.labels(stream_type="telemetry").inc()

    try:
        while True:
            # Keep connection alive; server pushes data via hub
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        await manager.disconnect(websocket, channel=f"telemetry:{asset_id}")


@app.websocket("/ws/anomaly-stream")
async def websocket_anomaly_stream(websocket: WebSocket) -> None:
    """Real-time anomaly event stream across all assets."""
    manager: WebSocketManager = app.state.websocket_manager
    await manager.connect(websocket, channel="anomaly:global")
    ACTIVE_WEBSOCKETS.labels(stream_type="anomaly").inc()

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await manager.disconnect(websocket, channel="anomaly:global")


@app.websocket("/ws/rul-predictions")
async def websocket_rul_predictions(websocket: WebSocket) -> None:
    """Real-time RUL prediction stream."""
    manager: WebSocketManager = app.state.websocket_manager
    await manager.connect(websocket, channel="rul:stream")
    ACTIVE_WEBSOCKETS.labels(stream_type="rul").inc()

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await manager.disconnect(websocket, channel="rul:stream")


@app.websocket("/ws/digital-twin/{twin_id}")
async def websocket_digital_twin(websocket: WebSocket, twin_id: str) -> None:
    """Real-time digital twin state stream."""
    manager: WebSocketManager = app.state.websocket_manager
    await manager.connect(websocket, channel=f"twin:{twin_id}")
    ACTIVE_WEBSOCKETS.labels(stream_type="twin").inc()

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await manager.disconnect(websocket, channel=f"twin:{twin_id}")


# ─────────────────────────────────────────────────────────────────
# Health Endpoints
# ─────────────────────────────────────────────────────────────────
@app.get("/health", tags=["health"])
async def health_check() -> dict:
    """Kubernetes liveness probe."""
    return {"status": "healthy", "version": settings.VERSION}


@app.get("/ready", tags=["health"])
async def readiness_check() -> dict:
    """Kubernetes readiness probe — checks all dependencies."""
    from app.db.session import check_database_health
    from app.core.cache import check_redis_health

    db_ok = await check_database_health()
    redis_ok = await check_redis_health()

    all_ok = db_ok and redis_ok
    return JSONResponse(
        status_code=200 if all_ok else 503,
        content={
            "status": "ready" if all_ok else "not_ready",
            "dependencies": {
                "database": "healthy" if db_ok else "unhealthy",
                "cache": "healthy" if redis_ok else "unhealthy",
            },
        },
    )


@app.get("/metrics/raw", include_in_schema=False)
async def raw_metrics() -> Response:
    """Raw Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
