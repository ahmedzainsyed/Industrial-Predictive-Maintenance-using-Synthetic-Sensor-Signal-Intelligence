# ─────────────────────────────────────────────────────────────────
# Industrial Predictive Maintenance Platform — Backend Dockerfile
# Multi-stage build: builder → production
# ─────────────────────────────────────────────────────────────────

# ── Stage 1: Build Dependencies ────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    gfortran \
    libffi-dev \
    libssl-dev \
    libpq-dev \
    libhdf5-dev \
    libblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python requirements
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --user -r requirements.txt

# ── Stage 2: Production Image ──────────────────────────────────────
FROM python:3.11-slim AS production

LABEL maintainer="Industrial PM Platform Team"
LABEL version="1.0.0"
LABEL description="Industrial Predictive Maintenance Backend API"

# Create non-root user
RUN groupadd -r pmuser && useradd -r -g pmuser -d /app -s /sbin/nologin pmuser

WORKDIR /app

# Install runtime system dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    libgomp1 \
    libhdf5-103 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /home/pmuser/.local
ENV PATH=/home/pmuser/.local/bin:$PATH
ENV PYTHONPATH=/app

# Copy application code
COPY --chown=pmuser:pmuser . .

# Create necessary directories
RUN mkdir -p /app/artifacts /app/logs && \
    chown -R pmuser:pmuser /app

# Switch to non-root user
USER pmuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Default command
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--loop", "uvloop", \
     "--http", "httptools", \
     "--log-level", "info", \
     "--access-log"]
