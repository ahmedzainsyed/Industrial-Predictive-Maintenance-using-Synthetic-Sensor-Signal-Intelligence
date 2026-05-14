# System Architecture Documentation

## Industrial Predictive Maintenance Platform
### Synthetic Sensor Signal Intelligence — Architecture Deep-Dive

---

## 1. System Design Principles

The platform is designed around six core engineering principles:

| Principle | Implementation |
|-----------|---------------|
| **Event-Driven** | MQTT → Redis Streams → WebSocket fanout |
| **Modular** | Each subsystem is independently deployable |
| **Observable** | Prometheus metrics on every service boundary |
| **Fault-Tolerant** | Circuit breakers, dead-letter queues, graceful degradation |
| **Scalable** | Stateless API layer, horizontal Celery worker scaling |
| **Reproducible** | DVC dataset versioning, MLflow experiment tracking |

---

## 2. Data Flow Architecture

```
Industrial Asset (Physical / Digital Twin)
           │
           │ Vibration + Temperature + Speed
           ▼
    ┌──────────────┐
    │  MQTT Broker │  eclipse-mosquitto:2.0
    │  (Mosquitto) │  Topics: industrial/pm/telemetry/{asset_id}
    └──────┬───────┘
           │ paho-mqtt subscriber
           ▼
    ┌──────────────────────────────────────────┐
    │         Streaming Engine                 │
    │  • Signal windowing (2048 samples)       │
    │  • Streaming Z-score anomaly detection   │
    │  • Redis Streams write (XADD)            │
    │  • WebSocket broadcast to dashboard      │
    └──────────────────┬───────────────────────┘
                       │
           ┌───────────┴────────────┐
           ▼                        ▼
    ┌─────────────┐         ┌──────────────┐
    │ Redis       │         │  FastAPI     │
    │ Streams     │         │  Backend     │
    │ (Buffer)    │         │  (REST + WS) │
    └──────┬──────┘         └──────┬───────┘
           │                       │
           ▼                       ▼
    ┌─────────────────────────────────────────┐
    │           Celery Workers                 │
    │  ┌──────────────┐  ┌───────────────┐    │
    │  │ Signal       │  │ AI Inference  │    │
    │  │ Processing   │  │ Queue         │    │
    │  │ Queue        │  │               │    │
    │  │ • FFT        │  │ • RUL LSTM    │    │
    │  │ • Wavelet    │  │ • Fault CNN   │    │
    │  │ • Features   │  │ • LSTM-VAE    │    │
    │  └──────────────┘  └───────────────┘    │
    └──────────────────┬──────────────────────┘
                       │
                       ▼
              ┌────────────────┐
              │   PostgreSQL   │  Results storage
              │   Database     │  Asset registry
              └────────────────┘
```

---

## 3. Signal Processing Pipeline

```
Raw Vibration Signal x[n]  (20 kHz, 2048 samples = 0.1s)
           │
           ▼
┌─────────────────────────────────┐
│    Preprocessing                │
│  • Linear detrend               │
│  • DC removal                   │
│  • Anti-aliasing check          │
└──────────────┬──────────────────┘
               │
      ┌────────┴─────────┐
      ▼                   ▼
┌──────────┐       ┌───────────────┐
│  FFT     │       │   Wavelet     │
│ Engine   │       │   Engine      │
│          │       │               │
│ • Welch  │       │ • DWT (db8)   │
│   PSD    │       │ • CWT (Morlet)│
│ • STFT   │       │ • Denoising   │
│ • Peaks  │       │ • Transients  │
│ • BPFO   │       │ • Entropy     │
└────┬─────┘       └───────┬───────┘
     │                     │
     └──────────┬───────────┘
                ▼
     ┌─────────────────────┐
     │  Feature Extraction │
     │                     │
     │  FFT features (64)  │
     │  Wavelet feat. (48) │
     │  Time-domain (8)    │
     │  ─────────────────  │
     │  Total: 120 dims    │
     └──────────┬──────────┘
                │
                ▼
     ┌─────────────────────┐
     │  AI Inference       │
     │                     │
     │  → RUL prediction   │
     │  → Fault diagnosis  │
     │  → Anomaly score    │
     └─────────────────────┘
```

---

## 4. AI Model Architecture Summary

### 4.1 RUL Prediction — AttentionLSTM

```
Input (B, T=30, F=14)
    │
    ▼
LayerNorm → Linear(14, 128)
    │
    ▼
BiLSTM × 3 layers (hidden=128, bidirectional)
    │         └─→ (B, T, 256)
    ▼
Linear(256, 128) + LayerNorm
    │
    ▼
MultiHead Attention (8 heads, causal)
    │
    ▼
FFN (128 → 512 → 128) + LayerNorm
    │
    ▼ Last timestep
Linear(128, 64) → GELU → Linear(64, 32) → GELU → Linear(32, 1)
    │
    ▼
Softplus (enforce RUL ≥ 0)
    │
    ▼
Output: RUL prediction (scalar)
```

**Parameters:** ~1.2M  
**Input:** 30-step multivariate sensor sequence  
**Output:** Scalar RUL estimate in cycles  
**Training loss:** MSE + NASA asymmetric score  

### 4.2 Bearing Fault — SpectralCNN

```
Input (B, 1, 257, 128)  ← STFT spectrogram
    │
    ▼
ResBlock(1→32) → MaxPool(2)
ResBlock(32→64) → MaxPool(2)
ResBlock(64→128) → MaxPool(2)
ResBlock(128→256) → MaxPool(2)
    │
    ▼
GlobalAveragePool → (B, 256)
    │
    ▼
Linear(256, 128) → GELU → Dropout(0.3)
Linear(128, 64) → GELU
Linear(64, 4)
    │
    ▼
Output: 4-class logits [healthy, inner, outer, ball]
```

**Parameters:** ~850K  
**CAM/Grad-CAM:** Supported for fault localization  

### 4.3 Anomaly Detection — LSTM-VAE

```
Encoder:
  Input (B, T=30, F=14)
  LSTM(14→64, layers=2) → h_n
  Linear(64, 16) → μ
  Linear(64, 16) → log σ²
  z = μ + ε·exp(0.5·log σ²)  [reparameterization]

Decoder:
  Linear(16, 64) → expand(T)
  LSTM(64→64, layers=2)
  Linear(64, 14)
  Output: x̂ ∈ ℝ^{T×F}

Loss: ELBO = E[log p(x|z)] - β·KL(q||p)
```

**Anomaly score:** Mean squared reconstruction error  
**AUROC:** > 0.97 on test set  

---

## 5. Digital Twin Physics Model

### 5.1 Degradation Equation (Paris Law)

$$\frac{da}{dt} = C \cdot (\Delta K)^m$$

$$\Delta K = Y \cdot \sigma \cdot \sqrt{\pi a}$$

Where:
- $a$ = crack depth (mm)
- $C$ = Paris coefficient (bearing steel: $10^{-10}$)
- $m$ = Paris exponent (bearing steel: 3.0)
- $Y$ = geometry factor (1.12 for surface crack)
- $\sigma$ = nominal stress (MPa)

### 5.2 Bearing Fault Characteristic Frequencies

For a rolling element bearing with:
- $N_b$ = number of rolling elements
- $d$ = ball diameter
- $D$ = pitch diameter  
- $\alpha$ = contact angle
- $f_s$ = shaft frequency

$$f_{BPFI} = \frac{N_b}{2} f_s \left(1 + \frac{d}{D}\cos\alpha\right)$$

$$f_{BPFO} = \frac{N_b}{2} f_s \left(1 - \frac{d}{D}\cos\alpha\right)$$

$$f_{BSF} = \frac{D}{2d} f_s \left[1 - \left(\frac{d}{D}\cos\alpha\right)^2\right]$$

$$f_{FTF} = \frac{f_s}{2} \left(1 - \frac{d}{D}\cos\alpha\right)$$

---

## 6. Streaming Architecture

```
MQTT Publisher (Digital Twin / Physical Asset)
    │ QoS=1, retained=false
    │ Topic: industrial/pm/telemetry/{asset_id}
    ▼
Mosquitto Broker (Port 1883)
    │
    ▼
asyncio-mqtt Subscriber (Streaming Engine)
    │ Parse JSON payload
    │ Validate schema
    ▼
Signal Window Buffer
    │ Sliding window: 2048 samples @ 20kHz = 102ms
    │ Overlap: 50%
    ▼
Streaming Anomaly Detector
    │ Rolling Z-score (window=50)
    │ Threshold: μ ± 3σ
    ▼
Redis XADD → Stream: industrial/pm/stream
    │ MAXLEN 10000
    ▼
WebSocket Manager (FastAPI)
    │ Channel fanout: telemetry:{asset_id}
    ▼
React Dashboard (recharts + framer-motion)
```

---

## 7. MLOps Workflow

```
New Training Data Available
         │
         ▼
    DVC Data Version (git + S3)
         │
         ▼
    Celery Task: run_training_pipeline
         │
    ┌────┴─────────────────┐
    │  MLflow Experiment   │
    │  • Log params        │
    │  • Log metrics       │
    │  • Log artifacts     │
    │  • Register model    │
    └────┬─────────────────┘
         │
         ▼
    Model Validation Gates
    • RMSE < 15 (RUL)
    • Accuracy > 97% (Bearing)
    • AUROC > 0.95 (Anomaly)
         │
    Pass ▼         Fail → Alert + manual review
    
    Promote to Staging
         │
    A/B Test (10% traffic)
         │
    Performance OK?
    Pass ▼
    
    Promote to Production
         │
         ▼
    Drift Monitor (daily)
    PSI + KS Test
    Performance tracking
```

---

## 8. API Design

### REST Endpoints

```
GET    /api/v1/health
GET    /api/v1/ready

# Assets
GET    /api/v1/assets
POST   /api/v1/assets
GET    /api/v1/assets/{asset_id}
GET    /api/v1/assets/{asset_id}/health

# Inference
POST   /api/v1/inference/rul
POST   /api/v1/inference/bearing-fault
POST   /api/v1/inference/anomaly
POST   /api/v1/inference/full-diagnosis

# Signal Processing
POST   /api/v1/signal/fft/analyze
POST   /api/v1/signal/fft/bearing-faults
POST   /api/v1/signal/fft/anomaly-map
POST   /api/v1/signal/fft/kurtogram
POST   /api/v1/signal/wavelet/analyze
POST   /api/v1/signal/multi-channel/analyze

# Digital Twin
POST   /api/v1/twin/create
GET    /api/v1/twin/{twin_id}/state
POST   /api/v1/twin/{twin_id}/advance
POST   /api/v1/twin/{twin_id}/inject-fault
POST   /api/v1/twin/{twin_id}/maintenance

# MLOps
GET    /api/v1/mlops/experiments
GET    /api/v1/mlops/models
POST   /api/v1/mlops/drift-check
POST   /api/v1/mlops/retrain

# Analytics
GET    /api/v1/analytics/fleet-summary
GET    /api/v1/analytics/maintenance-schedule
GET    /api/v1/analytics/reliability/{asset_id}

# Edge AI
POST   /api/v1/edge/benchmark
GET    /api/v1/edge/profiles
```

### WebSocket Channels

```
ws://host/ws/telemetry/{asset_id}     ← 10 Hz live telemetry
ws://host/ws/anomaly-stream           ← Real-time anomaly events
ws://host/ws/rul-predictions          ← RUL prediction stream
ws://host/ws/digital-twin/{twin_id}   ← Digital twin state
```

---

## 9. Deployment Architecture

### Docker Compose (Development/Staging)

```
pm_network (172.20.0.0/16)
├── pm_postgres      :5432
├── pm_redis         :6379
├── pm_mosquitto     :1883, :9001
├── pm_backend       :8000
├── pm_celery_worker
├── pm_celery_beat
├── pm_flower        :5555
├── pm_digital_twin  :8002
├── pm_streaming     :8003
├── pm_frontend      :3000
├── pm_mlflow        :5000
├── pm_prometheus    :9090
├── pm_grafana       :3001
└── pm_nginx         :80, :443
```

### Kubernetes (Production)

```
Namespace: industrial-pm
├── Deployment: backend (3 replicas, HPA)
├── Deployment: celery-worker (5 replicas, KEDA)
├── Deployment: digital-twin (2 replicas)
├── Deployment: streaming (2 replicas)
├── Deployment: frontend (2 replicas)
├── StatefulSet: postgres (primary + 1 read replica)
├── StatefulSet: redis (sentinel mode)
├── Service: LoadBalancer (nginx-ingress)
├── ConfigMap: platform-config
├── Secret: platform-secrets
├── PersistentVolumeClaim: model-artifacts
└── HorizontalPodAutoscaler: backend, celery-worker
```

---

## 10. Security Architecture

| Layer | Controls |
|-------|----------|
| Transport | TLS 1.3, HSTS |
| Authentication | JWT (RS256), API keys |
| Authorization | RBAC (viewer, operator, admin) |
| Secrets | Kubernetes Secrets / HashiCorp Vault |
| Scanning | Trivy (container), Bandit (code), Safety (deps) |
| Network | NetworkPolicies (k8s), Security Groups |
| Audit | Structured logging, OpenTelemetry traces |

---

*Document version: 1.0.0 | Last updated: 2025*
