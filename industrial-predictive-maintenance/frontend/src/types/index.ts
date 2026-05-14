// ─────────────────────────────────────────────────────────────────────────────
// Industrial Predictive Maintenance Platform — TypeScript Type Definitions
// ─────────────────────────────────────────────────────────────────────────────

// ── Asset & Sensor Types ──────────────────────────────────────────────────────

export type AssetState =
  | "healthy"
  | "incipient"
  | "degraded"
  | "severe"
  | "critical"
  | "failed";

export type FaultType =
  | "none"
  | "inner_race"
  | "outer_race"
  | "ball_fault"
  | "misalignment"
  | "imbalance"
  | "looseness"
  | "lubrication_failure";

export type AssetType =
  | "rotating_machinery"
  | "pump"
  | "fan"
  | "compressor"
  | "motor"
  | "turbofan"
  | "gearbox";

export interface Asset {
  id: string;
  name: string;
  type: AssetType;
  location: string;
  state: AssetState;
  healthScore: number; // 0-100
  rulHours: number;
  activeFault: FaultType;
  faultSeverity: number; // 0-1
  temperature: number; // °C
  vibrationRms: number; // g
  shaftRpm: number;
  operatingHours: number;
  lastMaintenanceHours: number;
  createdAt: string;
  updatedAt: string;
  tags: string[];
  metadata: Record<string, unknown>;
}

export interface AssetHealthSummary {
  totalAssets: number;
  healthyCount: number;
  incipientCount: number;
  degradedCount: number;
  severeCount: number;
  criticalCount: number;
  failedCount: number;
  avgHealthScore: number;
  avgRulHours: number;
  criticalAssets: Asset[];
}

// ── Telemetry Types ───────────────────────────────────────────────────────────

export interface TelemetryPoint {
  timestamp: number; // Unix ms
  assetId: string;
  vibrationRms: number;
  vibrationX: number;
  vibrationY: number;
  vibrationZ: number;
  bearingTemp: number;
  motorTemp: number;
  ambientTemp: number;
  shaftRpm: number;
  loadPercent: number;
  powerKw: number;
  kurtosis: number;
  crestFactor: number;
  peakG: number;
  healthScore: number;
  anomalyScore: number;
  rulEstimate: number;
}

export interface TelemetryStream {
  assetId: string;
  points: TelemetryPoint[];
  latestPoint: TelemetryPoint | null;
  isConnected: boolean;
  lastUpdated: number;
}

// ── Signal Processing Types ──────────────────────────────────────────────────

export interface FFTResult {
  frequencies: number[];
  psd: number[];
  dominantFrequencies: number[];
  dominantAmplitudes: number[];
  spectralEntropy: number;
  spectralCentroid: number;
  spectralBandwidth: number;
  spectralFlatness: number;
  totalPowerDb: number;
  bandPowers: Record<string, number>;
  harmonics: HarmonicSeries[];
  samplingRate: number;
  windowFunction: string;
}

export interface HarmonicSeries {
  fundamentalHz: number;
  harmonicFrequencies: number[];
  harmonicAmplitudes: number[];
  totalHarmonicDistortion: number;
  harmonicEnergyRatio: number;
}

export interface STFTResult {
  frequencies: number[];
  times: number[];
  magnitude: number[][]; // [freq_bins][time_frames]
}

export interface WaveletResult {
  scales: number[];
  frequencies: number[];
  cwtPower: number[][]; // [n_scales][n_samples]
  energyPerLevel: number[];
  energyRatios: number[];
  waveletEntropy: number;
  transientLocations: number[];
  transientAmplitudes: number[];
  levelKurtosis: number[];
  snrDb: number;
  denoisedSignal: number[];
}

export interface BearingFaultDetection {
  faultType: string;
  targetFrequencyHz: number;
  detectedFrequencyHz: number;
  peakAmplitudeDb: number;
  noiseFloorDb: number;
  snrDb: number;
  confidence: number;
  severity: string;
  detected: boolean;
}

export interface SpectralAnomalyMap {
  frequencies: number[];
  times: number[];
  anomalyScores: number[][]; // [freq_bins][time_frames]
  anomalyThreshold: number;
  anomalyRegions: AnomalyRegion[];
  severity: number;
}

export interface AnomalyRegion {
  freqStartHz: number;
  freqEndHz: number;
  timeStartS: number;
  timeEndS: number;
  maxScore: number;
  meanScore: number;
}

// ── AI Prediction Types ───────────────────────────────────────────────────────

export interface RULPrediction {
  predictedRul: number;
  uncertaintyLower: number;
  uncertaintyUpper: number;
  epistemicUncertainty: number;
  aleatoricUncertainty: number;
  healthIndex: number;
  degradationRate: number;
  failureProbability30d: number;
  attentionWeights: number[];
  modelName: string;
  inferenceLatencyMs: number;
  timestamp: number;
}

export interface FaultDiagnosis {
  predictedClass: string;
  predictedIndex: number;
  classProbabilities: Record<string, number>;
  confidence: number;
  camHeatmap: number[][];
  gradCam: number[][];
  shapeValues: number[] | null;
  severity: number;
  inferenceLatencyMs: number;
  timestamp: number;
}

export interface AnomalyDetection {
  isAnomaly: boolean;
  anomalyScore: number;
  threshold: number;
  reconstructionError: number;
  timestamps: number[];
  anomalyRegions: Array<{
    startIdx: number;
    endIdx: number;
    score: number;
    severity: string;
  }>;
}

export interface PredictionResult {
  assetId: string;
  rul: RULPrediction;
  faultDiagnosis: FaultDiagnosis;
  anomaly: AnomalyDetection;
  overallHealthScore: number;
  maintenanceUrgency: "none" | "monitor" | "schedule" | "immediate" | "emergency";
  recommendedActions: string[];
}

// ── Digital Twin Types ────────────────────────────────────────────────────────

export interface TwinState {
  twinId: string;
  assetId: string;
  timestamp: number;
  simulatedHours: number;
  bearingHealth: number;
  lubricationQuality: number;
  thermalStress: number;
  fatigueDamage: number;
  faultType: FaultType;
  faultSeverity: number;
  state: AssetState;
  rulHours: number;
  temperatureC: number;
  shaftRpm: number;
  powerKw: number;
  vibrationRmsG: number;
  degradationTrajectory: number[];
}

export interface TwinConfig {
  assetId: string;
  assetName: string;
  shaftRpm: number;
  ratedPowerKw: number;
  designLifeHours: number;
  simulationSpeed: number;
  noiseType: "gaussian" | "thermal" | "impulse" | "quantization";
  initialDegradation: number;
}

// ── Edge AI Types ─────────────────────────────────────────────────────────────

export interface EdgeBenchmark {
  modelName: string;
  modelType: string;
  precisionMode: "fp32" | "fp16" | "int8";
  latencyMs: {
    p50: number;
    p95: number;
    p99: number;
    mean: number;
    std: number;
  };
  throughputSamplesPerSec: number;
  memoryMb: number;
  flops: number;
  parameterCount: number;
  compressionRatio: number;
  accuracyDrop: number;
  powerWatts: number;
  efficiencyScore: number;
}

export interface EdgeDeploymentProfile {
  deviceType: string;
  deviceName: string;
  cpuCores: number;
  ramMb: number;
  hasGpu: boolean;
  hasFpga: boolean;
  maxPowerWatts: number;
  benchmarks: EdgeBenchmark[];
  recommendedModel: string;
}

// ── MLOps Types ───────────────────────────────────────────────────────────────

export interface MLExperiment {
  runId: string;
  experimentName: string;
  modelType: string;
  status: "running" | "finished" | "failed";
  startTime: number;
  endTime: number | null;
  metrics: {
    rmse?: number;
    mae?: number;
    nasaScore?: number;
    r2?: number;
    accuracy?: number;
    f1Macro?: number;
    aucRoc?: number;
  };
  params: Record<string, string | number>;
  tags: Record<string, string>;
  artifactPath: string;
}

export interface ModelInfo {
  modelId: string;
  name: string;
  version: string;
  stage: "staging" | "production" | "archived";
  createdAt: number;
  metrics: Record<string, number>;
  description: string;
  tags: Record<string, string>;
}

export interface DriftReport {
  modelId: string;
  timestamp: number;
  psiScore: number;
  ksPValue: number;
  driftDetected: boolean;
  driftSeverity: "none" | "minor" | "moderate" | "severe";
  affectedFeatures: string[];
  retrainingRecommended: boolean;
}

// ── Reliability Engineering Types ────────────────────────────────────────────

export interface WeibullAnalysis {
  shapeParameter: number;     // β
  scaleParameter: number;     // η (hours)
  locationParameter: number;  // γ (minimum life)
  mttf: number;               // Mean Time to Failure (hours)
  b10Life: number;            // Time to 10% failure probability
  b50Life: number;
  reliabilityAtT: (t: number) => number;
  hazardRateAtT: (t: number) => number;
  reliabilityCurve: Array<{ t: number; reliability: number; hazardRate: number }>;
  confidenceLower: number[];
  confidenceUpper: number[];
}

export interface MaintenanceSchedule {
  assetId: string;
  assetName: string;
  currentState: AssetState;
  faultType: FaultType;
  rulHours: number;
  priority: number; // 0-10
  recommendedAction: string;
  estimatedMaintenanceCostK: number;
  costOfFailureK: number;
  optimalMaintenanceWindowHours: number;
}

// ── Dashboard Types ───────────────────────────────────────────────────────────

export interface DashboardMetrics {
  totalAssets: number;
  assetsOnline: number;
  criticalAlerts: number;
  pendingMaintenance: number;
  avgFleetHealth: number;
  mtbf: number;
  mttr: number;
  oee: number;       // Overall Equipment Effectiveness
  uptimePercent: number;
  failuresPrevented: number;
  costSavingsK: number;
}

export interface Alert {
  id: string;
  assetId: string;
  assetName: string;
  type: "anomaly" | "fault" | "rul_warning" | "temperature" | "vibration";
  severity: "info" | "warning" | "critical" | "emergency";
  message: string;
  value: number;
  threshold: number;
  timestamp: number;
  acknowledged: boolean;
  resolvedAt: number | null;
}

export interface ChartDataPoint {
  timestamp: number;
  value: number;
  label?: string;
  color?: string;
}

export interface HeatmapCell {
  x: number; // time bin
  y: number; // frequency bin
  value: number;
}

// ── WebSocket Message Types ───────────────────────────────────────────────────

export type WSMessageType =
  | "telemetry"
  | "anomaly"
  | "rul_update"
  | "fault_detection"
  | "twin_state"
  | "alert"
  | "heartbeat";

export interface WSMessage<T = unknown> {
  type: WSMessageType;
  assetId?: string;
  timestamp: number;
  data: T;
}
