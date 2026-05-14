"""
Industrial PM Platform — Signal Processing API Endpoints

REST API for real-time signal processing:
- FFT spectral analysis
- Wavelet decomposition
- Bearing fault detection
- Spectral anomaly mapping
- Kurtogram computation
"""

from __future__ import annotations

import asyncio
import time
from typing import Annotated

import numpy as np
from fastapi import APIRouter, BackgroundTasks, Body, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from app.core.config import settings
from app.core.logging import get_logger
from app.services.signal.fft_service import FFTService
from app.services.signal.wavelet_service import WaveletService

router = APIRouter(prefix="/signal", tags=["signal"])
logger = get_logger(__name__)

# ── Shared service instances (initialized at startup) ─────────────────────────
_fft_service: FFTService | None = None
_wavelet_service: WaveletService | None = None


def get_fft_service() -> FFTService:
    global _fft_service
    if _fft_service is None:
        _fft_service = FFTService(
            sampling_rate=settings.SAMPLING_RATE_HZ,
            window_size=settings.FFT_WINDOW_SIZE,
            overlap=settings.FFT_OVERLAP,
        )
    return _fft_service


def get_wavelet_service() -> WaveletService:
    global _wavelet_service
    if _wavelet_service is None:
        _wavelet_service = WaveletService(
            sampling_rate=settings.SAMPLING_RATE_HZ,
            wavelet_family=settings.WAVELET_FAMILY,
        )
    return _wavelet_service


# ── Request / Response Schemas ────────────────────────────────────────────────

class SignalInput(BaseModel):
    signal: list[float] = Field(..., description="Raw signal samples")
    sampling_rate: int = Field(default=20000, ge=100, le=200000, description="Sampling rate in Hz")
    asset_id: str = Field(default="unknown", description="Asset identifier")

    @field_validator("signal")
    @classmethod
    def validate_signal_length(cls, v: list[float]) -> list[float]:
        if len(v) < 64:
            raise ValueError("Signal must have at least 64 samples")
        if len(v) > 131072:
            raise ValueError("Signal too long (max 131072 samples = 6.5s at 20kHz)")
        return v


class FFTAnalysisRequest(SignalInput):
    compute_stft: bool = Field(default=True, description="Compute STFT spectrogram")
    top_k_peaks: int = Field(default=10, ge=1, le=50)
    shaft_rpm: float | None = Field(default=None, ge=0, le=100000)


class WaveletAnalysisRequest(SignalInput):
    wavelet_family: str = Field(default="db8")
    n_levels: int = Field(default=6, ge=2, le=10)
    compute_cwt: bool = Field(default=True)
    detect_transients: bool = Field(default=True)


class BearingFaultRequest(SignalInput):
    shaft_rpm: float = Field(..., ge=1.0, le=100000.0, description="Shaft speed in RPM")
    bearing_type: str = Field(default="SKF_6205")
    method: str = Field(default="combined", pattern="^(fft|wavelet|combined)$")


class MultiChannelSignalInput(BaseModel):
    signals: dict[str, list[float]] = Field(
        ..., description="Dict of channel_name → signal samples"
    )
    sampling_rate: int = Field(default=20000)
    asset_id: str = Field(default="unknown")


# ── FFT Endpoints ─────────────────────────────────────────────────────────────

@router.post("/fft/analyze", summary="FFT Spectral Analysis")
async def fft_analyze(
    request: FFTAnalysisRequest,
    fft_svc: FFTService = Depends(get_fft_service),
) -> dict:
    """
    Perform complete FFT spectral analysis on a vibration signal.

    Returns:
    - Power spectral density (Welch method)
    - Dominant frequency peaks with amplitudes
    - Harmonic series analysis
    - Spectral features (entropy, centroid, bandwidth, etc.)
    - STFT spectrogram (optional)
    - Band powers (6 frequency bands)
    """
    t0 = time.perf_counter()

    try:
        signal_arr = np.array(request.signal, dtype=np.float32)

        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: fft_svc.analyze(
                signal_arr,
                shaft_rpm=request.shaft_rpm,
                return_stft=request.compute_stft,
                top_k_peaks=request.top_k_peaks,
            ),
        )

        response = {
            "asset_id": request.asset_id,
            "processing_time_ms": round((time.perf_counter() - t0) * 1000, 2),
            "signal_length": len(request.signal),
            "sampling_rate": request.sampling_rate,
            "frequency_resolution_hz": round(request.sampling_rate / settings.FFT_WINDOW_SIZE, 3),
            "frequencies": result.frequencies.tolist(),
            "power_spectral_density": result.power_spectral_density.tolist(),
            "dominant_frequencies": result.dominant_frequencies.tolist(),
            "dominant_amplitudes": result.dominant_amplitudes.tolist(),
            "spectral_features": {
                "entropy": round(float(result.spectral_entropy), 4),
                "centroid_hz": round(float(result.spectral_centroid), 2),
                "bandwidth_hz": round(float(result.spectral_bandwidth), 2),
                "rolloff_hz": round(float(result.spectral_rolloff), 2),
                "flatness": round(float(result.spectral_flatness), 4),
                "spectral_kurtosis": round(float(result.spectral_kurtosis), 4),
                "total_power_db": round(float(result.total_power), 2),
            },
            "time_domain_features": {
                "rms_g": round(float(result.rms), 4),
                "crest_factor": round(float(result.crest_factor), 4),
                "kurtosis": round(float(result.kurtosis), 4),
                "skewness": round(float(result.skewness), 4),
                "peak_to_peak": round(float(result.peak_to_peak), 4),
            },
            "band_powers_db": result.band_powers,
        }

        if request.compute_stft and result.stft_magnitude is not None:
            response["stft"] = {
                "frequencies": result.stft_frequencies.tolist(),
                "times": result.stft_times.tolist(),
                "magnitude": result.stft_magnitude.tolist(),
            }

        return response

    except Exception as e:
        logger.error("FFT analysis failed", error=str(e), asset_id=request.asset_id)
        raise HTTPException(status_code=422, detail=f"Signal processing failed: {str(e)}")


@router.post("/fft/bearing-faults", summary="Bearing Fault Frequency Detection")
async def detect_bearing_faults(
    request: BearingFaultRequest,
    fft_svc: FFTService = Depends(get_fft_service),
) -> dict:
    """
    Detect bearing fault signatures in vibration signal.

    Computes PSD and checks for spectral energy at bearing
    characteristic frequencies (BPFI, BPFO, BSF, FTF).

    Returns detection confidence, SNR, and severity for each fault type.
    """
    t0 = time.perf_counter()

    try:
        signal_arr = np.array(request.signal, dtype=np.float32)

        spectral_result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: fft_svc.analyze(signal_arr, shaft_rpm=request.shaft_rpm),
        )

        fault_detections = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: fft_svc.detect_bearing_faults(spectral_result, request.shaft_rpm),
        )

        # Determine overall fault verdict
        detected_faults = [
            name for name, det in fault_detections.items()
            if det.get("detected", False)
        ]
        max_confidence = max(
            (d["confidence"] for d in fault_detections.values()),
            default=0.0,
        )
        overall_severity = (
            "critical" if max_confidence > 0.8
            else "moderate" if max_confidence > 0.5
            else "incipient" if max_confidence > 0.3
            else "healthy"
        )

        shaft_hz = request.shaft_rpm / 60.0
        bearing_freqs = {
            "BPFI": round(7.29 * shaft_hz, 2),
            "BPFO": round(5.42 * shaft_hz, 2),
            "BSF": round(2.36 * shaft_hz, 2),
            "FTF": round(0.38 * shaft_hz, 2),
        }

        return {
            "asset_id": request.asset_id,
            "shaft_rpm": request.shaft_rpm,
            "processing_time_ms": round((time.perf_counter() - t0) * 1000, 2),
            "characteristic_frequencies_hz": bearing_freqs,
            "fault_detections": fault_detections,
            "detected_fault_types": detected_faults,
            "overall_max_confidence": round(float(max_confidence), 3),
            "overall_severity": overall_severity,
            "spectral_kurtosis": round(float(spectral_result.spectral_kurtosis), 3),
            "crest_factor": round(float(spectral_result.crest_factor), 3),
        }

    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.post("/fft/anomaly-map", summary="Spectral Anomaly Map")
async def compute_anomaly_map(
    request: SignalInput,
    z_score_threshold: float = Query(default=3.0, ge=1.0, le=10.0),
    fft_svc: FFTService = Depends(get_fft_service),
) -> dict:
    """
    Compute time-frequency anomaly map using STFT + Z-score analysis.

    Returns per-bin anomaly scores and detected anomaly regions.
    """
    t0 = time.perf_counter()

    try:
        signal_arr = np.array(request.signal, dtype=np.float32)

        anomaly_map = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: fft_svc.compute_spectral_anomaly_map(
                signal_arr, z_score_threshold=z_score_threshold
            ),
        )

        return {
            "asset_id": request.asset_id,
            "processing_time_ms": round((time.perf_counter() - t0) * 1000, 2),
            "frequencies": anomaly_map.frequencies.tolist(),
            "times": anomaly_map.times.tolist(),
            "anomaly_scores": anomaly_map.anomaly_scores.tolist(),
            "anomaly_threshold": anomaly_map.anomaly_threshold,
            "anomaly_regions": anomaly_map.anomaly_regions,
            "overall_severity": round(float(anomaly_map.severity), 3),
            "n_anomaly_regions": len(anomaly_map.anomaly_regions),
        }

    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.post("/fft/kurtogram", summary="Fast Kurtogram")
async def compute_kurtogram(
    request: SignalInput,
    n_levels: int = Query(default=3, ge=1, le=5),
    fft_svc: FFTService = Depends(get_fft_service),
) -> dict:
    """
    Compute Fast Kurtogram for optimal bandpass filter selection.

    Identifies frequency band with highest impulsive content
    (spectral kurtosis) for bearing fault envelope analysis.
    """
    signal_arr = np.array(request.signal, dtype=np.float32)
    kurtogram = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: fft_svc.compute_kurtogram(signal_arr, n_levels=n_levels),
    )
    return {
        "asset_id": request.asset_id,
        "kurtogram": kurtogram,
        "recommendation": (
            f"Apply bandpass filter at {kurtogram['optimal_band']['center_hz']:.0f} Hz "
            f"±{kurtogram['optimal_band']['bandwidth_hz']/2:.0f} Hz for envelope analysis"
        ),
    }


@router.post("/fft/cepstrum", summary="Cepstral Analysis")
async def compute_cepstrum(
    request: SignalInput,
    fft_svc: FFTService = Depends(get_fft_service),
) -> dict:
    """
    Compute real cepstrum for gear/bearing sideband analysis.
    c[n] = IFFT{log|FFT{x[n]}|}
    """
    signal_arr = np.array(request.signal, dtype=np.float32)
    quefrency, cepstrum = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: fft_svc.compute_cepstrum(signal_arr),
    )
    return {
        "asset_id": request.asset_id,
        "quefrency_s": quefrency[:len(quefrency)//2].tolist(),
        "cepstrum": cepstrum[:len(cepstrum)//2].tolist(),
    }


# ── Wavelet Endpoints ─────────────────────────────────────────────────────────

@router.post("/wavelet/analyze", summary="Wavelet Analysis")
async def wavelet_analyze(
    request: WaveletAnalysisRequest,
    wavelet_svc: WaveletService = Depends(get_wavelet_service),
) -> dict:
    """
    Complete wavelet analysis: CWT scalogram + DWT decomposition.

    Returns:
    - Energy per DWT level
    - Wavelet entropy
    - Transient event locations
    - Level-wise kurtosis (fault indicator)
    - Denoised signal
    - CWT scalogram power (optional)
    """
    t0 = time.perf_counter()

    try:
        signal_arr = np.array(request.signal, dtype=np.float32)

        result = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: wavelet_svc.analyze(
                signal_arr,
                compute_cwt=request.compute_cwt,
                compute_transients=request.detect_transients,
            ),
        )

        response = {
            "asset_id": request.asset_id,
            "processing_time_ms": round((time.perf_counter() - t0) * 1000, 2),
            "wavelet_family": result.wavelet_family,
            "n_levels": result.dwt_levels,
            "energy_per_level": result.energy_per_level.tolist(),
            "energy_ratios": result.energy_ratios.tolist(),
            "wavelet_entropy": round(float(result.wavelet_entropy), 4),
            "level_statistics": {
                "rms": result.level_rms.tolist(),
                "kurtosis": result.level_kurtosis.tolist(),
                "variance": result.level_variance.tolist(),
            },
            "transients": {
                "count": len(result.transient_locations),
                "locations_samples": result.transient_locations.tolist(),
                "amplitudes": result.transient_amplitudes.tolist(),
                "durations_samples": result.transient_durations.tolist(),
            },
            "denoising": {
                "snr_db": round(float(result.snr_db), 2),
                "denoised_signal": result.denoised_signal.tolist(),
            },
        }

        if request.compute_cwt and result.cwt_power.shape[0] > 1:
            response["scalogram"] = {
                "frequencies_hz": result.frequencies.tolist(),
                "power": result.cwt_power.tolist(),
            }

        return response

    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.post("/wavelet/bearing-faults", summary="Wavelet Bearing Fault Detection")
async def wavelet_bearing_faults(
    request: BearingFaultRequest,
    wavelet_svc: WaveletService = Depends(get_wavelet_service),
) -> dict:
    """
    Detect bearing faults using wavelet multi-resolution analysis.
    Complements FFT-based detection with time-frequency localization.
    """
    signal_arr = np.array(request.signal, dtype=np.float32)

    result = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: wavelet_svc.analyze(signal_arr, compute_cwt=False),
    )

    detections = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: wavelet_svc.detect_bearing_fault_wavelets(result, request.shaft_rpm),
    )

    return {
        "asset_id": request.asset_id,
        "shaft_rpm": request.shaft_rpm,
        "fault_detections": detections,
        "wavelet_entropy": round(float(result.wavelet_entropy), 4),
        "max_kurtosis": round(float(np.max(result.level_kurtosis)), 3),
    }


# ── Multi-sensor Endpoint ─────────────────────────────────────────────────────

@router.post("/multi-channel/analyze", summary="Multi-Channel Signal Analysis")
async def multi_channel_analyze(
    request: MultiChannelSignalInput,
    fft_svc: FFTService = Depends(get_fft_service),
) -> dict:
    """
    Analyze multiple sensor channels simultaneously.
    Returns per-channel spectral features for sensor fusion.
    """
    results = {}

    for channel_name, signal_samples in request.signals.items():
        try:
            signal_arr = np.array(signal_samples, dtype=np.float32)
            spectral = fft_svc.analyze(signal_arr)
            results[channel_name] = {
                "rms": round(float(spectral.rms), 4),
                "kurtosis": round(float(spectral.kurtosis), 4),
                "crest_factor": round(float(spectral.crest_factor), 4),
                "spectral_entropy": round(float(spectral.spectral_entropy), 4),
                "dominant_freq_hz": round(float(spectral.dominant_frequencies[0]), 2) if len(spectral.dominant_frequencies) > 0 else 0.0,
                "total_power_db": round(float(spectral.total_power), 2),
            }
        except Exception as e:
            results[channel_name] = {"error": str(e)}

    return {
        "asset_id": request.asset_id,
        "n_channels": len(request.signals),
        "channel_results": results,
    }
