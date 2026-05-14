"""
Industrial Signal Processing — Wavelet Intelligence Engine

Mathematical Foundation
-----------------------
Continuous Wavelet Transform (CWT):
    W_ψ(a, b) = (1/√a) ∫ x(t) · ψ*((t-b)/a) dt

where:
  a = scale parameter (inversely proportional to frequency)
  b = translation parameter (time localization)
  ψ = mother wavelet

Discrete Wavelet Transform (DWT) — Mallat algorithm:
    cA[n] = Σ_k h[k - 2n] · x[k]   (approximation, low-pass)
    cD[n] = Σ_k g[k - 2n] · x[k]   (detail, high-pass)

Wavelet Energy at level j:
    E_j = Σ_n |cD_j[n]|²

Wavelet Entropy:
    WE = -Σ_j p_j · log(p_j),  p_j = E_j / Σ E_j
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pywt
from scipy.interpolate import interp1d
from scipy.signal import correlate


@dataclass
class WaveletAnalysisResult:
    """Complete wavelet decomposition output."""
    
    # CWT scalogram
    scales: np.ndarray                  # Scale axis
    frequencies: np.ndarray             # Corresponding frequencies (Hz)
    cwt_coefficients: np.ndarray        # Shape: (n_scales, n_samples)
    cwt_power: np.ndarray               # |W|² scalogram
    
    # DWT decomposition
    dwt_coefficients: list[np.ndarray]  # [cA_N, cD_N, cD_{N-1}, ..., cD_1]
    dwt_levels: int
    dwt_frequency_bands: list[tuple]    # (f_lo, f_hi) for each level
    
    # Wavelet energy features
    energy_per_level: np.ndarray        # Energy at each DWT level
    energy_ratios: np.ndarray           # Normalized energy ratios
    wavelet_entropy: float              # Shannon entropy of energy distribution
    
    # Transient detection
    transient_locations: np.ndarray     # Sample indices of transients
    transient_amplitudes: np.ndarray    # Amplitude at each transient
    transient_durations: np.ndarray     # Duration in samples
    
    # Statistical features per level
    level_rms: np.ndarray
    level_kurtosis: np.ndarray
    level_variance: np.ndarray
    
    # Denoised signal
    denoised_signal: np.ndarray
    noise_estimate: np.ndarray
    snr_db: float
    
    # Metadata
    wavelet_family: str
    sampling_rate: int


@dataclass
class ScalogramAnnotation:
    """Annotations for scalogram visualization."""
    ridge_frequencies: np.ndarray       # Instantaneous frequency ridge
    ridge_times: np.ndarray
    ridge_amplitudes: np.ndarray
    
    impulse_locations: list[dict]       # [{time_s, freq_hz, amplitude, duration_ms}]
    modulation_frequency: float | None  # AM/FM modulation frequency
    chirp_rate: float | None            # Linear frequency sweep rate


class WaveletTransformEngine:
    """
    Production wavelet signal intelligence engine for industrial fault detection.
    
    Capabilities:
    - CWT scalogram with Morlet/Paul/DOG wavelets
    - DWT multi-resolution decomposition (Daubechies, Symlet, Coiflet)
    - Wavelet-based denoising (BayesShrink, SureShrink, VisuShrink)
    - Transient event localization
    - Wavelet energy entropy features
    - Multi-resolution bearing fault detection
    
    Usage
    -----
    engine = WaveletTransformEngine(sampling_rate=20000, wavelet_family='db8')
    result = engine.analyze(vibration_signal)
    features = engine.extract_features(result)
    """

    # Wavelet families for different industrial applications
    WAVELET_FAMILIES = {
        "bearing_fault": "db8",        # Good impulsive response
        "gear_fault": "db4",           # Moderate frequency resolution
        "unbalance": "sym8",           # Symmetric, good for sinusoids
        "misalignment": "coif3",       # Coiflet for smooth signals
        "transient": "meyer",          # Optimal frequency localization
        "general": "db8",
    }

    def __init__(
        self,
        sampling_rate: int = 20_000,
        wavelet_family: str = "db8",
        cwt_wavelet: str = "morl",     # Morlet for CWT
        n_dwt_levels: int = 6,
        n_cwt_scales: int = 128,
        denoising_method: Literal["bayes", "sure", "visu", "universal"] = "bayes",
        threshold_mode: Literal["soft", "hard"] = "soft",
    ) -> None:
        self.sampling_rate = sampling_rate
        self.wavelet_family = wavelet_family
        self.cwt_wavelet = cwt_wavelet
        self.n_dwt_levels = n_dwt_levels
        self.n_cwt_scales = n_cwt_scales
        self.denoising_method = denoising_method
        self.threshold_mode = threshold_mode
        self.dt = 1.0 / sampling_rate

        # Validate wavelet
        try:
            pywt.Wavelet(wavelet_family)
        except Exception as e:
            raise ValueError(f"Invalid wavelet family '{wavelet_family}': {e}")

        # Pre-compute CWT scales
        self._cwt_scales = self._compute_scales(n_cwt_scales, sampling_rate)

    def analyze(
        self,
        signal_data: np.ndarray,
        compute_cwt: bool = True,
        compute_transients: bool = True,
    ) -> WaveletAnalysisResult:
        """
        Complete wavelet analysis pipeline.
        
        Parameters
        ----------
        signal_data : np.ndarray
            Raw 1D vibration signal
        compute_cwt : bool
            Whether to compute CWT scalogram (expensive, O(N·S))
        compute_transients : bool
            Detect impulsive transient events
            
        Returns
        -------
        WaveletAnalysisResult
        """
        x = self._preprocess(signal_data)

        # ── DWT Decomposition ──────────────────────────────────────
        dwt_coeffs = pywt.wavedec(x, self.wavelet_family, level=self.n_dwt_levels)
        freq_bands = self._compute_dwt_frequency_bands()
        
        # ── Energy Analysis ────────────────────────────────────────
        energy_per_level = self._compute_energy_per_level(dwt_coeffs)
        energy_ratios = energy_per_level / (np.sum(energy_per_level) + 1e-12)
        wavelet_entropy = self._compute_wavelet_entropy(energy_ratios)
        
        # ── Statistical Features ───────────────────────────────────
        level_rms, level_kurtosis, level_variance = self._compute_level_statistics(dwt_coeffs)
        
        # ── Denoising ──────────────────────────────────────────────
        denoised, noise_est = self._denoise_signal(x, dwt_coeffs)
        signal_power = np.mean(denoised ** 2)
        noise_power = np.mean(noise_est ** 2) + 1e-12
        snr_db = float(10 * np.log10(signal_power / noise_power))
        
        # ── CWT Scalogram ──────────────────────────────────────────
        scales = np.zeros(1)
        freqs = np.zeros(1)
        cwt_coeffs = np.zeros((1, 1))
        cwt_power = np.zeros((1, 1))
        
        if compute_cwt:
            scales, freqs, cwt_coeffs, cwt_power = self._compute_cwt(x)
        
        # ── Transient Detection ────────────────────────────────────
        t_locs = np.array([])
        t_amps = np.array([])
        t_durs = np.array([])
        
        if compute_transients:
            t_locs, t_amps, t_durs = self._detect_transients(x, dwt_coeffs)
        
        return WaveletAnalysisResult(
            scales=scales,
            frequencies=freqs,
            cwt_coefficients=cwt_coeffs,
            cwt_power=cwt_power,
            dwt_coefficients=dwt_coeffs,
            dwt_levels=self.n_dwt_levels,
            dwt_frequency_bands=freq_bands,
            energy_per_level=energy_per_level,
            energy_ratios=energy_ratios,
            wavelet_entropy=wavelet_entropy,
            transient_locations=t_locs,
            transient_amplitudes=t_amps,
            transient_durations=t_durs,
            level_rms=level_rms,
            level_kurtosis=level_kurtosis,
            level_variance=level_variance,
            denoised_signal=denoised,
            noise_estimate=noise_est,
            snr_db=snr_db,
            wavelet_family=self.wavelet_family,
            sampling_rate=self.sampling_rate,
        )

    def detect_bearing_fault_wavelets(
        self,
        result: WaveletAnalysisResult,
        shaft_rpm: float,
        fault_types: list[str] | None = None,
    ) -> dict[str, dict]:
        """
        Wavelet-based bearing fault detection using multi-resolution analysis.
        
        Strategy:
        - Decompose to level where bearing fault frequency bands reside
        - Compute kurtosis of detail coefficients (impulsive content)
        - Correlate envelope of detail coefficients with fault frequencies
        """
        shaft_hz = shaft_rpm / 60.0
        fault_multipliers = {
            "BPFI": 7.29,
            "BPFO": 5.42,
            "BSF": 2.36,
            "FTF": 0.38,
        }
        
        if fault_types is None:
            fault_types = list(fault_multipliers.keys())
        
        detections = {}
        
        for fault_name in fault_types:
            mult = fault_multipliers.get(fault_name, 1.0)
            fault_freq = mult * shaft_hz
            
            # Find the DWT level corresponding to this frequency band
            target_level = self._find_fault_level(fault_freq)
            
            if target_level < len(result.dwt_coefficients):
                detail_coeffs = result.dwt_coefficients[-(target_level)]
                
                # Kurtosis of detail coefficients (>3 indicates impulsive content)
                kurt = float(self._kurtosis(detail_coeffs))
                
                # Envelope of detail coefficients
                from scipy.signal import hilbert
                envelope = np.abs(hilbert(detail_coeffs))
                
                # Autocorrelation to detect periodicity at fault frequency
                level_fs = self.sampling_rate / (2 ** target_level)
                period_samples = int(level_fs / (fault_freq + 1e-6))
                
                correlation_score = 0.0
                if 1 < period_samples < len(envelope) // 2:
                    lag = min(period_samples, len(envelope) // 2)
                    autocorr = correlate(envelope, envelope, mode="full")
                    mid = len(autocorr) // 2
                    autocorr = autocorr[mid:] / (autocorr[mid] + 1e-12)
                    if lag < len(autocorr):
                        correlation_score = float(autocorr[lag])
                
                energy_ratio = float(result.energy_ratios[min(target_level - 1, len(result.energy_ratios) - 1)])
                
                # Combined fault confidence
                confidence = float(np.clip(
                    0.4 * min(kurt / 10.0, 1.0) +
                    0.4 * max(0, correlation_score) +
                    0.2 * energy_ratio * 10,
                    0.0, 1.0
                ))
                
                detections[fault_name] = {
                    "fault_frequency_hz": float(fault_freq),
                    "wavelet_level": target_level,
                    "detail_kurtosis": kurt,
                    "correlation_score": float(max(0, correlation_score)),
                    "energy_ratio": energy_ratio,
                    "confidence": confidence,
                    "severity": self._classify_severity(kurt),
                    "detected": confidence > 0.35,
                }
        
        return detections

    def compute_instantaneous_frequency(
        self,
        result: WaveletAnalysisResult,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract instantaneous frequency ridge from CWT scalogram.
        
        The ridge corresponds to the scale (frequency) of maximum energy
        at each time point — useful for tracking speed variations in 
        rotating machinery.
        
        Returns (times, instantaneous_frequencies)
        """
        if result.cwt_power.shape[0] <= 1:
            n = len(result.dwt_coefficients[0])
            t = np.arange(n) * self.dt
            return t, np.ones(n) * self.sampling_rate / 4

        ridge_idx = np.argmax(result.cwt_power, axis=0)
        n_times = result.cwt_power.shape[1]
        times = np.arange(n_times) * self.dt
        
        ridge_freqs = result.frequencies[ridge_idx]
        
        # Smooth ridge
        from scipy.ndimage import uniform_filter1d
        ridge_freqs_smooth = uniform_filter1d(ridge_freqs.astype(float), size=5)
        
        return times, ridge_freqs_smooth

    def feature_vector(self, result: WaveletAnalysisResult) -> np.ndarray:
        """
        Extract fixed-length wavelet feature vector for ML models.
        
        Returns 48-dimensional feature vector:
        [energy_ratios × n_levels,
         level_rms × n_levels,
         level_kurtosis × n_levels,
         wavelet_entropy,
         snr_db,
         n_transients, mean_transient_amp,
         cwt_band_energies × 8]
        """
        features = list(result.energy_ratios[:self.n_dwt_levels])
        features += list(result.level_rms[:self.n_dwt_levels])
        features += list(result.level_kurtosis[:self.n_dwt_levels])
        features.append(result.wavelet_entropy)
        features.append(result.snr_db / 60.0)  # Normalize to ~[0,1]

        # Transient stats
        features.append(float(len(result.transient_locations)) / 100.0)
        features.append(float(np.mean(result.transient_amplitudes)) if len(result.transient_amplitudes) > 0 else 0.0)

        # CWT band energies (8 frequency bands)
        if result.cwt_power.shape[0] > 1:
            n_bands = 8
            band_size = result.cwt_power.shape[0] // n_bands
            for i in range(n_bands):
                lo = i * band_size
                hi = min((i + 1) * band_size, result.cwt_power.shape[0])
                band_energy = float(np.mean(result.cwt_power[lo:hi, :]))
                features.append(band_energy)
        else:
            features += [0.0] * 8

        # Pad or truncate to exactly 48
        features = features[:48]
        while len(features) < 48:
            features.append(0.0)

        return np.array(features, dtype=np.float32)

    # ── Private Methods ────────────────────────────────────────────

    def _preprocess(self, x: np.ndarray) -> np.ndarray:
        """Validate and detrend signal."""
        x = np.asarray(x, dtype=np.float64).ravel()
        if len(x) < 2 ** self.n_dwt_levels:
            raise ValueError(
                f"Signal length {len(x)} too short for {self.n_dwt_levels} DWT levels "
                f"(minimum {2 ** self.n_dwt_levels} samples)"
            )
        # Remove linear trend
        from scipy.signal import detrend
        return detrend(x, type="linear")

    def _compute_scales(
        self,
        n_scales: int,
        sampling_rate: int,
    ) -> np.ndarray:
        """Compute logarithmically spaced CWT scales."""
        # Scale range: cover 1 Hz to Nyquist/4
        f_max = sampling_rate / 4.0
        f_min = max(1.0, sampling_rate / (n_scales * 10))
        
        # For Morlet: scale ≈ sampling_rate / (2 * π * frequency)
        scale_max = sampling_rate / (2 * np.pi * f_min)
        scale_min = sampling_rate / (2 * np.pi * f_max)
        
        return np.geomspace(scale_min, scale_max, n_scales)

    def _compute_cwt(
        self,
        x: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute Continuous Wavelet Transform using PyWavelets."""
        # Limit signal length for CWT (expensive)
        max_cwt_samples = 4096
        if len(x) > max_cwt_samples:
            # Downsample for CWT computation
            step = len(x) // max_cwt_samples
            x_cwt = x[::step]
        else:
            x_cwt = x

        scales = self._cwt_scales
        
        try:
            coefficients, freqs = pywt.cwt(
                x_cwt,
                scales=scales,
                wavelet=self.cwt_wavelet,
                sampling_period=self.dt,
            )
        except Exception:
            # Fallback: use simpler CWT
            coefficients, freqs = pywt.cwt(
                x_cwt[:min(2048, len(x_cwt))],
                scales=scales[:min(64, len(scales))],
                wavelet="morl",
                sampling_period=self.dt,
            )

        power = np.abs(coefficients) ** 2
        
        return scales, freqs, coefficients, power

    def _compute_dwt_frequency_bands(self) -> list[tuple[float, float]]:
        """
        Compute frequency bands for each DWT decomposition level.
        
        Level j detail coefficients cover: [fs/(2^(j+1)), fs/2^j]
        """
        bands = []
        fs = float(self.sampling_rate)
        for level in range(1, self.n_dwt_levels + 1):
            f_hi = fs / (2 ** level)
            f_lo = fs / (2 ** (level + 1))
            bands.append((f_lo, f_hi))
        # Approximation at top level
        bands.append((0.0, fs / (2 ** (self.n_dwt_levels + 1))))
        return bands

    def _compute_energy_per_level(
        self,
        coefficients: list[np.ndarray],
    ) -> np.ndarray:
        """Compute signal energy at each decomposition level."""
        energies = np.array([
            float(np.sum(c ** 2)) for c in coefficients
        ])
        return energies

    def _compute_wavelet_entropy(self, energy_ratios: np.ndarray) -> float:
        """
        Shannon wavelet entropy: WE = -Σ p_j · log(p_j)
        
        Low entropy → energy concentrated in few bands (fault-like)
        High entropy → energy spread across bands (noise-like)
        """
        p = energy_ratios / (np.sum(energy_ratios) + 1e-12)
        entropy = -float(np.sum(p * np.log(p + 1e-12)))
        return entropy

    def _compute_level_statistics(
        self,
        coefficients: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute RMS, kurtosis, and variance for each DWT level."""
        level_rms = np.array([float(np.sqrt(np.mean(c ** 2))) for c in coefficients])
        level_kurtosis = np.array([float(self._kurtosis(c)) for c in coefficients])
        level_variance = np.array([float(np.var(c)) for c in coefficients])
        return level_rms, level_kurtosis, level_variance

    def _denoise_signal(
        self,
        x: np.ndarray,
        coefficients: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Wavelet-based signal denoising with threshold selection.
        
        Methods:
        - VisuShrink: λ = σ√(2 log N)
        - BayesShrink: level-dependent threshold
        - SureShrink: SURE-optimal threshold
        """
        sigma_mad = self._estimate_noise_mad(coefficients[-1])  # Finest detail
        
        # Copy coefficients
        denoised_coeffs = [c.copy() for c in coefficients]
        
        for level in range(1, len(coefficients)):
            detail = coefficients[level]
            
            if self.denoising_method == "visu":
                threshold = sigma_mad * np.sqrt(2 * np.log(len(x) + 1e-6))
            elif self.denoising_method == "bayes":
                # BayesShrink: level-specific threshold
                sigma_level = self._estimate_noise_mad(detail)
                signal_var = max(np.var(detail) - sigma_level ** 2, 1e-12)
                threshold = sigma_level ** 2 / np.sqrt(signal_var)
            elif self.denoising_method == "sure":
                threshold = self._sure_threshold(detail, sigma_mad)
            else:  # universal
                threshold = sigma_mad * np.sqrt(2 * np.log(len(detail) + 1e-6))
            
            denoised_coeffs[level] = pywt.threshold(
                detail,
                value=threshold,
                mode=self.threshold_mode,
            )
        
        denoised = pywt.waverec(denoised_coeffs, self.wavelet_family)
        # Ensure same length as input
        denoised = denoised[:len(x)]
        
        noise_est = x - denoised
        return denoised, noise_est

    def _detect_transients(
        self,
        x: np.ndarray,
        coefficients: list[np.ndarray],
        kurtosis_threshold: float = 4.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect impulsive transient events using DWT detail coefficients.
        
        Transients manifest as localized high-energy bursts in fine-scale
        detail coefficients — characteristic of bearing spalling impacts.
        """
        from scipy.signal import find_peaks
        
        # Use finest detail level for transient detection
        detail = coefficients[-1]  # Highest frequency band
        
        # Compute local energy envelope
        energy = detail ** 2
        window = max(10, len(detail) // 100)
        
        from scipy.ndimage import uniform_filter1d
        smooth_energy = uniform_filter1d(energy, size=window)
        
        # Adaptive threshold: mean + k×std
        threshold = np.mean(smooth_energy) + kurtosis_threshold * np.std(smooth_energy)
        
        peaks, properties = find_peaks(
            smooth_energy,
            height=threshold,
            distance=window,
            prominence=threshold * 0.3,
        )
        
        if len(peaks) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # Scale indices back to original signal
        scale_factor = len(x) / len(detail)
        original_indices = (peaks * scale_factor).astype(int)
        original_indices = np.clip(original_indices, 0, len(x) - 1)
        
        amplitudes = smooth_energy[peaks]
        
        # Estimate durations (width at half maximum)
        widths = np.ones(len(peaks)) * window
        for i, peak in enumerate(peaks):
            half_height = smooth_energy[peak] / 2
            lo = peak
            hi = peak
            while lo > 0 and smooth_energy[lo] > half_height:
                lo -= 1
            while hi < len(smooth_energy) - 1 and smooth_energy[hi] > half_height:
                hi += 1
            widths[i] = (hi - lo) * scale_factor
        
        durations = widths.astype(int)
        
        return original_indices, amplitudes, durations

    def _estimate_noise_mad(self, detail_coeffs: np.ndarray) -> float:
        """
        Estimate noise standard deviation using Median Absolute Deviation (MAD).
        
        σ̂ = median(|d|) / 0.6745
        
        Robust estimator that works even when signal contains impulsive content.
        """
        return float(np.median(np.abs(detail_coeffs)) / 0.6745)

    def _sure_threshold(self, coeffs: np.ndarray, sigma: float) -> float:
        """
        Compute SURE (Stein's Unbiased Risk Estimator) optimal threshold.
        
        Minimizes the expected mean-squared error of the threshold estimator.
        """
        n = len(coeffs)
        normalized = np.sort(np.abs(coeffs) / sigma) ** 2
        
        risks = np.zeros(n)
        for i in range(n):
            t2 = normalized[i]
            risks[i] = (n - 2 * (i + 1) + 
                       np.sum(np.minimum(normalized, t2)))
        
        best_idx = np.argmin(risks)
        return float(sigma * np.sqrt(normalized[best_idx]))

    def _find_fault_level(self, fault_freq: float) -> int:
        """Find the DWT level whose frequency band contains the fault frequency."""
        for level in range(1, self.n_dwt_levels + 1):
            f_hi = self.sampling_rate / (2 ** level)
            f_lo = self.sampling_rate / (2 ** (level + 1))
            if f_lo <= fault_freq <= f_hi:
                return level
        return self.n_dwt_levels  # Default to coarsest level

    def _kurtosis(self, x: np.ndarray) -> float:
        """Fisher's kurtosis (excess kurtosis, normal=0)."""
        if len(x) < 4:
            return 0.0
        mu = np.mean(x)
        sigma = np.std(x) + 1e-12
        return float(np.mean(((x - mu) / sigma) ** 4) - 3)

    def _classify_severity(self, kurtosis: float) -> str:
        """Classify fault severity from kurtosis value."""
        if kurtosis < 3:
            return "healthy"
        elif kurtosis < 5:
            return "incipient"
        elif kurtosis < 10:
            return "moderate"
        elif kurtosis < 20:
            return "severe"
        else:
            return "critical"
