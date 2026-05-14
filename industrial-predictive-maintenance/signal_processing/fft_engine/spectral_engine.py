"""
Industrial Signal Processing — FFT Spectral Intelligence Engine

Production-grade spectral analysis engine implementing:

Mathematical Foundation
-----------------------
DFT:     X[k] = Σ_{n=0}^{N-1} x[n] · e^{-j2πkn/N}
STFT:    X(τ,ω) = ∫ x(t)·w(t-τ)·e^{-jωt} dt
PSD:     S_xx(f) = (1/KU) · Σ|X_k(f)|²
Entropy: H_s = -Σ p(f)·log₂p(f)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import scipy.signal as signal
import scipy.stats as stats
from numpy.fft import fft, fftfreq, rfft, rfftfreq
from scipy.signal import welch, stft, find_peaks, peak_prominences

from signal_processing.feature_extraction.industrial_features import IndustrialFeatureSet


@dataclass
class SpectralAnalysisResult:
    """Complete spectral analysis output for one signal segment."""
    
    # Frequency domain
    frequencies: np.ndarray                 # Hz axis
    power_spectral_density: np.ndarray       # PSD (Welch)
    fft_magnitude: np.ndarray               # Raw FFT magnitude
    fft_phase: np.ndarray                   # Phase spectrum
    
    # STFT
    stft_frequencies: np.ndarray
    stft_times: np.ndarray
    stft_magnitude: np.ndarray              # Shape: (freq_bins, time_frames)
    
    # Peak analysis
    dominant_frequencies: np.ndarray        # Top-K peaks
    dominant_amplitudes: np.ndarray
    harmonic_frequencies: list[np.ndarray]  # Harmonics for each dominant freq
    harmonic_amplitudes: list[np.ndarray]
    
    # Statistical spectral features
    spectral_entropy: float
    spectral_centroid: float                # Hz
    spectral_bandwidth: float               # Hz
    spectral_rolloff: float                 # Hz (85% energy)
    spectral_flatness: float                # Wiener entropy
    spectral_kurtosis: float
    
    # Energy distribution
    total_power: float                      # dB
    band_powers: dict[str, float]           # Power in frequency bands
    
    # Industrial fault indicators
    rms: float
    crest_factor: float
    kurtosis: float
    skewness: float
    peak_to_peak: float
    
    # Metadata
    segment_length: int
    sampling_rate: int
    frequency_resolution: float
    window_function: str


@dataclass
class HarmonicAnalysis:
    """Result of harmonic series analysis."""
    fundamental_frequency: float           # Hz
    harmonics: list[float]                 # Frequencies of harmonics
    amplitudes: list[float]                # Amplitudes at each harmonic
    total_harmonic_distortion: float       # THD
    harmonic_energy_ratio: float           # Fraction of energy in harmonics
    snr_db: float                          # Signal-to-noise ratio


@dataclass  
class SpectralAnomalyMap:
    """Spectral anomaly localization output."""
    frequencies: np.ndarray
    times: np.ndarray
    anomaly_scores: np.ndarray             # Shape: (freq_bins, time_frames)
    anomaly_threshold: float
    anomaly_regions: list[dict]            # [{t_start, t_end, f_start, f_end, score}]
    severity: float                        # 0-1 normalized overall severity


class FFTSpectralEngine:
    """
    Production FFT spectral intelligence engine for industrial vibration analysis.
    
    Implements full spectral pipeline:
    1. Signal preprocessing (detrending, windowing)
    2. FFT / STFT / Welch PSD computation  
    3. Peak detection and harmonic analysis
    4. Spectral feature extraction
    5. Fault frequency detection
    6. Spectral anomaly mapping
    
    Usage
    -----
    engine = FFTSpectralEngine(sampling_rate=20000, window_size=1024)
    result = engine.analyze(vibration_signal)
    fault_indicators = engine.detect_bearing_faults(result, shaft_rpm=1800)
    """

    # Bearing fault frequency multipliers (dimensionless)
    BEARING_FAULT_MULTIPLIERS = {
        "BPFI": 7.29,   # Ball Pass Frequency Inner race
        "BPFO": 5.42,   # Ball Pass Frequency Outer race
        "BSF":  2.36,   # Ball Spin Frequency
        "FTF":  0.38,   # Fundamental Train Frequency
    }

    # Standard industrial frequency bands
    FREQUENCY_BANDS = {
        "sub_synchronous": (0.1, 1.0),      # Normalized to shaft freq
        "synchronous": (0.9, 1.1),
        "super_synchronous": (1.1, 10.0),
        "bearing_zone": (1000, 10000),      # Hz — absolute
        "ultrasonic": (10000, 40000),
    }

    def __init__(
        self,
        sampling_rate: int = 20_000,
        window_size: int = 1024,
        overlap: float = 0.5,
        window_function: str = "hann",
        n_harmonics: int = 10,
        peak_prominence_threshold: float = 0.05,
        spectral_resolution_hz: float | None = None,
    ) -> None:
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.overlap = overlap
        self.window_function = window_function
        self.n_harmonics = n_harmonics
        self.peak_prominence_threshold = peak_prominence_threshold
        self.hop_size = int(window_size * (1 - overlap))
        self.frequency_resolution = sampling_rate / window_size
        self.nyquist = sampling_rate / 2.0

        # Pre-compute window function
        self._window = self._create_window(window_function, window_size)

        # Frequency axis for full FFT
        self._freq_axis = rfftfreq(window_size, d=1.0 / sampling_rate)

        if spectral_resolution_hz is not None:
            required_window = int(sampling_rate / spectral_resolution_hz)
            if required_window > window_size:
                warnings.warn(
                    f"Window size {window_size} gives resolution "
                    f"{self.frequency_resolution:.2f} Hz, "
                    f"but {spectral_resolution_hz:.2f} Hz requested. "
                    f"Consider window_size={required_window}."
                )

    def analyze(
        self,
        signal_data: np.ndarray,
        shaft_rpm: float | None = None,
        return_stft: bool = True,
        top_k_peaks: int = 10,
    ) -> SpectralAnalysisResult:
        """
        Complete spectral analysis of a vibration signal.
        
        Parameters
        ----------
        signal_data : np.ndarray
            Raw vibration signal (1D), samples
        shaft_rpm : float, optional
            Shaft rotational speed for fault frequency computation
        return_stft : bool
            Whether to compute STFT spectrogram
        top_k_peaks : int
            Number of dominant frequency peaks to return
            
        Returns
        -------
        SpectralAnalysisResult
        """
        x = self._preprocess(signal_data)
        N = len(x)

        # ── FFT ────────────────────────────────────────────────────
        X = rfft(x * self._window[:N] if N <= self.window_size else rfft(x[:self.window_size] * self._window))
        if N > self.window_size:
            # Overlap-add for full signal FFT
            X = self._overlap_add_fft(x)
        
        magnitudes = np.abs(X)
        phases = np.angle(X)
        freqs = rfftfreq(len(x) if N <= self.window_size else self.window_size,
                         d=1.0 / self.sampling_rate)

        # ── Welch PSD ──────────────────────────────────────────────
        nperseg = min(self.window_size, N)
        noverlap = int(nperseg * self.overlap)
        psd_freqs, psd = welch(
            x,
            fs=self.sampling_rate,
            window=self.window_function,
            nperseg=nperseg,
            noverlap=noverlap,
            scaling="density",
            average="mean",
        )

        # ── STFT ───────────────────────────────────────────────────
        stft_f, stft_t, stft_Z = np.array([]), np.array([]), np.zeros((1, 1))
        if return_stft:
            stft_f, stft_t, stft_Z = stft(
                x,
                fs=self.sampling_rate,
                window=self.window_function,
                nperseg=min(self.window_size, N),
                noverlap=int(min(self.window_size, N) * self.overlap),
            )
            stft_Z = np.abs(stft_Z)

        # ── Peak Detection ─────────────────────────────────────────
        peak_freqs, peak_amps = self._detect_spectral_peaks(psd_freqs, psd, top_k_peaks)

        # ── Harmonic Analysis ──────────────────────────────────────
        harmonic_freqs, harmonic_amps = [], []
        for f0 in peak_freqs[:3]:  # Analyze harmonics for top-3 peaks
            hf, ha = self._compute_harmonics(psd_freqs, psd, f0)
            harmonic_freqs.append(hf)
            harmonic_amps.append(ha)

        # ── Spectral Features ──────────────────────────────────────
        features = self._compute_spectral_features(psd_freqs, psd, x)

        # ── Band Powers ────────────────────────────────────────────
        band_powers = self._compute_band_powers(psd_freqs, psd)

        return SpectralAnalysisResult(
            frequencies=psd_freqs,
            power_spectral_density=psd,
            fft_magnitude=magnitudes,
            fft_phase=phases,
            stft_frequencies=stft_f,
            stft_times=stft_t,
            stft_magnitude=stft_Z,
            dominant_frequencies=peak_freqs,
            dominant_amplitudes=peak_amps,
            harmonic_frequencies=harmonic_freqs,
            harmonic_amplitudes=harmonic_amps,
            spectral_entropy=features["entropy"],
            spectral_centroid=features["centroid"],
            spectral_bandwidth=features["bandwidth"],
            spectral_rolloff=features["rolloff"],
            spectral_flatness=features["flatness"],
            spectral_kurtosis=features["spectral_kurtosis"],
            total_power=features["total_power_db"],
            band_powers=band_powers,
            rms=features["rms"],
            crest_factor=features["crest_factor"],
            kurtosis=features["kurtosis"],
            skewness=features["skewness"],
            peak_to_peak=features["peak_to_peak"],
            segment_length=len(x),
            sampling_rate=self.sampling_rate,
            frequency_resolution=self.frequency_resolution,
            window_function=self.window_function,
        )

    def detect_bearing_faults(
        self,
        result: SpectralAnalysisResult,
        shaft_rpm: float,
        tolerance_hz: float = 2.0,
        energy_threshold_db: float = -20.0,
    ) -> dict[str, dict]:
        """
        Detect bearing fault signatures by comparing PSD peaks to
        theoretical fault characteristic frequencies.
        
        Parameters
        ----------
        result : SpectralAnalysisResult
        shaft_rpm : float
            Shaft rotational speed in RPM
        tolerance_hz : float
            Frequency tolerance for fault detection (±Hz)
        energy_threshold_db : float
            Minimum energy threshold for fault confirmation
            
        Returns
        -------
        dict mapping fault type → {frequency, amplitude, confidence, severity}
        """
        shaft_hz = shaft_rpm / 60.0
        fault_detections = {}

        for fault_name, multiplier in self.BEARING_FAULT_MULTIPLIERS.items():
            target_freq = multiplier * shaft_hz
            
            # Find nearest PSD bin
            freq_idx = np.argmin(np.abs(result.frequencies - target_freq))
            
            # Check energy in ±tolerance window
            tol_bins = max(1, int(tolerance_hz / self.frequency_resolution))
            lo = max(0, freq_idx - tol_bins)
            hi = min(len(result.frequencies), freq_idx + tol_bins + 1)
            
            local_psd = result.power_spectral_density[lo:hi]
            local_freqs = result.frequencies[lo:hi]
            
            peak_energy = 10 * np.log10(np.max(local_psd) + 1e-12)
            noise_floor = 10 * np.log10(
                np.median(result.power_spectral_density) + 1e-12
            )
            snr = peak_energy - noise_floor

            # Confidence based on SNR above noise floor
            confidence = float(np.clip(snr / 30.0, 0.0, 1.0))  # 30 dB = 100% conf

            actual_peak_freq = local_freqs[np.argmax(local_psd)]
            
            fault_detections[fault_name] = {
                "target_frequency_hz": float(target_freq),
                "detected_frequency_hz": float(actual_peak_freq),
                "frequency_deviation_hz": float(abs(actual_peak_freq - target_freq)),
                "peak_amplitude_db": float(peak_energy),
                "noise_floor_db": float(noise_floor),
                "snr_db": float(snr),
                "confidence": float(confidence),
                "severity": self._classify_severity(snr),
                "detected": confidence > 0.3,
            }

        return fault_detections

    def compute_spectral_anomaly_map(
        self,
        signal_data: np.ndarray,
        baseline_psd: np.ndarray | None = None,
        z_score_threshold: float = 3.0,
    ) -> SpectralAnomalyMap:
        """
        Compute time-frequency anomaly map using STFT + statistical deviation.
        
        Anomaly score at each (freq, time) bin:
            score(f,t) = |S(f,t) - μ(f)| / σ(f)  [Z-score]
        """
        x = self._preprocess(signal_data)
        
        stft_f, stft_t, stft_Z = stft(
            x,
            fs=self.sampling_rate,
            window=self.window_function,
            nperseg=self.window_size,
            noverlap=int(self.window_size * self.overlap),
        )
        magnitude = np.abs(stft_Z)

        if baseline_psd is not None:
            # Compare against provided baseline
            baseline_mean = baseline_psd[:magnitude.shape[0]]
            baseline_std = np.sqrt(baseline_psd[:magnitude.shape[0]]) * 0.1
        else:
            # Estimate baseline from signal itself (first 20% = healthy)
            baseline_frames = max(1, int(0.2 * magnitude.shape[1]))
            baseline_mean = np.mean(magnitude[:, :baseline_frames], axis=1, keepdims=True)
            baseline_std = np.std(magnitude[:, :baseline_frames], axis=1, keepdims=True) + 1e-8

        # Z-score anomaly map
        z_scores = (magnitude - baseline_mean) / baseline_std
        anomaly_scores = np.maximum(0, z_scores)  # Only positive deviations

        # Threshold and find regions
        binary_anomaly = anomaly_scores > z_score_threshold
        anomaly_regions = self._extract_anomaly_regions(
            stft_f, stft_t, anomaly_scores, binary_anomaly
        )

        severity = float(np.mean(anomaly_scores[anomaly_scores > z_score_threshold]) 
                        if np.any(anomaly_scores > z_score_threshold) else 0.0)
        severity = float(np.clip(severity / 10.0, 0.0, 1.0))

        return SpectralAnomalyMap(
            frequencies=stft_f,
            times=stft_t,
            anomaly_scores=anomaly_scores,
            anomaly_threshold=z_score_threshold,
            anomaly_regions=anomaly_regions,
            severity=severity,
        )

    def compute_cepstrum(self, signal_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute real cepstrum for gear/bearing sideband analysis.
        
        c[n] = IFFT{log|FFT{x[n]}|}
        
        Quefrency peaks indicate:
        - Gear mesh frequency and sidebands
        - Bearing defect frequency harmonics
        """
        x = self._preprocess(signal_data)
        X = rfft(x)
        log_spectrum = np.log(np.abs(X) + 1e-10)
        cepstrum = np.real(np.fft.irfft(log_spectrum))
        quefrency = np.arange(len(cepstrum)) / self.sampling_rate
        return quefrency, cepstrum

    def extract_envelope_spectrum(
        self,
        signal_data: np.ndarray,
        bandpass_center: float = 5000.0,
        bandpass_width: float = 2000.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        High-Frequency Resonance Technique (HFRT) for bearing fault detection.
        
        Steps:
        1. Bandpass filter around resonance frequency
        2. Hilbert transform to get envelope
        3. FFT of envelope (ESVD — Envelope Spectrum)
        """
        from scipy.signal import butter, filtfilt, hilbert
        
        x = self._preprocess(signal_data)
        
        # Bandpass filter
        lo = (bandpass_center - bandpass_width / 2) / self.nyquist
        hi = (bandpass_center + bandpass_width / 2) / self.nyquist
        lo, hi = np.clip([lo, hi], 0.001, 0.999)
        
        b, a = butter(N=4, Wn=[lo, hi], btype="bandpass")
        filtered = filtfilt(b, a, x)
        
        # Hilbert envelope
        analytic = hilbert(filtered)
        envelope = np.abs(analytic)
        
        # Envelope spectrum
        env_psd_f, env_psd = welch(
            envelope,
            fs=self.sampling_rate,
            nperseg=min(self.window_size, len(envelope)),
            scaling="density",
        )
        
        return env_psd_f, env_psd

    def compute_kurtogram(
        self,
        signal_data: np.ndarray,
        n_levels: int = 3,
    ) -> dict:
        """
        Fast Kurtogram for optimal bandpass filter selection.
        Identifies the frequency band with highest spectral kurtosis,
        indicating impulsive fault content.
        
        Returns dict with optimal center_freq, bandwidth, kurtosis_value.
        """
        x = self._preprocess(signal_data)
        
        results = []
        for level in range(1, n_levels + 1):
            n_bands = 2 ** level
            band_width = self.nyquist / n_bands
            
            for band_idx in range(n_bands):
                center = (band_idx + 0.5) * band_width
                lo = max(0.001, (center - band_width / 2) / self.nyquist)
                hi = min(0.999, (center + band_width / 2) / self.nyquist)
                
                from scipy.signal import butter, filtfilt
                b, a = butter(N=2, Wn=[lo, hi], btype="bandpass")
                try:
                    filtered = filtfilt(b, a, x)
                    kurt = float(stats.kurtosis(filtered))
                    results.append({
                        "level": level,
                        "center_hz": center,
                        "bandwidth_hz": band_width,
                        "kurtosis": kurt,
                    })
                except Exception:
                    pass
        
        if results:
            best = max(results, key=lambda r: r["kurtosis"])
        else:
            best = {"level": 1, "center_hz": self.nyquist / 2, "bandwidth_hz": self.nyquist, "kurtosis": 0.0}
        
        return {
            "optimal_band": best,
            "all_bands": results,
            "kurtogram_levels": n_levels,
        }

    # ── Private Methods ────────────────────────────────────────────

    def _preprocess(self, x: np.ndarray) -> np.ndarray:
        """Detrend, normalize, and validate signal."""
        x = np.asarray(x, dtype=np.float64).ravel()
        if len(x) < 16:
            raise ValueError(f"Signal too short: {len(x)} samples (minimum 16)")
        # Remove DC component
        x = signal.detrend(x, type="linear")
        return x

    def _create_window(self, window_name: str, size: int) -> np.ndarray:
        """Create normalized window function."""
        windows = {
            "hann": np.hanning,
            "hamming": np.hamming,
            "blackman": np.blackman,
            "rectangular": np.ones,
            "flattop": lambda n: signal.windows.flattop(n),
        }
        fn = windows.get(window_name, np.hanning)
        w = fn(size)
        return w / np.sqrt(np.mean(w ** 2))  # Normalize for consistent amplitude

    def _overlap_add_fft(self, x: np.ndarray) -> np.ndarray:
        """Compute average FFT over overlapping segments."""
        N = len(x)
        hop = self.hop_size
        n_frames = (N - self.window_size) // hop + 1
        
        avg_mag = np.zeros(self.window_size // 2 + 1)
        for i in range(n_frames):
            start = i * hop
            segment = x[start:start + self.window_size]
            if len(segment) == self.window_size:
                X_seg = rfft(segment * self._window)
                avg_mag += np.abs(X_seg)
        
        return avg_mag / max(n_frames, 1)

    def _detect_spectral_peaks(
        self,
        freqs: np.ndarray,
        psd: np.ndarray,
        top_k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Detect prominent spectral peaks using scipy peak detection."""
        min_distance = max(1, int(1.0 / self.frequency_resolution))  # 1 Hz separation
        
        peaks, properties = find_peaks(
            psd,
            distance=min_distance,
            prominence=np.max(psd) * self.peak_prominence_threshold,
        )
        
        if len(peaks) == 0:
            # Fallback: return top-K frequency bins by amplitude
            top_idx = np.argsort(psd)[-top_k:][::-1]
            return freqs[top_idx], psd[top_idx]
        
        # Sort by amplitude, take top-K
        peak_amps = psd[peaks]
        sorted_idx = np.argsort(peak_amps)[-top_k:][::-1]
        top_peaks = peaks[sorted_idx]
        
        return freqs[top_peaks], psd[top_peaks]

    def _compute_harmonics(
        self,
        freqs: np.ndarray,
        psd: np.ndarray,
        fundamental: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract harmonic amplitudes at f0, 2f0, 3f0, ..., Nf0."""
        harmonic_f = np.array([fundamental * (n + 1) for n in range(self.n_harmonics)])
        harmonic_a = np.zeros(self.n_harmonics)
        
        for i, hf in enumerate(harmonic_f):
            if hf > self.nyquist:
                break
            idx = np.argmin(np.abs(freqs - hf))
            # Average over ±1 bin for stability
            lo = max(0, idx - 1)
            hi = min(len(psd), idx + 2)
            harmonic_a[i] = np.max(psd[lo:hi])
        
        return harmonic_f, harmonic_a

    def _compute_spectral_features(
        self,
        freqs: np.ndarray,
        psd: np.ndarray,
        time_signal: np.ndarray,
    ) -> dict[str, float]:
        """Compute comprehensive spectral and time-domain features."""
        # Normalize PSD to probability distribution
        psd_sum = np.sum(psd) + 1e-12
        p = psd / psd_sum

        # Spectral entropy: H = -Σ p(f)·log₂p(f)
        entropy = float(-np.sum(p * np.log2(p + 1e-12)))

        # Spectral centroid: μ = Σ f·p(f)
        centroid = float(np.sum(freqs * p))

        # Spectral bandwidth: σ = sqrt(Σ (f-μ)²·p(f))
        bandwidth = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * p)))

        # Spectral rolloff: 85% energy
        cumulative_power = np.cumsum(psd)
        rolloff_idx = np.searchsorted(cumulative_power, 0.85 * cumulative_power[-1])
        rolloff = float(freqs[min(rolloff_idx, len(freqs) - 1)])

        # Spectral flatness (Wiener entropy): geometric/arithmetic mean of PSD
        log_psd = np.log(psd + 1e-12)
        geometric_mean = np.exp(np.mean(log_psd))
        arithmetic_mean = np.mean(psd) + 1e-12
        flatness = float(geometric_mean / arithmetic_mean)

        # Spectral kurtosis
        mu_f = centroid
        sigma_f = bandwidth + 1e-12
        sp_kurt = float(np.sum(((freqs - mu_f) / sigma_f) ** 4 * p))

        # Total power (dB)
        total_power_db = float(10 * np.log10(np.trapz(psd, freqs) + 1e-12))

        # Time domain
        rms = float(np.sqrt(np.mean(time_signal ** 2)))
        peak = float(np.max(np.abs(time_signal)))
        crest_factor = float(peak / (rms + 1e-12))
        kurtosis = float(stats.kurtosis(time_signal))
        skewness = float(stats.skew(time_signal))
        p2p = float(np.max(time_signal) - np.min(time_signal))

        return {
            "entropy": entropy,
            "centroid": centroid,
            "bandwidth": bandwidth,
            "rolloff": rolloff,
            "flatness": flatness,
            "spectral_kurtosis": sp_kurt,
            "total_power_db": total_power_db,
            "rms": rms,
            "crest_factor": crest_factor,
            "kurtosis": kurtosis,
            "skewness": skewness,
            "peak_to_peak": p2p,
        }

    def _compute_band_powers(
        self,
        freqs: np.ndarray,
        psd: np.ndarray,
    ) -> dict[str, float]:
        """Compute power in standard industrial frequency bands."""
        total = np.trapz(psd, freqs) + 1e-12
        
        bands = {
            "0-100Hz": (0, 100),
            "100-500Hz": (100, 500),
            "500-2kHz": (500, 2000),
            "2k-5kHz": (2000, 5000),
            "5k-10kHz": (5000, 10000),
            "10k+Hz": (10000, freqs[-1]),
        }
        
        result = {}
        for band_name, (f_lo, f_hi) in bands.items():
            mask = (freqs >= f_lo) & (freqs <= f_hi)
            if np.any(mask):
                band_power = np.trapz(psd[mask], freqs[mask])
                result[band_name] = float(10 * np.log10(band_power / total + 1e-12))
            else:
                result[band_name] = -60.0  # noise floor
        
        return result

    def _classify_severity(self, snr_db: float) -> str:
        """Classify fault severity from SNR."""
        if snr_db < 5:
            return "healthy"
        elif snr_db < 10:
            return "incipient"
        elif snr_db < 20:
            return "moderate"
        elif snr_db < 30:
            return "severe"
        else:
            return "critical"

    def _extract_anomaly_regions(
        self,
        freqs: np.ndarray,
        times: np.ndarray,
        scores: np.ndarray,
        binary: np.ndarray,
    ) -> list[dict]:
        """Extract contiguous anomaly regions from binary anomaly map."""
        from scipy.ndimage import label
        
        labeled, n_features = label(binary)
        regions = []
        
        for region_id in range(1, min(n_features + 1, 20)):  # Max 20 regions
            mask = labeled == region_id
            f_indices = np.where(np.any(mask, axis=1))[0]
            t_indices = np.where(np.any(mask, axis=0))[0]
            
            if len(f_indices) == 0 or len(t_indices) == 0:
                continue
            
            regions.append({
                "freq_start_hz": float(freqs[f_indices[0]]),
                "freq_end_hz": float(freqs[f_indices[-1]]),
                "time_start_s": float(times[t_indices[0]]),
                "time_end_s": float(times[t_indices[-1]]),
                "max_score": float(np.max(scores[mask])),
                "mean_score": float(np.mean(scores[mask])),
                "area_bins": int(np.sum(mask)),
            })
        
        # Sort by max score descending
        regions.sort(key=lambda r: r["max_score"], reverse=True)
        return regions

    def feature_vector(self, result: SpectralAnalysisResult) -> np.ndarray:
        """
        Extract a fixed-length feature vector for ML model input.
        
        Returns 64-dimensional feature vector:
        [rms, crest, kurtosis, skewness, p2p,
         entropy, centroid, bandwidth, rolloff, flatness, sp_kurtosis,
         total_power, band_powers×6,
         dominant_freqs×10, dominant_amps×10,
         harmonic_amps_f1×10, ...]
        """
        features = [
            result.rms,
            result.crest_factor,
            result.kurtosis,
            result.skewness,
            result.peak_to_peak,
            result.spectral_entropy,
            result.spectral_centroid / self.nyquist,  # Normalized
            result.spectral_bandwidth / self.nyquist,
            result.spectral_rolloff / self.nyquist,
            result.spectral_flatness,
            result.spectral_kurtosis,
            result.total_power,
        ]
        
        # Band powers (6 bands)
        for band_power in result.band_powers.values():
            features.append(band_power)
        
        # Top-10 dominant frequencies (normalized to Nyquist)
        n_peaks = 10
        dom_freqs = np.zeros(n_peaks)
        dom_amps = np.zeros(n_peaks)
        n = min(n_peaks, len(result.dominant_frequencies))
        dom_freqs[:n] = result.dominant_frequencies[:n] / self.nyquist
        dom_amps[:n] = result.dominant_amplitudes[:n]
        if np.max(dom_amps) > 0:
            dom_amps /= np.max(dom_amps)
        features.extend(dom_freqs.tolist())
        features.extend(dom_amps.tolist())
        
        # First harmonic series amplitudes
        if result.harmonic_amplitudes:
            ha = result.harmonic_amplitudes[0][:10]
            ha_padded = np.zeros(10)
            ha_padded[:len(ha)] = ha
            if np.max(ha_padded) > 0:
                ha_padded /= np.max(ha_padded)
            features.extend(ha_padded.tolist())
        else:
            features.extend([0.0] * 10)
        
        return np.array(features, dtype=np.float32)
