"""
Industrial Digital Twin Engine — Rotating Machinery Physics Simulation

Implements a production digital twin for industrial rotating machinery:
- Physics-based vibration generation
- Degradation progression models
- Synthetic fault injection
- Streaming telemetry generation
- Asset lifecycle simulation

Degradation Model (Paris Law modified):
    da/dt = C·(ΔK)^m·(1 + α·T/T_max)
    
where:
  a    = crack depth / damage parameter
  ΔK   = stress intensity factor range
  C, m = material constants
  T    = operating time
  α    = thermal coupling factor

Vibration Model:
    x(t) = Σ_i A_i·sin(2π·f_i·t + φ_i) + Σ_j B_j·s_j(t)·δ(t - T_j) + n(t)
    
where the first sum is healthy vibration (harmonics of shaft frequency),
the second sum is fault-induced impulsive content,
and n(t) is stochastic noise.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncGenerator, Optional
import uuid

import numpy as np
from scipy.signal import butter, filtfilt


# ─────────────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────────────

class AssetState(str, Enum):
    HEALTHY = "healthy"
    INCIPIENT = "incipient"
    DEGRADED = "degraded"
    SEVERE = "severe"
    CRITICAL = "critical"
    FAILED = "failed"


class FaultType(str, Enum):
    NONE = "none"
    INNER_RACE = "inner_race"
    OUTER_RACE = "outer_race"
    BALL_FAULT = "ball_fault"
    MISALIGNMENT = "misalignment"
    IMBALANCE = "imbalance"
    LOOSENESS = "looseness"
    LUBRICATION = "lubrication_failure"


@dataclass
class AssetConfig:
    """Configuration for a simulated industrial asset."""
    asset_id: str
    asset_name: str
    asset_type: str = "rotating_machinery"         # pump, fan, compressor, motor

    # Mechanical parameters
    shaft_rpm: float = 1800.0                      # Operating speed (RPM)
    rated_power_kw: float = 75.0
    bearing_type: str = "SKF_6205"

    # Bearing geometry (SKF 6205 defaults)
    n_rolling_elements: int = 9
    ball_diameter_mm: float = 7.938
    pitch_diameter_mm: float = 38.5
    contact_angle_deg: float = 0.0

    # Sampling
    sampling_rate_hz: int = 20_000
    segment_length: int = 4096

    # Operating limits
    max_temperature_c: float = 85.0
    max_vibration_g: float = 15.0
    design_life_hours: float = 50_000.0

    @property
    def shaft_hz(self) -> float:
        return self.shaft_rpm / 60.0

    @property
    def bearing_fault_frequencies(self) -> dict[str, float]:
        """Compute characteristic bearing fault frequencies."""
        shaft_hz = self.shaft_hz
        bd = self.ball_diameter_mm
        pd = self.pitch_diameter_mm
        ca = np.radians(self.contact_angle_deg)
        nb = self.n_rolling_elements

        bpfi = (nb / 2) * shaft_hz * (1 + (bd / pd) * np.cos(ca))
        bpfo = (nb / 2) * shaft_hz * (1 - (bd / pd) * np.cos(ca))
        bsf = (pd / (2 * bd)) * shaft_hz * (1 - ((bd / pd) * np.cos(ca)) ** 2)
        ftf = shaft_hz / 2 * (1 - (bd / pd) * np.cos(ca))

        return {
            "BPFI": float(bpfi),
            "BPFO": float(bpfo),
            "BSF": float(bsf),
            "FTF": float(ftf),
            "shaft": shaft_hz,
            "2x_shaft": 2 * shaft_hz,
            "3x_shaft": 3 * shaft_hz,
        }


@dataclass
class DegradationState:
    """Current degradation state of the digital twin."""
    asset_id: str
    timestamp: float = field(default_factory=time.time)
    operating_hours: float = 0.0

    # Degradation parameters (0=new, 1=failed)
    bearing_health: float = 1.0
    lubrication_quality: float = 1.0
    thermal_stress: float = 0.0
    fatigue_damage: float = 0.0

    # Active fault
    fault_type: FaultType = FaultType.NONE
    fault_severity: float = 0.0      # 0-1

    # Asset state
    state: AssetState = AssetState.HEALTHY
    rul_estimate: float = 50_000.0   # hours

    # Physical readings
    temperature_c: float = 35.0
    shaft_rpm: float = 1800.0
    power_kw: float = 75.0
    vibration_rms_g: float = 0.5


@dataclass
class TelemetryPacket:
    """Single telemetry observation from the digital twin."""
    asset_id: str
    timestamp: float
    sequence_number: int

    # Vibration
    vibration_x: np.ndarray      # Raw vibration signal (samples)
    vibration_y: np.ndarray
    vibration_z: np.ndarray
    vibration_rms_g: float

    # Temperature sensors
    bearing_temp_c: float
    motor_temp_c: float
    ambient_temp_c: float

    # Speed / load
    shaft_rpm: float
    load_percent: float
    power_kw: float

    # Health indicators
    kurtosis: float
    crest_factor: float
    rms_g: float
    peak_g: float

    # State labels (for synthetic data)
    true_state: AssetState
    true_fault: FaultType
    true_rul_hours: float


# ─────────────────────────────────────────────────────────────────
# Physics-Based Vibration Generator
# ─────────────────────────────────────────────────────────────────

class VibrationSignalGenerator:
    """
    Physics-based vibration signal generation for industrial machinery.

    Generates realistic multi-component vibration including:
    - Shaft harmonics (synchronous content)
    - Bearing fault signatures (sub/super-synchronous impulsive content)
    - Structural resonance
    - Gaussian noise floor
    - Sensor noise and drift
    """

    def __init__(
        self,
        asset_config: AssetConfig,
        random_seed: int = 42,
    ) -> None:
        self.config = asset_config
        self.rng = np.random.default_rng(random_seed)
        self.dt = 1.0 / asset_config.sampling_rate_hz

    def generate_healthy_vibration(
        self,
        n_samples: int,
        shaft_rpm: float | None = None,
        noise_std: float = 0.05,
    ) -> np.ndarray:
        """
        Generate healthy rotating machinery vibration signal.

        x(t) = Σ_k A_k·sin(2π·k·f_s·t + φ_k) + n(t)

        First 7 harmonics of shaft frequency, decreasing amplitude.
        """
        rpm = shaft_rpm or self.config.shaft_rpm
        shaft_hz = rpm / 60.0
        t = np.arange(n_samples) * self.dt

        signal = np.zeros(n_samples)

        # Shaft harmonics (balanced machine)
        harmonic_amplitudes = [0.5, 0.2, 0.1, 0.05, 0.03, 0.02, 0.01]
        for k, amp in enumerate(harmonic_amplitudes, start=1):
            phase = self.rng.uniform(0, 2 * np.pi)
            signal += amp * np.sin(2 * np.pi * k * shaft_hz * t + phase)

        # High-frequency structural resonance
        resonance_freq = self.rng.uniform(3000, 8000)
        resonance_amp = 0.05
        resonance_damping = self.rng.uniform(0.02, 0.1)
        envelope = np.exp(-resonance_damping * 2 * np.pi * resonance_freq * t)
        signal += resonance_amp * envelope * np.sin(2 * np.pi * resonance_freq * t)

        # Gaussian noise floor
        signal += self.rng.normal(0, noise_std, n_samples)

        return signal.astype(np.float32)

    def generate_bearing_fault_vibration(
        self,
        n_samples: int,
        fault_type: FaultType,
        fault_severity: float,
        shaft_rpm: float | None = None,
        noise_std: float = 0.05,
    ) -> np.ndarray:
        """
        Generate bearing fault vibration using impact model.

        Bearing fault creates periodic impacts at fault frequency f_fault.
        Each impact excites structural resonance:

        x_fault(t) = A_fault·Σ_n w(t - n/f_fault)·sin(2π·f_res·(t - n/f_fault))

        where w(t) is an exponential decay window and f_res is resonance frequency.
        """
        rpm = shaft_rpm or self.config.shaft_rpm
        fault_freqs = self.config.bearing_fault_frequencies

        # Select fault frequency
        fault_freq_map = {
            FaultType.INNER_RACE: fault_freqs["BPFI"],
            FaultType.OUTER_RACE: fault_freqs["BPFO"],
            FaultType.BALL_FAULT: fault_freqs["BSF"],
            FaultType.IMBALANCE: fault_freqs["shaft"],
            FaultType.MISALIGNMENT: fault_freqs["2x_shaft"],
        }
        fault_freq = fault_freq_map.get(fault_type, fault_freqs["BPFO"])

        t = np.arange(n_samples) * self.dt
        signal = self.generate_healthy_vibration(n_samples, rpm, noise_std * 0.5)

        # Impact period
        T_impact = 1.0 / (fault_freq + 1e-6)
        impact_times = np.arange(0, n_samples * self.dt, T_impact)

        # Structural resonance parameters
        f_resonance = self.rng.uniform(4000, 8000)
        damping = 0.03    # Lightly damped resonance
        decay_rate = damping * 2 * np.pi * f_resonance

        # Add modulation (speed variation causes slight frequency jitter)
        jitter = 0.01 * T_impact
        impact_times += self.rng.normal(0, jitter, len(impact_times))

        # Fault amplitude scales with severity
        fault_amplitude = fault_severity * 2.0

        for t_impact in impact_times:
            # Find sample index
            idx = int(t_impact * self.config.sampling_rate_hz)
            if idx >= n_samples:
                break

            # Number of samples for impact response
            n_response = min(int(5 * T_impact * self.config.sampling_rate_hz), n_samples - idx)
            if n_response < 2:
                continue

            t_local = np.arange(n_response) * self.dt
            impact = (
                fault_amplitude
                * np.exp(-decay_rate * t_local)
                * np.sin(2 * np.pi * f_resonance * t_local)
            )
            signal[idx:idx + n_response] += impact

        # Inner race modulation (amplitude modulated by shaft rotation)
        if fault_type == FaultType.INNER_RACE:
            shaft_modulation = 1 + 0.3 * np.sin(2 * np.pi * fault_freqs["shaft"] * t)
            signal *= shaft_modulation

        return signal.astype(np.float32)

    def generate_with_noise_corruption(
        self,
        signal: np.ndarray,
        noise_type: str = "gaussian",
        dropout_rate: float = 0.0,
        drift_rate: float = 0.0,
    ) -> np.ndarray:
        """
        Add realistic sensor noise, dropout, and drift to clean signal.

        Noise types:
        - gaussian: white Gaussian noise
        - thermal: 1/f (pink) noise simulating thermal effects
        - quantization: ADC quantization noise
        - impulse: random impulse noise (EMI spikes)
        """
        corrupted = signal.copy()
        n = len(signal)

        if noise_type == "gaussian":
            std = 0.02 * np.std(signal)
            corrupted += self.rng.normal(0, std, n)

        elif noise_type == "thermal":
            # Pink noise via spectral shaping
            f = np.fft.rfftfreq(n)
            f[0] = 1e-6  # Avoid division by zero
            power_spectrum = 1 / f
            pink = np.fft.irfft(
                np.fft.rfft(self.rng.normal(0, 1, n)) * np.sqrt(power_spectrum)
            )[:n]
            pink = pink / np.std(pink) * 0.01 * np.std(signal)
            corrupted += pink

        elif noise_type == "quantization":
            n_bits = 12  # 12-bit ADC
            q_step = 2 * np.max(np.abs(signal)) / (2 ** n_bits)
            corrupted = np.round(corrupted / q_step) * q_step

        elif noise_type == "impulse":
            n_impulses = max(1, int(n * 0.001))
            positions = self.rng.integers(0, n, n_impulses)
            amplitude = 3 * np.std(signal)
            corrupted[positions] += self.rng.choice([-1, 1], n_impulses) * amplitude

        # Sensor dropout (missing packets)
        if dropout_rate > 0:
            dropout_mask = self.rng.random(n) < dropout_rate
            # Interpolate over dropped samples
            indices = np.arange(n)
            valid = ~dropout_mask
            if np.any(valid):
                corrupted = np.interp(indices, indices[valid], corrupted[valid])

        # Sensor drift (linear + oscillatory)
        if drift_rate > 0:
            t = np.arange(n) / self.config.sampling_rate_hz
            drift = drift_rate * t + 0.01 * np.sin(2 * np.pi * 0.1 * t)
            corrupted += drift

        return corrupted.astype(np.float32)


# ─────────────────────────────────────────────────────────────────
# Degradation Progression Engine
# ─────────────────────────────────────────────────────────────────

class DegradationEngine:
    """
    Physics-based degradation progression for industrial machinery.

    Implements:
    1. Paris Law for fatigue crack propagation
    2. Arrhenius temperature-accelerated degradation
    3. Load-cycle accumulation (Palmgren-Miner)
    4. Lubrication film breakdown model
    """

    # Material constants (bearing steel)
    PARIS_C = 1e-10       # Paris law coefficient
    PARIS_M = 3.0         # Paris law exponent
    ACTIVATION_ENERGY = 0.8  # eV (Arrhenius)
    BOLTZMANN = 8.617e-5  # eV/K

    def __init__(
        self,
        asset_config: AssetConfig,
        initial_state: DegradationState | None = None,
        rng_seed: int = 42,
    ) -> None:
        self.config = asset_config
        self.rng = np.random.default_rng(rng_seed)

        if initial_state is None:
            self.state = DegradationState(asset_id=asset_config.asset_id)
        else:
            self.state = initial_state

        # Internal degradation trajectory
        self._degradation_history: list[DegradationState] = []
        self._total_cycles: float = 0.0
        self._crack_depth: float = 0.0001   # Initial surface crack depth (mm)
        self._critical_crack_depth: float = 2.0  # Critical crack depth (mm)

    def advance_time(
        self,
        delta_hours: float,
        operating_temp_c: float = 70.0,
        load_fraction: float = 1.0,
        shaft_rpm: float | None = None,
    ) -> DegradationState:
        """
        Advance degradation by delta_hours of operation.

        Parameters
        ----------
        delta_hours : float
            Hours of operation to simulate
        operating_temp_c : float
            Operating temperature (affects Arrhenius degradation)
        load_fraction : float
            Fraction of rated load (0-1)
        shaft_rpm : float, optional
            Shaft speed (defaults to config)
        """
        rpm = shaft_rpm or self.config.shaft_rpm
        self.state.operating_hours += delta_hours
        n_cycles = delta_hours * 3600 * rpm / 60
        self._total_cycles += n_cycles

        # 1. Paris Law fatigue crack growth
        stress_intensity = self._compute_stress_intensity(load_fraction)
        delta_k = stress_intensity * (1 + 0.2 * self.rng.normal())
        crack_growth = self.PARIS_C * (max(0, delta_k) ** self.PARIS_M) * n_cycles
        self._crack_depth += crack_growth

        # 2. Arrhenius temperature acceleration
        temp_k = operating_temp_c + 273.15
        ref_temp_k = 70.0 + 273.15
        acceleration_factor = np.exp(
            self.ACTIVATION_ENERGY / self.BOLTZMANN * (1 / ref_temp_k - 1 / temp_k)
        )
        thermal_damage = delta_hours * acceleration_factor * 0.0001

        # 3. Lubrication degradation (oil oxidation model)
        lube_degradation = delta_hours * load_fraction * 5e-5
        self.state.lubrication_quality = max(0, self.state.lubrication_quality - lube_degradation)

        # 4. Compute composite bearing health
        crack_fraction = np.clip(self._crack_depth / self._critical_crack_depth, 0, 1)
        self.state.bearing_health = 1.0 - crack_fraction
        self.state.fatigue_damage = float(crack_fraction)
        self.state.thermal_stress = float(
            np.clip(self.state.thermal_stress + thermal_damage, 0, 1)
        )

        # 5. Update temperature (rises with degradation)
        base_temp = 35.0 + load_fraction * 20.0
        degradation_heat = crack_fraction * 25.0 + (1 - self.state.lubrication_quality) * 15.0
        self.state.temperature_c = base_temp + degradation_heat + self.rng.normal(0, 0.5)

        # 6. Determine active fault and state
        self._update_fault_state()

        # 7. Estimate RUL
        self.state.rul_estimate = self._estimate_rul()

        # 8. Compute vibration indicators
        self._update_vibration_indicators()

        # Record history
        self._degradation_history.append(
            DegradationState(
                asset_id=self.state.asset_id,
                timestamp=time.time(),
                operating_hours=self.state.operating_hours,
                bearing_health=self.state.bearing_health,
                lubrication_quality=self.state.lubrication_quality,
                thermal_stress=self.state.thermal_stress,
                fatigue_damage=self.state.fatigue_damage,
                fault_type=self.state.fault_type,
                fault_severity=self.state.fault_severity,
                state=self.state.state,
                rul_estimate=self.state.rul_estimate,
                temperature_c=self.state.temperature_c,
                shaft_rpm=rpm,
                power_kw=self.config.rated_power_kw * load_fraction,
                vibration_rms_g=self.state.vibration_rms_g,
            )
        )

        return self.state

    def get_degradation_trajectory(self) -> np.ndarray:
        """Return bearing health trajectory as numpy array."""
        if not self._degradation_history:
            return np.array([1.0])
        return np.array([s.bearing_health for s in self._degradation_history])

    def inject_fault(
        self,
        fault_type: FaultType,
        severity: float = 0.3,
    ) -> None:
        """Manually inject a specific fault for testing/simulation."""
        self.state.fault_type = fault_type
        self.state.fault_severity = severity
        # Accelerate degradation
        self._crack_depth = self._critical_crack_depth * severity * 0.5

    def _compute_stress_intensity(self, load_fraction: float) -> float:
        """Compute stress intensity factor range ΔK based on load and geometry."""
        nominal_stress = 500 * load_fraction  # MPa (nominal)
        geometry_factor = 1.12               # Surface crack geometry
        return geometry_factor * nominal_stress * np.sqrt(np.pi * self._crack_depth * 1e-3)

    def _update_fault_state(self) -> None:
        """Update fault type and severity from degradation state."""
        damage = self.state.fatigue_damage
        lube_bad = self.state.lubrication_quality < 0.3

        if damage < 0.1:
            self.state.fault_type = FaultType.NONE
            self.state.fault_severity = 0.0
            self.state.state = AssetState.HEALTHY
        elif damage < 0.25:
            self.state.fault_type = FaultType.LUBRICATION if lube_bad else FaultType.OUTER_RACE
            self.state.fault_severity = damage
            self.state.state = AssetState.INCIPIENT
        elif damage < 0.5:
            self.state.fault_type = FaultType.OUTER_RACE
            self.state.fault_severity = damage
            self.state.state = AssetState.DEGRADED
        elif damage < 0.75:
            self.state.fault_type = FaultType.OUTER_RACE
            self.state.fault_severity = damage
            self.state.state = AssetState.SEVERE
        elif damage < 1.0:
            self.state.fault_type = FaultType.INNER_RACE
            self.state.fault_severity = damage
            self.state.state = AssetState.CRITICAL
        else:
            self.state.fault_type = FaultType.INNER_RACE
            self.state.fault_severity = 1.0
            self.state.state = AssetState.FAILED

    def _estimate_rul(self) -> float:
        """
        Estimate RUL using crack growth rate extrapolation.
        RUL = (a_critical - a_current) / (da/dt)
        """
        remaining_crack = max(0, self._critical_crack_depth - self._crack_depth)
        if self._total_cycles > 0:
            growth_rate = self._crack_depth / max(self._total_cycles, 1)
            if growth_rate > 0:
                remaining_cycles = remaining_crack / growth_rate
                rpm = self.config.shaft_rpm
                rul_hours = remaining_cycles / (rpm * 60)
                return max(0.0, float(rul_hours))
        return self.config.design_life_hours

    def _update_vibration_indicators(self) -> None:
        """Update vibration RMS from current degradation state."""
        base_rms = 0.5
        fault_contribution = self.state.fault_severity * 3.0
        thermal_contribution = self.state.thermal_stress * 0.5
        self.state.vibration_rms_g = float(
            base_rms + fault_contribution + thermal_contribution
            + self.rng.normal(0, 0.05)
        )


# ─────────────────────────────────────────────────────────────────
# Digital Twin Instance
# ─────────────────────────────────────────────────────────────────

class IndustrialDigitalTwin:
    """
    Complete industrial digital twin for an asset.

    Combines:
    - Physics simulation (DegradationEngine)
    - Signal generation (VibrationSignalGenerator)
    - State tracking
    - Telemetry streaming
    """

    def __init__(
        self,
        asset_config: AssetConfig,
        simulation_speed: float = 1.0,
        rng_seed: int = 42,
    ) -> None:
        self.config = asset_config
        self.simulation_speed = simulation_speed

        self.signal_gen = VibrationSignalGenerator(asset_config, rng_seed)
        self.degradation = DegradationEngine(asset_config, rng_seed=rng_seed)

        self._sequence_number = 0
        self._running = False

    @property
    def current_state(self) -> DegradationState:
        return self.degradation.state

    def generate_telemetry_packet(
        self,
        load_fraction: float = 1.0,
        noise_type: str = "gaussian",
        add_corruption: bool = True,
    ) -> TelemetryPacket:
        """Generate a single telemetry observation from current twin state."""
        state = self.degradation.state
        n_samples = self.config.segment_length

        # Generate raw vibration signal
        if state.fault_type == FaultType.NONE:
            vib_x = self.signal_gen.generate_healthy_vibration(n_samples, state.shaft_rpm)
            vib_y = self.signal_gen.generate_healthy_vibration(n_samples, state.shaft_rpm, 0.04)
            vib_z = self.signal_gen.generate_healthy_vibration(n_samples, state.shaft_rpm, 0.03)
        else:
            vib_x = self.signal_gen.generate_bearing_fault_vibration(
                n_samples, state.fault_type, state.fault_severity, state.shaft_rpm
            )
            vib_y = self.signal_gen.generate_bearing_fault_vibration(
                n_samples, state.fault_type, state.fault_severity * 0.8, state.shaft_rpm
            )
            vib_z = self.signal_gen.generate_healthy_vibration(n_samples, state.shaft_rpm)

        # Add noise corruption
        if add_corruption:
            vib_x = self.signal_gen.generate_with_noise_corruption(vib_x, noise_type)
            vib_y = self.signal_gen.generate_with_noise_corruption(vib_y, noise_type)

        # Compute statistical indicators
        rms_g = float(np.sqrt(np.mean(vib_x ** 2)))
        peak_g = float(np.max(np.abs(vib_x)))
        crest = float(peak_g / (rms_g + 1e-6))

        from scipy.stats import kurtosis
        kurt = float(kurtosis(vib_x))

        self._sequence_number += 1

        return TelemetryPacket(
            asset_id=self.config.asset_id,
            timestamp=time.time(),
            sequence_number=self._sequence_number,
            vibration_x=vib_x,
            vibration_y=vib_y,
            vibration_z=vib_z,
            vibration_rms_g=rms_g,
            bearing_temp_c=float(state.temperature_c),
            motor_temp_c=float(state.temperature_c * 0.9 + 5.0),
            ambient_temp_c=25.0,
            shaft_rpm=float(state.shaft_rpm),
            load_percent=float(load_fraction * 100),
            power_kw=float(self.config.rated_power_kw * load_fraction),
            kurtosis=kurt,
            crest_factor=crest,
            rms_g=rms_g,
            peak_g=peak_g,
            true_state=state.state,
            true_fault=state.fault_type,
            true_rul_hours=float(state.rul_estimate),
        )

    async def stream_telemetry(
        self,
        update_interval_s: float = 0.1,
        hours_per_update: float = 1.0,
        max_hours: float | None = None,
    ) -> AsyncGenerator[TelemetryPacket, None]:
        """
        Async generator streaming real-time digital twin telemetry.

        Parameters
        ----------
        update_interval_s : float
            Real-time update interval (seconds)
        hours_per_update : float
            Simulated operating hours per update
        max_hours : float, optional
            Stop after this many simulated hours
        """
        self._running = True
        total_hours = 0.0

        try:
            while self._running:
                if max_hours and total_hours >= max_hours:
                    break

                # Advance degradation
                load = self._compute_load_profile(total_hours)
                self.degradation.advance_time(
                    delta_hours=hours_per_update,
                    operating_temp_c=60 + load * 20,
                    load_fraction=load,
                )
                total_hours += hours_per_update

                # Generate telemetry
                packet = self.generate_telemetry_packet(
                    load_fraction=load,
                    noise_type="gaussian",
                )

                yield packet

                # Real-time delay (adjusted for simulation speed)
                await asyncio.sleep(update_interval_s / self.simulation_speed)

        except asyncio.CancelledError:
            self._running = False

    def stop(self) -> None:
        self._running = False

    def reset(self) -> None:
        """Reset twin to new-machine state."""
        self.degradation = DegradationEngine(self.config)
        self._sequence_number = 0

    def simulate_maintenance(self, maintenance_type: str = "bearing_replacement") -> None:
        """Simulate a maintenance event — partial or full degradation reset."""
        if maintenance_type == "bearing_replacement":
            self.degradation.state.bearing_health = 1.0
            self.degradation.state.fatigue_damage = 0.0
            self.degradation.state.fault_type = FaultType.NONE
            self.degradation.state.fault_severity = 0.0
            self.degradation._crack_depth = 0.0001
        elif maintenance_type == "lubrication":
            self.degradation.state.lubrication_quality = 1.0
        elif maintenance_type == "full_overhaul":
            self.reset()

    def _compute_load_profile(self, hours: float) -> float:
        """Time-varying load profile with daily cycles."""
        # Daily load variation (0.6-1.0)
        hour_of_day = (hours % 24)
        daily_cycle = 0.8 + 0.1 * np.sin(2 * np.pi * hour_of_day / 24)
        # Random variation
        noise = np.random.normal(0, 0.02)
        return float(np.clip(daily_cycle + noise, 0.5, 1.0))


# ─────────────────────────────────────────────────────────────────
# Digital Twin Fleet Manager
# ─────────────────────────────────────────────────────────────────

class DigitalTwinFleet:
    """
    Manages a fleet of industrial digital twins.
    Provides fleet-wide health monitoring and maintenance scheduling.
    """

    def __init__(self) -> None:
        self._twins: dict[str, IndustrialDigitalTwin] = {}

    def create_twin(
        self,
        asset_config: AssetConfig | None = None,
        rng_seed: int | None = None,
    ) -> IndustrialDigitalTwin:
        """Create and register a new digital twin."""
        if asset_config is None:
            asset_config = AssetConfig(
                asset_id=str(uuid.uuid4()),
                asset_name=f"Asset-{len(self._twins) + 1:03d}",
            )
        seed = rng_seed or len(self._twins) * 42
        twin = IndustrialDigitalTwin(asset_config, rng_seed=seed)
        self._twins[asset_config.asset_id] = twin
        return twin

    def get_twin(self, asset_id: str) -> IndustrialDigitalTwin | None:
        return self._twins.get(asset_id)

    def fleet_health_summary(self) -> dict:
        """Aggregate health status across all twins."""
        if not self._twins:
            return {}

        states = [t.current_state for t in self._twins.values()]
        health_scores = [s.bearing_health for s in states]
        rul_estimates = [s.rul_estimate for s in states]
        critical_assets = [
            s.asset_id for s in states
            if s.state in (AssetState.CRITICAL, AssetState.SEVERE)
        ]

        return {
            "total_assets": len(self._twins),
            "avg_health": float(np.mean(health_scores)),
            "min_health": float(np.min(health_scores)),
            "avg_rul_hours": float(np.mean(rul_estimates)),
            "min_rul_hours": float(np.min(rul_estimates)),
            "critical_count": len(critical_assets),
            "critical_asset_ids": critical_assets,
            "state_distribution": {
                state.value: sum(1 for s in states if s.state == state)
                for state in AssetState
            },
        }

    def maintenance_schedule(self) -> list[dict]:
        """Generate prioritized maintenance schedule."""
        schedule = []
        for twin in self._twins.values():
            state = twin.current_state
            priority = self._compute_priority(state)
            if state.state != AssetState.HEALTHY:
                schedule.append({
                    "asset_id": state.asset_id,
                    "current_state": state.state.value,
                    "fault_type": state.fault_type.value,
                    "rul_hours": state.rul_estimate,
                    "priority": priority,
                    "recommended_action": self._recommend_action(state),
                })

        schedule.sort(key=lambda x: x["priority"], reverse=True)
        return schedule

    def _compute_priority(self, state: DegradationState) -> float:
        """Priority score for maintenance scheduling (0-10)."""
        base_priority = {
            AssetState.HEALTHY: 0.0,
            AssetState.INCIPIENT: 2.0,
            AssetState.DEGRADED: 5.0,
            AssetState.SEVERE: 8.0,
            AssetState.CRITICAL: 9.5,
            AssetState.FAILED: 10.0,
        }.get(state.state, 0.0)

        # Urgency bonus for very low RUL
        rul_bonus = max(0, 2.0 * (1 - state.rul_estimate / 1000)) if state.rul_estimate < 1000 else 0.0

        return float(min(10.0, base_priority + rul_bonus))

    def _recommend_action(self, state: DegradationState) -> str:
        action_map = {
            AssetState.INCIPIENT: "Schedule inspection within 30 days",
            AssetState.DEGRADED: "Schedule maintenance within 7 days",
            AssetState.SEVERE: "Schedule immediate maintenance (24-48h)",
            AssetState.CRITICAL: "Emergency shutdown and replacement",
            AssetState.FAILED: "FAILED — immediate replacement required",
        }
        return action_map.get(state.state, "Monitor")
