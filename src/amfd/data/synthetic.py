from __future__ import annotations

import math
import random

from amfd.core.models import SensorWindow


def generate_machine_window(
    machine_id: str = "DEMO-ASSET-01",
    root_cause: str = "normal_operation",
    severity: str = "normal",
    sampling_rate_hz: int = 12_000,
    seconds: float = 1.0,
    seed: int = 7,
    fault_frequency_hz: float | None = None,
    fault_intensity: float | None = None,
    rpm_nominal: float = 1800.0,
) -> SensorWindow:
    """Generate a deterministic synthetic rotating-machine window for training and demos."""
    rng = random.Random(seed)
    samples = int(sampling_rate_hz * seconds)
    severity_scale = {
        "normal": 0.10,
        "warning": 0.35,
        "critical": 0.65,
    }.get(severity, 0.35)
    if fault_intensity is not None:
        severity_scale = fault_intensity

    vibration: list[float] = []
    rpm: list[float] = []
    running_speed_hz = rpm_nominal / 60.0
    fault_frequency_hz = fault_frequency_hz or _default_fault_frequency(root_cause, rng)

    for i in range(samples):
        t = i / sampling_rate_hz
        base = 0.03 * math.sin(2 * math.pi * 30 * t)
        harmonic = 0.01 * math.sin(2 * math.pi * 60 * t + 0.2)
        noise = rng.gauss(0, 0.012)

        if root_cause == "normal_operation":
            vibration_sample = base + harmonic + noise
            rpm_sample = rpm_nominal + rng.gauss(0, 1.5)
        elif root_cause == "bearing_defect":
            fault = severity_scale * math.sin(2 * math.pi * fault_frequency_hz * t)
            impulse_period = max(1, int(sampling_rate_hz / max(fault_frequency_hz, 1.0)))
            impulse = severity_scale * 0.9 if i % impulse_period == 0 else 0.0
            vibration_sample = base + fault + impulse + noise
            rpm_sample = rpm_nominal + rng.gauss(0, 2.5)
        elif root_cause == "rotor_imbalance_or_misalignment":
            imbalance = severity_scale * math.sin(2 * math.pi * running_speed_hz * t)
            coupling = 0.5 * severity_scale * math.sin(
                2 * math.pi * running_speed_hz * 2 * t + 0.7
            )
            vibration_sample = base + imbalance + coupling + noise
            rpm_sample = rpm_nominal + rng.gauss(0, 2.0)
        elif root_cause == "rpm_control_instability":
            drift = severity_scale * 60 * math.sin(2 * math.pi * 0.45 * t)
            modulation = 0.04 * math.sin(2 * math.pi * (30 + 3 * math.sin(2 * math.pi * t)) * t)
            vibration_sample = base + modulation + 0.015 * drift / 60.0 + noise
            rpm_sample = rpm_nominal + drift + rng.gauss(0, 4.5)
        else:
            vibration_sample = base + harmonic + noise
            rpm_sample = rpm_nominal + rng.gauss(0, 2.0)

        vibration.append(vibration_sample)
        rpm.append(rpm_sample)

    return SensorWindow(
        machine_id=machine_id,
        vibration=vibration,
        rpm=rpm,
        sampling_rate_hz=sampling_rate_hz,
    )


def generate_bearing_window(
    machine_id: str = "DEMO-BEARING-01",
    sampling_rate_hz: int = 12_000,
    seconds: float = 1.0,
    fault_frequency_hz: float = 260.0,
    fault_intensity: float = 0.35,
    seed: int = 7,
) -> SensorWindow:
    """Generate a deterministic vibration window with a bearing-fault impulse component."""
    severity = "critical" if fault_intensity >= 0.55 else "warning"
    return generate_machine_window(
        machine_id=machine_id,
        root_cause="bearing_defect",
        severity=severity,
        sampling_rate_hz=sampling_rate_hz,
        seconds=seconds,
        seed=seed,
        fault_frequency_hz=fault_frequency_hz,
        fault_intensity=fault_intensity,
    )


def _default_fault_frequency(root_cause: str, rng: random.Random) -> float:
    if root_cause == "bearing_defect":
        return rng.uniform(220.0, 360.0)
    if root_cause == "rotor_imbalance_or_misalignment":
        return rng.uniform(20.0, 55.0)
    if root_cause == "rpm_control_instability":
        return rng.uniform(40.0, 90.0)
    return rng.uniform(20.0, 40.0)
