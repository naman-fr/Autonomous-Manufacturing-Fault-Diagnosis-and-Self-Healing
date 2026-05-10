from __future__ import annotations

import math
import random

from amfd.core.models import SensorWindow


def generate_bearing_window(
    machine_id: str = "DEMO-BEARING-01",
    sampling_rate_hz: int = 12_000,
    seconds: float = 1.0,
    fault_frequency_hz: float = 260.0,
    fault_intensity: float = 0.35,
    seed: int = 7,
) -> SensorWindow:
    """Generate a deterministic vibration window with a bearing-fault impulse component."""
    rng = random.Random(seed)
    samples = int(sampling_rate_hz * seconds)
    vibration: list[float] = []
    rpm: list[float] = []

    for i in range(samples):
        t = i / sampling_rate_hz
        base = 0.04 * math.sin(2 * math.pi * 30 * t)
        fault = fault_intensity * math.sin(2 * math.pi * fault_frequency_hz * t)
        impulse = fault_intensity * 0.8 if i % max(1, int(sampling_rate_hz / fault_frequency_hz)) == 0 else 0
        noise = rng.gauss(0, 0.015)
        vibration.append(base + fault + impulse + noise)
        rpm.append(1800 + rng.gauss(0, 2))

    return SensorWindow(
        machine_id=machine_id,
        vibration=vibration,
        rpm=rpm,
        sampling_rate_hz=sampling_rate_hz,
    )

