from __future__ import annotations

from pathlib import Path

import pandas as pd

from amfd.core.models import SensorWindow


def load_sensor_csv(
    path: str | Path, machine_id: str, sampling_rate_hz: int = 12_000
) -> SensorWindow:
    frame = pd.read_csv(path)
    if "vibration" not in frame.columns:
        raise ValueError("CSV must include a 'vibration' column.")

    rpm = frame["rpm"].dropna().astype(float).tolist() if "rpm" in frame.columns else []
    return SensorWindow(
        machine_id=machine_id,
        vibration=frame["vibration"].astype(float).tolist(),
        rpm=rpm,
        sampling_rate_hz=sampling_rate_hz,
    )

