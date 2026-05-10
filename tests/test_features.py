from __future__ import annotations

from amfd.data.synthetic import generate_bearing_window
from amfd.ml.features import extract_features


def test_extract_features_identifies_fault_frequency_band() -> None:
    window = generate_bearing_window(
        sampling_rate_hz=2_000,
        seconds=1.0,
        fault_frequency_hz=250,
        fault_intensity=0.25,
    )

    features = extract_features(window)

    assert features.rms > 0.10
    assert 200 <= features.dominant_frequency_hz <= 300
    assert features.rpm_mean is not None

