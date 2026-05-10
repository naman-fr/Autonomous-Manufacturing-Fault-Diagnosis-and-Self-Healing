from __future__ import annotations

from amfd.core.models import FeatureVector, Severity
from amfd.ml.detector import HybridAnomalyDetector


def test_detector_flags_high_energy_bearing_pattern() -> None:
    detector = HybridAnomalyDetector(anomaly_threshold=0.50, high_risk_threshold=0.80)
    features = FeatureVector(
        rms=0.28,
        peak_to_peak=1.0,
        crest_factor=4.5,
        dominant_frequency_hz=260.0,
        spectral_energy=20.0,
        rpm_mean=1800.0,
    )

    result = detector.predict(features)

    assert result.severity in {Severity.warning, Severity.critical}
    assert result.anomaly_score >= 0.50
    assert any("dominant_frequency" in item for item in result.evidence)

