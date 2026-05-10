from __future__ import annotations

from amfd.core.models import DetectionResult, FeatureVector, Severity


class HybridAnomalyDetector:
    """Explainable baseline detector that can later be swapped for a trained model."""

    def __init__(self, anomaly_threshold: float, high_risk_threshold: float) -> None:
        self.anomaly_threshold = anomaly_threshold
        self.high_risk_threshold = high_risk_threshold

    def predict(self, features: FeatureVector) -> DetectionResult:
        score = self._score(features)
        if score >= self.high_risk_threshold:
            severity = Severity.critical
        elif score >= self.anomaly_threshold:
            severity = Severity.warning
        else:
            severity = Severity.normal

        evidence = [
            f"RMS={features.rms:.4f}",
            f"crest_factor={features.crest_factor:.2f}",
            f"dominant_frequency={features.dominant_frequency_hz:.1f}Hz",
            f"spectral_energy={features.spectral_energy:.4f}",
        ]
        if features.rpm_mean is not None:
            evidence.append(f"rpm_mean={features.rpm_mean:.0f}")

        confidence = min(0.99, 0.55 + abs(score - self.anomaly_threshold))
        return DetectionResult(
            severity=severity,
            confidence=confidence,
            anomaly_score=score,
            evidence=evidence,
        )

    @staticmethod
    def _score(features: FeatureVector) -> float:
        rms_component = min(features.rms / 0.30, 1.0)
        crest_component = min(max((features.crest_factor - 2.5) / 5.0, 0), 1.0)
        energy_component = min(features.spectral_energy / 25.0, 1.0)
        frequency_component = 1.0 if 120 <= features.dominant_frequency_hz <= 800 else 0.25
        return round(
            0.35 * rms_component
            + 0.25 * crest_component
            + 0.25 * energy_component
            + 0.15 * frequency_component,
            4,
        )
