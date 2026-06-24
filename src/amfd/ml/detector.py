from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from amfd.core.config import DiagnosisConfig
from amfd.core.models import DetectionResult, FeatureVector, Severity
from amfd.ml.bundle import ModelDiagnosis, SignalModelBundle


@dataclass(slots=True)
class HybridAnomalyDetector:
    """Compatibility wrapper around the trained signal model bundle.

    The public API stays stable for the rest of the application while the
    implementation shifts from hand-tuned thresholds to a persisted model bundle.
    """

    anomaly_threshold: float | None = None
    high_risk_threshold: float | None = None
    config: DiagnosisConfig | None = None
    artifact_dir: str | Path | None = None
    bundle: SignalModelBundle | None = None

    def __post_init__(self) -> None:
        config = self.config or DiagnosisConfig()
        self.config = config
        self.bundle = SignalModelBundle.load_or_train(
            config,
            artifact_dir=self.artifact_dir or config.artifact_dir,
        )

    def diagnose(self, features: FeatureVector) -> ModelDiagnosis:
        assert self.bundle is not None
        diagnosis = self.bundle.diagnose(features)
        return self._apply_thresholds(diagnosis)

    def predict(self, features: FeatureVector) -> DetectionResult:
        return self.diagnose(features).detection

    def _apply_thresholds(self, diagnosis: ModelDiagnosis) -> ModelDiagnosis:
        assert self.config is not None
        detection = diagnosis.detection
        anomaly_threshold = (
            self.anomaly_threshold
            if self.anomaly_threshold is not None
            else self.config.anomaly_threshold
        )
        high_risk_threshold = (
            self.high_risk_threshold
            if self.high_risk_threshold is not None
            else self.config.high_risk_threshold
        )

        anomaly_score = detection.anomaly_score
        severity = detection.severity
        if anomaly_score < anomaly_threshold:
            severity = Severity.normal
        elif anomaly_score >= high_risk_threshold:
            severity = Severity.critical
        elif severity is Severity.normal:
            severity = Severity.warning

        if severity == detection.severity:
            return diagnosis

        adjusted_detection = detection.model_copy(update={"severity": severity})
        return ModelDiagnosis(
            detection=adjusted_detection,
            root_cause=diagnosis.root_cause,
            evidence=diagnosis.evidence,
            severity_probabilities=diagnosis.severity_probabilities,
            root_cause_probabilities=diagnosis.root_cause_probabilities,
        )
