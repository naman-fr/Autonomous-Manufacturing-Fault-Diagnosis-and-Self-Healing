from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import numpy.typing as npt
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from amfd.core.config import DiagnosisConfig
from amfd.core.models import DetectionResult, FeatureVector, RootCause, Severity
from amfd.data.synthetic import generate_machine_window
from amfd.ml.features import extract_features

FEATURE_COLUMNS = (
    "rms",
    "peak_to_peak",
    "crest_factor",
    "dominant_frequency_hz",
    "spectral_energy",
    "rpm_mean",
)

ROOT_CAUSE_LABELS = (
    "normal_operation",
    "bearing_defect",
    "rotor_imbalance_or_misalignment",
    "rpm_control_instability",
)

SEVERITY_LABELS = ("normal", "warning", "critical")


@dataclass(frozen=True)
class TrainingScenario:
    root_cause: str
    severity: str
    intensity: float
    frequency_min_hz: float
    frequency_max_hz: float
    rpm_nominal: float = 1800.0


@dataclass(frozen=True)
class ModelDiagnosis:
    detection: DetectionResult
    root_cause: RootCause
    evidence: list[str]
    severity_probabilities: dict[str, float]
    root_cause_probabilities: dict[str, float]


@dataclass(slots=True)
class SignalModelBundle:
    severity_model: Pipeline
    root_cause_model: Pipeline
    feature_importances: dict[str, float]
    artifact_path: Path

    @classmethod
    def load_or_train(
        cls, config: DiagnosisConfig, artifact_dir: str | Path | None = None
    ) -> SignalModelBundle:
        base_dir = Path(artifact_dir or config.artifact_dir)
        base_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = base_dir / "signal_model_bundle.joblib"
        if artifact_path.exists():
            try:
                loaded: object = joblib.load(artifact_path)
                if isinstance(loaded, SignalModelBundle):
                    return loaded
            except Exception:  # noqa: BLE001 - corrupt cache should trigger retraining.
                pass

        bundle = cls._train(config, artifact_path)
        joblib.dump(bundle, artifact_path)
        return bundle

    def diagnose(self, features: FeatureVector) -> ModelDiagnosis:
        row: npt.NDArray[np.float64] = np.asarray([self._feature_row(features)], dtype=float)
        severity_probs = self._probabilities(self.severity_model, row, SEVERITY_LABELS)
        root_probs = self._probabilities(self.root_cause_model, row, ROOT_CAUSE_LABELS)

        severity_label = max(severity_probs, key=lambda label: severity_probs[label])
        root_label = max(root_probs, key=lambda label: root_probs[label])
        confidence = float(
            max(
                severity_probs[severity_label],
                root_probs[root_label],
            )
        )
        anomaly_score = float(1.0 - severity_probs.get("normal", 0.0))
        detection = DetectionResult(
            severity=_severity_from_label(severity_label),
            confidence=round(confidence, 4),
            anomaly_score=round(anomaly_score, 4),
            evidence=self._evidence_lines(features, severity_probs, root_probs),
        )
        root_cause = RootCause(
            label=root_label,
            probability=round(root_probs[root_label], 4),
            evidence=self._root_evidence(features, root_probs),
        )
        return ModelDiagnosis(
            detection=detection,
            root_cause=root_cause,
            evidence=detection.evidence + root_cause.evidence,
            severity_probabilities=severity_probs,
            root_cause_probabilities=root_probs,
        )

    def predict_detection(self, features: FeatureVector) -> DetectionResult:
        return self.diagnose(features).detection

    @classmethod
    def _train(cls, config: DiagnosisConfig, artifact_path: Path) -> SignalModelBundle:
        rows: list[list[float]] = []
        severity_labels: list[str] = []
        root_labels: list[str] = []

        scenarios = _load_training_scenarios(config.training_scenarios_path)
        samples_per_scenario = max(24, config.synthetic_training_cases // len(scenarios))

        seed = 13
        for scenario in scenarios:
            for index in range(samples_per_scenario):
                fault_frequency = _rand_uniform(
                    scenario.frequency_min_hz,
                    scenario.frequency_max_hz,
                    seed + index,
                )
                window = generate_machine_window(
                    machine_id=f"TRAIN-{scenario.root_cause[:4].upper()}-{index:03d}",
                    root_cause=scenario.root_cause,
                    severity=scenario.severity,
                    sampling_rate_hz=config.sampling_rate_hz,
                    seconds=config.window_seconds,
                    seed=seed + index,
                    fault_frequency_hz=fault_frequency,
                    fault_intensity=scenario.intensity,
                    rpm_nominal=scenario.rpm_nominal,
                )
                features = extract_features(window)
                rows.append(cls._feature_row(features))
                severity_labels.append(scenario.severity)
                root_labels.append(scenario.root_cause)
            seed += 97

        severity_model = _make_pipeline().fit(rows, severity_labels)
        root_cause_model = _make_pipeline().fit(rows, root_labels)
        feature_importances = _feature_importances(root_cause_model)
        return cls(
            severity_model=severity_model,
            root_cause_model=root_cause_model,
            feature_importances=feature_importances,
            artifact_path=artifact_path,
        )

    @staticmethod
    def _feature_row(features: FeatureVector) -> list[float]:
        return [
            features.rms,
            features.peak_to_peak,
            features.crest_factor,
            features.dominant_frequency_hz,
            features.spectral_energy,
            features.rpm_mean if features.rpm_mean is not None else float("nan"),
        ]

    @staticmethod
    def _probabilities(
        pipeline: Pipeline, row: npt.NDArray[np.float64], labels: tuple[str, ...]
    ) -> dict[str, float]:
        probabilities = pipeline.predict_proba(row)[0]
        classes = list(pipeline.named_steps["model"].classes_)
        mapped = {str(label): 0.0 for label in labels}
        for class_name, probability in zip(classes, probabilities, strict=False):
            mapped[str(class_name)] = float(probability)
        return mapped

    def _evidence_lines(
        self,
        features: FeatureVector,
        severity_probs: dict[str, float],
        root_probs: dict[str, float],
    ) -> list[str]:
        ranked = sorted(self.feature_importances.items(), key=lambda item: item[1], reverse=True)
        severity_label = max(severity_probs, key=lambda label: severity_probs[label])
        root_label = max(root_probs, key=lambda label: root_probs[label])
        top_features = ", ".join(
            f"{name}={self._feature_value(features, name):.3f}" for name, _ in ranked[:3]
        )
        return [
            f"Severity posterior: {severity_label} ({severity_probs[severity_label]:.2f})",
            f"Root-cause posterior: {root_label} ({root_probs[root_label]:.2f})",
            f"Top model signals: {top_features}",
        ]

    def _root_evidence(
        self,
        features: FeatureVector,
        root_probs: dict[str, float],
    ) -> list[str]:
        ranked = sorted(self.feature_importances.items(), key=lambda item: item[1], reverse=True)
        root_label = max(root_probs, key=lambda label: root_probs[label])
        evidence = [
            f"Model posterior for {root_label} is {root_probs[root_label]:.2f}",
        ]
        for name, importance in ranked[:3]:
            evidence.append(
                f"{name} contributes {importance:.2f} with observed value "
                f"{self._feature_value(features, name):.3f}"
            )
        return evidence

    @staticmethod
    def _feature_value(features: FeatureVector, name: str) -> float:
        return float(getattr(features, name))


def _make_pipeline() -> Pipeline:
    transformer = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                list(range(len(FEATURE_COLUMNS))),
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    model = RandomForestClassifier(
        n_estimators=256,
        max_depth=None,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([("transform", transformer), ("model", model)])


def _feature_importances(pipeline: Pipeline) -> dict[str, float]:
    model = pipeline.named_steps["model"]
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return {name: 0.0 for name in FEATURE_COLUMNS}
    return {FEATURE_COLUMNS[index]: float(score) for index, score in enumerate(importances)}


def _rand_uniform(low: float, high: float, seed: int) -> float:
    rng = np.random.default_rng(seed)
    return float(rng.uniform(low, high))


def _severity_from_label(label: str) -> Severity:
    if label == "critical":
        return Severity.critical
    if label == "warning":
        return Severity.warning
    return Severity.normal


def _load_training_scenarios(path: str | Path) -> list[TrainingScenario]:
    scenario_path = Path(path)
    if not scenario_path.exists():
        raise FileNotFoundError(f"Training scenarios not found: {scenario_path}")

    raw = json.loads(scenario_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Training scenarios must be a JSON array: {scenario_path}")

    scenarios: list[TrainingScenario] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        frequency = item.get("fault_frequency_hz", {})
        if not isinstance(frequency, dict):
            continue
        try:
            scenario = TrainingScenario(
                root_cause=str(item["root_cause"]),
                severity=str(item["severity"]),
                intensity=float(item["intensity"]),
                frequency_min_hz=float(frequency["min"]),
                frequency_max_hz=float(frequency["max"]),
                rpm_nominal=float(item.get("rpm_nominal", 1800.0)),
            )
        except (KeyError, TypeError, ValueError):
            continue
        scenarios.append(scenario)

    if not scenarios:
        raise ValueError(f"Training scenarios file is empty or invalid: {scenario_path}")
    return scenarios
