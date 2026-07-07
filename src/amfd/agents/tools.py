from __future__ import annotations

import json
from typing import Any, cast

import numpy as np
from langchain_core.tools import tool

from amfd.core.catalog import ActionCatalog
from amfd.core.config import DiagnosisConfig
from amfd.core.models import FeatureVector, MaintenanceAction, Severity
from amfd.core.safety import SafetyPolicy
from amfd.ml.bundle import SignalModelBundle
from amfd.ml.detector import HybridAnomalyDetector


@tool
def fft_signature_tool(vibration: list[float], sampling_rate_hz: int) -> dict[str, float]:
    """Compute dominant FFT frequency and spectral energy for a vibration window."""
    signal = np.asarray(vibration, dtype=float)
    centered = signal - signal.mean()
    spectrum = np.fft.rfft(centered)
    magnitudes = np.abs(spectrum)
    frequencies = np.fft.rfftfreq(signal.size, d=1 / sampling_rate_hz)
    index = int(np.argmax(magnitudes[1:]) + 1) if magnitudes.size > 1 else 0
    return {
        "dominant_frequency_hz": float(frequencies[index]),
        "spectral_energy": float(np.sum(np.square(magnitudes)) / max(signal.size, 1)),
    }


@tool
def anomaly_detector_tool(features: dict[str, float]) -> dict[str, Any]:
    """Score a feature vector with the explainable baseline anomaly detector."""
    config = DiagnosisConfig()
    detector = HybridAnomalyDetector(
        anomaly_threshold=config.anomaly_threshold,
        high_risk_threshold=config.high_risk_threshold,
        config=config,
    )
    result = detector.predict(FeatureVector(**features))
    return cast(dict[str, Any], result.model_dump(mode="json"))


@tool
def causal_inference_tool(features: dict[str, float], anomaly_score: float) -> list[dict[str, Any]]:
    """Rank likely rotating-machinery root causes from model-backed signal evidence."""
    config = DiagnosisConfig()
    bundle = SignalModelBundle.load_or_train(config)
    diagnosis = bundle.diagnose(FeatureVector(**features))
    candidates: list[dict[str, Any]] = []
    for label, probability in sorted(
        diagnosis.root_cause_probabilities.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        candidates.append(
            {
                "label": label,
                "score": round(float(probability), 4),
                "why": "Model-backed posterior from trained rotating-machine bundle.",
            }
        )
    if candidates:
        candidates[0]["why"] = (
            f"Highest posterior with anomaly score {anomaly_score:.2f} "
            f"and root-cause evidence from the bundle."
        )
    return candidates


@tool
def prescription_policy_tool(root_cause: str, severity: str) -> list[dict[str, Any]]:
    """Generate safety-gated maintenance prescriptions for the inferred root cause."""
    catalog = ActionCatalog.load(DiagnosisConfig().policy_path)
    actions = catalog.build(root_cause, Severity(severity))
    return [action.model_dump(mode="json") for action in actions]


@tool
def safety_validator_tool(actions_json: str, severity: str) -> dict[str, Any]:
    """Validate maintenance actions against the plant safety policy."""
    actions = [MaintenanceAction(**item) for item in json.loads(actions_json)]
    validation = SafetyPolicy(DiagnosisConfig().allowed_actions).validate(
        actions,
        Severity(severity),
    )
    return cast(dict[str, Any], validation.model_dump(mode="json"))


@tool
def oee_impact_tool(anomaly_score: float, minutes_saved: float = 30.0) -> dict[str, float]:
    """Estimate avoided downtime and OEE improvement for an early intervention."""
    avoided_minutes = max(0.0, minutes_saved * anomaly_score)
    return {
        "avoided_downtime_minutes": round(avoided_minutes, 2),
        "estimated_oee_gain_percent": round(min(8.0, avoided_minutes / 60 * 1.5), 2),
    }


def build_tool_registry() -> list[Any]:
    return [
        fft_signature_tool,
        anomaly_detector_tool,
        causal_inference_tool,
        prescription_policy_tool,
        safety_validator_tool,
        oee_impact_tool,
    ]
