from __future__ import annotations

import json
from typing import Any

import numpy as np
from langchain_core.tools import tool

from amfd.core.config import DiagnosisConfig
from amfd.core.models import FeatureVector, MaintenanceAction, Severity
from amfd.core.safety import SafetyPolicy
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
    detector = HybridAnomalyDetector(config.anomaly_threshold, config.high_risk_threshold)
    result = detector.predict(FeatureVector(**features))
    return result.model_dump(mode="json")


@tool
def causal_inference_tool(features: dict[str, float], anomaly_score: float) -> list[dict[str, Any]]:
    """Rank likely rotating-machinery root causes from signal evidence."""
    crest = features.get("crest_factor", 0.0)
    frequency = features.get("dominant_frequency_hz", 0.0)
    rpm = features.get("rpm_mean")
    candidates = [
        {
            "label": "bearing_defect",
            "score": min(0.98, 0.35 + anomaly_score + (0.15 if crest >= 3 else 0)),
            "why": "impulsive vibration or bearing-band frequency content",
        },
        {
            "label": "rotor_imbalance",
            "score": 0.50 if frequency < 120 else 0.28,
            "why": "low-frequency 1x running-speed dominated vibration",
        },
        {
            "label": "rpm_control_instability",
            "score": 0.70 if rpm is not None and rpm < 1750 else 0.22,
            "why": "RPM drift away from nominal band",
        },
    ]
    return sorted(candidates, key=lambda item: item["score"], reverse=True)


@tool
def prescription_policy_tool(root_cause: str, severity: str) -> list[dict[str, Any]]:
    """Generate safety-gated maintenance prescriptions for the inferred root cause."""
    sev = Severity(severity)
    if root_cause == "bearing_defect":
        priority = "immediate" if sev is Severity.critical else "high"
        return [
            {
                "action": "reduce_load",
                "priority": priority,
                "rationale": "Reduce bearing stress while maintenance prepares inspection.",
                "requires_human_approval": True,
            },
            {
                "action": "inspect_bearing",
                "priority": priority,
                "rationale": "Inspect race, cage, lubrication, and housing.",
                "requires_human_approval": True,
            },
        ]
    if root_cause == "rpm_control_instability":
        return [
            {
                "action": "recalibrate_rpm",
                "priority": "medium",
                "rationale": "RPM drift suggests drive or control-loop calibration issue.",
                "requires_human_approval": True,
            }
        ]
    return [
        {
            "action": "rebalance_rotor",
            "priority": "high",
            "rationale": "Vibration pattern is consistent with imbalance.",
            "requires_human_approval": True,
        }
    ]


@tool
def safety_validator_tool(actions_json: str, severity: str) -> dict[str, Any]:
    """Validate maintenance actions against the plant safety policy."""
    actions = [MaintenanceAction(**item) for item in json.loads(actions_json)]
    validation = SafetyPolicy(DiagnosisConfig().allowed_actions).validate(
        actions,
        Severity(severity),
    )
    return validation.model_dump(mode="json")


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
