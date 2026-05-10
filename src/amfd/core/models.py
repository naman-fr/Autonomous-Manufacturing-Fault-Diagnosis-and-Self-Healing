from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field


class Severity(str, Enum):
    normal = "normal"
    warning = "warning"
    critical = "critical"


class SensorWindow(BaseModel):
    machine_id: str
    vibration: list[float]
    rpm: list[float] = Field(default_factory=list)
    sampling_rate_hz: int = Field(default=12_000, gt=0)


class FeatureVector(BaseModel):
    rms: float
    peak_to_peak: float
    crest_factor: float
    dominant_frequency_hz: float
    spectral_energy: float
    rpm_mean: float | None = None


class DetectionResult(BaseModel):
    severity: Severity
    confidence: float = Field(ge=0, le=1)
    anomaly_score: float = Field(ge=0, le=1)
    evidence: list[str]


class RootCause(BaseModel):
    label: str
    probability: float = Field(ge=0, le=1)
    evidence: list[str]


class MaintenanceAction(BaseModel):
    action: str
    priority: Literal["low", "medium", "high", "immediate"]
    rationale: str
    requires_human_approval: bool = True


class SafetyValidation(BaseModel):
    approved: bool
    blocked_actions: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class IncidentReport(BaseModel):
    incident_id: str
    machine_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    features: FeatureVector
    detection: DetectionResult
    root_cause: RootCause
    actions: list[MaintenanceAction]
    validation: SafetyValidation
    context: list[str] = Field(default_factory=list)


class DiagnosisState(TypedDict, total=False):
    incident_id: str
    machine_id: str
    sensor_window: SensorWindow
    features: FeatureVector
    detection: DetectionResult
    root_cause: RootCause
    actions: list[MaintenanceAction]
    validation: SafetyValidation
    context: list[str]
    refinement_loops: int
    report: IncidentReport
    trace: list[str]
    metadata: dict[str, Any]

