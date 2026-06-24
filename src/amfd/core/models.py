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


class AgentMessage(BaseModel):
    sender: str
    receiver: str
    content: str
    message_type: Literal["handoff", "evidence", "decision", "review"] = "handoff"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class RetrievedEvidence(BaseModel):
    source: str
    text: str
    bm25_score: float
    rerank_score: float


class HumanReview(BaseModel):
    required: bool
    reason: str
    approved: bool | None = None
    reviewer: str | None = None


class GuardrailFinding(BaseModel):
    category: str
    action: Literal["allow", "redact", "block"]
    detail: str


class RuntimeMetrics(BaseModel):
    latency_ms: float = 0.0
    estimated_oee_gain_percent: float = 0.0
    tokens_used: int = 0
    tool_calls: int = 0


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
    rag_evidence: list[RetrievedEvidence] = Field(default_factory=list)
    guardrails: list[GuardrailFinding] = Field(default_factory=list)
    metrics: RuntimeMetrics = Field(default_factory=RuntimeMetrics)
    agent_messages: list[AgentMessage] = Field(default_factory=list)


class DiagnosisState(TypedDict, total=False):
    incident_id: str
    machine_id: str
    sensor_window: SensorWindow
    features: FeatureVector
    detection: DetectionResult
    model_diagnosis: Any
    root_cause: RootCause
    actions: list[MaintenanceAction]
    validation: SafetyValidation
    context: list[str]
    rag_evidence: list[RetrievedEvidence]
    guardrails: list[GuardrailFinding]
    human_review: HumanReview
    agent_messages: list[AgentMessage]
    metrics: RuntimeMetrics
    refinement_loops: int
    report: IncidentReport
    trace: list[str]
    metadata: dict[str, Any]
