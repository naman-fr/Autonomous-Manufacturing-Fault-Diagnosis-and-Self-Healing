from __future__ import annotations

from uuid import uuid4

try:
    from langgraph.graph import END, StateGraph
except ImportError:  # pragma: no cover - exercised only when optional dependency is absent
    END = "__end__"
    StateGraph = None  # type: ignore[assignment]

from amfd.core.config import DiagnosisConfig
from amfd.core.models import (
    DiagnosisState,
    IncidentReport,
    MaintenanceAction,
    RootCause,
    Severity,
)
from amfd.core.safety import SafetyPolicy
from amfd.ml.detector import HybridAnomalyDetector
from amfd.ml.features import extract_features
from amfd.rag.retriever import MaintenanceRetriever


class FaultDiagnosisWorkflow:
    def __init__(self, config: DiagnosisConfig | None = None) -> None:
        self.config = config or DiagnosisConfig()
        self.detector = HybridAnomalyDetector(
            anomaly_threshold=self.config.anomaly_threshold,
            high_risk_threshold=self.config.high_risk_threshold,
        )
        self.retriever = MaintenanceRetriever()
        self.safety_policy = SafetyPolicy(self.config.allowed_actions)
        self._graph = self._build_graph()

    def run(self, state: DiagnosisState) -> DiagnosisState:
        initial_state: DiagnosisState = {
            **state,
            "incident_id": state.get("incident_id", f"INC-{uuid4().hex[:10].upper()}"),
            "refinement_loops": state.get("refinement_loops", 0),
            "trace": state.get("trace", []),
        }
        if self._graph is None:
            return self._run_without_langgraph(initial_state)
        return self._graph.invoke(initial_state)

    def _build_graph(self):
        if StateGraph is None:
            return None

        graph = StateGraph(DiagnosisState)
        graph.add_node("extract_features", self._extract_features)
        graph.add_node("detect", self._detect)
        graph.add_node("refine", self._refine)
        graph.add_node("diagnose", self._diagnose)
        graph.add_node("prescribe", self._prescribe)
        graph.add_node("validate", self._validate)
        graph.add_node("report", self._report)

        graph.set_entry_point("extract_features")
        graph.add_edge("extract_features", "detect")
        graph.add_conditional_edges(
            "detect",
            self._route_after_detection,
            {"refine": "refine", "diagnose": "diagnose"},
        )
        graph.add_edge("refine", "extract_features")
        graph.add_edge("diagnose", "prescribe")
        graph.add_edge("prescribe", "validate")
        graph.add_edge("validate", "report")
        graph.add_edge("report", END)
        return graph.compile()

    def _run_without_langgraph(self, state: DiagnosisState) -> DiagnosisState:
        state = {**state, **self._extract_features(state)}
        state = {**state, **self._detect(state)}
        while self._route_after_detection(state) == "refine":
            state = {**state, **self._refine(state)}
            state = {**state, **self._extract_features(state)}
            state = {**state, **self._detect(state)}
        state = {**state, **self._diagnose(state)}
        state = {**state, **self._prescribe(state)}
        state = {**state, **self._validate(state)}
        state = {**state, **self._report(state)}
        return state

    def _extract_features(self, state: DiagnosisState) -> DiagnosisState:
        features = extract_features(state["sensor_window"])
        return {
            "features": features,
            "trace": [*state.get("trace", []), "features_extracted"],
        }

    def _detect(self, state: DiagnosisState) -> DiagnosisState:
        detection = self.detector.predict(state["features"])
        return {
            "detection": detection,
            "trace": [*state.get("trace", []), f"detected_{detection.severity.value}"],
        }

    def _route_after_detection(self, state: DiagnosisState) -> str:
        detection = state["detection"]
        loops = state.get("refinement_loops", 0)
        if detection.confidence < 0.90 and loops < self.config.max_refinement_loops:
            return "refine"
        return "diagnose"

    def _refine(self, state: DiagnosisState) -> DiagnosisState:
        return {
            "refinement_loops": state.get("refinement_loops", 0) + 1,
            "trace": [*state.get("trace", []), "refinement_requested"],
        }

    def _diagnose(self, state: DiagnosisState) -> DiagnosisState:
        features = state["features"]
        detection = state["detection"]
        context = self.retriever.retrieve(features)

        if detection.severity is Severity.normal:
            cause = RootCause(
                label="normal_operation",
                probability=0.86,
                evidence=["Anomaly score stayed below configured warning threshold."],
            )
        elif features.crest_factor >= 3.0 or 120 <= features.dominant_frequency_hz <= 800:
            cause = RootCause(
                label="bearing_defect",
                probability=min(0.96, detection.confidence + 0.05),
                evidence=[
                    "Impulsive vibration and mid-frequency spectral content match bearing wear.",
                    *detection.evidence,
                ],
            )
        elif features.rpm_mean and features.rpm_mean < 1750:
            cause = RootCause(
                label="rpm_control_instability",
                probability=0.78,
                evidence=["RPM mean is below operating baseline.", *detection.evidence],
            )
        else:
            cause = RootCause(
                label="rotor_imbalance_or_misalignment",
                probability=0.74,
                evidence=["Elevated vibration without strong bearing signature.", *detection.evidence],
            )

        return {
            "root_cause": cause,
            "context": context,
            "trace": [*state.get("trace", []), f"root_cause_{cause.label}"],
        }

    def _prescribe(self, state: DiagnosisState) -> DiagnosisState:
        severity = state["detection"].severity
        label = state["root_cause"].label

        if label == "normal_operation":
            actions = [
                MaintenanceAction(
                    action="inspect_bearing",
                    priority="low",
                    rationale="Continue routine inspection because no immediate fault was detected.",
                )
            ]
        elif label == "bearing_defect":
            priority = "immediate" if severity is Severity.critical else "high"
            actions = [
                MaintenanceAction(
                    action="reduce_load",
                    priority=priority,
                    rationale="Lower load to reduce bearing stress while maintenance prepares inspection.",
                ),
                MaintenanceAction(
                    action="inspect_bearing",
                    priority=priority,
                    rationale="Inspect bearing race, lubrication, and housing for early defect progression.",
                ),
            ]
            if severity is Severity.critical:
                actions.append(
                    MaintenanceAction(
                        action="schedule_shutdown",
                        priority="immediate",
                        rationale="Critical vibration signature warrants controlled shutdown approval.",
                    )
                )
        elif label == "rpm_control_instability":
            actions = [
                MaintenanceAction(
                    action="recalibrate_rpm",
                    priority="medium",
                    rationale="RPM drift suggests control-loop or drive calibration issue.",
                )
            ]
        else:
            actions = [
                MaintenanceAction(
                    action="rebalance_rotor",
                    priority="high",
                    rationale="Vibration pattern is consistent with imbalance.",
                ),
                MaintenanceAction(
                    action="align_coupling",
                    priority="medium",
                    rationale="Coupling alignment should be verified during maintenance window.",
                ),
            ]

        return {
            "actions": actions,
            "trace": [*state.get("trace", []), "prescription_generated"],
        }

    def _validate(self, state: DiagnosisState) -> DiagnosisState:
        validation = self.safety_policy.validate(
            state.get("actions", []),
            state["detection"].severity,
        )
        return {
            "validation": validation,
            "trace": [*state.get("trace", []), "safety_validated"],
        }

    def _report(self, state: DiagnosisState) -> DiagnosisState:
        report = IncidentReport(
            incident_id=state["incident_id"],
            machine_id=state["machine_id"],
            features=state["features"],
            detection=state["detection"],
            root_cause=state["root_cause"],
            actions=state.get("actions", []),
            validation=state["validation"],
            context=state.get("context", []),
        )
        return {
            "report": report,
            "trace": [*state.get("trace", []), "report_finalized"],
        }

