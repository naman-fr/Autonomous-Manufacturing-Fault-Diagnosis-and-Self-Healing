from __future__ import annotations

from time import perf_counter
from typing import Any
from uuid import uuid4

MemorySaver: Any = None
StateGraph: Any = None
END = "__end__"

try:
    from langgraph.checkpoint.memory import MemorySaver as _MemorySaver  # noqa: E402, I001
    from langgraph.graph import END as _END, StateGraph as _StateGraph  # noqa: E402, I001
except ImportError:  # pragma: no cover - optional dependency path
    pass
else:
    MemorySaver = _MemorySaver
    StateGraph = _StateGraph
    END = _END

from amfd.core.catalog import ActionCatalog  # noqa: E402
from amfd.core.config import DiagnosisConfig  # noqa: E402
from amfd.core.models import (  # noqa: E402
    AgentMessage,
    DiagnosisState,
    GuardrailFinding,
    HumanReview,
    IncidentReport,
    RuntimeMetrics,
)
from amfd.core.safety import SafetyPolicy  # noqa: E402
from amfd.ml.detector import HybridAnomalyDetector  # noqa: E402
from amfd.ml.features import extract_features  # noqa: E402
from amfd.rag.hybrid import HybridMaintenanceRetriever  # noqa: E402
from amfd.security.guardrails import GuardrailEngine  # noqa: E402

_langgraph_checkpoint_memory: Any | None
_langgraph_graph: Any | None


class FaultDiagnosisWorkflow:
    def __init__(self, config: DiagnosisConfig | None = None) -> None:
        self.config = config or DiagnosisConfig()
        self.detector = HybridAnomalyDetector(
            anomaly_threshold=self.config.anomaly_threshold,
            high_risk_threshold=self.config.high_risk_threshold,
            config=self.config,
        )
        self.retriever = HybridMaintenanceRetriever.from_config(self.config)
        self.action_catalog = ActionCatalog.load(self.config.policy_path)
        self.safety_policy = SafetyPolicy(self.config.allowed_actions)
        self.guardrails = GuardrailEngine()
        self._graph = self._build_graph()

    def run(self, state: DiagnosisState) -> DiagnosisState:
        started_at = perf_counter()
        initial_state: DiagnosisState = {
            **state,
            "incident_id": state.get("incident_id", f"INC-{uuid4().hex[:10].upper()}"),
            "refinement_loops": state.get("refinement_loops", 0),
            "trace": state.get("trace", []),
            "agent_messages": state.get("agent_messages", []),
            "guardrails": state.get("guardrails", []),
            "metrics": state.get("metrics", RuntimeMetrics()),
        }
        if self._graph is None:
            result = self._run_without_langgraph(initial_state)
        else:
            result = self._graph.invoke(
                initial_state,
                config={"configurable": {"thread_id": initial_state["incident_id"]}},
            )
        metrics = result.get("metrics", RuntimeMetrics())
        metrics.latency_ms = round((perf_counter() - started_at) * 1000, 2)
        result["metrics"] = metrics
        if "report" in result:
            result["report"].metrics = metrics
        return result

    def _build_graph(self) -> Any | None:
        if StateGraph is None:
            return None

        graph = StateGraph(DiagnosisState)
        graph.add_node("supervisor", self._supervisor)
        graph.add_node("guardrails", self._apply_guardrails)
        graph.add_node("data_aug", self._data_aug)
        graph.add_node("detector", self._detect)
        graph.add_node("refine", self._refine)
        graph.add_node("analyzer", self._diagnose)
        graph.add_node("rag", self._retrieve_context)
        graph.add_node("prescriber", self._prescribe)
        graph.add_node("validate", self._validate)
        graph.add_node("human_review", self._human_review)
        graph.add_node("report", self._report)

        graph.set_entry_point("supervisor")
        graph.add_edge("supervisor", "guardrails")
        graph.add_edge("guardrails", "data_aug")
        graph.add_edge("data_aug", "detector")
        graph.add_conditional_edges(
            "detector",
            self._route_after_detection,
            {"refine": "refine", "analyzer": "analyzer"},
        )
        graph.add_edge("refine", "data_aug")
        graph.add_edge("analyzer", "rag")
        graph.add_edge("rag", "prescriber")
        graph.add_edge("prescriber", "validate")
        graph.add_conditional_edges(
            "validate",
            self._route_after_validation,
            {"human_review": "human_review", "report": "report"},
        )
        graph.add_edge("human_review", "report")
        graph.add_edge("report", END)
        checkpointer = MemorySaver() if MemorySaver is not None else None
        return graph.compile(checkpointer=checkpointer)

    def _run_without_langgraph(self, state: DiagnosisState) -> DiagnosisState:
        state = {**state, **self._supervisor(state)}
        state = {**state, **self._apply_guardrails(state)}
        state = {**state, **self._data_aug(state)}
        state = {**state, **self._detect(state)}
        while self._route_after_detection(state) == "refine":
            state = {**state, **self._refine(state)}
            state = {**state, **self._data_aug(state)}
            state = {**state, **self._detect(state)}
        state = {**state, **self._diagnose(state)}
        state = {**state, **self._retrieve_context(state)}
        state = {**state, **self._prescribe(state)}
        state = {**state, **self._validate(state)}
        if self._route_after_validation(state) == "human_review":
            state = {**state, **self._human_review(state)}
        state = {**state, **self._report(state)}
        return state

    def _supervisor(self, state: DiagnosisState) -> DiagnosisState:
        return {
            "agent_messages": [
                *state.get("agent_messages", []),
                AgentMessage(
                    sender="supervisor",
                    receiver="guardrails",
                    content=(
                        "Start incident triage with security checks, then route to specialists."
                    ),
                    message_type="handoff",
                ),
            ],
            "trace": [*state.get("trace", []), "supervisor_started"],
        }

    def _apply_guardrails(self, state: DiagnosisState) -> DiagnosisState:
        notes = str(state.get("metadata", {}).get("operator_notes", ""))
        redacted, redaction_findings = self.guardrails.redact(notes)
        injection_findings = self.guardrails.validate_prompt(redacted)
        metadata = {**state.get("metadata", {}), "operator_notes": redacted}
        findings: list[GuardrailFinding] = [
            *state.get("guardrails", []),
            *redaction_findings,
            *injection_findings,
        ]
        return {
            "metadata": metadata,
            "guardrails": findings,
            "agent_messages": [
                *state.get("agent_messages", []),
                AgentMessage(
                    sender="guardrails",
                    receiver="data_aug",
                    content=f"Security pass completed with {len(findings)} finding(s).",
                    message_type="decision",
                ),
            ],
            "trace": [*state.get("trace", []), "guardrails_checked"],
        }

    def _data_aug(self, state: DiagnosisState) -> DiagnosisState:
        features = extract_features(state["sensor_window"])
        return {
            "features": features,
            "agent_messages": [
                *state.get("agent_messages", []),
                AgentMessage(
                    sender="data_aug",
                    receiver="detector",
                    content=(
                        "Feature window prepared; augmentation hooks remain isolated from "
                        "production evidence and only synthetic training data is used offline."
                    ),
                    message_type="evidence",
                ),
            ],
            "trace": [*state.get("trace", []), "data_aug_features_ready"],
        }

    def _detect(self, state: DiagnosisState) -> DiagnosisState:
        model_diagnosis = self.detector.diagnose(state["features"])
        detection = model_diagnosis.detection
        metrics = state.get("metrics", RuntimeMetrics())
        metrics.tool_calls += 2
        return {
            "detection": detection,
            "model_diagnosis": model_diagnosis,
            "metrics": metrics,
            "agent_messages": [
                *state.get("agent_messages", []),
                AgentMessage(
                    sender="detector",
                    receiver="analyzer",
                    content=(
                        f"Detected {detection.severity.value} state with "
                        f"score={detection.anomaly_score:.2f} "
                        f"confidence={detection.confidence:.2f}."
                    ),
                    message_type="evidence",
                ),
            ],
            "trace": [*state.get("trace", []), f"detector_{detection.severity.value}"],
        }

    def _route_after_detection(self, state: DiagnosisState) -> str:
        detection = state["detection"]
        loops = state.get("refinement_loops", 0)
        if (
            detection.confidence < self.config.review_probability_threshold
            and loops < self.config.max_refinement_loops
        ):
            return "refine"
        return "analyzer"

    def _refine(self, state: DiagnosisState) -> DiagnosisState:
        return {
            "refinement_loops": state.get("refinement_loops", 0) + 1,
            "agent_messages": [
                *state.get("agent_messages", []),
                AgentMessage(
                    sender="supervisor",
                    receiver="data_aug",
                    content="Detector confidence is below threshold; re-run feature extraction.",
                    message_type="decision",
                ),
            ],
            "trace": [*state.get("trace", []), "refinement_requested"],
        }

    def _diagnose(self, state: DiagnosisState) -> DiagnosisState:
        model_diagnosis = state.get("model_diagnosis")
        if model_diagnosis is None:
            model_diagnosis = self.detector.diagnose(state["features"])
        cause = model_diagnosis.root_cause

        return {
            "root_cause": cause,
            "agent_messages": [
                *state.get("agent_messages", []),
                AgentMessage(
                    sender="analyzer",
                    receiver="rag",
                    content=f"Top root-cause hypothesis: {cause.label} p={cause.probability:.2f}.",
                    message_type="decision",
                ),
            ],
            "trace": [*state.get("trace", []), f"analyzer_{cause.label}"],
        }

    def _retrieve_context(self, state: DiagnosisState) -> DiagnosisState:
        evidence = self.retriever.retrieve(state["features"], focus=state["root_cause"].label)
        return {
            "rag_evidence": evidence,
            "context": [item.text for item in evidence],
            "agent_messages": [
                *state.get("agent_messages", []),
                AgentMessage(
                    sender="rag",
                    receiver="prescriber",
                    content=f"Retrieved {len(evidence)} maintenance evidence snippets.",
                    message_type="evidence",
                ),
            ],
            "trace": [*state.get("trace", []), "rag_retrieved"],
        }

    def _prescribe(self, state: DiagnosisState) -> DiagnosisState:
        severity = state["detection"].severity
        label = state["root_cause"].label
        actions = self.action_catalog.build(label, severity)

        return {
            "actions": actions,
            "agent_messages": [
                *state.get("agent_messages", []),
                AgentMessage(
                    sender="prescriber",
                    receiver="safety",
                    content=f"Generated {len(actions)} proposed maintenance action(s).",
                    message_type="handoff",
                ),
            ],
            "trace": [*state.get("trace", []), "prescriber_actions_ready"],
        }

    def _validate(self, state: DiagnosisState) -> DiagnosisState:
        validation = self.safety_policy.validate(
            state.get("actions", []),
            state["detection"].severity,
        )
        metrics = state.get("metrics", RuntimeMetrics())
        metrics.tool_calls += 2
        metrics.estimated_oee_gain_percent = round(
            min(8.0, state["detection"].anomaly_score * 4.5), 2
        )
        return {
            "validation": validation,
            "metrics": metrics,
            "agent_messages": [
                *state.get("agent_messages", []),
                AgentMessage(
                    sender="safety",
                    receiver="supervisor",
                    content=f"Safety validation approved={validation.approved}.",
                    message_type="decision",
                ),
            ],
            "trace": [*state.get("trace", []), "safety_validated"],
        }

    def _route_after_validation(self, state: DiagnosisState) -> str:
        metadata = state.get("metadata", {})
        if (
            metadata.get("force_human_review")
            or not state["validation"].approved
            or state["detection"].confidence < self.config.review_probability_threshold
        ):
            return "human_review"
        return "report"

    def _human_review(self, state: DiagnosisState) -> DiagnosisState:
        metadata = state.get("metadata", {})
        review = HumanReview(
            required=True,
            reason=str(metadata.get("review_reason", "Policy requires operator confirmation.")),
            approved=metadata.get("human_approved"),
            reviewer=metadata.get("reviewer"),
        )
        return {
            "human_review": review,
            "agent_messages": [
                *state.get("agent_messages", []),
                AgentMessage(
                    sender="human_review",
                    receiver="supervisor",
                    content=f"Human review required; approved={review.approved}.",
                    message_type="review",
                ),
            ],
            "trace": [*state.get("trace", []), "human_review_recorded"],
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
            rag_evidence=state.get("rag_evidence", []),
            guardrails=state.get("guardrails", []),
            metrics=state.get("metrics", RuntimeMetrics()),
            agent_messages=state.get("agent_messages", []),
        )
        return {
            "report": report,
            "trace": [*state.get("trace", []), "report_finalized"],
        }
