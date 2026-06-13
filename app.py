from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd
import plotly.express as px

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from amfd.agents.workflow import FaultDiagnosisWorkflow  # noqa: E402
from amfd.backend.service import DiagnosisRequest, DiagnosisService  # noqa: E402
from amfd.core.config import load_config  # noqa: E402
from amfd.data.ingestion import load_sensor_csv  # noqa: E402
from amfd.data.synthetic import generate_bearing_window  # noqa: E402


def build_app(config_path: str | Path | None = None) -> Any:
    """Return the compiled LangGraph app for workers, tests, or deployment hooks."""
    workflow = FaultDiagnosisWorkflow(load_config(config_path))
    return workflow._graph  # noqa: SLF001 - intentional public entrypoint for deployment.


SERVICE = DiagnosisService(load_config(ROOT / "configs/default.yaml"))
graph = build_app(ROOT / "configs/default.yaml")


def _load_sensor_window(
    machine_id: str,
    uploaded_file: str | None,
    use_demo: bool,
    sampling_rate_hz: int,
) -> tuple[str, list[float], list[float], pd.DataFrame]:
    if use_demo or uploaded_file is None:
        window = generate_bearing_window(
            machine_id=machine_id,
            sampling_rate_hz=sampling_rate_hz,
            seconds=1.0,
            fault_frequency_hz=250.0,
            fault_intensity=0.50,
        )
        source = "Synthetic demo window"
    else:
        window = load_sensor_csv(uploaded_file, machine_id, sampling_rate_hz)
        source = Path(uploaded_file).name

    rpm_values: list[float] = window.rpm if window.rpm else [float("nan")] * len(window.vibration)
    frame = pd.DataFrame(
        {
            "sample": range(len(window.vibration)),
            "vibration": window.vibration,
            "rpm": rpm_values,
        }
    )
    return source, window.vibration, window.rpm, frame


def diagnose(
    machine_id: str,
    uploaded_file: str | None,
    use_demo: bool,
    operator_notes: str,
    sampling_rate_hz: int,
    force_human_review: bool,
) -> tuple[str, Any, pd.DataFrame, pd.DataFrame, pd.DataFrame, str, str]:
    source, vibration, rpm, frame = _load_sensor_window(
        machine_id=machine_id,
        uploaded_file=uploaded_file,
        use_demo=use_demo,
        sampling_rate_hz=sampling_rate_hz,
    )
    request = DiagnosisRequest(
        machine_id=machine_id,
        vibration=vibration,
        rpm=rpm,
        sampling_rate_hz=sampling_rate_hz,
        operator_notes=operator_notes,
        force_human_review=force_human_review,
    )
    response = SERVICE.diagnose_window(request)
    report = response.report

    figure = px.line(frame, x="sample", y="vibration", title="Vibration profile")
    if frame["rpm"].notna().any():
        rpm_frame = frame.dropna(subset=["rpm"])
        figure.add_scatter(
            x=rpm_frame["sample"],
            y=rpm_frame["rpm"],
            name="rpm",
            mode="lines",
            line=dict(dash="dot"),
            yaxis="y2",
        )
        figure.update_layout(
            yaxis2=dict(
                overlaying="y",
                side="right",
                title="rpm",
                showgrid=False,
            )
        )
    figure.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20),
        height=420,
        legend=dict(orientation="h"),
    )

    actions_df = pd.DataFrame([action.model_dump(mode="json") for action in report.actions])
    evidence_df = pd.DataFrame([item.model_dump(mode="json") for item in report.rag_evidence])
    guardrails_df = pd.DataFrame([item.model_dump(mode="json") for item in report.guardrails])

    summary = (
        f"### {report.machine_id}\n\n"
        f"**Incident:** `{report.incident_id}`  \n"
        f"**Source:** {source}  \n"
        f"**Severity:** {report.detection.severity.value.upper()}  \n"
        f"**Root cause:** {report.root_cause.label} "
        f"({report.root_cause.probability:.2f})  \n"
        f"**Validation:** {'approved' if report.validation.approved else 'review required'}  \n"
        f"**Latency:** {report.metrics.latency_ms:.0f} ms  \n"
        f"**Estimated OEE gain:** {report.metrics.estimated_oee_gain_percent:.2f}%"
    )

    report_json = json.dumps(report.model_dump(mode="json"), indent=2)
    trace = " -> ".join(response.trace)
    return summary, figure, actions_df, evidence_df, guardrails_df, trace, report_json


def build_ui() -> Any:
    with gr.Blocks(title="AMFD Industrial Diagnosis") as demo:
        gr.Markdown(
            "# Autonomous Manufacturing Fault Diagnosis and Self-Healing\n"
            "Upload a vibration CSV or run the deterministic demo window. "
            "The Space is self-contained and does not require external API keys."
        )

        with gr.Row():
            with gr.Column(scale=1):
                machine_id = gr.Textbox(
                    label="Machine ID",
                    value="PUMP-101",
                    placeholder="PUMP-101",
                )
                uploaded_file = gr.File(label="Vibration CSV", file_types=[".csv"], type="filepath")
                use_demo = gr.Checkbox(label="Use synthetic demo window", value=True)
                operator_notes = gr.Textbox(
                    label="Operator notes",
                    placeholder="Observed elevated vibration during startup.",
                    lines=4,
                )
                sampling_rate_hz = gr.Slider(
                    label="Sampling rate (Hz)",
                    minimum=1000,
                    maximum=24000,
                    step=500,
                    value=12000,
                )
                force_human_review = gr.Checkbox(label="Force human review", value=False)
                run_btn = gr.Button("Run Diagnosis", variant="primary")

            with gr.Column(scale=2):
                summary = gr.Markdown()
                plot = gr.Plot(label="Signal")

        with gr.Tabs():
            with gr.Tab("Actions"):
                actions_table = gr.Dataframe(
                    label="Recommended actions",
                    headers=["action", "priority", "rationale", "requires_human_approval"],
                    datatype=["str", "str", "str", "bool"],
                    interactive=False,
                    wrap=True,
                )
            with gr.Tab("RAG Evidence"):
                evidence_table = gr.Dataframe(
                    label="Retrieved evidence",
                    headers=["source", "text", "bm25_score", "rerank_score"],
                    datatype=["str", "str", "number", "number"],
                    interactive=False,
                    wrap=True,
                )
            with gr.Tab("Guardrails"):
                guardrails_table = gr.Dataframe(
                    label="Safety findings",
                    headers=["category", "action", "detail"],
                    datatype=["str", "str", "str"],
                    interactive=False,
                    wrap=True,
                )
            with gr.Tab("Trace"):
                trace_box = gr.Textbox(label="Agent trace", lines=4)
            with gr.Tab("JSON"):
                report_json = gr.Code(label="Incident report JSON", language="json")

        run_btn.click(
            fn=diagnose,
            inputs=[
                machine_id,
                uploaded_file,
                use_demo,
                operator_notes,
                sampling_rate_hz,
                force_human_review,
            ],
            outputs=[
                summary,
                plot,
                actions_table,
                evidence_table,
                guardrails_table,
                trace_box,
                report_json,
            ],
        )

        gr.Examples(
            examples=[
                [
                    "PUMP-101",
                    str(ROOT / "examples" / "bearing_sample.csv"),
                    False,
                    "Bearing sound is rough and hotter than baseline.",
                    12000,
                    False,
                ]
            ],
            inputs=[
                machine_id,
                uploaded_file,
                use_demo,
                operator_notes,
                sampling_rate_hz,
                force_human_review,
            ],
            label="Example input",
        )

    return demo


demo = build_ui()


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=8).launch(
        theme=gr.themes.Soft(),
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
    )
