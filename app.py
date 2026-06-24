from __future__ import annotations

import html
import json
import os
import sys
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from amfd.agents.workflow import FaultDiagnosisWorkflow  # noqa: E402
from amfd.backend.service import DiagnosisRequest, DiagnosisService  # noqa: E402
from amfd.core.config import load_config  # noqa: E402
from amfd.data.ingestion import load_sensor_csv  # noqa: E402
from amfd.data.synthetic import generate_bearing_window  # noqa: E402

INDUSTRIAL_CSS = """
:root {
  --amfd-bg: #0b1220;
  --amfd-panel: rgba(13, 20, 35, 0.86);
  --amfd-panel-strong: #111a2b;
  --amfd-line: rgba(148, 163, 184, 0.18);
  --amfd-ink: #e2e8f0;
  --amfd-muted: #94a3b8;
  --amfd-cyan: #22d3ee;
  --amfd-blue: #60a5fa;
  --amfd-green: #34d399;
  --amfd-amber: #fbbf24;
  --amfd-red: #fb7185;
  --amfd-shadow: 0 22px 48px rgba(2, 6, 23, 0.34);
}

body,
.gradio-container {
  background:
    radial-gradient(circle at top left, rgba(34, 211, 238, 0.12), transparent 24%),
    radial-gradient(circle at top right, rgba(96, 165, 250, 0.10), transparent 20%),
    linear-gradient(180deg, #0b1220 0%, #0f172a 100%);
  color: var(--amfd-ink);
}

.amfd-shell {
  max-width: 1560px;
  margin: 0 auto;
}

.amfd-hero {
  border: 1px solid var(--amfd-line);
  border-radius: 18px;
  background: linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(15, 23, 42, 0.78));
  box-shadow: var(--amfd-shadow);
  padding: 24px 28px;
  margin-bottom: 18px;
}

.amfd-eyebrow {
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--amfd-cyan);
  font-size: 11px;
  font-weight: 800;
  margin: 0 0 8px;
}

.amfd-title {
  margin: 0;
  font-size: 30px;
  font-weight: 800;
  line-height: 1.1;
}

.amfd-subtitle {
  margin: 8px 0 0;
  color: var(--amfd-muted);
  max-width: 76ch;
}

.amfd-layout {
  gap: 18px;
}

.amfd-card {
  border: 1px solid var(--amfd-line);
  border-radius: 16px;
  background: var(--amfd-panel);
  box-shadow: var(--amfd-shadow);
}

.amfd-card .gr-form {
  border: 0;
  background: transparent;
}

.amfd-card .gr-box,
.amfd-card .gr-panel {
  border: 0;
  background: transparent;
}

.amfd-control {
  padding: 20px;
}

.amfd-control h3,
.amfd-main h3 {
  margin: 0 0 12px;
  font-size: 14px;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  color: var(--amfd-muted);
}

.amfd-button button,
.amfd-demo button {
  height: 44px;
  border-radius: 10px;
  border: 1px solid transparent;
}

.amfd-button button {
  background: linear-gradient(135deg, #22d3ee, #2563eb);
}

.amfd-demo button {
  background: rgba(15, 23, 42, 0.88);
  border-color: var(--amfd-line);
}

.amfd-banner {
  border: 1px solid var(--amfd-line);
  border-radius: 16px;
  background: linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(23, 37, 84, 0.78));
  padding: 18px 20px;
}

.amfd-banner-grid {
  display: grid;
  gap: 12px;
  grid-template-columns: minmax(0, 1.2fr) repeat(4, minmax(120px, 1fr));
  align-items: stretch;
}

.amfd-banner h2 {
  margin: 4px 0 8px;
  font-size: 26px;
}

.amfd-banner .meta {
  color: var(--amfd-muted);
  font-size: 13px;
}

.chip {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 6px 10px;
  border-radius: 999px;
  font-size: 12px;
  font-weight: 700;
  border: 1px solid var(--amfd-line);
}

.chip.good {
  background: rgba(52, 211, 153, 0.12);
  color: #86efac;
}

.chip.warn {
  background: rgba(251, 191, 36, 0.12);
  color: #fcd34d;
}

.chip.bad {
  background: rgba(251, 113, 133, 0.12);
  color: #fda4af;
}

.amfd-kpis {
  display: grid;
  gap: 12px;
  grid-template-columns: repeat(4, minmax(140px, 1fr));
  margin-top: 14px;
}

.kpi {
  padding: 14px 16px;
  border: 1px solid var(--amfd-line);
  border-radius: 14px;
  background: rgba(15, 23, 42, 0.68);
}

.kpi span {
  display: block;
  color: var(--amfd-muted);
  font-size: 11px;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-bottom: 8px;
}

.kpi strong {
  font-size: 24px;
}

.kpi small {
  display: block;
  margin-top: 6px;
  color: var(--amfd-muted);
  font-size: 12px;
}

.amfd-plot-grid {
  display: grid;
  gap: 16px;
  grid-template-columns: minmax(0, 1.55fr) minmax(0, 1fr);
  margin-top: 16px;
}

.amfd-section {
  padding: 18px;
}

.amfd-section .gr-markdown,
.amfd-section .gr-plot,
.amfd-section .gr-dataframe,
.amfd-section .gr-code,
.amfd-section .gr-textbox {
  background: transparent;
}

.amfd-tabs {
  margin-top: 16px;
}

.amfd-empty {
  color: var(--amfd-muted);
}

@media (max-width: 1100px) {
  .amfd-banner-grid,
  .amfd-kpis,
  .amfd-plot-grid {
    grid-template-columns: 1fr;
  }
}
"""


def build_app(config_path: str | Path | None = None) -> Any:
    """Return the compiled LangGraph app for workers, tests, or deployment hooks."""
    workflow = FaultDiagnosisWorkflow(load_config(config_path))
    return workflow._graph  # noqa: SLF001 - intentional deployment entrypoint.


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

    rpm_values = window.rpm if window.rpm else [float("nan")] * len(window.vibration)
    frame = pd.DataFrame(
        {
            "sample": range(len(window.vibration)),
            "vibration": window.vibration,
            "rpm": rpm_values,
        }
    )
    return source, window.vibration, window.rpm, frame


def _table_frame(records: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(records)
    missing = [column for column in columns if column not in frame.columns]
    for column in missing:
        frame[column] = ""
    return frame.loc[:, columns]


def _build_waveform_figure(frame: pd.DataFrame, title: str) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=frame["sample"],
            y=frame["vibration"],
            name="Vibration",
            line=dict(color="#22d3ee", width=2),
        ),
        secondary_y=False,
    )
    if frame["rpm"].notna().any():
        rpm_frame = frame.dropna(subset=["rpm"])
        fig.add_trace(
            go.Scatter(
                x=rpm_frame["sample"],
                y=rpm_frame["rpm"],
                name="RPM",
                line=dict(color="#fbbf24", width=1.7, dash="dot"),
            ),
            secondary_y=True,
        )
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=410,
        margin=dict(l=18, r=18, t=50, b=18),
        legend=dict(orientation="h"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.96)",
        font=dict(color="#0f172a"),
    )
    fig.update_yaxes(title_text="Acceleration", secondary_y=False)
    fig.update_yaxes(title_text="RPM", secondary_y=True, showgrid=False)
    return fig


def _build_spectrum_figure(frame: pd.DataFrame, title: str, sampling_rate_hz: int) -> go.Figure:
    signal = np.asarray(frame["vibration"], dtype=float)
    centered = signal - signal.mean()
    spectrum = np.abs(np.fft.rfft(centered))
    frequencies = np.fft.rfftfreq(signal.size, d=1 / sampling_rate_hz)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frequencies,
            y=spectrum,
            name="Spectrum",
            line=dict(color="#60a5fa", width=2),
        )
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=410,
        margin=dict(l=18, r=18, t=50, b=18),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.96)",
        font=dict(color="#0f172a"),
    )
    fig.update_xaxes(title_text="Frequency (Hz)", range=[0, min(float(frequencies.max()), 1200.0)])
    fig.update_yaxes(title_text="Magnitude")
    return fig


def _render_banner(
    report: Any,
    source: str,
    sample_count: int,
    duration_seconds: float,
    rpm_present: bool,
) -> str:
    severity = report.detection.severity.value.upper()
    severity_class = {
        "NORMAL": "good",
        "WARNING": "warn",
        "CRITICAL": "bad",
    }.get(severity, "warn")
    validation_label = "AUTO-APPROVED" if report.validation.approved else "REVIEW REQUIRED"
    validation_class = "good" if report.validation.approved else "warn"
    sample_rate_label = f"{report.features.dominant_frequency_hz:.1f} Hz dominant"
    rpm_label = "RPM present" if rpm_present else "RPM absent"

    return f"""
    <div class="amfd-banner">
      <div class="amfd-banner-grid">
        <div>
          <div class="amfd-eyebrow">Industrial diagnosis console</div>
          <h2>{html.escape(report.machine_id)}</h2>
          <div class="meta">
            Incident <code>{html.escape(report.incident_id)}</code> &middot;
            Source {html.escape(source)} &middot;
            {sample_count} samples &middot;
            {duration_seconds:.2f}s window
          </div>
        </div>
        <div class="kpi">
          <span>Severity</span>
          <strong class="chip {severity_class}">{severity}</strong>
          <small>{sample_rate_label}</small>
        </div>
        <div class="kpi">
          <span>Validation</span>
          <strong class="chip {validation_class}">{validation_label}</strong>
          <small>{rpm_label}</small>
        </div>
        <div class="kpi">
          <span>Latency</span>
          <strong>{report.metrics.latency_ms:.0f} ms</strong>
          <small>End-to-end graph runtime</small>
        </div>
        <div class="kpi">
          <span>OEE gain</span>
          <strong>{report.metrics.estimated_oee_gain_percent:.2f}%</strong>
          <small>Estimated recovery uplift</small>
        </div>
      </div>
    </div>
    """


def _render_kpi_strip(report: Any, sample_count: int, duration_seconds: float) -> str:
    feature = report.features
    rpm_mean = f"{feature.rpm_mean:.0f}" if feature.rpm_mean is not None else "n/a"
    return f"""
    <div class="amfd-kpis">
      <div class="kpi">
        <span>Anomaly score</span>
        <strong>{report.detection.anomaly_score:.2f}</strong>
        <small>Thresholded severity signal</small>
      </div>
      <div class="kpi">
        <span>Confidence</span>
        <strong>{report.detection.confidence:.2f}</strong>
        <small>Decision confidence</small>
      </div>
      <div class="kpi">
        <span>RMS</span>
        <strong>{feature.rms:.4f}</strong>
        <small>Window vibration energy</small>
      </div>
      <div class="kpi">
        <span>Crest factor</span>
        <strong>{feature.crest_factor:.2f}</strong>
        <small>Impulsiveness indicator</small>
      </div>
      <div class="kpi">
        <span>Samples</span>
        <strong>{sample_count}</strong>
        <small>{duration_seconds:.2f}s captured</small>
      </div>
      <div class="kpi">
        <span>Dominant freq.</span>
        <strong>{feature.dominant_frequency_hz:.1f} Hz</strong>
        <small>Spectral peak</small>
      </div>
      <div class="kpi">
        <span>RPM mean</span>
        <strong>{rpm_mean}</strong>
        <small>Operating speed context</small>
      </div>
      <div class="kpi">
        <span>Tool calls</span>
        <strong>{report.metrics.tool_calls}</strong>
        <small>Graph/tool interactions</small>
      </div>
    </div>
    """


def diagnose(
    machine_id: str,
    uploaded_file: str | None,
    use_demo: bool,
    operator_notes: str,
    sampling_rate_hz: int,
    force_human_review: bool,
) -> tuple[str, str, go.Figure, go.Figure, pd.DataFrame, pd.DataFrame, pd.DataFrame, str, str]:
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

    sample_count = len(frame)
    duration_seconds = sample_count / sampling_rate_hz if sampling_rate_hz else 0.0
    rpm_present = bool(frame["rpm"].notna().any())

    banner_html = _render_banner(report, source, sample_count, duration_seconds, rpm_present)
    kpi_html = _render_kpi_strip(report, sample_count, duration_seconds)

    waveform_fig = _build_waveform_figure(frame, "Vibration and RPM trend")
    spectrum_fig = _build_spectrum_figure(frame, "Frequency spectrum", sampling_rate_hz)

    actions_df = _table_frame(
        [action.model_dump(mode="json") for action in report.actions],
        ["action", "priority", "rationale", "requires_human_approval"],
    )
    evidence_df = _table_frame(
        [item.model_dump(mode="json") for item in report.rag_evidence],
        ["source", "text", "bm25_score", "rerank_score"],
    )
    guardrails_df = _table_frame(
        [item.model_dump(mode="json") for item in report.guardrails],
        ["category", "action", "detail"],
    )

    trace = " -> ".join(response.trace)
    report_json = json.dumps(report.model_dump(mode="json"), indent=2)
    return (
        banner_html,
        kpi_html,
        waveform_fig,
        spectrum_fig,
        actions_df,
        evidence_df,
        guardrails_df,
        trace,
        report_json,
    )


def run_demo(
    machine_id: str,
    operator_notes: str,
    sampling_rate_hz: int,
    force_human_review: bool,
) -> tuple[str, str, go.Figure, go.Figure, pd.DataFrame, pd.DataFrame, pd.DataFrame, str, str]:
    return diagnose(
        machine_id=machine_id,
        uploaded_file=None,
        use_demo=True,
        operator_notes=operator_notes,
        sampling_rate_hz=sampling_rate_hz,
        force_human_review=force_human_review,
    )


(
    INITIAL_BANNER,
    INITIAL_KPIS,
    INITIAL_WAVEFORM,
    INITIAL_SPECTRUM,
    INITIAL_ACTIONS,
    INITIAL_EVIDENCE,
    INITIAL_GUARDRAILS,
    INITIAL_TRACE,
    INITIAL_REPORT_JSON,
) = run_demo("PUMP-101", "Synthetic demo bearing-fault window.", 12000, False)


def build_ui() -> Any:
    with gr.Blocks(title="AMFD Industrial Diagnosis") as demo:
        gr.HTML(
            """
            <div class="amfd-hero">
              <div class="amfd-eyebrow">Autonomous Manufacturing Fault Diagnosis</div>
              <h1 class="amfd-title">Industrial AI Operations Console</h1>
              <p class="amfd-subtitle">
                Upload a vibration CSV or run the synthetic demo stream. The interface is
                designed for operator triage, safety gating, evidence review, and maintenance
                handoff with no external API keys required.
              </p>
            </div>
            """
        )

        with gr.Row(elem_classes=["amfd-layout"]):
            with gr.Column(scale=1, min_width=360, elem_classes=["amfd-card", "amfd-control"]):
                gr.HTML("<h3>Acquisition & safety</h3>")
                machine_id = gr.Textbox(
                    label="Machine ID",
                    value="PUMP-101",
                    placeholder="PUMP-101",
                )
                uploaded_file = gr.File(
                    label="Vibration CSV",
                    file_types=[".csv"],
                    type="filepath",
                )
                use_demo = gr.Checkbox(label="Use synthetic demo window", value=True)
                operator_notes = gr.Textbox(
                    label="Operator notes",
                    placeholder="Observed elevated vibration during startup.",
                    lines=4,
                )
                with gr.Accordion("Advanced controls", open=False):
                    sampling_rate_hz = gr.Slider(
                        label="Sampling rate (Hz)",
                        minimum=1000,
                        maximum=24000,
                        step=500,
                        value=12000,
                    )
                    force_human_review = gr.Checkbox(
                        label="Force human review",
                        value=False,
                    )

                with gr.Row():
                    run_btn = gr.Button(
                        "Analyze CSV",
                        variant="primary",
                        elem_classes=["amfd-button"],
                    )
                    demo_btn = gr.Button("Run Demo", elem_classes=["amfd-demo"])

                gr.Markdown(
                    "Use the repo sample at `examples/bearing_sample.csv` or upload your own CSV."
                )

            with gr.Column(scale=2, min_width=760, elem_classes=["amfd-main"]):
                banner = gr.HTML(value=INITIAL_BANNER)
                kpis = gr.HTML(value=INITIAL_KPIS)
                with gr.Row(elem_classes=["amfd-plot-grid"]):
                    waveform = gr.Plot(label="Waveform", value=INITIAL_WAVEFORM)
                    spectrum = gr.Plot(label="Spectrum", value=INITIAL_SPECTRUM)

                with gr.Tabs(elem_classes=["amfd-tabs"]):
                    with gr.Tab("Actions"):
                        actions_table = gr.Dataframe(
                            label="Recommended actions",
                            value=INITIAL_ACTIONS,
                            headers=[
                                "action",
                                "priority",
                                "rationale",
                                "requires_human_approval",
                            ],
                            datatype=["str", "str", "str", "bool"],
                            interactive=False,
                            wrap=True,
                        )
                    with gr.Tab("Evidence"):
                        evidence_table = gr.Dataframe(
                            label="Retrieved evidence",
                            value=INITIAL_EVIDENCE,
                            headers=["source", "text", "bm25_score", "rerank_score"],
                            datatype=["str", "str", "number", "number"],
                            interactive=False,
                            wrap=True,
                        )
                    with gr.Tab("Guardrails"):
                        guardrails_table = gr.Dataframe(
                            label="Safety findings",
                            value=INITIAL_GUARDRAILS,
                            headers=["category", "action", "detail"],
                            datatype=["str", "str", "str"],
                            interactive=False,
                            wrap=True,
                        )
                    with gr.Tab("Trace"):
                        trace_box = gr.Textbox(
                            label="Agent trace",
                            lines=4,
                            value=INITIAL_TRACE,
                        )
                    with gr.Tab("JSON"):
                        report_json = gr.Code(
                            label="Incident report JSON",
                            language="json",
                            value=INITIAL_REPORT_JSON,
                        )

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
                banner,
                kpis,
                waveform,
                spectrum,
                actions_table,
                evidence_table,
                guardrails_table,
                trace_box,
                report_json,
            ],
        )

        demo_btn.click(
            fn=run_demo,
            inputs=[
                machine_id,
                operator_notes,
                sampling_rate_hz,
                force_human_review,
            ],
            outputs=[
                banner,
                kpis,
                waveform,
                spectrum,
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
        css=INDUSTRIAL_CSS,
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
    )
