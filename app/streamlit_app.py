from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from amfd.agents.workflow import FaultDiagnosisWorkflow
from amfd.core.config import load_config
from amfd.data.ingestion import load_sensor_csv
from amfd.data.synthetic import generate_bearing_window

st.set_page_config(page_title="Manufacturing Fault Diagnosis", layout="wide")

st.title("Autonomous Manufacturing Fault Diagnosis")

config = load_config(Path("configs/default.yaml"))
workflow = FaultDiagnosisWorkflow(config)

with st.sidebar:
    st.header("Asset")
    machine_id = st.text_input("Machine ID", value="PUMP-101")
    uploaded = st.file_uploader("Upload sensor CSV", type=["csv"])
    use_demo = st.button("Load synthetic bearing fault")

if uploaded is not None:
    tmp_path = Path(".streamlit_uploaded_sensor.csv")
    tmp_path.write_bytes(uploaded.getvalue())
    sensor_window = load_sensor_csv(tmp_path, machine_id, config.sampling_rate_hz)
elif use_demo:
    sensor_window = generate_bearing_window(machine_id=machine_id)
else:
    sensor_window = load_sensor_csv("examples/bearing_sample.csv", machine_id, config.sampling_rate_hz)

result = workflow.run({"machine_id": machine_id, "sensor_window": sensor_window})
report = result["report"]

samples = pd.DataFrame(
    {
        "sample": range(len(sensor_window.vibration)),
        "vibration": sensor_window.vibration,
    }
)

summary_cols = st.columns(4)
summary_cols[0].metric("Severity", report.detection.severity.value.upper())
summary_cols[1].metric("Anomaly Score", f"{report.detection.anomaly_score:.2f}")
summary_cols[2].metric("Confidence", f"{report.detection.confidence:.2f}")
summary_cols[3].metric("Root Cause", report.root_cause.label.replace("_", " ").title())

left, right = st.columns([1.5, 1])
with left:
    st.subheader("Vibration Window")
    st.plotly_chart(px.line(samples, x="sample", y="vibration"), use_container_width=True)

with right:
    st.subheader("Recommended Actions")
    for action in report.actions:
        st.write(f"**{action.priority.upper()}** · `{action.action}`")
        st.caption(action.rationale)

st.subheader("Evidence")
st.write(report.detection.evidence)

st.subheader("Agent Trace")
st.code(" -> ".join(result.get("trace", [])))

st.download_button(
    "Download incident report",
    data=json.dumps(report.model_dump(mode="json"), indent=2),
    file_name=f"{report.incident_id}.json",
    mime="application/json",
)

