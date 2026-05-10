from __future__ import annotations

from amfd.agents.workflow import FaultDiagnosisWorkflow
from amfd.core.config import DiagnosisConfig
from amfd.data.synthetic import generate_bearing_window


def test_workflow_returns_valid_incident_report() -> None:
    config = DiagnosisConfig(
        sampling_rate_hz=2_000,
        anomaly_threshold=0.45,
        high_risk_threshold=0.78,
        max_refinement_loops=1,
    )
    window = generate_bearing_window(
        machine_id="PUMP-TEST",
        sampling_rate_hz=2_000,
        seconds=1,
        fault_frequency_hz=250,
        fault_intensity=0.30,
    )

    result = FaultDiagnosisWorkflow(config).run(
        {"machine_id": "PUMP-TEST", "sensor_window": window}
    )

    report = result["report"]
    assert report.machine_id == "PUMP-TEST"
    assert report.root_cause.label == "bearing_defect"
    assert report.actions
    assert report.validation.approved
    assert "report_finalized" in result["trace"]
