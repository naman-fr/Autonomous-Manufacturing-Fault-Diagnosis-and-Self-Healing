from __future__ import annotations

from amfd.backend.service import DiagnosisRequest, DiagnosisService
from amfd.core.config import DiagnosisConfig
from amfd.data.synthetic import generate_bearing_window


def test_backend_service_diagnoses_window() -> None:
    config = DiagnosisConfig(
        sampling_rate_hz=2_000,
        anomaly_threshold=0.45,
        high_risk_threshold=0.82,
        max_refinement_loops=1,
    )
    window = generate_bearing_window(
        machine_id="API-TEST",
        sampling_rate_hz=2_000,
        seconds=1.0,
        fault_frequency_hz=250,
        fault_intensity=0.30,
    )
    response = DiagnosisService(config).diagnose_window(
        DiagnosisRequest(
            machine_id=window.machine_id,
            vibration=window.vibration,
            rpm=window.rpm,
            sampling_rate_hz=window.sampling_rate_hz,
            operator_notes="operator phone 9876543210",
        )
    )

    assert response.report.machine_id == "API-TEST"
    assert response.report.root_cause.label == "bearing_defect"
    assert response.report.rag_evidence
    assert response.report.guardrails[0].category == "pii.phone"

