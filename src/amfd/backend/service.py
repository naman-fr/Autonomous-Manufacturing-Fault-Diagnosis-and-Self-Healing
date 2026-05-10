from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

from pydantic import BaseModel, Field

from amfd.agents.workflow import FaultDiagnosisWorkflow
from amfd.core.config import DiagnosisConfig, load_config
from amfd.core.models import IncidentReport, SensorWindow
from amfd.data.ingestion import load_sensor_csv
from amfd.data.synthetic import generate_bearing_window


class DiagnosisRequest(BaseModel):
    machine_id: str = Field(default="PUMP-101", min_length=1)
    vibration: list[float] = Field(default_factory=list)
    rpm: list[float] = Field(default_factory=list)
    sampling_rate_hz: int | None = Field(default=None, gt=0)
    operator_notes: str = ""
    force_human_review: bool = False


class DiagnosisResponse(BaseModel):
    report: IncidentReport
    trace: list[str]


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "amfd-backend"
    version: str = "0.1.0"


class DiagnosisService:
    def __init__(self, config: DiagnosisConfig | None = None) -> None:
        self.config = config or load_config()
        self.workflow = FaultDiagnosisWorkflow(self.config)

    def diagnose_window(self, request: DiagnosisRequest) -> DiagnosisResponse:
        sampling_rate = request.sampling_rate_hz or self.config.sampling_rate_hz
        if not request.vibration:
            raise ValueError("request.vibration must contain at least four samples")
        sensor_window = SensorWindow(
            machine_id=request.machine_id,
            vibration=request.vibration,
            rpm=request.rpm,
            sampling_rate_hz=sampling_rate,
        )
        return self._run(sensor_window, request.operator_notes, request.force_human_review)

    def diagnose_csv_text(
        self,
        csv_text: str,
        machine_id: str,
        operator_notes: str = "",
        force_human_review: bool = False,
    ) -> DiagnosisResponse:
        with NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as handle:
            handle.write(csv_text)
            tmp_path = Path(handle.name)
        try:
            sensor_window = load_sensor_csv(tmp_path, machine_id, self.config.sampling_rate_hz)
            return self._run(sensor_window, operator_notes, force_human_review)
        finally:
            tmp_path.unlink(missing_ok=True)

    def demo(self, machine_id: str = "PUMP-101") -> DiagnosisResponse:
        sensor_window = generate_bearing_window(
            machine_id=machine_id,
            sampling_rate_hz=2_000,
            seconds=1.0,
            fault_frequency_hz=250,
            fault_intensity=0.50,
        )
        return self._run(sensor_window, "Synthetic demo bearing-fault window.", False)

    def _run(
        self,
        sensor_window: SensorWindow,
        operator_notes: str,
        force_human_review: bool,
    ) -> DiagnosisResponse:
        result = self.workflow.run(
            {
                "machine_id": sensor_window.machine_id,
                "sensor_window": sensor_window,
                "metadata": {
                    "operator_notes": operator_notes,
                    "force_human_review": force_human_review,
                    "review_reason": "Operator requested manual review.",
                },
            }
        )
        return DiagnosisResponse(report=result["report"], trace=result.get("trace", []))
