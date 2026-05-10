from __future__ import annotations

from typing import Annotated

from amfd.backend.service import DiagnosisRequest, DiagnosisService, HealthResponse

try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import Response
except ImportError as exc:  # pragma: no cover - optional runtime dependency
    raise RuntimeError("Install backend dependencies with `pip install -e .[backend]`.") from exc


service = DiagnosisService()
app = FastAPI(
    title="AMFD Industrial Diagnosis API",
    version="0.1.0",
    description="Agentic manufacturing fault diagnosis and self-healing recommendation API.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse()


@app.post("/api/v1/diagnose")
async def diagnose(request: DiagnosisRequest) -> dict[str, object]:
    try:
        response = service.diagnose_window(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return response.model_dump(mode="json")


@app.post("/api/v1/diagnose/csv")
async def diagnose_csv(
    file: Annotated[UploadFile, File()],
    machine_id: Annotated[str, Form()] = "PUMP-101",
    operator_notes: Annotated[str, Form()] = "",
    force_human_review: Annotated[bool, Form()] = False,
) -> dict[str, object]:
    csv_text = (await file.read()).decode("utf-8")
    response = service.diagnose_csv_text(csv_text, machine_id, operator_notes, force_human_review)
    return response.model_dump(mode="json")


@app.get("/api/v1/demo")
async def demo(machine_id: str = "PUMP-101") -> dict[str, object]:
    return service.demo(machine_id).model_dump(mode="json")


@app.get("/metrics")
async def metrics() -> Response:
    body = "\n".join(
        [
            "# HELP amfd_service_info AMFD service metadata",
            "# TYPE amfd_service_info gauge",
            'amfd_service_info{service="amfd-backend"} 1',
            "",
        ]
    )
    return Response(content=body, media_type="text/plain")

