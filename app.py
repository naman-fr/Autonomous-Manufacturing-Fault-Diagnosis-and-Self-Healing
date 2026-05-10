from __future__ import annotations

from amfd.agents.workflow import FaultDiagnosisWorkflow
from amfd.core.config import load_config


def build_app(config_path: str | None = None):
    """Return a compiled LangGraph app for workers, tests, or LangServe-style deployment."""
    workflow = FaultDiagnosisWorkflow(load_config(config_path))
    return workflow._graph  # noqa: SLF001 - intentionally exposes the compiled graph entrypoint.


graph = build_app()

