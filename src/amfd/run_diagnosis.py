from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from amfd.agents.workflow import FaultDiagnosisWorkflow
from amfd.core.config import load_config
from amfd.data.ingestion import load_sensor_csv

app = typer.Typer(help="Run an autonomous manufacturing fault diagnosis.")
console = Console()


@app.command()
def diagnose(
    csv_path: Annotated[Path, typer.Argument(exists=True, readable=True)],
    machine_id: Annotated[str, typer.Option(help="Machine or asset identifier.")] = "DEMO-001",
    config_path: Annotated[Path | None, typer.Option(help="Optional YAML config path.")] = None,
) -> None:
    config = load_config(config_path)
    sensor_window = load_sensor_csv(
        csv_path,
        machine_id=machine_id,
        sampling_rate_hz=config.sampling_rate_hz,
    )
    workflow = FaultDiagnosisWorkflow(config)
    result = workflow.run({"machine_id": machine_id, "sensor_window": sensor_window})
    report = result["report"].model_dump(mode="json")
    console.print(json.dumps(report, indent=2))


def main() -> None:
    app()


if __name__ == "__main__":
    main()

