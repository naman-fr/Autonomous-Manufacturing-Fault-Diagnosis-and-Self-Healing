from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class DiagnosisConfig(BaseModel):
    sampling_rate_hz: int = Field(default=12_000, gt=0)
    window_seconds: float = Field(default=1.0, gt=0)
    anomaly_threshold: float = Field(default=0.72, ge=0, le=1)
    high_risk_threshold: float = Field(default=0.88, ge=0, le=1)
    review_probability_threshold: float = Field(default=0.72, ge=0, le=1)
    max_refinement_loops: int = Field(default=2, ge=0, le=10)
    synthetic_training_cases: int = Field(default=512, ge=32, le=10_000)
    artifact_dir: str = "models"
    prompt_path: str = "configs/prompts.json"
    policy_path: str = "configs/actions.json"
    knowledge_base_path: str = "configs/knowledge_base.json"
    training_scenarios_path: str = "configs/training_scenarios.json"
    llm_provider: str = "auto"
    llm_model: str = ""
    allowed_actions: tuple[str, ...] = (
        "inspect_bearing",
        "reduce_load",
        "schedule_shutdown",
        "rebalance_rotor",
        "align_coupling",
        "lubricate_bearing",
        "recalibrate_rpm",
    )


def load_config(path: str | Path | None = None) -> DiagnosisConfig:
    if path is None:
        return DiagnosisConfig()

    config_path = Path(path)
    data = _read_simple_yaml(config_path)
    return DiagnosisConfig(**data)


def _read_simple_yaml(path: Path) -> dict[str, Any]:
    """Tiny YAML reader for this repo's flat config to avoid a hard PyYAML dependency."""
    result: dict[str, Any] = {}
    current_list_key: str | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("- ") and current_list_key:
            result.setdefault(current_list_key, []).append(line[2:].strip())
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        current_list_key = None
        if value == "":
            result[key] = []
            current_list_key = key
        elif value.lower() in {"true", "false"}:
            result[key] = value.lower() == "true"
        else:
            result[key] = _coerce_scalar(value)

    return result


def _coerce_scalar(value: str) -> Any:
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value
