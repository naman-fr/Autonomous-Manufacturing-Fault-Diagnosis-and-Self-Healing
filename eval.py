from __future__ import annotations

import json
from dataclasses import dataclass
from time import perf_counter

from amfd.agents.workflow import FaultDiagnosisWorkflow
from amfd.core.config import DiagnosisConfig
from amfd.data.synthetic import generate_bearing_window


@dataclass(frozen=True)
class EvalCase:
    label: str
    fault_frequency_hz: float
    intensity: float
    expected_root_cause: str


CASES = [
    EvalCase("bearing_outer_race_light", 220, 0.22, "bearing_defect"),
    EvalCase("bearing_inner_race_medium", 310, 0.30, "bearing_defect"),
    EvalCase("bearing_ball_fault_high", 420, 0.40, "bearing_defect"),
]


def run_eval() -> dict[str, float]:
    config = DiagnosisConfig(
        sampling_rate_hz=2_000,
        anomaly_threshold=0.45,
        high_risk_threshold=0.82,
        max_refinement_loops=1,
    )
    workflow = FaultDiagnosisWorkflow(config)
    correct = 0
    latencies: list[float] = []

    for case in CASES:
        window = generate_bearing_window(
            machine_id=f"EVAL-{case.label}",
            sampling_rate_hz=config.sampling_rate_hz,
            seconds=1.0,
            fault_frequency_hz=case.fault_frequency_hz,
            fault_intensity=case.intensity,
        )
        started = perf_counter()
        result = workflow.run({"machine_id": window.machine_id, "sensor_window": window})
        latencies.append((perf_counter() - started) * 1000)
        if result["report"].root_cause.label == case.expected_root_cause:
            correct += 1

    f1_proxy = correct / len(CASES)
    return {
        "cases": float(len(CASES)),
        "f1_proxy": round(f1_proxy, 4),
        "latency_p50_ms": round(sorted(latencies)[len(latencies) // 2], 2),
        "latency_max_ms": round(max(latencies), 2),
        "target_f1": 0.95,
        "target_latency_ms": 2000.0,
    }


if __name__ == "__main__":
    print(json.dumps(run_eval(), indent=2))

