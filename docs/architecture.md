# System Architecture

The system follows a stateful LangGraph workflow. Each node receives an incident state, adds evidence, and returns a typed update. Conditional edges route low-confidence cases through a refinement loop before the supervisor emits a final diagnosis.

## Agent Responsibilities

- **Detection agent** computes vibration features and classifies health status.
- **Diagnosis supervisor** decides whether enough evidence exists and selects the most likely fault family.
- **Root-cause specialist** maps feature evidence to mechanical causes.
- **Prescription agent** produces structured maintenance actions.
- **Safety validator** blocks unsafe or unsupported actions and records why.

## Production Notes

- Treat generated actions as advisory until connected to a plant-approved control layer.
- Persist each state transition for auditability.
- Keep thresholds machine-specific; a single global threshold is only suitable for demos.
- Benchmark latency separately for signal processing, model inference, retrieval, and orchestration.

