# Research Notes

This project adapts agentic self-healing manufacturing concepts to rotating machinery. The useful pattern is not "LLM controls factory"; it is a supervised, tool-using workflow where deterministic signal tools and safety policies bound the agent's decisions.

## Adapted Ideas

- Supervisor agent coordinates specialist tools.
- Causal reasoning is represented as evidence-backed root-cause hypotheses.
- Self-healing is implemented as validated maintenance prescriptions.
- Continual learning hooks are present through synthetic augmentation and future MLflow experiments.

## Dataset Path

The default demo uses generated vibration samples. The intended benchmark extensions are CWRU and PU:

1. Convert bearing recordings to fixed-length windows.
2. Extract time-domain and FFT-domain features.
3. Train baseline detectors and compare against augmented training sets.
4. Report F1, false-alarm rate, latency, and action-plan acceptance rate.

## Sources

- JISEM 2024: Agentic AI for Self-Healing Production Lines, DOI 10.52783/jisem.v9i4s.12427.
- PHM Society 2025: Exploring LLM-based Agentic Frameworks for Fault Diagnosis, DOI 10.36001/phmconf.2025.v17i1.4350.
- CWRU Bearing Data Center: seeded bearing faults with documented loads, speeds, and fault locations.
- Paderborn Bearing Data Center: vibration/current data for healthy, artificially damaged, and naturally damaged bearings.
