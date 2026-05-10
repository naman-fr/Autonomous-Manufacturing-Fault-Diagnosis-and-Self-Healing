# Research Notes

This project adapts agentic self-healing manufacturing concepts to rotating machinery. The useful pattern is not "LLM controls factory"; it is a supervised, tool-using workflow where deterministic signal tools and safety policies bound the agent's decisions.

## Adapted Ideas

- Supervisor agent coordinates specialist tools.
- Causal reasoning is represented as evidence-backed root-cause hypotheses.
- Self-healing is implemented as validated maintenance prescriptions.
- Continual learning hooks are present through synthetic augmentation and future MLflow experiments.

## Dataset Path

The default demo uses generated vibration samples. The intended benchmark extension is the CWRU bearing dataset:

1. Convert bearing recordings to fixed-length windows.
2. Extract time-domain and FFT-domain features.
3. Train baseline detectors and compare against augmented training sets.
4. Report F1, false-alarm rate, latency, and action-plan acceptance rate.

