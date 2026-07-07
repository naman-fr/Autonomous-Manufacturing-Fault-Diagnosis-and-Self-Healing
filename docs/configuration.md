# Configuration Reference

This project is driven by files so the diagnosis policy can evolve without editing code.

## Runtime Files

| File | Purpose |
| --- | --- |
| `configs/default.yaml` | Default runtime thresholds and file paths |
| `configs/actions.json` | Maintenance action catalog |
| `configs/knowledge_base.json` | Hybrid retrieval documents |
| `configs/prompts.json` | Prompt library for agent roles |
| `configs/training_scenarios.json` | Synthetic training policy for the model bundle |

## Core Settings

| Key | Meaning |
| --- | --- |
| `sampling_rate_hz` | Default sensor sampling rate |
| `window_seconds` | Synthetic and training window size |
| `anomaly_threshold` | Minimum anomaly score for warning state |
| `high_risk_threshold` | Score threshold for critical state |
| `review_probability_threshold` | Confidence threshold that triggers refinement or review |
| `max_refinement_loops` | Upper bound on detector refinement cycles |
| `synthetic_training_cases` | Approximate number of synthetic windows used during bundle training |
| `artifact_dir` | Directory where the persisted model bundle is stored |
| `prompt_path` | Prompt library for optional GenAI reasoning |
| `policy_path` | Action catalog file |
| `knowledge_base_path` | Retrieval corpus file |
| `training_scenarios_path` | Training scenario policy for synthetic model generation |
| `llm_provider` | `auto`, `openai`, `anthropic`, or `gemini` |
| `llm_model` | Explicit provider model name |

## Optional GenAI Secrets

The project runs without any secret. Add these only if you enable the optional LLM reasoning node:

| Secret / Env var | Use |
| --- | --- |
| `OPENAI_API_KEY` | OpenAI chat model auth |
| `ANTHROPIC_API_KEY` | Anthropic chat model auth |
| `GOOGLE_API_KEY` | Google Gemini auth |
| `OPENAI_MODEL` | Default model when `llm_provider=auto` and OpenAI is selected |
| `ANTHROPIC_MODEL` | Default model when Anthropic is selected |
| `GEMINI_MODEL` | Default model when Gemini is selected |

## CSV Input

The CSV loader expects at least one `vibration` column. A `rpm` column is optional.

## Action Policy

`configs/actions.json` maps root causes to structured maintenance actions. The runtime refuses to invent fallback maintenance plans if the catalog is missing or invalid.

## Training Policy

`configs/training_scenarios.json` defines the synthetic classes used to train the persisted detector bundle. This keeps the training recipe visible and editable without changing Python code.
