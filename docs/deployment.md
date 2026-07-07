# Deployment Guide

This project supports local development, container deployment, and CI/CD automation.

## Local Development

```bash
pip install -e ".[dev,backend,app,mlops,genai]"
python app.py
python scripts/local_server.py
uvicorn amfd.backend.api:app --reload
```

## Docker

```bash
docker compose up --build
```

Use this for a fully local container stack.

## Kubernetes

```bash
kubectl apply -f k8s/deployment.yaml
```

The GitHub Actions `cd` workflow can also update the deployment image when triggered manually and when `ENABLE_K8S_DEPLOY=true`.

Required secret for Kubernetes deploy:

- `KUBE_CONFIG`

## GitHub Actions

The `ci.yml` workflow currently runs:

- Ruff format check
- Ruff lint
- Unit tests
- Evaluation smoke test
- Backend service smoke test
- Docker image build
- Manifest validation

The `cd.yml` workflow publishes the container image and optionally deploys to Kubernetes.

## Hugging Face Space

The Space uses `app.py` as its entrypoint.

Secrets are only needed if you enable the optional GenAI reasoning path:

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY`

## Operational Checks

Before shipping a change, run:

```bash
python -m ruff format --check .
python -m ruff check .
python -m pytest -q
python eval.py
```

If you touch the UI, open the app and confirm:

- demo path works
- CSV upload works
- report JSON renders
- action table is populated
- evidence table is populated

