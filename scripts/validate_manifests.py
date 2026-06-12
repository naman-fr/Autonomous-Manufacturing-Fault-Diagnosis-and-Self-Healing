from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def main() -> None:
    deployment_path = Path("k8s/deployment.yaml")
    documents = list(yaml.safe_load_all(deployment_path.read_text(encoding="utf-8")))
    if len(documents) != 2:
        raise SystemExit("k8s/deployment.yaml must contain exactly two documents.")

    deployment, service = documents
    _assert_kind(deployment, "Deployment")
    _assert_kind(service, "Service")

    spec = deployment["spec"]
    template = spec["template"]
    container = template["spec"]["containers"][0]

    assert deployment["metadata"]["name"] == "amfd-api"
    assert spec["replicas"] == 2
    assert container["name"] == "api"
    assert container["ports"][0]["containerPort"] == 8000
    assert container["readinessProbe"]["httpGet"]["path"] == "/health"
    assert container["livenessProbe"]["httpGet"]["path"] == "/health"
    assert container["securityContext"]["allowPrivilegeEscalation"] is False
    assert service["spec"]["ports"][0]["targetPort"] == 8000

    print("k8s/deployment.yaml validated")


def _assert_kind(document: dict[str, Any], expected: str) -> None:
    kind = document.get("kind")
    if kind != expected:
        raise SystemExit(f"Expected {expected}, got {kind!r}.")


if __name__ == "__main__":
    main()
