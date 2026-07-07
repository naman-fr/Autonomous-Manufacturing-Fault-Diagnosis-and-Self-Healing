from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from amfd.core.models import MaintenanceAction, Severity

Priority = Literal["low", "medium", "high", "immediate"]


@dataclass(frozen=True)
class ActionTemplate:
    action: str
    priority: Priority
    rationale: str
    requires_human_approval: bool = True


class ActionCatalog:
    def __init__(self, templates: dict[str, list[ActionTemplate]]) -> None:
        self.templates = templates

    @classmethod
    def load(cls, path: str | Path | None = None) -> ActionCatalog:
        catalog_path = Path(path or "configs/actions.json")
        if not catalog_path.exists():
            raise FileNotFoundError(f"Action catalog not found: {catalog_path}")

        raw = json.loads(catalog_path.read_text(encoding="utf-8"))
        return cls(_parse_templates(raw))

    def build(self, root_cause: str, severity: Severity | str) -> list[MaintenanceAction]:
        severity_value = severity.value if isinstance(severity, Severity) else str(severity)
        templates = self.templates.get(root_cause) or self.templates.get("fallback", [])
        actions: list[MaintenanceAction] = []

        for template in templates:
            priority = template.priority
            if severity_value == "critical" and priority in {"low", "medium", "high"}:
                priority = "immediate" if priority != "low" else "high"
            rationale = template.rationale.format(
                root_cause=root_cause,
                severity=severity_value,
            )
            actions.append(
                MaintenanceAction(
                    action=template.action,
                    priority=_normalize_priority(priority),
                    rationale=rationale,
                    requires_human_approval=template.requires_human_approval,
                )
            )

        if actions:
            return actions

        raise ValueError(f"No maintenance actions available for root cause: {root_cause}")


def _parse_templates(raw: object) -> dict[str, list[ActionTemplate]]:
    if not isinstance(raw, dict):
        raise ValueError("Action catalog must be a JSON object.")

    templates: dict[str, list[ActionTemplate]] = {}
    for label, items in raw.items():
        if not isinstance(label, str) or not isinstance(items, list):
            continue
        parsed: list[ActionTemplate] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            action = str(item.get("action", "")).strip()
            priority = str(item.get("priority", "medium")).strip()
            rationale = str(item.get("rationale", "")).strip()
            if not action or not rationale:
                continue
            parsed.append(
                ActionTemplate(
                    action=action,
                    priority=_normalize_priority(priority),
                    rationale=rationale,
                    requires_human_approval=bool(item.get("requires_human_approval", True)),
                )
            )
        if parsed:
            templates[label] = parsed

    if not templates:
        raise ValueError("Action catalog is empty or invalid.")
    return templates


def _normalize_priority(priority: str) -> Priority:
    normalized = priority.strip().lower()
    if normalized not in {"low", "medium", "high", "immediate"}:
        return "medium"
    return cast(Priority, normalized)
