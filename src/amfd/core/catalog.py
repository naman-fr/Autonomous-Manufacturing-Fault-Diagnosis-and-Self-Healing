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
        if path is None:
            return cls(_default_templates())

        catalog_path = Path(path)
        if not catalog_path.exists():
            return cls(_default_templates())

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

        return [
            MaintenanceAction(
                action="inspect_bearing",
                priority="medium",
                rationale="Fallback inspection generated from the safety catalog.",
                requires_human_approval=True,
            )
        ]


def _parse_templates(raw: object) -> dict[str, list[ActionTemplate]]:
    if not isinstance(raw, dict):
        return _default_templates()

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

    return templates or _default_templates()


def _normalize_priority(priority: str) -> Priority:
    normalized = priority.strip().lower()
    if normalized not in {"low", "medium", "high", "immediate"}:
        return "medium"
    return cast(Priority, normalized)


def _default_templates() -> dict[str, list[ActionTemplate]]:
    return {
        "normal_operation": [
            ActionTemplate(
                action="inspect_bearing",
                priority="low",
                rationale=(
                    "Continue routine inspection and trend monitoring because the signal is within "
                    "the normal operating envelope."
                ),
            )
        ],
        "bearing_defect": [
            ActionTemplate(
                action="reduce_load",
                priority="high",
                rationale=(
                    "Reduce bearing stress while maintenance prepares an inspection and "
                    "lubrication check."
                ),
            ),
            ActionTemplate(
                action="inspect_bearing",
                priority="high",
                rationale=(
                    "Inspect the bearing race, cage, lubrication, and housing for progression "
                    "or pitting."
                ),
            ),
            ActionTemplate(
                action="schedule_shutdown",
                priority="immediate",
                rationale=(
                    "Critical bearing signatures warrant controlled shutdown approval before "
                    "damage escalates."
                ),
            ),
        ],
        "rotor_imbalance_or_misalignment": [
            ActionTemplate(
                action="rebalance_rotor",
                priority="high",
                rationale=(
                    "Vibration pattern is consistent with imbalance; balance correction "
                    "should be scheduled."
                ),
            ),
            ActionTemplate(
                action="align_coupling",
                priority="medium",
                rationale=(
                    "Coupling and shaft alignment should be verified during the maintenance "
                    "window."
                ),
            ),
        ],
        "rpm_control_instability": [
            ActionTemplate(
                action="recalibrate_rpm",
                priority="medium",
                rationale="RPM drift suggests drive or control-loop calibration issues.",
            )
        ],
        "fallback": [
            ActionTemplate(
                action="inspect_bearing",
                priority="medium",
                rationale=(
                    "Use a generic inspection plan when the model confidence is insufficient for a "
                    "narrower prescription."
                ),
            )
        ],
    }
