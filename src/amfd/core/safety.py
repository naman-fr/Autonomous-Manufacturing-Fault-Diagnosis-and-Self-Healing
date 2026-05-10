from __future__ import annotations

from amfd.core.models import MaintenanceAction, SafetyValidation, Severity


class SafetyPolicy:
    def __init__(self, allowed_actions: tuple[str, ...]) -> None:
        self.allowed_actions = set(allowed_actions)

    def validate(
        self, actions: list[MaintenanceAction], severity: Severity
    ) -> SafetyValidation:
        blocked: list[str] = []
        notes: list[str] = []

        for action in actions:
            if action.action not in self.allowed_actions:
                blocked.append(action.action)
            if severity is Severity.critical and not action.requires_human_approval:
                blocked.append(action.action)
                notes.append("Critical incidents require human approval for all recovery actions.")

        if not actions:
            notes.append("No recovery action was generated.")

        return SafetyValidation(
            approved=not blocked and bool(actions),
            blocked_actions=sorted(set(blocked)),
            notes=notes,
        )

