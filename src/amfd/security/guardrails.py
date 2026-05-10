from __future__ import annotations

import re

from amfd.core.models import GuardrailFinding

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?:\+?\d{1,3}[-.\s]?)?(?:\d{10}|\d{3}[-.\s]\d{3}[-.\s]\d{4})")


class GuardrailEngine:
    """Local guardrail layer; can be swapped for NeMo Guardrails in production."""

    def redact(self, text: str) -> tuple[str, list[GuardrailFinding]]:
        findings: list[GuardrailFinding] = []
        redacted = text

        if EMAIL_RE.search(redacted):
            redacted = EMAIL_RE.sub("[REDACTED_EMAIL]", redacted)
            findings.append(
                GuardrailFinding(
                    category="pii.email",
                    action="redact",
                    detail="Email address redacted before agent processing.",
                )
            )
        if PHONE_RE.search(redacted):
            redacted = PHONE_RE.sub("[REDACTED_PHONE]", redacted)
            findings.append(
                GuardrailFinding(
                    category="pii.phone",
                    action="redact",
                    detail="Phone number redacted before agent processing.",
                )
            )

        return redacted, findings

    def validate_prompt(self, text: str) -> list[GuardrailFinding]:
        lowered = text.lower()
        if "ignore previous instructions" in lowered or "bypass safety" in lowered:
            return [
                GuardrailFinding(
                    category="prompt_injection",
                    action="block",
                    detail="Prompt injection phrase detected.",
                )
            ]
        return []
