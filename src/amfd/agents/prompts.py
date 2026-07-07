from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PromptLibrary:
    prompts: dict[str, str]

    @classmethod
    def load(cls, path: str | Path) -> PromptLibrary:
        prompt_path = Path(path)
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt library not found: {prompt_path}")

        raw = json.loads(prompt_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Prompt library must be a JSON object: {prompt_path}")

        prompts = {
            str(role): str(text).strip()
            for role, text in raw.items()
            if isinstance(role, str) and isinstance(text, str) and text.strip()
        }
        if not prompts:
            raise ValueError(f"Prompt library is empty: {prompt_path}")
        return cls(prompts=prompts)

    def get(self, role: str) -> str:
        return self.prompts.get(role, "")
