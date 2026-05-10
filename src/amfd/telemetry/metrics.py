from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter


@dataclass
class Timer:
    name: str
    started_at: float = 0.0
    elapsed_seconds: float = 0.0

    def __enter__(self) -> "Timer":
        self.started_at = perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        self.elapsed_seconds = perf_counter() - self.started_at

