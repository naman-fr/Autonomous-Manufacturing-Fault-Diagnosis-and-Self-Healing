from __future__ import annotations

try:
    from prometheus_client import Counter, Histogram
except ImportError:  # pragma: no cover - optional production dependency
    Counter = Histogram = None  # type: ignore[assignment]


class PrometheusMetrics:
    def __init__(self) -> None:
        self.enabled = Counter is not None and Histogram is not None
        if self.enabled:
            self.incidents = Counter("amfd_incidents_total", "Total diagnosed incidents")
            self.latency = Histogram("amfd_latency_ms", "Diagnosis latency in milliseconds")

    def observe(self, latency_ms: float) -> None:
        if not self.enabled:
            return
        self.incidents.inc()
        self.latency.observe(latency_ms)

