from __future__ import annotations

from typing import Any

prometheus: Any = None
try:
    import prometheus_client as _prometheus

    prometheus = _prometheus
except ImportError:  # pragma: no cover - optional production dependency
    prometheus = None


class PrometheusMetrics:
    def __init__(self) -> None:
        self.incidents: Any = None
        self.latency: Any = None
        self.enabled = prometheus is not None
        if self.enabled:
            self.incidents = prometheus.Counter(
                "amfd_incidents_total",
                "Total diagnosed incidents",
            )
            self.latency = prometheus.Histogram(
                "amfd_latency_ms",
                "Diagnosis latency in milliseconds",
            )

    def observe(self, latency_ms: float) -> None:
        if not self.enabled:
            return
        self.incidents.inc()
        self.latency.observe(latency_ms)
