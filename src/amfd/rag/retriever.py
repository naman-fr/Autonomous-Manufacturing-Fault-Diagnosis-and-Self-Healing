from __future__ import annotations

from amfd.core.models import FeatureVector


class MaintenanceRetriever:
    def __init__(self, notes: list[str] | None = None) -> None:
        self.notes = notes or [
            "High crest factor with mid-frequency peaks often indicates bearing defects.",
            "Dominant 1x running-speed vibration suggests imbalance.",
            "Elevated vibration with RPM drift can indicate coupling misalignment.",
            "Critical bearing faults should trigger load reduction and human-approved shutdown.",
        ]

    def retrieve(self, features: FeatureVector, limit: int = 3) -> list[str]:
        ranked = list(self.notes)
        if features.crest_factor > 3.0:
            ranked.insert(0, "Impulsive vibration pattern is consistent with bearing wear.")
        if features.rpm_mean and features.rpm_mean < 1750:
            ranked.insert(0, "RPM sag can point to load, drive, or control-loop instability.")
        return ranked[:limit]
