from __future__ import annotations

from collections.abc import Mapping


class ExperimentTracker:
    def __init__(self, experiment_name: str = "amfd-diagnosis") -> None:
        self.experiment_name = experiment_name

    def log_metrics(self, metrics: Mapping[str, float]) -> None:
        try:
            import mlflow
        except ImportError:  # pragma: no cover - optional production dependency
            return

        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run():
            mlflow.log_metrics(dict(metrics))

