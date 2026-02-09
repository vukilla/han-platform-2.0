from __future__ import annotations

from typing import Dict


def log_metrics(metrics: Dict[str, float], run_name: str | None = None) -> None:
    try:
        import mlflow
    except Exception:
        return

    mlflow.start_run(run_name=run_name)
    for key, value in metrics.items():
        mlflow.log_metric(key, float(value))
    mlflow.end_run()
