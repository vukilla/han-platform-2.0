from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def build_eval_report(metrics: Dict[str, float], slices: Dict[str, float] | None = None) -> Dict[str, Any]:
    report = {"metrics": metrics}
    if slices:
        report["slices"] = slices
    return report


def save_eval_report(path: str | Path, report: Dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return output
