from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> dict:
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return {"ok": True, "stdout": completed.stdout.strip(), "stderr": completed.stderr.strip()}
    except subprocess.CalledProcessError as exc:
        return {"ok": False, "stdout": exc.stdout.strip() if exc.stdout else "", "stderr": exc.stderr.strip() if exc.stderr else ""}
    except FileNotFoundError as exc:
        return {"ok": False, "stdout": "", "stderr": str(exc)}


def main() -> None:
    report = {
        "python": sys.version,
        "platform": sys.platform,
        "nvidia_smi": _run(["nvidia-smi"]),
        "cuda_version": _run(["nvcc", "--version"]),
        "torch": {},
        "isaac_gym": {},
    }

    try:
        import torch  # type: ignore

        report["torch"] = {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception as exc:  # pragma: no cover
        report["torch"] = {"error": str(exc)}

    try:
        import isaacgym  # type: ignore

        report["isaac_gym"] = {"import_ok": True, "version": getattr(isaacgym, "__version__", "unknown")}
    except Exception as exc:
        report["isaac_gym"] = {"import_ok": False, "error": str(exc)}

    output = json.dumps(report, indent=2)
    print(output)
    Path("gpu_preflight_report.json").write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
