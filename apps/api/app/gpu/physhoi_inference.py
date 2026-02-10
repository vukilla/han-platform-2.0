from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from typing import Any


def _repo_root() -> Path:
    # apps/api/app/gpu/<file>.py -> repo root is parents[4]
    return Path(__file__).resolve().parents[4]


def _resolve_physhoi_root() -> Path:
    env = os.environ.get("PHYSHOI_ROOT")
    if env:
        return Path(env).expanduser().resolve()

    root = _repo_root()
    staged = root / "external" / "humanoid-projects" / "PhysHOI"
    if staged.exists():
        return staged
    return root / "external" / "physhoi"


def run_physhoi_inference(
    *,
    motion_file: Path,
    checkpoint: Path,
    out_dir: Path,
    num_envs: int = 16,
    task: str = "PhysHOI_BallPlay",
    extra_args: list[str] | None = None,
) -> dict[str, Any]:
    """Run PhysHOI inference and return paths to artifacts.

    Notes:
    - PhysHOI is Isaac Gym based and is Linux-first.
    - On Windows, this backend will likely not work. Prefer Isaac Lab for Windows-native GPU runs.
    """
    if os.name == "nt":
        raise RuntimeError(
            "PhysHOI inference is not supported on Windows in this repo (Isaac Gym is Linux-first). "
            "Run this backend on a Linux GPU worker, or use backend=isaaclab_teacher_ppo for Windows."
        )

    physhoi_root = _resolve_physhoi_root()
    if not physhoi_root.exists():
        raise FileNotFoundError(
            "PhysHOI repo not found. Clone into external/physhoi or set PHYSHOI_ROOT.\n"
            f"Expected: {physhoi_root}"
        )

    cfg_env = physhoi_root / "physhoi" / "data" / "cfg" / "physhoi.yaml"
    cfg_train = physhoi_root / "physhoi" / "data" / "cfg" / "train" / "rlg" / "physhoi.yaml"
    if not cfg_env.exists() or not cfg_train.exists():
        raise FileNotFoundError("PhysHOI config files not found. Is the repo layout correct?")

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "physhoi_inference.log"

    cmd = [
        sys.executable,
        str(physhoi_root / "physhoi" / "run.py"),
        "--test",
        "--task",
        str(task),
        "--num_envs",
        str(int(num_envs)),
        "--cfg_env",
        str(cfg_env),
        "--cfg_train",
        str(cfg_train),
        "--motion_file",
        str(motion_file),
        "--checkpoint",
        str(checkpoint),
    ]
    if extra_args:
        cmd.extend(list(extra_args))

    with log_path.open("w", encoding="utf-8") as handle:
        subprocess.run(cmd, cwd=str(physhoi_root), stdout=handle, stderr=handle, check=True)

    return {
        "physhoi_root": str(physhoi_root),
        "log_path": str(log_path),
    }

