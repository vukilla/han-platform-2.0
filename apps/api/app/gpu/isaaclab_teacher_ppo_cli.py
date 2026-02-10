from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback


def main() -> None:
    """CLI wrapper to run Isaac Lab teacher PPO as a subprocess.

    This is intended to be executed via Isaac Lab / Isaac Sim's python (e.g. `isaaclab.bat -p -m ...`).
    The parent process (Celery worker) should read `<out_dir>/result.json`.
    """

    parser = argparse.ArgumentParser(description="Train a teacher PPO policy in Isaac Lab and export a checkpoint.")
    parser.add_argument("--out-dir", required=True, help="Directory to write outputs into")
    parser.add_argument("--task", default="cargo_pickup_franka")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=4)
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    result_path = out_dir / "result.json"
    log_path = out_dir / "train.log"

    def _write_result(payload: dict) -> None:
        result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    try:
        from app.gpu.isaaclab_teacher_ppo import IsaacLabTeacherPPOConfig, train_teacher_ppo

        cfg = IsaacLabTeacherPPOConfig(
            task=str(args.task),
            device=str(args.device),
            num_envs=int(args.num_envs),
            seed=int(args.seed),
            rollout_steps=int(args.rollout_steps),
            updates=int(args.updates),
            gamma=float(args.gamma),
            gae_lambda=float(args.gae_lambda),
            clip_range=float(args.clip_range),
            lr=float(args.lr),
            epochs=int(args.epochs),
        )

        def log_cb(msg: str) -> None:
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(msg.rstrip() + "\n")

        metrics = train_teacher_ppo(cfg, out_dir=out_dir, log_cb=log_cb)
        _write_result({"ok": True, "metrics": metrics})
    except Exception as exc:
        _write_result(
            {
                "ok": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()

