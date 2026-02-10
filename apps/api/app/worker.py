import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from datetime import datetime
from time import sleep
from uuid import UUID

import numpy as np
from celery import Celery
from celery.signals import worker_ready, worker_shutdown
from kombu import Exchange, Queue
import redis

from app.core.config import get_settings
from app.core.alerts import send_alert
from app.core.storage import download_file, upload_bytes, upload_file, upload_text
from app import crud
from app.quality import evaluate_demo, evaluate_clip
from app.db import SessionLocal
from app import models

settings = get_settings()

celery_app = Celery(
    "han_worker",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

cpu_exchange = Exchange(settings.celery_cpu_queue, type="direct")
gpu_exchange = Exchange(settings.celery_gpu_queue, type="direct")
pose_exchange = Exchange(settings.celery_pose_queue, type="direct")

celery_app.conf.update(
    task_default_queue=settings.celery_default_queue,
    task_default_exchange=settings.celery_default_queue,
    task_default_exchange_type="direct",
    task_default_routing_key=settings.celery_default_queue,
    task_queues=[
        Queue(settings.celery_cpu_queue, cpu_exchange, routing_key=settings.celery_cpu_queue),
        Queue(settings.celery_gpu_queue, gpu_exchange, routing_key=settings.celery_gpu_queue),
        Queue(settings.celery_pose_queue, pose_exchange, routing_key=settings.celery_pose_queue),
    ],
    worker_prefetch_multiplier=settings.celery_prefetch_multiplier,
    task_acks_late=settings.celery_task_acks_late,
    task_reject_on_worker_lost=settings.celery_reject_on_worker_lost,
)

# Celery remote-control (inspect/ping) can be flaky across Docker<->Windows setups
# (e.g., clock drift, NAT quirks). We publish a lightweight heartbeat key in Redis
# so the API can reliably detect a running GPU worker even if `inspect.ping()` fails.
_heartbeat_stop = threading.Event()
_heartbeat_thread: threading.Thread | None = None


def _detect_worker_role() -> str:
    role = (os.environ.get("HAN_WORKER_ROLE") or "").strip().lower()
    if role:
        return role
    argv = " ".join(sys.argv).lower()
    if " -q " in f" {argv} " or " --queues " in f" {argv} ":
        if "gpu" in argv:
            return "gpu"
    return "cpu"


def _heartbeat_key(worker_name: str, role: str) -> str:
    # role is usually "cpu" or "gpu"
    return f"han:worker_heartbeat:{role}:{worker_name}"


def _start_heartbeat() -> None:
    global _heartbeat_thread  # noqa: PLW0603
    if _heartbeat_thread and _heartbeat_thread.is_alive():
        return

    role = _detect_worker_role()
    worker_name = f"celery@{socket.gethostname()}"
    key = _heartbeat_key(worker_name, role)

    def _run() -> None:
        client = redis.Redis.from_url(settings.redis_url)
        payload = json.dumps({"worker_name": worker_name, "role": role})
        while not _heartbeat_stop.is_set():
            try:
                # Keep a short TTL so stale workers disappear quickly.
                client.setex(key, 30, payload)
            except Exception:
                pass
            time.sleep(5)

    _heartbeat_stop.clear()
    _heartbeat_thread = threading.Thread(target=_run, name=f"han-heartbeat-{worker_name}", daemon=True)
    _heartbeat_thread.start()


def _stop_heartbeat() -> None:
    _heartbeat_stop.set()


@worker_ready.connect
def _on_worker_ready(**kwargs) -> None:  # noqa: ARG001
    _start_heartbeat()


@worker_shutdown.connect
def _on_worker_shutdown(**kwargs) -> None:  # noqa: ARG001
    _stop_heartbeat()


XGEN_STAGES = [
    "INGEST_VIDEO",
    "ESTIMATE_POSE",
    "RETARGET",
    "CONTACT_SYNTH",
    "NONCONTACT_SIM",
    "AUGMENT",
    "EXPORT_DATASET",
    "RENDER_PREVIEWS",
    "QUALITY_SCORE",
]

XMIMIC_STAGES = [
    "BUILD_ENV",
    "TRAIN_TEACHER",
    "DISTILL_STUDENT",
    "EVAL_POLICY",
    "EXPORT_CHECKPOINT",
    "QUALITY_SCORE",
]


def _update_job_status(db, job, status, error=None):
    job.status = status
    if status in {"COMPLETED", "FAILED"}:
        job.finished_at = datetime.utcnow()
    if job.started_at is None:
        job.started_at = datetime.utcnow()
    if error:
        job.error = error
    db.add(job)
    db.commit()
    db.refresh(job)


def _write_job_log(job_type: str, job_id: str, lines: list[str]) -> str:
    key = f"logs/{job_type}/{job_id}.log"
    return upload_text(key, "\n".join(lines))


def _npz_bytes(**arrays: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def _build_synthetic_clip(frames: int, nq: int, contact_dim: int, seed: int) -> tuple[bytes, dict, list[str]]:
    """Create a deterministic clip artifact for end-to-end platform plumbing.

    This keeps the platform usable even when the full GPU sim/training stack is not
    running on the control-plane machine.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, frames, dtype=np.float32)

    freqs = rng.uniform(0.5, 2.0, size=(nq,)).astype(np.float32)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=(nq,)).astype(np.float32)
    robot_qpos = 0.35 * np.sin(2.0 * np.pi * t[:, None] * freqs[None, :] + phases[None, :]).astype(np.float32)
    robot_qvel = np.vstack([np.zeros((1, nq), dtype=np.float32), np.diff(robot_qpos, axis=0)]).astype(np.float32)

    root_pose = np.zeros((frames, 7), dtype=np.float32)
    root_pose[:, 0] = 0.1 * t
    root_pose[:, 6] = 1.0

    object_pose = np.zeros((frames, 7), dtype=np.float32)
    object_pose[:, 0] = 0.35 + 0.02 * np.sin(2.0 * np.pi * t)
    object_pose[:, 1] = 0.05 + 0.02 * np.cos(2.0 * np.pi * t)
    object_pose[:, 2] = 0.30
    object_pose[:, 6] = 1.0

    contact_graph = np.zeros((frames, contact_dim), dtype=np.float32)
    c0 = int(frames * 0.35)
    c1 = int(frames * 0.70)
    contact_graph[c0:c1, :] = 1.0

    phase = np.zeros((frames,), dtype=np.int32)
    phase[:c0] = 0
    phase[c0:c1] = 1
    phase[c1:] = 2

    npz = _npz_bytes(
        robot_qpos=robot_qpos,
        robot_qvel=robot_qvel,
        root_pose=root_pose,
        object_pose=object_pose,
        contact_graph=contact_graph,
        phase=phase,
    )

    metadata = {
        "synthetic": True,
        "frames": frames,
        "nq": nq,
        "contact_dim": contact_dim,
        "contact_range": [c0, c1],
    }
    tags = [f"seed:{seed}", "synthetic:v0"]
    return npz, metadata, tags


def _ensure_dataset_for_xgen_job(db, job: models.XGenJob) -> models.Dataset:
    params = dict(job.params_json or {})
    dataset_id = params.get("dataset_id")
    if dataset_id:
        try:
            existing = crud.get_dataset(db, UUID(str(dataset_id)))
            if existing:
                return existing
        except Exception:
            pass

    demo = job.demo
    dataset = crud.create_dataset(
        db,
        demo.project_id,
        source_demo_id=demo.id,
        status="BUILDING",
        summary_json={"source": "xgen", "job_id": str(job.id)},
    )
    params["dataset_id"] = str(dataset.id)
    job.params_json = params
    db.add(job)
    db.commit()
    db.refresh(dataset)
    return dataset


def _run_isaaclab_teacher_ppo_subprocess(*, job_id: str, params: dict, log_lines: list[str]) -> dict:
    """Run Isaac Lab teacher PPO in a dedicated subprocess.

    Why:
    - Isaac Sim/Kit lifecycle is fragile on Windows. Closing a SimulationApp can terminate the
      entire process. If PPO runs inside the Celery worker process, it can kill the worker and
      leave jobs stuck in Redis as "delivered but unacked".
    - Running PPO in a subprocess isolates Kit shutdown from Celery.
    """
    if os.name != "nt":
        raise RuntimeError("isaaclab_teacher_ppo backend is only supported on Windows GPU workers.")

    repo_root = Path(__file__).resolve().parents[3]
    api_dir = repo_root / "apps" / "api"
    isaaclab_bat = repo_root / "external" / "isaaclab" / "isaaclab.bat"
    if not isaaclab_bat.exists():
        raise FileNotFoundError(
            f"Isaac Lab not found at {isaaclab_bat}. Run scripts\\\\windows\\\\bootstrap_isaaclab.ps1 on the GPU PC."
        )

    tmp_out = Path(tempfile.mkdtemp(prefix=f"isaaclab_ppo_{job_id}_"))
    stdout_path = tmp_out / "isaaclab_ppo.stdout.log"
    stderr_path = tmp_out / "isaaclab_ppo.stderr.log"

    def _get(name: str, default):
        v = params.get(name, default)
        return default if v is None else v

    cmd = [
        "cmd.exe",
        "/c",
        str(isaaclab_bat),
        "-p",
        "-m",
        "app.gpu.isaaclab_teacher_ppo_cli",
        "--out-dir",
        str(tmp_out),
        "--task",
        str(_get("isaaclab_task", "cargo_pickup_franka")),
        "--device",
        str(_get("device", "cuda:0")),
        "--num-envs",
        str(int(_get("num_envs", 64))),
        "--seed",
        str(int(_get("seed", 0))),
        "--rollout-steps",
        str(int(_get("rollout_steps", 128))),
        "--updates",
        str(int(_get("updates", 10))),
        "--gamma",
        str(float(_get("gamma", 0.99))),
        "--gae-lambda",
        str(float(_get("gae_lambda", 0.95))),
        "--clip-range",
        str(float(_get("clip_range", 0.2))),
        "--lr",
        str(float(_get("lr", 3e-4))),
        "--epochs",
        str(int(_get("epochs", 4))),
    ]

    log_lines.append(f"{datetime.utcnow().isoformat()}Z [isaaclab] starting subprocess")
    with stdout_path.open("w", encoding="utf-8") as out_handle, stderr_path.open("w", encoding="utf-8") as err_handle:
        proc = subprocess.run(cmd, cwd=str(api_dir), stdout=out_handle, stderr=err_handle, check=False)

    result_path = tmp_out / "result.json"
    payload = {}
    if result_path.exists():
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}

    # Always try to upload logs to help debugging from the web UI.
    try:
        stdout_key = f"policies/{job_id}/isaaclab_teacher_ppo.stdout.log"
        stderr_key = f"policies/{job_id}/isaaclab_teacher_ppo.stderr.log"
        upload_file(stdout_key, stdout_path, content_type="text/plain")
        upload_file(stderr_key, stderr_path, content_type="text/plain")
    except Exception:
        stdout_key = None
        stderr_key = None

    if proc.returncode != 0 or not payload.get("ok"):
        err = str(payload.get("error") or f"isaaclab teacher PPO failed (exit={proc.returncode})")
        if stdout_key or stderr_key:
            log_lines.append(
                f"{datetime.utcnow().isoformat()}Z [isaaclab] subprocess_failed logs stdout={stdout_key} stderr={stderr_key}"
            )
        raise RuntimeError(err)

    metrics = dict(payload.get("metrics") or {})
    metrics["subprocess_out_dir"] = str(tmp_out)
    if stdout_key:
        metrics["stdout_log_uri"] = stdout_key
    if stderr_key:
        metrics["stderr_log_uri"] = stderr_key
    try:
        train_log = tmp_out / "train.log"
        if train_log.exists():
            train_key = f"policies/{job_id}/isaaclab_teacher_ppo.train.log"
            upload_file(train_key, train_log, content_type="text/plain")
            metrics["train_log_uri"] = train_key
    except Exception:
        pass

    log_lines.append(f"{datetime.utcnow().isoformat()}Z [isaaclab] subprocess_ok")
    return metrics


def _run_gvhmr_pose_estimation(demo_id: str, job_id: str, video_uri: str, params: dict) -> dict:
    """Run GVHMR pose estimation on the local machine and upload artifacts to object storage.

    NOTE: GVHMR is GPU-heavy and requires extra Python deps. This function is intended to run
    on the Windows GPU worker (Isaac Sim Python).
    """
    tmp_root = Path(tempfile.mkdtemp(prefix=f"gvhmr_{demo_id}_{job_id}_"))
    local_video = tmp_root / "input.mp4"
    download_file(video_uri, local_video)

    # Optional: trim long videos to speed up interactive demos.
    # This trades completeness for responsiveness and is recommended for "golden path" UX.
    max_seconds = params.get("gvhmr_max_seconds", None)
    if max_seconds is None:
        max_seconds = params.get("pose_max_seconds", None)
    try:
        max_seconds_val = float(max_seconds) if max_seconds is not None else None
    except Exception:
        max_seconds_val = None
    if max_seconds_val and max_seconds_val > 0:
        try:
            import cv2

            cap = cv2.VideoCapture(str(local_video))
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            if w > 0 and h > 0 and cap.isOpened():
                trimmed_video = tmp_root / "input_trimmed.mp4"
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(trimmed_video), fourcc, fps, (w, h))
                if writer.isOpened():
                    max_frames = max(1, int(round(fps * float(max_seconds_val))))
                    written = 0
                    while written < max_frames:
                        ok_frame, frame = cap.read()
                        if not ok_frame:
                            break
                        writer.write(frame)
                        written += 1
                    writer.release()
                    if trimmed_video.exists() and trimmed_video.stat().st_size > 0 and written > 0:
                        local_video = trimmed_video
                        params["gvhmr_trimmed_seconds"] = float(max_seconds_val)
                        params["gvhmr_trimmed_frames"] = int(written)
            cap.release()
        except Exception:
            # Best-effort: if trimming fails, run GVHMR on the full video.
            pass

    out_dir = tmp_root / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    static_cam = bool(params.get("gvhmr_static_cam", True))
    use_dpvo = bool(params.get("gvhmr_use_dpvo", False))
    f_mm = params.get("gvhmr_f_mm", None)

    cmd = [
        os.environ.get("GVHMR_PYTHON", sys.executable),
        "-m",
        "app.gpu.gvhmr_runner",
        "--video",
        str(local_video),
        "--output-dir",
        str(out_dir),
    ]
    if static_cam:
        cmd.append("--static-cam")
    if use_dpvo:
        cmd.append("--use-dpvo")
    if f_mm is not None:
        cmd.extend(["--f-mm", str(int(f_mm))])

    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    stdout = (result.stdout or "").strip()
    payload: dict = {}
    try:
        payload = json.loads(stdout or "{}")
    except Exception:
        payload = {
            "ok": False,
            "error": "GVHMR runner did not emit valid JSON on stdout.",
            "returncode": int(getattr(result, "returncode", 1) or 1),
            "stdout_head": stdout[:4000],
            "stderr_head": (result.stderr or "").strip()[:4000],
        }

    ok = bool(payload.get("ok", False)) and int(getattr(result, "returncode", 0) or 0) == 0
    npz_path = Path(payload.get("npz_path", "")) if payload.get("npz_path") else None
    meta_path = Path(payload.get("meta_path", "")) if payload.get("meta_path") else None
    gvhmr_log_path = Path(payload.get("gvhmr_log_path", "")) if payload.get("gvhmr_log_path") else None
    preview_path = Path(payload.get("preview_mp4_path", "")) if payload.get("preview_mp4_path") else None

    pose_npz_key = f"demos/{demo_id}/poses/{job_id}/gvhmr_smplx.npz"
    pose_meta_key = f"demos/{demo_id}/poses/{job_id}/gvhmr_meta.json"
    pose_log_key = f"demos/{demo_id}/poses/{job_id}/gvhmr.log"
    pose_preview_key = f"demos/{demo_id}/poses/{job_id}/gvhmr_preview.mp4"

    # Always attempt to upload meta + log for debugging.
    pose_log_uri = None
    if gvhmr_log_path and gvhmr_log_path.exists():
        pose_log_uri = upload_file(pose_log_key, gvhmr_log_path, content_type="text/plain")
    else:
        # If GVHMR fails before producing gvhmr.log, capture stderr/stdout for visibility.
        fallback_log = out_dir / "gvhmr.log"
        try:
            fallback_log.write_text(
                "\n".join(
                    [
                        f"returncode={int(getattr(result, 'returncode', 1) or 1)}",
                        "stdout:",
                        stdout,
                        "",
                        "stderr:",
                        (result.stderr or "").strip(),
                        "",
                        f"payload_error: {payload.get('error')}",
                    ]
                ),
                encoding="utf-8",
            )
            pose_log_uri = upload_file(pose_log_key, fallback_log, content_type="text/plain")
        except Exception:
            pose_log_uri = None

    pose_fallback = None
    pose_error = None
    pose_preview_uri = None
    if ok and npz_path and meta_path and npz_path.exists() and meta_path.exists():
        upload_file(pose_npz_key, npz_path, content_type="application/octet-stream")
        upload_file(pose_meta_key, meta_path, content_type="application/json")
        if preview_path and preview_path.exists():
            pose_preview_uri = upload_file(pose_preview_key, preview_path, content_type="video/mp4")
    else:
        # If real GVHMR pose extraction fails (most commonly due to missing licensed SMPL-X model files),
        # fall back to a placeholder pose so the platform can still complete the end-to-end "golden path".
        # You can force failure by setting `params_json.fail_on_pose_error=true`.
        fail_on_error = bool(params.get("fail_on_pose_error", False))
        pose_error = str(payload.get("error") or f"GVHMR failed (exit={int(getattr(result, 'returncode', 1) or 1)})")
        if fail_on_error:
            raise RuntimeError(pose_error)

        import numpy as np

        pose_fallback = "placeholder"
        t = int(params.get("frames", 1) or 1)
        t = max(1, min(t, 10_000))

        placeholder_npz = out_dir / "placeholder_smplx.npz"
        # Minimal SMPL(-X)-like parameter set. Shapes are typical but not guaranteed to match any downstream model.
        np.savez_compressed(
            placeholder_npz,
            global_orient=np.zeros((t, 3), dtype=np.float32),
            body_pose=np.zeros((t, 63), dtype=np.float32),
            betas=np.zeros((10,), dtype=np.float32),
            transl=np.zeros((t, 3), dtype=np.float32),
        )

        placeholder_meta = out_dir / "placeholder_smplx_meta.json"
        placeholder_meta.write_text(
            json.dumps(
                {
                    "ok": False,
                    "pose_fallback": "placeholder",
                    "error": pose_error,
                    "frames": t,
                    "note": "GVHMR failed; placeholder pose generated so XGen can continue.",
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        upload_file(pose_npz_key, placeholder_npz, content_type="application/octet-stream")
        upload_file(pose_meta_key, placeholder_meta, content_type="application/json")

        # Best-effort placeholder preview: original video on the left, error banner on the right.
        try:
            import cv2
            import numpy as np

            cap = cv2.VideoCapture(str(local_video))
            fps_in = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
            w_in = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
            h_in = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 360)

            target_h = 360
            scale = float(target_h) / float(max(1, h_in))
            target_w = max(1, int(round(w_in * scale)))

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            preview_out = out_dir / "gvhmr_preview.mp4"
            writer = cv2.VideoWriter(str(preview_out), fourcc, fps_in, (target_w * 2, target_h))
            if writer.isOpened():
                frame_count = 0
                while frame_count < 240:  # cap preview to ~8s @30fps
                    ok_frame, frame = cap.read()
                    if not ok_frame:
                        break
                    frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
                    right = np.zeros((target_h, target_w, 3), dtype=np.uint8)
                    right[:] = (14, 12, 10)
                    cv2.putText(
                        right,
                        "GVHMR failed",
                        (16, 64),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (80, 120, 255),
                        3,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        right,
                        "See logs for details",
                        (16, 110),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (235, 235, 235),
                        2,
                        cv2.LINE_AA,
                    )
                    combined = np.concatenate([frame, right], axis=1)
                    writer.write(combined)
                    frame_count += 1
                writer.release()
                pose_preview_uri = upload_file(pose_preview_key, preview_out, content_type="video/mp4")
            cap.release()
        except Exception:
            pose_preview_uri = None

    return {
        "pose_estimator": "gvhmr",
        "pose_smplx_npz_uri": pose_npz_key,
        "pose_meta_uri": pose_meta_key,
        "pose_log_uri": pose_log_uri,
        "pose_preview_mp4_uri": pose_preview_uri,
        "pose_ok": bool(ok),
        "pose_fallback": pose_fallback,
        "pose_error": pose_error,
    }


@celery_app.task(bind=True, max_retries=3, retry_backoff=True)
def run_xgen_job(self, job_id: str):
    db = SessionLocal()
    job = None
    log_lines = [f"XGen job {job_id} started at {datetime.utcnow().isoformat()}Z"]
    try:
        job = db.query(models.XGenJob).filter(models.XGenJob.id == job_id).first()
        if not job:
            return
        dataset = _ensure_dataset_for_xgen_job(db, job)
        params = dict(job.params_json or {})
        only_pose = bool(params.get("only_pose", False))
        stages = ["INGEST_VIDEO", "ESTIMATE_POSE", "RENDER_PREVIEWS"] if only_pose else XGEN_STAGES
        for stage in stages:
            _update_job_status(db, job, stage)
            log_lines.append(f"{datetime.utcnow().isoformat()}Z stage={stage}")
            sleep(0.5)
            if stage == "ESTIMATE_POSE":
                # Default behavior is a placeholder (platform plumbing). Real pose extraction is opt-in.
                placeholder_pose = bool(params.get("placeholder_pose", False))
                estimator = str(params.get("pose_estimator", "none") or "none").lower()
                if placeholder_pose:
                    params["pose_ok"] = True
                    params["pose_estimator"] = "placeholder"
                    log_lines.append(f"{datetime.utcnow().isoformat()}Z pose=placeholder")
                elif estimator == "gvhmr":
                    log_lines.append(f"{datetime.utcnow().isoformat()}Z pose=gvhmr starting")
                    pose_artifacts = _run_gvhmr_pose_estimation(
                        demo_id=str(job.demo_id),
                        job_id=str(job.id),
                        video_uri=job.demo.video_uri,
                        params=params,
                    )
                    params.update(pose_artifacts)
                    if bool(pose_artifacts.get("pose_ok", False)):
                        log_lines.append(
                            f"{datetime.utcnow().isoformat()}Z pose=gvhmr ok npz={pose_artifacts.get('pose_smplx_npz_uri')}"
                        )
                    else:
                        log_lines.append(
                            f"{datetime.utcnow().isoformat()}Z pose=gvhmr failed fallback={pose_artifacts.get('pose_fallback')} "
                            f"err={pose_artifacts.get('pose_error')}"
                        )
                else:
                    params["pose_ok"] = True
                    params["pose_estimator"] = "none"
                    log_lines.append(f"{datetime.utcnow().isoformat()}Z pose=skipped")

                job.params_json = params
                db.add(job)
                db.commit()
                # Persist logs incrementally so the Web UI can show progress while GVHMR is running.
                try:
                    logs_uri = _write_job_log("xgen", job_id, log_lines)
                    job.logs_uri = logs_uri
                    db.add(job)
                    db.commit()
                except Exception:
                    pass
            if stage == "EXPORT_DATASET":
                if only_pose:
                    continue
                existing = crud.list_dataset_clips(db, dataset.id)
                if not existing:
                    frames = int(params.get("frames", 60))
                    nq = int(params.get("nq", 24))
                    contact_dim = int(params.get("contact_dim", 5))
                    clip_count = int(params.get("clip_count", 10))

                    manifest = {
                        "dataset_id": str(dataset.id),
                        "project_id": str(dataset.project_id),
                        "source_demo_id": str(dataset.source_demo_id) if dataset.source_demo_id else None,
                        "created_at": datetime.utcnow().isoformat() + "Z",
                        "clip_count": clip_count,
                        "frames": frames,
                        "nq": nq,
                        "contact_dim": contact_dim,
                    }

                    zip_buf = io.BytesIO()
                    with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                        zf.writestr("manifest.json", json.dumps(manifest, indent=2).encode("utf-8"))
                        for i in range(clip_count):
                            npz_bytes, meta, tags = _build_synthetic_clip(
                                frames=frames,
                                nq=nq,
                                contact_dim=contact_dim,
                                seed=1000 + i,
                            )
                            npz_key = f"datasets/{dataset.id}/clips/{i:04d}.npz"
                            upload_bytes(npz_key, npz_bytes, content_type="application/octet-stream")

                            clip = crud.create_dataset_clip(
                                db,
                                dataset.id,
                                uri_npz=npz_key,
                                # Until clip rendering is implemented, point previews at the original demo video.
                                uri_preview_mp4=job.demo.video_uri,
                                augmentation_tags=tags,
                                stats_json={"status": "Generated", "frames": frames},
                            )

                            zf.writestr(f"clips/{clip.clip_id}/clip.npz", npz_bytes)
                            zf.writestr(
                                f"clips/{clip.clip_id}/metadata.json",
                                json.dumps(meta, indent=2).encode("utf-8"),
                            )

                    dataset_zip_key = f"datasets/{dataset.id}/dataset.zip"
                    upload_bytes(dataset_zip_key, zip_buf.getvalue(), content_type="application/zip")

                    dataset.status = "READY"
                    dataset.summary_json = {
                        **(dataset.summary_json or {}),
                        "clip_count": clip_count,
                        "frames": frames,
                        "nq": nq,
                        "contact_dim": contact_dim,
                        "dataset_zip_key": dataset_zip_key,
                        "name": (dataset.summary_json or {}).get("name", "XGen Dataset"),
                    }
                    db.add(dataset)
                    db.commit()
            if stage == "RENDER_PREVIEWS":
                # When running pose-only mode, preview artifacts are already generated during ESTIMATE_POSE.
                # Keep this stage as a no-op so the UI can show a clean 3-step pipeline.
                pass
        _update_job_status(db, job, "COMPLETED")
        log_lines.append(f"{datetime.utcnow().isoformat()}Z status=COMPLETED")
        try:
            demo = job.demo
            frame_count = int((job.params_json or {}).get("frame_count", 0))
            pose_ok = bool((job.params_json or {}).get("pose_ok", True))
            quality = evaluate_demo(frame_count=frame_count, pose_ok=pose_ok)
            crud.create_quality_score(
                db,
                entity_type="demo",
                entity_id=demo.id,
                score=quality.score,
                breakdown_json=quality.breakdown,
                validator_status="pending",
            )
        except Exception:
            pass
    except Exception as exc:
        if job and self.request.retries < self.max_retries:
            _update_job_status(db, job, "RETRYING", error=str(exc))
            log_lines.append(f"{datetime.utcnow().isoformat()}Z status=RETRYING error={exc}")
            send_alert(
                "xgen_retry",
                {"job_id": job_id, "error": str(exc), "retries": self.request.retries},
            )
            raise self.retry(exc=exc)
        if job:
            _update_job_status(db, job, "FAILED", error=str(exc))
            send_alert("xgen_failed", {"job_id": job_id, "error": str(exc)})
        log_lines.append(f"{datetime.utcnow().isoformat()}Z status=FAILED error={exc}")
    finally:
        try:
            logs_uri = _write_job_log("xgen", job_id, log_lines)
            if job:
                job.logs_uri = logs_uri
                db.add(job)
                db.commit()
        except Exception:
            pass
        db.close()


@celery_app.task(bind=True, max_retries=3, retry_backoff=True)
def run_xmimic_job(self, job_id: str):
    db = SessionLocal()
    job = None
    log_lines = [f"XMimic job {job_id} started at {datetime.utcnow().isoformat()}Z"]
    try:
        job = db.query(models.XMimicJob).filter(models.XMimicJob.id == job_id).first()
        if not job:
            return
        params = dict(job.params_json or {})
        backend = str(params.get("backend", "synthetic") or "synthetic").lower()
        env_task = str(params.get("env_task", "cargo_pickup_v0") or "cargo_pickup_v0")

        # Default is synthetic artifact generation for platform plumbing.
        isaaclab_metrics = None
        physhoi_metrics = None
        for stage in XMIMIC_STAGES:
            _update_job_status(db, job, stage)
            log_lines.append(f"{datetime.utcnow().isoformat()}Z stage={stage}")
            sleep(0.5)
            if backend == "isaaclab_teacher_ppo" and stage == "TRAIN_TEACHER":
                log_lines.append(f"{datetime.utcnow().isoformat()}Z backend=isaaclab_teacher_ppo starting")
                isaaclab_metrics = _run_isaaclab_teacher_ppo_subprocess(
                    job_id=job_id,
                    params=params,
                    log_lines=log_lines,
                )
                params["backend"] = backend
                params["isaaclab_metrics"] = isaaclab_metrics
                job.params_json = params
                db.add(job)
                db.commit()
                log_lines.append(f"{datetime.utcnow().isoformat()}Z backend=isaaclab_teacher_ppo done")
            if backend == "physhoi_inference" and stage == "EVAL_POLICY":
                log_lines.append(f"{datetime.utcnow().isoformat()}Z backend=physhoi_inference starting")
                from app.gpu.physhoi_inference import run_physhoi_inference

                tmp_out = Path(tempfile.mkdtemp(prefix=f"physhoi_{job_id}_"))
                tmp_out.mkdir(parents=True, exist_ok=True)

                motion_local = None
                ckpt_local = None

                motion_path = params.get("physhoi_motion_path")
                ckpt_path = params.get("physhoi_checkpoint_path")
                motion_uri = params.get("physhoi_motion_uri")
                ckpt_uri = params.get("physhoi_checkpoint_uri")

                if motion_path:
                    motion_local = Path(str(motion_path)).expanduser().resolve()
                elif motion_uri:
                    motion_local = tmp_out / "motion.pt"
                    download_file(str(motion_uri), motion_local)

                if ckpt_path:
                    ckpt_local = Path(str(ckpt_path)).expanduser().resolve()
                elif ckpt_uri:
                    ckpt_local = tmp_out / "checkpoint.pth"
                    download_file(str(ckpt_uri), ckpt_local)

                if not motion_local or not motion_local.exists():
                    raise RuntimeError(
                        "PhysHOI backend requires a motion file. Provide params_json.physhoi_motion_path "
                        "(local path) or params_json.physhoi_motion_uri (s3 key/uri)."
                    )
                if not ckpt_local or not ckpt_local.exists():
                    raise RuntimeError(
                        "PhysHOI backend requires a checkpoint. Provide params_json.physhoi_checkpoint_path "
                        "(local path) or params_json.physhoi_checkpoint_uri (s3 key/uri)."
                    )

                physhoi_metrics = run_physhoi_inference(
                    motion_file=motion_local,
                    checkpoint=ckpt_local,
                    out_dir=tmp_out,
                    num_envs=int(params.get("num_envs", 16)),
                    task=str(params.get("physhoi_task", "PhysHOI_BallPlay") or "PhysHOI_BallPlay"),
                    extra_args=list(params.get("physhoi_extra_args", [])) if params.get("physhoi_extra_args") else None,
                )

                params["backend"] = backend
                params["physhoi_metrics"] = physhoi_metrics
                job.params_json = params
                db.add(job)
                db.commit()
                log_lines.append(f"{datetime.utcnow().isoformat()}Z backend=physhoi_inference done")
        _update_job_status(db, job, "COMPLETED")
        log_lines.append(f"{datetime.utcnow().isoformat()}Z status=COMPLETED")
        try:
            if backend == "isaaclab_teacher_ppo" and isaaclab_metrics:
                ckpt_path = Path(isaaclab_metrics["checkpoint_path"])
                ckpt_key = f"policies/{job.id}/checkpoint.pt"
                upload_file(ckpt_key, ckpt_path, content_type="application/octet-stream")

                report = {
                    "backend": backend,
                    "xmimic_job_id": str(job.id),
                    "dataset_id": str(job.dataset_id),
                    "mode": job.mode,
                    "env_task": env_task,
                    "metrics": isaaclab_metrics,
                    "created_at": datetime.utcnow().isoformat() + "Z",
                }
                report_key = f"policies/{job.id}/eval_report.json"
                upload_bytes(report_key, json.dumps(report, indent=2).encode("utf-8"), content_type="application/json")

                policy = crud.create_policy(
                    db,
                    job.id,
                    checkpoint_uri=ckpt_key,
                    metadata_json={"mode": job.mode, "backend": backend, "metrics": isaaclab_metrics},
                )
                crud.create_eval_run(
                    db,
                    policy.id,
                    env_task=env_task,
                    sr=None,
                    gsr=None,
                    eo=None,
                    eh=None,
                    report_uri=report_key,
                    videos_uri=None,
                )
            elif backend == "physhoi_inference" and physhoi_metrics:
                # Best-effort: upload the inference log and the referenced checkpoint file (if local).
                log_path = Path(str(physhoi_metrics.get("log_path", "")))
                if log_path.exists():
                    log_key = f"policies/{job.id}/physhoi_inference.log"
                    upload_file(log_key, log_path, content_type="text/plain")

                # If the caller used a local checkpoint path, mirror it into storage for reproducibility.
                ckpt_local_path = params.get("physhoi_checkpoint_path")
                if ckpt_local_path:
                    try:
                        ckpt_p = Path(str(ckpt_local_path)).expanduser().resolve()
                        if ckpt_p.exists():
                            ckpt_key = f"policies/{job.id}/checkpoint.pth"
                            upload_file(ckpt_key, ckpt_p, content_type="application/octet-stream")
                        else:
                            ckpt_key = None
                    except Exception:
                        ckpt_key = None
                else:
                    # If it came from storage already, re-use that pointer.
                    ckpt_key = str(params.get("physhoi_checkpoint_uri") or "")

                report = {
                    "backend": backend,
                    "xmimic_job_id": str(job.id),
                    "dataset_id": str(job.dataset_id),
                    "mode": job.mode,
                    "env_task": env_task,
                    "metrics": physhoi_metrics,
                    "created_at": datetime.utcnow().isoformat() + "Z",
                }
                report_key = f"policies/{job.id}/eval_report.json"
                upload_bytes(report_key, json.dumps(report, indent=2).encode("utf-8"), content_type="application/json")

                policy = crud.create_policy(
                    db,
                    job.id,
                    checkpoint_uri=ckpt_key or None,
                    metadata_json={"mode": job.mode, "backend": backend, "metrics": physhoi_metrics},
                )
                crud.create_eval_run(
                    db,
                    policy.id,
                    env_task=env_task,
                    sr=None,
                    gsr=None,
                    eo=None,
                    eh=None,
                    report_uri=report_key,
                    videos_uri=None,
                )
            else:
                checkpoint = {
                    "synthetic": True,
                    "xmimic_job_id": str(job.id),
                    "dataset_id": str(job.dataset_id),
                    "mode": job.mode,
                    "created_at": datetime.utcnow().isoformat() + "Z",
                }
                ckpt_key = f"policies/{job.id}/checkpoint.json"
                upload_bytes(
                    ckpt_key,
                    json.dumps(checkpoint, indent=2).encode("utf-8"),
                    content_type="application/json",
                )
                policy = crud.create_policy(
                    db,
                    job.id,
                    checkpoint_uri=ckpt_key,
                    metadata_json={"mode": job.mode, "synthetic": True},
                )
                crud.create_eval_run(
                    db,
                    policy.id,
                    env_task="cargo_pickup_v0",
                    sr=0.0,
                    gsr=0.0,
                    eo=None,
                    eh=None,
                    report_uri=None,
                    videos_uri=None,
                )
        except Exception:
            pass
        try:
            dataset = job.dataset
            quality = evaluate_clip(has_contact_graph=True, joint_limits_ok=True, object_spikes_ok=True)
            crud.create_quality_score(
                db,
                entity_type="dataset",
                entity_id=dataset.id,
                score=quality.score,
                breakdown_json=quality.breakdown,
                validator_status="pending",
            )
        except Exception:
            pass
    except Exception as exc:
        if job and self.request.retries < self.max_retries:
            _update_job_status(db, job, "RETRYING", error=str(exc))
            log_lines.append(f"{datetime.utcnow().isoformat()}Z status=RETRYING error={exc}")
            send_alert(
                "xmimic_retry",
                {"job_id": job_id, "error": str(exc), "retries": self.request.retries},
            )
            raise self.retry(exc=exc)
        if job:
            _update_job_status(db, job, "FAILED", error=str(exc))
            send_alert("xmimic_failed", {"job_id": job_id, "error": str(exc)})
        log_lines.append(f"{datetime.utcnow().isoformat()}Z status=FAILED error={exc}")
    finally:
        try:
            logs_uri = _write_job_log("xmimic", job_id, log_lines)
            if job:
                job.logs_uri = logs_uri
                db.add(job)
                db.commit()
        except Exception:
            pass
        db.close()
