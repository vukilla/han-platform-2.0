"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useParams, useSearchParams } from "next/navigation";
import Link from "next/link";
import { Badge } from "@/components/ui/Badge";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import {
  getDemo,
  getGvhmrSmplxModelStatus,
  getXgenJob,
  login,
  requeueXgenJob,
  runXmimic,
  uploadGvhmrSmplxModel,
  DemoOut,
  GVHMRSmplxModelStatus,
  XGenJobOut,
} from "@/lib/api";
import { getToken, setToken } from "@/lib/auth";

const DEFAULT_ISAACLAB_NUM_ENVS = 8;
const DEFAULT_ISAACLAB_UPDATES = 2;
const DEFAULT_ISAACLAB_ROLLOUT_STEPS = 64;

const fullStages = [
  "INGEST_VIDEO",
  "ESTIMATE_POSE",
  "RETARGET",
  "CONTACT_SYNTH",
  "NONCONTACT_SIM",
  "AUGMENT",
  "EXPORT_DATASET",
  "RENDER_PREVIEWS",
  "QUALITY_SCORE",
];

export default function JobProgressPage() {
  const params = useParams<{ id: string }>();
  const searchParams = useSearchParams();
  const jobId = params?.id;
  const [job, setJob] = useState<XGenJobOut | null>(null);
  const [demo, setDemo] = useState<DemoOut | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [poseReady, setPoseReady] = useState<boolean | null>(null);
  const [gpuReady, setGpuReady] = useState<boolean | null>(null);
  const [jobLogTail, setJobLogTail] = useState<string | null>(null);
  const [smplxStatus, setSmplxStatus] = useState<GVHMRSmplxModelStatus | null>(null);
  const [smplxFile, setSmplxFile] = useState<File | null>(null);
  const [smplxUploadStatus, setSmplxUploadStatus] = useState<string | null>(null);
  const requestedTrainMode: "nep" | "mocap" = searchParams?.get("train_mode") === "mocap" ? "mocap" : "nep";
  const requestedTrainBackend: "synthetic" | "isaaclab_teacher_ppo" =
    searchParams?.get("train_backend") === "isaaclab_teacher_ppo" ? "isaaclab_teacher_ppo" : "synthetic";
  const autoTrain = searchParams?.get("auto_train") === "1";

  const [trainMode, setTrainMode] = useState<"nep" | "mocap">(requestedTrainMode);
  const [trainBackend, setTrainBackend] = useState<"synthetic" | "isaaclab_teacher_ppo">(requestedTrainBackend);
  const [trainStatus, setTrainStatus] = useState<string | null>(null);
  const [trainError, setTrainError] = useState<string | null>(null);
  const [xmimicJobId, setXmimicJobId] = useState<string | null>(null);
  const autoTrainTriggered = useRef(false);
  const [requeueStatus, setRequeueStatus] = useState<string | null>(null);

  const onlyPose = Boolean(job?.params_json?.only_pose);

  async function ensureLoggedIn() {
    if (getToken()) return;
    const resp = await login("demo@humanx.local", "Demo");
    setToken(resp.token);
  }

  useEffect(() => {
    let timer: NodeJS.Timeout | null = null;

    async function fetchStatus() {
      if (!jobId) {
        return;
      }
      try {
        const response = await getXgenJob(jobId);
        setJob(response);
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load job");
      }
    }

    fetchStatus();
    timer = setInterval(fetchStatus, 3000);

    return () => {
      if (timer) {
        clearInterval(timer);
      }
    };
  }, [jobId]);

  useEffect(() => {
    const logsUri = job?.logs_uri;
    if (!logsUri) return;
    let timer: NodeJS.Timeout | null = null;
    let cancelled = false;

    async function fetchLogs() {
      try {
        const res = await fetch(logsUri as string);
        if (!res.ok) return;
        const text = await res.text();
        const tail = text.split("\n").slice(-200).join("\n");
        if (!cancelled) setJobLogTail(tail);
      } catch {
        // ignore
      }
    }

    fetchLogs();
    timer = setInterval(fetchLogs, 3000);
    return () => {
      cancelled = true;
      if (timer) clearInterval(timer);
    };
  }, [job?.logs_uri]);

  useEffect(() => {
    if (!job?.demo_id) return;
    let cancelled = false;
    void (async () => {
      try {
        const fetched = await getDemo(job.demo_id);
        if (!cancelled) setDemo(fetched);
      } catch {
        if (!cancelled) setDemo(null);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [job?.demo_id]);

  useEffect(() => {
    if (!onlyPose) return;
    let cancelled = false;
    void (async () => {
      try {
        await ensureLoggedIn();
        const status = await getGvhmrSmplxModelStatus();
        if (!cancelled) setSmplxStatus(status);
      } catch {
        if (!cancelled) setSmplxStatus(null);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [onlyPose]);

  async function handleUploadSmplx() {
    setSmplxUploadStatus(null);
    setError(null);
    if (!smplxFile) {
      setError("Select `SMPLX_NEUTRAL.npz` first.");
      return;
    }
    try {
      await ensureLoggedIn();
      setSmplxUploadStatus("Uploading SMPL-X model...");
      const status = await uploadGvhmrSmplxModel(smplxFile);
      setSmplxStatus(status);
      setSmplxUploadStatus("Uploaded. You can rerun the job now.");
    } catch (err) {
      setSmplxUploadStatus(null);
      setError(err instanceof Error ? err.message : "Failed to upload SMPL-X model");
    }
  }

  const status = job?.status ?? null;
  const jobIsComplete = status === "COMPLETED";
  const jobIsFailed = status === "FAILED";
  const jobIsQueued = status === "QUEUED";
  const datasetId = typeof job?.params_json?.dataset_id === "string" ? job?.params_json?.dataset_id : null;
  const stages = onlyPose ? (["INGEST_VIDEO", "ESTIMATE_POSE", "RENDER_PREVIEWS"] as const) : fullStages;
  const stageNames: Record<string, string> = onlyPose
    ? {
        INGEST_VIDEO: "Upload video",
        ESTIMATE_POSE: "Run GVHMR",
        RENDER_PREVIEWS: "Render preview",
      }
    : {};

  useEffect(() => {
    let timer: NodeJS.Timeout | null = null;
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    const poll = () => {
      fetch(`${apiUrl}/ops/workers?timeout=1.0`)
        .then((res) => (res.ok ? res.json() : null))
        .then((data) => {
          setPoseReady(Boolean(data?.has_pose_queue));
          setGpuReady(Boolean(data?.has_gpu_queue));
        })
        .catch(() => {
          setPoseReady(null);
          setGpuReady(null);
        });
    };
    poll();
    timer = setInterval(poll, 3000);
    return () => {
      if (timer) clearInterval(timer);
    };
  }, []);

  const startTraining = useCallback(async (mode: "nep" | "mocap", backend: "synthetic" | "isaaclab_teacher_ppo") => {
    if (!datasetId || !jobIsComplete) return;
    setTrainError(null);
    setXmimicJobId(null);
    setTrainStatus("Starting XMimic job...");
    try {
      const params: Record<string, unknown> = {};
      if (backend === "isaaclab_teacher_ppo") {
        params.backend = "isaaclab_teacher_ppo";
        params.env_task = "cargo_pickup_v0";
        params.isaaclab_task = "cargo_pickup_franka";
        // Default to a quick run that completes in a few minutes on a single GPU.
        params.num_envs = DEFAULT_ISAACLAB_NUM_ENVS;
        params.updates = DEFAULT_ISAACLAB_UPDATES;
        params.rollout_steps = DEFAULT_ISAACLAB_ROLLOUT_STEPS;
      }
      const xmimic = await runXmimic(datasetId, mode, params);
      setXmimicJobId(xmimic.id);
      setTrainStatus("XMimic job started.");
    } catch (err) {
      setTrainError(err instanceof Error ? err.message : "Failed to start training");
      setTrainStatus(null);
    }
  }, [datasetId, jobIsComplete]);

  async function handleStartTraining() {
    await startTraining(trainMode, trainBackend);
  }

  useEffect(() => {
    if (!autoTrain) return;
    if (!jobIsComplete || !datasetId) return;
    if (xmimicJobId) return;
    if (autoTrainTriggered.current) return;

    autoTrainTriggered.current = true;
    // Defer the state updates out of the effect body to avoid cascading render warnings.
    setTimeout(() => {
      void startTraining(requestedTrainMode, requestedTrainBackend);
    }, 0);
  }, [autoTrain, datasetId, jobIsComplete, requestedTrainBackend, requestedTrainMode, startTraining, xmimicJobId]);

  const poseOk = job?.params_json && typeof job.params_json.pose_ok === "boolean" ? Boolean(job.params_json.pose_ok) : null;
  const poseFallback =
    job?.params_json && typeof job.params_json.pose_fallback === "string" ? (job.params_json.pose_fallback as string) : null;
  const poseError =
    job?.params_json && typeof job.params_json.pose_error === "string" ? (job.params_json.pose_error as string) : null;
  const poseNpz =
    job?.params_json && typeof job.params_json.pose_smplx_npz_uri === "string"
      ? (job.params_json.pose_smplx_npz_uri as string)
      : null;
  const poseLog =
    job?.params_json && typeof job.params_json.pose_log_uri === "string" ? (job.params_json.pose_log_uri as string) : null;
  const posePreview =
    job?.params_json && typeof job.params_json.pose_preview_mp4_uri === "string"
      ? (job.params_json.pose_preview_mp4_uri as string)
      : null;
  const demoVideo = demo?.video_uri ?? null;
  const previewMessage = jobIsComplete
    ? poseOk === false
      ? "GVHMR fell back to a placeholder pose (no preview video). See the pose log below."
      : "Pose preview not available yet. See the pose log below if this persists."
    : jobIsFailed
      ? "Pose preview failed. See the worker error/logs."
      : status === "ESTIMATE_POSE"
        ? "GVHMR running..."
        : status === "RENDER_PREVIEWS"
          ? "Rendering preview..."
          : "Waiting for pose preview...";

  const currentIndex = status ? (stages as readonly string[]).indexOf(status) : -1;
  const stageLabel = (index: number) => {
    if (jobIsComplete) return "Completed";
    if (jobIsFailed) return "Failed";
    if (currentIndex === -1) return "Queued";
    if (index < currentIndex) return "Completed";
    if (index === currentIndex) return "Running";
    return "Queued";
  };

  const waitingForWorker = !jobIsComplete && !jobIsFailed && currentIndex === -1;

  async function handleRequeue() {
    if (!jobId) return;
    setRequeueStatus("Requeuing...");
    try {
      await requeueXgenJob(jobId);
      setRequeueStatus("Requeued. This page will update automatically.");
    } catch (err) {
      setRequeueStatus(null);
      setError(err instanceof Error ? err.message : "Failed to requeue job");
    }
  }

  return (
    <div className="space-y-8">
      <section className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <p className="section-eyebrow">Job progress</p>
          <h1 className="text-3xl font-semibold text-black">{onlyPose ? "GVHMR pose pipeline" : "XGen pipeline stages"}</h1>
          {jobId ? <p className="mt-2 text-sm text-black/60">{jobId}</p> : null}
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <Badge label={status || "Queued"} tone={jobIsComplete ? "emerald" : jobIsFailed ? "rose" : "amber"} />
          {datasetId && !onlyPose ? (
            <Link href={`/datasets/${datasetId}`} className="text-sm font-semibold text-black underline">
              Open dataset
            </Link>
          ) : null}
          {job?.logs_uri ? (
            <a className="text-sm font-semibold text-black underline" href={job.logs_uri} target="_blank" rel="noreferrer">
              Download logs
            </a>
          ) : null}
        </div>
      </section>

      {onlyPose && waitingForWorker && poseReady === false ? (
        <Card className="space-y-2">
          <h2 className="text-lg font-semibold text-black">Waiting for Windows pose worker</h2>
          <p className="text-sm text-black/70">
            This job requires the Windows worker to be running (queue: pose). Start it on your Windows PC, then this page will update automatically.
          </p>
          <p className="text-sm font-mono text-black/70">
            scripts\\windows\\one_click_gpu_worker.ps1 -MacIp &lt;MAC_LAN_IP&gt; -IsaacSimPath C:\\isaacsim -Queues pose
          </p>
        </Card>
      ) : null}

      {onlyPose && jobIsQueued && !job?.started_at && gpuReady !== false ? (
        <Card className="space-y-2">
          <h2 className="text-lg font-semibold text-black">Job is queued</h2>
          <p className="text-sm text-black/70">
            If you restarted Docker/Redis on your Mac, queued jobs can be lost. Requeue to start processing now.
          </p>
          <div className="flex flex-wrap items-center gap-3">
            <Button onClick={handleRequeue}>Requeue job</Button>
            {requeueStatus ? <span className="text-sm text-black/70">{requeueStatus}</span> : null}
          </div>
        </Card>
      ) : null}

      {onlyPose && jobIsComplete && poseOk === false ? (
        <Card className="space-y-2">
          <h2 className="text-lg font-semibold text-black">GVHMR ran with a fallback</h2>
          <p className="text-sm text-black/70">
            The most common cause is a missing licensed SMPL-X model file (<span className="font-mono">SMPLX_NEUTRAL.npz</span>).
            Upload it below (or in <span className="font-mono">/gvhmr</span>), then rerun this job.
          </p>
          {smplxStatus?.exists === false ? (
            <div className="space-y-2 rounded-2xl border border-black/10 bg-black/[0.02] p-4">
              <p className="text-sm font-semibold text-black">One-time setup: Upload SMPL-X model</p>
              <input
                type="file"
                accept=".npz"
                onChange={(event) => setSmplxFile(event.target.files?.[0] ?? null)}
                className="w-full rounded-2xl border border-black/15 bg-white px-4 py-3 text-sm"
              />
              <div className="flex flex-wrap items-center gap-3">
                <Button onClick={handleUploadSmplx} disabled={!smplxFile}>
                  Upload SMPL-X model
                </Button>
                {smplxUploadStatus ? <span className="text-sm text-black/70">{smplxUploadStatus}</span> : null}
              </div>
              <p className="text-xs text-black/60">Download instructions: `docs/GVHMR.md`</p>
            </div>
          ) : null}
          <div className="flex flex-wrap items-center gap-3">
            <Button onClick={handleRequeue}>Rerun GVHMR</Button>
            {requeueStatus ? <span className="text-sm text-black/70">{requeueStatus}</span> : null}
          </div>
        </Card>
      ) : null}

      <Card className="space-y-6">
        {stages.map((stage, index) => (
          <div key={stage} className="flex items-center justify-between border-b border-black/10 pb-4 last:border-none last:pb-0">
            <div>
              <p className="text-sm font-semibold text-black/60">Stage {index + 1}</p>
              <p className="text-lg font-semibold text-black">{stageNames[stage] ?? stage}</p>
            </div>
            <span className="text-sm text-black/70">{stageLabel(index)}</span>
          </div>
        ))}
      </Card>

      {demoVideo || posePreview ? (
        <Card className="space-y-4">
          <h2 className="text-xl font-semibold text-black">Video preview</h2>
          <div className="grid gap-6 lg:grid-cols-2">
            <div className="space-y-2">
              <p className="text-sm font-semibold text-black/60">Original</p>
              {demoVideo ? (
                <video className="w-full rounded-2xl border border-black/10 bg-black" controls playsInline src={demoVideo} />
              ) : (
                <p className="text-sm text-black/60">Waiting for upload...</p>
              )}
            </div>
            <div className="space-y-2">
              <p className="text-sm font-semibold text-black/60">GVHMR preview</p>
              {posePreview ? (
                <video className="w-full rounded-2xl border border-black/10 bg-black" controls playsInline src={posePreview} />
              ) : demoVideo ? (
                <div className="relative">
                  <video className="w-full rounded-2xl border border-black/10 bg-black" controls playsInline src={demoVideo} />
                  <div className="absolute inset-0 flex items-center justify-center rounded-2xl bg-black/40 p-4">
                    <p className="text-sm font-semibold text-white">{previewMessage}</p>
                  </div>
                </div>
              ) : (
                <div className="flex aspect-video items-center justify-center rounded-2xl border border-black/10 bg-black/[0.04] p-4">
                  <p className="text-sm text-black/60">{previewMessage}</p>
                </div>
              )}
            </div>
          </div>
        </Card>
      ) : null}

      {poseOk !== null || poseFallback || poseError || poseNpz || poseLog ? (
        <Card className="space-y-3">
          <h2 className="text-xl font-semibold text-black">Pose extraction</h2>
          <div className="flex flex-wrap items-center gap-3 text-sm">
            <Badge
              label={poseOk === true ? "GVHMR OK" : poseOk === false ? "GVHMR fallback" : "Unknown"}
              tone={poseOk === true ? "emerald" : poseOk === false ? "rose" : "amber"}
            />
            {poseFallback ? <span className="text-black/70">fallback: {poseFallback}</span> : null}
          </div>
          {poseError ? <p className="text-sm text-rose-700">{poseError}</p> : null}
          <div className="flex flex-wrap items-center gap-4">
            {poseNpz ? (
              <a className="text-sm font-semibold text-black underline" href={poseNpz} target="_blank" rel="noreferrer">
                Download pose artifact
              </a>
            ) : null}
            {poseLog ? (
              <a className="text-sm font-semibold text-black underline" href={poseLog} target="_blank" rel="noreferrer">
                Download pose log
              </a>
            ) : null}
            <span className="text-sm text-black/60">GVHMR setup: see `docs/GVHMR.md` in the repo</span>
          </div>
        </Card>
      ) : null}

      {jobLogTail ? (
        <Card className="space-y-3">
          <h2 className="text-xl font-semibold text-black">Live logs</h2>
          <p className="text-sm text-black/70">These update every few seconds while the job runs.</p>
          <pre className="max-h-80 overflow-auto rounded-2xl border border-black/10 bg-black p-4 text-xs text-white">
            {jobLogTail}
          </pre>
        </Card>
      ) : null}

      {datasetId && jobIsComplete && !onlyPose ? (
        <Card className="space-y-4">
          <h2 className="text-xl font-semibold text-black">Next: Train a policy</h2>
          <div className="grid gap-3 md:grid-cols-2">
            <label className="grid gap-2 text-sm font-semibold text-black">
              Mode
              <select
                className="w-full rounded-2xl border border-black/15 bg-white px-4 py-3 text-sm"
                value={trainMode}
                onChange={(event) => setTrainMode(event.target.value as "nep" | "mocap")}
              >
                <option value="nep">NEP (proprio only)</option>
                <option value="mocap">MoCap (object pose + dropout)</option>
              </select>
            </label>
            <label className="grid gap-2 text-sm font-semibold text-black">
              Backend
              <select
                className="w-full rounded-2xl border border-black/15 bg-white px-4 py-3 text-sm"
                value={trainBackend}
                onChange={(event) => setTrainBackend(event.target.value as "synthetic" | "isaaclab_teacher_ppo")}
              >
                <option value="synthetic">Synthetic (fast)</option>
                <option value="isaaclab_teacher_ppo">Isaac Lab PPO teacher (real, GPU)</option>
              </select>
            </label>
          </div>
          <div className="flex flex-wrap items-center gap-4">
            <Button onClick={handleStartTraining}>{autoTrain ? "Training will auto-start" : "Start training"}</Button>
            <Link href={`/training?dataset_id=${datasetId}`} className="text-sm font-semibold text-black underline">
              Advanced training page
            </Link>
            {xmimicJobId ? (
              <Link href={`/xmimic/${xmimicJobId}`} className="text-sm font-semibold text-black underline">
                View XMimic job
              </Link>
            ) : null}
          </div>
          {trainStatus ? <p className="text-sm text-emerald-700">{trainStatus}</p> : null}
          {trainError ? <p className="text-sm text-rose-700">{trainError}</p> : null}
        </Card>
      ) : null}

      {error ? <p className="text-sm text-rose-700">{error}</p> : null}
      {job?.demo_id ? (
        <p className="text-xs text-black/60">
          Demo id: <span className="font-mono">{job.demo_id}</span>
        </p>
      ) : null}
      {job?.error ? <p className="text-sm text-rose-700">Worker error: {job.error}</p> : null}
    </div>
  );
}
