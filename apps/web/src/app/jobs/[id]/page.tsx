"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useParams, useSearchParams } from "next/navigation";
import Link from "next/link";
import { Badge } from "@/components/ui/Badge";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { getXgenJob, runXmimic, XGenJobOut } from "@/lib/api";

const DEFAULT_ISAACLAB_NUM_ENVS = 8;
const DEFAULT_ISAACLAB_UPDATES = 2;
const DEFAULT_ISAACLAB_ROLLOUT_STEPS = 64;

const stages = [
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
  const [error, setError] = useState<string | null>(null);
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

  const status = job?.status ?? null;
  const jobIsComplete = status === "COMPLETED";
  const jobIsFailed = status === "FAILED";
  const datasetId = typeof job?.params_json?.dataset_id === "string" ? job?.params_json?.dataset_id : null;

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

  const currentIndex = status ? stages.indexOf(status) : -1;
  const stageLabel = (index: number) => {
    if (jobIsComplete) return "Completed";
    if (jobIsFailed) return "Failed";
    if (currentIndex === -1) return "Queued";
    if (index < currentIndex) return "Completed";
    if (index === currentIndex) return "Running";
    return "Queued";
  };

  return (
    <div className="space-y-8">
      <section className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <p className="section-eyebrow">Job progress</p>
          <h1 className="text-3xl font-semibold text-black">XGen pipeline stages</h1>
          {jobId ? <p className="mt-2 text-sm text-black/60">{jobId}</p> : null}
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <Badge label={status || "Queued"} tone={jobIsComplete ? "emerald" : jobIsFailed ? "rose" : "amber"} />
          {datasetId ? (
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

      <Card className="space-y-6">
        {stages.map((stage, index) => (
          <div key={stage} className="flex items-center justify-between border-b border-black/10 pb-4 last:border-none last:pb-0">
            <div>
              <p className="text-sm font-semibold text-black/60">Stage {index + 1}</p>
              <p className="text-lg font-semibold text-black">{stage}</p>
            </div>
            <span className="text-sm text-black/70">{stageLabel(index)}</span>
          </div>
        ))}
      </Card>

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

      {datasetId && jobIsComplete ? (
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
