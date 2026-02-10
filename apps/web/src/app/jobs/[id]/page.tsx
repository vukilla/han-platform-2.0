"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { Badge } from "@/components/ui/Badge";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { getXgenJob, runXmimic, XGenJobOut } from "@/lib/api";

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
  const jobId = params?.id;
  const [job, setJob] = useState<XGenJobOut | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [trainMode, setTrainMode] = useState<"nep" | "mocap">("nep");
  const [trainBackend, setTrainBackend] = useState<"synthetic" | "isaaclab_teacher_ppo">("synthetic");
  const [trainStatus, setTrainStatus] = useState<string | null>(null);
  const [trainError, setTrainError] = useState<string | null>(null);
  const [xmimicJobId, setXmimicJobId] = useState<string | null>(null);

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

  async function handleStartTraining() {
    if (!datasetId || !jobIsComplete) return;
    setTrainError(null);
    setXmimicJobId(null);
    setTrainStatus("Starting XMimic job...");
    try {
      const params: Record<string, unknown> = {};
      if (trainBackend === "isaaclab_teacher_ppo") {
        params.backend = "isaaclab_teacher_ppo";
        params.env_task = "cargo_pickup_v0";
        params.isaaclab_task = "cargo_pickup_franka";
        params.num_envs = 32;
        params.updates = 5;
        params.rollout_steps = 128;
      }
      const xmimic = await runXmimic(datasetId, trainMode, params);
      setXmimicJobId(xmimic.id);
      setTrainStatus("XMimic job started.");
    } catch (err) {
      setTrainError(err instanceof Error ? err.message : "Failed to start training");
      setTrainStatus(null);
    }
  }

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
            <Button onClick={handleStartTraining}>Start training</Button>
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
