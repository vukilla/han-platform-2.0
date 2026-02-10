"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { DatasetOut, listDatasets, runXmimic } from "@/lib/api";

export default function TrainingPage() {
  const [datasets, setDatasets] = useState<DatasetOut[]>([]);
  const [datasetId, setDatasetId] = useState<string>("");
  const [mode, setMode] = useState<"nep" | "mocap">("nep");
  const [distillation, setDistillation] = useState("teacher_student");
  const [backend, setBackend] = useState<"synthetic" | "isaaclab_teacher_ppo">("synthetic");
  const [numEnvs, setNumEnvs] = useState("32");
  const [updates, setUpdates] = useState("5");
  const [rolloutSteps, setRolloutSteps] = useState("128");
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const searchParams = useSearchParams();
  const requestedDatasetId = searchParams.get("dataset_id") || "";

  useEffect(() => {
    let cancelled = false;

    async function loadDatasets() {
      try {
        const response = await listDatasets();
        if (cancelled) return;
        setDatasets(response);
        setDatasetId((prev) => prev || requestedDatasetId || response[0]?.id || "");
      } catch (err) {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : "Failed to load datasets");
      }
    }

    loadDatasets();
    return () => {
      cancelled = true;
    };
  }, [requestedDatasetId]);

  const selectedDataset = useMemo(
    () => datasets.find((dataset) => dataset.id === datasetId) ?? null,
    [datasets, datasetId],
  );

  async function handleStartTraining() {
    if (!datasetId) {
      setError("Pick a dataset first.");
      return;
    }
    setError(null);
    setJobId(null);
    setStatus("Starting XMimic job...");
    try {
      const params: Record<string, unknown> = { distillation };
      if (backend === "isaaclab_teacher_ppo") {
        params.backend = "isaaclab_teacher_ppo";
        params.env_task = "cargo_pickup_v0";
        params.isaaclab_task = "cargo_pickup_franka";
        params.num_envs = Number(numEnvs);
        params.updates = Number(updates);
        params.rollout_steps = Number(rolloutSteps);
      }
      const job = await runXmimic(datasetId, mode, params);
      setJobId(job.id);
      setStatus(`XMimic job started (${job.status}).`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start training");
      setStatus(null);
    }
  }

  return (
    <div className="space-y-10">
      <section>
        <p className="section-eyebrow">XMimic</p>
        <h1 className="text-3xl font-semibold text-black">Train a teacher + student policy.</h1>
      </section>

      <div className="grid gap-6 lg:grid-cols-[1.2fr_0.8fr]">
        <Card className="space-y-4">
          <h2 className="text-xl font-semibold text-black">Configure run</h2>
          <label className="grid gap-2 text-sm font-semibold text-black">
            Dataset
            <select
              className="w-full rounded-2xl border border-black/15 bg-white px-4 py-3 text-sm"
              value={datasetId}
              onChange={(event) => setDatasetId(event.target.value)}
            >
              {datasets.length === 0 ? <option value="">No datasets found</option> : null}
              {datasets.map((dataset) => (
                <option key={dataset.id} value={dataset.id}>
                  {dataset.summary_json?.name ? String(dataset.summary_json.name) : `Dataset v${dataset.version}`}
                </option>
              ))}
            </select>
          </label>
          <label className="grid gap-2 text-sm font-semibold text-black">
            Backend
            <select
              className="w-full rounded-2xl border border-black/15 bg-white px-4 py-3 text-sm"
              value={backend}
              onChange={(event) => setBackend(event.target.value as "synthetic" | "isaaclab_teacher_ppo")}
            >
              <option value="synthetic">Synthetic (fast, plumbing)</option>
              <option value="isaaclab_teacher_ppo">Isaac Lab PPO teacher (real, GPU)</option>
            </select>
          </label>
          <div className="grid gap-3 md:grid-cols-2">
            <label className="grid gap-2 text-sm font-semibold text-black">
              Mode
              <select
                className="w-full rounded-2xl border border-black/15 bg-white px-4 py-3 text-sm"
                value={mode}
                onChange={(event) => setMode(event.target.value as "nep" | "mocap")}
              >
                <option value="nep">NEP (proprio only)</option>
                <option value="mocap">MoCap (object pose + dropout)</option>
              </select>
            </label>
            <label className="grid gap-2 text-sm font-semibold text-black">
              Distillation
              <select
                className="w-full rounded-2xl border border-black/15 bg-white px-4 py-3 text-sm"
                value={distillation}
                onChange={(event) => setDistillation(event.target.value)}
                disabled={backend === "isaaclab_teacher_ppo"}
              >
                <option value="teacher_student">Teacher + Student</option>
                <option value="student_only">Student only</option>
              </select>
            </label>
          </div>
          {backend === "isaaclab_teacher_ppo" ? (
            <div className="grid gap-3 md:grid-cols-3">
              <label className="grid gap-2 text-sm font-semibold text-black">
                Num envs
                <input
                  type="number"
                  className="w-full rounded-2xl border border-black/15 bg-white px-4 py-3 text-sm"
                  value={numEnvs}
                  onChange={(event) => setNumEnvs(event.target.value)}
                  min={1}
                  max={4096}
                />
              </label>
              <label className="grid gap-2 text-sm font-semibold text-black">
                Updates
                <input
                  type="number"
                  className="w-full rounded-2xl border border-black/15 bg-white px-4 py-3 text-sm"
                  value={updates}
                  onChange={(event) => setUpdates(event.target.value)}
                  min={1}
                  max={2000}
                />
              </label>
              <label className="grid gap-2 text-sm font-semibold text-black">
                Rollout steps
                <input
                  type="number"
                  className="w-full rounded-2xl border border-black/15 bg-white px-4 py-3 text-sm"
                  value={rolloutSteps}
                  onChange={(event) => setRolloutSteps(event.target.value)}
                  min={8}
                  max={4096}
                />
              </label>
            </div>
          ) : null}
          <Button onClick={handleStartTraining}>Start training</Button>
          {selectedDataset ? (
            <p className="text-xs text-black/60">Selected dataset: {selectedDataset.id}</p>
          ) : null}
          {jobId ? (
            <div className="flex flex-wrap items-center gap-3 text-sm">
              <Link href={`/xmimic/${jobId}`} className="font-semibold text-black underline">
                View XMimic job
              </Link>
              <Link href="/policies" className="font-semibold text-black underline">
                View policies
              </Link>
            </div>
          ) : null}
          {status ? <p className="text-sm text-emerald-700">{status}</p> : null}
          {error ? <p className="text-sm text-rose-700">{error}</p> : null}
        </Card>

        <Card className="space-y-4">
          <h2 className="text-xl font-semibold text-black">Unified reward</h2>
          <div className="flex flex-wrap gap-2">
            <Badge label="Body imitation" tone="amber" />
            <Badge label="Object tracking" tone="emerald" />
            <Badge label="Relative motion" tone="rose" />
            <Badge label="Contact graph" tone="amber" />
            <Badge label="Regularization" tone="stone" />
          </div>
          <p className="text-sm">
            Includes DI/IT/DR modules for generalization-first training across augmented distributions.
          </p>
        </Card>
      </div>
    </div>
  );
}
