"use client";

import { useEffect, useMemo, useState } from "react";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { DatasetOut, listDatasets, runXmimic } from "@/lib/api";

export default function TrainingPage() {
  const [datasets, setDatasets] = useState<DatasetOut[]>([]);
  const [datasetId, setDatasetId] = useState<string>("");
  const [mode, setMode] = useState<"nep" | "mocap">("nep");
  const [distillation, setDistillation] = useState("teacher_student");
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function loadDatasets() {
      try {
        const response = await listDatasets();
        if (cancelled) return;
        setDatasets(response);
        setDatasetId((prev) => prev || response[0]?.id || "");
      } catch (err) {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : "Failed to load datasets");
      }
    }

    loadDatasets();
    return () => {
      cancelled = true;
    };
  }, []);

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
    setStatus("Starting XMimic job...");
    try {
      const job = await runXmimic(datasetId, mode, { distillation });
      setStatus(`XMimic job ${job.id} started (${job.status}).`);
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
              >
                <option value="teacher_student">Teacher + Student</option>
                <option value="student_only">Student only</option>
              </select>
            </label>
          </div>
          <Button onClick={handleStartTraining}>Start training</Button>
          {selectedDataset ? (
            <p className="text-xs text-black/60">Selected dataset: {selectedDataset.id}</p>
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
