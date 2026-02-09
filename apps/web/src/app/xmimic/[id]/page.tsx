"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { Badge } from "@/components/ui/Badge";
import { Card } from "@/components/ui/Card";
import { getXmimicJob, XMimicJobOut } from "@/lib/api";

const stages = [
  "BUILD_ENV",
  "TRAIN_TEACHER",
  "DISTILL_STUDENT",
  "EVAL_POLICY",
  "EXPORT_CHECKPOINT",
  "QUALITY_SCORE",
];

export default function XmimicJobProgressPage() {
  const params = useParams<{ id: string }>();
  const jobId = params?.id;
  const [job, setJob] = useState<XMimicJobOut | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let timer: NodeJS.Timeout | null = null;

    async function fetchStatus() {
      if (!jobId) return;
      try {
        const response = await getXmimicJob(jobId);
        setJob(response);
        setError(null);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load job");
      }
    }

    fetchStatus();
    timer = setInterval(fetchStatus, 3000);

    return () => {
      if (timer) clearInterval(timer);
    };
  }, [jobId]);

  const status = job?.status ?? null;
  const jobIsComplete = status === "COMPLETED";
  const jobIsFailed = status === "FAILED";
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
          <h1 className="text-3xl font-semibold text-black">XMimic training stages</h1>
          {jobId ? <p className="mt-2 text-sm text-black/60">{jobId}</p> : null}
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <Badge label={status || "Queued"} tone={jobIsComplete ? "emerald" : jobIsFailed ? "rose" : "amber"} />
          {job?.dataset_id ? (
            <Link href={`/datasets/${job.dataset_id}`} className="text-sm font-semibold text-black underline">
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

      {job?.mode ? (
        <p className="text-xs text-black/60">
          Mode: <span className="font-mono">{job.mode}</span>
        </p>
      ) : null}
      {job?.error ? <p className="text-sm text-rose-700">Worker error: {job.error}</p> : null}
      {error ? <p className="text-sm text-rose-700">{error}</p> : null}
    </div>
  );
}

