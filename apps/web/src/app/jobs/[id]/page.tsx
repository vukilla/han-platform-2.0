"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { Badge } from "@/components/ui/Badge";
import { Card } from "@/components/ui/Card";
import { getXgenJob } from "@/lib/api";

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
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let timer: NodeJS.Timeout | null = null;

    async function fetchStatus() {
      if (!jobId) {
        return;
      }
      try {
        const job = await getXgenJob(jobId);
        setStatus(job.status);
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

  return (
    <div className="space-y-8">
      <section className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <p className="section-eyebrow">Job progress</p>
          <h1 className="text-3xl font-semibold text-black">XGen pipeline stages</h1>
        </div>
        <Badge label={status || "Queued"} tone={status === "COMPLETED" ? "emerald" : "amber"} />
      </section>

      <Card className="space-y-6">
        {stages.map((stage, index) => (
          <div key={stage} className="flex items-center justify-between border-b border-black/10 pb-4 last:border-none last:pb-0">
            <div>
              <p className="text-sm font-semibold text-black/60">Stage {index + 1}</p>
              <p className="text-lg font-semibold text-black">{stage}</p>
            </div>
            <span className="text-sm text-black/70">{status === stage ? "Running" : "Queued"}</span>
          </div>
        ))}
      </Card>
      {error ? <p className="text-sm text-rose-700">{error}</p> : null}
    </div>
  );
}
