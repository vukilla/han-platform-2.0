"use client";

import { useEffect, useMemo, useState } from "react";
import { useParams } from "next/navigation";
import { Card } from "@/components/ui/Card";
import { LinkButton } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { EvalRunOut, getEval } from "@/lib/api";

const formatPercent = (value?: number | null) => {
  if (value === null || value === undefined) return "—";
  return `${Math.round(value * 1000) / 10}%`;
};

const formatNumber = (value?: number | null) => {
  if (value === null || value === undefined) return "—";
  return value.toFixed(3);
};

export default function EvalReportPage() {
  const params = useParams();
  const evalId = typeof params?.id === "string" ? params.id : params?.id?.[0];
  const [evalRun, setEvalRun] = useState<EvalRunOut | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!evalId) return;
    const id = evalId;
    let cancelled = false;

    async function load() {
      try {
        setLoading(true);
        const response = await getEval(id);
        if (cancelled) return;
        setEvalRun(response);
        setError(null);
      } catch (err) {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : "Failed to load evaluation");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, [evalId]);

  const metrics = useMemo(
    () => [
      { label: "SR", value: formatPercent(evalRun?.sr ?? null) },
      { label: "GSR", value: formatPercent(evalRun?.gsr ?? null) },
      { label: "Eo", value: formatNumber(evalRun?.eo ?? null) },
      { label: "Eh", value: formatNumber(evalRun?.eh ?? null) },
    ],
    [evalRun],
  );

  const breakdown = evalRun?.report_uri ? null : evalRun?.env_task ? ["baseline", "mesh scale", "contact shift"] : [];

  return (
    <div className="space-y-10">
      <section className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <p className="section-eyebrow">Evaluation report</p>
          <h1 className="text-3xl font-semibold text-black">
            {evalRun ? `Eval run ${evalRun.id.slice(0, 8)}` : "Evaluation report"}
          </h1>
          {evalRun ? <p className="text-sm text-black/60">Env task: {evalRun.env_task}</p> : null}
        </div>
        <div className="flex flex-wrap items-center gap-3">
          {evalRun?.report_uri ? (
            <LinkButton href={evalRun.report_uri} variant="outline" target="_blank" rel="noreferrer">
              Download report
            </LinkButton>
          ) : null}
          {evalRun?.videos_uri ? (
            <LinkButton href={evalRun.videos_uri} target="_blank" rel="noreferrer">
              Download videos
            </LinkButton>
          ) : null}
          {loading ? <Badge label="Loading" tone="amber" /> : null}
        </div>
      </section>

      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-4">
        {metrics.map((metric) => (
          <Card key={metric.label} className="text-center">
            <p className="text-sm font-semibold text-black/60">{metric.label}</p>
            <p className="mt-2 text-3xl font-semibold text-black">{metric.value}</p>
          </Card>
        ))}
      </div>

      <Card>
        <h2 className="text-xl font-semibold text-black">Generalization slices</h2>
        <div className="mt-4 grid gap-3 text-sm">
          {breakdown?.length ? (
            breakdown.map((item) => (
              <div key={item} className="flex items-center justify-between border-b border-black/10 pb-3 last:border-none last:pb-0">
                <span>Augmentation: {item}</span>
                <span className="text-black/70">GSR —</span>
              </div>
            ))
          ) : (
            <p className="text-sm text-black/60">No generalization breakdown recorded yet.</p>
          )}
        </div>
      </Card>

      {error ? <p className="text-sm text-rose-700">{error}</p> : null}
    </div>
  );
}
