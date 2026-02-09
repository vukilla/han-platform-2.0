"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Card } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { DatasetOut, listDatasets } from "@/lib/api";

const statusTone = (status: string) => {
  if (status.toLowerCase().includes("ready") || status.toLowerCase().includes("complete")) return "emerald";
  if (status.toLowerCase().includes("fail")) return "rose";
  return "amber";
};

export default function DatasetsIndexPage() {
  const [datasets, setDatasets] = useState<DatasetOut[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        setLoading(true);
        const response = await listDatasets();
        if (cancelled) return;
        setDatasets(response);
        setError(null);
      } catch (err) {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : "Failed to load datasets");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="space-y-10">
      <section>
        <p className="section-eyebrow">Datasets</p>
        <h1 className="text-3xl font-semibold text-black">Generated datasets & clips.</h1>
      </section>

      <Card>
        {loading ? <p className="text-sm text-black/60">Loading…</p> : null}
        {error ? <p className="text-sm text-rose-700">{error}</p> : null}
        {!loading && !error && datasets.length === 0 ? (
          <p className="text-sm text-black/60">No datasets yet. Start from the demo wizard.</p>
        ) : null}

        <div className="mt-6 grid gap-4 text-sm">
          {datasets.map((dataset) => {
            const name = dataset.summary_json?.name ? String(dataset.summary_json.name) : `Dataset v${dataset.version}`;
            const clipCount =
              typeof dataset.summary_json?.clip_count === "number" ? dataset.summary_json.clip_count : undefined;
            return (
              <Link
                key={dataset.id}
                href={`/datasets/${dataset.id}`}
                className="flex items-center justify-between border-b border-black/10 pb-4 transition hover:bg-black/5"
              >
                <div>
                  <p className="font-semibold text-black">{name}</p>
                  <p className="text-xs text-black/60 font-mono">{dataset.id}</p>
                  {clipCount !== undefined ? <p className="text-xs text-black/60">{clipCount} clips</p> : null}
                </div>
                <Badge label={dataset.status} tone={statusTone(dataset.status)} />
              </Link>
            );
          })}
        </div>
      </Card>
    </div>
  );
}

