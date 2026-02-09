"use client";

import { useEffect, useMemo, useState } from "react";
import { useParams } from "next/navigation";
import { Card } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import {
  DatasetClipOut,
  DatasetOut,
  getDataset,
  getDatasetDownloadUrl,
  listDatasetClips,
} from "@/lib/api";

const statusTone = (status: string) => {
  if (status.toLowerCase().includes("complete")) return "emerald";
  if (status.toLowerCase().includes("fail")) return "rose";
  return "amber";
};

function summarizeTags(tags?: string[] | null) {
  if (!tags || tags.length === 0) {
    return "No augmentation tags";
  }
  return tags.join(" · ");
}

function clipLabel(clipId: string) {
  return `Clip ${clipId.slice(0, 8)}`;
}

export default function DatasetDetailPage() {
  const params = useParams();
  const datasetId = typeof params?.id === "string" ? params.id : params?.id?.[0];
  const [dataset, setDataset] = useState<DatasetOut | null>(null);
  const [clips, setClips] = useState<DatasetClipOut[]>([]);
  const [selectedClipId, setSelectedClipId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [downloadBusy, setDownloadBusy] = useState(false);

  useEffect(() => {
    if (!datasetId) {
      return;
    }
    let cancelled = false;

    async function load() {
      try {
        setLoading(true);
        const [datasetResponse, clipResponse] = await Promise.all([
          getDataset(datasetId),
          listDatasetClips(datasetId),
        ]);
        if (cancelled) return;
        setDataset(datasetResponse);
        setClips(clipResponse);
        setSelectedClipId((prev) => prev ?? clipResponse[0]?.clip_id ?? null);
        setError(null);
      } catch (err) {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : "Failed to load dataset");
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, [datasetId]);

  const selectedClip = useMemo(
    () => clips.find((clip) => clip.clip_id === selectedClipId) ?? null,
    [clips, selectedClipId],
  );

  const summary = dataset?.summary_json ?? {};
  const clipCount =
    typeof summary.clips === "number" ? summary.clips : typeof summary.clip_count === "number" ? summary.clip_count : clips.length;
  const augmentationCount = useMemo(() => {
    const tags = new Set<string>();
    clips.forEach((clip) => {
      clip.augmentation_tags?.forEach((tag) => tags.add(tag));
    });
    return tags.size;
  }, [clips]);
  const successRate =
    typeof summary.success_rate === "number"
      ? summary.success_rate
      : typeof summary.success === "number"
        ? summary.success
        : null;

  async function handleDownload() {
    if (!datasetId) return;
    try {
      setDownloadBusy(true);
      const { download_url } = await getDatasetDownloadUrl(datasetId);
      window.open(download_url, "_blank", "noopener,noreferrer");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to create download link");
    } finally {
      setDownloadBusy(false);
    }
  }

  return (
    <div className="space-y-10">
      <section className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <p className="section-eyebrow">Dataset detail</p>
          <h1 className="text-3xl font-semibold text-black">
            {summary.name ? String(summary.name) : dataset ? `Dataset v${dataset.version}` : "Dataset"}
          </h1>
          {dataset ? <p className="text-sm text-black/60">{dataset.id}</p> : null}
        </div>
        <div className="flex items-center gap-3">
          {dataset?.status ? <Badge label={dataset.status} tone={statusTone(dataset.status)} /> : null}
          <Button onClick={handleDownload} variant="outline" disabled={downloadBusy || !datasetId}>
            {downloadBusy ? "Preparing..." : "Download dataset"}
          </Button>
        </div>
      </section>

      <div className="grid gap-6 lg:grid-cols-[1.2fr_0.8fr]">
        <Card>
          <h2 className="text-xl font-semibold text-black">Clips</h2>
          {loading ? <p className="mt-4 text-sm text-black/60">Loading clips…</p> : null}
          {!loading && clips.length === 0 ? (
            <p className="mt-4 text-sm text-black/60">No clips found for this dataset yet.</p>
          ) : null}
          <div className="mt-6 grid gap-4">
            {clips.map((clip) => {
              const isSelected = clip.clip_id === selectedClipId;
              const statusLabel = clip.stats_json?.status ? String(clip.stats_json.status) : "Pending";
              return (
                <button
                  key={clip.clip_id}
                  type="button"
                  onClick={() => setSelectedClipId(clip.clip_id)}
                  className={`flex w-full items-center justify-between border-b border-black/10 pb-3 text-left transition ${
                    isSelected ? "bg-black/5" : "bg-transparent"
                  }`}
                >
                  <div>
                    <p className="text-sm font-semibold text-black">{clipLabel(clip.clip_id)}</p>
                    <p className="text-xs text-black/60">{summarizeTags(clip.augmentation_tags)}</p>
                  </div>
                  <Badge label={statusLabel} tone={statusLabel === "Validated" ? "emerald" : "amber"} />
                </button>
              );
            })}
          </div>
        </Card>

        <Card>
          <h2 className="text-xl font-semibold text-black">Preview</h2>
          {selectedClip?.uri_preview_mp4 ? (
            <video
              className="mt-4 h-52 w-full rounded-2xl border border-black/10 bg-black/5 object-cover"
              controls
              src={selectedClip.uri_preview_mp4}
            />
          ) : (
            <div className="mt-4 h-52 rounded-2xl border border-dashed border-black/20 bg-black/5" />
          )}
          <div className="mt-4 grid gap-2 text-sm">
            <p>Clips: {clipCount}</p>
            <p>Augmentations: {augmentationCount}</p>
            <p>Success checks: {successRate !== null ? `${successRate}%` : "—"}</p>
          </div>
        </Card>
      </div>

      {error ? <p className="text-sm text-rose-700">{error}</p> : null}
    </div>
  );
}
