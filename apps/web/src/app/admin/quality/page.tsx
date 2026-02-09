"use client";

import { useState } from "react";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { getQualityScore, QualityScoreOut, reviewQualityScore } from "@/lib/api";

export default function QualityReviewPage() {
  const [entityType, setEntityType] = useState("demo");
  const [entityId, setEntityId] = useState("");
  const [score, setScore] = useState<QualityScoreOut | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [notes, setNotes] = useState("");

  async function handleFetch() {
    setError(null);
    setStatus("Loading quality score...");
    try {
      const response = await getQualityScore(entityType, entityId);
      setScore(response);
      setStatus(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load quality score");
      setStatus(null);
    }
  }

  async function handleReview(nextStatus: "approved" | "rejected") {
    if (!entityId) return;
    setStatus("Submitting review...");
    try {
      const response = await reviewQualityScore(entityType, entityId, { status: nextStatus, notes });
      setScore(response);
      setStatus(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to submit review");
      setStatus(null);
    }
  }

  return (
    <div className="space-y-8">
      <section>
        <p className="section-eyebrow">Admin review</p>
        <h1 className="text-3xl font-semibold text-black">Quality review queue</h1>
      </section>

      <Card className="space-y-4">
        <h2 className="text-xl font-semibold text-black">Lookup entity</h2>
        <div className="grid gap-3 md:grid-cols-[160px_1fr_auto]">
          <select
            className="w-full rounded-2xl border border-black/15 bg-white px-4 py-3 text-sm"
            value={entityType}
            onChange={(event) => setEntityType(event.target.value)}
          >
            <option value="demo">Demo</option>
            <option value="clip">Clip</option>
            <option value="dataset">Dataset</option>
          </select>
          <input
            type="text"
            value={entityId}
            onChange={(event) => setEntityId(event.target.value)}
            placeholder="Entity ID"
            className="w-full rounded-2xl border border-black/15 bg-white px-4 py-3 text-sm"
          />
          <Button onClick={handleFetch}>Fetch</Button>
        </div>
        {status ? <p className="text-sm text-black/60">{status}</p> : null}
        {error ? <p className="text-sm text-rose-700">{error}</p> : null}
      </Card>

      {score ? (
        <Card className="space-y-4">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <h2 className="text-xl font-semibold text-black">Quality result</h2>
            <Badge label={score.validator_status || "Pending"} tone={score.validator_status === "approved" ? "emerald" : "amber"} />
          </div>
          <div className="grid gap-3 text-sm">
            <p>Score: {score.score ?? "—"}</p>
            <p>Entity: {score.entity_type} / {score.entity_id}</p>
          </div>
          <div className="rounded-2xl border border-black/10 bg-black/5 p-4 text-xs">
            <pre className="whitespace-pre-wrap">{JSON.stringify(score.breakdown_json, null, 2)}</pre>
          </div>
          <textarea
            value={notes}
            onChange={(event) => setNotes(event.target.value)}
            placeholder="Reviewer notes"
            className="min-h-[100px] w-full rounded-2xl border border-black/15 bg-white px-4 py-3 text-sm"
          />
          <div className="flex gap-3">
            <Button variant="outline" onClick={() => handleReview("rejected")}>
              Reject
            </Button>
            <Button onClick={() => handleReview("approved")}>Approve</Button>
          </div>
        </Card>
      ) : null}
    </div>
  );
}
