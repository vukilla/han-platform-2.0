"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { EvalRunOut, listEvalRuns, listPolicies, PolicyOut } from "@/lib/api";

type EvalByPolicy = Record<string, EvalRunOut[]>;

export default function PoliciesPage() {
  const [policies, setPolicies] = useState<PolicyOut[]>([]);
  const [evalByPolicy, setEvalByPolicy] = useState<EvalByPolicy>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        setLoading(true);
        const policyResponse = await listPolicies();
        if (cancelled) return;
        setPolicies(policyResponse);

        const evalPairs = await Promise.all(
          policyResponse.map(async (policy) => {
            const evalRuns = await listEvalRuns(policy.id);
            return [policy.id, evalRuns] as const;
          }),
        );
        if (cancelled) return;
        const mapping: EvalByPolicy = {};
        evalPairs.forEach(([policyId, runs]) => {
          mapping[policyId] = runs;
        });
        setEvalByPolicy(mapping);
        setError(null);
      } catch (err) {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : "Failed to load policies");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, []);

  const total = policies.length;
  const syntheticCount = useMemo(
    () => policies.filter((policy) => Boolean(policy.metadata_json?.synthetic)).length,
    [policies],
  );

  function policyLabel(policy: PolicyOut) {
    const mode = typeof policy.metadata_json?.mode === "string" ? policy.metadata_json.mode : "unknown";
    return `${mode} policy`;
  }

  return (
    <div className="space-y-10">
      <section className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <p className="section-eyebrow">Policies</p>
          <h1 className="text-3xl font-semibold text-black">Checkpoints & evaluation runs.</h1>
        </div>
        <div className="flex items-center gap-2 text-sm">
          <Badge label={`${total} total`} tone="amber" />
          <Badge label={`${syntheticCount} synthetic`} tone="stone" />
        </div>
      </section>

      <Card className="space-y-6">
        {loading ? <p className="text-sm text-black/60">Loading…</p> : null}
        {error ? <p className="text-sm text-rose-700">{error}</p> : null}
        {!loading && !error && policies.length === 0 ? (
          <p className="text-sm text-black/60">No policies yet. Start from the training page.</p>
        ) : null}

        <div className="grid gap-4">
          {policies.map((policy) => {
            const evalRuns = evalByPolicy[policy.id] || [];
            const latestEval = evalRuns[0] || null;
            const meta = (policy.metadata_json ?? {}) as Record<string, unknown>;
            const isSynthetic = Boolean(meta["synthetic"]);
            const mode = typeof meta["mode"] === "string" ? meta["mode"] : null;
            return (
              <div key={policy.id} className="rounded-2xl border border-black/10 p-4">
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <p className="text-sm font-semibold text-black">{policyLabel(policy)}</p>
                    <p className="text-xs font-mono text-black/60">{policy.id}</p>
                    <p className="mt-2 text-xs text-black/60">
                      XMimic job: <span className="font-mono">{policy.xmimic_job_id}</span>
                    </p>
                  </div>
                  <div className="flex flex-wrap items-center gap-3">
                    <Button
                      variant="outline"
                      onClick={() => window.open(policy.checkpoint_uri, "_blank", "noopener,noreferrer")}
                    >
                      Download checkpoint
                    </Button>
                    <Link href="/training" className="text-sm font-semibold text-black underline">
                      New run
                    </Link>
                  </div>
                </div>

                <div className="mt-4 flex flex-wrap items-center justify-between gap-3 text-sm">
                  <div className="flex flex-wrap items-center gap-2">
                    <Badge label={isSynthetic ? "Synthetic" : "Real"} tone="amber" />
                    <Badge label={mode ?? "mode?"} tone="stone" />
                  </div>
                  {latestEval ? (
                    <Link href={`/eval/${latestEval.id}`} className="text-sm font-semibold text-black underline">
                      Open eval ({latestEval.env_task})
                    </Link>
                  ) : (
                    <span className="text-black/60">No eval runs yet.</span>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </Card>
    </div>
  );
}
