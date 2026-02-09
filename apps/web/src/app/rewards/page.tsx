"use client";

import { useEffect, useMemo, useState } from "react";
import { Card } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { RewardEventOut, getRewardsMe } from "@/lib/api";

function formatPoints(points: number) {
  const sign = points >= 0 ? "+" : "";
  return `${sign}${points}`;
}

export default function RewardsPage() {
  const [events, setEvents] = useState<RewardEventOut[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    async function loadRewards() {
      try {
        setLoading(true);
        const response = await getRewardsMe();
        if (cancelled) return;
        setEvents(response);
        setError(null);
      } catch (err) {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : "Failed to load rewards");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    loadRewards();
    return () => {
      cancelled = true;
    };
  }, []);

  const totalPoints = useMemo(() => events.reduce((sum, event) => sum + event.points, 0), [events]);
  const hasApproved = events.some((event) => event.points > 0);

  return (
    <div className="space-y-10">
      <section>
        <p className="section-eyebrow">Rewards</p>
        <h1 className="text-3xl font-semibold text-black">Contributor points & attribution.</h1>
      </section>

      <div className="grid gap-6 lg:grid-cols-[1fr_1fr]">
        <Card>
          <p className="text-sm font-semibold text-black/60">Total points</p>
          <p className="mt-3 text-4xl font-semibold text-black">{totalPoints}</p>
          <p className="text-sm">Token staking opens in the incentives epic.</p>
        </Card>
        <Card>
          <p className="text-sm font-semibold text-black/60">Quality status</p>
          <div className="mt-4 flex flex-wrap gap-2">
            <Badge label={hasApproved ? "Validated demos" : "No validated demos"} tone={hasApproved ? "emerald" : "amber"} />
            <Badge label={loading ? "Loading" : "Pending review"} tone="amber" />
          </div>
          <p className="mt-4 text-sm">Quality scoring will unlock validator feedback and bonus rewards.</p>
        </Card>
      </div>

      <Card>
        <h2 className="text-xl font-semibold text-black">Recent events</h2>
        {loading ? <p className="mt-4 text-sm text-black/60">Loading rewards…</p> : null}
        {!loading && events.length === 0 ? <p className="mt-4 text-sm text-black/60">No reward events yet.</p> : null}
        <div className="mt-6 grid gap-4 text-sm">
          {events.map((event) => (
            <div key={event.id} className="flex items-center justify-between border-b border-black/10 pb-3">
              <div>
                <p className="font-semibold text-black">{event.reason}</p>
                <p className="text-xs text-black/60">{event.entity_type}</p>
              </div>
              <div className="flex items-center gap-3">
                <span className="font-semibold text-black">{formatPoints(event.points)}</span>
                <Badge label={event.points >= 0 ? "Approved" : "Deducted"} tone={event.points >= 0 ? "emerald" : "rose"} />
              </div>
            </div>
          ))}
        </div>
      </Card>

      {error ? <p className="text-sm text-rose-700">{error}</p> : null}
    </div>
  );
}
