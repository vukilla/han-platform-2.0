import Link from "next/link";
import { Card } from "@/components/ui/Card";

export default function DeployPage() {
  return (
    <div className="mx-auto max-w-2xl space-y-8">
      <section className="space-y-3">
        <p className="section-eyebrow">Deploy</p>
        <h1 className="text-3xl font-semibold text-black">Coming soon</h1>
        <p className="text-sm text-black/70">
          Real robot deployment is not available yet. For now, you can teach skills from videos and validate outputs in simulation.
        </p>
      </section>

      <Card className="space-y-3">
        <p className="text-sm font-semibold text-black">What you can do now</p>
        <div className="space-y-2 text-sm text-black/70">
          <p>
            1. Recover 3D motion from a phone video: <Link className="font-semibold underline" href="/studio">Open Studio</Link>
          </p>
          <p>
            2. Run the full pipeline (dataset + training):{" "}
            <Link className="font-semibold underline" href="/demos/new">
              Advanced pipeline
            </Link>
          </p>
        </div>
      </Card>
    </div>
  );
}
