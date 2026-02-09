import { Card } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { LinkButton } from "@/components/ui/Button";

export default function DashboardPage() {
  return (
    <div className="space-y-10">
      <section className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <p className="section-eyebrow">Dashboard</p>
          <h1 className="text-3xl font-semibold text-black">Projects, demos, datasets, policies.</h1>
        </div>
        <LinkButton href="/demos/new">New demo</LinkButton>
      </section>

      <div className="grid gap-6 lg:grid-cols-3">
        <Card>
          <p className="text-sm font-semibold text-black/60">Active demos</p>
          <p className="mt-3 text-3xl font-semibold text-black">3</p>
          <p className="text-sm">1 running XGen job, 2 ready for XMimic.</p>
        </Card>
        <Card>
          <p className="text-sm font-semibold text-black/60">Datasets</p>
          <p className="mt-3 text-3xl font-semibold text-black">5</p>
          <p className="text-sm">12,480 clips across 4 augmentations.</p>
        </Card>
        <Card>
          <p className="text-sm font-semibold text-black/60">Policies</p>
          <p className="mt-3 text-3xl font-semibold text-black">2</p>
          <p className="text-sm">Cargo pickup student + teacher pair.</p>
        </Card>
      </div>

      <Card>
        <div className="flex items-center justify-between">
          <h2 className="text-xl font-semibold text-black">Recent activity</h2>
          <Badge label="Live" tone="emerald" />
        </div>
        <div className="mt-6 grid gap-4 text-sm">
          <div className="flex items-center justify-between border-b border-black/10 pb-3">
            <span>Cargo pickup demo uploaded</span>
            <span className="text-black/60">2 minutes ago</span>
          </div>
          <div className="flex items-center justify-between border-b border-black/10 pb-3">
            <span>XGen augmentation 3 completed</span>
            <span className="text-black/60">15 minutes ago</span>
          </div>
          <div className="flex items-center justify-between">
            <span>XMimic student distillation queued</span>
            <span className="text-black/60">1 hour ago</span>
          </div>
        </div>
      </Card>
    </div>
  );
}
