import { LinkButton } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { StepList } from "@/components/ui/StepList";

const steps = [
  {
    title: "Upload a demo",
    detail: "Drop a human video or teleop clip and select the target robot and object.",
  },
  {
    title: "XGen synthesis",
    detail: "Retarget, synthesize contact, simulate non-contact, then augment at scale.",
  },
  {
    title: "XMimic training",
    detail: "Train teacher and student policies with unified imitation rewards.",
  },
];

export default function Home() {
  return (
    <div className="space-y-16">
      <section className="grid gap-10 lg:grid-cols-[1.1fr_0.9fr]">
        <div className="space-y-6">
          <p className="section-eyebrow">HumanX Data Factory</p>
          <h1 className="text-5xl font-semibold text-black">
            Turn human videos into deployable humanoid interaction skills.
          </h1>
          <p className="text-lg">
            Upload demonstrations, synthesize physically plausible interaction datasets with XGen, and train generalizable
            policies with XMimic. Publish datasets, checkpoints, and contributor attribution in a single workflow.
          </p>
          <div className="flex flex-wrap gap-4">
            <LinkButton href="/gvhmr">Run GVHMR</LinkButton>
            <LinkButton href="/dashboard" variant="outline">
              View dashboard
            </LinkButton>
          </div>
          <div className="flex flex-wrap gap-3">
            <Badge label="Video → Dataset" tone="amber" />
            <Badge label="Teacher–Student PPO" tone="emerald" />
            <Badge label="Incentivized Data" tone="rose" />
          </div>
        </div>
        <Card className="space-y-6">
          <div>
            <p className="text-sm font-semibold text-black/60">Live pipeline</p>
            <h2 className="text-2xl font-semibold text-black">Cargo pickup (MVP)</h2>
          </div>
          <div className="space-y-4 text-sm">
            <div className="flex items-center justify-between border-b border-black/10 pb-3">
              <span>Pose extraction</span>
              <span className="text-black">Queued</span>
            </div>
            <div className="flex items-center justify-between border-b border-black/10 pb-3">
              <span>Contact synthesis</span>
              <span className="text-black">Ready</span>
            </div>
            <div className="flex items-center justify-between border-b border-black/10 pb-3">
              <span>Augmentation</span>
              <span className="text-black">Pending</span>
            </div>
            <div className="flex items-center justify-between">
              <span>XMimic distillation</span>
              <span className="text-black">Waiting</span>
            </div>
          </div>
        </Card>
      </section>

      <section className="grid gap-8 lg:grid-cols-[0.9fr_1.1fr]">
        <Card className="space-y-6">
          <h2 className="text-2xl font-semibold text-black">End-to-end pipeline</h2>
          <StepList steps={steps} />
        </Card>
        <div className="space-y-6">
          <Card>
            <h3 className="text-xl font-semibold text-black">Why HumanX-style synthesis?</h3>
            <p className="mt-3 text-sm">
              Physical interaction data is scarce and fragmented. XGen standardizes human demonstrations into robot-contact
              trajectories, while XMimic learns policies that generalize beyond the original demos.
            </p>
          </Card>
          <Card>
            <h3 className="text-xl font-semibold text-black">Decentralized incentives</h3>
            <p className="mt-3 text-sm">
              Every demo is scored, validated, and attributed. Contributors earn points now and token rewards later.
            </p>
          </Card>
        </div>
      </section>
    </div>
  );
}
