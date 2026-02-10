"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { Card } from "@/components/ui/Card";
import { Button } from "@/components/ui/Button";
import { Badge } from "@/components/ui/Badge";
import { annotateDemo, createDemo, createProject, runXgen, uploadDemoVideo } from "@/lib/api";

export default function NewDemoPage() {
  const router = useRouter();
  const [projectName, setProjectName] = useState("Cargo Pickup Project");
  const [file, setFile] = useState<File | null>(null);
  const [robotModel, setRobotModel] = useState("unitree-g1");
  const [objectId, setObjectId] = useState("cargo_box");
  const [pipeline, setPipeline] = useState<"fast" | "real">("real");
  const [gvhmrStaticCam, setGvhmrStaticCam] = useState(true);
  const [autoTrain, setAutoTrain] = useState(true);
  const [gpuReady, setGpuReady] = useState<boolean | null>(null);
  const [tsStart, setTsStart] = useState("0.5");
  const [tsEnd, setTsEnd] = useState("8.0");
  const [anchorType, setAnchorType] = useState("palms_midpoint");
  const [keyBodies, setKeyBodies] = useState("left_hand,right_hand");
  const [objectPose, setObjectPose] = useState({
    x: "0",
    y: "0",
    z: "0.6",
    roll: "0",
    pitch: "0",
    yaw: "0",
  });
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);

  useEffect(() => {
    let timer: NodeJS.Timeout | null = null;
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    const poll = () => {
      fetch(`${apiUrl}/ops/workers?timeout=1.0`)
        .then((res) => (res.ok ? res.json() : null))
        .then((data) => setGpuReady(Boolean(data?.has_gpu_queue)))
        .catch(() => setGpuReady(null));
    };
    poll();
    timer = setInterval(poll, 3000);
    return () => {
      if (timer) clearInterval(timer);
    };
  }, []);

  async function handleRun() {
    setError(null);
    setJobId(null);
    if (!file) {
      setError("Please select a video file first.");
      return;
    }

    try {
      setStatus("Creating project...");
      const project = await createProject(projectName, "MVP cargo pickup demo");

      setStatus("Creating demo record...");
      const demo = await createDemo(project.id, robotModel, objectId);

      setStatus("Uploading video...");
      await uploadDemoVideo(demo.id, file);

      setStatus("Saving annotations...");
      await annotateDemo(demo.id, {
        ts_contact_start: Number(tsStart),
        ts_contact_end: Number(tsEnd),
        anchor_type: anchorType,
        key_bodies: keyBodies.split(",").map((val) => val.trim()).filter(Boolean),
      });

      setStatus("Starting XGen job...");
      const xgenParams: Record<string, unknown> = {
        object_id: objectId,
        object_pose: {
          x: Number(objectPose.x),
          y: Number(objectPose.y),
          z: Number(objectPose.z),
          roll: Number(objectPose.roll),
          pitch: Number(objectPose.pitch),
          yaw: Number(objectPose.yaw),
        },
      };

      if (pipeline === "fast") {
        xgenParams.placeholder_pose = true;
      } else {
        xgenParams.requires_gpu = true;
        xgenParams.pose_estimator = "gvhmr";
        xgenParams.gvhmr_static_cam = Boolean(gvhmrStaticCam);
        // Keep these small for interactive iteration.
        xgenParams.clip_count = 3;
        xgenParams.frames = 40;
        xgenParams.nq = 12;
        xgenParams.contact_dim = 4;
      }

      const job = await runXgen(demo.id, {
        ...xgenParams,
      });
      setJobId(job.id);
      setStatus("XGen job started. Redirecting...");
      const qp = autoTrain
        ? `?auto_train=1&train_mode=${pipeline === "real" ? "mocap" : "nep"}&train_backend=${
            pipeline === "real" ? "isaaclab_teacher_ppo" : "synthetic"
          }`
        : "";
      router.push(`/jobs/${job.id}${qp}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to run demo");
      setStatus(null);
    }
  }

  return (
    <div className="space-y-8">
      <section>
        <p className="section-eyebrow">Teach A Robot From A Video</p>
        <h1 className="text-3xl font-semibold text-black">Upload one phone video. Get a dataset and a policy.</h1>
        <p className="mt-2 max-w-2xl text-sm text-black/70">
          Minimal flow: upload, run XGen, then (optionally) train an XMimic policy. Advanced settings are optional.
        </p>
      </section>

      <div className="grid gap-6">
        <Card className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-semibold text-black">1. Upload video</h2>
            <Badge label="Required" tone="amber" />
          </div>
          <input
            type="file"
            onChange={(event) => setFile(event.target.files?.[0] ?? null)}
            className="w-full rounded-2xl border border-black/15 bg-white px-4 py-4 text-sm"
          />
          <p className="text-sm text-black/70">Use a clean, front-facing monocular video for MVP.</p>
        </Card>

        <Card className="space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-semibold text-black">2. Run</h2>
            <Badge
              label={gpuReady === true ? "GPU worker connected" : gpuReady === false ? "GPU worker offline" : "GPU status unknown"}
              tone={gpuReady === true ? "emerald" : gpuReady === false ? "rose" : "amber"}
            />
          </div>
          <p className="text-sm">
            Pick <strong>Real</strong> to run GVHMR pose extraction and Isaac Lab PPO on the Windows GPU worker, or{" "}
            <strong>Fast</strong> to run a stubbed pipeline for quick UI validation.
          </p>
          <div className="grid gap-3 md:grid-cols-2">
            <label className="grid gap-2 text-sm font-semibold text-black">
              Pipeline
              <select
                className="w-full rounded-2xl border border-black/15 bg-white px-4 py-3 text-sm"
                value={pipeline}
                onChange={(event) => setPipeline(event.target.value as "fast" | "real")}
              >
                <option value="real">Real (GVHMR + Isaac Lab PPO, GPU)</option>
                <option value="fast">Fast (placeholder + synthetic)</option>
              </select>
            </label>
            <label className="grid gap-2 text-sm font-semibold text-black">
              Auto-train after XGen
              <select
                className="w-full rounded-2xl border border-black/15 bg-white px-4 py-3 text-sm"
                value={autoTrain ? "yes" : "no"}
                onChange={(event) => setAutoTrain(event.target.value === "yes")}
              >
                <option value="yes">Yes</option>
                <option value="no">No</option>
              </select>
            </label>
          </div>
          {pipeline === "real" ? (
            <label className="flex items-center gap-2 text-sm text-black/70">
              <input
                type="checkbox"
                checked={gvhmrStaticCam}
                onChange={(event) => setGvhmrStaticCam(event.target.checked)}
              />
              Static camera (GVHMR recommended for phone videos)
            </label>
          ) : null}
          {pipeline === "real" ? (
            <p className="text-xs text-black/60">
              GVHMR requires licensed SMPL-X model files on the GPU PC. If pose extraction falls back, open the XGen job
              logs and follow <code>docs/GVHMR.md</code>.
            </p>
          ) : null}
          <Button onClick={handleRun}>Generate dataset and policy</Button>
          {status ? <p className="text-sm text-emerald-700">{status}</p> : null}
          {error ? <p className="text-sm text-rose-700">{error}</p> : null}
          {jobId ? (
            <Link href={`/jobs/${jobId}`} className="text-sm font-semibold text-black underline">
              View job {jobId}
            </Link>
          ) : null}
        </Card>

        <details className="rounded-3xl border border-black/10 bg-white p-6">
          <summary className="cursor-pointer text-sm font-semibold text-black">Advanced settings (optional)</summary>
          <div className="mt-6 grid gap-6">
            <Card className="space-y-4">
              <div className="flex items-center justify-between">
                <h2 className="text-xl font-semibold text-black">Project</h2>
                <Badge label="Required" tone="amber" />
              </div>
              <input
                type="text"
                value={projectName}
                onChange={(event) => setProjectName(event.target.value)}
                className="w-full rounded-2xl border border-black/15 bg-white px-4 py-3 text-sm"
              />
            </Card>

            <Card className="space-y-4">
              <h2 className="text-xl font-semibold text-black">Robot</h2>
              <select
                className="w-full rounded-2xl border border-black/15 bg-white px-4 py-3 text-sm"
                value={robotModel}
                onChange={(event) => setRobotModel(event.target.value)}
              >
                <option value="unitree-g1">Unitree G1 template</option>
                <option value="atlas">Atlas template</option>
                <option value="default-humanoid">Default humanoid</option>
              </select>
            </Card>

            <Card className="space-y-4">
              <h2 className="text-xl font-semibold text-black">Object</h2>
              <div className="grid gap-3 md:grid-cols-2">
                <select
                  className="w-full rounded-2xl border border-black/15 bg-white px-4 py-3 text-sm"
                  value={objectId}
                  onChange={(event) => setObjectId(event.target.value)}
                >
                  <option value="cargo_box">Cargo crate (default)</option>
                  <option value="basketball">Basketball</option>
                  <option value="badminton_shuttle">Badminton shuttle</option>
                </select>
                <input
                  type="text"
                  placeholder="Or paste mesh URL (not wired yet)"
                  className="w-full rounded-2xl border border-black/15 bg-white px-4 py-3 text-sm"
                />
              </div>
              <div className="grid gap-3 md:grid-cols-3">
                <input
                  type="number"
                  value={objectPose.x}
                  onChange={(event) => setObjectPose({ ...objectPose, x: event.target.value })}
                  placeholder="Object X"
                  className="w-full rounded-2xl border border-black/15 bg-white px-4 py-3 text-sm"
                />
                <input
                  type="number"
                  value={objectPose.y}
                  onChange={(event) => setObjectPose({ ...objectPose, y: event.target.value })}
                  placeholder="Object Y"
                  className="w-full rounded-2xl border border-black/15 bg-white px-4 py-3 text-sm"
                />
                <input
                  type="number"
                  value={objectPose.z}
                  onChange={(event) => setObjectPose({ ...objectPose, z: event.target.value })}
                  placeholder="Object Z"
                  className="w-full rounded-2xl border border-black/15 bg-white px-4 py-3 text-sm"
                />
                <input
                  type="number"
                  value={objectPose.roll}
                  onChange={(event) => setObjectPose({ ...objectPose, roll: event.target.value })}
                  placeholder="Roll"
                  className="w-full rounded-2xl border border-black/15 bg-white px-4 py-3 text-sm"
                />
                <input
                  type="number"
                  value={objectPose.pitch}
                  onChange={(event) => setObjectPose({ ...objectPose, pitch: event.target.value })}
                  placeholder="Pitch"
                  className="w-full rounded-2xl border border-black/15 bg-white px-4 py-3 text-sm"
                />
                <input
                  type="number"
                  value={objectPose.yaw}
                  onChange={(event) => setObjectPose({ ...objectPose, yaw: event.target.value })}
                  placeholder="Yaw"
                  className="w-full rounded-2xl border border-black/15 bg-white px-4 py-3 text-sm"
                />
              </div>
            </Card>

            <Card className="space-y-4">
              <h2 className="text-xl font-semibold text-black">Contact annotation</h2>
              <div className="grid gap-3 md:grid-cols-2">
                <input
                  type="number"
                  value={tsStart}
                  onChange={(event) => setTsStart(event.target.value)}
                  placeholder="Contact start (ts)"
                  className="w-full rounded-2xl border border-black/15 bg-white px-4 py-3 text-sm"
                />
                <input
                  type="number"
                  value={tsEnd}
                  onChange={(event) => setTsEnd(event.target.value)}
                  placeholder="Contact end (te)"
                  className="w-full rounded-2xl border border-black/15 bg-white px-4 py-3 text-sm"
                />
              </div>
              <div className="grid gap-3 md:grid-cols-2">
                <select
                  className="w-full rounded-2xl border border-black/15 bg-white px-4 py-3 text-sm"
                  value={anchorType}
                  onChange={(event) => setAnchorType(event.target.value)}
                >
                  <option value="palms_midpoint">Anchor: two palms midpoint</option>
                  <option value="single_body_part">Anchor: single body part</option>
                </select>
                <input
                  type="text"
                  value={keyBodies}
                  onChange={(event) => setKeyBodies(event.target.value)}
                  placeholder="Key bodies (comma separated)"
                  className="w-full rounded-2xl border border-black/15 bg-white px-4 py-3 text-sm"
                />
              </div>
            </Card>
          </div>
        </details>
      </div>
    </div>
  );
}
