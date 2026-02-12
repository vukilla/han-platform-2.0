"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import {
  createDemo,
  createProject,
  getGvhmrSmplxModelStatus,
  listProjects,
  runXgen,
  uploadDemoVideo,
  uploadGvhmrSmplxModel,
  GVHMRSmplxModelStatus,
} from "@/lib/api";
import { clearToken, getToken } from "@/lib/auth";

export default function StudioPage() {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [localPreviewUrl, setLocalPreviewUrl] = useState<string | null>(null);
  const [poseReady, setPoseReady] = useState<boolean | null>(null);
  const [smplxStatus, setSmplxStatus] = useState<GVHMRSmplxModelStatus | null>(null);
  const [smplxFile, setSmplxFile] = useState<File | null>(null);
  const [smplxUploadStatus, setSmplxUploadStatus] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let timer: NodeJS.Timeout | null = null;
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    const poll = () => {
      fetch(`${apiUrl}/ops/workers?timeout=1.0`)
        .then((res) => (res.ok ? res.json() : null))
        .then((data) => setPoseReady(Boolean(data?.has_pose_queue)))
        .catch(() => setPoseReady(null));
    };
    poll();
    timer = setInterval(poll, 3000);
    return () => {
      if (timer) clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    return () => {
      if (localPreviewUrl) {
        URL.revokeObjectURL(localPreviewUrl);
      }
    };
  }, [localPreviewUrl]);

  async function ensureLoggedIn() {
    if (getToken()) return;
    clearToken();
    router.push("/auth");
    throw new Error("Please sign in with Privy first.");
  }

  useEffect(() => {
    let cancelled = false;
    void (async () => {
      try {
        if (!getToken()) {
          clearToken();
          router.push("/auth");
          throw new Error("Please sign in with Privy first.");
        }
        const status = await getGvhmrSmplxModelStatus();
        if (!cancelled) setSmplxStatus(status);
      } catch (err) {
        if (cancelled) return;
        const message = err instanceof Error ? err.message : "Failed to load SMPL-X status";
        if (message.toLowerCase().includes("session expired")) {
          setError(message);
          clearToken();
          router.push("/auth");
          return;
        }
        setSmplxStatus(null);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [router]);

  async function handleUploadSmplx() {
    setSmplxUploadStatus(null);
    setError(null);
    if (!smplxFile) {
      setError("Select `SMPLX_NEUTRAL.npz` first.");
      return;
    }
    try {
      await ensureLoggedIn();
      setSmplxUploadStatus("Uploading SMPL-X model...");
      const status = await uploadGvhmrSmplxModel(smplxFile);
      setSmplxStatus(status);
      setSmplxUploadStatus("Uploaded. You can run motion recovery now.");
    } catch (err) {
      setSmplxUploadStatus(null);
      setError(err instanceof Error ? err.message : "Failed to upload SMPL-X model");
    }
  }

  async function handleRun() {
    setError(null);
    setStatus(null);
    if (!file) {
      setError("Select a video file first.");
      return;
    }
    if (poseReady === false) {
      setError("Pose worker is offline. Start it, then try again.");
      return;
    }
    if (smplxStatus?.exists === false) {
      setError("One-time setup required: upload `SMPLX_NEUTRAL.npz` first.");
      return;
    }

    try {
      await ensureLoggedIn();

      setStatus("Finding project...");
      const projects = await listProjects();
      const existing = projects.find((p) => p.name === "Studio");
      const project = existing ?? (await createProject("Studio", "Phone video -> 3D motion preview"));

      setStatus("Creating run...");
      const demo = await createDemo(project.id, "human", "none");

      setStatus("Uploading video...");
      await uploadDemoVideo(demo.id, file);

      setStatus("Starting motion recovery...");
      const job = await runXgen(demo.id, {
        requires_gpu: true,
        only_pose: true,
        pose_estimator: "gvhmr",
        gvhmr_static_cam: true,
        fail_on_pose_error: true,
      });

      setStatus("Opening preview...");
      router.push(`/jobs/${job.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to start motion recovery");
      setStatus(null);
    }
  }

  return (
    <div className="mx-auto max-w-3xl space-y-8">
      <section className="space-y-3">
        <p className="section-eyebrow">Studio</p>
        <h1 className="text-3xl font-semibold text-black">Upload one video. Get a 3D motion preview.</h1>
        <p className="max-w-2xl text-sm text-black/70">
          This runs world-grounded motion recovery on your pose worker (Pegasus or Windows) and renders a 3D preview you can compare side-by-side with the original.
        </p>
      </section>

      <Card className="space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h2 className="text-xl font-semibold text-black">1. Upload</h2>
          <div className="flex flex-wrap items-center gap-3">
            <Badge
              label={
                poseReady === true ? "Pose worker connected" : poseReady === false ? "Pose worker offline" : "Worker status unknown"
              }
              tone={poseReady === true ? "emerald" : poseReady === false ? "rose" : "amber"}
            />
            <Badge
              label={
                smplxStatus?.exists === true
                  ? "SMPL-X model uploaded"
                  : smplxStatus?.exists === false
                    ? "SMPL-X model missing"
                    : "SMPL-X status unknown"
              }
              tone={smplxStatus?.exists === true ? "emerald" : smplxStatus?.exists === false ? "rose" : "amber"}
            />
          </div>
        </div>

        <input
          type="file"
          accept="video/*"
          onChange={(event) => {
            const nextFile = event.target.files?.[0] ?? null;
            setFile(nextFile);
            if (localPreviewUrl) URL.revokeObjectURL(localPreviewUrl);
            setLocalPreviewUrl(nextFile ? URL.createObjectURL(nextFile) : null);
          }}
          className="w-full rounded-2xl border border-black/15 bg-white px-4 py-4 text-sm"
        />

        {localPreviewUrl ? (
          <div className="space-y-2">
            <p className="text-sm font-semibold text-black/60">Selected video</p>
            <video className="w-full rounded-2xl border border-black/10 bg-black" controls playsInline src={localPreviewUrl} />
          </div>
        ) : null}
      </Card>

      {smplxStatus?.exists === false ? (
        <Card className="space-y-3">
          <h2 className="text-xl font-semibold text-black">One-time setup</h2>
          <p className="text-sm text-black/70">
            Upload the licensed SMPL-X model file <span className="font-mono">SMPLX_NEUTRAL.npz</span> to enable 3D preview rendering.
            This is stored in your object storage and reused for all runs.
          </p>
          <input
            type="file"
            accept=".npz"
            onChange={(event) => setSmplxFile(event.target.files?.[0] ?? null)}
            className="w-full rounded-2xl border border-black/15 bg-white px-4 py-3 text-sm"
          />
          <div className="flex flex-wrap items-center gap-3">
            <Button onClick={handleUploadSmplx} disabled={!smplxFile}>
              Upload SMPL-X model
            </Button>
            {smplxUploadStatus ? <span className="text-sm text-black/70">{smplxUploadStatus}</span> : null}
          </div>
          <p className="text-xs text-black/60">
            Download instructions: <span className="font-mono">docs/GVHMR.md</span>
          </p>
        </Card>
      ) : null}

      <Card className="space-y-4">
        <h2 className="text-xl font-semibold text-black">2. Run</h2>
        <div className="flex flex-wrap items-center gap-3">
          <Button onClick={handleRun} disabled={!file}>
            Run motion recovery
          </Button>
          {status ? <span className="text-sm text-black/70">{status}</span> : null}
        </div>
        {error ? <p className="text-sm text-rose-700">{error}</p> : null}
        <p className="text-xs text-black/60">
          You will be taken to a progress page showing the original video and the 3D preview side-by-side.
        </p>
      </Card>
    </div>
  );
}
