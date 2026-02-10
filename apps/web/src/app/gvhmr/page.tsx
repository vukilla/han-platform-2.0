"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Card } from "@/components/ui/Card";
import { createDemo, createProject, getDemoUploadUrl, listProjects, login, runXgen } from "@/lib/api";
import { getToken, setToken } from "@/lib/auth";

export default function GVHMRPage() {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [localPreviewUrl, setLocalPreviewUrl] = useState<string | null>(null);
  const [gvhmrStaticCam, setGvhmrStaticCam] = useState(true);
  const [gpuReady, setGpuReady] = useState<boolean | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
    fetch(`${apiUrl}/ops/workers?timeout=1.0`)
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => setGpuReady(Boolean(data?.has_gpu_queue)))
      .catch(() => setGpuReady(null));
  }, []);

  useEffect(() => {
    if (!file) {
      setLocalPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setLocalPreviewUrl(url);
    return () => {
      URL.revokeObjectURL(url);
    };
  }, [file]);

  async function ensureLoggedIn() {
    if (getToken()) return;
    const resp = await login("demo@humanx.local", "Demo");
    setToken(resp.token);
  }

  async function uploadToPresignedUrl(uploadUrl: string, uploadFile: File) {
    const response = await fetch(uploadUrl, {
      method: "PUT",
      body: uploadFile,
      headers: {
        "Content-Type": uploadFile.type || "video/mp4",
      },
    });
    if (!response.ok) {
      throw new Error("Upload failed");
    }
  }

  async function handleRun() {
    setError(null);
    setStatus(null);
    if (!file) {
      setError("Select a video file first.");
      return;
    }
    if (gpuReady === false) {
      setError("GPU worker is offline. Start the Windows GPU worker first, then try again.");
      return;
    }
    try {
      await ensureLoggedIn();

      setStatus("Finding project...");
      const projects = await listProjects();
      const existing = projects.find((p) => p.name === "GVHMR Demos");
      const project = existing ?? (await createProject("GVHMR Demos", "Pose recovery from monocular phone video"));

      setStatus("Creating demo record...");
      const demo = await createDemo(project.id, "human", "none");

      setStatus("Requesting upload URL...");
      const upload = await getDemoUploadUrl(demo.id);

      setStatus("Uploading video...");
      await uploadToPresignedUrl(upload.upload_url, file);

      setStatus("Starting GVHMR...");
      const job = await runXgen(demo.id, {
        requires_gpu: true,
        only_pose: true,
        pose_estimator: "gvhmr",
        gvhmr_static_cam: Boolean(gvhmrStaticCam),
        // Keep fallback enabled so the platform can still complete even if licensed SMPL-X assets are missing.
        // Once GVHMR is fully configured, set `fail_on_pose_error=true` to hard-fail instead.
        fail_on_pose_error: false,
      });

      setStatus("Redirecting...");
      router.push(`/jobs/${job.id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to run GVHMR");
      setStatus(null);
    }
  }

  return (
    <div className="mx-auto max-w-2xl space-y-8">
      <section className="space-y-3">
        <p className="section-eyebrow">GVHMR</p>
        <h1 className="text-3xl font-semibold text-black">Upload a video and recover 3D motion</h1>
        <p className="text-sm text-black/70">
          This runs GVHMR pose extraction on the Windows GPU worker and shows a side-by-side preview: original video plus a
          3D skeleton render.
        </p>
      </section>

      <Card className="space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h2 className="text-xl font-semibold text-black">Upload</h2>
          <Badge
            label={gpuReady === true ? "GPU worker connected" : gpuReady === false ? "GPU worker offline" : "GPU status unknown"}
            tone={gpuReady === true ? "emerald" : gpuReady === false ? "rose" : "amber"}
          />
        </div>
        <input
          type="file"
          accept="video/*"
          onChange={(event) => setFile(event.target.files?.[0] ?? null)}
          className="w-full rounded-2xl border border-black/15 bg-white px-4 py-4 text-sm"
        />
        {localPreviewUrl ? (
          <div className="space-y-2">
            <p className="text-sm font-semibold text-black/60">Selected video</p>
            <video className="w-full rounded-2xl border border-black/10 bg-black" controls playsInline src={localPreviewUrl} />
          </div>
        ) : null}
        <label className="flex items-center gap-2 text-sm text-black/70">
          <input type="checkbox" checked={gvhmrStaticCam} onChange={(event) => setGvhmrStaticCam(event.target.checked)} />
          Static camera (recommended)
        </label>
        <Button onClick={handleRun} disabled={!file}>
          Run GVHMR
        </Button>
        {status ? <p className="text-sm text-emerald-700">{status}</p> : null}
        {error ? <p className="text-sm text-rose-700">{error}</p> : null}
      </Card>
    </div>
  );
}
