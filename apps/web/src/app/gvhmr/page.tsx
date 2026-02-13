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
  API_URL,
  runXgen,
  uploadDemoVideo,
  uploadGvhmrSmplxModel,
  GVHMRSmplxModelStatus,
} from "@/lib/api";
import { clearToken, getToken } from "@/lib/auth";

export default function GVHMRPage() {
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
    const poll = () => {
      fetch(`${API_URL}/ops/workers?timeout=1.0`)
        .then((res) => (res.ok ? res.json() : null))
        .then((data) => {
          const hasPoseQueue =
            Boolean(data?.has_pose_queue) ||
            Boolean(data?.has_pose_queue_pegasus) ||
            Boolean(data?.has_pose_queue_windows) ||
            Boolean(data?.has_pose_queue_legacy);
          if (typeof data?.has_pose_queue === "boolean" || hasPoseQueue) {
            setPoseReady(hasPoseQueue);
          } else {
            setPoseReady(false);
          }
        })
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
      setSmplxUploadStatus("Uploaded. New runs will produce a real 3D preview.");
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
      setError("Pose worker is offline. Start a pose worker (Pegasus or Windows), then try again.");
      return;
    }
    if (smplxStatus?.exists === false) {
      setError("Upload the licensed SMPL-X file `SMPLX_NEUTRAL.npz` first (one-time setup), then rerun.");
      return;
    }
    try {
      await ensureLoggedIn();

      setStatus("Finding project...");
      const projects = await listProjects();
      const existing = projects.find((p) => p.name === "Studio");
      const project = existing ?? (await createProject("Studio", "Phone video -> 3D motion preview"));

      setStatus("Creating demo record...");
      const demo = await createDemo(project.id, "human", "none");

      setStatus("Uploading video...");
      await uploadDemoVideo(demo.id, file);

      setStatus("Starting motion recovery...");
      const job = await runXgen(demo.id, {
        requires_gpu: true,
        only_pose: true,
        pose_estimator: "gvhmr",
        gvhmr_static_cam: true,
        gvhmr_max_seconds: 12,
        // On the GVHMR-only page, require the licensed SMPL-X model file and fail fast if it's missing.
        // The job page provides an inline uploader + requeue to recover from this.
        fail_on_pose_error: true,
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
        <p className="section-eyebrow">Motion Recovery</p>
        <h1 className="text-3xl font-semibold text-black">Upload a video and recover 3D motion</h1>
        <p className="text-sm text-black/70">
          This runs motion recovery on your pose worker (Pegasus or Windows) and shows a side-by-side preview: original video plus a 3D
          skeleton render. (Powered by GVHMR.)
        </p>
      </section>

      <Card className="space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <h2 className="text-xl font-semibold text-black">Upload</h2>
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

        {smplxStatus?.exists === false ? (
          <div className="space-y-2 rounded-2xl border border-black/10 bg-black/[0.02] p-4">
            <p className="text-sm font-semibold text-black">One-time setup: SMPL-X model file</p>
            <p className="text-sm text-black/70">
              To generate the real 3D skeleton preview, upload the licensed file <span className="font-mono">SMPLX_NEUTRAL.npz</span>.
              See <span className="font-mono">docs/GVHMR.md</span> for where to download it.
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
          </div>
        ) : null}

        <input
          type="file"
          accept="video/*"
          onChange={(event) => {
            const nextFile = event.target.files?.[0] ?? null;
            setFile(nextFile);
            if (localPreviewUrl) {
              URL.revokeObjectURL(localPreviewUrl);
            }
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
        <Button onClick={handleRun} disabled={!file || poseReady === false || smplxStatus?.exists === false}>
          Run motion recovery
        </Button>
        {status ? <p className="text-sm text-emerald-700">{status}</p> : null}
        {error ? <p className="text-sm text-rose-700">{error}</p> : null}
      </Card>
    </div>
  );
}
