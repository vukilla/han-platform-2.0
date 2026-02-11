# Han Platform Status

## Verified (2026-02-10)
- Mac control-plane starts cleanly:
  - `/Users/robertvukosa/Downloads/Python/han-platform/scripts/mac/control_plane_up.sh`
  - `GET http://localhost:8000/health` returns `status=ok` (db/redis/s3).
- Worker detection endpoint includes pose vs gpu:
  - `GET http://localhost:8000/ops/workers?timeout=2.0`
  - Returns `has_pose_queue` and `has_gpu_queue` plus heartbeat entries.
- Web UI boots and builds:
  - Dev UI: `http://localhost:3000`
  - Production build (sanity): `npm run build` in `apps/web` succeeds.
- Motion-recovery-only flow is available (simplest UX):
  - `http://localhost:3000/studio`
  - Upload video, click `Run motion recovery`, job page shows side-by-side:
    - Original video
    - GVHMR 3D skeleton preview (same resolution as the original)
- CLI smoke test (motion recovery) works:
  - `/Users/robertvukosa/Downloads/Python/han-platform/scripts/smoke_motion_recovery.sh`
- Navigation simplified:
  - Top nav: Dashboard, Studio, Rewards, Deploy (Coming soon)
  - `/` redirects to `/dashboard`

## GVHMR Prerequisite (Licensed)
Real GVHMR requires the SMPL-X model file `SMPLX_NEUTRAL.npz` (licensed, not bundled).

One-time setup:
- Upload via Web UI: `http://localhost:3000/studio` (preferred)
- Or API: `POST /admin/gvhmr/smplx-model` (multipart field `file`)

Details: `/Users/robertvukosa/Downloads/Python/han-platform/docs/GVHMR.md`

If missing, the platform:
- Marks pose as fallback (`pose_ok=false`)
- Shows a banner-style preview and a clear action to upload SMPL-X and rerun.

## Windows Worker Commands
- Start pose worker (GVHMR-only):
  - `scripts\\windows\\one_click_gpu_worker.ps1 -MacIp <MAC_LAN_IP> -IsaacSimPath C:\\isaacsim -Queues pose`
- Start GPU training worker (XMimic / PPO):
  - `scripts\\windows\\one_click_gpu_worker.ps1 -MacIp <MAC_LAN_IP> -IsaacSimPath C:\\isaacsim -Queues gpu`

## Not Yet Paper-Fidelity
- XGen stages beyond pose estimation (retarget/contact/noncontact/augment) are still placeholders.
- XMimic training defaults to a synthetic backend for fast end-to-end UI plumbing.
  - The "real" Isaac Lab PPO backend exists but is still experimental and may require additional iteration.
