# han-platform

Humanoid Network platform scaffolding: Next.js web app, FastAPI backend, XGen/XMimic services, and local infra.

## Quick start
1. `docker compose -f infra/docker-compose.yml up --build`
2. Run migrations:
   - `docker compose -f infra/docker-compose.yml exec api alembic upgrade head`
3. Open:
   - Web: `http://localhost:3000`
   - API health: `http://localhost:8000/health`

Fastest end-to-end (Mac control-plane + Windows GPU worker):
1. On Mac: `./scripts/mac/run_full_e2e.sh`
2. When it prints your Mac IP, run the printed Windows command:
   - `powershell -ExecutionPolicy Bypass -File .\\scripts\\windows\\one_click_gpu_worker.ps1 -MacIp <MAC_IP> -IsaacSimPath C:\\isaacsim`

Fastest REAL end-to-end (GVHMR + Isaac Lab PPO, requires extra GVHMR checkpoints on Windows):
1. On Mac: `./scripts/mac/run_full_e2e_real.sh`
2. When it prints your Mac IP, run the printed Windows command:
   - `powershell -ExecutionPolicy Bypass -File .\\scripts\\windows\\one_click_gpu_worker_real.ps1 -MacIp <MAC_IP> -IsaacSimPath C:\\isaacsim -SetupGVHMR -DownloadLightCheckpoints`

What to test (golden path):
- `docs/WHAT_TO_TEST.md`

## GPU PC Setup (Windows, Isaac Sim + Isaac Lab)
The current supported path (no WSL/virtualization required) is:
- Windows gaming PC runs Isaac Sim + Isaac Lab (GPU compute only).
- Mac runs the control-plane stack via Docker Compose.

Start here:
- `docs/ISAACLAB_WINDOWS_SETUP.md`
- `docs/GPU_PC_SETUP.md`
- `docs/FIRST_PROMPT_GPU_PC.md`

## Dual Worker Setup (Pegasus + Windows fallback)
If you want motion recovery available while your home gaming PC is off, add a Pegasus worker and keep Windows as fallback:
- `docs/PEGASUS_DUAL_WORKER_SETUP.md`
- `scripts/mac/start_pegasus_worker_ssh.sh`
- `scripts/mac/stop_pegasus_worker_ssh.sh`
- `scripts/mac/tail_pegasus_worker_logs_ssh.sh`

Legacy Isaac Gym/PhysHOI scripts remain under `scripts/gpu/` but assume Linux and are not the recommended path.

## GitHub Notes
- `.gitignore` excludes `external/`, `tmp/`, `.env`, venvs, node_modules, and large checkpoints.
- Keep the reference paper PDF local (recommended path: `docs/references/paper.pdf`). Do not commit it.

## Structure
- `apps/web` Next.js UI
- `apps/api` FastAPI API
- `services/xgen` XGen pipeline
- `services/xmimic` XMimic pipeline
- `infra/docker-compose.yml` local stack
- `docs/notion` Notion page drafts
- `docs/figma` Figma frame checklist
- `docs/STATUS.md` progress tracker
