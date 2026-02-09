# han-platform-2.0

HumanX Data Factory scaffolding: Next.js web app, FastAPI backend, XGen/XMimic services, and local infra.

## Quick start
1. `docker compose -f infra/docker-compose.yml up --build`
2. Run migrations:
   - `docker compose -f infra/docker-compose.yml exec api alembic upgrade head`
3. Open:
   - Web: `http://localhost:3000`
   - API health: `http://localhost:8000/health`

## GPU PC Setup (PhysHOI/GVHMR/Isaac Gym)
External repos + checkpoints are not committed. On the gaming PC:
1. Clone this repo.
2. From repo root run:
   - `scripts/gpu/bootstrap.sh`
3. If Isaac Gym is not installed yet, follow the prompt and run:
   - `scripts/gpu/install_isaacgym.sh`

Codex agent instructions for the GPU PC are in `AGENTS.md`.

## GitHub Notes
- `.gitignore` excludes `external/`, `tmp/`, `.env`, venvs, node_modules, and large checkpoints.
- Keep the HumanX PDF local (recommended path: `docs/references/HumanX.pdf`). Do not commit it.

## Structure
- `apps/web` Next.js UI
- `apps/api` FastAPI API
- `services/xgen` XGen pipeline
- `services/xmimic` XMimic pipeline
- `infra/docker-compose.yml` local stack
- `docs/notion` Notion page drafts
- `docs/figma` Figma frame checklist
- `docs/STATUS.md` progress tracker
