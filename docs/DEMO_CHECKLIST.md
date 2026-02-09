# Demo & Launch Readiness Checklist

## Golden path (local)
1. `docker compose -f infra/docker-compose.yml up`
2. Open web app at `http://localhost:3000`
3. Log in via `/auth` (any email)
4. Create demo in wizard, upload sample video, annotate, run XGen
5. Open job page and verify stage updates
6. Open dataset detail and preview clip/download
7. Start XMimic run (stub)
8. Open eval report with an eval id
9. View rewards + admin quality review pages

## Artifacts
- Dataset `clip.npz` + metadata JSON
- Logs in MinIO `logs/xgen` and `logs/xmimic`

## Known limitations
- ML training is stubbed; replace with real Isaac Gym training.
- Quality review approvals are UI-only (no persistence yet).
