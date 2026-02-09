# HumanX Data Factory Build Playbook (Code-First)

This playbook is a consolidated, end-to-end process to build the HumanX-style data factory with XGen and XMimic. It is designed to be executed by a Codex agent in dependency order, using Linear as the execution driver. Figma is optional.

## Product decisions (locked)
- Incentives chain: Solana first; off-chain points for MVP; EVM later via adapter.
- Data collection: video-only first; teleop reserved for Phase 2.
- First public task: cargo pickup.
- Storage: local MinIO for dev; prod TBD.
- Compute: MacBook runs web+api+db+queue+minio; RTX 5090 PC runs GPU worker.
- Tracking: MLflow local (no W&B yet).
- Notifications: in-app/logs only for MVP.

## One-sentence goal
Build a web app that turns human videos into standardized, augmented humanoid interaction datasets (XGen), trains deployable policies (XMimic), and publishes datasets + checkpoints + quality scores + attribution.

## Execution system
- Use Linear as the source of truth.
- Always pull the next unblocked issue.
- If blocked by assets or infrastructure, leave a BLOCKED comment and pick the next unblocked issue.

## Architecture
- Frontend: Next.js (TypeScript), pre-signed direct uploads, job polling.
- Backend: FastAPI, Postgres, Redis queue.
- Workers: CPU worker for validation/packaging, GPU worker for pose + training.
- Storage: MinIO (S3-compatible), later S3/R2/IPFS for prod.
- ML: XGen + XMimic pipelines, MLflow tracking.

## Milestones
### 1. Monorepo + local infra
- apps/web, apps/api, services/xgen, services/xmimic, infra/docker-compose.yml
- Done when docker compose up runs web+api+db+redis+minio.

### 2. DB + migrations
- Implement schema for users, projects, demos, jobs, datasets, policies, evals, rewards.
- Done when CRUD works for projects and demos.

### 3. Upload pipeline
- API issues pre-signed URL, web uploads directly, demo status updated.
- Done when demo row points to playable video URI.

### 4. Annotation UI
- Capture ts/te, anchor type, key bodies.
- Done when annotations persist and load.

### 5. Job system
- Celery + Redis wiring, job state machine, logs, retries.
- Done when dummy job updates stages in DB and UI shows progress.

### 6. XGen MVP
- Manual SMPL or placeholder motion.
- Segment, anchors, contact synth, non-contact sim stub, augment, export.
- Done when dataset has >=10 clips and previews render.

### 7. XGen full adapters
- Pose estimator adapter, retargeter adapter, object init estimator (optional).
- Done when real phone video produces dataset without manual SMPL.

### 8. XMimic tiny-scale
- Cargo pickup task, teacher PPO + student PPO + BC distillation.
- NEP and MoCap modes with dropout.
- Done when checkpoint + SR/GSR metrics exist.

### 9. Eval reporting UI
- Metric cards, report download, checkpoint download.
- Done when eval report accessible end-to-end.

### 10. Quality + incentives
- Automated checks, validator review queue, points ledger.
- Done when accepted demos grant points.

### 11. Hardening
- Rate limits, antivirus scanning, structured logs, observability.
- Done when repeated uploads + retries are stable.

## XGen module boundaries
- pose/
- retarget/
- interaction/ (segment, anchors, object_init, contact_synth, force_closure_refine, noncontact_sim, stitch)
- augment/ (mesh, trajectory, noncontact)
- export/ (clip writer with contact graph)

## XMimic essentials
- Teacher-student PPO + BC distillation.
- Unified imitation reward (body, object, relative, contact, regularization).
- DI/IT/DR modules.
- NEP and MoCap observation modes.

## Dataset output format
Each clip exports clip.npz with:
- robot_qpos, robot_qvel, root_pose, object_pose, contact_graph, phase, metadata.json.

## Local dev assets
- Sample videos stored in assets/sample_videos/ (placeholder mp4s acceptable).
- Robot asset: simulator default humanoid for MVP; Unitree G1 later.
- Object mesh: procedural cargo box (simple cuboid).

## GPU worker (RTX 5090 PC)
- Run worker in native Python env (no Docker for MVP).
- Use .env.worker.example for env vars.
- Set REDIS_URL and S3_ENDPOINT to MacBook LAN IP.
- Ensure Mac firewall allows inbound 6379 and 9000 (and 9001 if using MinIO console).

## Outreach email (Yinhuai Wang)
- Introduce Humanoid Network, describe overlap with HumanX pipeline, mention Noshaba Cheema (MPII/DFKI), ask about collaboration and code release timing, request short call.

## Automation
- Run continuously on next unblocked Linear issue.
- Update docs/STATUS.md after each run.
