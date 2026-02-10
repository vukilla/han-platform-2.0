# Han Platform 2.0 Status

## Current state (2026-02-10)
- Laptop local stack is running: `docker compose up` (Postgres/Redis/MinIO/API/worker/web)
- API health check passes: `GET http://localhost:8000/health`
- Web dev server loads (fixed linux native deps by isolating container `node_modules` volume)
- Large external artifacts live inside the repo under `external/humanoid-projects/` (no separate `~/humanoid-projects` required on laptop)
- Browser upload UX is reliable (no MinIO CORS required):
  - `POST /demos/{id}/upload` streams the video to MinIO
  - `/gvhmr` and `/demos/new` use the API upload path by default
- `/gvhmr` supports a "Quick preview" mode (trim to the first 12 seconds) to keep GVHMR demos responsive
- Windows GPU worker (Isaac Sim + Isaac Lab) can now run:
  - Real XGen pose extraction via GVHMR (video -> SMPL-X NPZ), uploaded to MinIO
  - Real PPO teacher checkpoint generation via Isaac Lab (Franka cube lift as the current "cargo pickup" baseline), uploaded to MinIO
  - `scripts/smoke_e2e_with_gpu_real.sh` runs XGen(GVHMR) + XMimic(IsaacLab PPO) end-to-end
- PhysHOI baseline inference hook exists as an optional backend, but is Linux-first (not Windows)
- Added golden-path runbook + UI affordances for testing:
  - `docs/WHAT_TO_TEST.md`
  - `/datasets` index page
  - `/policies` index page
  - `/xmimic/<job_id>` progress page
  - XGen job page now exposes demo id, dataset id, and presigned logs download
  - XGen job API presigns GVHMR pose artifacts in `params_json` (pose NPZ + meta + log) so the UI can link them
- Quality approval now mints MVP points (reward events) for approved demos/datasets
- Paper-fidelity work continues on laptop: Table IV obs/reward schema parity and CPU-side contact refinement improvements
  - Relative motion reward parity tightened (Eq. 10 uses mean squared L2 norm).
  - Contact refinement upgraded: optional CPU robot IK refinement for contact frames (pre-physics).
  - Simplified web wizard: minimal upload + single "Generate dataset and policy" button. All robot/object/annotation knobs moved under "Advanced settings (optional)".
  - Added Mac scripts to start the Windows GPU worker via SSH and run the REAL E2E in one command:
    - `scripts/mac/start_windows_gpu_worker_ssh.sh`
    - `scripts/mac/run_full_e2e_real_ssh.sh`

## Current state (2026-02-04)
- Monorepo scaffolded with apps/web, apps/api, services/xgen, services/xmimic, infra/docker-compose.yml
- FastAPI skeleton, DB models, Alembic migration, Celery worker stubs, S3 presigned helpers
- Auth login now returns JWT; /me and create project/demo use bearer auth
- Next.js UI shell with key pages and visual system
- Linear project and epics/stories created
- Notion/Figma draft docs stored locally
## Product decisions (2026-02-04)
- Incentives chain: Solana first (off-chain points for MVP), EVM later via adapter interface
- Data collection: video-only first; teleop later (reserved interfaces)
- First public task: cargo pickup
- Storage: local MinIO (S3-compatible) for dev, prod TBD
- Compute: MacBook Pro runs web+api+db+queue+minio; RTX 5090 PC runs GPU worker
- Experiment tracking: MLflow local for MVP
- Notifications: in-app/logs only for MVP
## Assets (2026-02-04)
- Sample videos created in assets/sample_videos/ (placeholder mp4s for golden path)
- Robot asset: simulator default humanoid for MVP (Unitree G1 later)
- Object mesh: procedural cargo box (simple cuboid)

## Next focus
1. Wire API endpoints to real DB + storage interactions
2. Add dataset/policy creation flow in workers
3. Integrate frontend with API
4. Implement XGen MVP path (manual inputs)
5. Implement XMimic tiny-scale training loop

## Recent updates
- Added Project update/delete endpoints and CRUD helpers (02.02).
- Added Demo list/update/delete endpoints and CRUD helpers (02.03).
- Added XGen job list endpoint and auth guard for run (02.04).
- Added dataset list endpoint, XMimic job list, and eval list endpoints (02.05-02.07).
- Wired dataset detail UI to API with clip list + preview player + download action (03.03).
- Wired training launch UI to datasets + XMimic run endpoint with NEP/MoCap toggle (03.04).
- Wired eval report UI to fetch eval run metrics and download report/videos (03.05).
- Wired rewards UI to /rewards/me and summary totals (03.06).
- Added worker log uploads to object storage for XGen/XMimic jobs (04.03).
- Celery/Redis worker wiring and job state updates in place (04.01-04.02).
- Added job idempotency keys + retry policy in workers (04.04).
- Added XGen clip schema + export helpers for NPZ/metadata (05.01).
- Added phase segmenter for ts/te contact labeling (05.02).
- Added anchor definitions for palms midpoint vs single body part (05.03).
- Added contact synthesis helper to propagate anchor-object transforms (05.04).
- Added MVP force-closure refine smoothing pass (05.05).
- Added MVP non-contact simulation placeholder (05.06).
- Added augmentation helpers for mesh scaling and trajectory transforms (05.07).
- Added MVP preview renderer stub for clip previews (05.08).
- Added XMimic environment scaffolding with CargoPickupEnv stub (06.01).
- Added unified reward term scaffolding (body/object/relative/contact/reg) (06.02).
- Added teacher PPO training loop stub (06.03).
- Added student distillation loop stub with BC loss (06.04).
- Added NEP/MoCap observation builders with dropout (06.05).
- Added DI/IT/DR generalization module stubs (06.06).
- Added eval metric calculators for SR/GSR, tracking errors, and generalization sampler (07.01-07.03).
- Added automated quality check scaffolding for demos/clips (08.01).
- Added quality score persistence helper with breakdown storage (08.02).
- Added admin quality review UI (lookup + approve/reject controls) (08.03).
- Local docker-compose stack confirmed for dev (10.01).
- Added GitHub Actions CI for web lint + API compile check (10.02).
- Added GPU worker deployment notes (10.03).
- Added basic API rate limiting middleware + security notes (11.00).
- Added demo readiness checklist for golden-path walkthrough (12.00).
- Added MediaPipe pose estimation adapter for real video pose extraction (13.01).
- Added IK-based retargeting for upper-body joints with root alignment (13.02).
- Added object pose estimator via tracking + manual pose inputs in wizard (13.03).
- Added MuJoCo-based noncontact simulation with ballistic fallback (13.04).
- Added constraint-based force-closure refine with anchor offsets (13.05).
- Added contact graph label generator based on body/object distances (13.06).
- Added mesh substitution augmentation via trimesh (13.07).
- Added dataset manifest + versioning helper for packaging (13.08).
- Added Isaac Gym cargo pickup env scaffolding with real sim step/reset (14.01).
- Added PPO-based teacher training loop with privileged obs adapter (14.02).
- Added student PPO training with BC distillation support (14.03).
- Added NEP/MoCap observation builder for Isaac root states (14.04).
- Integrated DI/IT/DR into PPO training loop (14.05).
- Added training config helpers and checkpoint save/load (14.06).
- Added eval report generator for metrics and slices (14.07).
- Added policy registry list endpoint + manifest helper (14.08).
- Added quality review persistence + API endpoint (15.02).
- Wired admin review UI to approve/reject API (15.03).
- Added FFmpeg-based preview rendering with overlay text (15.01).
- Integrated quality scoring into job pipeline (15.04).
- Added MLflow logging hook for training metrics (16.01).
- Added CUDA-based GPU worker Dockerfile (16.02).
- Added Celery CPU/GPU queue routing, backpressure checks, and worker concurrency defaults (16.03).
- Added health checks, failed-jobs ops endpoint, alert webhook hook, and admin failed-jobs UI (16.04).
- Added GVHMR pose estimator wrapper producing SMPL-X NPZ outputs (18.01).
- Added SMPL → SMPL-X conversion utility with fitting and docs (18.02).
- Cloned PhysHOI base repo and documented build/inference steps (17.01).
- Added SMPL-X → PhysHOI motion conversion utility (18.03).
- Implemented exponential XMimic reward terms (body/object/rel/contact) with scales + doc (21.01).
- Added inverse-dynamics contact inference helpers with Pinocchio (21.02).
- Added PhysHOI training adapter to map YAML → PPOConfig and distill student checkpoints (21.03).
- Added SAM-3D object init estimator wrapper and docs (20.01).
- Added QP-based force-closure solver (cvxpy) with exports and docs (20.02).
- Added contact refinement pipeline (anchor propagation + force-closure + optional QP forces) (20.03).
- Added simulation-based augmentation sweeps for scale/velocity (20.04).
- Added PhysHOI → policy registry adapter with manifest upload (17.03).
- Added PhysHOI motion ingest → XGen clip converter (17.04).
- Added forward/backward non-contact sim helpers with damping control (19.02).
- Added Isaac Lab migration assessment doc (19.03).
- Added HumanX reward config loader + YAML defaults for gamma/lambda terms (21.04).
- Added PhysHOI motion export CLI + validation helpers (17.05).
- Added GPU preflight script for Isaac Gym/CUDA/Torch checks (17.06).
- Added PhysHOI motion packer CLI and YAML generator (17.07).
- Added CPU orchestration script + runbook for GVHMR → PhysHOI prep (17.08).
- Added SMPL-X resample + smoothing utility for control-rate alignment (18.05).
- Added retargeting validator for EEF errors, joint limits, and foot skating (18.04).
- Added observation pipeline with normalization + history buffers (19.06).
- Added Table IV mapping doc and enforced observation schemas (teacher vs student, NEP vs MoCap) with tests (23.01).
- Corrected Table IV student observation schema: student includes `pd_error` (matches Table IV) and updated docs/tests (23.05).
- Tightened relative-motion reward error metric to match Eq. 10 (mean squared L2 norm) + expanded reg/relative tests (23.02).
- Made `xmimic` top-level imports lazy so schema/tests work without torch installed (23.03).
- Added optional CPU robot IK refinement into XGen contact refinement pipeline (23.04).
- Upgraded CPU contact refinement: multi-tip SQP-style IK solve with posture regularization (23.06).
- Reward parity: when deriving Eq. 10 relative vectors, prefer `key_body_pos` over `body_pos` and added regression test (23.07).
- Tightened unified reward parity: derived relative vectors `u_t` from key bodies + added missing `r_reg` terms with tests (23.02).
- Upgraded XGen contact refinement with CPU IK refinement hook for joint-level alignment (23.03).
- Added phase transition smoothing utility for body/object/vel series (20.05).
- Extended dataset clip schema with root/object velocities + metadata guidance (20.06).
- Added interaction termination helper with contact-relative error (21.06).
- Added external push scheduling hooks + DR helpers in training loop (21.07).
- Added multi-skill sampling helper for student training (21.08).
- Added AMP discriminator module + reward hook with config support (21.09).
- Added PD control interface module + Isaac Gym PD action hook (19.04).
- Added robot spec loader + contact/key body definitions config (19.05).
- Added v0 repo-local humanoid/object URDFs + spec/URDF validator script (19.05).
- Added PhysHOI inference wrapper script to standardize runs (17.09).
- Added PhysHOI checkpoint installer script for the Google Drive zip layout.
- Made `pinocchio` optional for `xmimic` imports; inference functions now raise a clear ImportError when called without it.
- Fixed dataclass defaults in XMimic teacher/student configs (avoid mutable defaults on Python 3.12).
- Added deployment runtime loop scaffold with 100Hz policy / 1000Hz PD (22.01).
- Added MoCap dropout + transform scaffolding for runtime parity (22.02).
- Added runtime safety scaffold (torque limits, joint limits, fall detect, watchdog) (22.03).
- Platform plumbing: XGen jobs now create `datasets` + `dataset_clips` records and upload real artifacts (`.npz` + `dataset.zip`) to MinIO/S3, and dataset clip URIs are returned as presigned URLs (24.01).
- Platform plumbing: XMimic jobs now register a checkpoint artifact and create `policies` + `eval_runs` records on completion (24.02).
- Platform ops: added `/ops/workers` to verify Celery CPU/GPU workers, added Mac `scripts/mac/run_full_e2e.sh`, added Windows `scripts/windows/one_click_gpu_worker.ps1`, fixed GPU smoke-script parsing, and corrected Table IV teacher schema (no `skill_label`) + relative-vector convention (24.03).

## Open items needed
- Replace synthetic artifact generation with real XGen outputs (GVHMR/PhysHOI/Isaac Lab), including clip-aligned preview renders.
- Wire XMimic training loop to Isaac Lab environments so checkpoints and eval metrics (SR/GSR/Eo/Eh) are non-synthetic.
