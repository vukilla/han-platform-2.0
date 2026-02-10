# HumanX Paper Reconciliation (Han Platform 2.0)

Source of truth:
- HumanX paper PDF (keep local, do not commit). Recommended path: `docs/references/HumanX.pdf`

This document maps the HumanX paper components (XGen + XMimic + deployment) to concrete code in this repo, and flags what is:
- OK: implemented in code and exercised by at least a smoke test or runnable script
- PARTIAL: implemented as scaffolding or an approximation; needs correctness work to match the paper
- GPU-BLOCKED: implemented in code, but requires GPU PC deps (Windows-native Isaac Sim / Isaac Lab, plus GVHMR runtime) to run end-to-end
- MISSING: not implemented yet

## Repo Entry Points

Local platform (laptop):
- `infra/docker-compose.yml` (Postgres, Redis, MinIO, API, worker, web)
- API health: `curl http://localhost:8000/health`
- Web: `http://localhost:3000`

CPU pipeline orchestration (laptop-side prep):
- `scripts/run_end_to_end_cpu.sh`

PhysHOI baseline inference (GPU PC):
- `scripts/install_physhoi_checkpoints.sh`
- `scripts/run_physhoi_inference.sh`

## 1) Simulator Stack (Humanoid + Objects + PD Torque Interface)

- PD torque interface + action semantics parity (OK)
  - `services/xmimic/xmimic/controllers/pd.py`
  - `services/xmimic/xmimic/deploy/runtime.py`
- Robot asset + joint map + contact bodies (OK for v0 asset; PARTIAL for real G1 fidelity)
  - `assets/robots/generic_humanoid.urdf`
  - `assets/objects/cargo_box.urdf`
  - `services/xmimic/configs/robot_spec.yaml`
  - `services/xmimic/scripts/robot_spec_validate.py`
- Isaac Sim / Isaac Lab parallel sim harness (GPU-BLOCKED; Windows GPU worker required)
  - Isaac Lab PPO baseline (real checkpoint artifact):
    - `apps/api/app/gpu/isaaclab_teacher_ppo.py`
    - Triggered via XMimic job params: `backend="isaaclab_teacher_ppo"`
  - Notes:
    - `docs/ISAACLAB_WINDOWS_SETUP.md`
    - `docs/ISAAC_LAB_MIGRATION.md`

## 2) XGen (Video -> Physically Plausible Interaction Clips)

### 2.1 Human motion from monocular video (GVHMR -> SMPL/SMPL-X)

- GVHMR wrapper + checkpoint wiring (GPU-BLOCKED to run; code OK)
  - `services/xgen/xgen/pose/gvhmr_pose.py`
  - `docs/GVHMR.md`
- SMPL -> SMPL-X conversion + resample/smoothing (PARTIAL; depends on body model assets)
  - `services/xgen/xgen/pose/smplx_convert.py`
  - `services/xgen/xgen/pose/smplx_resample.py`
- CPU fallback pose extraction (OK, not paper-accurate but keeps pipeline usable)
  - `services/xgen/xgen/pose/mediapipe_pose.py`

### 2.2 Retarget human motion to humanoid (GMR-style IK)

- Retargeting scaffold + validator (PARTIAL; needs tighter parity with paper + target robot)
  - `services/xgen/xgen/retarget/retarget.py`
  - `services/xgen/xgen/retarget/simple_retarget.py`
  - `services/xgen/xgen/retarget/validate.py`

### 2.3 Contact/non-contact segmentation and anchors

- Manual ts/te segmentation model + phase labels (OK)
  - `services/xgen/xgen/interaction/segment.py`
- Anchor definitions (two-hand midpoint vs single body part) (OK)
  - `services/xgen/xgen/interaction/anchors.py`

### 2.4 Contact phase synthesis (anchor invariance) + contact graph labels

- Object init (manual + estimator hooks) (PARTIAL)
  - `services/xgen/xgen/interaction/object_init.py`
  - `external/sam3d` (estimator wrapper)
- Contact synthesis (anchor-object relative transform propagation) (OK)
  - `services/xgen/xgen/interaction/contact_synth.py`
- Contact graph label generator (PARTIAL; currently heuristic)
  - `services/xgen/xgen/interaction/contact_graph.py`

### 2.5 Force-closure contact refinement

- QP force solver + object-pose refinement helper + CPU IK hook (PARTIAL: IK exists for joint-level alignment, but full force-closure SQP/contact constraints are not yet implemented)
  - `services/xgen/xgen/interaction/force_closure_refine.py`
  - Unit test: `services/xgen/tests/test_force_closure_ik.py`

### 2.6 Non-contact phase synthesis via physics simulation (forward/backward, inverted damping)

- Non-contact simulation helpers (OK for MuJoCo/ballistic fallback; GPU-BLOCKED for Isaac Gym parity)
  - `services/xgen/xgen/interaction/noncontact_sim.py`

### 2.7 Stitching/smoothing around boundaries

- Stitch + smoothing utilities (OK)
  - `services/xgen/xgen/interaction/stitch.py`

### 2.8 Data augmentation (object geometry, trajectory, velocity)

- Mesh scaling/substitution and trajectory transforms (PARTIAL; requires real mesh library assets)
  - `services/xgen/xgen/augment/mesh.py`
  - `services/xgen/xgen/augment/trajectory.py`
  - `services/xgen/xgen/augment/noncontact.py`
  - `services/xgen/xgen/augment/augmentations.py`

### 2.9 Export format (NPZ clips + metadata)

- Clip schema + export writer (OK)
  - `services/xgen/xgen/export/schema.py`
  - `services/xgen/xgen/export/clip_export.py`
- Preview rendering (PARTIAL; CPU ffmpeg path exists but not validated against all clips)
  - `services/xgen/xgen/export/preview.py`

## 3) XMimic Stage 1 (Teacher PPO, Privileged State, Unified Interaction Reward)

- PPO actor-critic scaffold + BC hook (PARTIAL)
  - Paper-faithful imitation PPO lives under `services/xmimic/xmimic/train/` and is still being wired to a full humanoid env.
  - A *real Isaac Lab PPO baseline* (task reward, not HumanX imitation reward) exists for GPU validation and checkpoint export:
    - `apps/api/app/gpu/isaaclab_teacher_ppo.py`
- Unified reward terms (PARTIAL; closer to equation-level parity but env wiring is still pending)
  - `services/xmimic/xmimic/rewards/unified.py`
  - `services/xmimic/configs/humanx_reward.yaml`
  - Eq (12) weighted contact mismatch support.
  - Relative motion vectors can be derived from `body_pos` + `object_pos` (paper Eq. 10 helper).
  - Table IV reg terms support added: feet orientation/slippage, DoF/torque limit penalties.
  - Unit tests: `services/xmimic/tests/test_rewards_unified.py`
- AMP discriminator scaffold (PARTIAL; training integration not end-to-end validated)
  - `services/xmimic/xmimic/amp.py`
- Disturbed initialization + interaction termination + domain randomization + sustained pushes (PARTIAL; wired in PPO scaffold)
  - `services/xmimic/xmimic/generalization.py`
  - `services/xmimic/xmimic/utils/di_it_dr.py`

## 4) XMimic Stage 2 (Student PPO + BC Distillation, NEP/MoCap Modes)

- Student PPO BC distillation hook (PARTIAL; env + teacher policy wiring pending real sim run)
  - `services/xmimic/xmimic/train/student.py`
  - `services/xmimic/xmimic/train/multiskill.py`
- NEP / MoCap observation builders + dropout (PARTIAL; Table IV schema now enforced, env still needs to emit full signals)
  - `services/xmimic/xmimic/obs.py`
  - `services/xmimic/xmimic/obs_pipeline.py`
  - Helper schemas: `humanx_teacher_obs_config(...)`, `humanx_student_obs_config(...)`
  - Added support for distinct `history_fields` (Table IV style: current terms + history terms subset).
  - Mapping doc + schema tests: `docs/HUMANX_TABLE_IV_MAPPING.md`, `services/xmimic/tests/test_obs_pipeline.py`

## 5) Evaluation (SR/GSR/Eo/Eh + Reports)

- Metrics + report generator (PARTIAL; depends on real rollouts and clip definitions)
  - `services/xmimic/xmimic/eval/metrics.py`
  - `services/xmimic/xmimic/eval/tracking.py`
  - `services/xmimic/xmimic/eval/report.py`

## 6) Deployment Runtime (100 Hz policy, 1000 Hz PD, MoCap dropout parity, safety)

- Runtime loop scaffold (OK)
  - `services/xmimic/xmimic/deploy/runtime.py`
- MoCap ingestion + transforms + dropout scaffolding (PARTIAL)
  - `services/xmimic/xmimic/deploy/mocap.py`
- Safety/failsafes scaffold (PARTIAL; hardware integration still needed)
  - `services/xmimic/xmimic/deploy/safety.py`

## 7) Platform (Web + API + Jobs + Storage)

- Docker local stack (OK)
  - `infra/docker-compose.yml`
- Backend (FastAPI, DB models, Alembic, Celery queues, MinIO presigned ops) (OK)
  - `apps/api/app`
  - `apps/api/alembic`
- Frontend (Next.js UI shell + flows) (OK for dev; depends on backend endpoints)
  - `apps/web/src`

## Gaps That Still Block Full Paper-Fidelity

These are the main remaining deltas vs the paper, even aside from GPU availability:
- Force-closure refinement needs a full SQP/contact constraint loop (joint-level IK exists; friction cone / penetration constraints remain).
- Reward parity still needs integration into a real sim environment step (current code + tests exist, but env stubs do not emit all signals).
- Observation parity is enforced at the schema level (Table IV mapping + tests), but needs end-to-end env instrumentation.
- Isaac Sim / Isaac Lab runs are GPU PC dependent (see `HAN-142`, `HAN-150` equivalents once renamed).
