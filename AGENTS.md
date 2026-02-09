# Codex Agent Instructions (han-platform-2.0)

This repo is a local-first implementation of the HumanX Data Factory (XGen + XMimic) with a web app + API + job system.

## Primary Goal (GPU PC)

Get the GPU-blocked pieces running end-to-end on the gaming PC:
1. PhysHOI baseline inference runs successfully (proves Isaac Gym + GPU stack).
2. GVHMR demo runs on a sample video and produces motion outputs.
3. Run the pipeline: GVHMR -> SMPL-X -> PhysHOI motion pack -> PhysHOI inference.

## What To Run First (one command)

From the repo root:
```bash
scripts/gpu/bootstrap.sh
```

If Isaac Gym is not installed yet, bootstrap will tell you where to place the Preview 4 archive and which command to run next:
```bash
scripts/gpu/install_isaacgym.sh
```

## After Isaac Gym Is Installed

1. Install PhysHOI checkpoints into the cloned PhysHOI repo:
```bash
scripts/install_physhoi_checkpoints.sh
```

2. Run PhysHOI baseline inference (proves the env):
```bash
scripts/run_physhoi_inference.sh \
  external/physhoi/physhoi/data/motions/BallPlay/pass.pt \
  external/physhoi/physhoi/data/models/pass/nn/PhysHOI.pth \
  16 PhysHOI_BallPlay
```

3. Run the CPU-side pipeline (GVHMR -> SMPL-X -> PhysHOI motion pack). This can be done on GPU PC too:
```bash
scripts/run_end_to_end_cpu.sh <video.mp4> <smplx_model_dir> <output_dir>
```

## Repo Notes

- External dependencies and checkpoints are **not committed**. They are rehydrated under `external/` via the GPU bootstrap scripts.
- Do not add secrets to the repo. Use `.env` files locally; `.gitignore` excludes them.

## References

- GPU PC setup details: `docs/GPU_PC_SETUP.md`
- Paper reconciliation status: `docs/HUMANX_PAPER_RECONCILIATION.md`
- Table IV mapping (obs + reward parity): `docs/HUMANX_TABLE_IV_MAPPING.md`

