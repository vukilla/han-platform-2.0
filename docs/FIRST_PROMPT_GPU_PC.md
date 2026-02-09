# First Prompt (GPU PC)

If you start a Codex agent on the gaming PC, paste this as the first message:

```
You are on the GPU PC in the repo root. Follow AGENTS.md exactly.
1) Run scripts/gpu/bootstrap.sh and summarize results (driver, nvidia-smi, clones, downloads).
2) If Isaac Gym is missing, stop and tell me exactly where to download Preview 4 and where to place the archive, then run scripts/gpu/install_isaacgym.sh once it exists.
3) Run PhysHOI baseline inference (pass.pt + checkpoint) and capture logs/errors.
4) If Isaac Gym fails on RTX 5090 (sm_120), propose the fastest pivot (Isaac Lab) and open a Linear blocker issue with details.
```

