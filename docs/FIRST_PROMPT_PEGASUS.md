# First Prompt (Pegasus)

If you start a Codex agent inside a Pegasus VS Code Remote-SSH session, paste this:

```
You are on Pegasus in the repo root. Follow docs/PEGASUS_SETUP.md.

Goal: run GPU work on Pegasus (Slurm), no local gaming PC.

1) Confirm we are NOT on a login node for heavy work. If we are, start an interactive GPU job using scripts/pegasus/srun_gpu_shell.sh and continue inside it.
2) From repo root, run scripts/gpu/bootstrap.sh and summarize: clones, checkpoint downloads, missing license-gated Isaac Gym.
3) If Isaac Gym is missing, tell me exactly where to download Preview 4 and where to place it under external/humanoid-projects/isaacgym/.
4) Run PhysHOI baseline inference (pass.pt + checkpoint) and capture logs/errors.
5) If Isaac Gym is incompatible with available GPUs, propose the fastest pivot (Isaac Lab) and update docs/STATUS.md with the decision.
```

