# First Prompt (GPU PC, Windows Native)

If you start a Codex agent on the gaming PC (Windows), paste this as the first message:

```text
You are on the Windows GPU PC in the repo root (han-platform-2.0). Follow AGENTS.md exactly.

Constraints:
- Do not use WSL2/virtualization.
- Use Windows-native Isaac Sim + Isaac Lab.

Do these steps in order and summarize after each:
1) Run `scripts\\windows\\preflight.ps1` and report:
   - GPU + driver (nvidia-smi)
   - git + conda availability
   - disk space
2) If Isaac Sim is not installed, stop and tell me EXACTLY:
   - which Isaac Sim Windows download page to use
   - where to extract it (target: C:\\isaacsim)
   - what file(s) you expect to exist (isaac-sim.bat)
3) Run `scripts\\windows\\bootstrap_isaaclab.ps1 -IsaacSimPath C:\\isaacsim`.
4) Run `scripts\\windows\\run_gpu_worker.ps1 -MacIp <MAC_LAN_IP>` and confirm it can connect to Redis/MinIO on the Mac.
   - Optional detached: `scripts\\windows\\start_gpu_worker_detached.ps1 -MacIp <MAC_LAN_IP>`
5) If any step fails, write a short diagnosis and the minimal next action.
```
