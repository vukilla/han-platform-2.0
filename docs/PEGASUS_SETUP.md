# Pegasus (DFKI) GPU Compute Setup

This repo can run the GPU-heavy steps (GVHMR, PhysHOI/Isaac Gym, later training) on the Pegasus
cluster so you do not need a local gaming PC.

Important: use **your own Pegasus account**. Do not share passwords or use someone else's login.

## What You Need

1. Your Pegasus username and access to login nodes:
   - `login1.pegasus.kl.dfki.de` (or `login2/3`)
   - Note: these hostnames resolve to a private `192.168.33.x` address. From home, you must be on the DFKI network
     (or connected via the DFKI Saarbrücken VPN) or you will see `Network is unreachable`.
2. SSH access (prefer SSH keys, not passwords).
3. GPU partition access (A100/H100/etc) and a basic Slurm workflow.

## 0) Fix "Network is unreachable" (VPN / Routing)

If this fails:
```bash
ssh <USER>@login1.pegasus.kl.dfki.de
```
and you see `Network is unreachable`, you are not on the right network yet. Confirm:

```bash
nslookup login1.pegasus.kl.dfki.de 8.8.8.8
nc -vz -w 3 login1.pegasus.kl.dfki.de 22
```

If DNS returns `192.168.33.x` and `nc` fails, connect to the DFKI Saarbrücken VPN (ask your DFKI IT/contact for the
VPN profile if you do not have it yet). Re-test `nc` until port 22 is reachable.

## 1) SSH Key Setup (from your laptop)

```bash
ssh-keygen -t ed25519 -C "<your_email>" -f ~/.ssh/id_ed25519
```

Add the public key to Pegasus (you will be prompted for your Pegasus password once):
```bash
ssh-copy-id -i ~/.ssh/id_ed25519.pub <USER>@login1.pegasus.kl.dfki.de
```

If `ssh-copy-id` is not installed on macOS, use:
```bash
cat ~/.ssh/id_ed25519.pub | ssh <USER>@login1.pegasus.kl.dfki.de \
  'mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys'
```

## 2) Clone Repo On Pegasus

On Pegasus (recommended location is your scratch):
```bash
mkdir -p /netscratch/$USER
cd /netscratch/$USER
git clone https://github.com/vukilla/han-platform-2.0.git han-platform-2.0
cd han-platform-2.0
```

## 3) Run GPU Bootstrap On Pegasus

On a **GPU job** (not on the login node), run:
```bash
scripts/gpu/bootstrap.sh
```

This clones external repos under `external/` and downloads what it can. Isaac Gym remains
license-gated and must be downloaded manually.

## 4) Slurm Templates

We include templates under:
- `scripts/pegasus/`

Start with:
```bash
scripts/pegasus/srun_gpu_shell.sh
```

Then, inside the allocated GPU shell:
```bash
cd /netscratch/$USER/han-platform-2.0
scripts/gpu/bootstrap.sh
```

## 5) VS Code Remote SSH

Use Remote-SSH to open `/netscratch/$USER/han-platform-2.0` on Pegasus.
First prompt for a Codex agent on Pegasus: `docs/FIRST_PROMPT_PEGASUS.md`.
