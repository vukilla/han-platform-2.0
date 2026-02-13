# Pegasus + Windows Onboarding (New Teammate Guide)

This guide is for setting up the platform on a new machine with Pegasus-first execution and optional Windows fallback.

## 1) What `PEGASUS_HOST` is

`PEGASUS_HOST` is the SSH target for the Pegasus server where the worker is started.

- It can be a hostname such as `pegasus.internal`, a full SSH URL like `user@10.10.0.12`, or an SSH alias from `~/.ssh/config`.
- The value is expected to match one command like:
  - `ssh pegasus`
  - `ssh user@pegasus.example.org`
- In this repo, scripts use either:
  - env var: `PEGASUS_HOST=<value>`
  - positional arg: `./scripts/mac/start_pegasus_worker_ssh.sh <value>`

## 2) What each teammate needs before setup

1. SSH access to Pegasus control node (host, user, private key).
2. Optional: access to Windows GPU machine IP for fallback mode.
3. Docker, Node.js, Python runtime, and git installed on local Mac.
4. Repository cloned locally at a known path, for example:
   - `/Users/<you>/Downloads/Python/han-platform`

## 3) Verify Pegasus SSH works first

From the terminal:

```bash
ssh <pegasus_host_or_alias>
```

If that succeeds, note the string used to connect, that is your host value.

Example:

```bash
ssh pegasus
```

Then use `pegasus` as `PEGASUS_HOST`.

## 4) Set one-time SSH config (recommended)

Add in `~/.ssh/config`:

```text
Host pegasus
  HostName <pegasus_host_or_ip>
  User <pegasus_user>
  IdentityFile ~/.ssh/<your_pegasus_key>
  IdentitiesOnly yes
```

Then validate:

```bash
ssh pegasus "hostname"
```

## 5) Export required environment variables

```bash
export PEGASUS_HOST=pegasus
export SSH_KEY=~/.ssh/<your_pegasus_key>   # skip if using ssh-agent or ~/.ssh/config IdentityFile
export SSH_USER=<pegasus_user>              # optional if alias already includes user
```

Optional fallback to Windows:

```bash
export WINDOWS_GPU_IP=<windows_lan_ip>
```

## 6) Start control plane and worker

```bash
cd /Users/<you>/Downloads/Python/han-platform

# one-time Pegasus bootstrap
./scripts/mac/bootstrap_pegasus_control_plane_ssh.sh

# start Pegasus control plane
./scripts/mac/start_pegasus_control_plane_ssh.sh
./scripts/mac/status_pegasus_control_plane_ssh.sh

# start Pegasus pose worker with auto fallback behavior
export CONTROL_PLANE_MODE=auto
export HAN_WORKER_QUEUES=pose
./scripts/mac/start_pegasus_worker_ssh.sh
```

## 7) End-to-end checks

Run a quick studio smoke test:

```bash
PEGASUS_HOST=pegasus ./scripts/mac/run_gvhmr_studio_ssh.sh
```

Or run full pipeline smoke (requires worker + model assets configured for your chosen worker path):

```bash
PEGASUS_HOST=pegasus /Users/<you>/Downloads/Python/han-platform/scripts/mac/run_full_e2e_real_ssh.sh
```

If Pegasus is unavailable, set only `WINDOWS_GPU_IP` and use the same `run_*_ssh.sh` scripts for fallback mode.

## 8) What to check when onboarding a new teammate

Ask them to capture and confirm:

- `PEGASUS_HOST` value used and why it resolves.
- SSH connectivity result.
- Output from `./scripts/mac/status_pegasus_control_plane_ssh.sh`.
- `start_pegasus_control_plane_ssh.sh` and `start_pegasus_worker_ssh.sh` logs show running.
- One successful job reaches `/jobs/<id>` and stages past `INGEST_VIDEO` and `ESTIMATE_POSE`.

## 9) Reference docs

- `docs/PEGASUS_DUAL_WORKER_SETUP.md`
- `docs/WHAT_TO_TEST.md`
- `README.md` quickstart section
