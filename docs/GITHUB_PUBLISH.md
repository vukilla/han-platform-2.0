# Publish To GitHub (One-Time)

This repo is not initialized as git by default. Use the commands below from the repo root.

## 1) Initialize + Commit
```bash
git init
git add .
git commit -m "Initial import: han-platform-2.0"
```

## 2) Add Remote + Push
Create an empty GitHub repository, then:
```bash
git branch -M main
git remote add origin <YOUR_GITHUB_REMOTE_URL>
git push -u origin main
```

## Alternative (Recommended): gh Creates Repo For You

If you have `gh` installed and authenticated:
```bash
gh auth login --hostname github.com --git-protocol https --web
./scripts/gh_create_repo_and_push.sh han-platform-2.0 private
```

## Notes
- `external/`, `tmp/`, `.env`, venvs, and large checkpoints are excluded by `.gitignore`.
- If you already have a git repo, skip `git init` and only do the remote/push steps.
