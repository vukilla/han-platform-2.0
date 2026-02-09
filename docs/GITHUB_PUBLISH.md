# Publish To GitHub (One-Time)

This repo is already a git repo locally. Publishing is just: authenticate, create remote, push.

## Option A: Push To An Existing Empty Repo (Most Reliable)

Create an empty GitHub repository, then:
```bash
git remote add origin <YOUR_GITHUB_REMOTE_URL>
git push -u origin main
```

## Option B: gh Creates Repo For You (Fastest When GitHub Web Auth Works)

If you have `gh` installed and authenticated:
```bash
gh auth login --hostname github.com --git-protocol https --web
./scripts/gh_create_repo_and_push.sh han-platform-2.0 private
```

If GitHub's web/device auth endpoints are returning `503`, use a Personal Access Token instead:
```bash
export GH_TOKEN="<YOUR_PAT>"
echo "$GH_TOKEN" | gh auth login --hostname github.com --with-token
./scripts/gh_create_repo_and_push.sh han-platform-2.0 private
```

## Option C: No GitHub Yet (Bundle For USB/Local Transfer)

If GitHub auth is blocked or you just want to move to the GPU PC immediately:
```bash
./scripts/make_git_bundle.sh
```

Then copy the generated `.bundle` file to the other machine and:
```bash
git clone han-platform-2.0.bundle han-platform-2.0
```

## Notes
- `external/`, `tmp/`, `.env`, venvs, and large checkpoints are excluded by `.gitignore`.
- The GPU bootstrap scripts rehydrate `external/` on the gaming PC.
