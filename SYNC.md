# Keeping this repo in sync

This private repo tracks the public [sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench)
repo, plus long-lived project branches (`hyper-tau`, `soham/tau-multilingual`).

```
public tau2-bench main
        │  (daily merge — automated)
        ▼
private main            <- the sync spine; carries no project work of its own
        │  (daily sync PRs — automated; conflict-free PRs auto-merge,
        │   owners resolve conflicted ones)
        ├──► hyper-tau
        └──► soham/tau-multilingual
```

## The rules

1. **Horizontal / core changes land upstream-first.** A fix to shared code
   (`src/tau2` core, evaluator, runner, orchestrator, etc.) goes to **public
   main** if it can be public, otherwise to **private main** — never directly
   to a project branch. It reaches the project branches through the daily
   sync. Landing core fixes on a project branch and cherry-picking around is
   what created the 4-month divergence this setup replaces.
2. **Project-specific work stays on the project branch.** PRs target
   `hyper-tau` or `soham/tau-multilingual` as usual.
3. **Sync PRs are merged with a merge commit — never squash.** Squashing
   rewrites main's commits and guarantees the same conflicts return next week.

## The automation

[`.github/workflows/sync-from-public.yml`](.github/workflows/sync-from-public.yml)
runs daily (and on manual dispatch):

1. Merges public `main` into private `main` and pushes. This is expected to be
   conflict-free (private main carries almost nothing of its own); the job
   fails loudly if not.
2. For each project branch, force-pushes `sync/main-into-<branch>` to the
   current main and opens a PR into the branch, assigned to the branch owner.
   If a previous sync PR is still open, it is left untouched (no new PR, no
   force-push over manual conflict resolutions).
3. Conflict-free sync PRs are merged automatically (merge commit — never
   squash). Conflicted ones stay open for the branch owner to resolve.
   Note: the repo has no blocking test CI on PRs today, so "conflict-free"
   is the only automated gate; if a required check is ever added to the
   project branches, the auto-merge waits for it via GitHub auto-merge.

## Resolving a conflicted sync PR

```bash
git fetch origin
git checkout sync/main-into-<branch>
git config merge.ours.driver true   # once per clone; enables uv.lock 'ours' merge
git merge origin/<branch>           # resolve conflicts
uv lock                             # regenerate the lockfile, don't hand-merge it
make test
git push origin sync/main-into-<branch>
```

Then merge the PR (merge commit). Resolution conventions that have worked
well: keep the project branch's intentional divergences (model defaults,
branch-specific features), adopt main's structural/content updates, and treat
conflicts in files the branch never deliberately edited as "take main's side".

## Secrets

Auth preference order:

1. **GitHub App** (`tau2-sync-bot`, installed on this repo by IT): secrets
   `SYNC_APP_ID` + `SYNC_APP_PRIVATE_KEY`. The workflow mints a short-lived
   installation token per run. Preferred: org-managed, not tied to a person,
   no expiry rotation, and CI triggers on the PRs it creates.
2. `SYNC_PAT` repo secret (fine-grained PAT with `contents: write` +
   `pull-requests: write` on this repo). Legacy fallback.
3. Default `GITHUB_TOKEN` — works for the main merge, but the org policy
   blocks it from creating PRs, and it does not trigger CI.
