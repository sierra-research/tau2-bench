# Keeping this repo in sync

This private repo tracks the public [sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench)
repo, plus long-lived project branches (`hyper-tau`, `soham/tau-multilingual`).

```
public tau2-bench main
        │  (daily merge — automated)
        ▼
private main            <- the sync spine; carries no project work of its own
        │  (daily sync PRs — automated, owners resolve conflicts)
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
   If last week's sync PR is still open, it is left untouched (no new PR, no
   force-push over manual conflict resolutions).

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

The workflow prefers a `SYNC_PAT` repo secret (fine-grained PAT with
`contents: write` + `pull-requests: write` on this repo). Without it the
default `GITHUB_TOKEN` is used, which works but does not trigger CI on the
pushed main or on the sync PRs.
