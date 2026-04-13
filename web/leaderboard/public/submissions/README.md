# τ-bench Leaderboard Submissions

This directory contains model evaluation results for the τ-bench leaderboard at [taubench.com](https://taubench.com).

## Directory Structure

```
submissions/
├── manifest.json          # Lists all active submissions (text, voice, legacy)
├── schema.json            # JSON schema for submission.json files
├── {model}_{org}_{date}/  # Individual submission directories
│   └── submission.json    # Submission metadata and metrics
└── A_EXAMPLE_*/           # Example submissions for reference
```

Trajectory files are **not** stored in this directory — they are hosted on S3.

## Schema

Your `submission.json` must conform to [`schema.json`](schema.json) in this directory.

**Note:** `schema.json` is auto-generated from the Pydantic models in `src/tau2/scripts/leaderboard/submission.py`. Do not edit it by hand. Run `make generate-schema` to regenerate, or `make check-schema` to verify it is up-to-date.

## Hosting

Submission metadata (`submission.json`, `manifest.json`) in this directory is synced to the `sierra-tau-bench-public` S3 bucket on merge to `main` (via the `sync-submissions-s3.yml` GitHub Actions workflow).

Trajectory files are hosted on S3 only and are uploaded by a maintainer after reviewing the PR. Contributors provide a link to their trajectory data in the PR description.

The production website at [taubench.com](https://taubench.com) fetches all data (metadata and trajectories) from S3.

## Full Submission Guide

For complete instructions on how to run evaluations, prepare submissions, and submit a pull request, see the **[Leaderboard Submission Guide](../../../../docs/leaderboard-submission.md)**.
