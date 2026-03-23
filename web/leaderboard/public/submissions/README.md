# τ-bench Leaderboard Submissions

This directory contains model evaluation results for the τ-bench leaderboard at [taubench.com](https://taubench.com).

## Directory Structure

```
submissions/
├── manifest.json          # Lists all active submissions (text, voice, legacy)
├── schema.json            # JSON schema for submission.json files
├── {model}_{org}_{date}/  # Individual submission directories
│   ├── submission.json    # Submission metadata and metrics
│   └── trajectories/      # Trajectory files (text submissions only)
└── A_EXAMPLE_*/           # Example submissions for reference
```

## Schema

Your `submission.json` must conform to [`schema.json`](schema.json) in this directory.

## Hosting

Files in this directory are synced to the `sierra-tau-bench-public` S3 bucket on merge to `main` (via the `sync-submissions-s3.yml` GitHub Actions workflow). The production website at [taubench.com](https://taubench.com) fetches submission data from S3, not from GitHub Pages directly.

Contributors still submit new results by adding files here and opening a PR — the S3 sync is automatic.

## Full Submission Guide

For complete instructions on how to run evaluations, prepare submissions, and submit a pull request, see the **[Leaderboard Submission Guide](../../../../docs/leaderboard-submission.md)**.
