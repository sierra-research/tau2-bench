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

## Full Submission Guide

For complete instructions on how to run evaluations, prepare submissions, and submit a pull request, see the **[Leaderboard Submission Guide](../../../../docs/leaderboard-submission.md)**.
