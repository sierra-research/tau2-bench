# τ-bench Leaderboard Submissions

This directory contains model evaluation results for the τ-bench leaderboard at [taubench.com](https://taubench.com).

## Directory Structure

```
submissions/
├── manifest.json          # Lists active submissions per tab (text, voice, voice_v2, legacy)
├── schema.json            # JSON schema for 1.x submission.json files
├── schema_benchmark_metadata.json  # JSON schema for 2.0 benchmark-metadata.json
├── schema_benchmark_quality.json   # JSON schema for voice-v2 benchmark-quality.json
├── {model}_{org}_{date}/  # A 1.x submission directory
│   └── submission.json    # Submission metadata and metrics
├── {model}_{org}_{date}/  # A 2.0 package (listed under voice_v2_submissions)
│   ├── benchmark-metadata.json     # Submitter fields, agent identity, S3 prefix
│   ├── benchmark-submission.json   # Frame, trials and pass@1 summary
│   ├── benchmark-diagnostics.json  # Judge-measure summaries (voice-v2)
│   └── benchmark-quality.json      # Call-level Interaction and Generation quality (voice-v2)
└── A_EXAMPLE_*/           # Example submissions for reference
```

Trajectory files are **not** stored in this directory — they are hosted on S3.

## Schema

A 1.x `submission.json` must conform to [`schema.json`](schema.json); a 2.0 `benchmark-metadata.json` must conform to [`schema_benchmark_metadata.json`](schema_benchmark_metadata.json) and a voice-v2 `benchmark-quality.json` to [`schema_benchmark_quality.json`](schema_benchmark_quality.json). The other 2.0 files are written by `tau2 submit prepare` and checked by `tau2 submit validate`.

**Note:** The schemas are auto-generated from the Pydantic models in `src/tau2/scripts/leaderboard/submission.py`, `benchmark_package.py` and `benchmark_quality.py`. Do not edit them by hand. Run `make generate-schema` to regenerate, or `make check-schema` to verify it is up-to-date.

## Hosting

Submission metadata (`submission.json`, `manifest.json`) in this directory is synced to the `sierra-tau-bench-public` S3 bucket on merge to `main` (via the `sync-submissions-s3.yml` GitHub Actions workflow).

Trajectory files are hosted on S3 only and are uploaded by a maintainer after reviewing the PR. Contributors provide a link to their trajectory data in the PR description.

The production website at [taubench.com](https://taubench.com) fetches all data (metadata and trajectories) from S3.

## Full Submission Guide

For complete instructions on how to run evaluations, prepare submissions, and submit a pull request, see the **[Leaderboard Submission Guide](../../../../docs/leaderboard-submission.md)**.
