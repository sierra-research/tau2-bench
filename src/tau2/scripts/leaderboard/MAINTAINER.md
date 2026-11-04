# Leaderboard Maintainer Guide

## Prerequisites

- AWS CLI configured with `tau-bench-ci` profile
- S3 bucket: `sierra-tau-bench-public`
- S3 prefix: `submissions/`

## Handling a Submission PR

A PR adds either a 1.x / voice-v1 `submission.json` (this section) or a 2.0
package (`benchmark-submission.json`; see
[Handling a 2.0 package PR](#handling-a-20-package-pr)).

When someone opens a PR with a new `submission.json`:

### 1. Review the PR

Check the submission.json manually:
- Directory name follows `{model_name}_{org}_{date}` convention
- `manifest.json` is updated (correct array: `submissions`, `voice_submissions`, or `legacy_submissions`; 2.0 packages use `voice_v2_submissions`)
- `submission_type` is `"standard"` or `"custom"` (if custom: check `methodology.notes` and `references`)
- Contact info is present
- For voice: `modality` is `"voice"`, `voice_config` is present

### 2. Download their trajectories

The PR description should include a link to externally hosted trajectory files.
Download them locally:

```bash
# Example: download from Google Drive, HuggingFace, etc.
mkdir -p /tmp/review-trajectories
# ... download to /tmp/review-trajectories/
```

The trajectory source should contain either:
- **Text**: One or more `results.json` (or `*.json`) files, one per domain
- **Voice**: One or more experiment directories, each containing `results.json` + `simulations/` + `artifacts/*/audio/`

### 3. Run the review script

```bash
python -m tau2.scripts.leaderboard.review_submission \
  web/leaderboard/public/submissions/<submission-dir> \
  /tmp/review-trajectories/

# Or with --upload to also push to S3:
python -m tau2.scripts.leaderboard.review_submission \
  web/leaderboard/public/submissions/<submission-dir> \
  /tmp/review-trajectories/ \
  --upload --aws-profile tau-bench-ci
```

The script will:
1. Parse the PR's `submission.json`
2. Load and validate trajectory files (format, task coverage, trial counts)
3. Recompute metrics and compare against submitted scores
4. (Voice) Recompute `interaction_metrics` from the tick data, replacing any submitter-provided values
5. Update `submission.json` with `trajectories_available: true` and `trajectory_files`
6. (With `--upload`) Upload trajectories and updated `submission.json` to S3
7. (With `--upload`) Verify the upload (HTTP 200, file sizes match)

### 4. Verify upload separately (optional)

If you uploaded manually or want to re-check:

```bash
python -m tau2.scripts.leaderboard.review_submission \
  web/leaderboard/public/submissions/<submission-dir> \
  /tmp/review-trajectories/ \
  --verify-only --aws-profile tau-bench-ci
```

### 5. Commit and merge

After the script updates `submission.json`, commit the change and merge the PR.
The GitHub Actions workflow will sync `submission.json` and `manifest.json` to S3
(trajectories are excluded from the sync — they were uploaded directly by the script).

## Handling a 2.0 package PR

A 2.0 package is a few files in `web/leaderboard/public/submissions/<name>/`:
`benchmark-submission.json`, `benchmark-diagnostics.json` and
`benchmark-quality.json` (both voice-v2 only), and `benchmark-metadata.json`.
`benchmark-quality.json` holds each scored call's Interaction and Generation
quality (`experience`) verdicts, which feed the leaderboard's Interaction and
Generation quality columns. voice-v2 packages are listed in the manifest's
`voice_v2_submissions` array and are the rows of the τ³-Voice **τ-i18n
(Multilingual)** leaderboard.
voice-v2-banking and voice-v2-entity packages (`benchmark-submission.json` and
`benchmark-metadata.json`, plus `benchmark-pass3.json` for entity) go in
`voice_v2_domain_submissions`. Intake is scored by crossed Pass^3: the share of
tasks passed in all three voice conditions (regular, heavy channel, heavy
speech; `tau2.runner.pass3`). `benchmark-pass3.json` holds each task's three
outcomes; `benchmark-submission.json` keeps the regular condition's rewards.
The review tool uploads the other two conditions to
`pass3/<language>/intake/<condition>/`. Each fills
the Banking or Intake row of the voice-v2 package with the same model name,
organization, agent provider, agent model and submission type, and never
changes its Overall. Without a matching voice-v2 package it is shown only
under its own domain. 1.x tabs are unaffected.

1. **Review** the PR: the directory name equals
   `{model_name}_{model_organization}_{submission_date}` (slugified; the review
   script checks it). voice-v2-banking and voice-v2-entity packages append
   `_banking` / `_entity`, because they share the voice-v2 package's metadata.
   Also check that `benchmark-metadata.json` has a contact email and that
   `submission_type` is right.
2. **Download** the submitter's matrix root (`benchmark-frame.json` plus every
   `<language>-<domain>/` cell with `results.json`, `simulations/`,
   `artifacts/` and `diagnostics/`).
3. **Validate, then upload:**

   ```bash
   python -m tau2.scripts.leaderboard.review_submission \
     web/leaderboard/public/submissions/<name> /tmp/review-matrix/          # dry run
   python -m tau2.scripts.leaderboard.review_submission \
     web/leaderboard/public/submissions/<name> /tmp/review-matrix/ \
     --upload --aws-profile tau-bench-ci
   ```

   The script recomputes rewards, judge diagnostics, call quality and the
   recorded agent identity from the matrix (`tau2 submit validate`), then prints the upload
   plan. With `--upload` it syncs each cell to
   `s3://sierra-tau-bench-public/submissions/<name>/trajectories/<language>/<domain>/`
   and the root's JSON files to `trajectories/`, sets `trajectories` in
   `benchmark-metadata.json` to `s3://sierra-tau-bench-public/submissions/<name>`,
   uploads the package files and checks that every cell's `results.json` is
   publicly readable. `--verify-only` repeats that check.
4. **Commit** the updated `benchmark-metadata.json` and merge. The sync workflow
   copies the package files; it never deletes uploaded evidence.

To re-validate a package from S3, download `trajectories/<language>/<domain>/`
back into `<language>-<domain>/` cells next to `trajectories/benchmark-frame.json`.

### The frozen 2026-10-02 release

Release packages cite the release export instead of a matrix
(`"evidence": "historical-voice-v2-2026-10-02"`). Build and review them with
`--historical-voice-v2` and one `--judges-root` per judge suite; see
[docs/voice-v2-publication.md](../../../../docs/voice-v2-publication.md). The
upload also copies the export's `correction.json` to `source_correction.json`
and the verdicts to `judges/<agent_provider>/<language>/`, matching the
uploaded Gemini entry. The voice-v2 release package's `benchmark-quality.json`
is recomputed from the export's saved ticks and verdicts; validation needs the
same `--results` and `--judges-root` inputs.

The same export's banking and entity calls package as `voice-v2-banking` and
`voice-v2-entity` with `--historical-voice-v2 --benchmark banking|entity` and
no `--judges-root`. These frames are not judged. Their cells predate the
manifests, so the adapter checks the recorded bindings against the release and
accepts only the verified `runtime_rules_sha256` digests in
`ACCEPTED_RUNTIME_RULES` (`voice_v2_publication.py`). Never extend that
allowlist without verifying the source diff it covers.

## Trajectory visualizer data

The visualizer (`/trajectory-visualizer`) lists submissions from
`visualizer-index.json`, generated at build time from the manifest and the
committed metadata (`web/leaderboard/scripts/generate-visualizer-index.mjs`).
It then loads a cell's small `results.json` index and fetches one
`simulations/<id>.json` per opened conversation.

Text submissions upload one monolithic file per domain; `review_submission
--upload` also uploads it split into `trajectories/<domain>/` (gzipped, with
caching headers). For submissions uploaded before that:

```bash
# Stage (gzipped) under a local directory; skips cells already staged
python -m tau2.scripts.leaderboard.viewer_data split --staging ~/.cache/tau2/viewer-staging \
    [<submission> ...]

# Preview locally: serves staged files, proxies the rest from S3
node web/leaderboard/scripts/serve-submissions.mjs
VITE_SUBMISSIONS_BASE_URL=http://localhost:8787/submissions npm run dev --prefix web/leaderboard

# Upload with Content-Encoding: gzip
python -m tau2.scripts.leaderboard.viewer_data upload --staging ~/.cache/tau2/viewer-staging \
    --aws-profile tau-bench-ci [--dry-run]
```

A text cell without the split layout still works: the visualizer falls back
to the monolithic file.

Voice cells are already saved in this layout. Never re-upload them: an
overwritten run cannot be recovered from S3, and `aws s3 sync` and Python
readers (`Results.load`) do not decode `Content-Encoding: gzip`. `upload`
refuses a staging tree holding anything other than text submissions' viewer
cells, and its sync never deletes.

## Manual S3 Commands

If you need to do things manually:

```bash
# Upload trajectories
aws s3 sync \
  /local/trajectories/ \
  s3://sierra-tau-bench-public/submissions/<submission-dir>/trajectories/ \
  --profile tau-bench-ci

# Upload submission.json
aws s3 cp \
  web/leaderboard/public/submissions/<submission-dir>/submission.json \
  s3://sierra-tau-bench-public/submissions/<submission-dir>/submission.json \
  --profile tau-bench-ci

# Verify a file exists and check size
aws s3api head-object \
  --bucket sierra-tau-bench-public \
  --key submissions/<submission-dir>/trajectories/<filename> \
  --profile tau-bench-ci \
  --query '[ContentLength, ETag]'

# Delete and re-upload (if corrupted)
aws s3 rm s3://sierra-tau-bench-public/submissions/<dir>/trajectories/<file> --profile tau-bench-ci
aws s3 cp /local/path s3://sierra-tau-bench-public/submissions/<dir>/trajectories/<file> --profile tau-bench-ci
```

## Backfilling Interaction Metrics

If the interaction-metric code changes in a way that warrants recomputation
(major refactor — see `INTERACTION_METRICS_VERSION`), or a voice submission is
missing its `interaction_metrics` block, recompute from the public S3
trajectories:

```bash
# All voice submissions in the manifest (downloads trajectories to ~/.cache/tau2/)
python -m tau2.scripts.leaderboard.backfill_interaction_metrics

# Specific submissions, or a dry run
python -m tau2.scripts.leaderboard.backfill_interaction_metrics \
  --submissions gpt-realtime-2_openai_2026-06-24 --dry-run
```

The script patches the local `web/leaderboard/public/submissions/<dir>/submission.json`
files; commit them via PR and the GitHub Actions sync pushes them to S3.

## Troubleshooting

### Large voice trajectory files get corrupted during upload

Files over ~500 MB can get corrupted during S3 multipart upload. Symptoms:
- S3 file size differs from local
- JSON parse errors when fetching from S3

Verify integrity:
```bash
# Compare local vs S3 sizes
local_size=$(stat -f%z /local/path/results.json)
s3_size=$(aws s3api head-object --bucket sierra-tau-bench-public --key <key> --profile tau-bench-ci --query ContentLength --output text)
echo "local: $local_size, s3: $s3_size"
```

If corrupted, delete and re-upload with `aws s3 cp` (not sync).

### Voice trajectory files are very large

Voice experiments use a directory-based format: `results.json` contains only metadata,
while simulation data is split into individual files under `simulations/`. The total
size can still be 100 MB - 1 GB+ across all files. The `prepare_submission` script
copies only `results.json`, `simulations/`, and canonical audio from `artifacts/`.

If uploading large directories manually, consider syncing entire experiment directories:

```bash
aws s3 sync /local/experiment_dir/ \
  s3://sierra-tau-bench-public/submissions/<dir>/trajectories/<exp>/ \
  --profile tau-bench-ci
```
