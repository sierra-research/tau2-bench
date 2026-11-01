# Task grading corrections (opt-in)

Each JSON file here is a named correction set loaded only by
`tau2 evaluate-trajs --task-corrections <name>` (see
`src/tau2/evaluator/task_corrections.py`). Official grading never reads these
files, and corrected rewards are not leaderboard-comparable: report them next to
the official rewards, never instead of them.

- `voice-v2-task-defects-20261029`: airline/retail grading corrections described below.

Each entry pins `task_digests`: per call language, the `task_content_digest`
(sha256 of the canonical JSON of `user_scenario` + `evaluation_criteria`) of
every shipped task copy it was written against. Grading refuses an entry when
the graded task's digest is not pinned (a different task version), and refuses
a non-English call unless its recorded scenario mode is `regional`, because the
per-language values are written for the regional bundles. After a deliberate
task change, re-validate the entry and update its digests;
`tests/test_evaluator/test_task_corrections.py` fails until they match.

## Scope and compatibility

This is a grading-only overlay for saved voice-v2 trajectories, not a new
benchmark release. It keeps the released `benchmark_version`, task digests and
manifest binding revisions unchanged. The original result files remain the
source of official scores; write corrected results to a separate directory.

The set covers retail 105 (an unaffordable exchange), airline 39 (an ineligible
cancellation), retail 38/66 (either cancellation reason), airline 44 (journey
rather than segment duration), airline 7 (the cost of the other reservations),
retail 64 (either qualifying camera), airline 29 (either insurance answer), and
localized retail 36/38 assertions (generic camera wording). Airline 39 becomes
stricter for trajectories that cancel the ineligible reservation. Airline 44
retains the original interpretation as an accepted alternative because its
wording is ambiguous; this is a diagnostic sensitivity result.

Retail 52 needs a clearer caller instruction and a new simulation; it is not
included. Changes to NL assertions and communication requirements require a new
semantic judge verdict. A DB-only replay must report these cells as pending,
not turn them into passes.

Direct task-data fixes have precedent (#402), but the current released manifests
pin task content: see `docs/running-the-benchmark.md`, “For maintainers:
regenerate manifest pins”. Changing those pins requires a new binding revision
and invalidates old results against that revision. This overlay allows historical
re-grading without silently changing the frozen benchmark. It does not replace a
future task-data release or claim comparability with unregraded submissions.

Every corrected reward records the correction-set name and canonical content
SHA-256 as well as the matching entry and alternatives. `--fresh-tasks` cannot
be combined with `--task-corrections`: replacing saved instructions before the
content check would not establish which scenario the conversation used.
