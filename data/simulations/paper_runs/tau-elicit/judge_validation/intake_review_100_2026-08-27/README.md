# intake_review_100_2026-08-27

Human annotation returns for the intake domain user review: 100 voice
calls from `intake_bm200_2026-08-26_chanlight_speechheavy`
(sims in `data/simulations/intake_bm200_2026-08-26_chanlight_speechheavy/`),
reviewed by two raters. Packet built by `tau2 annotate` intake-review
(packet version `intake-review-packet-v1`, batch `4e9409b1731c`, PR #971).

Results summary: [results.md](results.md).

## Files

| file | rows | what |
|---|---|---|
| `niko_review.csv` | 200 | niko: per-call verdicts, 100 calls × 2 phases (blind `pre_reveal`, then `post_reveal`) |
| `niko_decisions.csv` | 41 | niko: confirm/reject on each fidelity-judge finding |
| `ian_belcher_review.csv` | 200 | Ian Belcher: per-call verdicts, same structure |
| `ian_belcher_decisions.csv` | 41 | Ian Belcher: finding decisions, same structure |

## Schemas

**Review**: `batch, rater, clip_id, simulation_id, task_id, bank, tier,
reward, phase, error_source (agent/user/no_error), error_subtype,
first_error_turn, error_notes, fidelity_severity (0–3), fidelity_notes,
provenance_json, completed, created_at`.

**Decisions**: `batch, rater, clip_id, simulation_id, task_id, finding_id,
category, utterance_index, judge_severity, judge_issue,
decision (confirmed/rejected), note, provenance_json, completed,
created_at`.

## Caveats

- Calls 1–40 were the rater calibration segment for error-type
  annotation; use calls 41–100 for blind (pre_reveal) error-type numbers.
  Post-reveal and decision numbers are valid over all 100.
- `first_error_turn` is populated in `pre_reveal` rows only.
