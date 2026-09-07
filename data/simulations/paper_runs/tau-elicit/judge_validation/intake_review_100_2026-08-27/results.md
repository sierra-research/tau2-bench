# Intake domain — human review results (intake_review_100_2026-08-27)

Two raters (niko, Ian Belcher) reviewed 100 intake voice calls from the
`intake_bm200_2026-08-26_chanlight_speechheavy` run: 70 passing (reward 1)
and 30 failing (reward 0). Each call was reviewed twice — first blind
(audio only), then with the reward and transcript revealed. Raters also
adjudicated all 41 findings raised by the fidelity judge on these calls.
The first 40 calls were used for rater calibration; blind error-type
numbers below use calls 41–100.

## Error types on failing calls (calls 41–100, blind)

When both raters marked an error on a failing call, they agreed on the
error type **88%** of the time. The consensus errors split:

| error type | n | share |
|---|---|---|
| transcription error | 12 | 86% |
| logical error (premature submit / early hang-up) | 2 | 14% |
| hallucination | 0 | 0% |

A similar split holds post-reveal on all 30 failing calls: 20
transcription / 5 logical (80/20). Neither rater ever agreed with the other
on a "hallucination" label — in practice the taxonomy is two-class.

Transcription failures concentrate in spelling-heavy banks: emails (6),
coined names (4), properties (3). Every failing call got a consensus
explanation; no unexplained reward-0 calls.

## Fidelity judge — precision / recall / F1 (all 100 calls, 41 findings)

Strict = both raters confirm a finding; ground truth = defects both raters
marked. Lenient = either rater confirms; ground truth = defects either
rater marked. Recall is clip-level (did the judge flag the call at all).

| | precision | recall | F1 |
|---|---|---|---|
| strict | 0.73 | 1.00 | 0.85 |
| lenient | 0.85 | 0.80 | 0.82 |

On every call where both raters marked a fidelity defect (21 calls), the
judge had at least one confirmed finding. It missed 10 defects that only
one rater caught — mostly soft mispronunciations and small omissions.

By factor:

| factor | n | P strict | F1 strict | P lenient | R lenient | F1 lenient | rater agreement |
|---|---|---|---|---|---|---|---|
| email_url_code | 7 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| number_date_currency | 2 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| acronym_brand_name | 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| missing_word | 9 | 0.78 | 0.88 | 1.00 | 0.82 | 0.90 | 0.78 |
| mispronunciation | 12 | 0.75 | 0.86 | 0.83 | 0.67 | 0.74 | 0.92 |
| extra_or_hallucinated_word | 4 | 0.75 | 0.86 | 0.75 | 0.75 | 0.75 | 1.00 |
| word_substitution | 3 | 0.33 | 0.50 | 0.67 | 1.00 | 0.80 | 0.67 |
| punctuation_or_formatting | 3 | 0.00 | 0.00 | 0.33 | 0.50 | 0.40 | 0.67 |
| **overall** | **41** | **0.73** | **0.85** | **0.85** | **0.80** | **0.82** | **0.88 (κ 0.63)** |

Per-factor recall rests on mapping 9 single-rater catches to factors by
note text, and factor sample sizes are small (n = 1–12) — treat per-factor
F1 as directional. The two weak factors (punctuation_or_formatting,
word_substitution) are also the two where the humans themselves agree
least, pointing at rubric definitions rather than judge quality.

## Data

Raw returns (per-rater review and decision CSVs) live next to this file in
`data/annotation/intake_review_100_2026-08-27/` — see the README there for
file schemas and provenance.
