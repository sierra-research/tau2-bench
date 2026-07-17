# Tau2 Additional Sampling Plan

## Current evidence

The current tau2 analysis contains 6 eligible shifts and 93 retained real task-level outcomes. API-Bank is excluded from this plan because its negative labels are synthetic API-call corruptions.

| Shift | Source n | Target n | Source rate | Target rate | Delta_y | 95% CI | CI width | d=0.05 | d=0.10 | d=0.15 |
|---|---:|---:|---:|---:|---:|---|---:|---|---|---|
| tau2_retail_to_airline | 46 | 47 | 0.4130 | 0.2979 | -0.1152 | [-0.2969, 0.0774] | 0.3742 | inconclusive | inconclusive | inconclusive |
| tau2_no_write_to_write_required | 26 | 67 | 0.4615 | 0.3134 | -0.1481 | [-0.3567, 0.0625] | 0.4191 | inconclusive | inconclusive | inconclusive |
| tau2_zero_or_one_write_to_two_plus_writes | 69 | 24 | 0.4493 | 0.0833 | -0.3659 | [-0.4975, -0.1583] | 0.3392 | candidate_harmful | candidate_harmful | candidate_harmful |
| tau2_few_to_many_expected_actions | 25 | 36 | 0.4400 | 0.3611 | -0.0789 | [-0.3122, 0.1591] | 0.4714 | inconclusive | inconclusive | inconclusive |
| tau2_short_to_long_trajectory | 27 | 28 | 0.5185 | 0.2500 | -0.2685 | [-0.4818, -0.0124] | 0.4694 | inconclusive | inconclusive | inconclusive |
| tau2_few_to_many_tool_calls | 25 | 26 | 0.4400 | 0.2308 | -0.2092 | [-0.4336, 0.0478] | 0.4814 | inconclusive | inconclusive | inconclusive |

## Candidate-harmful shift

`tau2_zero_or_one_write_to_two_plus_writes`

This shift should be prioritized for additional real tau2 outcomes, but the next collection should still be staged rather than collecting the full calculated sample size in one pass.

## Inconclusive shifts

`tau2_retail_to_airline`, `tau2_no_write_to_write_required`, `tau2_few_to_many_expected_actions`, `tau2_short_to_long_trajectory`, `tau2_few_to_many_tool_calls`

## Precision-based sample estimates

Planning estimates use standard normal approximations for two independent proportions. They are planning estimates, not guarantees.

| Shift | CI half-width | Required final n/group | Add source | Add target | Add total |
|---|---:|---:|---:|---:|---:|
| tau2_retail_to_airline | 0.15 | 78 | 32 | 31 | 63 |
| tau2_retail_to_airline | 0.10 | 174 | 128 | 127 | 255 |
| tau2_retail_to_airline | 0.05 | 694 | 648 | 647 | 1295 |
| tau2_no_write_to_write_required | 0.15 | 80 | 54 | 13 | 67 |
| tau2_no_write_to_write_required | 0.10 | 179 | 153 | 112 | 265 |
| tau2_no_write_to_write_required | 0.05 | 713 | 687 | 646 | 1333 |
| tau2_zero_or_one_write_to_two_plus_writes | 0.15 | 56 | 0 | 32 | 32 |
| tau2_zero_or_one_write_to_two_plus_writes | 0.10 | 125 | 56 | 101 | 157 |
| tau2_zero_or_one_write_to_two_plus_writes | 0.05 | 498 | 429 | 474 | 903 |
| tau2_few_to_many_expected_actions | 0.15 | 82 | 57 | 46 | 103 |
| tau2_few_to_many_expected_actions | 0.10 | 184 | 159 | 148 | 307 |
| tau2_few_to_many_expected_actions | 0.05 | 734 | 709 | 698 | 1407 |
| tau2_short_to_long_trajectory | 0.15 | 75 | 48 | 47 | 95 |
| tau2_short_to_long_trajectory | 0.10 | 168 | 141 | 140 | 281 |
| tau2_short_to_long_trajectory | 0.05 | 672 | 645 | 644 | 1289 |
| tau2_few_to_many_tool_calls | 0.15 | 73 | 48 | 47 | 95 |
| tau2_few_to_many_tool_calls | 0.10 | 163 | 138 | 137 | 275 |
| tau2_few_to_many_tool_calls | 0.05 | 652 | 627 | 626 | 1253 |

## Power-based sample estimates

Power estimates use two-sided alpha = 0.05 and power = 0.80.

| Shift | Effect size | Required final n/group | Add source | Add target | Add total |
|---|---:|---:|---:|---:|---:|
| tau2_retail_to_airline | 0.15 | 159 | 113 | 112 | 225 |
| tau2_retail_to_airline | 0.10 | 359 | 313 | 312 | 625 |
| tau2_retail_to_airline | 0.05 | 1437 | 1391 | 1390 | 2781 |
| tau2_no_write_to_write_required | 0.15 | 159 | 133 | 92 | 225 |
| tau2_no_write_to_write_required | 0.10 | 359 | 333 | 292 | 625 |
| tau2_no_write_to_write_required | 0.05 | 1437 | 1411 | 1370 | 2781 |
| tau2_zero_or_one_write_to_two_plus_writes | 0.15 | 159 | 90 | 135 | 225 |
| tau2_zero_or_one_write_to_two_plus_writes | 0.10 | 359 | 290 | 335 | 625 |
| tau2_zero_or_one_write_to_two_plus_writes | 0.05 | 1437 | 1368 | 1413 | 2781 |
| tau2_few_to_many_expected_actions | 0.15 | 166 | 141 | 130 | 271 |
| tau2_few_to_many_expected_actions | 0.10 | 374 | 349 | 338 | 687 |
| tau2_few_to_many_expected_actions | 0.05 | 1498 | 1473 | 1462 | 2935 |
| tau2_short_to_long_trajectory | 0.15 | 164 | 137 | 136 | 273 |
| tau2_short_to_long_trajectory | 0.10 | 370 | 343 | 342 | 685 |
| tau2_short_to_long_trajectory | 0.05 | 1481 | 1454 | 1453 | 2907 |
| tau2_few_to_many_tool_calls | 0.15 | 154 | 129 | 128 | 257 |
| tau2_few_to_many_tool_calls | 0.10 | 348 | 323 | 322 | 645 |
| tau2_few_to_many_tool_calls | 0.05 | 1395 | 1370 | 1369 | 2739 |

## Available task-pool audit

- `retail`: 114 local tasks; 46 retained outcomes; 68 without retained outcomes.
- `airline`: 50 local tasks; 47 retained outcomes; 3 without retained outcomes.
- `telecom`: 2285 local tasks; 0 retained outcomes; 2285 without retained outcomes.
- `banking_knowledge`: 97 local tasks; 0 retained outcomes; 97 without retained outcomes.
- Airline all local tasks previously sampled: True.
- Retail unused by retained outcome: 68; retail not previously attempted in the 2026-07-14 run: 64.
- `tau2_retail_to_airline`: smaller group `source`; unused retail source=68, target=0, unknown=0; repeated trials required now: False.
- `tau2_no_write_to_write_required`: smaller group `source`; unused retail source=5, target=63, unknown=0; repeated trials required now: False.
- `tau2_zero_or_one_write_to_two_plus_writes`: smaller group `target`; unused retail source=36, target=32, unknown=0; repeated trials required now: False.
- `tau2_few_to_many_expected_actions`: smaller group `source`; unused retail source=20, target=12, unknown=36; repeated trials required now: False.
- `tau2_short_to_long_trajectory`: smaller group `source`; shift uses observed trajectory length or observed tool-call count; unused task membership is unknown before running; repeated trials required now: True.
- `tau2_few_to_many_tool_calls`: smaller group `source`; shift uses observed trajectory length or observed tool-call count; unused task membership is unknown before running; repeated trials required now: True.

Task IDs can be selected without using outcome `y` for the proposed Stage 1 batch. Observed trajectory-length and observed tool-call-count membership is unavailable for unused tasks until new runs produce trajectories.

## Recommended collection priority

- Prioritize tau2_zero_or_one_write_to_two_plus_writes because it is currently candidate_harmful at all practical thresholds.
- Use unused retail tasks first, especially tasks with two or more expected write actions, because they increase the smaller target group for the candidate-harmful write-complexity shift.
- Add a small number of no-write and few-action retail tasks to preserve coverage for inconclusive shifts with smaller source groups.
- Reassess confidence intervals after Stage 1 before expanding.

## Proposed next batch

Stage 1: collect 12 unused retail tasks, then rerun the tau2 uncertainty analysis and reassess confidence intervals. Stage 2 should expand only if the Stage 1 evidence remains decision-relevant and inconclusive.

| Domain | Task ID | Expected actions | Expected writes | Selection reason |
|---|---:|---:|---:|---|
| retail | 64 | 8 | 2 | retail; unused by prior simulation; two or more expected write actions |
| retail | 54 | 12 | 3 | retail; unused by prior simulation; two or more expected write actions |
| retail | 55 | 13 | 4 | retail; unused by prior simulation; two or more expected write actions |
| retail | 71 | 2 | 2 | retail; unused by prior simulation; two or more expected write actions |
| retail | 72 | 2 | 2 | retail; unused by prior simulation; two or more expected write actions |
| retail | 74 | 2 | 2 | retail; unused by prior simulation; two or more expected write actions |
| retail | 76 | 2 | 2 | retail; unused by prior simulation; two or more expected write actions |
| retail | 81 | 2 | 2 | retail; unused by prior simulation; two or more expected write actions |
| retail | 57 | 0 | 0 | retail; unused by prior simulation; no expected write actions |
| retail | 62 | 5 | 0 | retail; unused by prior simulation; no expected write actions |
| retail | 50 | 1 | 1 | retail; unused by prior simulation; few expected actions |
| retail | 70 | 1 | 1 | retail; unused by prior simulation; few expected actions |

## Cost and runtime constraints

Telecom is technically runnable, but the current feasibility test was slow, rate-limited, reached max steps, and cost approximately $0.095 for one unsuccessful task. Do not recommend large telecom runs without a separate cost-control plan.

## Limitations

- Planning estimates use standard normal approximations for two independent proportions. They are planning estimates, not guarantees.
- Current tau2 evidence has only 93 retained real task-level outcomes.
- Precision and power estimates assume independent Bernoulli outcomes and stable success rates.
- Equivalence estimates are CI-screening approximations where available, not rigorous TOST power guarantees.
- Observed trajectory-length and tool-call-count groups cannot be assigned to unused tasks before running them.
- No task is selected based on observed success or failure.
