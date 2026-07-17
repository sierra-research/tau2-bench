# Tau2 Stage 1 Shift Comparison

## Purpose

Compare the baseline tau2 shift uncertainty outputs with the versioned Stage 1 tau2-only outputs.

## Results

| Shift | Base n | Stage 1 n | Base rates | Stage 1 rates | Base delta | Stage 1 delta | Base CI | Stage 1 CI | CI widths | d=0.05 | d=0.10 | d=0.15 | Class changed | Stage 1 membership |
|---|---:|---:|---|---|---:|---:|---|---|---|---|---|---|---|---|
| tau2_retail_to_airline | 46/47 | 58/47 | 0.4130/0.2979 | 0.3966/0.2979 | -0.1152 | -0.0987 | [-0.2969, 0.0774] | [-0.2687, 0.0844] | 0.3742/0.3531; narrowed=True | inconclusive -> inconclusive | inconclusive -> inconclusive | inconclusive -> inconclusive | False | source=12, target=0, both=0, neither=0 |
| tau2_no_write_to_write_required | 26/67 | 29/76 | 0.4615/0.3134 | 0.4483/0.3158 | -0.1481 | -0.1325 | [-0.3567, 0.0625] | [-0.3320, 0.0658] | 0.4191/0.3978; narrowed=True | inconclusive -> inconclusive | inconclusive -> inconclusive | inconclusive -> inconclusive | False | source=3, target=9, both=0, neither=0 |
| tau2_zero_or_one_write_to_two_plus_writes | 69/24 | 73/32 | 0.4493/0.0833 | 0.4521/0.1250 | -0.3659 | -0.3271 | [-0.4975, -0.1583] | [-0.4634, -0.1371] | 0.3392/0.3264; narrowed=True | candidate_harmful -> candidate_harmful | candidate_harmful -> candidate_harmful | candidate_harmful -> inconclusive | True | source=4, target=8, both=0, neither=0 |
| tau2_few_to_many_expected_actions | 25/36 | 28/39 | 0.4400/0.3611 | 0.4643/0.3590 | -0.0789 | -0.1053 | [-0.3122, 0.1591] | [-0.3263, 0.1252] | 0.4714/0.4516; narrowed=True | inconclusive -> inconclusive | inconclusive -> inconclusive | inconclusive -> inconclusive | False | source=3, target=3, both=0, neither=6 |
| tau2_short_to_long_trajectory | 27/28 | 28/35 | 0.5185/0.2500 | 0.5357/0.2571 | -0.2685 | -0.2786 | [-0.4818, -0.0124] | [-0.4833, -0.0371] | 0.4694/0.4461; narrowed=True | inconclusive -> inconclusive | inconclusive -> inconclusive | inconclusive -> inconclusive | False | source=1, target=7, both=0, neither=4 |
| tau2_few_to_many_tool_calls | 25/26 | 26/33 | 0.4400/0.2308 | 0.4615/0.2424 | -0.2092 | -0.2191 | [-0.4336, 0.0478] | [-0.4355, 0.0226] | 0.4814/0.4581; narrowed=True | inconclusive -> inconclusive | inconclusive -> inconclusive | inconclusive -> inconclusive | False | source=1, target=7, both=0, neither=4 |

## Candidate-Harmful Shift

`tau2_zero_or_one_write_to_two_plus_writes` remains candidate_harmful at all three practical thresholds after adding the 12 targeted Stage 1 records: False. Post-Stage-1 classifications are d=0.05 `candidate_harmful`, d=0.10 `candidate_harmful`, and d=0.15 `inconclusive`.

## Interpretation

Stage 1 task selection was targeted by X/task characteristics rather than observed outcomes, and Stage 1 had 4/12 successes. The sample remains exploratory and relatively small. The comparison should not be interpreted causally and does not prove that any shift is harmful or harmless.

## Changes

- CI widths narrowed: tau2_retail_to_airline, tau2_no_write_to_write_required, tau2_zero_or_one_write_to_two_plus_writes, tau2_few_to_many_expected_actions, tau2_short_to_long_trajectory, tau2_few_to_many_tool_calls.
- Classifications changed: tau2_zero_or_one_write_to_two_plus_writes.
