# Tau2 Shift Uncertainty Analysis

## Purpose

Estimate statistical uncertainty for the eligible tau2 candidate shifts only. API-Bank rows are excluded because their labels include synthetic corrupted negatives and do not support harmful/harmless deployment claims.

## Statistical formulation

For each preserved source/target definition, `Delta_Y = P_target(Y=1) - P_source(Y=1)`. The primary interval is a 95% Newcombe-Wilson difference-in-proportions interval. A deterministic nonparametric bootstrap interval uses 10,000 replicates with seed 1, resampling within source and target groups separately. Equality-of-proportions p-values use Fisher's exact test when expected counts are small and a two-proportion z-test otherwise, followed by Benjamini-Hochberg adjustment across the eligible tau2 shifts.

## Practical-significance thresholds

Classifications are reported separately for `delta_practical = [0.05, 0.10, 0.15]`. A shift is `candidate_harmful` only when the full primary 95% CI is below `-d`, `candidate_harmless` only when the full CI lies within `[-d, +d]`, and `candidate_beneficial` only when the full CI is above `+d`. All other cases are `inconclusive`.

## Results table

| Shift | Source n | Target n | Source rate | Target rate | Delta_y | 95% CI | Raw p | BH p | d=0.05 | d=0.10 | d=0.15 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| tau2_retail_to_airline | 46 | 47 | 0.4130 | 0.2979 | -0.1152 | [-0.2969, 0.0774] | 0.2458 | 0.2950 | inconclusive | inconclusive | inconclusive |
| tau2_no_write_to_write_required | 26 | 67 | 0.4615 | 0.3134 | -0.1481 | [-0.3567, 0.0625] | 0.1803 | 0.2705 | inconclusive | inconclusive | inconclusive |
| tau2_zero_or_one_write_to_two_plus_writes | 69 | 24 | 0.4493 | 0.0833 | -0.3659 | [-0.4975, -0.1583] | 0.0012 | 0.0075 | candidate_harmful | candidate_harmful | candidate_harmful |
| tau2_few_to_many_expected_actions | 25 | 36 | 0.4400 | 0.3611 | -0.0789 | [-0.3122, 0.1591] | 0.5351 | 0.5351 | inconclusive | inconclusive | inconclusive |
| tau2_short_to_long_trajectory | 27 | 28 | 0.5185 | 0.2500 | -0.2685 | [-0.4818, -0.0124] | 0.0405 | 0.1214 | inconclusive | inconclusive | inconclusive |
| tau2_few_to_many_tool_calls | 25 | 26 | 0.4400 | 0.2308 | -0.2092 | [-0.4336, 0.0478] | 0.1131 | 0.2261 | inconclusive | inconclusive | inconclusive |

## Classification sensitivity

| Shift | Classification stability | d=0.05 | d=0.10 | d=0.15 |
|---|---|---|---|---|
| tau2_retail_to_airline | stable_across_thresholds | inconclusive | inconclusive | inconclusive |
| tau2_no_write_to_write_required | stable_across_thresholds | inconclusive | inconclusive | inconclusive |
| tau2_zero_or_one_write_to_two_plus_writes | stable_across_thresholds | candidate_harmful | candidate_harmful | candidate_harmful |
| tau2_few_to_many_expected_actions | stable_across_thresholds | inconclusive | inconclusive | inconclusive |
| tau2_short_to_long_trajectory | stable_across_thresholds | inconclusive | inconclusive | inconclusive |
| tau2_few_to_many_tool_calls | stable_across_thresholds | inconclusive | inconclusive | inconclusive |

Stable across all three thresholds:

- `tau2_retail_to_airline`
- `tau2_no_write_to_write_required`
- `tau2_zero_or_one_write_to_two_plus_writes`
- `tau2_few_to_many_expected_actions`
- `tau2_short_to_long_trajectory`
- `tau2_few_to_many_tool_calls`

Changes with threshold:

None.

## Candidate harmful shifts

- `tau2_zero_or_one_write_to_two_plus_writes`

## Candidate harmless shifts

None.

## Candidate beneficial shifts

None.

## Inconclusive shifts

- `tau2_retail_to_airline`
- `tau2_no_write_to_write_required`
- `tau2_few_to_many_expected_actions`
- `tau2_short_to_long_trajectory`
- `tau2_few_to_many_tool_calls`

## Small-sample and overlap warnings

- `tau2_retail_to_airline`: Candidate shifts reuse tau2 records across non-independent exploratory definitions.
- `tau2_no_write_to_write_required`: Candidate shifts reuse tau2 records across non-independent exploratory definitions.; small group size warning: source_n=26, target_n=67, threshold=30
- `tau2_zero_or_one_write_to_two_plus_writes`: Candidate shifts reuse tau2 records across non-independent exploratory definitions.; small group size warning: source_n=69, target_n=24, threshold=30
- `tau2_few_to_many_expected_actions`: Candidate shifts reuse tau2 records across non-independent exploratory definitions.; small group size warning: source_n=25, target_n=36, threshold=30
- `tau2_short_to_long_trajectory`: Candidate shifts reuse tau2 records across non-independent exploratory definitions.; small group size warning: source_n=27, target_n=28, threshold=30; unstable interval warning: CI width exceeds 0.50
- `tau2_few_to_many_tool_calls`: Candidate shifts reuse tau2 records across non-independent exploratory definitions.; small group size warning: source_n=25, target_n=26, threshold=30; unstable interval warning: CI width exceeds 0.50

## Retraining interpretation

`candidate_harmful` may motivate additional evaluation or adaptation. `candidate_harmless` is evidence consistent with a practically small outcome change and may suggest retraining is not immediately justified, but it is not proof that retraining is unnecessary. `inconclusive` means more data are needed. These results do not independently authorize a deployment decision.

## Limitations

The dataset is small, the analysis is exploratory, candidate shift definitions reuse records and are not independent, and the estimates should not be interpreted causally. No predictive model is trained here, and group membership is not redefined using `y`.

## Next step

Collect additional real tau2-style task outcomes for the most decision-relevant shifts, then rerun this uncertainty analysis before deciding whether adaptation or retraining is warranted.
