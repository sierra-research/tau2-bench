# BFCL Shift Uncertainty

## Purpose

Estimate statistical uncertainty for BFCL v4 non-live single-turn category contrasts using the already evaluated 1,240 sample-level labels. This is exploratory analysis only.

## Statistical Formulation

For each source/target category contrast, `Delta_Y = P_target(Y=1) - P_source(Y=1)`. The primary interval is a 95% Newcombe-Wilson difference-in-proportions interval. A deterministic nonparametric bootstrap interval uses 10,000 replicates with seed 1, resampling within source and target groups separately. Equality-of-proportions p-values use Fisher's exact test when expected counts are small and a two-proportion z-test otherwise, followed by Benjamini-Hochberg adjustment across the analyzed BFCL shifts.

## Practical-Significance Thresholds

Classifications are reported separately for `delta_practical = [0.05, 0.10, 0.15]`.

- `candidate_harmful`: upper 95% CI < `-d`
- `candidate_harmless`: full 95% CI is inside `[-d, d]`
- `candidate_beneficial`: lower 95% CI > `d`
- `inconclusive`: all other cases

## Results Table

| Shift | Type | Source n | Target n | Source rate | Target rate | Delta_y | 95% CI | Bootstrap 95% CI | Raw p | BH p | Test | d=0.05 | d=0.10 | d=0.15 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| bfcl_simple_python_to_multiple | primary_complexity | 400 | 200 | 0.8750 | 0.8800 | 0.0050 | [-0.0548, 0.0574] | [-0.0500, 0.0575] | 0.8606 | 0.8622 | two_proportion_z_test | inconclusive | candidate_harmless | candidate_harmless |
| bfcl_simple_python_to_parallel | primary_complexity | 400 | 200 | 0.8750 | 0.8700 | -0.0050 | [-0.0659, 0.0486] | [-0.0625, 0.0500] | 0.8622 | 0.8622 | two_proportion_z_test | inconclusive | candidate_harmless | candidate_harmless |
| bfcl_simple_python_to_parallel_multiple | primary_complexity | 400 | 200 | 0.8750 | 0.8000 | -0.0750 | [-0.1424, -0.0137] | [-0.1400, -0.0100] | 0.0153 | 0.0873 | two_proportion_z_test | inconclusive | inconclusive | candidate_harmless |
| bfcl_multiple_to_parallel_multiple | primary_complexity | 200 | 200 | 0.8800 | 0.8000 | -0.0800 | [-0.1518, -0.0079] | [-0.1500, -0.0100] | 0.0291 | 0.0873 | two_proportion_z_test | inconclusive | inconclusive | inconclusive |
| bfcl_parallel_to_parallel_multiple | primary_complexity | 200 | 200 | 0.8700 | 0.8000 | -0.0700 | [-0.1427, 0.0030] | [-0.1400, 0.0000] | 0.0593 | 0.1186 | two_proportion_z_test | inconclusive | inconclusive | candidate_harmless |
| bfcl_simple_python_to_irrelevance | behavioral_abstention | 400 | 240 | 0.8750 | 0.8292 | -0.0458 | [-0.1059, 0.0098] | [-0.1033, 0.0108] | 0.1080 | 0.1620 | two_proportion_z_test | inconclusive | inconclusive | candidate_harmless |

## Classification Summary by Threshold

### d=0.05
- `candidate_harmful`: None.
- `candidate_harmless`: None.
- `candidate_beneficial`: None.
- `inconclusive`: `bfcl_simple_python_to_multiple`, `bfcl_simple_python_to_parallel`, `bfcl_simple_python_to_parallel_multiple`, `bfcl_multiple_to_parallel_multiple`, `bfcl_parallel_to_parallel_multiple`, `bfcl_simple_python_to_irrelevance`

### d=0.10
- `candidate_harmful`: None.
- `candidate_harmless`: `bfcl_simple_python_to_multiple`, `bfcl_simple_python_to_parallel`
- `candidate_beneficial`: None.
- `inconclusive`: `bfcl_simple_python_to_parallel_multiple`, `bfcl_multiple_to_parallel_multiple`, `bfcl_parallel_to_parallel_multiple`, `bfcl_simple_python_to_irrelevance`

### d=0.15
- `candidate_harmful`: None.
- `candidate_harmless`: `bfcl_simple_python_to_multiple`, `bfcl_simple_python_to_parallel`, `bfcl_simple_python_to_parallel_multiple`, `bfcl_parallel_to_parallel_multiple`, `bfcl_simple_python_to_irrelevance`
- `candidate_beneficial`: None.
- `inconclusive`: `bfcl_multiple_to_parallel_multiple`

## Constraints

- No causal claims are made.
- No deployment-safe or retraining-required claims are made.
- BFCL rows are not mixed with tau2 or API-Bank as IID samples.
- `label_scope=test_case_level` and `label_origin=bfcl_evaluator` are preserved.
- Candidate groups are defined without using `y`.
- The analysis does not use partial BFCL leaderboard overall scores.
- The `simple_python -> irrelevance` contrast is reported as behavioral/abstention, not as a primary complexity shift.

## Limitations

The category contrasts are exploratory and reuse some category groups across multiple candidate shifts. The intervals describe uncertainty in this processed 1,240-row BFCL subset and should not be interpreted as causal evidence about context changes.
