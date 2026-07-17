# Tau2 Shift Uncertainty Analysis After Stage 1

## Purpose

Rerun the tau2-only uncertainty analysis after adding 12 retained Stage 1 retail records to the original 93 tau2 records. Baseline files are not overwritten, API-Bank data are not modified, and no predictive model is trained.

## Methods

The analysis uses the same six baseline tau2 shift definitions and thresholds, Newcombe-Wilson 95% confidence intervals, deterministic bootstrap with 10,000 replicates and seed 1, Fisher or two-proportion testing as previously implemented, Benjamini-Hochberg adjustment, and practical thresholds 0.05, 0.10, 0.15.

## Results

| Shift | Source n | Target n | Source rate | Target rate | Delta_y | 95% CI | d=0.05 | d=0.10 | d=0.15 |
|---|---:|---:|---:|---:|---:|---|---|---|---|
| tau2_retail_to_airline | 58 | 47 | 0.3966 | 0.2979 | -0.0987 | [-0.2687, 0.0844] | inconclusive | inconclusive | inconclusive |
| tau2_no_write_to_write_required | 29 | 76 | 0.4483 | 0.3158 | -0.1325 | [-0.3320, 0.0658] | inconclusive | inconclusive | inconclusive |
| tau2_zero_or_one_write_to_two_plus_writes | 73 | 32 | 0.4521 | 0.1250 | -0.3271 | [-0.4634, -0.1371] | candidate_harmful | candidate_harmful | inconclusive |
| tau2_few_to_many_expected_actions | 28 | 39 | 0.4643 | 0.3590 | -0.1053 | [-0.3263, 0.1252] | inconclusive | inconclusive | inconclusive |
| tau2_short_to_long_trajectory | 28 | 35 | 0.5357 | 0.2571 | -0.2786 | [-0.4833, -0.0371] | inconclusive | inconclusive | inconclusive |
| tau2_few_to_many_tool_calls | 26 | 33 | 0.4615 | 0.2424 | -0.2191 | [-0.4355, 0.0226] | inconclusive | inconclusive | inconclusive |

## Interpretation

Stage 1 task selection was targeted by X/task characteristics, not by observed outcomes. Stage 1 itself had 4 successes in 12 retained records. These results are exploratory, relatively small-sample estimates; they do not establish causality and do not prove that any shift is harmful or harmless.

## Summary

```json
{
  "bootstrap_configuration": {
    "replicates": 10000,
    "resampling": "within source and target groups separately",
    "seed": 1
  },
  "classification_counts_by_threshold": {
    "0.05": {
      "candidate_harmful": 1,
      "inconclusive": 5
    },
    "0.10": {
      "candidate_harmful": 1,
      "inconclusive": 5
    },
    "0.15": {
      "inconclusive": 6
    }
  },
  "confidence_interval_methods": {
    "bootstrap_delta_y_ci_95": "deterministic nonparametric bootstrap percentile interval",
    "primary_delta_y_ci_95": "Newcombe-Wilson score interval for difference in proportions"
  },
  "eligible_tau2_shift_count": 6,
  "exploratory_analysis_warning": "This is an exploratory small-sample tau2 analysis; the results do not imply causal effects and do not independently authorize deployment decisions.",
  "largest_group_size": 76,
  "multiple_testing_method": "Benjamini-Hochberg false-discovery-rate adjustment",
  "overlapping_group_shift_count": 0,
  "small_sample_warning_count": 4,
  "smallest_group_size": 26,
  "stable_classification_shifts": [
    "tau2_retail_to_airline",
    "tau2_no_write_to_write_required",
    "tau2_few_to_many_expected_actions",
    "tau2_short_to_long_trajectory",
    "tau2_few_to_many_tool_calls"
  ],
  "stage1_note": "Stage 1 task selection was targeted by X/task characteristics, not by observed outcomes; Stage 1 retained outcomes had 4 successes in 12 records.",
  "threshold_sensitive_classification_shifts": [
    "tau2_zero_or_one_write_to_two_plus_writes"
  ]
}
```
