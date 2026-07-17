# Tool-Calling Candidate Shift Inventory

## Formulation

This inventory defines deterministic source and target groups from existing metadata and structural features, then computes descriptive statistics only. Group membership is defined without using `y`, `variant`, `corruption_type`, `label_origin`, `is_synthetic`, validation status, or validation error fields.

`Delta_Y = P_target(Y=1) - P_source(Y=1)` is reported descriptively. A negative `delta_y` is not a final harmful-shift label, and a near-zero `delta_y` is not a final harmless-shift label. Final classification requires confidence intervals and a practical significance threshold.

## Grouping Rules

Minimum group size: 10.

- `tau2_retail_to_airline`: Source group has domain == retail; target group has domain == airline. Fields: domain. Thresholds: {}.
- `tau2_no_write_to_write_required`: Source has expected_write_action_count == 0; target has expected_write_action_count >= 1. Fields: x_numeric_features.expected_write_action_count. Thresholds: {"source_max": 0, "target_min": 1}.
- `tau2_zero_or_one_write_to_two_plus_writes`: Source has expected_write_action_count <= 1; target has expected_write_action_count >= 2. Fields: x_numeric_features.expected_write_action_count. Thresholds: {"source_max": 1, "target_min": 2}.
- `tau2_few_to_many_expected_actions`: Source is the lower quartile of x_numeric_features.expected_action_count; target is the upper quartile. Membership is defined before looking at y. Fields: x_numeric_features.expected_action_count. Thresholds: {"lower_quantile": 0.25, "lower_threshold": 1.0, "upper_quantile": 0.75, "upper_threshold": 6.0}.
- `tau2_short_to_long_trajectory`: Source is the lower quartile of s_structural_features.trajectory_length; target is the upper quartile. Membership is defined before looking at y. Fields: s_structural_features.trajectory_length. Thresholds: {"lower_quantile": 0.25, "lower_threshold": 19.0, "upper_quantile": 0.75, "upper_threshold": 33.0}.
- `tau2_few_to_many_tool_calls`: Source is the lower quartile of metadata.num_tool_calls; target is the upper quartile. Membership is defined before looking at y. Fields: metadata.num_tool_calls. Thresholds: {"lower_quantile": 0.25, "lower_threshold": 3.0, "upper_quantile": 0.75, "upper_threshold": 9.0}.
- `api_bank_no_auth_to_auth_required`: Source has no authentication signal in the pre-call context and candidate API; target has requires_authentication == 1 or an authentication-like API name. Fields: x_structural_features.requires_authentication, s_raw.api_name. Thresholds: {"source_value": 0, "target_value": 1}.
- `api_bank_short_to_long_dialogue_history`: Source is the lower quartile of x_numeric_features.history_length; target is the upper quartile. Membership is defined before looking at y. Fields: x_numeric_features.history_length. Thresholds: {"lower_quantile": 0.25, "lower_threshold": 3.0, "upper_quantile": 0.75, "upper_threshold": 8.0}.
- `api_bank_one_tool_to_multiple_tools_available`: Source has available_api_count == 1; target has available_api_count >= 2. Fields: x_numeric_features.available_api_count. Thresholds: {"source_value": 1, "target_min": 2}.
- `api_bank_few_to_many_arguments`: Source is the lower quartile of s_structural_features.argument_count; target is the upper quartile. Membership is defined before looking at y. Fields: s_structural_features.argument_count. Thresholds: {"lower_quantile": 0.25, "lower_threshold": 1.0, "upper_quantile": 0.75, "upper_threshold": 3.0}.
- `api_bank_simple_call_to_multi_step_context`: Source has previous_api_call_count == 0; target has previous_api_call_count >= 1 in the pre-call dialogue history. Fields: x_numeric_features.previous_api_call_count. Thresholds: {"source_value": 0, "target_min": 1}.
- `api_bank_domain_or_tool_family_comparison`: Domain/tool-family comparison is recorded as unsupported because the unified API-Bank records have no reliable domain metadata. Fields: domain. Thresholds: {"observed_non_null_domain_values": []}.

## Candidate Shifts By Dataset

### tau2

Outcome type: `real_benchmark_task_outcome`.

| Shift | Status | Source n | Target n | Delta_y | X distance | S distance | Direction |
|---|---:|---:|---:|---:|---:|---:|---|
| tau2_retail_to_airline | eligible | 46 | 47 | -0.1152 | 4.2353 | 4.6796 | candidate negative-outcome shift |
| tau2_no_write_to_write_required | eligible | 26 | 67 | -0.1481 | 4.1591 | 10.4401 | candidate negative-outcome shift |
| tau2_zero_or_one_write_to_two_plus_writes | eligible | 69 | 24 | -0.3659 | 4.3299 | 14.0699 | candidate negative-outcome shift |
| tau2_few_to_many_expected_actions | eligible | 25 | 36 | -0.0789 | 8.0470 | 12.6480 | candidate negative-outcome shift |
| tau2_short_to_long_trajectory | eligible | 27 | 28 | -0.2685 | 4.6089 | 28.4241 | candidate negative-outcome shift |
| tau2_few_to_many_tool_calls | eligible | 25 | 26 | -0.2092 | 4.9653 | 28.5383 | candidate negative-outcome shift |

### API-Bank

Outcome type: `synthetic_api_call_correctness`.

| Shift | Status | Source n | Target n | Delta_y | X distance | S distance | Direction |
|---|---:|---:|---:|---:|---:|---:|---|
| api_bank_no_auth_to_auth_required | eligible | 472 | 544 | 0.0000 | 151.2239 | 0.9240 | candidate stable-outcome shift |
| api_bank_short_to_long_dialogue_history | eligible | 276 | 350 | 0.0000 | 697.2969 | 1.4473 | candidate stable-outcome shift |
| api_bank_one_tool_to_multiple_tools_available | eligible | 212 | 804 | 0.0000 | 265.9932 | 0.4071 | candidate stable-outcome shift |
| api_bank_few_to_many_arguments | eligible | 374 | 306 | 0.1150 | 294.6998 | 2.8494 | candidate positive-outcome shift |
| api_bank_simple_call_to_multi_step_context | eligible | 522 | 494 | 0.0000 | 451.7896 | 0.7280 | candidate stable-outcome shift |
| api_bank_domain_or_tool_family_comparison | failed | 0 | 0 |  |  |  |  |

## Validity Warnings

- tau2 labels are real benchmark task outcomes.
- API-Bank labels are synthetic API-call correctness labels, not task-level deployment success labels.
- API-Bank delta_y values cannot be interpreted as real deployment success-rate shifts because the negative samples are synthetic corruptions and labels are balanced by construction.
- Task-level and API-call-level labels are kept separate and are not treated as equivalent.
- The terms candidate negative-outcome shift, candidate stable-outcome shift, and candidate positive-outcome shift are descriptive only.

## Later Analysis Suitability

Potentially suitable for later harmful/harmless analysis after confidence intervals and a practical threshold are defined:

- `tau2_retail_to_airline`
- `tau2_no_write_to_write_required`
- `tau2_zero_or_one_write_to_two_plus_writes`
- `tau2_few_to_many_expected_actions`
- `tau2_short_to_long_trajectory`
- `tau2_few_to_many_tool_calls`

Unsuitable for final harmful/harmless conclusions at this stage because labels are synthetic, domains are unavailable, or groups are too small:

- `api_bank_no_auth_to_auth_required`
- `api_bank_short_to_long_dialogue_history`
- `api_bank_one_tool_to_multiple_tools_available`
- `api_bank_few_to_many_arguments`
- `api_bank_simple_call_to_multi_step_context`
- `api_bank_domain_or_tool_family_comparison`
