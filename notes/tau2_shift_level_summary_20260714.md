# tau2 Shift-Level Summary - 2026-07-14

## Dataset
- Input: `/Users/xuyida/Research/llm-toolcalling-benchmarks/tau2-bench/data/processed/tau2_l2t_success_retail_airline_50_filtered_20260714.pkl`
- Total N: 93
- Positive count (`y == 1`): 33
- Overall success rate: 0.354839
- Rates are proportions. `drop_pp` is percentage points, computed as `100 * (source_success_rate - target_success_rate)`.
- `harmful_candidate` is true when `drop_pp > 10`.

## Shift Summary
| shift_name | shift_type | source_group | target_group | source_n | target_n | source_positive | target_positive | source_success_rate | target_success_rate | drop_pp | harmful_candidate | notes |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| zero or one expected write -> two or more expected writes | action_type | zero or one expected write | two or more expected writes | 69 | 24 | 31 | 2 | 0.449275 | 0.083333 | 36.594203 | true | source expected_write_action_count <= 1; target >= 2 |
| short messages -> long messages | trajectory_complexity | short messages | long messages | 48 | 45 | 21 | 12 | 0.4375 | 0.266667 | 17.083333 | true | median num_messages=23; source <= median, target > median |
| no expected write actions -> expected write actions > 0 | action_type | no expected write actions | expected write actions > 0 | 26 | 67 | 12 | 21 | 0.461538 | 0.313433 | 14.810563 | true | source expected_write_action_count == 0; target > 0 |
| few tool calls -> many tool calls | tool_complexity | few tool calls | many tool calls | 51 | 42 | 21 | 12 | 0.411765 | 0.285714 | 12.605042 | true | median num_tool_calls=6; source <= median, target > median |
| retail -> airline | domain | retail | airline | 46 | 47 | 19 | 14 | 0.413043 | 0.297872 | 11.517114 | true | domain comparison |
| low expected actions -> high expected actions | action_type | low expected actions | high expected actions | 57 | 36 | 20 | 13 | 0.350877 | 0.361111 | -1.023392 | false | median expected_action_count=5; source <= median, target > median |
| low expected read actions -> high expected read actions | action_type | low expected read actions | high expected read actions | 58 | 35 | 19 | 14 | 0.327586 | 0.4 | -7.241379 | false | median expected_read_action_count=4; source <= median, target > median |

