# Shift Group Analysis - 2026-07-14

## 1. Dataset summary
- Input path: `/Users/xuyida/Research/llm-toolcalling-benchmarks/tau2-bench/data/processed/tau2_l2t_success_retail_airline_50_filtered_20260714.pkl`
- Total N: 93
- X shape: `(93, 12)`
- traj['s'] shape: `(93, 64)`
- Feature names: `['domain_retail', 'domain_airline', 'expected_action_count', 'expected_read_action_count', 'expected_write_action_count', 'requires_db_mutation', 'has_communication_checks', 'has_nl_assertions', 'has_env_assertions', 'reward_basis_has_DB', 'reward_basis_has_COMMUNICATE', 'reward_basis_has_NL_ASSERTION']`
- Metadata fields: `['db_match', 'domain', 'expected_action_count', 'expected_read_action_count', 'expected_write_action_count', 'num_messages', 'num_tool_calls', 'reward', 'source_result_folder', 'task_id', 'termination_reason']`
- Total positive y count: 33
- Overall success rate: 0.355 (35.5%)

## 2. Success rates by domain
| Group | N | Positive | Success rate |
| --- | ---: | ---: | ---: |
| retail | 46 | 19 | 41.3% |
| airline | 47 | 14 | 29.8% |

Retail minus airline drop: 11.5 pp

## 3. Success rates by write requirement
| Group | N | Positive | Success rate |
| --- | ---: | ---: | ---: |
| no expected write actions | 26 | 12 | 46.2% |
| expected write actions > 0 | 67 | 21 | 31.3% |

No-write minus write-required drop: 14.8 pp

Expected write count detail:

| Group | N | Positive | Success rate |
| --- | ---: | ---: | ---: |
| zero write | 26 | 12 | 46.2% |
| one write | 43 | 19 | 44.2% |
| two or more writes | 24 | 2 | 8.3% |

## 4. Success rates by message length
Threshold/median: 23

| Group | N | Positive | Success rate |
| --- | ---: | ---: | ---: |
| short messages | 48 | 21 | 43.8% |
| long messages | 45 | 12 | 26.7% |

Short minus long drop: 17.1 pp

## 5. Success rates by tool-call count
Threshold/median: 6

| Group | N | Positive | Success rate |
| --- | ---: | ---: | ---: |
| few tool calls | 51 | 21 | 41.2% |
| many tool calls | 42 | 12 | 28.6% |

Few minus many drop: 12.6 pp

## 6. Candidate harmful-shift ranking table
| Rank | Shift | Source group | Target group | Source N | Target N | Source success | Target success | Drop |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | short messages -> long messages | short messages | long messages | 48 | 45 | 43.8% | 26.7% | 17.1 pp |
| 2 | no-write -> write-required | no expected write actions | expected write actions > 0 | 26 | 67 | 46.2% | 31.3% | 14.8 pp |
| 3 | few tool calls -> many tool calls | few tool calls | many tool calls | 51 | 42 | 41.2% | 28.6% | 12.6 pp |
| 4 | retail -> airline | retail | airline | 46 | 47 | 41.3% | 29.8% | 11.5 pp |
| 5 | low expected actions -> high expected actions | low expected actions | high expected actions | 57 | 36 | 35.1% | 36.1% | -1.0 pp |
| 6 | low read count -> high read count | low read count | high read count | 58 | 35 | 32.8% | 40.0% | -7.2 pp |

## 7. Short interpretation
The largest candidate harmful shift is `short messages -> long messages`, where success falls from 43.8% in `short messages` to 26.7% in `long messages` (17.1 pp). These comparisons are descriptive groupings over the existing dataset and should be treated as candidate shift signals rather than causal explanations.

