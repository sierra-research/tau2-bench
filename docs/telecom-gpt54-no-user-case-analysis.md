# Telecom GPT-5.4 No-User Case Analysis

This note summarizes the local telecom no-user comparison between Qwen 3.6 27B and Azure GPT-5.4 with no reasoning. The goal is to identify whether the GPT-5.4 regressions look like model errors, task/evaluation issues, or run-configuration artifacts.

## Runs Compared

| Run | Result file |
| --- | --- |
| Qwen 3.6 27B, no reasoning, no user, max steps 40 | `data/simulations/telecom-qwen36-noreason-nouser-sessionstarted-max40-mc8-20260522-155743/results.json` |
| Azure GPT-5.4, no reasoning, no user, max steps 40 | `data/simulations/telecom-azure-gpt54-noreason-none-nouser-sessionstarted-max40-mc8-20260522-161024/results.json` |

Both runs contain 114 common telecom tasks with `dummy_user` and one trial.

## Score Summary

| Metric | Qwen 3.6 27B | Azure GPT-5.4 |
| --- | ---: | ---: |
| Passed tasks | 97 / 114 | 81 / 114 |
| Pass rate | 85.1% | 71.1% |

Pairwise deltas:

| Delta type | Count |
| --- | ---: |
| Qwen passed, GPT-5.4 failed | 20 |
| GPT-5.4 passed, Qwen failed | 4 |
| Both failed | 13 |

Azure GPT-5.4 failures by family:

| Family | Failed | Total |
| --- | ---: | ---: |
| `mobile_data_issue` | 1 | 36 |
| `service_issue` | 2 | 29 |
| `mms_issue` | 30 | 49 |

The strongest pattern is inside MMS:

| Subset | Qwen 3.6 27B | Azure GPT-5.4 |
| --- | ---: | ---: |
| MMS with `data_usage_exceeded` | 17 / 30 | 0 / 30 |
| MMS without `data_usage_exceeded` | 18 / 19 | 19 / 19 |

## Main Finding

The majority of the GPT-5.4 delta is not a general MMS failure. GPT-5.4 solved every MMS task without `data_usage_exceeded`, but failed every MMS task that included it.

This looks like a task/evaluation-policy tension:

- MMS success depends on mobile data working. `TelecomUserTools._can_send_mms()` returns false if `_get_mobile_data_working()` is false, and `_get_mobile_data_working()` returns false when `mobile_data_usage_exceeded` is true.
- `TelecomEnvironment.sync_tools()` sets `mobile_data_usage_exceeded` when line usage is greater than or equal to `plan.data_limit_gb + line.data_refueling_gb`.
- The gold fix for `data_usage_exceeded` is `assistant:refuel_data(customer_id="C1001", line_id="L1002", gb_amount=2.0)`.
- The policy says the agent should confirm the refuel price before applying the refuel. The non-solo policy also says to ask how much data the user wants to refuel.
- In no-user mode, there is no live user turn to provide that paid-action confirmation. The user scenario says the simulated user is willing to refuel 2.0 GB if necessary, but the agent-facing ticket often does not make that authorization explicit.

GPT-5.4 appears to behave conservatively around this paid action. In representative failed cases it finds the data-cap condition, notes that a refuel would be needed, but transfers instead of calling `refuel_data` because the amount or charge was not confirmed. Qwen is more willing to directly call `refuel_data`, which happens to satisfy the gold action.

## Mismatch Categories

| Category | Count | Assessment |
| --- | ---: | --- |
| MMS data cap / missing `refuel_data` | 16 | Likely dataset or no-user harness issue. The gold action requires a paid refuel, while policy asks for price confirmation. |
| Max-step cap | 1 | Run-configuration artifact. The task reached `max_steps` before evaluation. |
| Wrong line plus missing refuel/roaming | 1 | Model/tool-affordance issue. GPT-5.4 used `L1001` even though the phone number maps to `L1002`. |
| Service/APN path missed | 2 | Model behavior issue. GPT-5.4 resolved billing/SIM pieces but missed APN reset actions. |

## Representative Cases

### MMS data cap and paid-action confirmation

Task:

`[mms_issue]break_app_storage_permission|data_usage_exceeded[PERSONA:Easy]`

Observed GPT-5.4 path:

- Identified John Smith and the target phone number.
- Granted the missing messaging storage permission.
- Eventually checked line `L1002`.
- Found usage `15.1 GB / 15.0 GB` with `0.0 GB` refueled.
- Transferred instead of calling `refuel_data`, saying the ticket did not specify/confirm the refuel amount or charge acceptance.

Gold expectation:

- `assistant:refuel_data(customer_id="C1001", line_id="L1002", gb_amount=2.0)`

Assessment:

This is the clearest evidence that GPT-5.4 is being penalized for following a conservative interpretation of the paid-action policy in a no-user setting.

### Wrong line in a multi-line account

Task:

`[mobile_data_issue]bad_vpn|data_mode_off|data_usage_exceeded|user_abroad_roaming_disabled_off[PERSONA:None]`

Observed GPT-5.4 path:

- The affected phone number is `555-123-2002`, which maps to `L1002`.
- GPT-5.4 enabled roaming and checked data usage for `L1001`.
- It concluded data usage was not over the limit and transferred.

Gold expectation:

- Use `L1002`, then call `enable_roaming` and `refuel_data` for `L1002`.

Assessment:

This looks like a real model/tooling failure rather than a dataset issue. The account has multiple lines, and the available tools make it easy to inspect the wrong line unless the agent carefully maps phone number to line id.

### Service issue misses APN reset

Tasks:

- `[service_issue]break_apn_settings|overdue_bill_suspension|unseat_sim_card[PERSONA:Hard]`
- `[service_issue]airplane_mode_on|break_apn_settings|overdue_bill_suspension|unseat_sim_card[PERSONA:None]`

Observed GPT-5.4 path:

- Paid the overdue bill.
- Resumed the suspended line.
- Reseated SIM / handled airplane mode where applicable.
- Transferred while service was still not connected.
- Missed the APN reset action required by the gold evaluation.

Assessment:

These are likely model behavior misses. The task contains multiple independent service blockers, and GPT-5.4 stopped after resolving billing/SIM symptoms.

### Max-step cap

Task:

`[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_app_storage_permission|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:Hard]`

Observed GPT-5.4 path:

- Took many correct repair actions.
- Hit `max_steps`.
- Reward was forced to 0 with premature termination metadata, so normal env/action checks were not evaluated.

Assessment:

This should be treated separately from correctness failures. It indicates that high-complexity composite MMS tasks may still be sensitive to the max-step setting.

## Full Qwen-Pass / GPT-5.4-Fail List

| Category | Task | GPT-5.4 termination | Missing gold actions |
| --- | --- | --- | --- |
| Max-step cap | `[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_app_storage_permission|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:Hard]` | max_steps | n/a |
| MMS data cap / missing refuel | `[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_on[PERSONA:Easy]` | agent_stop | assistant:refuel_data |
| MMS data cap / missing refuel | `[mms_issue]airplane_mode_on|bad_network_preference|break_apn_mms_setting|break_app_both_permissions|data_mode_off|data_usage_exceeded|user_abroad_roaming_enabled_off[PERSONA:None]` | agent_stop | assistant:refuel_data |
| MMS data cap / missing refuel | `[mms_issue]airplane_mode_on|bad_network_preference|break_apn_mms_setting|data_usage_exceeded[PERSONA:Hard]` | agent_stop | assistant:refuel_data |
| MMS data cap / missing refuel | `[mms_issue]airplane_mode_on|bad_network_preference|break_app_both_permissions|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_on[PERSONA:Hard]` | agent_stop | assistant:refuel_data |
| MMS data cap / missing refuel | `[mms_issue]airplane_mode_on|bad_network_preference|data_usage_exceeded[PERSONA:None]` | agent_stop | assistant:refuel_data |
| MMS data cap / missing refuel | `[mms_issue]airplane_mode_on|break_app_both_permissions|data_usage_exceeded|user_abroad_roaming_disabled_off[PERSONA:None]` | agent_stop | assistant:refuel_data |
| MMS data cap / missing refuel | `[mms_issue]bad_network_preference|bad_wifi_calling|break_app_both_permissions|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_off[PERSONA:None]` | agent_stop | assistant:refuel_data |
| MMS data cap / missing refuel | `[mms_issue]bad_network_preference|bad_wifi_calling|break_app_both_permissions|data_usage_exceeded|user_abroad_roaming_disabled_off[PERSONA:Hard]` | agent_stop | assistant:refuel_data |
| MMS data cap / missing refuel | `[mms_issue]bad_network_preference|break_app_sms_permission|data_mode_off|data_usage_exceeded|user_abroad_roaming_enabled_off[PERSONA:None]` | agent_stop | assistant:refuel_data |
| MMS data cap / missing refuel | `[mms_issue]bad_wifi_calling|break_apn_mms_setting|break_app_sms_permission|data_usage_exceeded[PERSONA:Hard]` | agent_stop | assistant:refuel_data |
| MMS data cap / missing refuel | `[mms_issue]bad_wifi_calling|break_apn_mms_setting|data_mode_off|data_usage_exceeded|unseat_sim_card[PERSONA:None]` | agent_stop | assistant:refuel_data |
| MMS data cap / missing refuel | `[mms_issue]bad_wifi_calling|data_mode_off|data_usage_exceeded[PERSONA:Easy]` | agent_stop | assistant:refuel_data |
| MMS data cap / missing refuel | `[mms_issue]break_apn_mms_setting|data_mode_off|data_usage_exceeded|user_abroad_roaming_disabled_on[PERSONA:Hard]` | agent_stop | assistant:refuel_data |
| MMS data cap / missing refuel | `[mms_issue]break_app_both_permissions|data_usage_exceeded[PERSONA:Hard]` | agent_stop | assistant:refuel_data |
| MMS data cap / missing refuel | `[mms_issue]break_app_sms_permission|data_usage_exceeded|user_abroad_roaming_disabled_on[PERSONA:None]` | agent_stop | assistant:refuel_data |
| MMS data cap / missing refuel | `[mms_issue]break_app_storage_permission|data_usage_exceeded[PERSONA:Easy]` | agent_stop | assistant:refuel_data |
| Wrong line plus missing refuel/roaming | `[mobile_data_issue]bad_vpn|data_mode_off|data_usage_exceeded|user_abroad_roaming_disabled_off[PERSONA:None]` | agent_stop | assistant:refuel_data, assistant:enable_roaming |
| Service/APN path missed | `[service_issue]airplane_mode_on|break_apn_settings|overdue_bill_suspension|unseat_sim_card[PERSONA:None]` | agent_stop | user:reset_apn_settings, user:reboot_device, user:reboot_device |
| Service/APN path missed | `[service_issue]break_apn_settings|overdue_bill_suspension|unseat_sim_card[PERSONA:Hard]` | agent_stop | user:reset_apn_settings |

## GPT-5.4-Pass / Qwen-Fail List

| Task | Qwen termination |
| --- | --- |
| `[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_both_permissions|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:Hard]` | max_steps |
| `[service_issue]airplane_mode_on|break_apn_settings|contract_end_suspension|unseat_sim_card[PERSONA:Easy]` | agent_stop |
| `[service_issue]airplane_mode_on|break_apn_settings|unseat_sim_card[PERSONA:None]` | max_steps |
| `[service_issue]contract_end_suspension|unseat_sim_card[PERSONA:Hard]` | agent_stop |

## Recommendations

1. Make paid-action authorization explicit in no-user telecom tasks that expect `refuel_data`.

   For `data_usage_exceeded` tasks, especially MMS tasks, add agent-visible ticket text such as "The customer has already authorized adding 2.0 GB of data and accepts the associated charge if data usage is the blocker." This aligns the gold action with the policy requirement to know the amount and confirm the price.

2. Alternatively, change the gold evaluation for no-user paid actions.

   If no-user mode should test strict policy compliance, the benchmark should not require `refuel_data` unless the run includes a simulated confirmation step. In that setup, a conservative transfer or request for confirmation should not be scored as the same type of failure as missing a free device-side fix.

3. Add a phone-number-specific data/line helper or strengthen line-selection instructions.

   The wrong-line case shows that multi-line accounts can produce false failures. A helper such as `get_line_by_phone_number` or `get_data_usage_by_phone_number` would reduce accidental use of `L1001` when the affected phone number maps to `L1002`.

4. Track max-step failures separately from semantic failures.

   The high-complexity MMS task that hit `max_steps` should be tagged as a run cap artifact. If these tasks are kept, use a higher max-step cap for analysis runs or exclude premature terminations from model-behavior conclusions.

5. Add regression metadata for suspected ambiguous tasks.

   A simple tag such as `requires_paid_action_confirmation` on refuel-dependent tasks would make it easier to slice benchmark results and avoid over-interpreting model differences caused by policy ambiguity.
