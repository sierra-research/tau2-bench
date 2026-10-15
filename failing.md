# Failing Smoke Tasks — Root Cause Analysis

> Doc now covers two runs: **SMOKE** (6 tasks, section below) and the **DEV set** (24 tasks, `data/simulations/dev_test/results.json` — see "DEV set failures" section further down). DEV analysis was produced fast by 5 parallel agents (no Codex review this round), one compact block per failing task.

Source: `data/simulations/smoke_test/results.json` (SMOKE run, 6 tasks, see `smoke.md`). 5 of 6 tasks scored reward 0. Analysis is based on transcripts (`simulations[].messages`) and expected ground-truth actions (`tasks[].evaluation_criteria.actions`) — no re-run performed. Every root cause below was independently checked by Codex (read-only, no re-run, full repo + results.json access) against the raw transcript and doc IDs; Codex's verdict is folded into each task's "Codex check" line, and two of my initial diagnoses (task_024, task_072) were corrected based on its evidence. This document is treated as the verified source of truth for DEV iteration planning.

---

## task_014 — reward 0.0, DB ✅ (1/1 action check failed)

**What happened:** Customer received a suspicious mail offer for a "Crypto-Cash Back" card referral bonus. Agent correctly searched the KB multiple times, correctly concluded no such referral program exists, correctly identified it as a probable scam, and correctly called `transfer_to_human_agents`.

**Failure:** Agent used `reason="fraud_or_security_concern"`. Expected: `reason="unconfirmed_external_communication"`.

**Root cause: Incorrect policy interpretation (wrong classification label).** The agent's judgment and actions were substantively correct — it just picked the wrong reason-code taxonomy entry for the transfer. This is a labeling/vocabulary mismatch, not a reasoning failure.

**Codex check:** CONFIRMED.

---

## task_024 — reward 0.0, DB ❌ (1/1 action check failed)

**What happened:** Customer wanted a business credit card recommendation for a ~$40k truck purchase. The task's `required_documents` are exclusively about the **Business Bronze** and **Business Silver** Rewards Cards. The agent ran ~15 KB searches.

**Failure:** Agent recommended and the customer applied for a **Business Platinum Rewards Card**. Expected: `apply_for_credit_card(card_type="Business Bronze Rewards Card", ...)`.

**Root cause (revised — original "retrieval omission" diagnosis was wrong): Incorrect synthesis over already-retrieved context, not a retrieval failure.** Verification against the raw tool-result content (not just the truncated first hit) shows the real Bronze and Silver product docs (`doc_business_credit_cards_business_bronze_rewards_card_001/002/004/005`, `doc_business_credit_cards_business_silver_rewards_card_001/002/012`) **were** returned across multiple KB_search calls (messages 8, 12, 21, 23, 24, 26, 30, 34, 37). The agent had the correct evidence in context repeatedly but never surfaced or compared Bronze against Platinum in its final recommendation — a reasoning/synthesis failure over its own retrieved results, not a missing document.

**Codex check:** REFUTED my original diagnosis, then re-verified by me directly against the raw message content (see `Bronze`/`Silver` substring hits across messages 8/12/21/23/24/26/30/34/37) — confirmed the docs were present in-context and ignored.

---

## task_072 — reward 0.0, DB ❌ (8/9 action checks passed)

**What happened:** Customer asked the agent to audit ATM fees across two accounts. Agent correctly found and credited $14.00 to the first account (Bluest Account) — this matched expected exactly. For the second account (Light Green, `chk_538bfb9cba`), the agent computed a $5.00 credit from three line items ($1.50 + $0.50 + $3.00).

**Failure:** Expected credit for the second account was **$3.50**, not $5.00 — a $1.50 over-credit.

**Root cause (refined by Codex): Incorrect policy interpretation, but specifically a wrongly-applied eligibility condition, not foreign-fee tier math.** Per `doc_checking_accounts_light_green_account_013`, the two foreign-ATM-fee line items were correctly identified and correctly priced (the $80 withdrawal's $5 fee was $3 excessive; the $200 withdrawal's $4 fee was $0.50 excessive — together $3.50, matching the expected credit exactly). The extra $1.50 came from a **third, incorrect** line item: the agent also refunded a domestic ATM fee from 11/05, believing it fell within a "4 free domestic withdrawals/month" allowance the customer hadn't actually used up — a miscounted/misapplied eligibility condition, not a tier-pricing error.

**Codex check:** REFINED — isolated the exact wrong line item (the domestic-fee refund) via the account's fee-schedule doc; the foreign-fee tier math was actually correct.

---

## task_081 — reward 0.0, DB ❌ (9/37 action checks failed)

**What happened:** Customer reported a stolen wallet with 3 debit + 2 credit cards, and asked to freeze all debit cards. (Task also tested policy rule 7 — transfer to human only on the 4th request — which the agent handled **correctly**.) Agent went straight to `close_debit_card_4721` (permanent, irreversible) for all 3 debit cards.

**Failure:** Expected trajectory: `freeze_debit_card_3892` all 3 cards first (reversible), then `unfreeze_debit_card_3893` the one card the customer later found in a jacket pocket, and only `close_debit_card_4721` the 2 genuinely-stolen cards. Because the agent closed all 3 immediately, when the customer found the "Green Account" card was never stolen, it could not be unfrozen (permanently closed) — the agent had to order an unnecessary replacement instead. This one wrong initial tool choice cascaded into 9 downstream action-check failures (wrong tool called at all for freeze/unfreeze, and knock-on argument mismatches on the replacement-card order).

**Root cause: Incorrect policy interpretation + missing verification before an irreversible action.** The KB doc (`doc_bank_accounts_bank_accounts_(general)_026`) states CLOSE is for "confirmed lost/stolen" but also lists "suspicious activity, wants to investigate before closing" as a freeze reason. A same-report-covers-multiple-cards scenario is inherently unconfirmed per-card until the customer checks pockets/bags — the agent should have frozen (safe default, reversible) before irreversibly closing card-by-card as each was confirmed. This is the **earliest supported cause**; everything downstream (unable to recover the wrongly-closed card, extra replacement order) is a consequence, not an independent failure.

**Codex check:** CONFIRMED, with a refinement — the doc doesn't unambiguously say "always freeze first," so this also touches "failure to recover" (agent had no path to undo the incorrect close once discovered) rather than pure misinterpretation alone. Recommend treating this as a **partial-observability / consequence-prediction gap**: the agent didn't preserve uncertainty about which cards were "confirmed" stolen before taking an irreversible action.

---

## task_087 — reward 0.0, DB ❌ (2/20 action checks failed, otherwise near-perfect run)

Two separate, unrelated failures in an otherwise fully correct 18/20 execution:

### (a) Debit card dispute — wrong liability cap
Expected `file_debit_card_transaction_dispute_6281(..., customer_max_liability_amount=50, ...)` (the Regulation E $50 cap for debit card fraud). Agent submitted `customer_max_liability_amount=499.99` — the full disputed amount, verified by direct JSON diff of expected vs. actual tool-call arguments (only that one field differs).

**Root cause: Incorrect policy interpretation.** Agent never applied/looked up the Reg E consumer-liability cap for debit disputes and instead used the disputed transaction amount itself.

### (b) Replacement debit card order — false failure (confirmed harness bug, not an agent error)
Expected and actual tool-call arguments for `order_debit_card_5739` (action `087_18`) were compared via JSON parse-and-diff: **they are semantically identical** — same account_id, user_id, delivery_option, delivery_fee, card_design, design_fee, shipping_address. The only difference is JSON whitespace/indentation (expected is single-line, the agent's tool call was pretty-printed/multi-line).

**Root cause: Evaluation ambiguity — confirmed as a real, systemic grader bug, traced to source.** `Action.compare_with_tool_call()` in `src/tau2/data_model/tasks.py` (~line 194) does `tool_args == action_args`, a plain dict-equality check. For any action routed through `call_discoverable_agent_tool`, the compared `arguments` value is itself a **raw JSON string** (never parsed), so any whitespace/key-order difference in that string causes a false failure — this affects every discoverable-tool action check in the harness, not just this one task. It should be fixed by parsing both sides as JSON before comparing (when the field is a JSON string) rather than comparing raw strings.

**Important scoring nuance (caught by Codex, verified):** for task_087 specifically, `reward_info.reward_basis == ["DB"]`, so this particular false action-check failure did **not** by itself zero task_087's reward — the DB mismatch (driven by the wrong $499.99 liability value in check (a)) is what zeroed it. The bug is still real and worth fixing (it corrupts the action-check-level diagnostics used for failure analysis, and would directly zero the reward on any task whose `reward_basis` includes `ACTION`), but it should not be read as "task_087 failed for two independent agent-visible reasons" — only (a) is reward-relevant here.

**Codex check:** CONFIRMED both — (a) diff isolates the liability-cap field exactly; (b) `json.loads(expected) == json.loads(actual)` is `True` confirming the false failure, and Codex additionally traced it to the exact line in `tasks.py` and flagged the `reward_basis` nuance above, which I independently verified against `reward_info` in `results.json`.

---


---

## DEV set failures (`data/simulations/dev_test/results.json`)

24 of the 25 DEV tasks ran (task_102 didn't complete). **17/24 failed (reward 0.0)**, 7 passed (task_014, 016, 018, 028, 057, 072, 098). Below: one compact block per failing task, produced by 5 parallel agents, no Codex cross-check this round (per instruction — treat as first-pass, not verified-to-100% like the SMOKE section above).

### task_015 — reward 0.0
**Failed:** `give_discoverable_user_tool`/referral-link grant never issued.
**Cause:** Agent demanded a 2nd ID beyond phone+user_id (msg 27); user couldn't supply one and hung up. Ground truth expected the tool granted on user_id+card_name alone.
**Category:** Incorrect policy interpretation (over-strict verification bar).

### task_024 — reward 0.0
**Failed:** `apply_for_credit_card` expected Business Bronze; agent had user apply for Business Silver.
**Cause:** Agent recommended Silver citing a temp promo (~$800 back) despite its own earlier analysis (msg 42) noting the truck purchase earns only base rate — miscalculated best-card comparison.
**Category:** Incorrect policy interpretation / retrieval-comparison error.

### task_026 — reward 0.0
**Failed:** Wrong transaction IDs disputed — missing `txn_b7e2d4c5f506`, two wrong IDs substituted in.
**Cause:** Misidentified which transactions were affected by the cash-back miscalculation when reading the transaction list (msg 26/34).
**Category:** Retrieval omission (misidentified affected records).

### task_048 — reward 0.0
**Failed:** Skipped `get_user_dispute_history_7291` / `get_pending_replacement_orders_5765` before closing 4 credit card accounts.
**Cause:** Went straight from closure-reason lookup to `close_credit_card_account_7834` every time, never discovered the two required pre-closure check tools.
**Category:** Missing prerequisite/verification.

### task_070 — reward 0.0
**Failed:** Opened business checking with `account_class="Lime Green"`, expected `"Sky Blue"`.
**Cause:** Two promo docs retrieved: Oct promo (expired 11/12, ranks Lime Green first) vs Nov promo (active, ranks Sky Blue first). `get_current_time` returned 2025-11-14 (Nov promo active) but agent applied the expired Oct promo anyway.
**Category:** Incorrect policy interpretation (didn't cross-check promo dates vs. current date).

### task_066 — reward 0.0
**Failed:** (1) savings opened as `"Green Account (savings)"` not `"Green Account"`; (2) `apply_for_credit_card` never happened; (3) `close_bank_account_7392` had an extra unrequested `reason` arg.
**Cause:** (1)/(3) argument-formatting errors; (2) agent's final recap never surfaced the credit-card recommendation step, so the user's follow-on application never triggered.
**Category:** Tool-calling error (malformed args) + retrieval omission/incomplete task (credit-card step dropped).

### task_071 — reward 0.0
**Failed:** Same as task_070 — checking opened as Lime Green not Sky Blue, plus savings opened as Silver Plus Saver not Gold Saver.
**Cause:** Same expired-Oct-promo-vs-active-Nov-promo bug as task_070 (`get_current_time` = 2025-11-14, same miss), compounding into a second wrong-tier recommendation.
**Category:** Incorrect policy interpretation — **same recurring bug as task_070.**

### task_074 — reward 0.0
**Failed:** All 4 fee-refund credits under-computed (e.g. $22.00 vs correct $27.00 on account `_1`).
**Cause:** Verified by hand-summing the 33-record transaction history: $39.50 in fees − $12.50 in rebates = $27.00 owed; agent's $22.00 missed at least one fee line item.
**Category:** Retrieval omission (undercounted line items in a long transaction history).

### task_075 — reward 0.0
**Failed:** Opened `"Bluest Account"` instead of expected `"Green Fee-Free Account"`.
**Cause:** Only one KB_search run up front (surfaced World Blue, a business account). After user clarified they needed a personal account, agent never re-queried for personal low-fee options — defaulted to Bluest Account without comparison.
**Category:** Retrieval omission (failed to re-query after initial recommendation was ruled out).

### task_078 — reward 0.0
**Failed:** Expected freeze→verify→selective-unfreeze sequence on 2 cards; agent closed (irreversible) both instead, and froze the wrong card_id.
**Cause:** Called `close_debit_card_4721` (reason "lost") on both target cards instead of `freeze_debit_card_3892`; separately froze an unrelated card (`_lg`) by mistake.
**Category:** Incorrect policy interpretation (irreversible close over reversible freeze) + tool-calling error (wrong card_id). **Same class of bug as task_081 (see below).**

### task_081 — reward 0.0
**Failed:** Same as SMOKE-run task_081 (see above) — expected freeze-all→verify-per-card→selective unfreeze; agent closed all 3 cards immediately instead.
**Cause:** **Confirmed exact repeat of the smoke-test failure pattern** — `close_debit_card_4721` reason "stolen" on all 3 cards without ever calling `get_bank_account_transactions_9173` to verify per-card before acting irreversibly.
**Category:** Incorrect policy interpretation + missing verification. **Reproduced identically across two independent runs — strong evidence this is a systemic pattern, not a one-off.**

### task_082 — reward 0.0
**Failed:** 4 dispute filings with wrong liability/fact fields; 1 debit-card order with wrong delivery option.
**Cause:** Reg E $50 liability cap misapplied (values of `0`/`500` used instead of `50`) across multiple filings; one filing also copied `card_in_possession`/`pin_compromised` facts from the *previous* transaction instead of the current one (stale-state carryover between sequential filings); debit card order added an unrequested paid EXPEDITED upgrade.
**Category:** Incorrect policy interpretation (Reg E cap) + stale/incorrect state (fact carryover).

### task_083 — reward 0.0
**Failed:** All 4 dispute filings mismatched.
**Cause:** Agent invented an extra `customer_max_liability_amount` param not in the expected schema on all 4; the 4th also had wrong `disputed_amount` ($425 vs $475) and used `card_action: "keep_active"` instead of freezing despite an admitted compromised PIN.
**Category:** Tool-calling error (hallucinated param) + incorrect policy interpretation (no freeze on compromised PIN) + stale/incorrect state (wrong amount).

### task_084 — reward 0.0
**Failed:** All 3 dispute filings had wrong `customer_max_liability_amount`.
**Cause:** Reg E $50 cap never correctly applied — used $47.50, then $100, then a rounded $500 instead of an exact $412.88 uncapped-loss figure.
**Category:** Incorrect policy interpretation (Reg E cap).

### task_087 — reward 0.0
**Failed:** `clear_debit_card_fraud_alert_4892` never called (card stuck FROZEN); 1 dispute filing had wrong liability amount.
**Cause:** Agent never discovered/called the fraud-alert-clear tool at all despite the card showing `status: "FROZEN"`; separately used $500 liability instead of the $50 cap on a dispute filing.
**Category:** Retrieval/tool-discovery omission (new failure mode, not seen in SMOKE run) + incorrect policy interpretation (Reg E cap — **recurs from SMOKE run**, though the wrong value differs: $500 here vs. full disputed amount there). **Partial match to SMOKE task_087, not identical — cap bug recurs, discovery-omission bug is new.**

### task_099 — reward 0.0
**Failed:** `log_verification` never called.
**Cause:** Transcript shows account actions taken with no identity-verification step anywhere beforehand.
**Category:** Missing prerequisite/verification.

### task_101 — reward 0.0
**Failed:** Second referral submitted `"Lime Green Account"`, expected `"Sky Blue Account"`.
**Cause:** Agent's KB_search calls only covered Purple/Lime Green/Light Green — never searched "Sky Blue Account" — so its "best combined bonus" recommendation was built without that option, and the user submitted the recommended (wrong) choice.
**Category:** Retrieval omission (option never searched) + incorrect policy interpretation (recommendation built on incomplete option set).

### Cross-task patterns (DEV set)

The 17 DEV failures are not 17 independent bugs — they cluster hard into a small number of repeat mechanisms, which sharpens (not changes) the priority call made in the SMOKE section above:

- **Regulation E $50 liability cap misapplied — 4 tasks (task_082, 083, 084, 087).** By far the single most common recurring failure. Every single dispute-filing call across these tasks either omits the cap, uses the wrong number, or invents a nonexistent parameter. This is exactly the kind of numeric-policy-value error that a **pre-write verification gate requiring the argument to cite its source rule** (Proposal #1/#2 in "Root causes and proposed fixes" above) would catch mechanically — it's the highest-value target for that fix, more valuable than the SMOKE sample alone suggested.
- **Irreversible close instead of reversible freeze — 3 tasks (task_078, and task_081 in both SMOKE and DEV runs).** task_081 reproduced *identically* across two independent simulation runs — this is not noise, it's a systemic gap in how the agent weighs reversibility before acting under an unconfirmed "stolen/lost" claim. Directly validates Proposal #2 (reversibility metadata + plan/verify/commit on irreversible tools) as the top-priority fix.
- **Stale KB coverage / not re-querying after a premise changes — task_075, task_101, and arguably task_070/071 (didn't re-check promo validity against current date).** The agent tends to search once, anchor on the first plausible answer, and not re-verify once new information (a date, a ruled-out option) should have invalidated it. This is a different flavor from task_024's SMOKE finding (info was retrieved but ignored) — here the info was never retrieved at all after the premise changed. Both point at the same fix direction: force a citation/evidence check at decision time, not just at retrieval time.
- **Missing identity verification before acting — task_048 (partial, pre-closure checks) and task_099 (full skip of `log_verification`).** A cheap, mechanical pre-condition check (some tools require verification/prerequisite calls before they're used) — could likely be caught by a simple required-tool-sequence guard, cheaper than the full verification-gate proposal.

**Net effect on the earlier priority call:** unchanged in ranking, strengthened in confidence. The DEV sample gives task_081's freeze/close bug and the Reg E cap bug enough repeat evidence (3-4 tasks each) to say these are systemic, not sample noise — supporting shipping Proposal #2 (risk-aware plan/verify/commit) first, exactly as recommended above.

---

## Summary table

| Task | Failed checks | Root cause | Category |
|---|---|---|---|
| task_014 | 1/1 | Wrong transfer-reason code, correct decision otherwise | Incorrect policy interpretation |
| task_024 | 1/1 | Correct docs (Bronze/Silver) retrieved repeatedly but ignored in final recommendation → wrong card applied for | Incorrect synthesis over already-retrieved context (not retrieval failure) |
| task_072 | 1/9 | Wrongly refunded a domestic fee under a misapplied "4 free/month" eligibility condition; foreign-fee tier math was actually correct | Incorrect policy interpretation (eligibility condition, not arithmetic) |
| task_081 | 9/37 | Closed (irreversible) instead of froze (reversible) debit cards before per-card confirmation | Incorrect policy interpretation + missing verification / partial observability |
| task_087 | 2/20 | (a) Missed Reg E $50 liability cap; (b) grader false-failure on JSON whitespace | (a) Incorrect policy interpretation; (b) Evaluation ambiguity (harness bug, not agent) |

**Net read:** Of 6 SMOKE tasks, only 1 (`task_057`) truly passed. Excluding the confirmed harness false-failure (task_087b), all 4 remaining agent-attributable failures are variants of the same underlying gap: the agent has correct information available (in KB retrieval, in its own tool results, or in a specific policy doc) but **fails to verify/apply it precisely** before acting — wrong reason-code labels, a misapplied eligibility condition, a missed statutory cap, an ignored retrieved doc, and an irreversible action taken before per-item confirmation. This points squarely at the "Partial observability" and "Consequence prediction" focus areas from the handoff pack, not at retrieval coverage. See "Root causes and proposed fixes" below for concrete harness changes.

---

## Root causes and proposed fixes

### My independent proposal (formed before consulting Codex)

Reading `src/tau2/agent/llm_agent.py` and the retrieval toolkit code (`src/tau2/domains/banking_knowledge/{retrieval.py,retrieval_mixins.py,retrieval_toolkits.py}`) shows the baseline is a bare LLM tool-loop: one system prompt (policy + tools), a message-history state object, no verification step, no memory across turns or runs, and no notion of "this action is irreversible." All 4 agent-attributable failures share one mechanism: the agent has the correct information somewhere in context (a retrieved doc, a policy rule, its own prior tool result) but nothing in the loop forces it to check that information precisely before writing state. So instead of 4 separate fixes, I'd prioritize one core mechanism plus two smaller structural changes:

- **Pre-write verification gate (highest leverage).** Wrap write-tool dispatch so that before any tool call carrying a policy-derived numeric or categorical argument (fee amount, liability cap, reason code), the agent must emit a short structured claim — value, source doc/tool-result ID it came from, confidence — before the call is allowed through. Doesn't need to hard-block, just needs to force the citation step. Directly targets task_072 (fee amount), task_087a (Reg E cap), and task_014 (reason-code label) — all three are "argument value should trace to a specific doc/rule and didn't."
- **Reversibility metadata on tools.** Tag tools like `close_debit_card` as irreversible and `freeze_debit_card` as reversible in the tool schema/registry (`src/tau2/environment/tool.py` `Tool`/`as_tool`). When a claim behind an irreversible action ("this card is stolen") hasn't been independently confirmed per-item, require the reversible action first. Targets task_081 directly, and generalizes to any "reported vs. confirmed" ambiguity elsewhere in the domain.
- **Forced retrieval-to-decision traceability, not more retrieval.** task_024 proves the retrieval layer already works — the right docs were pulled multiple times and ignored. So I would *not* prioritize retrieval-strategy changes (denser embeddings, reranking, etc.) as the first move here. Instead, require the agent's final recommendation/decision to cite the doc ID(s) it's comparing against; if it recommends a product it never named while comparing, that gap becomes visible in the trace.
- **Explicitly deprioritized for now: cross-execution memory.** None of these 5 SMOKE failures are memory-shaped (nothing repeats across tasks in this sample). I'd sequence a "Previous-execution memory" build (per the handoff pack's own table) after 1–3 are tried on DEV and shown to move the needle, rather than build it speculatively first.

### Codex's independent proposal (full repo read access, reached before seeing mine)

Codex was given full read access to this repo (not just `results.json`) — `src/tau2/agent/`, `src/tau2/domains/banking_knowledge/{retrieval.py, retrieval_mixins.py, retrieval_toolkits.py, tools.py, data_model.py}`, `src/tau2/environment/` — plus this document, and asked to reach its own conclusions independently, before seeing my proposal above. Its analysis follows, lightly formatted; disagreements with the diagnoses above are called out explicitly where Codex raised any.

**1. Structured evidence and decision state.** Replace "retrieval output as conversation text" with a typed working state. Today `_format_kb_search_result()` in `retrieval_mixins.py` returns one large string, while `LLMAgentState` in `llm_agent.py` retains only raw messages. Add structured `EvidenceItem`/`DecisionRecord` models (document ID/version, quoted rule span, applicable conditions, entities, numeric values, candidate options, confidence). Retrieval tools should return/store these records; before a write, a synthesis stage should produce a decision table linking every selected argument to evidence. Directly targets task_024's ignored Bronze/Silver evidence, and makes the reason code, ATM allowance, and $50 cap in 014/072/087 explicit rather than buried in context. Measure via argument accuracy, evidence recall/precision, unsupported-write rate, numeric-rule accuracy on held-out DEV, ablating retrieval quality separately so synthesis gains aren't mistaken for retrieval gains.

**2. Risk-aware plan/verify/commit execution.** Extend `@is_tool` metadata (`src/tau2/environment/toolkit.py`) with impact properties (`reversible`, `financial`, `irreversible`, required facts/confirmations). Intercept assistant writes in `Environment.get_response()` (`src/tau2/environment/environment.py`) or an execution-policy wrapper. High-impact calls first produce a preview (state diff, affected entities, computed amounts, governing evidence, unmet preconditions, reversible alternatives); a separate verifier (deterministic for arithmetic/enums, model-driven for policy applicability) must approve before commit. Must unwrap `call_discoverable_agent_tool` to inspect the underlying tool. Irreversible actions like `close_debit_card_4721` should require per-card confirmation or explicit override, with `freeze_debit_card_3892` surfaced as the safe interim action. Addresses 014, 072, 081, 087 without encoding task IDs. Measure via prevented-invalid-write rate, false-block rate on correct writes, recovery rate under ambiguous scenarios, final DB success, added turns/latency — on held-out and counterfactual DEV cases.

**3. Versioned execution traces as reusable lessons.** Persist an audit record per decision (observations, retrieved doc IDs/versions, derived facts, proposed call, verifier findings, state diff, result, evaluator feedback) — extend `src/tau2/data_model/simulation.py`, build a training-only lesson store keyed by policy-rule IDs/tool fields/failure type, not task ID. Give every KB rule a stable ID/version and record `rule_id -> evidence -> decision -> tool argument -> lesson/test` dependencies, so a policy-doc change can invalidate/revalidate affected lessons and surface regression tests automatically. Measure via transfer to unseen tasks sharing rule structures, leave-one-policy-family-out eval, repeat-error rate across runs, stale-lesson rate after simulated policy edits, contamination checks (eval trajectories must never enter memory).

**Codex's diagnosis check:** No contradiction found with the confirmed agent-error diagnoses above. One refinement: task_081's KB doesn't mandate "always freeze first" as an explicit rule — the conclusion follows from uncertainty + irreversibility, not a stated universal rule (i.e., this is a generalizable *pattern* to design for, not a policy the agent literally violated). Also independently found and traced the task_087(b) grading bug to `Action.compare_with_tool_call()` in `tasks.py`, including the `reward_basis` nuance — both verified by me directly above.

### Reconciled recommendation

Three passes went into this: my independent proposal, Codex's independent proposal (full repo access), and a second Codex review specifically stress-testing the "fast and cheap" version against the DEV evidence (with two of Codex's own claims cross-checked against the raw task data below — one held up, one didn't).

**Diagnosis, final.** Across SMOKE (5 failures) + DEV (17 failures), two narrow, mechanical bugs account for a large share of all agent-attributable failures:
- **Regulation E debit-liability cap misapplied** — `file_debit_card_transaction_dispute_6281` gets the wrong `customer_max_liability_amount` repeatedly (SMOKE task_087a; DEV task_082, 084, and — with one caveat below — 083).
- **Irreversible `close_debit_card_4721` used instead of reversible `freeze_debit_card_3892`** before per-card verification — reproduced identically across SMOKE task_081, DEV task_081 (same trajectory, same mistake), and DEV task_078. Codex flags a real tension worth documenting rather than papering over: at least one KB doc instructs closing a reported lost/stolen card immediately, which is in some tension with a "freeze first, verify, then close" guard — the fix should encode the actual documented exception conditions (per-card confirmation status), not a blanket "always freeze first" rule.
