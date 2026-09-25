# Plan: Verified, policy-legal post-training to improve Qwen 3.5 9B on tau2-bench retail

## Goal

Improve **Qwen 3.5 9B** on the tau2-bench **retail test tasks**
([data/tau2/domains/retail/tasks.json](data/tau2/domains/retail/tasks.json),
114 tasks) by post-training on ~1000 diverse, complex, trajectories that are
both **outcome-verified** (`reward==1.0`) **and policy-legal** (no policy
violation along the path). The pipeline feeds a **post-training ladder** —
SFT first, then preference tuning (KTO/DPO), then optionally on-policy RL
(GRPO) — over a single shared, method-agnostic data + reward substrate.

> **Implemented so far (v1):** the **data-generation half** — synthetic tasks +
> trajectories in tau2-compliant format. v1 is a focused first batch targeting
> three measured failure modes (222 tasks), not yet the full taxonomy/sizing
> below. Post-training is handed off to a separate owner. See
> **[Implemented approach (v1)](#implemented-approach-v1--failure-mode-augmentation)**.

## Why this works

tau2's reward is **programmatic and path-flexible**. Default `reward_basis` is
`[DB, COMMUNICATE]`: a trajectory passes (`reward == 1.0`) when the final DB
state matches the target — computed by replaying `evaluation_criteria.actions`
on a fresh env — **and** all required strings appear in agent messages. Any
agent path that reaches the target state passes. This gives us two things:
trajectories can be **auto-verified — no human labeling**, and synthetic tasks
can be **proven valid by execution** rather than trusted from an LLM.

**But path-flexibility is a liability for training data.** Because the reward
only scores the end state, ~71% of `reward==1` retail trajectories take at least
one policy-illegal step (measured by the policy validator below; consistent with
the 27–78% corrupt-success range in the procedure-aware-evaluation literature).
SFT on `reward==1` alone teaches the model to reach correct end-states via
illegal shortcuts (skipping auth, batching writes, acting before confirmation,
fabricating IDs) — exactly the brittle behavior that erodes **pass^k**
reliability even when pass^1 looks fine. So we filter training data on
**legality**, not just outcome, while leaving the eval reward unchanged.

Key references:
- Evaluator dispatch: [src/tau2/evaluator/evaluator.py:88](src/tau2/evaluator/evaluator.py#L88)
- Task schema: [src/tau2/data_model/tasks.py:560](src/tau2/data_model/tasks.py#L560)
- Retail env factory: [src/tau2/domains/retail/environment.py:17](src/tau2/domains/retail/environment.py#L17)
- Retail policy (skill + legality source): [data/tau2/domains/retail/policy.md](data/tau2/domains/retail/policy.md)
- Programmatic task-build pattern to mirror: [src/tau2/domains/telecom/tasks/create_tasks.py](src/tau2/domains/telecom/tasks/create_tasks.py)
- Seed/style reference (10 hand-built complex tasks): [data/tau2/domains/retail/tasks_complex.json](data/tau2/domains/retail/tasks_complex.json)
- Legality validator spec + checker: `Policy-validator-algorithm.md`, `retail_policy_validator.py`

---

## Architectural decisions & rationale

Each decision is tied to evidence from the reference pipelines — **APIGen-MT**
(verified blueprint → interplay), **JOSH** (sparse-reward self-alignment via
beam search), and **Simia** (LLM-simulated trajectories + GRPO).

| # | Decision | Choice | Rationale & evidence |
|---|---|---|---|
| **D1** | Verification substrate | **Real tau2 env + `reward==1.0` rejection sampling** (not LLM-simulated trajectories) | tau2 ships a programmatic reward, so we can *prove* correctness by execution. Simia gives this up (correct-by-construction, no success check) and is bounded by synthesizer correctness; JOSH and APIGen-MT both anchor on a real reward and discard failures. With a real reward available, simulating it away is strictly worse. |
| **D2** | Data source: **self vs teacher distillation** | **Decided empirically in pre-work, not locked.** Default to **self-distillation (Qwen's own r=1, legal rollouts)** if Qwen's legal keep-rate is workable; else blend / use Claude teacher. | Qwen 3.5 9B already scores ~79 aggregate — retail headroom over a Claude teacher may be thin, and SFT on Claude traces imports Claude's phrasing/CoT/turn-shape, fighting Qwen's priors. JOSH's gains came from self-alignment on a model's own outputs; APIGen-MT shows cross-model BC works too. Measure both the *legal* keep-rate per candidate agent (Pre-work) — a teacher that follows retail policy more faithfully raises legal yield and may justify distillation despite the style mismatch. |
| **D3** | Training signal | **Positives = legal r=1 paths (SFT); negatives = r=0 rollouts AND per-step illegal steps from the legality validator.** Default preference method KTO. | JOSH's KTO beat its SFT variant on every metric (96% fewer malformed APIs). APIGen-MT names discarded failures as untapped contrastive signal. Rejection sampling already produces whole-trajectory negatives; D11's validator adds *step-localized* negatives, a sharper signal (closer to JOSH's turn-level extraction). |
| **D4** | Task quality gates | **Execution validity + policy compliance + intent alignment** (this validates the *task/gold solution*) | Execute-validate alone admits "valid but wrong" tasks whose gold actions don't match intent. APIGen-MT defends this with format/execution + policy-as-unit-tests + an LLM-committee alignment vote over the diff_patch. Distinct from D11, which validates the *agent's path*. |
| **D5** | Complex-task construction | **Reverse task recombination from validated simple blocks** | APIGen-MT found direct long-horizon generation fails validation often; composing validated simple tasks (same persona) then re-checking policy + alignment raises yield on the hardest cells and gives compositional difficulty control. |
| **D6** | Reward storage | **Persist decomposed `[DB, COMMUNICATE]` components + realized DB diff, not just binary** | Needed for error-driven iteration (wrong-DB vs failed-to-inform) and for the `communicate_info` gaming guard. |
| **D7** | Decontamination — **protect the held-out *test split***  | **Two layers: (1) EXCLUDE the *test-split* tasks' entities (`user_id`/`order_id`) as synthesis seeds; (2) reject any synthetic task whose target-DB-diff (write-set) equals a *test-split* task's.** | The retail benchmark labels splits in `split_tasks.json`: **train 74 / test 40 / base 114**. Only the **test split (40)** is the held-out post-training eval set, so only *its* entities/solutions are forbidden as seeds (**19 users / 66 orders**); the **train split (74) is usable seed material**. Layer 1 (entity exclusion) keeps the model off the exact users/orders it's later evaluated on. Layer 2 (DB-diff equivalence) is the right *dedup* key because synthetic and benchmark tasks share `db.json` — surface ids collide spuriously, so a (user_id, order_id) match is the wrong dedup signal; the write-set catches solution leakage across ids and duplicate synthetic tasks. Protected scope is one flag (`synth/lib.py:PROTECTED_SPLITS`); set `("train","test")` if eval will use `base`. **Decision taken (v1): STRICT — because train↔test share entities, drop any task touching a test user/order (164→92), then top up from-scratch; the 222-task set shares 0 users / 0 orders / 0 write-sets with the test split.** |
| **D8** | Format & thinking mode | **Lock thinking/non-thinking mode + chat template + tool schema byte-identical across sampling, training, and eval** | Simia warns to eval Qwen3 on tau2 in *non-thinking* mode (thinking → runaway CoT, near-zero scores). Format drift is the #1 silent failure. Also interacts with D2: Claude CoT won't map to Qwen thinking tags → another reason to prefer self-distillation. |
| **D9** | Eval protocol | **pass^1 over ≥8–10 trials, fixed seeds, report pass^k (k=1..4) + CIs** | JOSH ran τ-bench 10× due to ~115-task variance; our set is 114. Legality-filtering's payoff is mainly in pass^k reliability, so report the curve, not a point. |
| **D10** | Diversity control | **Skill × complexity taxonomy, cell quotas weighted by pre-work failure analysis** | Systematic coverage beats ad-hoc generation; weight toward cells where Qwen fails but a verified+legal solution exists (real, fixable headroom). |
| **D11** | **Trajectory legality gate** | **Keep a trajectory only if `reward==1.0` AND the policy validator reports no high-precision violation** (strict mode + LLM-judged confirmation) | tau2's reward is path-flexible by design, so ~71% of `reward==1` retail trajectories are policy-illegal. Outcome-only filtering teaches illegal shortcuts that erode pass^k. The validator (`Policy-validator-algorithm.md`) is a guarded transition system that replays a trajectory and flags `action_before_auth`, `cross_user_action`, `ungrounded_id`, `guard_violation`, `multiple_tool_calls`, `msg_and_toolcall_same_turn`, `missing_confirmation`. Its per-step labels also feed D3 negatives and D12's GRPO reward shaping. |
| **D12** | **Post-training ladder over a shared substrate** | **One method-agnostic data+reward substrate; climb SFT → KTO/DPO → GRPO based on where pass^k plateaus.** | KTO/DPO are *offline* and consume the same rollout pool (difference = example formatting + trainer flag); LLaMA-Factory/TRL cover SFT/DPO/KTO over one dataset interface. GRPO is *online* (samples from the live policy, scores against the env each step; Simia used RAGEN+VeRL) — it reuses our env + reward + legality validator but is a separate training loop, planned for round 2+. Build the substrate now; add each method as a thin adapter; tune one rung at a time. |

**Locked from v0:** Claude Opus/Sonnet 4.x as the fallback/blend teacher and the
pre-work ceiling probe; grounded + execute-validate task synthesis; factored
pipeline so preference/RL rungs bolt on without rework.

---

## Implemented approach (v1) — failure-mode augmentation

**Scope.** This repo implements the **data-generation half** only — synthetic
**tasks** and **trajectories** in tau2-compliant format. Post-training (D12 SFT/
KTO/DPO/GRPO; D8 format/serving; D9 eval) is handed off to a separate owner; the
phases below are the *receiving* plan, not built here.

**What v1 targets.** Instead of the full skill×complexity taxonomy (Phase 1,
deferred), v1 focuses on the **three failure modes** found in baseline
Qwen3.5-9B error analysis on the retail test set:
**`conditional_fallback`**, **`multi_goal`** (multi-goal phrasing), and
**`mid_call_mind_change`**.

**Method — seed-transform augmentation (a concrete instance of D5).** Take the
**74 train-split tasks as seeds** and inject one failure-mode pattern each:
- **Gold actions / DB target are derived in code** (deterministic,
  execute-validated). The **LLM only rewrites the prose** (`reason_for_call`,
  mind-change script); it never selects the answer → correctness is independent
  of the LLM (D1).
- Each pattern is built so **DB-state reward alone catches its failure** (pick
  the unavailable variant → wrong DB; drop a goal → wrong DB; commit the
  mind-change decoy → wrong DB), so `reward_basis=[DB]` suffices — no judge
  needed for the outcome.

**Decontamination — strict + top-up (the D7 decision actually taken).** Train and
test splits *share* users/orders (shared `db.json`), so naive augmentation leaked
72 test entities (164 → 92 clean). Chosen policy: **drop any augmented task
touching a test user/order** (→ 92 strict-clean), then **top up from-scratch on
`free_users`** (test-excluded by construction) to a per-pattern target.

**Result — `synth/tasks_failuremode.json`: 222 tasks (74 per failure mode)** =
92 train-seed augmented + 130 from-scratch top-up. **0 test-user / 0 test-order /
0 test-write-set overlap, all unique, all execute-validated.** Registered as the
tau2 task set **`retail_failuremode`**.

**Task-validity gates applied (D4):** (a) execute-validate gold on a fresh env,
(b) DB moves as expected, (c) strict entity decontam + DB-diff dedup vs test
(D7), (d) `Task.model_validate`. Policy-compliance (D4c/D11) and LLM
intent-alignment (D4d) are **pluggable**: alignment runs inside the LLM rewrite;
the legality validator activates when `retail_policy_validator.py` is provided.

**Code map (`synth/`):**

| File | Role |
|---|---|
| `lib.py` | grounding helpers, execute-validate, DB-diff decontam, `PROTECTED_SPLITS` |
| `seeds.py` | load the 74 train seeds (68 write-bearing) |
| `patterns.py` | the 3 failure-mode transforms (native + compose) |
| `build_augmented.py` | strict-clean train-seed augmentation → 92 |
| `build_topup.py` | from-scratch fill to 74/pattern → combined 222 |
| `rewrite.py` | LLM prose rewriter + alignment gate (needs API key) |
| `synth_tasks.py` | register `retail_failuremode` for `tau2 run` |
| `run_trajectories.py` / `filter_legal.py` | trajectory gen via tau2 + `reward==1 ∧ legal` filter |

*(`generators.py` / `build_tasks.py` are an earlier from-scratch coverage path,
retained but not used for v1.)*

**Run-later (needs API key / agent endpoint):** (1) `build_augmented.py --llm`
for natural prose (gold/validation identical to the template path); (2)
trajectories via `run_trajectories.py → filter_legal.py`.

**Known limits of v1.** Phase 1's full taxonomy and the ~500–600-task / ~1000-
trajectory sizing are **deferred**. Top-up tasks reuse templated
`reason_for_call` (lower surface diversity until `--llm`). `mid_call_mind_change`
only manifests at rollout time (the user simulator changes its mind) — the static
task is valid regardless, and wrong-DB filters the failures.

---

## Pipeline overview

```
(A) TASK SYNTHESIS  → validated tasks      [gates: execute + policy + alignment]   (D4,D5)
                         │
(B) TRAJECTORY GEN  → rollouts scored by tau2 reward + replayed through validator
                     → keep r=1 AND legal           → SFT positives                 (D1,D11)
                     → r=0 rollouts + per-step illegal steps → preference negatives  (D3,D11)
                         │
(C) SUBSTRATE       → canonical rollout records + reward service                    (D12)
                         │
(D) POST-TRAIN      → SFT → (KTO | DPO) → (GRPO, round 2+)                          (D12)
                         │
(E) EVAL + ITERATE  → multi-trial eval on real 114 (pass^k) → error-driven round    (D9,D10)
```

Three independent quality gates: **task validity** (D4: executes + policy +
intent-aligned), **trajectory outcome** (D1: `reward==1.0`), and **trajectory
legality** (D11: no policy violation along the path).

---

## Phase 0 — Foundations (build once)

1. **Task validation harness (D4).** Given a candidate `Task`: instantiate the
   retail env via `get_environment`, apply `initial_state` + `evaluation_criteria.actions`,
   confirm (a) no tool errors, (b) deterministic target DB hash, (c) policy
   compliance (policy clauses as executable checks), (d) **intent alignment**
   (strong judge / 3-judge majority over the DB diff confirms the gold actions
   fulfill the instruction). Fail → one self-repair attempt, else discard.
2. **Policy legality validator (D11).** Stand up `retail_policy_validator.py`
   per `Policy-validator-algorithm.md` — a guarded labeled transition system
   built from `policy.md` (Σ = one node per tool; obligations = requires /
   cross_user / provenance / confirm / guard; effects = status transitions +
   one-shot locks). `CHECK(messages, env_from_db)` replays a trajectory and
   returns violations + **per-step legality labels**. Run **grounded** (init
   statuses/ownership/availability from `db.json`).
   - **Severity tiering:** treat WRITE-containing `multiple_tool_calls`,
     `guard_violation`, `ungrounded_id`, `cross_user_action`, `action_before_auth`,
     `msg_and_toolcall_same_turn` as hard rejects; treat read-only batches as a
     soft signal (down-weight, don't hard-reject).
   - **Upgrade `missing_confirmation`** from the regex matcher to an LLM judge
     over the 2–3 turns preceding each write before using it as a gate (regex
     already mis-handled "please cancel it because it's no longer needed").
   - **Calibrate:** hand-check a sample of (reward==1, flagged) trajectories to
     confirm the false-positive rate before trusting it as a hard filter.
3. **Decontamination filter (D7) — protect the held-out test split.** The retail
   benchmark labels splits in `split_tasks.json` (**train 74 / test 40 / base
   114**). The **test split (40)** is the post-training eval set; synthetic data
   must use **none of its entities and none of its solutions**. The **train
   split (74) is usable seed material**.
   - **Layer 1 (entity exclusion, primary guard):** never seed on a `user_id` or
     `order_id` referenced by the test split (**19 users / 66 orders**);
     synthesis draws from the rest. `synth/lib.py` (`free_users`, `orders_of`,
     `PROTECTED_SPLITS`).
   - **Layer 2 (solution equivalence):** index the test split by target-DB-diff
     signature; reject matching synthetic write-sets. `synth/build_tasks.py`.
   - **Status (v1):** train↔test share entities, so the chosen policy is
     **strict** — drop any synthetic task touching a test user/order, then top up
     from `free_users`. The 222-task set is verified clean (0 users / 0 orders /
     0 write-sets shared with test).
   - **Decision:** if post-training eval will run on `base` (all 114) instead of
     the test split, set `PROTECTED_SPLITS=("train","test")` and regenerate.
4. **Format contract (D8).** Lock chat template / tool-schema rendering to eval
   time **and** fix thinking/non-thinking mode. Serve Qwen via a vLLM
   OpenAI-compatible endpoint, identical for sampling and eval.
5. **Canonical rollout record (D12).** Define one record consumed by every rung:
   `{task_id, path_id, messages, tool_calls, db_reward, communicate_reward,
   realized_db_diff, per_step_legality[], agent_model, format_version}`. All
   downstream formatters (SFT / KTO / DPO / GRPO buffers) read this.
6. **Reward service (D12).** Wrap tau2 reward **and** the legality validator
   behind one callable `score(trajectory) -> {db, communicate, legal, step_labels}`,
   usable both as an *offline batch filter* (Phases 3–4) and as an *online
   reward* (GRPO, Phase 5).
7. **Eval protocol (D9).** Fix trial count (≥8–10), seeds, pass^k report format.

## Phase 1 — Difficulty taxonomy (drives diversity)

Coverage of **skill × complexity** cells, per-cell quota **weighted by pre-work
failure analysis** (D10).

- **Skills** (retail policy sections): auth (name+zip / email) · get/cancel
  pending order · modify pending (items / address / payment) · exchange delivered ·
  return delivered · product+variant search · refuse-and-explain · transfer_to_human.
- **Complexity modifiers:** multi-intent (≥2 goals) · disambiguation · conditional
  fallback · cross-tool dependency · **policy-refusal traps** · constraint
  satisfaction · single-call constraints.

Give refusal/transfer tasks real weight.

## Phase 2 — Grounded task synthesis (over-generate ~1.5–2×, filter to target)

> **v1 status:** implemented as **failure-mode seed-transform augmentation +
> from-scratch top-up** (see [Implemented approach (v1)](#implemented-approach-v1--failure-mode-augmentation)),
> producing 222 tasks across 3 patterns. The full skill×complexity taxonomy and
> the ~500–600 target below remain future work.

Per cell: (1) sample real grounding from `db.json`; (2) LLM composes the
scenario; (3) derive `evaluation_criteria.actions` (programmatic for structured
intents; LLM-propose-then-verify for complex); (4) **validate** through Phase 0
gates 1a–1d (incl. alignment); (5) auto-fill `communicate_info` from executed
results. **Complex cells via reverse recombination (D5):** concatenate validated
same-persona tasks, re-run policy + alignment, synthesize one umbrella
instruction. Target: ~500–600 valid tasks.

## Phase 3 — Trajectory generation (the fourth gate lives here)

> **v1 status:** wired but not yet run (needs an agent endpoint + API key).
> `synth/run_trajectories.py` registers `retail_failuremode` and calls tau2's own
> runner (tau2 computes + stores decomposed `reward_info`, so no custom reward
> capture is needed); `synth/filter_legal.py` keeps `reward==1 ∧ legal` with the
> legality validator pluggable. The canonical-record/reward-service substrate
> (Phase 0.5/0.6) is deferred to the post-training owner.

1. Run the chosen agent (**self: Qwen; else Claude teacher**, per D2) as
   `--agent-llm` vs the standard user simulator, **k samples each at temp > 0**
   (`tau2 run --num-trials k`), user-sim model+temp pinned to eval.
   **Generation-time legality emphasis:** instruct the agent to make one tool
   call per turn, authenticate first, and confirm before every write — raises
   legal yield instead of relying only on the filter.
2. Score every rollout through the **reward service** (D6, D12): tau2 reward +
   validator. **Accept as SFT positive iff `reward==1.0` AND legal (D11, tiered).**
3. **Keep the negatives:** r=0 rollouts and per-step illegal steps from otherwise
   passing trajectories → preference pool (D3).
4. DB-reward is path-flexible → keep **multiple distinct *legal* paths/task**
   (≥2; more when tool ordering / read-API use differs) to teach outcome-based,
   legal behavior rather than a memorized path.
5. Emit everything as **canonical rollout records** (Phase 0.5).

## Phase 4 — Curation & formatting (method-agnostic)

- **SFT set:** legal r=1 trajectories; **mask loss to agent/assistant tokens
  only** (text + tool calls); system prompt = retail policy + tool schemas,
  byte-identical to eval (D8).
- **Preference data (D3):**
  - *KTO (default):* legal r=1 turns = upvotes; r=0 turns and per-step illegal
    steps = downvotes (unpaired, imbalance-tolerant).
  - *DPO (optional):* requires genuine **shared-context contrast pairs** — build
    them from the validator's per-step labels (same context up to a step, one
    legal next-step vs one illegal next-step). Do **not** pair arbitrary r=1/r=0
    whole trajectories (loose pairs degrade DPO).
- **`communicate_info` gaming guard:** LLM-judge a sample of COMMUNICATE-passing
  trajectories ("was the info actually conveyed?"); keep some `nl_assertions`.
- Re-run DB-diff decontamination; dedupe; rebalance to taxonomy quotas; sanity
  filters (no degenerate loops; refusal tasks *actually* refuse).
- Carve a **synthetic dev split** (never the 114; sized to a few hundred so the
  iteration signal isn't itself noisy).

## Phase 5 — Post-training ladder (D12)

Climb only as far as pass^k requires; tune one rung at a time. LLaMA-Factory/TRL
cover SFT/DPO/KTO over one dataset interface; GRPO via RAGEN+VeRL or TRL's
GRPOTrainer.

- **Rung 1 — SFT (v1).** On legal r=1 data. Establish the baseline and confirm
  the data is right before adding method complexity.
- **Rung 2 — Preference (v1.5).** KTO by default on the Phase-4 preference pool;
  DPO only if shared-context contrast pairs were harvested. Offline, cheap,
  reuses the rollout pool.
- **Rung 3 — GRPO (v2, only if preference plateaus).** On-policy: sample groups
  from the current Qwen policy, score each rollout via the **reward service**
  online. **Use legality as a reward shaper / hard penalty**, not just the
  outcome reward — this is what stops GRPO from reward-hacking into corrupt
  successes (the 71% failure mode), the single strongest reason to expose the
  validator as a service. Budget for cost/instability: Simia got only *slight*
  gains from 64 GRPO steps; JOSH's PPO baseline collapsed until rewards were
  hand-shaped; every step runs multi-turn LLM-user rollouts.

Support the data/infra for all three; **commit tuning effort to one rung at a
time** (DPO β, KTO desirable/undesirable weights + reference, GRPO KL/group
size/clip are separate failure surfaces).

## Phase 6 — Eval, iterate

- Eval on the real 114 via `tau2 run` with the fixed protocol (D9): pass^1 over
  ≥8–10 trials + pass^k (k=1..4) curve + CIs. Expect the win mainly in pass^k
  (reliability), since the eval reward itself does not score legality — a
  deliberate train/eval asymmetry (we bet legal behavior generalizes to higher
  reliability).
- **Generalization guard:** hold out a slice of the 114 (or use airline) looked
  at *once*, at the end, so cross-round error analysis doesn't leak the test set.
- **Error analysis by taxonomy cell + reward component + violation type** →
  regenerate failing cells → expert-iteration round 2 with on-policy Qwen
  successes (feeds Rung 3). Two–three rounds beats one static dump.

---

## Risks & mitigations

- **Validator false positives (D11).** It's a hand-built model of prose policy.
  *Mitigation:* tier severities, LLM-judge confirmation, calibrate FP rate on a
  sample before hard-filtering; it's retail-coupled, so regenerate if `policy.md`
  changes.
- **Legality keep-rate collapse.** Strict filtering of all 488 `multiple_tool_calls`
  would gut volume. *Mitigation:* tier (only 144 batch a WRITE); raise legal
  yield at generation time; size `k` off the *tiered* corrupt rate, not 71%.
- **Teacher ceiling / format mismatch (D2, D8).** *Mitigation:* measure legal
  keep-rate per agent; prefer self-distillation; reformat teacher CoT to Qwen's
  mode if used.
- **"Valid but wrong" tasks (D4).** *Mitigation:* the alignment gate.
- **GRPO cost / instability / reward hacking (D12).** *Mitigation:* defer to
  round 2+; legality-as-shaper inside the GRPO reward; small step budget first.
- **Premature multi-method tuning (D12).** *Mitigation:* substrate supports
  three; tune one rung at a time.
- **Eval noise (D9) / test-set leakage (D7, Phase 6).** *Mitigation:* multi-trial
  CIs; DB-diff decontamination + one-look held-out slice.

## Pre-work (before writing synthesis code)

1. **Baseline:** untuned Qwen 3.5 9B on the 114, fixed protocol → "before" + CIs.
2. **Two keep-rate probes (drives D2), measured for BOTH outcome and legality:**
   - **Teacher:** Claude on the 114 → ceiling, outcome keep-rate, and **legal**
     keep-rate.
   - **Self:** Qwen zero-shot on the 114 → its outcome and **legal** keep-rates.
     Prefer self-distillation if Qwen's legal keep-rate is workable.
3. **Corrupt-rate calibration:** run `retail_policy_validator.py --strict` on a
   teacher/self sample, tier by severity → the *true* legal-given-passing rate
   that sets `k` (Sizing).
4. **Failure map:** base-Qwen failures by skill × complexity cell + violation
   type → seeds Phase 1 quotas (D10).

## Sizing

~600 valid tasks × ~2 kept legal r=1 paths ≈ **~1000–1200 SFT trajectories**.
Set `k` from the *tiered legality* keep-rate observed in pre-work (not the raw
71% corrupt rate, and not a fixed k=4): if legal-given-passing ≈ p_L and reward
pass-rate ≈ p_R, expect ≈ k·p_R·p_L legal positives per task, so choose `k` to
hit ≥2. The r=0 rollouts plus per-step illegal steps form the preference pool
(KTO/DPO) at no extra generation cost; GRPO (Rung 3) generates on-policy and
needs no static pool.
