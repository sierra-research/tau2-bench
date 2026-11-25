# Implementation plan — PLAN_v2 (policy-legal post-training for Qwen3-8B on tau2 retail)

Companion to [PLAN_v2.md](PLAN_v2.md). Translates the strategy (D1–D12) into a
sequenced, buildable plan with milestones, contracts, and decision gates.

## Decisions folded in

- **Agent model:** **Qwen3-8B** (closest real ~9B Qwen3). Eval + sampling +
  training all in **non-thinking mode** (D8 — Qwen3 thinking → runaway CoT →
  ~0 on tau2). Thinking-toggle handling is part of the M0 format fixture.
- **Legality validator:** **user-provided** (`Policy-validator-algorithm.md`,
  `retail_policy_validator.py`). Not yet on disk → M1 is *integrate + calibrate*,
  not author. The 71% illegal / 488 / 144 figures in PLAN_v2 are **reproduced
  from a real run** before being used as filter thresholds.
- **Compute:** **undecided.** Plan is compute-agnostic; M0/M5/M6 carry an
  explicit "compute TBD" dependency and conditional effort.

## Ordering insight

PLAN_v2 says "pre-work before synthesis code," but the pre-work probes depend on
foundations: no *legal* keep-rate without the validator (D11), nothing without
Qwen served under the locked format (D8) + eval protocol (D9) + decomposed
reward capture (D6). So the build front-loads the substrate, runs pre-work as a
**gate**, then generates data.

**Critical path:**
`M0 → M1 → M2 → M3 (GATE) → M4 → M5 → M6`

```
M0 Harness ─┬───────────────┐
            │               ▼
            └─> M1 Validator ─> M2 Reward service ─> M3 Pre-work ─GATE─> M4 Synthesis ─> M5 Traj+Curate ─> M6 Train+Eval
   (D8 fmt, D9 eval,        (D11)        (D6,D12)    (D2, k, quotas)   (D4,D5,D7)     (D3,D11)        (D12,D9)
    D6 reward capture)
```
M4 generators can be drafted in parallel (extend `synth/lib.py`), but M4's gates
need M1 + a judge.

---

## Milestones

### M0 — Harness bring-up *(unblocks everything; no data yet)*
- `infra/serve_qwen.{sh,md}` — Qwen3-8B on vLLM OpenAI-compatible; register as a
  litellm `--agent-llm` target; smoke-test one retail task through `tau2 run`.
- **Format contract (D8):** pin chat template, tool-schema rendering, and
  **non-thinking mode**; commit the exact rendered system prompt + tool block as
  `infra/format_fixture.json`. All downstream asserts against it.
- `eval/protocol.py` (D9): fixed seeds, ≥8–10 trials, pass^1 + pass^k (k=1..4)
  with bootstrap CIs.
- `eval/reward_capture.py` (D6): `Results`/`SimulationRun` → `{db_reward,
  communicate_reward, realized_db_diff}` (extend `RewardInfo` with DB-diff).
- **Exit:** one command runs an N-trial retail eval for any agent and emits
  pass^k + CIs + per-task reward components. Fixture committed.
- **Depends:** compute (TBD). **Effort:** M.

### M1 — Legality validator (D11) *(integrate user-provided + calibrate)*
- Integrate `retail_policy_validator.py`; confirm `CHECK(messages, env_from_db)
  -> {violations[], per_step_labels[]}` runs **grounded** from `db.json`.
- Wire **severity tiering** (WRITE-containing `multiple_tool_calls` + the five
  structural flags = hard reject; read-only batches = soft).
- **Upgrade `missing_confirmation`** to an LLM judge over the 2–3 turns before
  each write (regex mishandles "cancel it because it's no longer needed").
- **Calibrate:** hand-label a sample of `(reward==1, flagged)` rollouts → FP
  rate; **reproduce the 71%/488/144 figures** from a real run before trusting
  thresholds.
- **Exit:** validator emits per-trajectory + per-step labels on a `results.json`;
  FP rate calibrated; tiers set from real data.
- **Depends:** M0 (sample rollouts to calibrate), user delivers the files.
  **Effort:** M. **Risk:** high (hand-built model of prose policy).

### M2 — Reward service + canonical record (D6, D12)
- `substrate/record.py` — canonical record read by every rung: `{task_id,
  path_id, messages, tool_calls, db_reward, communicate_reward, realized_db_diff,
  per_step_legality[], agent_model, format_version}`.
- `substrate/reward_service.py` — `score(trajectory) -> {db, communicate, legal,
  step_labels}` wrapping tau2 reward + M1. One path, two call sites: offline
  filter (M4–M5) and online reward (M6 GRPO) — the anti-reward-hacking lever.
- **Exit:** any trajectory → one record + one score call; JSONL round-trip.
- **Depends:** M1. **Effort:** S–M.

### M3 — Pre-work probes *(decision GATE — do not skip)*
- Baseline Qwen3-8B on the 114 → "before" + CIs.
- Keep-rate probes for Qwen (self) **and** Claude (teacher): outcome **and
  legal** keep-rate.
- Corrupt-rate calibration: validator `--strict` on a sample, tier → true
  *legal-given-passing* rate `p_L`.
- Failure map: base-Qwen failures by skill×complexity cell + violation type.
- **Exit GATE decisions:** (a) **D2** self vs teacher vs blend; (b) **k** from
  `k·p_R·p_L ≥ 2`; (c) **taxonomy quotas** weighted to cells where Qwen fails
  but a legal solution exists.
- **Depends:** M0, M1, M2, compute. **Effort:** M (compute-bound).

### M4 — Task synthesis (Phases 1–2; D4, D5, D7)
- Build on existing `synth/lib.py` (grounding + execute-validate already done).
- `synth/taxonomy.py` — skill×complexity cells + quotas (seeded by M3).
- `synth/generators.py` — grounded per-cell builders (resumes the interrupted v1
  work; reuses `lib.py`).
- **D4 gate** = execute-validate (have it) **+ policy-compliance** (gold actions
  through M1) **+ intent-alignment** (3-judge majority over the DB diff). One
  self-repair retry, else discard.
- **D7 decontamination** = **target-DB-diff signature** (supersedes `lib.py`'s
  entity-overlap check — too weak, db is shared).
- **D5 reverse recombination** for complex cells: concatenate validated
  same-persona simple tasks → re-run policy+alignment → umbrella instruction.
- **Exit:** ~500–600 validated, quota-balanced, decontaminated tasks.
- **Depends:** M1, a judge model, M3 (quotas). Builders can start pre-M3.
  **Effort:** L.

### M5 — Trajectory generation + curation (Phases 3–4)
- `gen/rollout.py` — batch `tau2 run`, `k` from M3, **legality-emphasis prompt**
  (one tool call/turn, auth first, confirm before writes); user-sim model+temp
  pinned to eval.
- Score via M2 → **positives = r=1 ∧ legal**, keep **≥2 distinct legal
  paths/task**; **harvest negatives** (r=0 + per-step illegal) → preference pool.
  Emit canonical records.
- `curate/` formatters (all read canonical record): **SFT** (assistant-token
  mask, byte-identical sys prompt), **KTO** (turn up/downvotes), **DPO** (only
  genuine shared-context contrast pairs from per-step labels — never arbitrary
  r=1/r=0 pairs). Plus communicate-gaming LLM guard, dedupe, rebalance,
  **synthetic dev split** (never the 114).
- **Exit:** ~1000–1200 SFT positives + preference pool; dev split carved.
- **Depends:** M2, M4, compute. **Effort:** L.

### M6 — Post-training ladder + eval loop (Phases 5–6, D12)
- **Rung 1 SFT** (LLaMA-Factory/TRL) on legal r=1 → eval via M0.
- **Rung 2 KTO** (default; DPO only if contrast pairs harvested) → eval.
- **Rung 3 GRPO** (round 2+, only if preference plateaus): on-policy groups
  scored online by M2, **legality as reward shaper/penalty**.
- Error analysis by cell × reward-component × violation type → regenerate failing
  cells (round 2). **Generalization guard:** one-look held-out slice.
- **Exit:** pass^k improvement on the 114 with CIs (win expected mainly in
  pass^k reliability).
- **Depends:** M5, compute. **Effort:** L.

---

## Load-bearing contracts (lock before building around them)

| Contract | Where | Why |
|---|---|---|
| Canonical rollout record | `substrate/record.py` (M2) | Every formatter + GRPO buffer reads it. |
| `score()` signature | `substrate/reward_service.py` (M2) | Same call offline + online; anti-reward-hacking. |
| Validator `CHECK()` + label vocab | `synth/retail_policy_validator.py` (M1) | Feeds D4 gate, D3 negatives, D11 filter, D12 shaping. |
| Format fixture (non-thinking) | `infra/format_fixture.json` (M0) | D8; train/sample/eval drift = #1 silent failure. |

## Proposed layout
```
synth/      lib.py (done) · taxonomy.py · generators.py · retail_policy_validator.py (provided)
substrate/  record.py · reward_service.py
eval/       protocol.py · reward_capture.py
gen/        rollout.py
curate/     sft.py · kto.py · dpo.py · guards.py
infra/      serve_qwen.{sh,md} · format_fixture.json
train/      sft/ · kto/ · dpo/ · grpo/   (configs per rung)
```

## Open dependencies
- **Validator files** to be delivered by user (M1 blocked until then; M0 can proceed).
- **Compute** undecided → M0/M5/M6 sequencing + effort conditional.
- **Judge model** for D4 alignment + `missing_confirmation` (Claude assumed).

## Suggested first step
M0 in two parallel tracks: (1) serve Qwen3-8B + lock the non-thinking format
fixture; (2) `eval/protocol.py` + `reward_capture.py`. Both are compute-light to
*write*; only the actual eval runs need GPUs. That makes M3's baseline runnable
the moment compute lands and the validator arrives.
```
