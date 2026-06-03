# GLM-5 banking_knowledge replication — findings (overnight 2026-06-03)

**Goal:** confirm our pipeline reproduces the leaderboard's GLM-5 `banking_knowledge` **pass@1 = 9.79%**, so we know the pipeline is faithful before running our own experiments.

**Branch:** `tau3-glm-replication` @ github.com/lilyzhng/tau2-bench · **bug log:** Supabase `tau3_replication_log`

## What we measured (clean, dedup'd from `tau2_rollouts_glm5`)
| Run | pass@1 | notes |
|---|---|---|
| Board (GLM-5-think) | **9.79%** | 4 trials, tau2-bench **0.2.1-dev** |
| Our 06-02 (bootstrap) | **13.4%** (13/97) | 1 trial |
| Our 06-03 | **18.6%** (18/97) | 1 trial |

- **Reward is binary** here (avg reward == pass-rate), so avg reward IS pass@1, no metric artifact.
- Our two single-trial runs differ by 5 tasks (13 vs 18) → **single-trial variance is large** (~±5%). The board used **4 trials**.

## Config audit — we MATCH the board on everything documented
Board's canonical config (`submissions/glm-5-think_sierra_2026-03-02/submission.json`):
user-sim **gpt-5.2 reasoning_effort low** ✓ · **seed 300** ✓ · **temp 1.0 / top_p 0.95** ✓ · retrieval **text-emb-3-large** (= our openai_embeddings) ✓ · 97 tasks ✓ · binary reward ✓.

## Root cause of the gap (why we read HIGHER than 9.79%)
1. **Version mismatch (the big one).** Board = `tau2_bench_version: 0.2.1-dev`. We're on **v1.0.0**. `banking_knowledge` is **new in 1.0.0**, and the board's 0.2.1-dev is an **untagged dev snapshot** (git tags jump v0.2.0 → v1.0.0, no 0.2.1). We cannot cleanly check it out.
2. **Scoring changed between those versions.** Per CHANGELOG: hallucinated tool calls are now treated as **no-ops** during `set_state` replay, so **trajectories that hallucinate-then-recover now score as successes**; previously they failed / were binned as INFRASTRUCTURE_ERROR. This **inflates our pass@1** vs the board, exactly our symptom (we're higher, not lower).
3. **Single-trial variance** on top (board ran 4 trials, averaged).

**Conclusion:** the pipeline is **configured faithfully**; the residual gap is a **benchmark-version difference (0.2.1-dev → 1.0.0 scoring) + variance**, not a config bug. Exact 9.79% on v1.0.0 is likely **not reproducible** because the scoring rules changed.

## Recommended next steps (your call — each costs a bit)
1. **Run 4 trials** on v1.0.0 to kill variance and get our stable v1.0.0 number (cheap-ish, GLM). Expect ~12–16%.
2. **Document the version caveat** in the report: "our numbers are on v1.0.0, which scores recovered-hallucination trajectories more leniently than the board's 0.2.1-dev."
3. If exact 0.2.1-dev replication is required → **deep research** to pin the exact dev commit + scoring, or contact Sierra for the 0.2.1-dev ref. (This is the "if still stuck, do deep research" path.)

**No more spend was incurred after the GLM run.** M3 + Opus were stopped early. Nothing else is running.
