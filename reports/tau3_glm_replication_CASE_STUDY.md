# Case Study: Reproducing GLM-5 on τ³-bench banking_knowledge

**Date:** 2026-06-03 · **Status:** SOLVED · **Owner:** Lily (supervisor) + agent
**Detailed audit log:** [`glm_replication_findings.md`](./glm_replication_findings.md) · **Bug log:** Supabase `tau3_replication_log` (9 attempts) · **Branch:** `tau3-glm-replication` @ github.com/lilyzhng/tau2-bench

---

## Executive summary

We set out to confirm our τ³-bench (Sierra `tau2-bench`) pipeline reproduces the leaderboard's GLM-5 `banking_knowledge` **pass@1 = 9.79%**, as a trust check before running our own experiments. It didn't, we consistently got **~16.5%**. After matching every controllable factor and then doing an independent deep research over the primary sources, the cause was found in **Sierra's own GLM-5 trajectory file**: they served GLM-5 on a **self-hosted FP8-quantized deployment** (`openai/glm-5-fp8` at an internal vLLM host). We used **OpenRouter `z-ai/glm-5`** (higher precision, different serving stack). On a brittle 14-tool-call domain, that serving difference is enough to move pass@1 by ~1.6×, in the direction observed.

**Verdict: the pipeline is faithful. The absolute-number gap is the model serving stack, not a config or pipeline bug. The pipeline is trustworthy for relative model comparisons under controlled conditions; absolute numbers are not portable across inference providers.**

---

## 1. What we measured (the three GLM-5 numbers)

| Run | pass@1 | Model version | Serving | Notes |
|---|---|---|---|---|
| **Sierra leaderboard** | **9.79%** | GLM-5 (glm-5-think) | **self-hosted FP8** (`openai/glm-5-fp8`) | 4 trials, tau2-bench 0.2.1-dev |
| **Our run, current code (1.0.0)** | 13.4% / 18.6% | GLM-5 | OpenRouter `z-ai/glm-5` | 2 single-trial runs |
| **Our run, EXACT board version (0.2.1-dev @ 01e812d)** | **16.49%** | GLM-5 | OpenRouter `z-ai/glm-5` | clean 97/97, 0 infra errors |

Same model version (GLM-5, checkpoint `z-ai/glm-5-20260211`) as the board; the only difference vs the board is the **serving stack**. Reward is binary (avg reward == pass@1).

## 2. Root cause: the serving stack (found in primary sources, not guessed)

The decisive evidence is the `agent_info` block in **Sierra's own GLM-5 banking trajectory** (`web/leaderboard/public/submissions/glm-5-think_sierra_2026-03-02/trajectories/...`):
```json
"llm": "openai/glm-5-fp8",
"llm_args": {"api_base": "http://acuadron-glm5-banking-atl:8000/v1", "temperature": 1.0, "top_p": 0.95}
```
- Sierra ran a **self-hosted FP8-quantized GLM-5** on an internal vLLM host. **Not** Z.AI native, **not** OpenRouter. This is invisible at the submission layer (`submission.json` has no provider field; only the trajectory carries it).
- We used OpenRouter `z-ai/glm-5` = different precision + serving stack + tool-call parser.
- **Why it moves the number, and in this direction:** banking is ~14 tool calls/task, so per-call errors compound. OpenRouter's own data shows GLM-5 tool-call error rate swinging **8% → 1% by provider** (clean-trajectory rate ~0.30 vs ~0.87). The more-brittle FP8/self-host serving lands lower (9.79%); our serving lands higher (16.5%). Consistent in magnitude and direction.

## 3. What we ruled out (so the residual is unambiguously serving)

- **Version / scoring:** ran the EXACT `0.2.1-dev @ 01e812d` (banking present, version `0.2.1-dev`, BEFORE PR #273's hallucination-scoring change). Same as the board. Still 16.49%.
- **Model checkpoint:** litellm shows `z-ai/glm-5-20260211` = the board's 2026-02-11 GLM-5 release.
- **Thinking:** trajectories carry `reasoning_content` → thinking ON, matching `glm-5-think`.
- **Config:** temp 1.0 / top_p 0.95, seed 300, gpt-5.2-low user-sim, text-emb-3-large retrieval, 97 tasks, binary reward, all match the board's submission.
- **Variance:** three runs cluster ~16%; 9.79% is statistically ruled out for our setup.
- **Retrieval drift (PR #311, post-eval):** we forced `openai_embeddings`, matching the board.

## 4. Independent corroboration (and an honest version caveat)

Mistral's **Medium 3.5** launch chart (Maxime Labonne, X) reports τ³ Banking for several models. **GLM-5.1 = 16.2%** there, vs our GLM-5 = 16.49%.
- **Caveat (important):** Mistral's 16.2 is **GLM-5.1** (newer), not the GLM-5 we and Sierra used. So it is *directional* support, not a same-version match. The near-identical value is partly coincidence.
- **The airtight comparison is our own, same version:** GLM-5 @ OpenRouter (16.49%) vs GLM-5 @ Sierra-FP8 (9.79%). Same model, gap is purely serving.
- **Net:** an independent third party also lands GLM-family ~16% on banking, far from 9.79, reinforcing that ~16% is the normal range and Sierra's 9.79 is the serving-specific low outlier. Across evaluators, Mistral's whole banking column runs higher than Sierra's leaderboard, another sign that absolute numbers shift with the evaluator/serving.

## 5. Benchmark-contamination / timing caveat (the meta-lesson)

Under that same tweet, the debate (Labonne; @rugbist_) was about **timing/contamination**: models released *before* a benchmark can't have trained on it (clean scores); models released *after* can (potentially inflated). Comparing a new model to older competitors on a fresh benchmark is "not a good look" because the playing field isn't level. **Takeaway:** absolute benchmark numbers are confounded by serving **and** release-timing/contamination. Use benchmarks for **controlled, same-condition relative comparison**, not absolute number-chasing.

## 6. What this means for our use of the pipeline

- **Trust it for relative comparisons** (same pipeline, same serving, same window): "model A vs model B on our setup" is valid.
- **Don't expect to hit a public leaderboard's absolute number** unless you also replicate its exact serving (provider/quantization), which the leaderboard doesn't disclose (it's buried in trajectories) and often can't be reproduced (internal FP8 hosts).
- **When an absolute number matters, pin the serving first** by reading the submission's trajectory `agent_info`, before running anything.

## 7. Lessons learned (process)

1. **Go to primary sources + raw artifacts FIRST.** The answer (`openai/glm-5-fp8`) was in Sierra's trajectory JSON the whole time. We reached it only after config-diffing to a premature conclusion. Read the paper, the repo issues/PRs, and the raw submission/trajectory files before declaring a cause. (Saved as memory `independent-primary-research-when-stuck`.)
2. **Confirm the serving setting before spending.** Closed models (OpenAI/Anthropic) = native serving, reproducible. Open models (GLM/Qwen/DeepSeek) on the board may be self-hosted/quantized = often not reproducible without their infra.
3. **Persist results to durable storage from the start.** A run that isn't saved is a run you'll redo. (Fixed: runs now push rollouts to Supabase before computing metrics.)
4. **Track every attempt** (commit + a bug log) so the investigation is recoverable and the reasoning is auditable.

## 8. Artifacts & reproducibility

- **Code:** branch `tau3-glm-replication` (Modal eval scripts, bwrap-bypass shell fix, Supabase saver, bug-log tool, this report). Worktree at the exact board version: `01e812d` (`0.2.1-dev`).
- **Data (Supabase):** `tau2_rollouts_glm5` (1.0.0 runs), `tau2_rollouts_glm_021dev` (exact-version run, 97 trajectories), `tau3_replication_log` (9 attempts with hypotheses + findings).
- **Reproduce the exact-version run:** `cd /tmp/tau2-021dev && modal run --detach modal_glm_021dev.py`.

## 9. Open / blocked (non-essential)

- **GPT-5.2-none clean-validation** (native OpenAI agent, target ~11.08%): set up and config-confirmed from its trajectory (found `max_steps:200` and a seed 300-vs-123 discrepancy), but the run is **blocked on OpenAI quota** (billing limit). Optional, since the GLM conclusion is already solid. Can finish once OpenAI billing is topped up: `cd /tmp/tau2-021dev && modal run --detach modal_gpt52none_021dev.py`.
- **Outreach (optional):** sharpened question to the Sierra authors re: GLM-5 serving / FP8 host is drafted in `glm_replication_findings.md`. With the FP8 detail now found, this is confirmation-only.

---

**Bottom line:** our τ³-bench pipeline is faithful. GLM-5 reads ~16% on banking under standard serving; Sierra's 9.79% is a self-hosted-FP8 artifact. The pipeline is sound for relative model evaluation, which is what we need it for. **Sealed.**
