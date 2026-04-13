# tau2-TRACE: Trajectory-Aware Observability for tau2-bench

> High-speed, heuristic-based process metrics for tau2-bench -- answering not just *"did the agent succeed?"* but *"how did it get there?"*

## Problem and Motivation

tau2-bench evaluates agents with a binary `pass^k` metric: does the final database state match ground truth? While mathematically rigorous, this hides operational realities:

| Question | tau2-bench Today | tau2-TRACE Adds |
|---|---|---|
| Did the agent succeed? | Final DB state match (`pass^k`) | Preserved -- we don't replace this |
| Were tools called in order? | Presence check only (`ActionEvaluator`) | Phase-ordering check against transcribed workflow constants + read-after-write verification |
| Was communication clear? | Substring match (`CommunicateEvaluator`, has `TODO: This could be improved!` in source) | Guidance precision (keyword matching for 18 domain terms) + token-to-action ratio |
| Was the agent efficient? | Not measured | Turns vs. expected, action density, loop detection, redundancy detection |
| Why did the agent fail? | `reward=0` -- no detail | Error bursts, recovery pairs, and signature error classification narrow debugging scope |
| Is the agent robust to user noise? | Not measured | Adversarial wrapper injects interruptions and self-corrections via Proxy Pattern |

## Approach

tau2-TRACE is a **heuristic-based observability layer** that analyses existing `SimulationRun.messages` across three tiers:

```
                    SimulationRun.messages
                             |
              +--------------+--------------+
              v              v              v
  +------------------+ +-----------+ +------------------+
  | Layer 1:         | | Layer 2:  | | Layer 3:         |
  | Efficiency       | | Adherence | | Interaction      |
  | (Deterministic)  | | (Determ.) | | Quality          |
  |                  | |           | |                  |
  | Redundancy       | | Phase-    | | Action density   |
  | Loops            | | order     | | Token ratio      |
  | Error bursts     | | constants | | Guidance prec.   |
  | Recovery pairs   | | Read-     | | Repeated info    |
  | Orphan tracking  | | after-    | | ----------       |
  |                  | | write     | | Opt-in LLM judge |
  +--------+---------+ +-----+-----+ +--------+---------+
           +------------------+------------------+
                              v
                    CompositeScorecard
                  (merged into DataFrame)
```

**Design constraints:**
- **Zero LLM calls by default** -- all Layer 1-2 metrics are deterministic heuristics, O(n)
- **Opt-in semantic evaluation** -- Layer 3 offers a Chain-of-Thought LLM judge (`--llm-judge`) via `tau2.utils.llm_utils.generate()`
- **Zero core changes** -- lives entirely in `src/experiments/tau2_trace/`; verified by `git diff src/tau2/` = empty
- **Zero dependency additions** -- uses only packages already in `pyproject.toml`
- **Domain-aware** -- telecom gets phase-order checks using transcribed workflow constants; retail/airline get read-after-write verification

## Illustrative Case Studies

End-to-end validated against live `gpt-4.1-mini` simulations across all four production domains (mock, telecom, retail, airline). These are single-task runs -- they demonstrate the *kind* of diagnostic gap tau2-TRACE exposes, not statistically representative benchmarks. All numbers below come from actual E2E runs; full output is in [EXAMPLE_OUTPUT.md](EXAMPLE_OUTPUT.md).

**Telecom baseline (reward=1.0, `$0.017`):**
The agent passed but required 39 turns for a 1-action task (`trace_turns_vs_expected=39.0`), made 17 tool calls (8 agent, 9 user), and matched workflow `path1_no_service` with 100% phase-ordering adherence. Guidance precision was 83.3% (keyword heuristic). The binary reward hides that this agent is operationally expensive.

**Telecom adversarial (reward=0.0, 6 perturbations injected):**
Same task, same model, but with `AdversarialSimulatorWrapper` injecting interruptions and self-corrections at 30% rate. The agent failed -- policy adherence dropped to 0.0, action density fell from 0.44 to 0.28, and the agent could not recover from user perturbations. This demonstrates the wrapper's impact on agent robustness evaluation.

**Retail baseline (reward=1.0, `$0.009`):**
19 turns for 5 tool calls, `trace_turns_vs_expected=3.8`. Zero redundant calls, zero loops, zero errors. Cheaper and more efficient than the telecom agent despite both passing.

**Airline baseline (reward=1.0, `$0.003`):**
9 turns for 1 tool call, `trace_turns_vs_expected=9.0`. Read-after-write score 1.0, policy adherence 1.0. Telecom-only metrics (guidance precision, workflow matching) correctly skipped by domain routing. Cheapest of all domains.

**Deterministic vs. LLM judge on the same simulation:**

| Metric | Deterministic | CoT LLM Judge (gpt-4.1-mini) |
|---|---|---|
| `trace_repeated_info_requests` | 6 | 0 |
| `trace_guidance_precision` | 0.833 | 0.667 |

The deterministic heuristic over-counted repeated-info (token-in-string false positives on common substrings). The CoT LLM judge was stricter on guidance precision, requiring specific actionable steps rather than just keyword presence. Both modes produce valid but different signals.

**Before and after on the same simulation:**

```
# tau2-bench alone:
task_id=[mobile_data_issue]user_abroad...  reward=1.0  agent_cost=$0.017

# tau2-bench + tau2-TRACE:
task_id=[mobile_data_issue]user_abroad...  reward=1.0  agent_cost=$0.017
  trace_turns_vs_expected=39.0  trace_redundant_tool_calls=0
  trace_matched_workflow=path1_no_service  trace_guidance_precision=0.833
  trace_error_recovery_rate=1.0  trace_error_burst_count=0
  trace_orphan_tool_messages=0  trace_read_after_write_score=1.0
  trace_policy_adherence=1.0  trace_action_density=0.436
```

## Quick Start

```bash
# Post-hoc analysis of existing results (deterministic, zero-cost, no API key)
python -m experiments.tau2_trace.run_experiment analyze \
    --results-file data/simulations/my_run.json

# With configurable error recovery window
python -m experiments.tau2_trace.run_experiment analyze \
    --results-file data/simulations/my_run.json \
    --recovery-window 5

# With opt-in CoT LLM judge (requires API key)
python -m experiments.tau2_trace.run_experiment analyze \
    --results-file data/simulations/my_run.json \
    --llm-judge --llm-judge-model gpt-4.1

# Live simulation with adversarial user wrapper (requires API key)
python -m experiments.tau2_trace.run_experiment run \
    --domain telecom --agent-llm gpt-4.1 --user-llm gpt-4.1 \
    --adversarial --perturbation-rate 0.3 --num-tasks 1

# Live simulation without adversarial (baseline comparison)
python -m experiments.tau2_trace.run_experiment run \
    --domain telecom --agent-llm gpt-4.1 --user-llm gpt-4.1 \
    --num-tasks 3
```

Output: CSV with all standard tau2-bench columns (27) plus `trace_*` columns (31), totalling 58 columns, ready for pandas analysis.

### Broader Validation

For statistically meaningful results, run a sweep across multiple tasks, models, and domains. Use `--task-split base` to evaluate on the full standard task set:

```bash
# Broader sweep: 10 tasks x 2 trials across domains
for domain in telecom retail airline; do
  for model in gpt-4.1-mini gpt-4.1; do
    python -m experiments.tau2_trace.run_experiment run \
      --domain $domain --agent-llm $model --user-llm $model \
      --task-split base --num-tasks 10 --num-trials 2 \
      --output results/${domain}_${model}
  done
done

# Then compare across models via pandas
python -c "
import pandas as pd, glob
dfs = [pd.read_csv(f) for f in glob.glob('results/*.csv')]
df = pd.concat(dfs)
print(df.groupby('info_domain')[['trace_action_density','trace_turns_vs_expected','trace_policy_adherence']].describe())
"
```

## Technical Implementation

- **Pydantic models** throughout -- consistent with `tau2.data_model.*` conventions; uses `ToolRequestor` Literal from `tau2.data_model.message`
- **Error recovery redesign** -- consecutive failures on the same tool are grouped into *Error Bursts* rather than inflating error counts; each recovery returns a `RecoveryPair(failed, recovered)` for diff inspection; signature errors (wrong function name, detected via scoped pattern matching) allow cross-name recovery
- **Dict comparison via `get_dict_hash`** from `tau2.utils.utils` -- handles key ordering and nested structures reliably; previous hash cached to avoid recomputation
- **Orphan tool message tracking** -- unmatched ToolMessages are logged via `loguru.warning` and surfaced as `trace_orphan_tool_messages` rather than silently absorbed
- **Token-to-action ratio** -- prefers actual `AssistantMessage.usage["completion_tokens"]` when available, falls back to character count as proxy
- **LLM judge with Chain-of-Thought** -- lazy-imports `tau2.utils.llm_utils.generate()`, prompts require explicit step-by-step reasoning in a `"reasoning"` field before the score; parses responses via `extract_json_from_llm_response()` (handles markdown code blocks); gracefully falls back to deterministic evaluation on any failure (safe for CI/CD)
- **Adversarial testing** -- `AdversarialSimulatorWrapper` inherits `HalfDuplexUser` and injects seeded interruptions and self-corrections via the Proxy Pattern. Forwards `set_seed()` to the base simulator for Orchestrator compatibility. Integrated into the `run` subcommand with `--adversarial`, `--perturbation-rate`, `--adversarial-seed` flags. E2E verified: 6 perturbations injected, caused agent failure (1.0 -> 0.0). Zero core code touched.
- **Phase constant validation** -- 6 CI-ready tests parse actual `@is_tool(...)` decorated functions from telecom/retail/airline source files and validate phase constants against them. Caught 3 real drift bugs (`get_line_details` never existed; `get_customer_by_name` and `get_data_usage` were missing). DOT workflow file existence and step coverage also verified.

## Known Limitations

These are intentional trade-offs for speed and simplicity, not bugs:

- **Redundancy detection is strict exact-match.** Consecutive calls with identical name + arguments (compared via `get_dict_hash`). Semantically equivalent calls with different parameters are not flagged. This prevents false positives at the cost of false negatives.
- **Loop detection uses a fixed sliding window (size 3).** Misses interleaved loops or near-loops with slight variations. Chosen for O(n) speed and zero false positives over recall.
- **Guidance precision uses a curated set of 18 telecom-specific terms** (e.g., "airplane mode", "APN", "roaming"). An agent mentioning the right keyword in the wrong context would score as precise. This is a fast heuristic, not semantic understanding.
- **Repeated-info detection is token-in-string matching** with a 6-character minimum token length. Works well for structured identifiers (customer IDs, order numbers) but can over-count on common substrings. Our E2E testing confirmed this: deterministic mode reported 6 repeated-info requests where the CoT LLM judge found 0 on the same simulation.
- **Phase-order evaluation uses manually transcribed constants**, not runtime DOT file parsing. The phase matching checks non-decreasing first-occurrence indices -- it validates high-level sequencing, not strict causal dependencies. We chose this over adding `networkx`/`pydot` dependencies. Validation tests parse actual tool source files to catch drift (and already caught 3 real bugs).
- **LLM judge uses Chain-of-Thought but is not calibrated.** The opt-in prompts require step-by-step reasoning before scoring, which improves reliability over raw JSON output, but there is no multi-rater calibration or rubric anchoring. Recommended for nightly or release-candidate evaluations, not every CI run.
- **Retail/airline metrics are thinner than telecom.** No DAG workflow files exist for these domains -- they only get read-after-write verification and no guidance precision.
- **Case studies are single-task runs.** They demonstrate the kind of diagnostic insight tau2-TRACE provides, not statistical claims about agent quality.

## Verification

All verification was performed end-to-end and is reproducible. See [EXAMPLE_OUTPUT.md](EXAMPLE_OUTPUT.md) for the full testing strategy and output.

- **75/75 unit tests passing** (1.35s) across 6 test files -- unit tests, integration tests with file I/O, end-to-end DataFrame merge, mock-verified LLM judge tests, phase constant validation against source, and Orchestrator compatibility
- **Core tau2-bench tests passing** -- the 1 failure is a pre-existing LLM-dependent flaky test (`test_run_tasks_action_checks`), unrelated to tau2-TRACE
- **Synced with latest upstream** -- merged latest sierra-research/tau2-bench changes; migrated from removed `BaseUser` to `HalfDuplexUser`, added `set_seed` forwarding for Orchestrator compatibility
- **6 live E2E simulations** against real `gpt-4.1-mini` API: mock baseline, telecom baseline, telecom adversarial, retail baseline, airline baseline, telecom with LLM judge
- **`ruff check`** -- all lint checks passed
- **`ruff format`** -- all formatting verified
- **`git diff src/tau2/`** -- empty (zero core files modified)
- Result artifacts in `results/` are **example-only snapshots** from E2E runs at a specific point in time (gitignored). See [EXAMPLE_OUTPUT.md](EXAMPLE_OUTPUT.md) for reproduction steps

## File Structure

```
src/experiments/tau2_trace/
├── models.py                  # Pydantic: ToolCallRecord, RecoveryPair, ErrorBurst, CompositeScorecard
├── trajectory_analyzer.py     # Parse messages -> efficiency metrics (redundancy, loops, burst recovery)
├── tool_order_evaluator.py    # Phase-order constants (telecom) + read-after-write (all domains)
├── interaction_quality.py     # Action density, token ratio, guidance precision, CoT LLM judge
├── domain_router.py           # Domain-aware dispatch -> CompositeScorecard
├── adversarial_wrapper.py     # UserSimulator proxy (interruptions, self-corrections)
├── run_experiment.py          # CLI: analyze (post-hoc) + run (live with --adversarial)
├── results/                   # Example-only snapshots (gitignored; see EXAMPLE_OUTPUT.md to reproduce)
├── tests/                     # 75 tests (6 files)
├── README.md
└── EXAMPLE_OUTPUT.md          # Full testing strategy, E2E outputs, claim-by-claim verification
```

---

<details>
<summary><strong>Metrics Reference</strong> (click to expand)</summary>

### Trajectory Efficiency (Heuristic, Deterministic)

| Metric | Definition | Method |
|---|---|---|
| `trace_redundant_tool_calls` | Consecutive calls with identical name + arguments | Strict exact-match via `get_dict_hash` |
| `trace_loop_count` | Repeating tool call patterns | Sliding window (size 3), name-sequence only |
| `trace_error_count` | Raw count of tool calls that returned errors | Direct count |
| `trace_error_recovery_rate` | Errors followed by successful retry within configurable window | Same-name match (or any-name for signature errors) |
| `trace_error_burst_count` | Consecutive same-tool failures grouped into logical events | Burst grouping by tool name |
| `trace_error_bursts_recovered` | How many bursts ended with a successful recovery | Burst-level recovery tracking |
| `trace_orphan_tool_messages` | ToolMessages that could not be matched to a pending call | ID-based matching |

### Policy Adherence (Heuristic, Deterministic)

| Metric | Definition | Method |
|---|---|---|
| `trace_policy_adherence` | Telecom: phase-ordering score. Retail/Airline: read-after-write score | Non-decreasing first-occurrence indices across transcribed phase constants |
| `trace_matched_workflow` | Best-matching telecom path (no_service / mobile_data / mms) | Best score across 3 workflow phase sets |
| `trace_read_after_write_score` | Write operations verified by subsequent read within 5 calls | Domain-specific write-to-read mapping |

### Interaction Quality (Heuristic + Opt-In LLM)

| Metric | Definition | Method |
|---|---|---|
| `trace_action_density` | `tool_calls / turns` | Direct ratio |
| `trace_token_to_action_ratio` | Agent tokens per tool call | Prefers `msg.usage`, falls back to char count |
| `trace_turns_vs_expected` | `actual_turns / expected_actions` | Direct ratio (closer to 1.0 = better) |
| `trace_guidance_precision` | (Telecom) Fraction of messages with domain terms | 18-term keyword set matching |
| `trace_repeated_info_requests` | Agent asks for data already in prior tool results | Token-in-string (>=6 chars) or CoT LLM judge |

### Domain Routing

| Domain | Phase-Order | Read-After-Write | Guidance | LLM Judge |
|---|---|---|---|---|
| Telecom | 3 workflow phase sets | Yes | Yes (18 terms) | Opt-in (CoT) |
| Retail | -- | Yes | -- | Opt-in (CoT) |
| Airline | -- | Yes | -- | Opt-in (CoT) |

</details>

<details>
<summary><strong>Future Work</strong> (click to expand)</summary>

- **Deeper redundancy detection**: Semantic similarity between tool calls (e.g., same-intent but different parameter formatting) using embedding distance.
- **DAG parsing at runtime**: Parse the actual DOT files with `pydot`/`networkx` to derive causal dependency edges, enabling true dependency violation detection.
- **Retail/airline workflow DAGs**: Author workflow phase constants for these domains (requires domain expertise).
- **LLM judge calibration**: Multi-rater agreement metrics, rubric anchoring, and confidence intervals for the semantic evaluation layer.
- **Adversarial persona expansion**: Additional perturbation types (confused user, multilingual switches, emotional escalation) calibrated against real user distributions.

</details>

<details>
<summary><strong>Related Work</strong> (click to expand)</summary>

Draws on: TRAJECT-Bench (tool usage diagnostics), TRACE (evidence banks), Agent GPA (temporal phase evaluation), SWE-eval (info-gain metrics). Adapted for tau2-bench's Dec-POMDP dual-control environment with a deterministic-first, heuristic-based approach.

</details>
