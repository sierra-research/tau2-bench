# Synthetic retail tasks (τ²-bench)

Synthetic, tau2-compliant **retail tasks** for post-training Qwen3-8B, targeting
three failure modes found in baseline error analysis: **conditional fallback**,
**multi-goal phrasing**, and **mid-call mind-change**. Full design rationale is in
[../PLAN_v2.md](../PLAN_v2.md) → *Implemented approach (v1)*.

## Where the tasks are

| File | Tasks | Use this if… |
|---|---|---|
| **`tasks_failuremode.json`** | **222** (74 / failure mode) | **the main set** — train-seed augmented + from-scratch top-up |
| `tasks_augmented.json` | 92 | only the train-seed-augmented subset |
| `tasks_topup.json` | 130 | only the from-scratch top-up subset |
| `tasks_synth.json` | 20 | an earlier from-scratch coverage demo (not failure-mode-specific) |

Each file is a JSON **`list[Task]`** that validates against tau2's own model
(`tau2.data_model.tasks.Task`) — drop-in for `tau2 run` or the evaluator.

```python
import json
from tau2.data_model.tasks import Task
tasks = [Task.model_validate(t) for t in json.load(open("synth/tasks_failuremode.json"))]
```

Task ids encode provenance and pattern, e.g.
`retail_aug_<seedid>_mid_call_mind_change` (augmented from train task `<seedid>`)
or `retail_topup_syn_<user>_conditional_fallback` (from-scratch).

## How to use them

**1 — Generate trajectories** (needs an agent endpoint + API key):
```bash
# registers the tasks as the tau2 task set `retail_failuremode`, then runs the
# agent vs the user simulator; tau2 scores + saves reward in the results file
uv run python synth/run_trajectories.py --agent-llm <model> [--api-base URL] --num-trials 4
```

**2 — Keep only good trajectories** (`reward==1` AND policy-legal):
```bash
uv run python synth/filter_legal.py data/simulations/<run>/results.json
```
(The legality filter activates once `retail_policy_validator.py` is on the path;
until then it reports legality as *skipped* and filters on `reward==1` only.)

To point tau2 at a different task file: `SYNTH_TASKS_FILE=path uv run python synth/run_trajectories.py …`

## How they were created

Each task augments a real benchmark scenario by injecting **one** failure-mode
pattern. The key invariant: **the gold solution (`evaluation_criteria.actions`,
the verifiable DB target) is derived in code** — execution-validated on a fresh
retail env — while an **LLM only writes the natural-language prose**. The model
never decides the answer, so correctness is independent of the LLM, and DB-state
reward alone catches each failure mode.

Two sources, combined into `tasks_failuremode.json`:

1. **Train-seed augmentation** (`build_augmented.py`) — takes the **74 train-split
   tasks** as seeds (`seeds.py`) and applies the three pattern transforms
   (`patterns.py`).
2. **From-scratch top-up** (`build_topup.py`) — builds fresh base tasks on the
   test-excluded user pool and applies the same transforms, filling each pattern
   to 74.

**Every task passes:** (a) gold actions execute with no error, (b) the DB changes
as intended, (c) **decontamination** — touches **no test-split user, order, or
solution** (the 40-task test split is held out for evaluation), (d) tau2 schema
validation.

Regenerate from source:
```bash
uv run python synth/build_augmented.py   # → tasks_augmented.json (add --llm for natural prose)
uv run python synth/build_topup.py       # → tasks_topup.json + tasks_failuremode.json
```

## Good to know

- **Prose is currently deterministic templates.** Re-run `build_augmented.py --llm`
  (needs an API key) for natural phrasing; the gold actions and validation are
  identical either way.
- **Decontamination is strict**: nothing here overlaps the test split. If you will
  instead evaluate on `base` (all 114 tasks), set `PROTECTED_SPLITS=("train","test")`
  in `lib.py` and regenerate.
- **`mid_call_mind_change`** is realized during the rollout (the user simulator
  asks for one thing, then switches); the static task is valid regardless, and a
  wrong final DB state filters out trajectories that mishandle the switch.
