# Running Simulations

This guide covers the `tau2.runner` API for running simulations at different levels of control. Whether you want a one-liner CLI command or full programmatic control over every component, the runner framework has you covered.

## Architecture Overview

The runner framework is organized into three layers:

| Layer | Module | Purpose |
|-------|--------|---------|
| **Layer 1** | `runner.simulation` | Execute a pre-built orchestrator. No registry, no config. |
| **Layer 2** | `runner.build` | Build instances (environment, agent, user, orchestrator) from names/config via the registry. |
| **Layer 3** | `runner.batch` | Batch execution with concurrency, checkpointing, retries, logging, and side effects. |

Each layer builds on the one below it, but you can enter at any level depending on your needs.

---

## Level 1: CLI (Simplest)

The fastest way to run simulations is through the `tau2 run` command:

```bash
# Run all airline tasks with GPT-4.1
tau2 run --domain airline --agent llm_agent --agent-llm openai/gpt-4.1

# Run specific tasks with multiple trials
tau2 run --domain retail --agent llm_agent --agent-llm openai/gpt-4.1 \
    --task-ids 0 1 --num-trials 3

# Run with concurrency and auto-resume
tau2 run --domain telecom --agent llm_agent --agent-llm openai/gpt-4.1 \
    --max-concurrency 4 --auto-resume
```

See `tau2 run --help` or [CLI Reference](cli-reference.md) for all options.

---

## Level 2: Config-Based Python API (Intermediate)

For programmatic use, create a `TextRunConfig` (for text/half-duplex) or `VoiceRunConfig` (for voice/full-duplex) and call `run_domain()`:

### Text (half-duplex) simulations

```python
from tau2 import TextRunConfig
from tau2.runner import run_domain

config = TextRunConfig(
    domain="airline",
    agent="llm_agent",
    llm_agent="openai/gpt-4.1",
    llm_user="openai/gpt-4.1-mini",
    num_trials=3,
    max_concurrency=4,
    seed=42,
)

results = run_domain(config)
# compute_metrics() returns an AgentMetrics object with avg_reward, pass_hat_ks, etc.
from tau2.metrics.agent_metrics import compute_metrics
metrics = compute_metrics(results)
print(f"Average reward: {metrics.avg_reward}")
```

### Voice (full-duplex) simulations

```python
from tau2 import VoiceRunConfig
from tau2.data_model.simulation import AudioNativeConfig
from tau2.runner import run_domain

config = VoiceRunConfig(
    domain="airline",
    audio_native_config=AudioNativeConfig(
        provider="openai",
        model="gpt-4o-realtime-preview",
    ),
    llm_user="openai/gpt-4.1",
    speech_complexity="regular",
)

results = run_domain(config)
```

### Knowledge retrieval simulations

```python
from tau2 import TextRunConfig
from tau2.runner import run_domain

config = TextRunConfig(
    domain="banking_knowledge",
    agent="llm_agent",
    llm_agent="openai/gpt-4.1",
    llm_user="openai/gpt-4.1",
    retrieval_config="bm25",  # or "openai_embeddings", "terminal_use", etc.
    num_trials=3,
)

results = run_domain(config)
```

Both types share a common `BaseRunConfig` with fields like `domain`, `num_trials`, `seed`, `hallucination_retries`, etc. The `RunConfig` type alias is `Union[TextRunConfig, VoiceRunConfig]`.

This handles everything: task loading, filtering, concurrency, checkpointing, metrics display.

### Running specific tasks

```python
from pathlib import Path
from tau2 import TextRunConfig
from tau2.runner import get_tasks, run_tasks

config = TextRunConfig(
    domain="airline",
    agent="llm_agent",
    llm_agent="openai/gpt-4.1",
)

# Load and filter tasks manually
tasks = get_tasks("airline", task_ids=["0", "1"])

# Run with custom save path
results = run_tasks(
    config,
    tasks,
    save_path=Path("my_results/results.json"),
    save_dir=Path("my_results/"),
)
```

### Running a single task

```python
from tau2 import TextRunConfig
from tau2.runner import run_single_task, get_tasks

config = TextRunConfig(domain="airline", agent="llm_agent", llm_agent="openai/gpt-4.1")
tasks = get_tasks("airline")

result = run_single_task(config, tasks[0], seed=42)
print(f"Task {result.task_id}: reward={result.reward_info.reward}")
```

---

## Level 3: Instance-Based Python API (Power User)

For maximum control, build instances yourself and use `run_simulation()`. This is ideal for:
- Custom agents or environments not in the registry
- Fine-grained control over agent/user construction
- Testing and development
- Integration into custom pipelines

### Using build helpers with the registry

```python
from tau2.runner import (
    build_environment,
    build_agent,
    build_user,
    build_orchestrator,
    run_simulation,
    get_tasks,
)
from tau2.evaluator.evaluator import EvaluationType

# Load a task
tasks = get_tasks("airline")
task = tasks[0]

# Option A: Build orchestrator from config (uses registry)
from tau2 import TextRunConfig
config = TextRunConfig(domain="airline", agent="llm_agent", llm_agent="openai/gpt-4.1")
orchestrator = build_orchestrator(config, task, seed=42)
result = run_simulation(orchestrator, evaluation_type=EvaluationType.ALL)

# Option B: Build components individually
env = build_environment("airline")
agent = build_agent("llm_agent", env, llm="openai/gpt-4.1")
user = build_user("user_simulator", env, task, llm="openai/gpt-4.1-mini")
```

### Fully custom instances (no registry needed)

```python
from tau2.runner import run_simulation
from tau2.orchestrator.orchestrator import Orchestrator
from tau2.user.user_simulator import UserSimulator

# Your custom environment and agent
env = MyCustomEnvironment()
agent = MyCustomAgent(tools=env.get_tools(), domain_policy=env.get_policy())

# Standard user simulator
user = UserSimulator(
    tools=env.get_user_tools(),
    instructions=str(task.user_scenario),
    llm="openai/gpt-4.1-mini",
)

# Wire everything into an orchestrator
orchestrator = Orchestrator(
    domain="my_domain",
    agent=agent,
    user=user,
    environment=env,
    task=task,
    max_steps=100,
    max_errors=10,
    seed=42,
)

# Run and evaluate
result = run_simulation(orchestrator)
print(f"Reward: {result.reward_info.reward}")
```

### Audio-native (voice) simulations

```python
from tau2.runner import build_voice_orchestrator, run_simulation, get_tasks
from tau2 import VoiceRunConfig
from tau2.data_model.simulation import AudioNativeConfig

config = VoiceRunConfig(
    domain="airline",
    audio_native_config=AudioNativeConfig(
        provider="openai",
        model="gpt-4o-realtime-preview",
    ),
    llm_user="openai/gpt-4.1",
    speech_complexity="regular",
)

tasks = get_tasks("airline")
orchestrator = build_voice_orchestrator(config, tasks[0], seed=42)
result = run_simulation(orchestrator)
```

---

## API Reference

### Layer 1: Execution

- **`run_simulation(orchestrator, *, evaluation_type=EvaluationType.ALL, env_kwargs=None)`** -- Run a pre-built orchestrator and evaluate the result. Returns `SimulationRun` with `reward_info` attached.

### Layer 2: Build

- **`build_environment(domain, *, solo_mode=False, env_kwargs=None)`** -- Build an environment from a domain name.
- **`build_agent(agent_name, environment, *, llm, llm_args, task, audio_native_config, solo_mode, audio_taps_dir)`** -- Build an agent from a registered name.
- **`build_user(user_name, environment, task, *, llm, llm_args, persona_config, solo_mode)`** -- Build a half-duplex user from a registered name.
- **`build_voice_user(environment, task, audio_native_config, *, llm, llm_args, voice_settings, persona_config, speech_complexity, seed, domain, hallucination_feedback, audio_taps_dir)`** -- Build a full-duplex voice user simulator.
- **`build_text_orchestrator(config, task, *, seed, simulation_id, user_persona_config)`** -- Build a half-duplex `Orchestrator` from a `TextRunConfig`.
- **`build_voice_orchestrator(config, task, *, seed, simulation_id, user_voice_settings, user_persona_config, hallucination_feedback, audio_taps_dir)`** -- Build a full-duplex `FullDuplexOrchestrator` from a `VoiceRunConfig`.
- **`build_orchestrator(config, task, *, seed, simulation_id, user_voice_settings, user_persona_config, hallucination_feedback, audio_taps_dir)`** -- Dispatcher that calls `build_text_orchestrator` or `build_voice_orchestrator` based on config type.

### Layer 3: Batch Execution

- **`run_domain(config)`** -- Run all tasks for a domain from a `RunConfig`. Handles task loading, filtering, save paths, metrics.
- **`run_tasks(config, tasks, *, save_path, save_dir, evaluation_type=EvaluationType.ALL_WITH_NL_ASSERTIONS, console_display)`** -- Run a list of tasks with concurrency, checkpointing, and retries. Note: defaults to `ALL_WITH_NL_ASSERTIONS` (unlike `run_simulation` which defaults to `ALL`).
- **`run_single_task(config, task, *, seed, evaluation_type, save_dir, ...)`** -- Run one task with logging and optional side effects (auto-review, audio saving).

### Helpers

- **`get_options()`** -- List available domains, agents, users, and task sets.
- **`get_tasks(task_set_name, task_split_name, task_ids, num_tasks)`** -- Load tasks with optional filtering.
- **`load_tasks(task_set_name, task_split_name)`** -- Load raw tasks from the registry.
- **`get_info(config, **overrides)`** -- Create run metadata (`Info` object).
- **`make_run_name(config)`** -- Generate a timestamped run name.

---

## Choosing the Right Level

| Scenario | Recommended Level |
|----------|------------------|
| Quick evaluation from the command line | Level 1 (CLI) |
| Running evals in a Python script | Level 2 (Config-based) |
| Custom agent development and testing | Level 3 (Instance-based) |
| Integrating into a larger pipeline | Level 3 (Instance-based) |
| Hyperparameter sweeps | Level 2 (Config-based) |
| Contributing a new domain/agent | Level 3 for testing, Level 2 for benchmarking |
