# tau2.runner

Simulation execution framework with a layered architecture for running evaluations at different levels of control.

## Module Structure

```
runner/
├── __init__.py        # Package exports
├── simulation.py      # Layer 1: run_simulation()
├── build.py           # Layer 2: build_* functions
├── batch.py           # Layer 3: run_domain(), run_tasks(), run_single_task()
├── helpers.py         # Task loading, run metadata, utilities
├── checkpoint.py      # Save/resume logic for batch runs
├── progress.py        # Retry logic and status monitoring
└── README.md          # This file
```

## Layers

### Layer 1: `simulation.py` -- Execute

Pure simulation execution and evaluation. Takes a fully constructed orchestrator, runs it, evaluates the result, and returns a `SimulationRun` with `reward_info` attached.

- **No registry dependency**: Everything is encapsulated in the orchestrator.
- **No config parsing**: No `RunConfig` needed.
- **No side effects**: No logging setup, no file saving, no auto-review.

```python
from tau2.runner import run_simulation

result = run_simulation(orchestrator)
```

### Layer 2: `build.py` -- Build

Turns names and configuration into live instances using the registry for name resolution.

**Low-level builders** (take individual parameters):
- `build_environment(domain)` -- Resolve domain name to environment instance.
- `build_agent(agent_name, environment)` -- Resolve agent name to agent instance.
- `build_user(user_name, environment, task)` -- Resolve user name to user instance.
- `build_voice_user(environment, task, audio_native_config)` -- Build voice user with all config wiring.

**High-level builders** (take `RunConfig`):
- `build_text_orchestrator(config, task)` -- Build a half-duplex `Orchestrator` from `TextRunConfig`.
- `build_voice_orchestrator(config, task)` -- Build a full-duplex `FullDuplexOrchestrator` from `VoiceRunConfig`.
- `build_orchestrator(config, task)` -- Dispatcher; delegates to the appropriate builder based on config type.

```python
from tau2.runner import build_text_orchestrator, run_simulation
from tau2 import TextRunConfig

config = TextRunConfig(domain="airline", agent="llm_agent", llm_agent="openai/gpt-4.1")
orchestrator = build_text_orchestrator(config, task, seed=42)
result = run_simulation(orchestrator)
```

### Layer 3: `batch.py` -- Batch

High-level batch execution with all operational concerns:

- **Concurrency**: Thread pool with configurable `max_concurrency`.
- **Checkpointing**: Atomic save/resume via `checkpoint.py`.
- **Retries**: Configurable retry with delay via `progress.py`.
- **Hallucination retries**: When `hallucination_retries > 0` (full-duplex only), re-runs simulations where the user simulator hallucinates, using feedback from `check_hallucination()` in `evaluator.reviewer`.
- **LiveKit pre-registration**: Calls `preregister_livekit_plugins()` on the main thread before spawning workers (required for LiveKit provider).
- **Status monitoring**: Periodic progress display (every 30s).
- **Side effects**: Auto-review, audio saving, per-task logging.

Entry points:
- `run_domain(config)` -- Full pipeline: load tasks, filter, run batch, display metrics.
- `run_tasks(config, tasks)` -- Run a list of tasks with all batch features.
- `run_single_task(config, task)` -- Run one task with logging and side effects.

```python
from tau2.runner import run_domain
from tau2 import TextRunConfig

config = TextRunConfig(domain="airline", agent="llm_agent", llm_agent="openai/gpt-4.1")
results = run_domain(config)
```

## Supporting Modules

- **`helpers.py`**: Task loading (`get_tasks`, `load_tasks`), run metadata (`get_info`, `make_run_name`), registry queries (`get_options`, `get_environment_info`).
- **`checkpoint.py`**: `try_resume()` for resuming from existing results, `create_checkpoint_saver()` for atomic saves.
- **`progress.py`**: `run_with_retry()` for retry logic, `StatusMonitor` for periodic progress display.

## Relationship to `tau2.run`

The `tau2.run` module is the original monolithic implementation. The `tau2.runner` package is its modular replacement. Both coexist:

- **`tau2.run`**: Still works for all existing callers (CLI, tests, scripts). Not deprecated yet.
- **`tau2.runner`**: The recommended API for new code. Cleaner separation of concerns.

Over time, `tau2.run` will be thinned out to re-export from `tau2.runner`.
