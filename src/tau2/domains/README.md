# Tau2 Domains

This folder contains the domains for the Tau2 project.

## Domain Structure

Each domain has its own folder with the following structure:

- `data_model.py`: Defines the data models for the domain. 
    - Implements a subclass of `DB` (e.g. `FlightDB(DB)`) which is the base class for all domain databases (defined in `tau2/environment/db.py`).
- `user_data_model.py`: Defines the data models for the user data for the domain. (Optional)
    - Implements a subclass of `DB` for the domain's user database.
- `tools.py`: Defines the tools for the domain. 
    - Implements a subclass of `ToolKitBase` (e.g. `AirlineTools(ToolKitBase)`) which is the base class for all domain toolkits (defined in `tau2/environment/toolkit.py`). 
- `user_tools.py`: Defines the user tools for the domain. (Optional)
    - Implements a subclass of `ToolKitBase` for user-facing tools. 
- `environment.py`: Defines the environment for the domain. 
    - Implements `get_environment()` function that returns an `Environment` instance for the domain.
    - Implements `get_tasks()` function that returns a list of tasks for the domain.
    - Implements `get_tasks_split()` function that returns a dictionary mapping split names to lists of task IDs.
- `utils.py`: Defines the utility functions for the domain (e.g. data paths).

## Data Storage

All the data for the domain is stored in `data/tau2/domains/<domain_name>` folder.
Should contain:
- `tasks.json`: A JSON file containing the tasks for the domain.
- `tasks_voice.json`: A JSON file containing the voice tasks for the domain.
- `split_tasks.json`: A JSON file containing the task splits for the domain.
    - The split file naming follows the pattern `split_<tasks_file_stem>.json` (e.g. `split_tasks.json` for `tasks.json`).
    - The task splits are defined as a dictionary mapping split names to lists of task IDs.
    - The task IDs are defined in the `tasks.json` file.
    - It must at minimum implement a task split called `base`. This will be the default task split.
- `policy.md`: A markdown file containing the policy for the domain.
- `db.json` or `db.toml`: A JSON or TOML file containing the database for the domain.
- `user_db.json` or `user_db.toml`: A JSON or TOML file containing the user database for the domain. (Optional)

## Task Schema and Evaluation

Each entry in `tasks.json` is a `Task` (see
[`src/tau2/data_model/tasks.py`](../data_model/tasks.py)) with an
`evaluation_criteria` block. Note: `evaluation_criteria.actions` is
*one* reference trajectory used to derive the target DB end state, not
a per-call requirement on the agent (unless `RewardType.ACTION` is in
`reward_basis`, which the standard airline / retail / telecom tasks do
not use). See [`docs/evaluation.md`](../../../docs/evaluation.md) for
the full breakdown.


## Tests
All the tests for the domain are stored in the `tests/test_domains/test_<domain_name>` folder.
- `test_tools_<domain_name>.py`: Contains tests for the tools for the domain.
- `test_user_tools_<domain_name>.py`: Contains tests for the user tools for the domain (if any)


To run tests:
```sh
pytest tests/test_domains/test_<domain_name>
```

## Available Domains

| Domain | Description |
|--------|-------------|
| `mock` | Lightweight test domain for development |
| `airline` | Flight booking, cancellation, and customer support |
| `retail` | Order management, returns, and product inquiries |
| `telecom` | Telecom account management and troubleshooting |
| `banking_knowledge` | Knowledge-retrieval-based banking customer service with configurable RAG pipelines |
| `intake` | Outbound elicitation with an agent-directed default and a scaffolded strategy ablation |

The `intake` domain and task set use `src/tau2/domains/intake/`,
`data/tau2/domains/intake/`, and `tests/test_domains/test_intake/`. Agent-directed
voice elicitation is the default. Select `--elicitation-strategy scaffolded` for
the scaffolded ablation, which also supports text. Both strategies use the same
frozen task collection with strategy-specific policies and tool schemas.
See the [elicitation guide](../../../docs/elicitation.md) and
`tau2 intake-tasks --help` for usage and reproduction.

### `banking_knowledge` Domain

The `banking_knowledge` domain differs from standard domains in several ways:

- **Configurable retrieval**: Uses `--retrieval-config` to select how the agent accesses the knowledge base (e.g., `qwen_embeddings`, `grep_only`, `terminal_use`). Different configs give the agent different tools and system prompts.
- **Dual data sources**: A `TransactionalDB` for user/account data and a `KnowledgeBase` of 700+ documents for retrieval.
- **Extended data directory**: Contains `documents/`, `prompts/` (per-variant policy templates), and `tasks/` subdirectories.
- **Retrieval pipeline**: Uses the `src/tau2/knowledge/` module for embeddings, BM25, grep, reranking, and sandboxed shell access. See `src/tau2/knowledge/README.md` for config details.

## Registering your domain
To make it easy for people to use your domain, you need to register your `get_environment`, `get_tasks`, and `get_tasks_split` functions in Tau2 `registry.py` file.

In `registry.py`:
```python
from tau2.domains.your_domain.environment import get_environment as your_domain_get_environment
from tau2.domains.your_domain.environment import get_tasks as your_domain_get_tasks
from tau2.domains.your_domain.environment import get_tasks_split as your_domain_get_tasks_split
...
registry.register_domain(your_domain_get_environment, "your_domain_name")
registry.register_tasks(your_domain_get_tasks, "your_domain_name", get_task_splits=your_domain_get_tasks_split)
```

## Domains in another package

A package outside this repo can register its domains, agents and users without
editing `registry.py`, through the `tau2.plugins` entry point group. Each entry
point names a function that takes the registry:

```toml
# your package's pyproject.toml
[project.entry-points."tau2.plugins"]
your_domain = "your_package.plugin:register"
```

```python
from tau2.registry import DomainOptions

def register(registry):
    registry.register_domain(
        get_environment,
        "your_domain",
        DomainOptions(env_kwargs=lambda task: {"task": task}),
    )
    registry.register_tasks(get_tasks, "your_domain", get_task_splits=get_tasks_split)
```

`DomainOptions` covers domains that need more than the default wiring:

- `build_orchestrator`: used by the batch runner in place of
  `tau2.runner.build.build_orchestrator` (same signature), for domains that
  build their own environment or user simulator.
- `env_kwargs`: `(task) -> kwargs`, extra environment kwargs used when
  evaluating a run and when re-evaluating a saved one.
- `evaluation_env_kwargs`: `(simulation) -> kwargs`, extra environment kwargs
  when re-evaluating a saved run.

Environments can also override `get_evaluation_info()` (diagnostics saved with
the reward) and `get_run_record()` (merged into the simulation's `info`), and
tool results can carry `raw_data` that is saved but never sent to a model.
