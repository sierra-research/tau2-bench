# Tau2 Domains

This folder contains the domains for the Tau2 project.

## Domain Structure

Each domain has its own folder with the following structure:

- `data_model.py`: Defines the data models for the domain. 
    - This implements the `DB` class, which is a base class for all domain databases.
- `user_data_model.py`: Defines the data models for the user data for the domain. (Optional)
    - This implements the `DB` class, which is a base class for all domain user databases.
- `tools.py`: Defines the tools for the domain. 
    - Implements `ToolKitBase` class, which is a base class for all domain toolkits. 
- `user_tools.py`: Defines the user tools for the domain. (Optional)
    - Implements `ToolKitBase` class, which is a base class for all domain toolkits. 
- `environment.py`: Defines the environment for the domain. 
    - Implements `get_environment()` functions that returns an `Environment` instance for the domain.
    - Implements `get_tasks()` functions that returns a list of tasks for the domain.
- `utils.py`: Defines the utility functions for the domain.

## Data Storage

All the data for the domain is stored in `data/tau2/domains/<domain_name>` folder.
Should contain:
- `tasks.json`: A JSON file containing the tasks for the domain.
- `policy.md`: A markdown file containing the policy for the domain.
- `db.json` or `db.toml`: A JSON or TOML file containing the database for the domain.
- `user_db.json` or `user_db.toml`: A JSON or TOML file containing the user database for the domain. (Optional)


## Tests
All the tests for the domain are stored in the `tests/domain_tests/<domain_name>` folder.
- `test_tools_<domain_name>.py`: Contains tests for the tools for the domain.
- `test_user_tools_<domain_name>.py`: Contains tests for the user tools for the domain (if any)


To run tests:
```sh
pytest tests/domain_tests/<domain_name>
```

## Registering your domain
To make it easy for people to use your domain, you need to register your `get_environment` function in Tau2 `registry.py` file.

In `registry.py`:
```python
from tau2.your_domain.environment import get_environment as your_domain_get_environment
...
registry.register_domain(your_domain_get_environment, "your_domain_name")
registry.register_tasks(your_domain_get_tasks, "your_domain_name")
```

## User Action Omission Testing

To test the robustness of agents when users claim to have performed actions without actually executing them, you can enable the optional user action omission feature. This simulates real-world situations in which users occasionally misreport their own behavior in some external environment due to confusion or other human errors.

### How It Works

The feature only affects WRITE-type user tools in telecom domains by default. When enabled, it replaces single user tool calls with natural language claims such as "I turned mobile data off". The environment state remains unchanged, no fabricated tool outputs are generated, and the feature is disabled by default. Running with user-action-omission enabled tests whether agents implicitly verify user actions and retry when needed.

### Configuration

Enable via the `--user-action-omission` CLI flag with JSON configuration:

```bash
tau2 run --domain telecom \
  --user-action-omission '{
    "enabled": true,
    "tool_name_filter": ["toggle_data", "toggle_wifi"],
    "p0": 0.1,
    "max_failures": 1
  }'
```

#### Available Settings

- **`enabled`** (boolean, required): Master switch to activate omission behavior
- **`p0`** (float, 0.0-1.0): Initial omission probability for first attempt to call a function (default: `0.05`). Probability decays on each subsequent attempt according to `p = p0 / (2 ** attempt_index)`.
- **`max_failures`** (integer ≥ 1): Maximum omission attempts per operation before stopping (default: `1`)
- **`tool_name_filter`** (list[str], optional): Only omit specific tools (default: all WRITE tools eligible)
- **`max_omitted_tools`** (integer, optional): Randomly sample N tools per simulation (default: all eligible)
- **`seed`** (integer, optional): RNG seed for reproducible omission patterns (default: uses simulation seed)