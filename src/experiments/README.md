# Experiments

This directory contains **experimental code** that is offered as-is and should be treated as experimental components, not part of the core tau2 benchmark.

> ⚠️ **Important**: The code in this directory is experimental and may not be fully tested or supported. Use at your own discretion.

## Overview

The `experiments/` folder is used for experimental features and research code that extends beyond the core tau2 benchmark. It can be used for new features, prototypes, and innovative approaches that are not part of the core evaluation framework. These components are provided for research purposes and to enable advanced use cases.

## Directory Structure

This directory is organized into subdirectories for different types of experimental components. Each subdirectory should contain its own README with specific documentation and usage instructions.

```
experiments/
├── domains/              # Community-contributed domains
├── hyperparam/           # Hyperparameter tuning experiments
└── agentify_tau_bench/   # Agent framework experiments
```

### Community-Contributed Domains (`domains/`)

The `domains/` subdirectory contains community-contributed tau2 domains that are not part of the core benchmark but follow tau2 domain conventions. These domains:

- **Are self-contained**: Include their own data, tests, and documentation
- **Follow tau2 conventions**: Implement `get_environment()`, `get_tasks()`, and proper tool structure
- **Are auto-discovered**: Registered automatically with an `experimental:` prefix
- **May include original code**: Can bundle the contributor's original code for reference

#### Structure for Contributed Domains

```
domains/<domain_name>/
├── original/            # Original contributor's code (optional, for reference)
├── domain/              # Clean tau2-compatible implementation
│   ├── __init__.py
│   ├── data_model.py    # Pydantic DB model
│   ├── tools.py         # ToolKit implementation
│   ├── environment.py   # get_environment() and get_tasks()
│   └── tests/           # Unit tests
├── data/                # Domain data (db.json, policy.md, tasks.json)
└── README.md            # Documentation and attribution
```

#### Using a Contributed Domain

**Via CLI** (auto-discovered with `experimental:` prefix):
```bash
tau2 run --domain "experimental:<domain_name>" --agent-llm gpt-4o-mini --user-llm gpt-4o-mini
```

**Via Python** (direct import):
```python
from experiments.domains.<domain_name>.domain import (
    get_environment,
    get_tasks,
)

env = get_environment()
tasks = get_tasks()
```

**Via Registry**:
```python
from tau2.registry import registry

env_constructor = registry.get_env_constructor("experimental:<domain_name>")
env = env_constructor()
```

## Quick Start

To contribute experimental code:

1. Create a new subdirectory for your experiment
2. Add a comprehensive README.md explaining the purpose and usage
3. Include example scripts and basic tests
4. Follow the development guidelines below 

## Development Guidelines

When working with experimental code:

1. **Backward Compatibility**: Maintain compatibility with core tau2 interfaces when possible
2. **Documentation**: Each experimental component should have its own README
3. **Testing**: Include basic testing scripts and examples
4. **Dependencies**: Manage dependencies carefully to avoid conflicts with core tau2
5. **Isolation**: Keep experimental code self-contained within this directory

## Contributing

Experimental contributions are welcome! Please:

1. Add comprehensive documentation in your subfolder's README
2. Include example usage and test scripts
3. Mark any breaking changes or dependencies clearly
4. Consider the experimental nature - code doesn't need to be production-ready

## Support

Since this is experimental code:

- **No guarantees** of stability or continued support
- **Community-driven** - contributions and improvements welcome
- **Use at your own risk** - test thoroughly before production use
- **Documentation-first** - refer to individual README files for detailed usage

For core tau2 benchmark support, see the main project documentation.
