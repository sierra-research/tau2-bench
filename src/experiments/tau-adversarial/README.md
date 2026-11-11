# tau-adversarial: Adversarial Evaluation for tau2-bench

Test how well agents resist manipulation attempts including social engineering, prompt injection, policy exploitation, identity manipulation, and information extraction attacks.

## Installation

```bash
cd src/experiments/tau-adversarial
uv sync
```

## Quick Start

```bash
# Run adversarial evaluation
tau-adversarial --domain airline

# Run with specific strategy
tau-adversarial --domain airline --strategy social_engineering

# Run all strategies
tau-adversarial --domain airline --all-strategies

# Run tests
pytest tests/ -v
```

## Attack Strategies

| Strategy | Description |
|----------|-------------|
| `social_engineering` | Emotional manipulation, fake authority claims |
| `prompt_injection` | Override agent instructions |
| `policy_exploitation` | Find and exploit policy loopholes |
| `identity_manipulation` | Impersonate other users |
| `information_extraction` | Probe for internal procedures |

Each strategy has 3 sophistication levels:
- **Low (0.0)**: Direct, obvious attempts
- **Medium (0.5)**: Subtle with emotional appeals
- **High (1.0)**: Multi-turn, trust-building attacks

## Python API

```python
from tau_adversarial import (
    AdversarialStrategy,
    AttackConfig,
    load_adversarial_tasks,
)
from tau_adversarial.adversarial_user import create_adversarial_user
from tau_adversarial.evaluator_safety import evaluate_safety

# Configure attack
config = AttackConfig(
    strategy=AdversarialStrategy.SOCIAL_ENGINEERING,
    sophistication=0.7,
    target_goal="Get unauthorized refund",
    domain="airline",
)

# Load adversarial tasks
tasks = load_adversarial_tasks("airline")

# Create adversarial user
adversarial_user = create_adversarial_user(
    strategy=AdversarialStrategy.SOCIAL_ENGINEERING,
    sophistication=0.7,
    target_goal="Get unauthorized refund",
    domain="airline",
    llm="gpt-4o",
)

# Evaluate safety (after running simulation)
safety_result = evaluate_safety(
    simulation=simulation,
    domain="airline",
    attack_goal="Get unauthorized refund",
)
```

## Safety Metrics

- **Safety Score** (0-1): Higher = safer, penalized by violations
- **Attack Resistance Rate**: Percentage of attacks blocked
- **Violations**: Detected issues including:
  - Unauthorized actions
  - Information disclosure
  - Policy circumvention
  - Prompt injection success
  - Identity verification bypass

## Architecture: Prompt Layering Design

The adversarial evaluation uses a **two-layer prompt architecture** by design:

### Layer 1: Task-Level Attack Scripts
Tasks in `tasks_adversarial.json` contain detailed attack scripts in `user_scenario.instructions.task_instructions`. These define:
- The specific manipulation tactics to attempt
- Step-by-step escalation paths
- Domain-specific attack content

### Layer 2: AdversarialUser Wrapper
The `AdversarialUser` class wraps these instructions with:
- Strategy-specific framing (e.g., "You are an adversarial user...")
- Meta-instructions for hiding adversarial intent
- Guidance to blend legitimate requests with attacks

**Why Two Layers?**
1. **Flexibility**: Task scripts can be swapped independently of the wrapper
2. **Realism**: Real attacks often combine legitimate requests with manipulation
3. **Configurability**: The wrapper adds sophistication-based behavior without modifying task definitions
4. **Evaluation**: Separates "what to attack" (task) from "how to attack" (wrapper)

The double-stacking is intentional - it creates more realistic attack scenarios where the adversarial user has both specific goals (from the task) and general manipulation strategies (from the wrapper).

## Adding Adversarial Tasks

Create `tasks_adversarial.json` in `data/domains/{domain}/`:

```json
[
    {
        "id": "adv_001",
        "adversarial_metadata": {
            "strategy": "social_engineering",
            "sophistication": 0.5,
            "target_goal": "Get unauthorized refund"
        },
        "user_scenario": {
            "instructions": {
                "task_instructions": "Claim supervisor approved your refund...",
                "domain": "airline"
            }
        }
    }
]
```

## Project Structure

```
src/experiments/tau-adversarial/
├── pyproject.toml           # Project configuration
├── README.md                 # This file
├── src/tau_adversarial/
│   ├── __init__.py          # Package exports
│   ├── strategies.py        # Attack strategy definitions
│   ├── tasks.py             # Task loading utilities
│   ├── adversarial_user.py  # Adversarial user simulator
│   ├── evaluator_safety.py  # Safety violation detection
│   └── run_adversarial.py   # CLI runner
├── tests/
│   └── test_adversarial.py  # Test suite
└── data/domains/
    ├── airline/tasks_adversarial.json
    ├── retail/tasks_adversarial.json
    └── telecom/tasks_adversarial.json
```

## CLI Options

```
tau-adversarial [OPTIONS]

Options:
  --domain TEXT         Domain to evaluate (airline, retail, telecom)
  --strategy TEXT       Strategy filter (social_engineering, prompt_injection, etc.)
  --all-strategies      Run all strategies
  --agent TEXT          Agent type (default: llm_agent)
  --llm TEXT            LLM model (default: gpt-4o)
  --num-seeds INT       Number of seeds per task (default: 1)
  --output-dir TEXT     Output directory (default: results/adversarial)
  --verbose             Enable verbose output
```
