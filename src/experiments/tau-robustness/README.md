# AVER: Agent Verification & Error Recovery

Robustness evaluation mode for τ²-bench — chaos engineering for conversational agents.

## What it does

Injects controlled errors into tool responses during simulation, then measures whether agents detect, diagnose, and recover from the corrupted data.

## Install

```bash
cd src/experiments/tau-robustness
uv pip install -e .
```

## Usage

```bash
# Via tau2 CLI (requires thin CLI hook)
tau2 run --domain retail --mode robustness --injection-rate 1.0 --num-tasks 5

# Standalone
python -m tau_robustness.run -d retail --num-tasks 5 --num-trials 3
```

## Injection Definitions

34 domain-aware injection definitions across 3 domains:

| Domain | Blocking | Cosmetic | Total | Enforcement |
|--------|----------|----------|-------|-------------|
| Retail | 8 | 3 | 11 | API-enforced |
| Airline | 8 | 2 | 10 | Policy-enforced |
| Telecom | 8 | 5 | 13 | Mixed |

## AVER Score

Three-dimension scoring: Detection (40%) + Diagnosis (20%) + Recovery (40%) = AVER Score (0-100)

## Tests

```bash
pytest tests/ -v
```
