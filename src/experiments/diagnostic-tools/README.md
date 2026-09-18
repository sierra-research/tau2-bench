# Diagnostic Tools for τ²-bench

Three analysis tools for understanding and improving agent performance on τ²-bench.

## Tools

### 1. Failure Pattern Analyzer

Groups task failures by root cause pattern (identity not verified, missing confirmation, policy violation, etc.) and provides actionable recommendations.

```bash
python failure_pattern_analyzer.py results.json
python failure_pattern_analyzer.py results.json --output json
python failure_pattern_analyzer.py results.json --top 3
```

### 2. Difficulty-Graded Scoring

Categorizes tasks into Easy/Medium/Hard tiers based on number of action checks, then computes pass rate per tier. Reveals whether an agent struggles with complex multi-fault tasks.

```bash
python difficulty_graded_scoring.py --tasks tasks.json --results results.json
python difficulty_graded_scoring.py --tasks tasks.json --results results.json --domain airline
python difficulty_graded_scoring.py --tasks tasks.json --results results.json --output json
```

### 3. Cross-Domain Transfer Test

Compares agent performance across multiple domains to measure generalization. Computes a generalization score and identifies domain-specific weaknesses.

```bash
python cross_domain_transfer.py --results airline_results.json retail_results.json telecom_results.json
python cross_domain_transfer.py --results *.json --domains airline retail telecom --output json
```

## Requirements

- Python 3.10+
- No external dependencies (stdlib only)

## Input Format

All tools expect JSON files following the τ²-bench result format:

```json
[
  {
    "task_id": "task_1",
    "reward": 1.0,
    "action_checks": [
      {"description": "check description", "passed": true}
    ]
  }
]
```
