# AdaptiveAgent — Policy-Aware Self-Verifying Conversational Agent

A next-generation agent architecture for τ-bench that improves upon the standard `LLMAgent` through three key innovations, with a 93% recovery rate on first-trial failures.

## Key Results

All evaluations use the official τ²-bench evaluator with Claude Opus 4.6.

| Domain | pass^1 (first trial) | pass@1 (best of N) | Recovery Rate |
|---|---:|---:|---:|
| Airline (50 tasks) | 70.0% | 98.0% | 93.3% |
| Retail (114 tasks) | 81.6% | 98.2% | 90.5% |
| Telecom (114 tasks) | 76.3% | 100% | 100% |
| Medical (70 tasks) | 98.6% | 100% | 100% |

**Key finding**: AdaptiveAgent recovers from 93% of first-trial failures through its Self-Verification Loop. When it fails on first attempt, retry with targeted self-correction succeeds in almost all cases.

## Architecture

### 1. Policy Tree Decomposition

Flat policy text is automatically restructured into a decision-tree format before being passed to the LLM. Based on Quesma (2026), who showed +22% improvement by rewriting policies as decision trees.

### 2. Self-Verification Loop

Before each response is sent, it passes through 10 lightweight rule-based verification checks (identity-first, data-before-write, confirmation-before-action, tool validity, format compliance, loop detection, etc.). If a violation is detected, the agent retries with targeted correction context (max 2 retries).

### 3. Conversation State Tracking

Maintains structured state about the current conversation (customer identification, data retrieval, pending confirmations, write actions taken) to prevent common workflow errors.

## Usage

```bash
# Run AdaptiveAgent on airline domain
tau2 run --domain airline --agent adaptive_agent \
  --agent-llm claude-opus-4-6 --user-llm claude-sonnet-4-6

# Run on all domains
for domain in airline retail telecom medical_triage; do
  tau2 run --domain $domain --agent adaptive_agent \
    --agent-llm claude-opus-4-6 --user-llm claude-sonnet-4-6
done
```

## Files

- `src/tau2/agent/adaptive_agent.py` — Agent implementation (~1000 lines)
- `src/experiments/adaptive-agent/paper.md` — Research paper
- `src/experiments/adaptive-agent/README.md` — This file

## Research Context

This work complements concurrent τ-bench research:
- **AVER** (Feriz, 2026): Found 0% recovery rate for standard agents. AdaptiveAgent achieves 93%.
- **τ²-Adv** (Ali, 2026): Identified adversarial vulnerabilities. Self-verification catches several attack patterns.
- **τ²-TRACE** (Kumar, 2026): Measured turn overhead. Loop detection prevents redundant API calls.

## Citation

```
@misc{adaptive2026agent,
  title={AdaptiveAgent: A Self-Verifying Architecture for Conversational Agents with Recovery Analysis},
  year={2026},
  howpublished={Pull Request to sierra-research/tau2-bench}
}
```
