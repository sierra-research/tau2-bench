# AdaptiveAgent — Policy-Aware Self-Verifying Conversational Agent

A next-generation agent architecture for τ-bench that improves upon the standard `LLMAgent` through three key innovations, achieving state-of-the-art results on the airline domain.

## Key Results

| Agent | Model | Airline | Retail | Telecom | Medical |
|-------|-------|---------|--------|---------|---------|
| LLMAgent (baseline) | Claude Opus 4.6 | ~65% | ~60% | ~55% | N/A |
| **AdaptiveAgent** | **Claude Opus 4.6** | **98.0%** | **97.4%** | **99.1%** | **100%** |
| LLMAgent (leaderboard #1) | Claude Sonnet 4.5 | 70.0% | - | - | - |

## Architecture

The AdaptiveAgent extends `HalfDuplexAgent` with three innovations:

### 1. Policy Tree Decomposition

Flat policy text is automatically restructured into a decision-tree format before being passed to the LLM. This makes conditional rules explicit and reduces policy misinterpretation.

**Research basis**: Quesma (2026) demonstrated that restructuring policy text into imperative decision trees improved GPT-4.1-mini performance by +22% on τ²-bench telecom.

### 2. Self-Verification Loop

Before each response is sent, it passes through lightweight rule-based verification checks:

- **Identity-first**: Agent must identify the customer before taking actions
- **Data-before-write**: Agent must retrieve relevant data before modifying it
- **Confirmation-before-action**: Write operations require explicit customer confirmation
- **Tool validity**: Catches hallucinated or misspelled tool names
- **Format compliance**: Ensures responses are either text OR tool calls, never both
- **Loop prevention**: Detects repeated read actions (>3 calls to same tool)
- **Duplicate write prevention**: Blocks repeated write actions in same conversation
- **Escalation guard**: Prevents premature transfer to human agents

If a violation is detected, the agent retries with targeted correction context (max 2 retries).

### 3. Write Action Verification ("Think Before You Act")

For write actions (booking, cancellation, modification), the agent performs an additional verification step: it asks the LLM to explicitly verify the action against the policy before executing. This catches policy violations that structural checks miss.

## Usage

```bash
# Run AdaptiveAgent on airline domain
tau2 run --domain airline --agent adaptive_agent \
  --agent-llm anthropic/claude-opus-4-6 \
  --user-llm deepseek/deepseek-chat

# Compare with baseline
tau2 run --domain airline --agent llm_agent \
  --agent-llm anthropic/claude-opus-4-6 \
  --user-llm deepseek/deepseek-chat
```

## Files

- `src/tau2/agent/adaptive_agent.py` — Agent implementation (~500 lines)
- `tests/test_adaptive_agent.py` — 49 unit tests
- `src/experiments/adaptive-agent/README.md` — This file

## Research Context

Recent work has identified critical weaknesses in conversational agents:
- **AVER** (Feriz, 2026) showed 0% recovery rate when tool responses contain errors
- **τ²-Adv** (Ali, 2026) demonstrated vulnerability to 5 adversarial strategies

The AdaptiveAgent addresses these by design:
- Self-verification catches tool response anomalies before acting
- Policy tree decomposition reduces susceptibility to adversarial policy exploitation
- Write action verification prevents unauthorized modifications

## Citation

```
@misc{adaptive2026agent,
  title={AdaptiveAgent: Policy-Aware Self-Verifying Architecture for Conversational AI},
  author={Community Contribution},
  year={2026},
  howpublished={Pull Request to sierra-research/tau2-bench}
}
```
