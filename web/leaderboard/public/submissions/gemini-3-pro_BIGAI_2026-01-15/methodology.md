## TongAgents Core Optimization Design

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Original LLMAgent                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   User/Tool Message ──► LLM Generate ──► Direct Output                      │
│                                                                             │
│                      (Single-pass generation, no validation)                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                        TongAgents LLMAgent                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   User/Tool Message ──► LLM Generate ──► Multi-Stage Verify ──► Output      │
│                                                │                            │
│                              ┌─────────────────┴─────────────────┐          │
│                              │                                   │          │
│                              ▼                                   ▼          │
│                        Structural                          Semantic         │
│                        Validation                          Validation       │
│                      (Tool Schema)                   (Policy Compliance)    │
│                              │                                   │          │
│                              └─────────────────┬─────────────────┘          │
│                                                │                            │
│                                                ▼                            │
│                                       Feedback-based                        │
│                                       Regeneration                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Multi-Stage Verification Framework

Introduces a **verification layer** between LLM output and final execution, upgrading the paradigm from "generate-and-execute" to "generate-verify-execute":

| Stage | Validation Type | Design Objective |
|:-----:|:---------------:|:-----------------|
| **L1** | Structural Validation | Ensures tool call parameters conform to Schema definitions, intercepting malformed requests |
| **L2** | Semantic Validation | Leverages LLM self-review to verify actions align with provided policy constraints |

**Core Value**: Reduces execution errors caused by malformed tool calls or policy misinterpretation, a common failure mode in agentic systems.

---

## 2. Verification-Driven Refinement

When validation fails, the agent attempts iterative refinement rather than immediate failure:

```
Validation Failed
    │
    └─► Feedback-based regeneration with error context injected into prompt
```

**Core Value**: Enables the agent to recover from validation errors through feedback loops, improving task completion rates without modifying the underlying policy logic.

---
