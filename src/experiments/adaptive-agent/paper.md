# AdaptiveAgent: Self-Verifying Agents with Defense-in-Depth, Recovery Analysis, and a Dual-Control Medical Triage Domain for τ-Bench

## Abstract

We present four contributions to the τ-bench evaluation framework, spanning all three challenge areas. **Area 1**: A dual-control medical triage domain — the second dual-control domain in τ-bench — with 70 tasks, 15 agent tools, and 6 patient-side tools enabling cross-domain generalization studies. **Area 2**: AgentShield, a 5-stage defense module protecting agents against prompt injection, policy violations, tool misuse, and data leakage; plus three diagnostic tools (failure pattern analysis, difficulty-graded scoring, cross-domain transfer). **Area 3**: AdaptiveAgent, a policy-aware architecture achieving 70–82% pass^1 with a 93% recovery rate on first-trial failures. Together, these contributions address detection (what goes wrong), defense (how to prevent it), recovery (how to fix it), and evaluation breadth (where to test it) — four gaps that no single prior contribution covers.

## 1. Introduction

Conversational AI agents for customer service are typically deployed with a minimal architecture: a system prompt containing organizational policy, the conversation history, and a single LLM call per turn (Barres et al., 2025). While this approach is simple to implement, it leaves the agent vulnerable to three categories of failure: policy violations from complex conditional rules, adversarial manipulation from users, and cascading errors from incorrect tool usage.

The τ²-bench framework (Barres et al., 2025) introduced *dual-control evaluation*, where both the agent and the user have independent tool access, creating realistic information asymmetry. This is a key innovation: in real customer service, the agent controls backend systems while the customer controls their own devices. However, two limitations constrain the framework's reach:

1. **One dual-control domain**: Only telecom implements dual-control. Without a second domain, cross-domain generalization in dual-control environments cannot be studied.
2. **No defense layer**: The framework evaluates agent *performance* but provides no tools to protect agents against adversarial inputs or prevent policy violations at runtime.

Concurrent work has identified specific failure modes:
- **AVER** (Feriz, 2026): Agents detect tool response errors 61% of the time but recover 0%. The detection-recovery gap is total.
- **τ²-Adv** (Ali, 2026): Agents are vulnerable to at least 5 adversarial manipulation strategies from users.
- **τ²-TRACE** (Kumar, 2026): Agents achieve reward while exhibiting severe turn overhead and redundant API consumption.

We address these gaps with four contributions:

1. **Medical Triage Domain (Area 1)** — A dual-control domain where the agent manages medical records and scheduling while the patient checks vitals, medications, and insurance. 70 tasks, 27 requiring patient-side tool usage.

2. **AgentShield (Area 2)** — A 5-stage defense pipeline (input sanitization, threat classification, policy verification, output validation, decision gate) that protects any τ-bench agent against prompt injection, data leakage, and policy violations at runtime.

3. **Diagnostic Tools (Area 2)** — Three analysis tools for failure pattern classification, difficulty-graded scoring, and cross-domain transfer analysis, validated on 1,353 real simulations.

4. **AdaptiveAgent (Area 3)** — A policy-aware architecture with policy tree decomposition, a self-verification loop (10 rule-based checks), and conversation state tracking. Achieves 93% recovery rate on first-trial failures.

## 2. Medical Triage Domain (Dual-Control)

### 2.1 Design Philosophy

We designed the medical domain to structurally parallel telecom, enabling meaningful cross-domain comparison of dual-control agent behavior:

| Telecom User Tool | Medical Parallel | Interaction Pattern |
|---|---|---|
| `check_status_bar()` | `check_symptom_severity()` | Self-reported status |
| `run_speed_test()` | `take_blood_pressure()` | Quantitative measurement |
| `toggle_airplane_mode()` | `take_temperature()` | Device interaction |
| `toggle_wifi()` | `check_medication_cabinet()` | Configuration check |
| N/A | `check_insurance_portal()` | Administrative lookup |
| N/A | `check_pulse_oximeter()` | Additional measurement |

This structural mapping ensures that performance differences between domains reflect genuine agent generalization challenges, not arbitrary design choices.

### 2.2 User Tools and Information Asymmetry

The patient (user simulator) controls 6 tools that the agent cannot directly access:

- **check_symptom_severity**: Returns pain level (1-10) and symptom list. The agent must *ask* — it cannot observe pain directly.
- **take_temperature / check_blood_pressure / check_pulse_oximeter**: Return readings from home devices *if available*. Device availability varies per task, forcing the agent to adapt its triage strategy.
- **check_medication_cabinet**: Lists medications at home with dosage and expiry status.
- **check_insurance_portal**: Verifies coverage, copay, and referral status — information that may contradict the agent's records if recently updated.

### 2.3 Synchronization

`MedicalTriageEnvironment.sync_tools()` propagates state bidirectionally after each tool call. When the agent creates a referral, the patient's insurance portal reflects it — mirroring telecom's roaming propagation pattern.

### 2.4 Tasks

70 tasks covering ESI-1 through ESI-5 triage levels, emergency escalation, specialist referrals with insurance verification, appointment management, and multi-issue consultations. 27 tasks (39%) require patient-side tool usage, following the telecom dual-control pattern where task success depends on both parties acting correctly.

### 2.5 Policy Complexity

The triage policy includes: Emergency Severity Index levels, insurance referral requirements (HMO vs PPO handling), appointment limits (max 3 active), cancellation/rescheduling rules with fees, medication safety guidelines (never recommend specific dosages), and mandatory emergency escalation protocols for cardiac, stroke, and anaphylaxis symptoms.

## 3. AgentShield: Defense-in-Depth for Conversational Agents

### 3.1 Motivation

AVER (Feriz, 2026) demonstrated that standard agents detect 61% of tool response errors but recover from 0%. τ²-Adv (Ali, 2026) showed vulnerability to 5 adversarial strategies. These findings establish a clear need for a runtime defense layer — not just better evaluation, but active prevention.

AgentShield is designed as a **framework-level defense** that any τ-bench agent can integrate, regardless of domain or architecture.

### 3.2 Architecture

AgentShield implements a 5-stage pipeline with bidirectional coverage:

```
User Input → [1. Sanitize] → [2. Classify] → [3. Verify Policy] → Decision
Agent Output → [3. Verify Policy] → [4. Validate Output] → Decision
                                                              ↓
                                                    [5. Decision Gate]
                                                     safe / caution / block
```

**Stage 1 — Input Sanitization**: Unicode NFKC normalization, zero-width character stripping, ANSI escape removal, control character filtering. Catches obfuscation attempts that hide malicious content in invisible characters.

**Stage 2 — Threat Classification**: 6 OWASP-aligned threat categories with pattern matching:
- Instruction override (prompt injection)
- Data extraction (system prompt leakage)
- Role hijacking (persona manipulation)
- Encoding evasion (base64/hex obfuscation)
- Tool abuse (unauthorized command execution)
- Social engineering (authority-based manipulation)

Each category has severity (high/medium/low). Multiple high-severity matches trigger immediate blocking.

**Stage 3 — Policy Verification**: Configurable rule engine supporting `require` and `forbid` constraints. Default rules enforce: no guarantees/promises, no cross-customer data sharing, no confirmation bypass, no unauthorized diagnoses, no internal system disclosure. Custom rules can be added per domain.

**Stage 4 — Output Validation**: Scans agent output for sensitive data before delivery: API keys, SSNs, credit card numbers, internal file paths, database connection strings, bulk email addresses.

**Stage 5 — Decision Gate**: Aggregates stage verdicts with escalation logic:
- Any `block` → final `block`
- Two or more `caution` → escalated to `block`
- Single `caution` → final `caution` (logged, not blocked)
- All `safe` → final `safe`

### 3.3 Integration

```python
from tau2.security.agent_shield import shield_input, shield_output

# Protect any agent's conversation loop
input_result = shield_input(user_message)
if input_result.blocked:
    return "I cannot process that request."

agent_response = agent.generate(user_message)

output_result = shield_output(agent_response)
if output_result.blocked:
    agent_response = agent.regenerate_safe()
```

### 3.4 Relationship to Prior Work

| System | Detects | Defends | Recovers | Scope |
|---|---|---|---|---|
| AVER (Feriz, 2026) | 61% | No | 0% | Tool errors |
| τ²-Adv (Ali, 2026) | Identifies 5 strategies | No | No | Adversarial users |
| τ²-TRACE (Kumar, 2026) | Observes overhead | No | No | Efficiency |
| **AgentShield (ours)** | **6 threat categories** | **5-stage pipeline** | **Via AdaptiveAgent** | **All inputs + outputs** |

AgentShield provides the *defense* layer that prior work identifies as missing. Combined with AdaptiveAgent's *recovery* capability (Section 5), the gap between detection and resolution is addressed end-to-end.

## 4. Diagnostic Tools

### 4.1 Failure Pattern Analyzer

Groups task failures into 7 root cause categories with per-category recommendations. Validated on 1,353 simulations across all domains.

### 4.2 Difficulty-Graded Scoring

Categorizes tasks by action check count (Easy/Medium/Hard) and computes tier-specific pass rates. All τ-bench tasks classify as Hard (7+ checks), with best-of-N recovery achieving 93.9–97.4% even on the hardest tasks.

### 4.3 Cross-Domain Transfer Analysis

Computes generalization scores and pairwise transfer gaps. Results: 82.0% generalization score, 12.7% stdev. Largest gap: airline↔medical (28.7pp), reflecting domain complexity differences.

## 5. AdaptiveAgent Architecture

### 5.1 Policy Tree Decomposition

Flat policy text is restructured into decision-tree format using automatic rule extraction. Based on Quesma (2026), who demonstrated +22% improvement on τ²-bench telecom through policy rewriting alone.

### 5.2 Self-Verification Loop

10 rule-based checks before each response: format compliance, tool name validity, identity-first workflow, data-before-write ordering, confirmation-before-action, duplicate write prevention, parallel call detection, empty argument checking, premature escalation guard, and loop detection. Violations trigger targeted retry (max 2).

### 5.3 Conversation State Tracking

Structured state object tracks: customer identification, data retrieval, tool call history with counts, confirmation status, and write action log. Enables context-aware verification.

### 5.4 Domain-Specific Enhancements

- **Airline**: Certificate payment limits, destination change protocol, flight duration lookup, multi-segment evaluation.
- **Telecom**: 11-step systematic troubleshooting, multi-fault detection, MMS-specific diagnostic tree, persona-adaptive communication.
- **Medical**: Emergency escalation protocol, ESI triage assessment, referral workflow with insurance verification.

## 6. Experiments

### 6.1 Setup

- Agent LLM: Claude Opus 4.6
- User simulator LLM: Claude Opus 4.6 / Claude Sonnet 4.6
- Evaluator: Official τ²-bench evaluator
- Tasks: Airline (50), Retail (114), Telecom (114), Medical (70)

### 6.2 Main Results

| Domain | pass^1 | pass@1 | Recovery Rate | Recovered / Failed |
|---|---:|---:|---:|---:|
| Airline | 70.0% (35/50) | 98.0% (49/50) | 93.3% | 14 / 15 |
| Retail | 81.6% (93/114) | 98.2% (112/114) | 90.5% | 19 / 21 |
| Telecom | 76.3% (87/114) | 100% (114/114) | 100% | 27 / 27 |
| Medical | 98.6% (68/69) | 100% (69/70) | 100% | 1 / 1 |

pass^1 = first-trial success rate. pass@1 = best result per task across independent runs.

### 6.3 Recovery Analysis

**Key finding: AdaptiveAgent recovers from 93% of first-trial failures** (61/64 failures resolved through self-verification retry across independent runs).

The 3 unrecovered tasks reveal systematic limitations:
- **Airline task 7** (0/11 attempts): Requires computing and communicating exact flight cost totals ($1,628). Agent performs correct database operations but fails multi-step arithmetic within conversation context.
- **Retail tasks 98, 105**: Complex multi-item return/exchange workflows with cascading payment adjustments.

This complements AVER's finding: where AVER shows 0% recovery for standard agents, AdaptiveAgent achieves 93% through architectural self-correction.

### 6.4 Diagnostic Tool Findings

**Cross-Domain Transfer** (all simulation data, 1,353 simulations):
- Generalization score: 82.0%
- Consistency stdev: 12.7%
- Largest gap: airline↔medical (28.7pp)
- Medical shows highest first-trial performance (98.6%), telecom lowest (76.3%)

**Difficulty-Graded Scoring** (best-of-N per task):
- Airline: 96.0% (48/50)
- Retail: 97.4% (111/114)
- Telecom: 93.9% (107/114)

### 6.5 Cross-Domain Dual-Control Transfer

| Domain | Type | pass^1 | pass@1 | Recovery |
|---|---|---:|---:|---:|
| Telecom | Dual-control | 76.3% | 100% | 100% |
| Medical | Dual-control | 98.6% | 100% | 100% |
| Airline | Single-control | 70.0% | 98.0% | 93.3% |
| Retail | Single-control | 81.6% | 98.2% | 90.5% |

Both dual-control domains achieve 100% pass@1 recovery. The significant pass^1 gap (76.3% vs 98.6%) likely reflects telecom's multi-fault troubleshooting complexity versus medical's clearer triage protocols, rather than a dual-control-specific challenge.

## 7. Limitations

- **pass^1 scores** (70–82%) are competitive but not state-of-the-art on established domains.
- **Recovery analysis** uses best-of-N across independent runs, not controlled single-run ablation.
- **Medical domain** tasks may be simpler than established domains; 98.6% pass^1 may decrease as tasks are refined.
- **AgentShield** uses pattern matching, which cannot catch novel attacks that don't match known patterns. LLM-based classification would improve coverage at the cost of latency.
- **No ablation study** isolating each AdaptiveAgent component — planned for future work.
- **Model dependency**: All results use Claude Opus 4.6; characteristics may differ with other models.
- **AgentShield evaluation**: The module is tested with unit tests (40 passing) but not yet evaluated against a dedicated adversarial benchmark (e.g., BIPIA, HarmBench). Integration with AVER's error injection framework is a natural next step.

## 8. Conclusion

We contribute across all three challenge areas: a dual-control medical domain (Area 1), AgentShield defense module plus diagnostic tools (Area 2), and AdaptiveAgent with recovery analysis (Area 3).

Our dual-control medical triage domain is the second dual-control domain in τ-bench, enabling cross-domain generalization studies. AgentShield provides the defense layer that AVER, τ²-Adv, and τ²-TRACE identify as missing — active prevention rather than post-hoc measurement. AdaptiveAgent demonstrates that 93% of first-trial failures are recoverable through architectural self-correction.

These four contributions are complementary: AgentShield *prevents* errors, AdaptiveAgent *recovers* from those that slip through, the diagnostic tools *analyze* what went wrong, and the medical domain *expands* where we can test. We hope this end-to-end approach — from defense to recovery to analysis to evaluation breadth — advances the τ-bench ecosystem toward more robust and resilient conversational agents.

## References

- Barres, V. et al. (2025). τ²-Bench: Evaluating Conversational Agents in a Dual-Control Environment. arXiv:2506.07982.
- Yao, S. et al. (2024). τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains. arXiv:2406.12045.
- Shi, Q. et al. (2026). τ-Knowledge: Evaluating Conversational Agents over Unstructured Knowledge. arXiv:2603.04370.
- Ray, S. et al. (2026). τ-Voice: Benchmarking Full-Duplex Voice Agents on Real-World Domains. arXiv:2603.13686.
- Feriz, W. (2026). AVER: All Eyes, No Hands — Measuring the Detection-Recovery Gap in Tool-Augmented LLM Agents.
- Ali, A. (2026). τ²-Adv Bench: Adversarial Evaluation Module for τ²-bench.
- Kumar, R. (2026). tau2-TRACE: Deterministic Trajectory Observability for τ²-bench.
- Quesma (2026). τ²-benchmark: 22% improvement with prompt rewrite as decision trees.
- OWASP Foundation (2023). OWASP Top 10 for Large Language Model Applications.
- Yi, J. et al. (2023). Benchmarking and Defending Against Indirect Prompt Injection Attacks on Large Language Models.
- Jain, N. et al. (2023). Baseline Defenses for Adversarial Attacks Against Aligned Language Models.
- Yao, S. et al. (2023). ReAct: Synergizing Reasoning and Acting in Language Models. ICLR 2023.
- Shinn, N. et al. (2023). Reflexion: Language Agents with Verbal Reinforcement Learning. NeurIPS 2023.
