# τ²-Adv Bench: Comprehensive Evaluation Results

## Executive Summary

**Models Evaluated:** 6 models
**Total Evaluations:** 506 task evaluations
**Domains:** Airline, Retail, Telecom
**Attack Strategies:** Social Engineering, Prompt Injection, Policy Exploitation, Identity Manipulation, Information Extraction
**Sophistication Levels:** Low (0.0), Medium (0.5), High (1.0)

---

## Overall Model Ranking

| Rank | Model | Evaluations | Safety Score | Defense Rate | ASR |
|------|-------|-------------|--------------|--------------|-----|
| 🥇 | **gpt-oss-120b** | 202 | **70.7%** | 100% | 0% |
| 🥈 | **grok-4.1-fast** | 72 | **53.3%** | 100% | 0% |
| 🥉 | **deepseek-v3.2** | 72 | **52.7%** | 100% | 0% |
| 4 | **kimi-k2.5** | 68 | **34.5%** | 100% | 0% |
| 5 | **mimo-v2-flash** | 72 | **29.4%** | 100% | 0% |
| 6 | **nemotron-3-nano-30b** | 20 | **0.7%** | 100% | 0% |

---

## Results by Domain

| Domain | Safety Score | Vulnerability Ranking |
|--------|--------------|----------------------|
| **Retail** | 65.7% | Most Resilient |
| **Telecom** | 60.4% | Moderate |
| **Airline** | 32.2% | Most Vulnerable |

---

## Results by Attack Strategy

| Strategy | Safety Score | Attack Effectiveness |
|----------|--------------|---------------------|
| **Prompt Injection** | 39.8% | Most Effective |
| **Policy Exploitation** | 47.0% | Moderate |
| **Social Engineering** | 52.0% | Moderate |
| **Identity Manipulation** | 64.0% | Moderate |
| **Information Extraction** | 71.5% | Least Effective |

---

## Model × Domain Breakdown

| Model | Airline | Retail | Telecom |
|-------|---------|--------|---------|
| **gpt-oss-120b** | 52.7% | 83.3% | 75.6% |
| **grok-4.1-fast** | 29.4% | 55.6% | 74.8% |
| **deepseek-v3.2** | 30.2% | 73.8% | 54.2% |
| **kimi-k2.5** | 22.5% | 46.0% | 32.9% |
| **mimo-v2-flash** | 14.0% | 34.4% | 40.0% |
| **nemotron-3-nano-30b** | 0.7% | N/A | N/A |

---

## Model × Strategy Breakdown

| Model | Social Eng. | Prompt Inj. | Policy Exp. | Identity Man. | Info Extract. |
|-------|-------------|-------------|-------------|---------------|---------------|
| **gpt-oss-120b** | 73.2% | 62.1% | 61.2% | 79.4% | 81.4% |
| **grok-4.1-fast** | 52.4% | 43.3% | 46.1% | 62.2% | 73.9% |
| **deepseek-v3.2** | 55.2% | 34.4% | 56.1% | 65.6% | 65.6% |
| **kimi-k2.5** | 40.6% | 13.9% | 20.6% | 46.7% | 59.4% |
| **mimo-v2-flash** | 27.9% | 13.8% | 28.3% | 35.0% | 59.4% |
| **nemotron-3-nano-30b** | 1.1% | 0.0% | 0.0% | N/A | N/A |

---

## Key Insights

### 1. Model Performance Gap
- **Best Model:** gpt-oss-120b (70.7% safety score)
- **Worst Model:** nemotron-3-nano-30b (0.7% safety score)
- **Performance Gap:** 70.0 percentage points

### 2. Domain Vulnerability
- **Most Resilient:** Retail (65.7%)
- **Most Vulnerable:** Airline (32.2%)
- Complex policy structures in airline domain create more attack opportunities

### 3. Attack Strategy Effectiveness
- **Most Effective:** Prompt Injection (39.8% safety)
- **Least Effective:** Information Extraction (71.5% safety)
- Prompt injection remains the most effective attack vector across all models

### 4. Universal Defense Success
- All models achieved **100% defense rate** (0% ASR)
- However, safety scores reveal significant variation in *how cleanly* models handle adversarial interactions

---

## Visualizations

- [Comprehensive Results](all_models_comprehensive_results.png)
- [Adversarial Vulnerability Scores](adversarial_vulnerability_scores.png)

---

*Generated: February 05, 2026*
*Framework: τ²-Adv Bench v1.0*
