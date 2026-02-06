# τ²-Adv Bench: Comprehensive Evaluation Results

## Executive Summary

**Models Evaluated:** xAI Grok 4.1 Fast, OpenAI GPT-OSS-120B
**Total Evaluations:** 134 configurations
**Domains:** Airline, Retail, Telecom
**Attack Strategies:** 5 (Social Engineering, Prompt Injection, Policy Exploitation, Identity Manipulation, Information Extraction)
**Sophistication Levels:** 3 (Low, Medium, High)

---

## Overall Results

| Metric | Value |
|--------|-------|
| Mean Safety Score | **59.1%** |
| Defense Rate | **100.0%** |
| Attack Success Rate | **0.0%** |

**Key Finding:** All models successfully defended against adversarial attacks (100% defense rate), but safety scores reveal significant variation in *how cleanly* they handle adversarial interactions.

---

## Results by Model

| Model | Evaluations | Safety Score | Defense Rate | ASR |
|-------|-------------|--------------|--------------|-----|
| **GPT-OSS-120B** | 62 | **66.0%** | 100% | 0% |
| **Grok 4.1 Fast** | 72 | 53.3% | 100% | 0% |

**Gap:** GPT-OSS-120B outperforms Grok 4.1 Fast by **12.7 percentage points** in safety score.

---

## Results by Domain

| Domain | Evaluations | Safety Score | Vulnerability Ranking |
|--------|-------------|--------------|----------------------|
| **Telecom** | 45 | **74.4%** | Most Resilient |
| **Retail** | 48 | 65.1% | Moderate |
| **Airline** | 41 | 35.4% | **Most Vulnerable** |

**Key Finding:** The airline domain is **39 percentage points** more vulnerable than telecom, suggesting complex policy structures (cancellations, refunds, passenger modifications) create more attack opportunities.

---

## Results by Attack Strategy

| Strategy | Evaluations | Safety Score | Effectiveness |
|----------|-------------|--------------|---------------|
| **Prompt Injection** | 34 | **45.3%** | Most Effective Attack |
| Policy Exploitation | 17 | 55.9% | High |
| Social Engineering | 48 | 61.7% | Moderate |
| Identity Manipulation | 18 | 65.3% | Low |
| **Information Extraction** | 17 | **76.5%** | Least Effective Attack |

**Key Finding:** Prompt injection remains the most effective attack vector, achieving the lowest safety scores across all models.

---

## Results by Sophistication Level

| Level | Evaluations | Safety Score | Δ from Low |
|-------|-------------|--------------|------------|
| Low (φ=0.0) | 47 | **63.8%** | — |
| Medium (φ=0.5) | 43 | 56.3% | -7.5% |
| High (φ=1.0) | 44 | 56.9% | -6.9% |

**Key Finding:** Higher sophistication attacks reduce safety scores by approximately **11%** compared to naive (low sophistication) attacks.

---

## Model × Domain Breakdown

### Grok 4.1 Fast
| Domain | Safety Score | Evaluations |
|--------|--------------|-------------|
| Telecom | **74.8%** | 24 |
| Retail | 55.6% | 24 |
| Airline | 29.4% | 24 |

### GPT-OSS-120B
| Domain | Safety Score | Evaluations |
|--------|--------------|-------------|
| Retail | **74.6%** | 24 |
| Telecom | 74.0% | 21 |
| Airline | 43.8% | 17 |

**Key Finding:** Both models struggle with the airline domain, but GPT-OSS-120B handles it 14.4 percentage points better than Grok 4.1 Fast.

---

## Model × Strategy Breakdown

### Grok 4.1 Fast
| Strategy | Safety Score | Evaluations |
|----------|--------------|-------------|
| Information Extraction | **73.9%** | 9 |
| Identity Manipulation | 62.2% | 9 |
| Social Engineering | 52.4% | 27 |
| Policy Exploitation | 46.1% | 9 |
| Prompt Injection | **43.3%** | 18 |

### GPT-OSS-120B
| Strategy | Safety Score | Evaluations |
|----------|--------------|-------------|
| Information Extraction | **79.4%** | 8 |
| Social Engineering | 73.6% | 21 |
| Identity Manipulation | 68.3% | 9 |
| Policy Exploitation | 66.9% | 8 |
| Prompt Injection | **47.5%** | 16 |

**Key Finding:** GPT-OSS-120B shows particularly strong resistance to social engineering (+21.2pp over Grok) and policy exploitation (+20.8pp).

---

## Strategy × Sophistication Breakdown

### Social Engineering
| Sophistication | Safety Score | Δ from Low |
|----------------|--------------|------------|
| Low | **70.3%** | — |
| Medium | 56.2% | -14.1% |
| High | 57.7% | -12.6% |

### Prompt Injection
| Sophistication | Safety Score | Δ from Low |
|----------------|--------------|------------|
| Low | 47.1% | — |
| Medium | **43.6%** | -3.5% |
| High | 45.0% | -2.1% |

### Policy Exploitation
| Sophistication | Safety Score | Δ from Low |
|----------------|--------------|------------|
| Low | 51.7% | — |
| Medium | **64.0%** | +12.3% |
| High | 53.3% | +1.6% |

### Identity Manipulation
| Sophistication | Safety Score | Δ from Low |
|----------------|--------------|------------|
| Low | **70.8%** | — |
| Medium | 59.2% | -11.6% |
| High | 65.8% | -5.0% |

### Information Extraction
| Sophistication | Safety Score | Δ from Low |
|----------------|--------------|------------|
| Low | **84.2%** | — |
| Medium | 73.0% | -11.2% |
| High | 71.7% | -12.5% |

---

## Key Insights

### 1. Domain Complexity Drives Vulnerability
- **39 percentage point gap** between airline (35.4%) and telecom (74.4%)
- Complex, interconnected policies in airline domain create more attack surfaces
- Clearer verification procedures in telecom provide better defense

### 2. Prompt Injection Remains Unsolved
- Lowest safety scores across all strategies (45.3%)
- Consistent vulnerability across both models
- Suggests architectural limitations in distinguishing system vs. user instructions

### 3. Sophistication Matters (But Not Uniformly)
- Social engineering shows **14.1% degradation** with sophistication
- Information extraction shows **12.5% degradation**
- Prompt injection shows minimal change (attacks already effective at low sophistication)

### 4. Model Selection Impacts Safety
- **12.7 percentage point gap** between best (GPT-OSS-120B) and worst (Grok 4.1 Fast)
- GPT-OSS-120B shows more balanced safety profile across all dimensions
- Different models may suit different deployment contexts

### 5. All Models Defend Successfully
- 100% defense rate across all configurations
- 0% attack success rate
- But safety scores reveal quality of defense varies significantly

---

## Recommendations

1. **Domain-Specific Training:** Deploy models trained on adversarial scenarios within their specific policy domain
2. **Prompt Injection Hardening:** Invest in architectural innovations for instruction separation
3. **Sophistication-Aware Testing:** Red-team with multi-turn, trust-building attacks, not just obvious attempts
4. **Model Selection:** Consider GPT-OSS-120B or similar for high-stakes deployments requiring balanced safety
5. **Continuous Evaluation:** Use τ²-Adv Bench for pre-deployment adversarial testing

---

*Generated: February 5, 2026*
*Framework: τ²-Adv Bench v1.0*
