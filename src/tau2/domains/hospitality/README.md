# Hospitality Domain

A full-service restaurant simulation for evaluating conversational AI agents in high-stakes service environments.

## Motivation

The AI in restaurants market is valued at USD 6.1 billion in 2024 and projected to reach USD 48.3 billion by 2033 (CAGR 23.5%) [1]. Voice AI adoption is accelerating, with companies like SoundHound AI now powering over 10,000 restaurant locations for drive-thru and phone ordering [2]. However, current deployments remain limited to **low-complexity, transactional interactions** — drive-thru orders, phone reservations, and basic FAQs.

Full-service dining presents fundamentally different challenges:
- **Stateful, multi-turn conversations** with emotionally varied customers
- **Liability-sensitive decisions** (food allergies, safety incidents)
- **Hierarchical authority constraints** requiring appropriate escalation
- **Conflicting objectives** (customer satisfaction vs. policy compliance)

This domain addresses the gap between current benchmark coverage and the requirements of full-service hospitality AI.

**Grounded in Real Operations:** The entire domain is modeled after an actual restaurant operation. This includes:
- **Seating configuration**: Table types, capacities, and expansion limits based on real floor plans
- **Staff hierarchy**: Role definitions, authority levels, and escalation paths reflecting actual restaurant org charts
- **Menu and pricing**: Soup bases, food items, and pricing structures from operational menus
- **Policies**: Discount limits, reservation rules, and service recovery procedures from real staff handbooks
- **Tasks**: All 116 scenarios are derived from actual customer interactions and operational incidents

**References:**
1. Dataintelo. "AI in Restaurants Market." 2024. https://dataintelo.com/report/ai-in-restaurants-market
2. SoundHound AI. "Next-Generation AI Platform for Restaurants." Feb 2025. https://investors.soundhound.com/news-releases/

## Overview

This domain simulates a Chinese hot pot restaurant (Berkeley Hot Pot), testing agents on complex interactions that require balancing customer satisfaction with strict operational policies.

**Key characteristics:**
- Multi-turn conversations with emotionally varied customers
- Role-based authority limits (Server vs. Manager)
- Food safety and liability considerations
- Temporal and capacity constraints

## Domain Features

### High-Stakes Decision Making
Agents must prioritize safety and liability over customer satisfaction. For example, strictly enforcing allergy protocols (e.g., verifying hidden allergens) even when customers insist otherwise.

### Hierarchical Reasoning & Escalation
Agents must judge their authority limits — deciding whether to resolve issues independently (e.g., small discounts) or escalate to management — mirroring real-world role-based access control.

### Investigation-First SOPs
Agents must verify facts via tools before taking action, reducing confabulation. For instance, checking kitchen status or bill details *before* offering apologies or compensation.

### Deterministic Evaluation
All 116 tasks are scored via verifiable actions (tool calls) and database state assertions, ensuring reproducible results without relying on unstable LLM-as-judge metrics.

## Task Categories

**116 tasks** (101 base + 15 kitchen coordination variants) organized by staff role:

| Category | Count | Description |
|----------|-------|-------------|
| `host_phone` | 13 | Phone reservations, inquiries, complaint calls |
| `host_seating` | 6 | Table assignment, party size changes |
| `host_walkin` | 1 | Walk-in customer handling |
| `server_food_safety` | 11 | Allergy and dietary restriction handling |
| `server_promotion` | 16 | Discounts, loyalty points, secret codes |
| `server_food_issue` | 7 | Order accuracy, out-of-stock items |
| `server_billing` | 6 | Payment and billing inquiries |
| `server_celebration` | 4 | Birthday, anniversary coordination |
| `server_incident` | 13 | Complaints, accidents, escalations |
| `server_special_policy` | 6 | Special amenities and policies |
| `server_misc` | 33 | Menu knowledge, seating preferences, misc |

### Kitchen Coordination Variants

Additional task variants simulate internal operational challenges:
- `_overload`: Kitchen overwhelmed
- `_understaffed`: Short-staffed scenarios
- `_equipment`: Equipment malfunctions
- `_attitude`: Difficult colleague interactions

Agents must handle customer-facing issues without exposing internal problems.

## Usage

```bash
# Run specific tasks
tau2 run --domain hospitality --task-ids hospitality_007_hidden_allergy --agent-llm gpt-4o

# Run full benchmark
tau2 run --domain hospitality --task-split base --agent-llm gpt-4o
```

## Evaluation

Tasks are evaluated using:
- **ACTION**: Required tool calls (e.g., `check_allergy_safety`, `create_reservation`)
- **ENV_ASSERTION**: Database state verification (e.g., `assert_escalated_to_manager`)

No LLM-as-judge; all evaluations are deterministic.

## Model Performance

Baseline results on the 11-task `base` split:

| Model | Pass Rate | Avg Reward | Avg Cost/Conv |
|-------|-----------|------------|---------------|
| GPT-4o-mini | 63.6% | 0.636 | $0.004 |

*Evaluated with `--max-concurrency 1` on 2026-01-31.*

## Files

```
data/tau2/domains/hospitality/
├── db.json              # Restaurant database (menu, tables, customers)
├── policy.md            # Operational policies (466 lines)
├── split_tasks.json     # Task split definitions
└── tasks.json           # Task definitions (116 tasks)

src/tau2/domains/hospitality/
├── data_model.py        # Pydantic models (agent-side)
├── user_data_model.py   # Pydantic models (user-side)
├── environment.py       # Environment setup
├── tools.py             # Agent tool implementations (33 tools)
├── user_tools.py        # User tool implementations (8 tools)
└── utils.py             # Helper functions
```
