# CRM Domain - Commitment-Trust Theory Framework

This CRM domain for tau2-bench implements relationship management based on **Commitment-Trust Theory (Morgan & Hunt, 1994)**.

## Overview

The CRM domain enables agents to:
- Build trust through reliable, competent, and transparent interactions
- Strengthen commitment through value delivery and relationship investment
- Navigate relationship stages from prospect to loyal advocate
- Measure and track trust/commitment metrics

## Architecture

### Domain Components

```
src/tau2/domains/crm/
├── __init__.py          # Package initialization
├── data_model.py        # CRM data models (Person, Company, Opportunity, Task)
├── tools.py             # CRM tools with trust/commitment metrics
├── environment.py       # Environment setup and task loading
└── utils.py             # Data paths configuration

data/tau2/domains/crm/
├── db.json              # Database initialization
├── policy.md            # Domain policy based on Commitment-Trust Theory
├── tasks.json           # 10 relationship-building tasks
└── split_tasks.json     # Train/test splits
```

### Data Models

- **Person**: trust_level, commitment_level, lead_score, relationship_stage
- **Company**: name, industry, revenue, employee_count
- **Opportunity**: stage, value, probability, person_id, company_id
- **Task**: status, priority, associated person/company

### Tools

See `tools.py` for all available tools, including:

#### Read Operations
- `find_people()` - Search people by criteria
- `find_companies()` - Search companies by criteria
- `find_opportunities()` - Find opportunities by stage/person
- `find_tasks()` - Locate tasks by status/person
- `get_relationship_metrics()` - Get trust/commitment metrics

#### Write Operations
- `create_person()` - Create person with trust/commitment levels
- `update_person_trust_commitment()` - Update relationship metrics
- `create_company()` - Add company record
- `create_opportunity()` - Create opportunity with stage
- `update_opportunity_stage()` - Progress opportunity through pipeline
- `create_task()` - Create follow-up task

## Relationship Stages

Based on Commitment-Trust Theory:

1. **Prospect** (trust: 0-20, commitment: 0-20) - Initial exploration
2. **Qualified** (trust: 20-40, commitment: 20-40) - Evaluating options
3. **Proposal** (trust: 40-60, commitment: 40-60) - Solution evaluation
4. **Negotiation** (trust: 60-80, commitment: 60-80) - Terms discussion
5. **Closed-Won** (trust: 80-100, commitment: 80-100) - Agreement reached
6. **Loyal** (trust: 90-100, commitment: 90-100) - Repeat business
7. **At-Risk** (trust: <50, commitment: <50) - Relationship at risk

## Tasks

The domain includes 10 tasks covering the relationship lifecycle:

| ID | Purpose | Focus |
|----|---------|-------|
| 0 | Initial inquiry | Building first impression |
| 1 | Stubborn user | Overcoming skepticism |
| 2 | Reliability test | Demonstrating competence |
| 3 | Evidence seeking | Providing proof |
| 4 | Small commitments | Testing reliability |
| 5 | Long-term value | Demonstrating sustained value |
| 6 | Problem resolution | Maintaining trust during challenges |
| 7 | Referral request | Relationship strength |
| 8 | Loyalty reinforcement | Deepening relationship |
| 9 | Churn prevention | Trust restoration |

## Usage

### Basic Execution

```bash
tau2 run --domain crm --agent-llm gpt-4.1 --user-llm gpt-4.1 \
  --num-trials 1 --num-tasks 5
```

### With Twenty CRM Integration

The domain is designed to work with Twenty CRM's MCP server:

1. Configure `.env` with Twenty CRM credentials
2. The domain tools can be extended to directly call Twenty CRM MCP
3. Current implementation provides CRM-agnostic tools that can be connected

### Environment Setup

```bash
cd tau2-bench-copy
cp .env.example .env
# Edit .env with your API keys and Twenty CRM credentials
```

## Key Features

### Trust-Building Tools

- `update_person_trust_commitment()` - Track relationship metrics
- `get_relationship_metrics()` - Analyze relationship health
- `update_person_lead_score()` - Update sales readiness

### Commitment-Strengthening Features

- Opportunity progression through pipeline stages
- Task creation for follow-up actions
- Long-term value communication

### Evaluation Criteria

Tasks evaluate:
- Trust building behaviors
- Commitment development
- Stage progression
- Communication quality
- Problem resolution effectiveness

## Integration with Twenty CRM

The CRM domain is designed to work with Twenty CRM's MCP server:

**Connection**: `http://x1000c1s1b0n0:3000/mcp?mode=direct`

**Authentication**: Bearer token from `TWENTY_MCP_TOKEN` in the `.env` file (see `.env.example`)

**Tools**: 280+ CRM tools available via MCP protocol

### Integration Pattern

1. Use CRM tools for relationship management
2. Connect to Twenty CRM MCP for data operations
3. Track trust/commitment metrics
4. Guide relationship progression

## Theory Background

### Commitment-Trust Theory (Morgan & Hunt, 1994)

**Core Premise**: Commitment and trust are mutually reinforcing and critical for long-term success.

**Trust Components**:
- **Reliability**: Consistent performance
- **Competence**: Expertise and knowledge
- **Integrity**: Honesty and transparency

**Commitment Elements**:
- **Attachment**: Emotional connection
- **Investment**: Resources committed
- **Satisfaction**: Perceived value

**Relationship Outcomes**:
- Transaction frequency
- Customer lifetime value
- Referral behavior
- Price tolerance

## References

- Morgan, R. M., & Hunt, S. D. (1994). The Commitment-Trust Theory of Relationship Marketing. *Journal of Marketing*, 58(3), 20-38.

- Ganesan, S. (1994). Determinants of Long-Term Orientation in Buyer-Seller Relationships. *Journal of Marketing*, 58(2), 1-19.

## Future Enhancements

- [ ] Direct Twenty CRM MCP integration
- [ ] Advanced trust/commitment measurement
- [ ] Multi-agent relationship management
- [ ] Voice interaction support
- [ ] Custom user simulator for CRM scenarios
