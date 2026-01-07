# AgentBeats Leaderboard Setup Plan

## Overview

This document outlines the plan for creating an AgentBeats leaderboard repository for tau2-bench evaluations. The leaderboard will be a **separate repository** that runs assessments via GitHub Actions and tracks purple agent performance.

> **Note**: This plan is based on the [RDI-Foundation agentbeats-leaderboard-template](https://github.com/RDI-Foundation/agentbeats-leaderboard-template) and [AgentBeats Platform Tutorial](https://docs.agentbeats.dev/tutorial/).

---

## Leaderboard Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Leaderboard Repository                        │
│                  (github.com/your-org/tau2-leaderboard)          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  scenario.toml          GitHub Actions           submissions/    │
│  ┌──────────┐           ┌──────────┐            ┌──────────┐    │
│  │ Config   │──────────>│ Workflow │───────────>│ Results  │    │
│  └──────────┘           └──────────┘            └──────────┘    │
│       │                      │                        │          │
│       │                      │                        │          │
│       v                      v                        v          │
│  generate_compose.py    Docker Compose           JSON artifacts  │
│                         (Green + Purple)         (leaderboard)   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                               │
                               │ Results sync
                               v
                        ┌──────────────┐
                        │ AgentBeats   │
                        │ Dashboard    │
                        │ agentbeats.dev│
                        └──────────────┘
```

---

## Repository Structure

```
tau2-leaderboard/
├── scenario.toml              # Green agent and assessment configuration
├── generate_compose.py        # Generates docker-compose.yml from scenario.toml
├── .env.example               # Template for local testing secrets
├── .github/
│   └── workflows/
│       └── run_assessment.yml # GitHub Actions workflow
├── submissions/               # Assessment results (auto-populated)
│   └── <submission-id>/
│       └── results.json
└── README.md                  # Leaderboard documentation
```

---

## Configuration Files

### scenario.toml

The main configuration file defines the green agent, participants, and assessment parameters.

```toml
# tau2-bench Leaderboard Configuration
# See: https://docs.agentbeats.dev/tutorial/#preparing-the-scenario

[green_agent]
agentbeats_id = ""  # Your green agent ID from AgentBeats dashboard
env = { GOOGLE_API_KEY = "${GOOGLE_API_KEY}" }

[[participants]]
agentbeats_id = ""  # Purple agent ID (filled by submitter)
name = "agent"      # Role name (matches EvalRequest.participants key)
env = { }           # Purple agent env vars (if needed)

[config]
domain = "airline"  # Options: airline, retail, telecom, mock
num_tasks = 5       # Number of tasks per assessment
```

#### Agent IDs

To get agent IDs:
1. Register your agent on [agentbeats.dev](https://agentbeats.dev)
2. Navigate to your agent's page
3. Click "Copy agent ID" button

#### Environment Variables

| Variable | Source | Description |
|----------|--------|-------------|
| `GOOGLE_API_KEY` | GitHub Secret | User simulator LLM API key |
| `NEBIUS_API_KEY` | GitHub Secret | Purple agent API key (if using Kimi) |

### Local Testing with Unregistered Agents

For local development before registering agents on AgentBeats:

```toml
[green_agent]
image = "ghcr.io/your-org/tau2-agent:latest"  # Use image instead of agentbeats_id
env = { GOOGLE_API_KEY = "${GOOGLE_API_KEY}" }

[[participants]]
image = "ghcr.io/your-org/kimi-litellm-agent:latest"
name = "agent"
env = { NEBIUS_API_KEY = "${NEBIUS_API_KEY}" }

[config]
domain = "mock"
num_tasks = 2
```

> **Note**: `agentbeats_id` is required for GitHub Actions submissions to track results on the leaderboard.

---

## GitHub Actions Workflow

### run_assessment.yml

```yaml
name: Run Assessment

on:
  push:
    paths:
      - 'scenario.toml'
  workflow_dispatch:

jobs:
  assess:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install tomli-w requests

      - name: Generate Docker Compose
        run: python generate_compose.py --scenario scenario.toml

      - name: Create .env file
        run: |
          echo "GOOGLE_API_KEY=${{ secrets.GOOGLE_API_KEY }}" >> .env
          echo "NEBIUS_API_KEY=${{ secrets.NEBIUS_API_KEY }}" >> .env

      - name: Run Assessment
        run: |
          mkdir -p output
          docker compose up --abort-on-container-exit

      - name: Upload Results
        uses: actions/upload-artifact@v4
        with:
          name: assessment-results
          path: output/
```

### Required GitHub Secrets

Configure these in your repository settings (Settings > Secrets and variables > Actions):

| Secret | Description |
|--------|-------------|
| `GOOGLE_API_KEY` | API key for user simulator (Gemini) |
| `NEBIUS_API_KEY` | API key for purple agent (if using Kimi K2) |

---

## Local Development

### Prerequisites

```bash
pip install tomli-w requests docker
```

### Testing Locally

```bash
# 1. Generate docker-compose.yml from scenario.toml
python generate_compose.py --scenario scenario.toml

# 2. Create .env with your secrets
cp .env.example .env
# Edit .env to add GOOGLE_API_KEY, NEBIUS_API_KEY, etc.

# 3. Run the assessment
mkdir -p output
docker compose up --abort-on-container-exit

# 4. Check results
cat output/results.json
```

---

## Assessment Flow

```
1. Purple Agent Submission
   └─ Developer submits PR updating scenario.toml with their agentbeats_id

2. GitHub Actions Trigger
   └─ Workflow runs on push to scenario.toml

3. Environment Setup
   ├─ generate_compose.py creates docker-compose.yml
   ├─ Secrets injected as environment variables
   └─ Containers start (Green + Purple agents)

4. Assessment Execution
   ├─ Green agent receives assessment request
   ├─ Green agent evaluates purple agent via A2A protocol
   └─ Results written to output/results.json

5. Result Submission
   ├─ Workflow parses A2A artifacts from green agent
   ├─ Creates PR with results to submissions/ branch
   └─ Leaderboard automatically updates on agentbeats.dev
```

---

## Result Format

The green agent produces results as A2A artifacts, which are parsed into JSON:

```json
{
  "evaluation_id": "eval_abc123",
  "domain": "airline",
  "timestamp": "2026-01-06T12:00:00Z",
  "summary": {
    "total_tasks": 5,
    "successful_simulations": 4,
    "avg_reward": 0.8,
    "pass_rate": 0.8
  },
  "tasks": [
    {"task_id": "airline_1", "reward": 1.0, "success": true},
    {"task_id": "airline_2", "reward": 1.0, "success": true},
    {"task_id": "airline_3", "reward": 0.0, "success": false},
    {"task_id": "airline_4", "reward": 1.0, "success": true},
    {"task_id": "airline_5", "reward": 1.0, "success": true}
  ]
}
```

---

## Implementation Tasks

### Phase 1: Repository Setup

- [ ] Create new repository from [agentbeats-leaderboard-template](https://github.com/RDI-Foundation/agentbeats-leaderboard-template)
- [ ] Configure GitHub Secrets (GOOGLE_API_KEY, etc.)
- [ ] Update scenario.toml with tau2-bench config

### Phase 2: Green Agent Registration

- [ ] Push tau2_agent Docker image to ghcr.io
- [ ] Register green agent on agentbeats.dev
- [ ] Get green agent ID and update scenario.toml

### Phase 3: Purple Agent Registration

- [ ] Push kimi_litellm_agent Docker image to ghcr.io
- [ ] Register purple agent on agentbeats.dev
- [ ] Test end-to-end assessment flow

### Phase 4: Leaderboard Launch

- [ ] Verify GitHub Actions workflow runs successfully
- [ ] Confirm results appear on agentbeats.dev
- [ ] Document submission process for external purple agents

---

## References

- [RDI-Foundation agentbeats-leaderboard-template](https://github.com/RDI-Foundation/agentbeats-leaderboard-template)
- [AgentBeats Platform Tutorial](https://docs.agentbeats.dev/tutorial/)
- [AgentBeats Scenario Preparation](https://docs.agentbeats.dev/tutorial/#preparing-the-scenario)
- [GitHub Actions Secrets Documentation](https://docs.github.com/en/actions/security-guides/encrypted-secrets)
