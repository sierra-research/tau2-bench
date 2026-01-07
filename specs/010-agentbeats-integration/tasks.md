# AgentBeats Integration Tasks

## Overview

This document defines the implementation tasks for AgentBeats integration. The approach leverages the **official AgentBeats template** which provides all orchestration infrastructure, eliminating the need for custom trigger scripts or utilities.

**Key Insight**: The `agentbeats-client` container (provided by the template) handles:
- Waiting for agent readiness
- Sending A2A assessment requests
- Collecting results
- Signaling completion

We only need to configure and customize, not build from scratch.

---

## Target Repositories

| Repository | Purpose |
|------------|---------|
| `tau2-bench-agent` | Source code for agents (already complete) |
| `tau2-bench-agent-leaderboard` | Leaderboard configuration (new, from template) |

**Container Registry**: `ghcr.io/wutims/`
- `ghcr.io/wutims/tau2-agent:latest` (Green Agent)
- `ghcr.io/wutims/kimi-litellm-agent:latest` (Purple Agent)

---

## Leaderboard Query Reference

Use this JSON when configuring the green agent's leaderboard queries on AgentBeats:

```json
[
  {
    "name": "Overall Performance",
    "query": "SELECT results.participants.agent AS id, ROUND(results.summary.pass_rate * 100, 1) AS \"Pass Rate %\", results.summary.total_tasks AS \"Tasks\", results.summary.successful_simulations AS \"Passed\", ROUND(results.summary.avg_reward, 2) AS \"Avg Reward\" FROM results ORDER BY \"Pass Rate %\" DESC"
  }
]
```

> **Note**: This query matches our results.json schema. May need adjustment after first run if agentbeats-client wraps the output differently.

---

## User LLM Configuration

The Green Agent (tau2_agent) uses an LLM for the **user simulator** that simulates customer interactions during evaluation. This is separate from the Purple Agent's LLM.

### LiteLLM Model Path Format

The user LLM model must be specified as a **full LiteLLM model path** with provider prefix:

| Provider | Model Path Format | Example |
|----------|-------------------|---------|
| Nebius | `nebius/<org>/<model>` | `nebius/moonshotai/Kimi-K2-Instruct` |
| Google Gemini | `gemini/<model>` | `gemini/gemini-2.0-flash` |
| OpenAI | `<model>` (no prefix) | `gpt-4o` |
| Anthropic | `anthropic/<model>` | `anthropic/claude-3-5-sonnet-20241022` |

### For AgentBeats (Using Nebius)

Both agents can share the same `NEBIUS_API_KEY`:

```toml
[green_agent]
image = "ghcr.io/wutims/tau2-agent:latest"
env = {
  USER_LLM_MODEL = "nebius/moonshotai/Kimi-K2-Instruct",
  USER_LLM_API_KEY = "${NEBIUS_API_KEY}"
}

[[participants]]
image = "ghcr.io/wutims/kimi-litellm-agent:latest"
name = "agent"
env = { NEBIUS_API_KEY = "${NEBIUS_API_KEY}" }
```

---

## Phase 0: Prerequisites

### 0.1 Verify Docker Images Published

```bash
# Verify both images exist on GHCR
docker pull ghcr.io/wutims/tau2-agent:latest
docker pull ghcr.io/wutims/kimi-litellm-agent:latest
```

If images are missing, trigger the workflow:
```bash
gh workflow run docker-publish.yml
gh run list --workflow=docker-publish.yml --limit=3
```

### 0.2 Create Leaderboard Repository from Template

**Important**: Use the official template, do not create from scratch.

1. Navigate to [RDI-Foundation/agentbeats-leaderboard-template](https://github.com/RDI-Foundation/agentbeats-leaderboard-template)
2. Click **"Use this template"** > **"Create a new repository"**
3. Owner: `wuTims`, Name: `tau2-bench-agent-leaderboard`
4. Visibility: **Public** (required for AgentBeats)
5. Click **"Create repository"**

```bash
# Clone to workspace
cd /home/ubuntu/workspace
git clone https://github.com/wuTims/tau2-bench-agent-leaderboard.git
```

### 0.3 Configure Repository Settings

1. Navigate to **Settings** > **Actions** > **General**
2. Under "Workflow permissions", select **"Read and write permissions"**
3. Click **Save**

### 0.4 Configure GitHub Secrets

Navigate to **Settings** > **Secrets and variables** > **Actions** > **New repository secret**

Add:
- `NEBIUS_API_KEY`: API key for Nebius (used by both agents)

---

## Phase 1: Leaderboard Configuration

> **Working Directory**: `/home/ubuntu/workspace/tau2-bench-agent-leaderboard`

The template provides these files (do not recreate):
- `generate_compose.py` - Generates docker-compose.yml with agentbeats-client
- `.github/workflows/run-scenario.yml` - Assessment workflow
- `record_provenance.py` - Metadata recording

### 1.1 Customize scenario.toml

Replace the template's `scenario.toml`:

```toml
# tau2-bench Leaderboard Configuration
# See: https://docs.agentbeats.dev/tutorial/

[green_agent]
# Green Agent: tau2-bench evaluation orchestrator
# For local testing, use `image` instead of `agentbeats_id`
image = "ghcr.io/wutims/tau2-agent:latest"
# User simulator LLM: full LiteLLM model path required
env = { USER_LLM_MODEL = "nebius/moonshotai/Kimi-K2-Instruct", USER_LLM_API_KEY = "${NEBIUS_API_KEY}" }

[[participants]]
# Purple Agent: The agent being evaluated
# Submitters: Replace with your agent's agentbeats_id
image = "ghcr.io/wutims/kimi-litellm-agent:latest"
name = "agent"
env = { NEBIUS_API_KEY = "${NEBIUS_API_KEY}" }

[config]
# Assessment configuration passed to the green agent
domain = "airline"
num_tasks = 5
```

> **Note**: TOML inline tables (`env = { ... }`) must be single-line. Comments go on separate lines above.

### 1.2 Create .env.example

```bash
# tau2-bench Leaderboard Environment Variables
# Copy this file to .env and fill in your credentials

# Nebius API key (used by both user simulator and purple agent when using Nebius models)
NEBIUS_API_KEY=your-nebius-api-key-here

# Optional: Other LLM provider API keys (uncomment if using)
# GOOGLE_API_KEY=your-google-api-key-here
# OPENAI_API_KEY=your-openai-api-key-here
# ANTHROPIC_API_KEY=your-anthropic-api-key-here
```

### 1.3 Update README.md

Update with tau2-bench specific documentation:
- Overview of tau2-bench evaluation benchmark
- Local testing instructions
- Submission process for external agents
- Configuration reference (domain, num_tasks)
- LiteLLM model path format table

### 1.4 Local Validation

```bash
cd /home/ubuntu/workspace/tau2-bench-agent-leaderboard

# Create virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies for generate_compose.py
pip install tomli tomli-w pyyaml requests

# Generate docker-compose.yml (validates scenario.toml syntax)
python generate_compose.py --scenario scenario.toml

# Optional: Run full assessment locally (requires credentials)
cp .env.example .env
# Edit .env with your NEBIUS_API_KEY
docker compose up --abort-on-container-exit
cat output/results.json
```

> **Minimum validation**: Ensure `generate_compose.py` succeeds without errors. Full `docker compose up` requires valid API credentials.

### 1.5 Commit and Push

```bash
git add scenario.toml .env.example README.md .gitignore
git commit -m "Configure tau2-bench leaderboard

- Customize scenario.toml for tau2-bench agents
- Add .env.example with credential templates
- Update README with tau2-bench documentation
- Update .gitignore to track .env.example, ignore .venv/"

git push -u origin main
```

---

## Phase 2: Agent Registration

> **Prerequisite**: Phase 1 complete, images published

### 2.1 Register Green Agent (tau2-agent)

1. Navigate to https://agentbeats.dev
2. Click "Register Agent"
3. Fill in:
   - Name: `tau2-agent`
   - Description: `tau2-bench evaluation agent (Green Agent)`
   - Image: `ghcr.io/wutims/tau2-agent:latest`
   - Port: `8001`
   - Agent Card Path: `/a2a/tau2_agent/.well-known/agent-card.json`
4. Copy the generated `agentbeats_id`

### 2.2 Register Purple Agent (kimi-litellm-agent)

1. Navigate to https://agentbeats.dev
2. Click "Register Agent"
3. Fill in:
   - Name: `kimi-litellm-agent`
   - Description: `Kimi K2 agent via LiteLLM (Purple Agent)`
   - Image: `ghcr.io/wutims/kimi-litellm-agent:latest`
   - Port: `8002`
   - Agent Card Path: `/a2a/kimi_litellm_agent/.well-known/agent-card.json`
4. Copy the generated `agentbeats_id`

### 2.3 Connect Leaderboard to Green Agent

1. Navigate to your green agent's page on AgentBeats
2. Click **"Edit Agent"**
3. Add leaderboard repository URL: `https://github.com/wuTims/tau2-bench-agent-leaderboard`
4. Add leaderboard query config:

```json
[
  {
    "name": "Overall Performance",
    "query": "SELECT
      id,
      ROUND(pass_rate, 1) AS \"Pass Rate\",
      ROUND(time_used, 1) AS \"Time\",
      total_tasks AS \"# Tasks\"
    FROM (
      SELECT *,
             ROW_NUMBER() OVER (PARTITION BY id ORDER BY pass_rate DESC, time_used ASC) AS rn
      FROM (
        SELECT
          results.participants.agent AS id,
          res.pass_rate AS pass_rate,
          res.time_used AS time_used,
          SUM(res.max_score) OVER (PARTITION BY results.participants.agent) AS total_tasks
        FROM results
        CROSS JOIN UNNEST(results.results) AS r(res)
      )
    )
    WHERE rn = 1
    ORDER BY \"Pass Rate\" DESC;"
  }
]
```

5. Click **Save**

### 2.4 Set Up Webhook

1. On green agent page, open **"Webhook Integration"** box
2. Copy the webhook URL
3. Go to leaderboard repo: **Settings** > **Webhooks** > **Add webhook**
4. Fill in:
   - **Payload URL**: The copied webhook URL
   - **Content type**: `application/json` (important!)
5. Click **"Add webhook"**

### 2.5 Update scenario.toml with Agent IDs

```bash
cd /home/ubuntu/workspace/tau2-bench-agent-leaderboard

# Edit scenario.toml to use agentbeats_id instead of image
# [green_agent]
# agentbeats_id = "<copied-green-agent-id>"
#
# [[participants]]
# agentbeats_id = "<copied-purple-agent-id>"

git add scenario.toml
git commit -m "Use registered AgentBeats agent IDs"
git push
```

---

## Phase 3: Final Verification

### 3.1 Verify GitHub Actions Assessment

1. Push to `tau2-bench-agent-leaderboard` triggers workflow
2. Navigate to Actions tab
3. Watch the `Run Assessment` workflow
4. Verify:
   - Docker Compose generated correctly
   - Both containers start and pass health checks
   - Evaluation completes without errors
   - Results artifact is uploaded

### 3.2 Verify AgentBeats Dashboard

1. Navigate to https://agentbeats.dev/leaderboard
2. Verify agents appear
3. Check evaluation results are recorded

### 3.3 Verify Webhook

1. Make a small commit to leaderboard repo
2. Check that AgentBeats dashboard updates automatically

---

## Task Checklist

### Phase 0: Prerequisites
- [x] Verify Docker images published to ghcr.io/wutims/
- [x] Create tau2-bench-agent-leaderboard repository from template
- [x] Configure repository settings (workflow permissions)
- [x] Configure GitHub secrets (NEBIUS_API_KEY)

### Phase 1: Leaderboard Configuration
- [x] 1.1: Customize scenario.toml
- [x] 1.2: Create .env.example
- [x] 1.3: Update README.md
- [x] 1.4: Local validation passes (`generate_compose.py` succeeded)
- [x] 1.5: Commit and push to main

### Phase 2: Agent Registration
- [ ] 2.1: Register tau2-agent (green) on AgentBeats
- [ ] 2.2: Register kimi-litellm-agent (purple) on AgentBeats
- [ ] 2.3: Connect leaderboard to green agent
- [ ] 2.4: Set up webhook for automatic updates
- [ ] 2.5: Update scenario.toml with agent IDs

### Phase 3: Final Verification
- [ ] GitHub Actions workflow runs successfully
- [ ] Results appear on AgentBeats dashboard
- [ ] Webhook triggers leaderboard updates

---

## References

- [integration-plan.md](./integration-plan.md) - Integration architecture
- [integration-analysis.md](./integration-analysis.md) - Technical analysis
- [RDI-Foundation agentbeats-leaderboard-template](https://github.com/RDI-Foundation/agentbeats-leaderboard-template)
- [AgentBeats Platform Tutorial](https://docs.agentbeats.dev/tutorial/)
