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
| Nebius | `nebius/<org>/<model>` | `nebius/Qwen/Qwen3-235B-A22B-Thinking-2507` |
| Google Gemini | `gemini/<model>` | `gemini/gemini-2.0-flash` |
| OpenAI | `<model>` (no prefix) | `gpt-4o` |
| Anthropic | `anthropic/<model>` | `anthropic/claude-3-5-sonnet-20241022` |

### For AgentBeats (Using Nebius)

Both agents can share the same `NEBIUS_API_KEY`:

```toml
[green_agent]
image = "ghcr.io/wutims/tau2-agent:latest"
env = {
  USER_LLM_MODEL = "nebius/Qwen/Qwen3-235B-A22B-Thinking-2507",
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
env = { USER_LLM_MODEL = "nebius/Qwen/Qwen3-235B-A22B-Thinking-2507", USER_LLM_API_KEY = "${NEBIUS_API_KEY}" }

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

**Implemented differently**: Simplified to only the required key, following the principle of minimal configuration.

```bash
NEBIUS_API_KEY=
```

**Rationale**: The template's `generate_compose.py` auto-extracts required env vars from `${VAR}` patterns in scenario.toml. A minimal `.env.example` avoids confusion and ensures users only configure what's actually needed. Additional API keys are added only when used.

### 1.3 Update README.md

Update with tau2-bench specific documentation:
- Overview of tau2-bench evaluation benchmark
- Local testing instructions
- Submission process for external agents
- Configuration reference (domain, num_tasks)
- LiteLLM model path format table

### 1.4 Local Validation ✅

**Completed**: Both compose generation and full end-to-end testing passed.

```bash
cd /home/ubuntu/workspace/tau2-bench-agent-leaderboard

# Create virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies for generate_compose.py
pip install tomli tomli-w pyyaml requests

# Generate docker-compose.yml (validates scenario.toml syntax)
python generate_compose.py --scenario scenario.toml

# Full assessment locally (requires credentials and local images)
cp .env.example .env
# Edit .env with your NEBIUS_API_KEY
docker compose up --abort-on-container-exit
cat output/results.json
```

**Test Results**:
- `generate_compose.py` succeeded without errors
- Full `docker compose up` completed with exit code 0
- Purple agent achieved **100% pass rate** (2/2 tasks on airline domain)
- Results successfully written to `output/results.json`

> **Note**: For local testing, use `local-scenario.toml` which references locally-built images (`tau2-agent:local`, `kimi-litellm-agent:local`) instead of GHCR images.

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
>
> **Status**: ✅ Agents registered, IDs captured in scenario.toml

### 2.1 Register Green Agent (tau2-agent) ✅

**Completed**: Agent registered on AgentBeats.

| Field | Value |
|-------|-------|
| Name | `tau2-agent` |
| Image | `ghcr.io/wutims/tau2-agent:latest` |
| Port | `9009` (note: updated from original plan of 8001) |
| Agent Card Path | `/a2a/tau2_agent/.well-known/agent-card.json` |
| **AgentBeats ID** | `019b950f-0070-7aa0-9135-085aab814ed7` |

### 2.2 Register Purple Agent (kimi-litellm-agent) ✅

**Completed**: Agent registered on AgentBeats.

| Field | Value |
|-------|-------|
| Name | `kimi-litellm-agent` |
| Image | `ghcr.io/wutims/kimi-litellm-agent:latest` |
| Port | `9009` (note: updated from original plan of 8002) |
| Agent Card Path | `/a2a/kimi_litellm_agent/.well-known/agent-card.json` |
| **AgentBeats ID** | `019b9515-47bd-7e80-8ad3-86c33d0175c9` |

**Port Change Rationale**: Both agents now use port 9009 for consistency. In Docker networking, each container has its own network namespace, so port conflicts don't occur.

### 2.3 Connect Leaderboard to Green Agent ✅

**Completed**: Leaderboard repository connected to green agent on AgentBeats dashboard.

Configuration:
- Leaderboard repository URL: `https://github.com/wuTims/tau2-bench-agent-leaderboard`
- Leaderboard query: Configured for tau2-bench results schema

### 2.4 Set Up Webhook ✅

**Completed**: Webhook configured for automatic leaderboard updates.

- Webhook URL added to leaderboard repo settings
- Content type: `application/json`

### 2.5 Update scenario.toml with Agent IDs ✅

**Completed**: scenario.toml updated with registered agent IDs (commit `29bea57`).

Current `scenario.toml`:
```toml
[green_agent]
agentbeats_id = "019b950f-0087-7d42-ad31-8965c08d1ed7"
env = { USER_LLM_MODEL = "nebius/Qwen/Qwen3-235B-A22B-Thinking-2507", USER_LLM_API_KEY = "${NEBIUS_API_KEY}" }

[[participants]]
agentbeats_id = "019b9515-47bd-7e80-8ad3-86c33d0175c9"
name = "agent"
env = { NEBIUS_API_KEY = "${NEBIUS_API_KEY}" }

[config]
domain = "airline"
num_tasks = 5
```

**Note**: For local testing, replace `agentbeats_id` with `image` field pointing to local images.

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
- [x] 2.1: Register tau2-agent (green) on AgentBeats
- [x] 2.2: Register kimi-litellm-agent (purple) on AgentBeats
- [x] 2.3: Connect leaderboard to green agent
- [x] 2.4: Set up webhook for automatic updates
- [x] 2.5: Update scenario.toml with agent IDs

### Phase 3: Final Verification
- [ ] GitHub Actions workflow runs successfully
- [ ] Results appear on AgentBeats dashboard
- [ ] Webhook triggers leaderboard updates

---

## Implementation Notes (Phase 1-2)

### Features Implemented Beyond Original Plan

The following features were added during implementation to enable successful end-to-end AgentBeats integration:

#### 1. Python-Based Agent Card URL Configuration

**Original approach**: Shell entrypoint scripts (`entrypoint.sh`) to modify `agent.json` before server start.

**Implemented approach**: Python function `_update_agent_card_url()` in `server.py` for both agents.

**Rationale**:
- Shell scripts added complexity and an extra layer of indirection
- Python approach integrates naturally with the FastAPI server startup sequence
- Better error handling and logging via loguru
- No shell dependency in container (simpler Dockerfile)
- Easier to test and debug

**Files changed**:
- `tau2_agent/server.py`: Added `_update_agent_card_url()` function (lines 29-51)
- `kimi_litellm_agent/server.py`: Same pattern
- Removed: `tau2_agent/docker_setup/entrypoint.sh` and `kimi_litellm_agent/docker_setup/entrypoint.sh`

#### 2. LiteLLM Model Support for Orchestrator

**Original assumption**: tau2_agent orchestrator always uses Gemini.

**Implemented**: Support for any LiteLLM-compatible model via `TAU2_AGENT_MODEL` env var.

**Usage**:
```bash
# Gemini (default)
TAU2_AGENT_MODEL=gemini-2.0-flash

# LiteLLM with Nebius
TAU2_AGENT_MODEL=litellm/nebius/Qwen/Qwen3-235B-A22B-Thinking-2507

# LiteLLM with OpenAI
TAU2_AGENT_MODEL=litellm/openai/gpt-4o
```

**Rationale**:
- AgentBeats/Nebius integration required non-Gemini model support
- Reuses existing `google.adk.models.lite_llm.LiteLlm` class from ADK
- Consistent with user LLM configuration pattern

**Files changed**:
- `tau2_agent/agent.py`: Added `create_model()` with prefix detection (lines 53-83)

#### 3. Environment Variable Fallback for User LLM Credentials

**Original assumption**: User LLM credentials passed via HTTP headers only.

**Implemented**: Fallback to `USER_LLM_MODEL` and `USER_LLM_API_KEY` env vars.

**Rationale**:
- AgentBeats deployments configure credentials via env vars, not per-request headers
- Maintains backward compatibility (headers take precedence)
- Documented in middleware module docstring

**Files changed**:
- `tau2_agent/middleware.py`: Added env var fallback (lines 123-132)

#### 4. Generate Compose Enhancements for Google ADK Agents

**Original template**: Basic A2A path support.

**Implemented**: `agent_name` field for ADK-style `/a2a/<agent_name>/` paths.

**Rationale**:
- Google ADK agents serve at `/a2a/<agent_name>/.well-known/agent-card.json`
- Health checks need correct path for proper container orchestration
- Enables automatic endpoint discovery via scenario.toml

**Files changed**:
- `tau2-bench-agent-leaderboard/generate_compose.py`: Added `agent_name` support and helper functions

### Local Testing Configuration

The `docker-compose.yml` in the leaderboard repo is configured for local testing:

```yaml
# Local images (for development)
image: tau2-agent:local
pull_policy: never

# Key environment variables
CARD_URL: http://green-agent:9009/a2a/tau2_agent  # Dynamic URL for Docker networking
TAU2_AGENT_MODEL: litellm/nebius/Qwen/Qwen3-235B-A22B-Thinking-2507  # Orchestrator model
USER_LLM_MODEL: nebius/Qwen/Qwen3-235B-A22B-Thinking-2507  # User simulator model
```

### Verified End-to-End Workflow

Local testing confirmed the complete AgentBeats integration:

1. **agentbeats-client** starts, waits for agents to be healthy
2. **green-agent** (tau2_agent) receives A2A assessment request
3. **tau2-bench** runs evaluation using **purple-agent** (kimi_litellm_agent)
4. Results written to `output/results.json`
5. Exit code 0 indicates success

Test results: 100% pass rate on airline domain tasks (2/2 successful)

### Registered Agent IDs

| Agent | AgentBeats ID | Port |
|-------|---------------|------|
| tau2-agent (green) | `019b950f-0070-7aa0-9135-085aab814ed7` | 9009 |
| kimi-litellm-agent (purple) | `019b9515-47bd-7e80-8ad3-86c33d0175c9` | 9009 |

---

## References

- [integration-plan.md](./integration-plan.md) - Integration architecture
- [integration-analysis.md](./integration-analysis.md) - Technical analysis
- [RDI-Foundation agentbeats-leaderboard-template](https://github.com/RDI-Foundation/agentbeats-leaderboard-template)
- [AgentBeats Platform Tutorial](https://docs.agentbeats.dev/tutorial/)
