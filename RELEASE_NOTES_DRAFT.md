# Release Notes Draft — Major Version

Working draft for release messaging. Written for someone who only knows the
current public release (text-only evaluation of airline, retail, telecom, and
mock domains).

---

## Positioning

τ-bench grows from a text-only agent evaluation framework into a multimodal,
knowledge-aware benchmark:

- **Voice evaluation** — evaluate real-time voice agents end-to-end, across
  every existing domain, with realistic audio conditions.
- **Knowledge retrieval domain** — a new `banking_knowledge` domain where agents
  must find relevant information in a large document corpus before acting.
- **Higher-quality tasks** — 75+ task corrections across airline, retail, and
  banking to improve evaluation reliability.
- **Better developer experience** — modular installation, richer CLI, layered
  runner API, and comprehensive documentation.

---

## 1) Voice Evaluation

Previously, τ-bench only evaluated agents via text. This release adds full
voice evaluation — agents receive audio and respond with audio in real time.

### What you can do now

- **Evaluate any domain in voice mode.** Airline, retail, telecom, and mock all
  work out of the box with voice. Each domain ships with voice-specific task
  metadata (difficulty ratings, voice task variants).
- **Run against multiple real-time voice providers.** A provider-agnostic
  adapter layer means the same evaluation works across different APIs:
  - Fully supported: **OpenAI Realtime**, **xAI Grok Voice**, **Gemini Live**
  - Experimental: **Nova Sonic**, **Qwen**, **Deepgram** (cascaded),
    **LiveKit** (cascaded)
- **Full-duplex conversations.** User and agent speak simultaneously — the
  simulated user can yield, interrupt, or wait, just like a real caller. This
  goes beyond simple turn-taking to test how agents handle overlapping speech,
  interruptions, and natural conversational dynamics.
- **Realistic audio conditions.** An audio effects pipeline adds background
  noise, burst sounds (car horns, dog barks), telephony compression, and
  frame drops. Evaluations can simulate real-world call-center conditions
  rather than clean studio audio.
- **Record and inspect audio.** An AudioTap system captures audio at each stage
  of the pipeline (user speech, effects applied, agent response) for
  debugging and qualitative analysis.
- **Catch user simulator errors.** A hallucination reviewer automatically
  detects when the simulated user deviates from its instructions and can
  re-run affected evaluations, improving result reliability.

### Example

```bash
tau2 run --domain retail \
  --audio-native \
  --audio-native-provider openai \
  --audio-native-model gpt-realtime-1.5 \
  --num-trials 1 \
  --num-tasks 5 \
  --audio-taps \
  --verbose-logs
```

---

## 2) Knowledge Retrieval + `banking_knowledge` Domain

Previously, all τ-bench domains gave agents a fixed policy document and a set of
tools. The agent's job was to follow the policy and use the tools correctly.

This release introduces a new evaluation dimension: **the agent must also find
the right information** before it can act.

### New domain: `banking_knowledge`

- **97 tasks** spanning account management, credit cards, disputes, transfers,
  and more.
- **698 policy and procedure documents** that the agent can search — but only a
  few are relevant to any given task.
- Combines transactional tools (look up accounts, process transfers) with
  knowledge retrieval, testing both capabilities together.

### Configurable retrieval strategies

The `--retrieval-config` flag controls how the agent accesses the knowledge
base, enabling apples-to-apples comparison of different retrieval approaches:

| Category | Configs | Notes |
|----------|---------|-------|
| Offline (no API keys) | `no_knowledge`, `full_kb`, `golden_retrieval`, `bm25`, `bm25_grep`, `grep_only` | Good for development and ablation |
| Embedding-backed | `openai_embeddings`, `qwen_embeddings` (+ `_reranker`, `_grep` variants) | Requires embedding API key |
| Agentic (sandboxed shell) | `terminal_use`, `terminal_use_write` | Agent searches documents via shell commands |

Embedding results are cached on disk to avoid recomputation across runs.

### Example

```bash
tau2 run --domain banking_knowledge \
  --retrieval-config bm25 \
  --agent-llm gpt-4.1 \
  --user-llm gpt-4.1 \
  --num-trials 1 \
  --num-tasks 5 \
  --verbose-logs
```

---

## 3) Task Quality

Evaluation results are only meaningful if the tasks themselves are correct.
This release includes broad corrections across all domains:

### Airline — 27 task fixes

- Removed incorrect expected actions (e.g., compensation for passengers not
  eligible under policy)
- Clarified ambiguous user instructions (e.g., "economy" vs "basic economy")
- Fixed impossible constraints (e.g., payment methods not in user profile)
- Closed policy loopholes (e.g., cancel-and-rebook workarounds)
- Added missing fallback behaviors and corrected passenger/date data

### Retail — 26 task fixes

- Removed invalid expected actions (e.g., PayPal refunds, which the system
  does not support)
- Clarified ambiguous instructions (e.g., "similar one" → "the same one")
- Fixed impossible same-item exchanges
- Added fallback behaviors for unavailable items
- Added `get_item_details` tool for product information retrieval

### Banking — 20+ task and document fixes

- Corrected required documents, expected actions, and reward calculations
  across ~20 tasks
- Cleaned up escaping issues across 155+ policy documents
- Ported missing tool validations to ensure tool behavior matches
  specification
- Fixed policy documents (fee rules, cooldown constraints, dispute
  eligibility criteria)

---

## 4) Developer Experience

### Installation

τ-bench now uses [uv](https://docs.astral.sh/uv/) and optional dependency
groups. Install only what you need:

```bash
uv sync                    # core text-mode evaluation
uv sync --extra voice      # + voice/audio-native
uv sync --extra knowledge  # + banking_knowledge domain
uv sync --extra gym        # + gymnasium RL interface
uv sync --extra dev        # + testing and linting
uv sync --all-extras       # everything
```

Python requirement is now `>=3.12, <3.14`.

### CLI

- **`tau2 intro`** — guided introduction to the framework and available
  domains.
- **`tau2 view`** — improved simulation viewer with richer output.
- **Timeout control** — `--timeout` flag for capping evaluation time.
- **Multiple results paths** — compare results across directories.

### Programmatic runner API

A new layered runner package (`tau2.runner`) replaces the previous monolithic
execution path. You can now use τ-bench programmatically at three levels:

```python
from tau2.runner import run_simulation          # low-level: run one orchestrator
from tau2.runner import build_text_orchestrator  # mid-level: build from config
from tau2.runner import run_domain              # high-level: full batch pipeline
```

Each layer adds capabilities (registry resolution, concurrency, checkpointing,
retries) without forcing you to use them all.

### Evaluation enhancements

- **LLM-based conversation review** — automated quality checks on completed
  conversations.
- **Hallucination detection** — identifies when the user simulator deviates
  from task instructions (especially important for voice evaluations).
- **Per-task summaries** — detailed per-task breakdowns for diagnosing
  evaluation patterns.

### Documentation

New guides:
- [Getting Started](docs/getting-started.md) — installation, setup, first run
- [CLI Reference](docs/cli-reference.md) — all commands and options
- [Knowledge Retrieval](src/tau2/knowledge/README.md) — retrieval pipeline
  setup and configuration
- [Audio Native Mode](src/tau2/voice/audio_native/README.md) — voice provider
  integration

Per-module READMEs and developer guides added throughout the codebase.

### Testing

53 new test files covering voice providers, banking tools, retrieval
pipelines, and full-duplex integration. Voice provider tests are gated behind
environment variables to avoid requiring live API access in CI.

---

## What's New (short version, for README)

- **Voice Evaluation** — Full-duplex voice with 7 real-time providers
  (OpenAI, xAI, Gemini, Nova, Qwen, Deepgram, LiveKit). Realistic audio
  effects, turn-taking simulation, and audio recording. Every existing
  domain works in voice mode.
- **Knowledge Domain** — New `banking_knowledge` domain (97 tasks, 698
  documents) with configurable retrieval: BM25, embeddings, rerankers, and
  sandboxed agentic search.
- **Task Quality** — 75+ task fixes across airline (27), retail (26), and
  banking (20+) covering incorrect actions, ambiguous instructions,
  impossible constraints, and policy loopholes.
- **Developer Experience** — `uv`-based install with optional extras, layered
  runner API, richer CLI, LLM-based evaluation review, and comprehensive
  documentation.

---

## Open Decisions

- Final version number (e.g., `0.3.0` vs `1.0.0`).
- Release title / tagline.
- Whether to ship a standalone `WHATS_NEW.md` or fold everything into
  `CHANGELOG.md` + `README.md`.
- Whether to include a migration guide for users of 0.2.x (installation
  method changed from pip to uv, some CLI flags renamed).
