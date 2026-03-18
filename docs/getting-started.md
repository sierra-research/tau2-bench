# Getting Started

This guide walks you through installing τ-bench, configuring API keys, and running your first evaluation.

## Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) — Python package and project manager
- Python 3.12+ (uv will download it automatically if not present)

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/sierra-research/tau2-bench
cd tau2-bench
```

### 2. Install τ-bench

```bash
uv sync                        # core only (text-mode: airline, retail, telecom, mock)
```

This creates a virtual environment, installs core dependencies from the lockfile, and enables the `tau2` command. The Python version is pinned via `.python-version` (3.12) — uv will download it automatically if needed.

#### Optional extras

Install the extras you need:

```bash
uv sync --extra voice          # + voice/audio-native features
uv sync --extra knowledge      # + banking_knowledge domain (retrieval pipeline)
uv sync --extra gym            # + gymnasium RL interface
uv sync --extra dev            # + pytest, ruff, pre-commit (required for contributing)
uv sync --extra experiments    # + plotting libs for src/experiments/
uv sync --all-extras           # everything
```

#### System dependencies (voice only)

If using voice features (`--extra voice`), install:

**macOS:**
```bash
brew install portaudio ffmpeg
```

> **Note:** If you install without `-e` mode (e.g., `uv pip install .`), you'll need to set the `TAU2_DATA_DIR` environment variable to point to your data directory:
> ```bash
> export TAU2_DATA_DIR=/path/to/your/tau2-bench/data
> ```

### 3. Verify your installation

```bash
uv run tau2 check-data
```

This checks that your data directory is correctly configured and all required files are present.

## Setting Up API Keys

We use [LiteLLM](https://github.com/BerriAI/litellm) to manage LLM APIs, so you can use any LLM provider supported by LiteLLM.

Copy `.env.example` as `.env` and edit it to include your API keys:

```bash
cp .env.example .env
```

### Voice API Keys (for voice-enabled features)

If you're using voice features, add the following to your `.env` file:
- `ELEVENLABS_API_KEY` — for voice synthesis
- `DEEPGRAM_API_KEY` — for voice transcription

## Running Your First Evaluation

### Standard text-based evaluation (half-duplex)

```bash
tau2 run --domain airline --agent-llm gpt-4.1 --user-llm gpt-4.1 \
  --num-trials 1 --num-tasks 5
```

Results are saved in `data/simulations/`.

### Audio native mode (voice full-duplex)

```bash
tau2 run --domain retail --audio-native --num-tasks 1 --verbose-logs
```

See the [Audio Native Documentation](../src/tau2/voice/audio_native/README.md) for provider configuration and all options.

### Knowledge retrieval evaluation

```bash
tau2 run --domain banking_knowledge --retrieval-config bm25 \
  --agent-llm gpt-4.1 --user-llm gpt-4.1 --num-tasks 5
```

See the [Knowledge Retrieval Documentation](../src/tau2/knowledge/README.md) for retrieval configuration options.


> **tip**: for full agent evaluation that matches the original τ-bench methodology, remove `--num-tasks` to evaluate on the complete task set (the `base` split is used by default).

## Simulation Output Structure

Each simulation run creates a directory. The standard text-based run produces:

```
data/simulations/<timestamp>_<domain>_<agent>_<user>/
└── results.json             # Simulation results and metrics
```

When using `--audio-native --verbose-logs`, the output includes additional data:

```
data/simulations/<timestamp>_<domain>_<agent>_<user>/
├── results.json                        # Simulation results and metrics
└── tasks/
    └── task_<task_id>/
        └── sim_<uuid>/
            ├── sim_status.json         # Simulation status
            ├── task.log                # Per-task log
            ├── audio/
            │   ├── both.wav            # Full conversation audio (stereo)
            │   ├── assistant_labels.txt # Audacity labels for agent speech
            │   ├── user_labels.txt     # Audacity labels for user speech
            │   ├── assistant_tool_calls_labels.txt  # Audacity labels for agent tool calls (when present)
            │   └── user_tool_calls_labels.txt       # Audacity labels for user tool calls (when present)
            └── llm_debug/
                └── *.json              # LLM call logs
```

## Viewing Results

```bash
tau2 view
```

This allows you to browse simulation files, view agent performance metrics, inspect individual simulations, and view task details. Works for both standard text and audio native runs.

## Configuration

The framework is configured via [`src/tau2/config.py`](../src/tau2/config.py).

### LLM Call Caching

LLM call caching is disabled by default. To enable it:

1. Install the `redis` Python package: `uv pip install redis`
2. Make sure a Redis server is running
3. Update the redis config in `config.py` if necessary
4. Set `LLM_CACHE_ENABLED` to `True` in `config.py`

## Cleanup

To remove all generated files and the virtual environment:

```bash
make clean
```

## Next Steps

- [CLI Reference](cli-reference.md) — all `tau2` commands and options
- [Agent Developer Guide](../src/tau2/agent/README.md) — build and evaluate your own agent
- [Domain Documentation](../src/tau2/domains/README.md) — understand the available domains
- [Communication Modes](../src/tau2/orchestrator/README.md) — half-duplex and full-duplex orchestration
- [Knowledge Retrieval](../src/tau2/knowledge/README.md) — retrieval pipeline setup and configuration for banking_knowledge domain
- [Voice (Full-Duplex)](../src/tau2/voice/README.md) — providers, speech complexity, and CLI options for voice evaluation
- [Gym/RL Interface](../src/tau2/gym/README.md) — Gymnasium-compatible environment for RL training
