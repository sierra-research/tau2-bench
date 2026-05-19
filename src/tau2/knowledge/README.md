# Knowledge Retrieval

Domains with a knowledge base (currently just `banking_knowledge`) use a `--retrieval-config` flag that controls how the agent accesses the knowledge base.

```bash
tau2 run --domain banking_knowledge --retrieval-config <config_name> --agent-llm gpt-4.1 --user-llm gpt-4.1
```

If `--retrieval-config` is omitted for `banking_knowledge`, the default is **`alltools`**: BM25 search, dense embedding search, and read-only shell (see below). Choose an offline-only config such as **`bm25`** if you want no API keys or sandbox.

### AllTools (`alltools`, `alltools-qwen`)

| Tool | Role |
|------|------|
| `KB_search_bm25` | BM25 sparse retrieval; pass **`k`** (default 10) for result count |
| `KB_search_dense` | Dense embeddings; pass **`k`** (default 10); backend selected by retrieval config |
| `shell` | Same read-only sandboxed shell as `terminal_use` |

Requirements: **sandbox-runtime** for `shell`, and an embedding API for dense search:

- **`alltools`**: uses OpenAI embeddings — set **`OPENAI_API_KEY`**. Model: **`text-embedding-3-large`**.
- **`alltools-qwen`**: uses OpenRouter/Qwen embeddings — set **`OPENROUTER_API_KEY`**. Model: **`qwen3-embedding-8b`**.

## Retrieval Configs

| Config | Tools | Requirements |
|--------|-------|--------------|
| `no_knowledge` | None | None (offline) |
| `full_kb` | None | None (offline) |
| `golden_retrieval` | None | None (offline) |
| `grep_only` | `grep` | None (offline) |
| `bm25` | `KB_search` | None (offline) |
| `openai_embeddings` | `KB_search` | `OPENAI_API_KEY` |
| `qwen_embeddings` | `KB_search` | `OPENROUTER_API_KEY` |
| `terminal_use` | `shell` | `sandbox-runtime` (see below) |
| `terminal_use_write` | `shell` | `sandbox-runtime` (see below) |
| `alltools` | `KB_search_bm25`, `KB_search_dense`, `shell` | BM25 offline + OpenAI dense embeddings + sandbox-runtime |
| `alltools-qwen` | `KB_search_bm25`, `KB_search_dense`, `shell` | BM25 offline + Qwen dense embeddings + sandbox-runtime |

The `bm25`, `openai_embeddings`, and `qwen_embeddings` configs can also be combined with:
- `_reranker` suffix — adds an LLM reranker postprocessor (requires `OPENAI_API_KEY`)
- `_grep` suffix — adds a `grep` tool
- Both (e.g. `openai_embeddings_reranker_grep`)

Note: `*_reranker` variants always require `OPENAI_API_KEY` for the pointwise LLM reranker, even when the base embedder uses a different provider (e.g. `qwen_embeddings_reranker` needs both `OPENROUTER_API_KEY` and `OPENAI_API_KEY`).

## Embedding Cache

Embedding-based configs (`openai_embeddings*`, `qwen_embeddings*`, `alltools`, `alltools-qwen`) cache document embeddings on disk at `data/.embeddings_cache` (gitignored). This avoids re-computing embeddings on repeated runs. The cache is automatically invalidated when document content changes.

## Additional Setup

### OpenRouter API Key

The `qwen_embeddings*` and `alltools-qwen` configs route through [OpenRouter](https://openrouter.ai/). Set the `OPENROUTER_API_KEY` environment variable (or add it to your `.env` file — see `.env.example`).

### sandbox-runtime

The `terminal_use`, `terminal_use_write`, `alltools`, and `alltools-qwen` configs require [Anthropic's sandbox-runtime](https://github.com/anthropic-experimental/sandbox-runtime) for secure filesystem isolation. **All of the following are required** — installing just the npm package is not sufficient.

```bash
npm install -g @anthropic-ai/sandbox-runtime@0.0.23
```

**macOS**: Also requires `ripgrep`:
```bash
brew install ripgrep
```

**Linux**: Also requires `ripgrep`, `bubblewrap`, and `socat`:
```bash
# Ubuntu/Debian
sudo apt-get install ripgrep bubblewrap socat
```

#### Verifying setup

To confirm the sandbox dependencies are installed:

```bash
which srt rg bwrap socat   # Linux
which srt rg               # macOS
```

All listed binaries must be present on `PATH`. As of **tau2 1.0.x**, `SandboxManager` will raise `SandboxRuntimeError` at construction time if any are missing, so a misconfigured machine fails loudly at the start of a run rather than silently passing "Sandbox dependencies are not available on this system" back to the agent for every shell tool call (which the agent will dutifully treat as a normal tool result and learn to give up on).

If you do not need the shell tool, use a retrieval config that doesn't require it (e.g., `--retrieval-config bm25`, `openai_embeddings`, or `qwen_embeddings`).
