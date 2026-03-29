# Knowledge Retrieval

Domains with a knowledge base (currently just `banking_knowledge`) use a `--retrieval-config` flag that controls how the agent accesses the knowledge base.

```bash
tau2 run --domain banking_knowledge --retrieval-config <config_name> --agent-llm gpt-4.1 --user-llm gpt-4.1
```

If `--retrieval-config` is omitted, the default is `bm25` (offline, no API keys needed). A warning is printed to remind you to choose a config explicitly.

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

The `bm25`, `openai_embeddings`, and `qwen_embeddings` configs can also be combined with:
- `_reranker` suffix — adds an LLM reranker postprocessor (requires `OPENAI_API_KEY`)
- `_grep` suffix — adds a `grep` tool
- Both (e.g. `openai_embeddings_reranker_grep`)

Note: `*_reranker` variants always require `OPENAI_API_KEY` for the pointwise LLM reranker, even when the base embedder uses a different provider (e.g. `qwen_embeddings_reranker` needs both `OPENROUTER_API_KEY` and `OPENAI_API_KEY`).

## Embedding Cache

Embedding-based configs (`openai_embeddings*`, `qwen_embeddings*`) cache document embeddings on disk at `data/.embeddings_cache` (gitignored). This avoids re-computing embeddings on repeated runs. The cache is automatically invalidated when document content changes.

## Additional Setup

### OpenRouter API Key

The `qwen_embeddings*` configs route through [OpenRouter](https://openrouter.ai/). Set the `OPENROUTER_API_KEY` environment variable (or add it to your `.env` file — see `.env.example`).

### sandbox-runtime

The `terminal_use` and `terminal_use_write` configs require [Anthropic's sandbox-runtime](https://github.com/anthropic-experimental/sandbox-runtime) for secure filesystem isolation:

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
