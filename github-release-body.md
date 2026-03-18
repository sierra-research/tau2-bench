# τ-bench 1.0.0 — Voice, Knowledge, Task Quality

This release transforms τ-bench from a text-only agent evaluation framework into a multimodal, knowledge-aware benchmark. You can now evaluate voice agents in realistic conditions and test agents that must retrieve information from large document corpora.

## 🎙️ Voice Evaluation

**Evaluate any domain in voice mode.** All existing domains (airline, retail, telecom, mock) now support full voice evaluation out of the box.

**7 real-time voice providers** with a provider-agnostic adapter system:
- **Fully supported:** OpenAI Realtime, xAI Grok Voice, Gemini Live  
- **Experimental:** Nova Sonic, Qwen, Deepgram (cascaded), LiveKit (cascaded)

**Full-duplex conversations** where user and agent speak simultaneously. The simulated user can yield, interrupt, or wait — testing how agents handle overlapping speech and natural conversational dynamics beyond simple turn-taking.

**Realistic audio conditions** via an audio effects pipeline: background noise, burst sounds (car horns, dog barks), telephony compression, and frame drops. Simulate real call-center conditions rather than clean studio audio.

**Quality assurance** with automatic hallucination detection. The system identifies when the simulated user deviates from task instructions and can re-run affected evaluations.

```bash
# Evaluate a voice agent on retail tasks
tau2 run --domain retail \
  --audio-native \
  --audio-native-provider openai \
  --audio-native-model gpt-realtime-1.5 \
  --num-tasks 5 \
  --audio-taps
```

## 📚 Knowledge Retrieval + Banking Domain

**New evaluation paradigm:** Agents must now find relevant information in large document corpora before they can act, not just follow a fixed policy.

**New `banking_knowledge` domain:**
- **97 tasks** spanning account management, credit cards, disputes, transfers
- **698 policy and procedure documents** — only a few are relevant to any given task
- Tests both information retrieval and transactional tool use together

**12 configurable retrieval strategies** for apples-to-apples comparison:
- **Offline:** `no_knowledge`, `full_kb`, `golden_retrieval`, `bm25`, `bm25_grep`, `grep_only`
- **Embedding-backed:** `openai_embeddings`, `qwen_embeddings` (+ reranker/grep variants)  
- **Agentic:** `terminal_use`, `terminal_use_write` (sandboxed shell access)

```bash
# Evaluate knowledge retrieval with BM25
tau2 run --domain banking_knowledge \
  --retrieval-config bm25 \
  --agent-llm gpt-4.1 \
  --user-llm gpt-4.1 \
  --num-tasks 5
```

## 🎯 Task Quality (75+ fixes)

**Airline (27 tasks):** Removed incorrect expected actions, clarified ambiguous instructions, fixed impossible constraints, closed policy loopholes, added missing fallback behaviors.

**Retail (26 tasks):** Removed invalid expected actions (e.g., unsupported PayPal refunds), clarified ambiguous instructions, fixed impossible same-item exchanges, added fallback behaviors.

**Banking (20+ tasks):** Corrected required documents, expected actions, and reward calculations. Cleaned up escaping issues across 155+ policy documents. Ported missing tool validations.

## 🛠️ Developer Experience

**Simpler installation** with `uv` and optional dependency groups:
```bash
uv sync                    # core text-mode evaluation  
uv sync --extra voice      # + voice/audio-native
uv sync --extra knowledge  # + banking_knowledge domain
uv sync --all-extras       # everything
```

**Richer CLI:**
- `tau2 intro` — guided introduction to domains and capabilities
- `tau2 view` — improved simulation viewer  
- `--timeout` — evaluation time limits
- Multiple results comparison

**Programmatic API** with three levels of control:
```python
from tau2.runner import run_simulation          # low-level: run one orchestrator
from tau2.runner import build_text_orchestrator  # mid-level: build from config  
from tau2.runner import run_domain              # high-level: full batch pipeline
```

**Enhanced evaluation:**
- LLM-based conversation review and quality checks
- Hallucination detection for user simulator reliability  
- Per-task summary analysis and diagnostics

**Comprehensive documentation:**
- [Getting Started](docs/getting-started.md) — installation, setup, first run
- [CLI Reference](docs/cli-reference.md) — all commands and options
- [Knowledge Retrieval](src/tau2/knowledge/README.md) — retrieval pipeline setup
- [Audio Native Mode](src/tau2/voice/audio_native/README.md) — voice provider integration
- Per-module READMEs and developer guides throughout

## 🔧 Breaking Changes

- **Installation method:** Now uses `uv` instead of `pip install -e .`
- **Python requirement:** Now `>=3.12, <3.14` (was `>=3.10`)
- Some internal APIs refactored (affects custom agent implementations)

## 📊 By the Numbers

- **172 new source files** in `src/tau2/`
- **53 new test files** covering voice providers and knowledge retrieval
- **7 voice providers** with full-duplex support
- **97 banking tasks** with **698 documents** in the knowledge domain
- **75+ task fixes** across all domains
- **4 new documentation guides**

---

**Full changelog:** https://github.com/sierra-research/tau2-bench/blob/main/CHANGELOG.md  
**Installation guide:** https://github.com/sierra-research/tau2-bench/blob/main/docs/getting-started.md