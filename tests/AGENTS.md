# AGENTS.md — tests/

## Test Layout

```
tests/
├── conftest.py                 # Shared fixtures (mock domain, base tasks)
├── test_agent.py               # Core agent tests
├── test_environment.py         # Environment base class tests
├── test_orchestrator.py        # Orchestrator tests
├── test_run.py                 # End-to-end run tests
├── test_tasks.py               # Task loading/validation
├── test_user.py                # User simulator tests
├── test_domains/               # Per-domain tool tests
│   ├── test_<domain>/
│   └── test_banking_knowledge/ # Knowledge domain tests
│       ├── tasks/              # Per-task scenario tests (test_task_*.py)
│       ├── test_retrieval_system.py  # Retrieval pipeline tests
│       └── test_tools_knowledge.py   # Domain tool tests
├── test_streaming/             # Full-duplex / streaming tests
├── test_gym/                   # Gymnasium interface tests
└── test_voice/                 # Voice + audio native provider tests
    └── test_audio_native/
        └── test_<provider>/
```

## Running Tests

```bash
make test                         # All tests
pytest tests/test_domains/test_airline/   # Domain-specific
pytest tests/test_agent.py        # Single file
pytest -m "not full_duplex_integration"   # Skip live API tests
```

## Key Conventions

### Fixtures

Shared fixtures are in `conftest.py` and default to the `mock` domain:
- `domain_name` — returns `"mock"`
- `get_environment` — returns the mock environment constructor
- `base_task` — returns `create_task_1` from mock domain
- `task_with_*` — various task fixtures for different evaluation scenarios

Use the `mock` domain for unit tests. It's fast, has no external dependencies, and covers all evaluation criteria types.

### Test Markers

- `@pytest.mark.full_duplex_integration` — requires live LLM APIs; skipped by default in CI
- `@pytest.mark.skipif(not os.environ.get("{PROVIDER}_TEST_ENABLED"))` — audio native provider tests require `{PROVIDER}_TEST_ENABLED=1`

### Provider Test Pattern

Audio native provider tests in `tests/test_voice/test_audio_native/test_<provider>/`:
- Gated by environment variable: `{PROVIDER}_TEST_ENABLED=1`
- Use shared test audio from `tests/test_voice/test_audio_native/testdata/`
- Required test classes: `TestProviderConnection`, `TestProviderConfiguration`, `TestProviderAudioSend`, `TestProviderAudioReceive`, `TestProviderTranscription`, `TestProviderToolFlow`
- Run: `{PROVIDER}_TEST_ENABLED=1 pytest tests/test_voice/test_audio_native/test_<provider>/ -v`

### Domain Test Pattern

Domain tool tests in `tests/test_domains/test_<domain>/`:
- Test tools via `environment.get_response(ToolCall(...))`
- Test both success and failure cases (wrong IDs, invalid amounts, etc.)
- Use domain-specific fixtures for DB and environment setup

The `banking_knowledge` domain has an extended test structure:
- `test_tools_knowledge.py` — standard domain tool tests
- `test_retrieval_system.py` — tests for the retrieval pipeline (embeddings, BM25, grep, reranking)
- `test_retrieval_e2e.py` — end-to-end retrieval config tests with dependency-gated variants
- `tasks/test_task_*.py` — per-task scenario tests with shared fixtures in `tasks/conftest.py`

Retrieval e2e tests use skip markers to gate tests that require external dependencies:
- `requires_openai` — skips when `OPENAI_API_KEY` is not set (openai_embeddings variants)
- `requires_openrouter` — skips when `OPENROUTER_API_KEY` is not set (qwen_embeddings variants)
- `requires_sandbox_runtime` — skips when `srt` CLI is not installed (terminal_use variants)

### Asyncio

`asyncio_default_fixture_loop_scope = "function"` is set in `pyproject.toml`. Each async test gets a fresh event loop.

### What NOT to Do

- Do not add tests that require live API keys to the default test suite — gate them with markers or environment variables.
- Do not modify shared test audio files in `testdata/` without regenerating via `generate_test_audio.py`.
- Do not use `mock` domain tasks for testing domain-specific behavior — use the actual domain's fixtures.
