# Model Serving Experiments

This directory contains self-contained experiments for serving, exercising, and
measuring language-model inference endpoints.

## Experiments

- [`token_cache_benchmark/`](token_cache_benchmark/) measures streaming latency,
  throughput, prompt-token buckets, and optional vLLM prefix-cache behavior for
  local or authenticated OpenAI-compatible endpoints. Reports include an explicit
  network label so direct and company-VPN runs are not accidentally compared as if
  they used the same path.

Each experiment owns its configuration, dependencies, documentation, and tests.
