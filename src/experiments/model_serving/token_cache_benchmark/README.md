# Token Cache Benchmark

This self-contained experiment measures latency, throughput, and optional vLLM
prefix-cache behavior across prompt-token buckets. It sends streaming,
OpenAI-compatible chat completion requests while maintaining realistic multi-turn
conversation histories. It can target either a local server or an authenticated
hosted endpoint such as OpenRouter.

The benchmark client needs the tokenizer and chat template that match the served
model. It does not load model weights. The model weights are owned by the inference
server configured through `BASE_URL`.

## What it measures

- generated-token TTFT mean, P90, and P95 (first reasoning or content delta)
- visible-content TTFT, reported separately when reasoning is streamed first
- end-to-end and post-first-generated-delta latency
- Request success rate, timeouts, and measured requests per second
- completion, reasoning, and visible-content token counts and throughput
- optional vLLM prefix-cache hit rate from a Prometheus metrics endpoint

Reasoning and content are deliberately separate. A reasoning-capable model may
start generating before it emits user-visible content, so generated-token TTFT and
content TTFT answer different questions. Completion throughput uses the server's
reported completion-token count when available and otherwise falls back to local
tokenization of the streamed reasoning and content. Post-first-delta throughput
subtracts all locally tokenized tokens delivered in the initial SSE delta; it does
not assume that the first network chunk contains exactly one token.
Each bucket also reports completion- and reasoning-token source counts so a result
that mixes server usage with tokenizer fallback is visible during comparison.

The benchmark uses closed-loop concurrency: it sends one wave of concurrent
requests, waits for the whole wave to finish, and then sends the next wave. It is
not a fixed-QPS load generator.

The default prompt thresholds extend through 28K tokens, including 20K, 24K, and
28K tiers. `MAX_PROMPT_TOKENS=32256` leaves room to finish sampling the final tier,
and the 320-turn safety cap lets closed-loop histories grow far enough to reach it.
Collection still stops as soon as every reachable bucket has enough samples.

## Setup

```bash
cd /path/to/tau2-bench
cp src/experiments/model_serving/token_cache_benchmark/.env.example .env
# Edit TOKENIZER_PATH, BASE_URL, and MODEL_NAME in the repository-level .env.
# For a hosted endpoint, also set its API key and usually DISABLE_METRICS=1.
cd src/experiments/model_serving/token_cache_benchmark
uv sync
```

`TOKENIZER_PATH` can point to a tokenizer-only directory, a complete local model
directory, or a Hugging Face repository ID. A tokenizer-only directory is enough;
the benchmark does not require local model weights. It must, however, match the
served model closely enough for prompt and fallback completion-token counts to be
meaningful.

`BASE_URL` accepts all of the following forms:

```text
http://127.0.0.1:9019
https://openrouter.ai/api/v1
https://example.test/v1/chat/completions
```

The client appends `/v1/chat/completions` to a server root, appends
`/chat/completions` to a URL ending in `/v1`, and leaves a complete
`.../chat/completions` URL unchanged.

For authentication, `API_KEY_ENV` names the environment variable that contains the
Bearer token. Its default is `BENCHMARK_API_KEY`. The secret value is used only to
construct the request header; it is never written to the report. Keep real secrets
in the ignored repository-level `.env` file or in the process environment, never
in `.env.example`. Set `ENV_FILE=/path/to/another.env` only when an explicit
override is required.

`REQUEST_EXTRA_JSON` can add provider-specific request fields. It must be a JSON
object. For example, OpenRouter users can pass `provider` preferences or disable
reasoning when they want visible-content-only latency. The report records only the
extra object's top-level field names, not their values.

When reasoning stays enabled, choose a large enough `MAX_OUTPUT_TOKENS` value for
the model to reach visible content. A reasoning-only response remains a valid
generated-token performance sample, but its content TTFT and content throughput are
reported as unavailable and that virtual conversation is not continued.

## Run

```bash
./run_benchmark.sh
```

Before a real run, choose the actual network path and record it with
`NETWORK_LABEL`, for example `direct` or `company-vpn`. This is report metadata;
the script does not connect or disconnect the VPN itself. Do not combine runs with
different network labels when attributing latency to the model server. The runner
prints a warning when the label is left as `unspecified`.

By default, the script writes a human-readable `.log` report and a detailed `.json`
report to:

```text
data/exp/token_cache_benchmark/<run-id>/
```

The text report begins with the safely redacted endpoint, model, tokenizer,
network label, seed, output limit, metrics state, and source commit. Cell warnings
and up to 20 request errors are included inline; full detail remains in JSON.

That directory is ignored by the repository. The settings listed in `.env.example`
can be overridden through the repository-level `.env` or environment variables.
Other parser options, such as `--seed` and `--conversation-pool-size`, can be
appended to `run_benchmark.sh`; appended command-line arguments take precedence.

Custom tokenizer code is disabled by default. Enable `TRUST_REMOTE_CODE=1` only for
a tokenizer repository you trust and only when the standard Transformers tokenizer
implementation cannot load it.

Set `METRICS_URL` when the vLLM Prometheus endpoint is not derived correctly from
`BASE_URL`. Set `DISABLE_METRICS=1` for hosted endpoints that do not expose a
dedicated metrics endpoint. Prefix-cache counters are process-global, not tagged by
request or token bucket, so the report presents them at cell scope only. The values
are trustworthy only when the benchmark has exclusive use of the inference server;
unrelated traffic during a cell changes the same counters. Metrics coverage reports
how much of the measured cell has a valid counter delta or snapshot pair. Chat
Bearer credentials are never forwarded to the metrics endpoint.

## Request shape

Each measured request is sent to the normalized chat-completions URL described
above. The payload contains:

```json
{
  "model": "<MODEL_NAME>",
  "messages": ["<full conversation history>"],
  "stream": true,
  "stream_options": {"include_usage": true},
  "temperature": 0.7,
  "max_tokens": 64
}
```

`max_tokens` is present when `--max-output-tokens` is set; direct CLI callers can
omit that option to use the server default. `stream_options` requests authoritative
usage counts when the endpoint supports them and can be replaced through
`REQUEST_EXTRA_JSON` for an incompatible provider. Extra fields cannot replace
`model`, `messages`, `stream`, `temperature`, or `max_tokens`. Each successful
visible assistant response is appended to that user's history and the complete
history is sent again on the next turn.

Cache modes have fixed, non-overlapping meanings:

- `shared_system`: every virtual user shares one exact system prompt; its effective
  shared ratio is always `1.0`.
- `isolated_system`: every virtual user has a unique system prefix; its effective
  shared ratio is always `0.0`.
- `mixed_system`: `SHARED_SYSTEM_RATIO` controls the fraction of the conversation
  pool using the shared prompt. Only this mode iterates multiple ratio values.

The report records both the requested ratio and the actual shared-user count and
ratio after rounding to the finite conversation pool.

## Verify without a model

The local verification script injects a deterministic fake tokenizer and starts a
mock streaming endpoint. It exercises URL normalization, authentication, reasoning
and content deltas, immediate termination on `[DONE]`, cache-metric coverage, cache
mode ratios, fatal HTTP errors, and the total request timeout. It also confirms that
a first generated token arriving after three seconds is measured rather than
cancelled.

```bash
uv run python tests/verify_benchmark.py
```
