#!/usr/bin/env python3
"""Hermetic integration checks for the token-cache benchmark.

This verifier needs neither model weights nor network access.  It injects a
small tokenizer double and exercises an OpenAI-compatible SSE server running on
localhost.  The checks intentionally focus on report correctness and on safety
properties that matter for long benchmark runs.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import re
import sys
import tempfile
import time
import types
from pathlib import Path
from typing import Any

from aiohttp import web

TARGET = Path(
    os.environ.get(
        "TOKEN_CACHE_BENCHMARK_TARGET",
        str(Path(__file__).resolve().parents[1] / "benchmark_token_cache_buckets.py"),
    )
)
TEMP_DIR = tempfile.TemporaryDirectory(prefix="token-cache-benchmark-test-")
ARTIFACTS = Path(TEMP_DIR.name)
API_KEY_ENV = "TOKEN_CACHE_BENCHMARK_TEST_API_KEY"
API_KEY = "unit-test-super-secret-value"
EXTRA_SENTINEL = "unit-test-extra-value-must-not-be-in-report"


class FakeTokenizer:
    """Enough of the Transformers tokenizer API for deterministic counts."""

    _token_pattern = re.compile(r"\w+|[^\w\s]", re.UNICODE)
    init_kwargs = {"revision": "fake-revision"}

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = True,
        add_generation_prompt: bool = True,
    ) -> list[int]:
        assert tokenize is True
        count = 0
        for message in messages:
            count += 3
            count += len(self._token_pattern.findall(message.get("content", "")))
        if add_generation_prompt:
            count += 2
        return list(range(count))

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return list(range(len(self._token_pattern.findall(text))))


class FakeAutoTokenizer:
    @classmethod
    def from_pretrained(cls, *_args: Any, **_kwargs: Any) -> FakeTokenizer:
        return FakeTokenizer()


def load_target() -> Any:
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = FakeAutoTokenizer
    sys.modules["transformers"] = fake_transformers
    spec = importlib.util.spec_from_file_location("benchmark_under_test", TARGET)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {TARGET}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class MockOpenAIServer:
    """OpenAI SSE double with auth, metrics, errors, and delayed socket close."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.metrics_requests: list[dict[str, Any]] = []
        self.request_count_by_mode: dict[str, int] = {}
        self.cache_hits = 10.0
        self.cache_queries = 20.0
        self.app = web.Application()
        self.app.router.add_get("/metrics-explicit", self.metrics)
        self.app.router.add_get("/{mode}/metrics", self.metrics)
        self.app.router.add_post("/{mode}/v1/chat/completions", self.chat)
        self.runner = web.AppRunner(self.app)
        self.site: web.TCPSite | None = None
        self.port: int | None = None

    async def __aenter__(self) -> "MockOpenAIServer":
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await self.site.start()
        sockets = self.site._server.sockets  # pylint: disable=protected-access
        self.port = sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        await self.runner.cleanup()

    def base_url(self, mode: str) -> str:
        assert self.port is not None
        return f"http://127.0.0.1:{self.port}/{mode}"

    def v1_url(self, mode: str) -> str:
        return f"{self.base_url(mode)}/v1"

    def chat_url(self, mode: str) -> str:
        return f"{self.v1_url(mode)}/chat/completions"

    def metrics_url(self) -> str:
        assert self.port is not None
        return f"http://127.0.0.1:{self.port}/metrics-explicit"

    async def metrics(self, request: web.Request) -> web.Response:
        self.metrics_requests.append(
            {
                "path": request.path,
                "authorization": request.headers.get("Authorization"),
            }
        )
        body = (
            f"vllm:prefix_cache_hits_total {self.cache_hits}\n"
            f"vllm:prefix_cache_queries_total {self.cache_queries}\n"
            "vllm:gpu_cache_usage_perc 0.25\n"
        )
        return web.Response(text=body)

    async def chat(self, request: web.Request) -> web.StreamResponse:
        mode = request.match_info["mode"]
        self.request_count_by_mode[mode] = self.request_count_by_mode.get(mode, 0) + 1
        payload = await request.json()
        self.requests.append(
            {
                "mode": mode,
                "path": request.path,
                "authorization": request.headers.get("Authorization"),
                "payload": payload,
            }
        )

        if mode == "unauthorized":
            return web.Response(status=401, text="bad test credential")
        if mode == "rate-limited":
            return web.Response(status=429, text="mock rate limit")
        if mode == "overloaded":
            return web.Response(status=503, text="mock overload")

        response = web.StreamResponse(
            status=200,
            headers={"Content-Type": "text/event-stream"},
        )
        await response.prepare(request)
        try:
            if mode in {"slow", "timeout"}:
                await asyncio.sleep(3.15)

            # The first generated token is private reasoning.  Visible content
            # begins later, and usage arrives in a choices=[] terminal chunk.
            lines = [
                'data: {"choices":[{"delta":{"role":"assistant"}}]}\n\n',
                "data: this-is-not-json\n\n",
                'data: {"choices":[{"delta":{"reasoning":"think deeply"}}]}\n\n',
            ]
            for line in lines:
                await response.write(line.encode("utf-8"))
                await asyncio.sleep(0.01)

            await asyncio.sleep(0.04)
            await response.write(
                b'data: {"choices":[{"delta":{"content":"Hello there"}}]}\n\n'
            )
            if mode == "early-eof":
                await response.write_eof()
                return response
            await asyncio.sleep(0.02)
            await response.write(
                b'data: {"choices":[{"delta":{"content":" world"}}]}\n\n'
            )
            self.cache_hits += 1
            self.cache_queries += 1
            await response.write(
                b'data: {"choices":[],"usage":{"completion_tokens":7}}\n\n'
            )
            await response.write(b"data: [DONE]\n\n")

            # A correct client returns at [DONE] rather than waiting for this
            # delayed socket close (or parsing the unexpected later event).
            await asyncio.sleep(1.0)
            await response.write(
                b'data: {"choices":[{"delta":{"content":" ignored"}}]}\n\n'
            )
            await response.write_eof()
        except (ConnectionResetError, asyncio.CancelledError):
            pass
        return response


def parse_args(
    module: Any,
    base_url: str,
    output: Path,
    *,
    timeout: float = 8.0,
    cache_mode: str = "shared_system",
    metrics_url: str | None = None,
    disable_metrics: bool = True,
    extra_argv: list[str] | None = None,
) -> argparse.Namespace:
    argv = [
        "--base-url",
        base_url,
        "--model-name",
        "mock-model",
        "--tokenizer-path",
        "fake-tokenizer-only",
        "--api-key-env",
        API_KEY_ENV,
        "--system-prompt-tokens",
        "2048",
        "--system-cache-modes",
        cache_mode,
        "--concurrency",
        "1",
        "--prompt-token-buckets",
        "2304",
        "--min-samples-per-bucket",
        "1",
        "--max-prompt-tokens",
        "2304",
        "--max-turns-per-user",
        "1",
        "--conversation-pool-size",
        "1",
        # Deliberately provide two ratios. shared_system and isolated_system
        # must collapse them to one effective cell.
        "--shared-system-ratio",
        "0.25,0.75",
        "--prompt-warmup",
        "0",
        "--exclude-first-turns",
        "0",
        "--timeout",
        str(timeout),
        "--temperature",
        "0.2",
        "--max-output-tokens",
        "16",
        "--seed",
        "41",
        "--network-label",
        "local-test",
        "--output",
        str(output),
    ]
    if disable_metrics:
        argv.append("--disable-metrics")
    if metrics_url is not None:
        argv.extend(["--metrics-url", metrics_url])
    if extra_argv:
        argv.extend(extra_argv)
    return module.build_parser().parse_args(argv)


def required(mapping: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    raise AssertionError(f"expected one of {names!r} in {sorted(mapping)}")


def source_and_helper_checks(module: Any) -> None:
    source = TARGET.read_text(encoding="utf-8")
    assert "asyncio.timeout(" not in source
    assert "--first-token-timeout" not in source
    assert "first_token_timeout_s" not in source

    normalize = getattr(module, "build_chat_completions_url", None)
    assert callable(normalize), "missing build_chat_completions_url(base_or_endpoint)"
    cases = {
        "https://example.test/service": (
            "https://example.test/service/v1/chat/completions"
        ),
        "https://example.test/service/": (
            "https://example.test/service/v1/chat/completions"
        ),
        "https://example.test/service/v1": (
            "https://example.test/service/v1/chat/completions"
        ),
        "https://example.test/service/v1/chat/completions": (
            "https://example.test/service/v1/chat/completions"
        ),
    }
    for supplied, expected in cases.items():
        assert normalize(supplied) == expected, (supplied, normalize(supplied))

    ratios = getattr(module, "effective_shared_system_ratios", None)
    assert callable(ratios), (
        "missing effective_shared_system_ratios(cache_mode, requested_ratios)"
    )
    requested = [0.25, 0.75]
    assert ratios("shared_system", requested) == [1.0]
    assert ratios("isolated_system", requested) == [0.0]
    assert ratios("mixed_system", requested) == requested

    cache = module.cache_summary(
        {
            "vllm:prefix_cache_hits_total": 10,
            "vllm:prefix_cache_queries_total": 20,
            "vllm:gpu_prefix_cache_hits_total": 100,
            "vllm:gpu_prefix_cache_queries_total": 200,
        },
        {
            "vllm:prefix_cache_hits_total": 11,
            "vllm:prefix_cache_queries_total": 22,
            "vllm:gpu_prefix_cache_hits_total": 150,
            "vllm:gpu_prefix_cache_queries_total": 201,
        },
    )
    assert cache["kv_cache_hits"] == 1
    assert cache["kv_cache_queries"] == 2
    assert cache["kv_cache_hit_rate"] == 0.5


async def run_and_checkpoint(
    module: Any, args: argparse.Namespace, output: Path
) -> dict[str, Any]:
    results = await module.run_benchmark(
        args,
        lambda current: module.write_results_checkpoint(
            output,
            current,
            args.show_first_turn_metrics,
        ),
    )
    return results


def assert_reasoning_record(record: dict[str, Any]) -> None:
    assert record["generated_ttft_ms"] >= 0
    assert record["content_ttft_ms"] > record["generated_ttft_ms"] + 20, record
    assert record["reasoning_tokens"] == 2, record
    assert record["content_tokens"] == 3, record
    assert record["completion_tokens"] == 7, record
    assert record["first_generated_chunk_tokens"] == 2, record
    assert record["first_content_chunk_tokens"] == 2, record
    assert record["token_count_source"] in {"server_usage", "usage"}, record
    assert record["reasoning_token_source"] == "local_tokenizer", record
    assert record["total_time_ms"] < 700, (
        "the client appears to have waited for the socket to close after [DONE]",
        record,
    )
    assert record["completion_tps"] is not None
    assert record["content_tps"] is not None
    expected_completion_tps = 5_000 / record["generation_time_ms"]
    expected_content_tps = 1_000 / record["content_generation_time_ms"]
    assert abs(record["completion_tps"] - expected_completion_tps) < 0.1, record
    assert abs(record["content_tps"] - expected_content_tps) < 0.1, record


async def disabled_metrics_auth_reasoning_check(
    module: Any, server: MockOpenAIServer
) -> None:
    output = ARTIFACTS / "disabled_metrics.json"
    request_extra = {
        "provider": {"order": [EXTRA_SENTINEL], "allow_fallbacks": False},
        "reasoning": {"enabled": False},
    }
    args = parse_args(
        module,
        server.v1_url("reasoning"),
        output,
        extra_argv=["--request-extra-json", json.dumps(request_extra)],
    )
    metrics_before = len(server.metrics_requests)
    started = time.perf_counter()
    results = await run_and_checkpoint(module, args, output)
    elapsed = time.perf_counter() - started

    assert elapsed < 0.75, "[DONE] did not terminate stream consumption immediately"
    assert len(results["summaries"]) == 1, "shared mode duplicated ratio cells"
    assert results["summaries"][0]["shared_system_ratio"] == 1.0
    assert len(results["records"]) == 1
    assert_reasoning_record(results["records"][0])
    assert len(server.metrics_requests) == metrics_before

    populated_buckets = [
        bucket
        for bucket in results["summaries"][0]["buckets"].values()
        if bucket["count"]
    ]
    assert len(populated_buckets) == 1
    bucket = populated_buckets[0]
    assert bucket["completion_tokens_mean"] == 7.0, bucket
    assert bucket["content_tokens_mean"] == 3.0, bucket
    assert bucket["reasoning_tokens_mean"] == 2.0, bucket
    assert bucket["completion_token_source_counts"] == {"server_usage": 1}, bucket
    assert bucket["reasoning_token_source_counts"] == {"local_tokenizer": 1}, bucket
    assert bucket["completion_tps_mean"] is not None, bucket
    assert bucket["content_tps_mean"] is not None, bucket

    kv = results["summaries"][0]["kv_cache"]
    assert kv["available"] is False
    assert kv.get("enabled", False) is False
    assert required(kv, "metrics_attempted_waves", "attempted_waves") == 0

    request = next(item for item in server.requests if item["mode"] == "reasoning")
    assert request["path"].endswith("/reasoning/v1/chat/completions")
    assert request["authorization"] == f"Bearer {API_KEY}"
    assert request["payload"]["stream_options"]["include_usage"] is True
    assert request["payload"]["max_tokens"] == 16
    assert request["payload"]["provider"] == request_extra["provider"]
    assert request["payload"]["reasoning"] == request_extra["reasoning"]

    config = results["config"]
    assert config["network_label"] == "local-test"
    assert config["seed"] == 41
    assert config["bearer_auth_enabled"] is True
    assert config["tokenizer_revision"] == "fake-revision"
    extra_keys = required(config, "request_extra_keys", "request_extra_json_keys")
    assert sorted(extra_keys) == ["provider", "reasoning"]

    serialized = json.dumps(results, ensure_ascii=False)
    assert API_KEY not in serialized
    assert EXTRA_SENTINEL not in serialized
    assert output.exists() and output.with_suffix(".log").exists()
    assert API_KEY not in output.read_text(encoding="utf-8")
    assert API_KEY not in output.with_suffix(".log").read_text(encoding="utf-8")
    log_text = output.with_suffix(".log").read_text(encoding="utf-8")
    for expected in (
        "Token Cache Benchmark",
        "Model=mock-model",
        "Network=local-test",
        "Tokenizer=fake-tokenizer-only",
        "Seed=41",
        "MaxOutputTokens=16",
        "Metrics=disabled",
    ):
        assert expected in log_text, expected


async def explicit_metrics_and_full_url_check(
    module: Any, server: MockOpenAIServer
) -> None:
    output = ARTIFACTS / "explicit_metrics.json"
    args = parse_args(
        module,
        server.chat_url("explicit"),
        output,
        cache_mode="isolated_system",
        metrics_url=server.metrics_url(),
        disable_metrics=False,
    )
    requests_before = len(server.metrics_requests)
    results = await run_and_checkpoint(module, args, output)
    new_metrics = server.metrics_requests[requests_before:]

    assert len(results["summaries"]) == 1, "isolated mode duplicated ratio cells"
    summary = results["summaries"][0]
    assert summary["shared_system_ratio"] == 0.0
    assert len(new_metrics) >= 2
    assert all(item["path"] == "/metrics-explicit" for item in new_metrics)

    kv = summary["kv_cache"]
    assert kv["available"] is True, kv
    assert kv.get("enabled", True) is True
    assert kv.get("scope") == "cell", kv
    attempted = required(kv, "metrics_attempted_waves", "attempted_waves")
    observed = required(kv, "metrics_observed_waves", "observed_waves")
    assert attempted >= 1 and observed == attempted
    assert kv["kv_cache_hits"] == 1.0, kv
    assert kv["kv_cache_queries"] == 1.0, kv
    assert kv["kv_cache_hit_rate"] == 1.0, kv

    # Global server counters cannot truthfully be attributed to prompt buckets.
    for bucket in summary["buckets"].values():
        assert not any(key.startswith("kv_cache") for key in bucket), bucket

    request = next(item for item in server.requests if item["mode"] == "explicit")
    assert request["path"] == "/explicit/v1/chat/completions"
    assert request["authorization"] == f"Bearer {API_KEY}"


async def reserved_request_field_check(module: Any, server: MockOpenAIServer) -> None:
    requests_before = len(server.requests)
    parse_extra = getattr(module, "parse_request_extra_json", None)
    assert callable(parse_extra), "missing parse_request_extra_json"
    try:
        parse_extra('{"model":"must-not-win"}')
    except argparse.ArgumentTypeError as exc:
        assert "model" in str(exc) and "reserved" in str(exc).lower(), exc
    else:
        raise AssertionError("request-extra JSON must not override reserved model")
    assert len(server.requests) == requests_before


async def mixed_ratio_check(module: Any, server: MockOpenAIServer) -> None:
    output = ARTIFACTS / "mixed_ratio.json"
    args = parse_args(
        module,
        server.v1_url("mixed"),
        output,
        cache_mode="mixed_system",
        extra_argv=[
            "--shared-system-ratio",
            "0.5",
            "--conversation-pool-size",
            "3",
        ],
    )
    results = await run_and_checkpoint(module, args, output)
    assert len(results["summaries"]) == 1
    summary = results["summaries"][0]
    assert summary["requested_shared_system_ratio"] == 0.5, summary
    assert summary["shared_user_count"] == 2, summary
    assert summary["conversation_pool_size"] == 3, summary
    assert abs(summary["actual_shared_system_ratio"] - (2 / 3)) < 1e-9, summary


async def failed_wave_circuit_breaker_check(
    module: Any, server: MockOpenAIServer
) -> None:
    output = ARTIFACTS / "failed_wave_breaker.json"
    args = parse_args(
        module,
        server.v1_url("overloaded"),
        output,
        extra_argv=[
            "--min-samples-per-bucket",
            "2",
            "--max-turns-per-user",
            "5",
            "--max-consecutive-failed-waves",
            "1",
        ],
    )
    results = await run_and_checkpoint(module, args, output)
    assert server.request_count_by_mode.get("overloaded") == 1
    summary = results["summaries"][0]
    assert summary["attempted_requests"] == 1
    assert summary["successful_requests"] == 0
    assert summary["failed_requests"] == 1
    assert any("HTTP 503" in error for error in summary["errors"]), summary
    assert any(
        "stopped after 1 consecutive all-failed waves" in warning
        for warning in summary["warnings"]
    ), summary
    log_text = output.with_suffix(".log").read_text(encoding="utf-8")
    assert "Warnings:" in log_text
    assert "stopped after 1 consecutive all-failed waves" in log_text
    assert "Errors:" in log_text and "HTTP 503" in log_text


async def early_eof_is_failure_check(module: Any, server: MockOpenAIServer) -> None:
    output = ARTIFACTS / "early_eof.json"
    args = parse_args(
        module,
        server.v1_url("early-eof"),
        output,
        extra_argv=["--max-consecutive-failed-waves", "1"],
    )
    results = await run_and_checkpoint(module, args, output)
    summary = results["summaries"][0]
    assert results["records"] == []
    assert summary["successful_requests"] == 0
    assert any(
        "stream ended before [DONE] or a finish_reason event" in error
        for error in summary["errors"]
    ), summary


async def strict_bucket_validation_check(module: Any, server: MockOpenAIServer) -> None:
    invalid_thresholds = (
        [2304, 2304],
        [2500, 2304],
        [2048, 2304],
    )
    requests_before = len(server.requests)
    for index, thresholds in enumerate(invalid_thresholds):
        output = ARTIFACTS / f"invalid_buckets_{index}.json"
        args = parse_args(module, server.v1_url("invalid"), output)
        args.prompt_token_buckets = thresholds
        try:
            await module.run_benchmark(args)
        except ValueError as exc:
            message = str(exc).lower()
            assert "bucket" in message and (
                "strict" in message
                or "ascending" in message
                or "greater than 2048" in message
            ), exc
        else:
            raise AssertionError(f"invalid bucket thresholds accepted: {thresholds}")
    assert len(server.requests) == requests_before


async def slow_first_token_check(module: Any, server: MockOpenAIServer) -> None:
    output = ARTIFACTS / "slow_first_token.json"
    args = parse_args(module, server.base_url("slow"), output)
    started = time.perf_counter()
    results = await run_and_checkpoint(module, args, output)
    elapsed = time.perf_counter() - started
    assert elapsed >= 3.1, elapsed
    assert len(results["records"]) == 1
    assert results["records"][0]["generated_ttft_ms"] >= 3000


async def total_timeout_check(module: Any, server: MockOpenAIServer) -> None:
    output = ARTIFACTS / "total_timeout.json"
    args = parse_args(
        module,
        server.base_url("timeout"),
        output,
        timeout=0.15,
    )
    started = time.perf_counter()
    results = await run_and_checkpoint(module, args, output)
    elapsed = time.perf_counter() - started
    summary = results["summaries"][0]
    assert elapsed < 1.0, elapsed
    assert summary["attempted_requests"] == 1
    assert summary["successful_requests"] == 0
    assert summary["failed_requests"] == 1
    assert summary["timeout_requests"] == 1
    assert "total request timeout" in summary["errors"][0]


async def fatal_http_check(module: Any, server: MockOpenAIServer) -> None:
    fatal_type = getattr(module, "FatalBenchmarkError", None)
    assert isinstance(fatal_type, type), "missing FatalBenchmarkError"
    output = ARTIFACTS / "fatal_http.json"
    args = parse_args(module, server.v1_url("unauthorized"), output)
    try:
        await run_and_checkpoint(module, args, output)
    except fatal_type as exc:
        assert "HTTP 401" in str(exc), exc
    else:
        raise AssertionError("HTTP 401 must abort the benchmark")
    assert server.request_count_by_mode.get("unauthorized") == 1


async def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    os.environ[API_KEY_ENV] = API_KEY
    module = load_target()
    source_and_helper_checks(module)
    try:
        async with MockOpenAIServer() as server:
            await disabled_metrics_auth_reasoning_check(module, server)
            await explicit_metrics_and_full_url_check(module, server)
            await reserved_request_field_check(module, server)
            await mixed_ratio_check(module, server)
            await failed_wave_circuit_breaker_check(module, server)
            await early_eof_is_failure_check(module, server)
            await strict_bucket_validation_check(module, server)
            await slow_first_token_check(module, server)
            await total_timeout_check(module, server)
            await fatal_http_check(module, server)
    finally:
        os.environ.pop(API_KEY_ENV, None)

    print("PASS URL normalization: root, /v1, and full endpoint")
    print("PASS authentication: Bearer header sent and secret absent from reports")
    print("PASS SSE: reasoning/content/usage counted and [DONE] returns immediately")
    print("PASS cache metrics: disabled mode and explicit cell-scoped endpoint")
    print("PASS request extras: merged safely and reserved fields rejected")
    print("PASS reproducibility config: network label and seed recorded")
    print("PASS cache modes: fixed ratios and mixed actual pool ratio reported")
    print("PASS circuit breaker: one all-failed HTTP 503 wave stops the cell")
    print("PASS early EOF: partial streams are failures, not completed responses")
    print("PASS bucket validation: thresholds are unique, increasing, and above 2k")
    print("PASS slow TTFT: responses beyond three seconds are measured")
    print("PASS total timeout: complete request deadline remains enforced")
    print("PASS fatal HTTP: authorization failure aborts after one request")


if __name__ == "__main__":
    asyncio.run(main())
