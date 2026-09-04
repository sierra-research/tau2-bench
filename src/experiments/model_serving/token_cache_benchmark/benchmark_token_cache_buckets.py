#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import random
import re
import statistics
import subprocess
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import aiohttp
from transformers import AutoTokenizer

DEFAULT_BUCKETS = "4096,6144,8192,10240,12288,16384,20480,24576,28672"
DEFAULT_SYSTEM_TOKENS = "2048,4096"
DEFAULT_CONCURRENCY = "8,16,32,48,64"
DEFAULT_OUTPUT = "token_cache_benchmark_results.json"
BUCKET_FLOOR = 2048
DEFAULT_API_KEY_ENV = "BENCHMARK_API_KEY"
RESERVED_REQUEST_FIELDS = {
    "model",
    "messages",
    "stream",
    "temperature",
    "max_tokens",
}

BUSINESS_POLICY = """
You are a professional customer service assistant for a ride-sharing company.
Your task is to resolve customer complaints through a multi-turn conversation.
Always answer politely and concisely. Try to keep each reply within 50 words.
Follow the process: greet the customer, ask for order information, verify identity,
confirm expense details, offer a voucher, offer a partial refund if needed, and
escalate to a human support specialist if the customer asks for further help.
Use function-call style text only when the user gives enough information to verify
an order or an expense. Otherwise ask one clear follow-up question.
"""

USER_TOPICS = [
    "The passenger reports that the driver took a longer route than expected.",
    "The passenger says the car arrived late and caused a missed appointment.",
    "The passenger asks why a cancellation fee was charged incorrectly.",
    "The passenger complains that the driver ended the trip at the wrong location.",
    "The passenger requests a refund because the car was not clean.",
    "The passenger says the app showed one price but the final charge was higher.",
    "The passenger wants to know whether a voucher can be used on the next trip.",
    "The passenger asks to connect with a human support specialist.",
]


@dataclass
class RequestRecord:
    system_tokens_target: int
    cache_mode: str
    concurrency: int
    shared_system_ratio: float
    uses_shared_system_prompt: bool
    virtual_user_id: str
    turn_index: int
    is_first_turn: bool
    bucket: str
    prompt_tokens: int
    generated_ttft_ms: float
    content_ttft_ms: float | None
    total_time_ms: float
    generation_time_ms: float
    content_generation_time_ms: float | None
    completion_tokens: int
    content_tokens: int
    reasoning_tokens: int
    first_generated_chunk_tokens: int
    first_content_chunk_tokens: int
    token_count_source: str
    reasoning_token_source: str
    timestamp: float


@dataclass
class ChatMeasurement:
    generated_ttft_ms: float
    content_ttft_ms: float | None
    total_time_ms: float
    content_text: str
    reasoning_text: str
    first_generated_content_text: str
    first_generated_reasoning_text: str
    first_content_text: str
    server_completion_tokens: int | None
    server_reasoning_tokens: int | None


@dataclass
class VirtualUser:
    user_id: str
    messages: list[dict[str, str]]
    uses_shared_system_prompt: bool
    turn_index: int = 0
    attempt_index: int = 0


@dataclass
class CellResult:
    system_tokens_target: int
    cache_mode: str
    concurrency: int
    shared_system_ratio: float = 0.0
    requested_shared_system_ratio: float = 0.0
    conversation_pool_size: int = 0
    shared_system_count: int = 0
    records: list[RequestRecord] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    attempted_requests: int = 0
    successful_requests: int = 0
    timeout_requests: int = 0
    cache_metrics_enabled: bool = True
    cache_metrics_available: bool = False
    cache_metrics_attempted_waves: int = 0
    cache_metrics_observed_waves: int = 0
    cache_metrics_counter_resets: int = 0
    cache_hits: float = 0.0
    cache_queries: float = 0.0
    measured_elapsed_s: float = 0.0
    elapsed_s: float = 0.0


class FatalBenchmarkError(RuntimeError):
    """An invalid request/configuration that should abort the benchmark."""


class TransientBenchmarkError(RuntimeError):
    """A server/network failure that a later wave may recover from."""


def parse_int_list(raw: str) -> list[int]:
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("value list cannot be empty")
    if any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("all values must be positive")
    return values


def parse_ratio_list(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("value list cannot be empty")
    if any(value < 0 or value > 1 for value in values):
        raise argparse.ArgumentTypeError("all ratios must be between 0 and 1")
    return values


def parse_request_extra_json(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(
            f"--request-extra-json must be valid JSON: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError("--request-extra-json must be a JSON object")
    reserved = sorted(RESERVED_REQUEST_FIELDS & value.keys())
    if reserved:
        raise argparse.ArgumentTypeError(
            "--request-extra-json cannot override reserved fields: "
            + ", ".join(reserved)
        )
    return value


def normalize_chat_completions_url(value: str) -> str:
    """Accept an API root, a /v1 root, or the full chat-completions URL."""
    parsed = urlsplit(value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("--base-url must be an absolute http(s) URL")
    path = parsed.path.rstrip("/")
    if path.endswith("/chat/completions"):
        normalized_path = path
    elif path.endswith("/v1"):
        normalized_path = f"{path}/chat/completions"
    else:
        normalized_path = f"{path}/v1/chat/completions"
    return urlunsplit(
        (parsed.scheme, parsed.netloc, normalized_path, parsed.query, parsed.fragment)
    )


def normalize_metrics_url(base_url: str, metrics_url: str | None) -> str:
    """Return an explicit metrics URL or derive the server-root /metrics URL."""
    if metrics_url:
        parsed = urlsplit(metrics_url.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("--metrics-url must be an absolute http(s) URL")
        return urlunsplit(
            (
                parsed.scheme,
                parsed.netloc,
                parsed.path.rstrip("/") or "/metrics",
                parsed.query,
                parsed.fragment,
            )
        )

    parsed = urlsplit(normalize_chat_completions_url(base_url))
    path = parsed.path
    for suffix in ("/v1/chat/completions", "/chat/completions"):
        if path.endswith(suffix):
            path = path[: -len(suffix)]
            break
    return urlunsplit(
        (parsed.scheme, parsed.netloc, f"{path.rstrip('/')}/metrics", "", "")
    )


def redact_url_for_report(value: str | None) -> str | None:
    if value is None:
        return None
    parsed = urlsplit(value)
    host = parsed.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    return urlunsplit((parsed.scheme, host, parsed.path, "", ""))


def effective_cache_ratios(cache_mode: str, requested: list[float]) -> list[float]:
    if cache_mode == "shared_system":
        return [1.0]
    if cache_mode == "isolated_system":
        return [0.0]
    return requested


def installed_version(distribution: str) -> str | None:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def source_git_commit() -> str | None:
    repository = Path(__file__).resolve().parents[4]
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            capture_output=True,
            check=False,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = completed.stdout.strip()
    return commit if completed.returncode == 0 and commit else None


def tokenizer_revision(tokenizer: Any) -> str | None:
    init_kwargs = getattr(tokenizer, "init_kwargs", {})
    if not isinstance(init_kwargs, dict):
        return None
    return init_kwargs.get("_commit_hash") or init_kwargs.get("revision")


def bucket_label(value: int, thresholds: list[int]) -> str:
    if value <= BUCKET_FLOOR:
        return f"<= {fmt_tokens(BUCKET_FLOOR)}"

    prev = BUCKET_FLOOR
    for threshold in thresholds:
        if value <= threshold:
            return f"{fmt_tokens(prev)}-{fmt_tokens(threshold)}"
        prev = threshold
    return f"> {fmt_tokens(thresholds[-1])}"


def bucket_labels(thresholds: list[int]) -> list[str]:
    labels = [bucket_label(BUCKET_FLOOR, thresholds)]
    prev = BUCKET_FLOOR
    for threshold in thresholds:
        labels.append(f"{fmt_tokens(prev)}-{fmt_tokens(threshold)}")
        prev = threshold
    labels.append(bucket_label(thresholds[-1] + 1, thresholds))
    return labels


def fmt_tokens(value: int) -> str:
    if value % 1024 == 0:
        return f"{value // 1024}k"
    if value >= 1000:
        return f"{value / 1000:.1f}k"
    return str(value)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * p
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def count_prompt_tokens(
    tokenizer: Any,
    messages: list[dict[str, str]],
    add_generation_prompt: bool = True,
) -> int:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    if isinstance(encoded, list):
        return len(encoded)
    if hasattr(encoded, "get") and encoded.get("input_ids") is not None:
        return len(encoded["input_ids"])
    raise TypeError(f"unexpected tokenizer output: {type(encoded)}")


def fit_message_content(
    tokenizer: Any,
    base_messages: list[dict[str, str]],
    role: str,
    seed_text: str,
    target_tokens: int,
    min_tokens: int = 0,
) -> str:
    """Build content so base_messages + candidate role roughly reaches target_tokens."""
    content = seed_text.strip() or "details"
    current_messages = base_messages + [{"role": role, "content": content}]
    while count_prompt_tokens(tokenizer, current_messages) < target_tokens:
        content = f"{content}\n{seed_text}"
        current_messages = base_messages + [{"role": role, "content": content}]

    low = 1
    high = len(content)
    best = content
    best_count = count_prompt_tokens(tokenizer, current_messages)
    lower_bound = min_tokens if min_tokens > 0 else 1

    while low <= high:
        mid = (low + high) // 2
        candidate = content[:mid].strip() or content[:1]
        candidate_messages = base_messages + [{"role": role, "content": candidate}]
        count = count_prompt_tokens(tokenizer, candidate_messages)
        if count <= target_tokens:
            if count >= lower_bound or best_count > target_tokens:
                best = candidate
                best_count = count
            low = mid + 1
        else:
            high = mid - 1

    return best


def build_system_prompt(
    tokenizer: Any,
    target_tokens: int,
    cache_mode: str,
    user_id: str | None = None,
) -> str:
    unique_prefix = ""
    if cache_mode == "isolated_system":
        unique_prefix = f"Session isolation marker: {user_id or 'system'}.\n"

    seed = (
        unique_prefix
        + BUSINESS_POLICY
        + "\nAdditional policy context:\n"
        + "\n".join(
            f"- Policy detail {idx}: {BUSINESS_POLICY.strip()}" for idx in range(1, 9)
        )
    )
    return fit_message_content(tokenizer, [], "system", seed, target_tokens)


def build_natural_user_message(
    virtual_user_id: str,
    turn_index: int,
    rng: random.Random,
) -> str:
    topic = rng.choice(USER_TOPICS)
    nonce = (
        f"user_id={virtual_user_id}; turn={turn_index}; "
        f"nonce={rng.getrandbits(64):016x}"
    )
    return (
        f"{nonce}\n"
        f"Customer message: {topic}\n"
        "The customer adds new ride details, timestamps, route notes, payment records, "
        "and follow-up questions. This is one natural conversation turn; do not repeat "
        "other users' text.\n"
    )


def build_chat_completions_url(value: str) -> str:
    """Compatibility-named wrapper used by tests and other experiment helpers."""
    return normalize_chat_completions_url(value)


def effective_shared_system_ratios(
    cache_mode: str, requested: list[float]
) -> list[float]:
    return effective_cache_ratios(cache_mode, requested)


def _stream_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_stream_text(item) for item in value)
    if isinstance(value, dict):
        for key in ("text", "content"):
            text = _stream_text(value.get(key))
            if text:
                return text
    return ""


def _reasoning_delta(delta: dict[str, Any]) -> str:
    # Providers often expose both a flattened reasoning field and structured
    # details. Prefer the flattened value to avoid double-counting the same text.
    for key in ("reasoning", "reasoning_content"):
        text = _stream_text(delta.get(key))
        if text:
            return text
    return _stream_text(delta.get("reasoning_details"))


def _tokens_from_usage(data: dict[str, Any]) -> tuple[int | None, int | None]:
    usage = data.get("usage")
    if not isinstance(usage, dict):
        return None, None
    value = usage.get("completion_tokens")
    completion_tokens = value if isinstance(value, int) and value >= 0 else None
    details = usage.get("completion_tokens_details")
    reasoning_tokens = None
    if isinstance(details, dict):
        value = details.get("reasoning_tokens")
        if isinstance(value, int) and value >= 0:
            reasoning_tokens = value
    return completion_tokens, reasoning_tokens


def _measurement_from_stream(
    *,
    t_start: float,
    t_end: float,
    t_first_generated: float | None,
    t_first_content: float | None,
    content_chunks: list[str],
    reasoning_chunks: list[str],
    first_generated_content_text: str,
    first_generated_reasoning_text: str,
    first_content_text: str,
    server_completion_tokens: int | None,
    server_reasoning_tokens: int | None,
) -> ChatMeasurement:
    if t_first_generated is None:
        raise TransientBenchmarkError(
            "stream ended without a content or reasoning token"
        )
    return ChatMeasurement(
        generated_ttft_ms=(t_first_generated - t_start) * 1000,
        content_ttft_ms=(
            (t_first_content - t_start) * 1000 if t_first_content is not None else None
        ),
        total_time_ms=(t_end - t_start) * 1000,
        content_text="".join(content_chunks).strip(),
        reasoning_text="".join(reasoning_chunks),
        first_generated_content_text=first_generated_content_text,
        first_generated_reasoning_text=first_generated_reasoning_text,
        first_content_text=first_content_text,
        server_completion_tokens=server_completion_tokens,
        server_reasoning_tokens=server_reasoning_tokens,
    )


async def measure_chat(
    session: aiohttp.ClientSession,
    base_url: str,
    messages: list[dict[str, str]],
    model_name: str,
    temperature: float,
    max_output_tokens: int | None,
    request_headers: dict[str, str] | None = None,
    request_extra: dict[str, Any] | None = None,
) -> ChatMeasurement:
    url = build_chat_completions_url(base_url)
    extra = request_extra or {}
    reserved = RESERVED_REQUEST_FIELDS & extra.keys()
    if reserved:
        raise FatalBenchmarkError(
            "request extras cannot override reserved fields: "
            + ", ".join(sorted(reserved))
        )
    payload: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "stream": True,
        "temperature": temperature,
        "stream_options": {"include_usage": True},
        **extra,
    }
    if max_output_tokens is not None:
        payload["max_tokens"] = max_output_tokens

    t_start = time.perf_counter()
    t_first_generated: float | None = None
    t_first_content: float | None = None
    content_chunks: list[str] = []
    reasoning_chunks: list[str] = []
    first_generated_content_text = ""
    first_generated_reasoning_text = ""
    first_content_text = ""
    server_completion_tokens: int | None = None
    server_reasoning_tokens: int | None = None
    finish_reason_seen = False

    async with session.post(url, json=payload, headers=request_headers) as resp:
        if 400 <= resp.status < 500:
            await resp.read()
            raise FatalBenchmarkError(
                f"HTTP {resp.status} {resp.reason or ''} from chat endpoint".strip()
            )
        if resp.status != 200:
            await resp.read()
            raise TransientBenchmarkError(
                f"HTTP {resp.status} {resp.reason or ''} from chat endpoint".strip()
            )

        # StreamReader iteration is line-oriented, preserving a partial SSE line
        # until its newline arrives. Returning on [DONE] avoids charging a server's
        # delayed socket close to generation latency.
        async for raw_line in resp.content:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload_str = line[5:].strip()
            if payload_str == "[DONE]":
                return _measurement_from_stream(
                    t_start=t_start,
                    t_end=time.perf_counter(),
                    t_first_generated=t_first_generated,
                    t_first_content=t_first_content,
                    content_chunks=content_chunks,
                    reasoning_chunks=reasoning_chunks,
                    first_generated_content_text=first_generated_content_text,
                    first_generated_reasoning_text=first_generated_reasoning_text,
                    first_content_text=first_content_text,
                    server_completion_tokens=server_completion_tokens,
                    server_reasoning_tokens=server_reasoning_tokens,
                )
            try:
                data = json.loads(payload_str)
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict):
                continue
            usage_completion, usage_reasoning = _tokens_from_usage(data)
            if usage_completion is not None:
                server_completion_tokens = usage_completion
            if usage_reasoning is not None:
                server_reasoning_tokens = usage_reasoning
            if data.get("error"):
                raise TransientBenchmarkError("stream returned a provider error")
            choices = data.get("choices", [])
            if not choices or not isinstance(choices[0], dict):
                continue
            if choices[0].get("finish_reason") is not None:
                finish_reason_seen = True
            delta = choices[0].get("delta", {})
            if not isinstance(delta, dict):
                continue
            content = _stream_text(delta.get("content"))
            reasoning = _reasoning_delta(delta)
            now = time.perf_counter()
            if (content or reasoning) and t_first_generated is None:
                t_first_generated = now
                first_generated_content_text = content
                first_generated_reasoning_text = reasoning
            if content:
                content_chunks.append(content)
                if t_first_content is None:
                    t_first_content = now
                    first_content_text = content
            if reasoning:
                reasoning_chunks.append(reasoning)

    if not finish_reason_seen:
        raise TransientBenchmarkError(
            "stream ended before [DONE] or a finish_reason event"
        )
    return _measurement_from_stream(
        t_start=t_start,
        t_end=time.perf_counter(),
        t_first_generated=t_first_generated,
        t_first_content=t_first_content,
        content_chunks=content_chunks,
        reasoning_chunks=reasoning_chunks,
        first_generated_content_text=first_generated_content_text,
        first_generated_reasoning_text=first_generated_reasoning_text,
        first_content_text=first_content_text,
        server_completion_tokens=server_completion_tokens,
        server_reasoning_tokens=server_reasoning_tokens,
    )


def cache_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    return {key: after[key] - before[key] for key in set(before) & set(after)}


def cache_summary(before: dict[str, float], after: dict[str, float]) -> dict[str, Any]:
    if not before or not after:
        return {"status": "unavailable", "available": False}
    delta = cache_delta(before, after)
    hits_suffix = "prefix_cache_hits_total"
    queries_suffix = "prefix_cache_queries_total"
    hit_families = {
        key[: -len(hits_suffix)]: key
        for key in sorted(delta)
        if key.endswith(hits_suffix) and "external" not in key.lower()
    }
    query_families = {
        key[: -len(queries_suffix)]: key
        for key in sorted(delta)
        if key.endswith(queries_suffix) and "external" not in key.lower()
    }
    common_families = hit_families.keys() & query_families.keys()
    if not common_families:
        return {"status": "unavailable", "available": False}
    family = min(common_families, key=lambda item: (len(item), item))
    hits_key = hit_families[family]
    queries_key = query_families[family]
    hits = delta[hits_key]
    queries = delta[queries_key]
    if hits < 0 or queries < 0:
        return {"status": "counter_reset", "available": False}
    return {
        "status": "available",
        "available": True,
        "kv_cache_hits": hits,
        "kv_cache_queries": queries,
        "kv_cache_hit_rate": hits / queries if queries > 0 else 0,
    }


async def natural_single_turn(
    session: aiohttp.ClientSession,
    user: VirtualUser,
    tokenizer: Any,
    base_url: str,
    model_name: str,
    system_tokens_target: int,
    cache_mode: str,
    concurrency: int,
    shared_system_ratio: float,
    thresholds: list[int],
    max_prompt_tokens: int,
    max_turns_per_user: int,
    result: CellResult,
    rng_seed: int,
    temperature: float,
    max_output_tokens: int | None,
    request_headers: dict[str, str] | None = None,
    request_extra: dict[str, Any] | None = None,
) -> RequestRecord | None:
    rng = random.Random(rng_seed)

    if user.attempt_index >= max_turns_per_user:
        return None

    is_first_turn = user.turn_index == 0
    next_turn_index = user.turn_index + 1
    user.attempt_index += 1
    user_content = build_natural_user_message(
        user.user_id,
        next_turn_index,
        rng,
    )
    request_messages = user.messages + [{"role": "user", "content": user_content}]
    prompt_tokens = count_prompt_tokens(tokenizer, request_messages)
    if prompt_tokens > max_prompt_tokens:
        user.attempt_index = max_turns_per_user
        return None

    request_bucket = bucket_label(prompt_tokens, thresholds)

    try:
        result.attempted_requests += 1
        measurement = await measure_chat(
            session,
            base_url,
            request_messages,
            model_name,
            temperature,
            max_output_tokens,
            request_headers=request_headers,
            request_extra=request_extra,
        )
        content_tokens = len(
            tokenizer.encode(measurement.content_text, add_special_tokens=False)
        )
        local_reasoning_tokens = len(
            tokenizer.encode(measurement.reasoning_text, add_special_tokens=False)
        )
        first_generated_chunk_tokens = len(
            tokenizer.encode(
                measurement.first_generated_reasoning_text,
                add_special_tokens=False,
            )
        ) + len(
            tokenizer.encode(
                measurement.first_generated_content_text,
                add_special_tokens=False,
            )
        )
        first_content_chunk_tokens = len(
            tokenizer.encode(
                measurement.first_content_text,
                add_special_tokens=False,
            )
        )
        reasoning_tokens = (
            measurement.server_reasoning_tokens
            if measurement.server_reasoning_tokens is not None
            else local_reasoning_tokens
        )
        reasoning_token_source = (
            "server_usage"
            if measurement.server_reasoning_tokens is not None
            else "local_tokenizer"
        )
        if measurement.server_completion_tokens is not None:
            completion_tokens = measurement.server_completion_tokens
            token_count_source = "server_usage"
        else:
            completion_tokens = content_tokens + reasoning_tokens
            token_count_source = "local_tokenizer"
    except asyncio.TimeoutError:
        result.timeout_requests += 1
        result.errors.append(
            f"{user.user_id} attempt={user.attempt_index}: total request timeout"
        )
        return None
    except FatalBenchmarkError:
        raise
    except Exception as exc:  # pylint: disable=broad-exception-caught
        result.errors.append(f"{user.user_id} attempt={user.attempt_index}: {exc}")
        return None

    result.successful_requests += 1
    user.turn_index = next_turn_index

    record = RequestRecord(
        system_tokens_target=system_tokens_target,
        cache_mode=cache_mode,
        concurrency=concurrency,
        shared_system_ratio=shared_system_ratio,
        uses_shared_system_prompt=user.uses_shared_system_prompt,
        virtual_user_id=user.user_id,
        turn_index=next_turn_index,
        is_first_turn=is_first_turn,
        bucket=request_bucket,
        prompt_tokens=prompt_tokens,
        generated_ttft_ms=measurement.generated_ttft_ms,
        content_ttft_ms=measurement.content_ttft_ms,
        total_time_ms=measurement.total_time_ms,
        generation_time_ms=max(
            measurement.total_time_ms - measurement.generated_ttft_ms, 0
        ),
        content_generation_time_ms=(
            max(measurement.total_time_ms - measurement.content_ttft_ms, 0)
            if measurement.content_ttft_ms is not None
            else None
        ),
        completion_tokens=completion_tokens,
        content_tokens=content_tokens,
        reasoning_tokens=reasoning_tokens,
        first_generated_chunk_tokens=first_generated_chunk_tokens,
        first_content_chunk_tokens=first_content_chunk_tokens,
        token_count_source=token_count_source,
        reasoning_token_source=reasoning_token_source,
        timestamp=time.time(),
    )
    if measurement.content_text:
        user.messages = request_messages + [
            {"role": "assistant", "content": measurement.content_text}
        ]
    else:
        # A reasoning-only response is still a valid latency/throughput sample,
        # but cannot seed the next visible assistant turn.
        user.attempt_index = max_turns_per_user

    return record


async def collect_cache_stats(metrics_url: str) -> dict[str, float]:
    connector = aiohttp.TCPConnector(limit=4)
    async with aiohttp.ClientSession(connector=connector) as session:
        return await fetch_vllm_metrics(session, metrics_url)


async def fetch_vllm_metrics(
    session: aiohttp.ClientSession,
    metrics_url: str,
) -> dict[str, float]:
    """Scrape vLLM /metrics for prefix-cache and cache usage counters."""
    try:
        async with session.get(
            metrics_url, timeout=aiohttp.ClientTimeout(total=5)
        ) as resp:
            if resp.status != 200:
                return {}
            text = await resp.text()
    except Exception:
        return {}

    result: dict[str, float] = {}
    for line in text.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([\w:]+)(?:\{[^}]*\})?\s+([\d.eE+-]+(?:nan|inf)?)$", line)
        if not match:
            continue
        name = match.group(1)
        try:
            val = float(match.group(2))
        except ValueError:
            continue
        key = name.lower()
        if any(
            keyword in key
            for keyword in (
                "prefix_cache",
                "gpu_cache_usage",
                "cpu_cache_usage",
                "num_requests_running",
                "num_requests_waiting",
            )
        ):
            result[name] = result.get(name, 0.0) + val
    return result


async def warmup_prompt_cache(
    tokenizer: Any,
    base_url: str,
    model_name: str,
    system_tokens_target: int,
    cache_mode: str,
    warmup_requests: int,
    timeout_s: float,
    temperature: float,
    max_output_tokens: int | None,
    request_headers: dict[str, str] | None = None,
    request_extra: dict[str, Any] | None = None,
) -> int:
    if warmup_requests <= 0:
        return 0

    connector = aiohttp.TCPConnector(limit=max(warmup_requests, 4))
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    system_prompt = build_system_prompt(
        tokenizer,
        system_tokens_target,
        cache_mode,
        user_id="prompt-warmup",
    )
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = []
        for idx in range(warmup_requests):
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"prompt_warmup={idx}; please respond briefly to warm up "
                        "the shared prompt cache."
                    ),
                },
            ]
            tasks.append(
                asyncio.create_task(
                    measure_chat(
                        session=session,
                        base_url=base_url,
                        messages=messages,
                        model_name=model_name,
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                        request_headers=request_headers,
                        request_extra=request_extra,
                    )
                )
            )
        responses = await asyncio.gather(*tasks, return_exceptions=True)
    for response in responses:
        if isinstance(response, FatalBenchmarkError):
            raise response
    return sum(isinstance(response, BaseException) for response in responses)


def target_bucket_labels(
    system_tokens_target: int,
    thresholds: list[int],
    max_prompt_tokens: int,
) -> set[str]:
    return {
        bucket_label(min(threshold, max_prompt_tokens), thresholds)
        for threshold in thresholds
        if min(threshold, max_prompt_tokens) > system_tokens_target
    }


async def run_cell(
    tokenizer: Any,
    system_tokens_target: int,
    cache_mode: str,
    concurrency: int,
    base_url: str,
    model_name: str,
    thresholds: list[int],
    min_samples_per_bucket: int,
    max_prompt_tokens: int,
    max_turns_per_user: int,
    prompt_warmup: int,
    conversation_pool_size: int,
    shared_system_ratio: float,
    timeout_s: float,
    seed: int,
    run_id: str,
    temperature: float,
    max_output_tokens: int | None,
    request_headers: dict[str, str] | None = None,
    request_extra: dict[str, Any] | None = None,
    metrics_url: str | None = None,
    disable_metrics: bool = False,
    max_consecutive_failed_waves: int = 3,
) -> CellResult:
    del run_id  # Run identity belongs in metadata, not deterministic prompt content.
    pool_size = max(conversation_pool_size, concurrency)
    if cache_mode == "shared_system":
        shared_count = pool_size
    elif cache_mode == "isolated_system":
        shared_count = 0
    else:
        shared_count = min(pool_size, max(0, round(pool_size * shared_system_ratio)))
    actual_shared_ratio = shared_count / pool_size
    target_labels = target_bucket_labels(
        system_tokens_target, thresholds, max_prompt_tokens
    )
    if not target_labels:
        raise ValueError(
            "no reachable measured prompt bucket: increase --max-prompt-tokens "
            "or add a threshold above the system prompt size"
        )

    resolved_metrics_url = None
    if not disable_metrics:
        resolved_metrics_url = metrics_url or normalize_metrics_url(base_url, None)
    result = CellResult(
        system_tokens_target=system_tokens_target,
        cache_mode=cache_mode,
        concurrency=concurrency,
        shared_system_ratio=actual_shared_ratio,
        requested_shared_system_ratio=shared_system_ratio,
        conversation_pool_size=pool_size,
        shared_system_count=shared_count,
        cache_metrics_enabled=not disable_metrics,
    )
    warmup_failures = await warmup_prompt_cache(
        tokenizer=tokenizer,
        base_url=base_url,
        model_name=model_name,
        system_tokens_target=system_tokens_target,
        cache_mode=cache_mode,
        warmup_requests=prompt_warmup,
        timeout_s=timeout_s,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        request_headers=request_headers,
        request_extra=request_extra,
    )
    if warmup_failures:
        result.warnings.append(
            f"{warmup_failures}/{prompt_warmup} prompt warmup requests failed"
        )

    users: list[VirtualUser] = []
    shared_system_prompt = None
    if cache_mode in {"shared_system", "mixed_system"}:
        shared_system_prompt = build_system_prompt(
            tokenizer, system_tokens_target, "shared_system"
        )

    rng = random.Random(seed)
    for idx in range(pool_size):
        user_id = (
            f"{cache_mode}-sys{system_tokens_target}-c{concurrency}-"
            f"r{actual_shared_ratio:.6f}-u{idx}-seed{seed}"
        )
        if idx < shared_count:
            system_prompt = shared_system_prompt
        else:
            system_prompt = build_system_prompt(
                tokenizer, system_tokens_target, "isolated_system", user_id
            )
        users.append(
            VirtualUser(
                user_id=user_id,
                messages=[{"role": "system", "content": system_prompt}],
                uses_shared_system_prompt=idx < shared_count,
            )
        )

    connector = aiohttp.TCPConnector(limit=max(concurrency * 2, 32))
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    completed_labels: set[str] = set()
    consecutive_failed_waves = 0
    started = time.perf_counter()
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        while any(
            sum(1 for record in result.records if record.bucket == label)
            < min_samples_per_bucket
            for label in target_labels
        ) and any(user.attempt_index < max_turns_per_user for user in users):
            available_users = [
                user for user in users if user.attempt_index < max_turns_per_user
            ]
            if not available_users:
                break

            selected_users = rng.sample(
                available_users,
                min(concurrency, len(available_users)),
            )
            wave_before: dict[str, float] = {}
            if resolved_metrics_url is not None:
                result.cache_metrics_attempted_waves += 1
                wave_before = await fetch_vllm_metrics(session, resolved_metrics_url)
            wave_started = time.perf_counter()
            tasks = [
                asyncio.create_task(
                    natural_single_turn(
                        session=session,
                        user=user,
                        tokenizer=tokenizer,
                        base_url=base_url,
                        model_name=model_name,
                        system_tokens_target=system_tokens_target,
                        cache_mode=cache_mode,
                        concurrency=concurrency,
                        shared_system_ratio=actual_shared_ratio,
                        thresholds=thresholds,
                        max_prompt_tokens=max_prompt_tokens,
                        max_turns_per_user=max_turns_per_user,
                        result=result,
                        rng_seed=seed + user.attempt_index + idx,
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                        request_headers=request_headers,
                        request_extra=request_extra,
                    )
                )
                for idx, user in enumerate(selected_users)
            ]
            if not tasks:
                break
            wave_results = await asyncio.gather(*tasks)
            wave_elapsed_s = time.perf_counter() - wave_started
            result.measured_elapsed_s += wave_elapsed_s
            if resolved_metrics_url is not None:
                wave_after = await fetch_vllm_metrics(session, resolved_metrics_url)
                wave_kv = cache_summary(wave_before, wave_after)
                if wave_kv.get("available"):
                    result.cache_metrics_available = True
                    result.cache_metrics_observed_waves += 1
                    result.cache_hits += wave_kv.get("kv_cache_hits", 0.0)
                    result.cache_queries += wave_kv.get("kv_cache_queries", 0.0)
                elif wave_kv.get("status") == "counter_reset":
                    result.cache_metrics_counter_resets += 1
            wave_records = [
                record for record in wave_results if isinstance(record, RequestRecord)
            ]
            if not wave_records:
                consecutive_failed_waves += 1
                if consecutive_failed_waves >= max_consecutive_failed_waves:
                    result.warnings.append(
                        "stopped after "
                        f"{consecutive_failed_waves} consecutive all-failed waves"
                    )
                    break
                continue
            consecutive_failed_waves = 0
            result.records.extend(wave_records)

            for label in bucket_labels(thresholds):
                if label not in target_labels or label in completed_labels:
                    continue
                sample_count = sum(
                    1 for record in result.records if record.bucket == label
                )
                if sample_count >= min_samples_per_bucket:
                    completed_labels.add(label)
                    print(
                        f"Completed bucket {label}: "
                        f"{sample_count}/{min_samples_per_bucket} samples",
                        flush=True,
                    )

        for label in sorted(target_labels):
            sample_count = sum(1 for record in result.records if record.bucket == label)
            if sample_count < min_samples_per_bucket:
                result.warnings.append(
                    f"incomplete bucket {label}: "
                    f"{sample_count}/{min_samples_per_bucket} samples"
                )
        if result.cache_metrics_enabled and not result.cache_metrics_available:
            result.warnings.append(
                "prefix-cache metrics unavailable from the configured metrics endpoint"
            )
        elif (
            result.cache_metrics_enabled
            and result.cache_metrics_observed_waves
            < result.cache_metrics_attempted_waves
        ):
            result.warnings.append(
                "prefix-cache metrics were available for only "
                f"{result.cache_metrics_observed_waves}/"
                f"{result.cache_metrics_attempted_waves} measured waves"
            )
        if result.cache_metrics_counter_resets:
            result.warnings.append(
                "ignored prefix-cache counter resets in "
                f"{result.cache_metrics_counter_resets} measured waves"
            )
    result.elapsed_s = time.perf_counter() - started
    return result


def summarize_bucket(
    records: list[RequestRecord],
    exclude_first_turns: int,
) -> dict[str, Any]:
    empty = {
        "count": 0,
        "first_turn_count": 0,
        "steady_count": 0,
        "prompt_tokens_mean": None,
        "prompt_tokens_min": None,
        "prompt_tokens_max": None,
        "generated_ttft_mean_ms": None,
        "generated_ttft_p90_ms": None,
        "generated_ttft_p95_ms": None,
        "content_ttft_mean_ms": None,
        "content_ttft_p90_ms": None,
        "content_ttft_p95_ms": None,
        "first_turn_generated_ttft_p95_ms": None,
        "steady_generated_ttft_mean_ms": None,
        "steady_generated_ttft_p90_ms": None,
        "steady_generated_ttft_p95_ms": None,
        "total_time_p95_ms": None,
        "generation_time_p95_ms": None,
        "completion_tokens_mean": None,
        "content_tokens_mean": None,
        "reasoning_tokens_mean": None,
        "first_generated_chunk_tokens_mean": None,
        "first_content_chunk_tokens_mean": None,
        "completion_token_source_counts": {},
        "reasoning_token_source_counts": {},
        "completion_tps_mean": None,
        "completion_tps_p95": None,
        "content_tps_mean": None,
        "content_tps_p95": None,
    }
    if not records:
        return empty

    prompt_tokens = [record.prompt_tokens for record in records]
    generated_ttft = [record.generated_ttft_ms for record in records]
    content_ttft = [
        record.content_ttft_ms
        for record in records
        if record.content_ttft_ms is not None
    ]
    first_turn_generated_ttft = [
        record.generated_ttft_ms for record in records if record.is_first_turn
    ]
    steady_generated_ttft = [
        record.generated_ttft_ms
        for record in records
        if record.turn_index > exclude_first_turns
    ]
    generation = [record.generation_time_ms for record in records]
    total = [record.total_time_ms for record in records]
    completion_tokens = [record.completion_tokens for record in records]
    content_tokens = [record.content_tokens for record in records]
    reasoning_tokens = [record.reasoning_tokens for record in records]
    completion_tps = [
        (record.completion_tokens - record.first_generated_chunk_tokens)
        * 1000
        / record.generation_time_ms
        for record in records
        if record.completion_tokens > record.first_generated_chunk_tokens
        and record.generation_time_ms > 0
    ]
    content_tps = [
        (record.content_tokens - record.first_content_chunk_tokens)
        * 1000
        / record.content_generation_time_ms
        for record in records
        if record.content_tokens > record.first_content_chunk_tokens
        and record.content_generation_time_ms is not None
        and record.content_generation_time_ms > 0
    ]
    return {
        "count": len(records),
        "first_turn_count": len(first_turn_generated_ttft),
        "steady_count": len(steady_generated_ttft),
        "prompt_tokens_mean": round(statistics.mean(prompt_tokens), 1),
        "prompt_tokens_min": min(prompt_tokens),
        "prompt_tokens_max": max(prompt_tokens),
        "generated_ttft_mean_ms": round(statistics.mean(generated_ttft), 1),
        "generated_ttft_p90_ms": round(percentile(generated_ttft, 0.90), 1),
        "generated_ttft_p95_ms": round(percentile(generated_ttft, 0.95), 1),
        "content_ttft_mean_ms": (
            round(statistics.mean(content_ttft), 1) if content_ttft else None
        ),
        "content_ttft_p90_ms": (
            round(percentile(content_ttft, 0.90), 1) if content_ttft else None
        ),
        "content_ttft_p95_ms": (
            round(percentile(content_ttft, 0.95), 1) if content_ttft else None
        ),
        "first_turn_generated_ttft_p95_ms": (
            round(percentile(first_turn_generated_ttft, 0.95), 1)
            if first_turn_generated_ttft
            else None
        ),
        "steady_generated_ttft_mean_ms": (
            round(statistics.mean(steady_generated_ttft), 1)
            if steady_generated_ttft
            else None
        ),
        "steady_generated_ttft_p90_ms": (
            round(percentile(steady_generated_ttft, 0.90), 1)
            if steady_generated_ttft
            else None
        ),
        "steady_generated_ttft_p95_ms": (
            round(percentile(steady_generated_ttft, 0.95), 1)
            if steady_generated_ttft
            else None
        ),
        "total_time_p95_ms": round(percentile(total, 0.95), 1),
        "generation_time_p95_ms": round(percentile(generation, 0.95), 1),
        "completion_tokens_mean": round(statistics.mean(completion_tokens), 1),
        "content_tokens_mean": round(statistics.mean(content_tokens), 1),
        "reasoning_tokens_mean": round(statistics.mean(reasoning_tokens), 1),
        "first_generated_chunk_tokens_mean": round(
            statistics.mean(record.first_generated_chunk_tokens for record in records),
            1,
        ),
        "first_content_chunk_tokens_mean": round(
            statistics.mean(record.first_content_chunk_tokens for record in records),
            1,
        ),
        "completion_token_source_counts": dict(
            sorted(Counter(record.token_count_source for record in records).items())
        ),
        "reasoning_token_source_counts": dict(
            sorted(Counter(record.reasoning_token_source for record in records).items())
        ),
        "completion_tps_mean": (
            round(statistics.mean(completion_tps), 2) if completion_tps else None
        ),
        "completion_tps_p95": (
            round(percentile(completion_tps, 0.95), 2) if completion_tps else None
        ),
        "content_tps_mean": (
            round(statistics.mean(content_tps), 2) if content_tps else None
        ),
        "content_tps_p95": (
            round(percentile(content_tps, 0.95), 2) if content_tps else None
        ),
    }


def summarize_cell(
    result: CellResult,
    thresholds: list[int],
    exclude_first_turns: int,
) -> dict[str, Any]:
    if not result.cache_metrics_enabled:
        kv_status = "disabled"
    elif not result.cache_metrics_observed_waves:
        kv_status = (
            "counter_reset" if result.cache_metrics_counter_resets else "unavailable"
        )
    elif result.cache_metrics_observed_waves < result.cache_metrics_attempted_waves:
        kv_status = "partial"
    else:
        kv_status = "available"
    kv = {
        "scope": "cell",
        "status": kv_status,
        "available": result.cache_metrics_available,
        "attempted_waves": result.cache_metrics_attempted_waves,
        "observed_waves": result.cache_metrics_observed_waves,
        "coverage": (
            result.cache_metrics_observed_waves / result.cache_metrics_attempted_waves
            if result.cache_metrics_attempted_waves
            else 0.0
        ),
        "counter_reset_waves": result.cache_metrics_counter_resets,
        "kv_cache_hits": result.cache_hits if result.cache_metrics_available else None,
        "kv_cache_queries": (
            result.cache_queries if result.cache_metrics_available else None
        ),
        "kv_cache_hit_rate": (
            result.cache_hits / result.cache_queries
            if result.cache_metrics_available and result.cache_queries > 0
            else None
        ),
        "caveat": (
            "Server-global counters are attributable only to this cell when the "
            "benchmark has exclusive use of the endpoint."
        ),
    }
    grouped: dict[str, list[RequestRecord]] = {
        label: [] for label in bucket_labels(thresholds)
    }
    for record in result.records:
        grouped.setdefault(record.bucket, []).append(record)
    shared_request_count = sum(
        record.uses_shared_system_prompt for record in result.records
    )
    return {
        "system_tokens_target": result.system_tokens_target,
        "cache_mode": result.cache_mode,
        "concurrency": result.concurrency,
        "shared_system_ratio": result.shared_system_ratio,
        "requested_shared_system_ratio": result.requested_shared_system_ratio,
        "actual_shared_system_ratio": result.shared_system_ratio,
        "conversation_pool_size": result.conversation_pool_size,
        "shared_user_count": result.shared_system_count,
        "shared_request_count": shared_request_count,
        "measured_shared_request_ratio": (
            shared_request_count / len(result.records) if result.records else None
        ),
        "elapsed_s": round(result.elapsed_s, 3),
        "measured_elapsed_s": round(result.measured_elapsed_s, 3),
        "attempted_requests": result.attempted_requests,
        "successful_requests": result.successful_requests,
        "recorded_requests": len(result.records),
        "failed_requests": result.attempted_requests - result.successful_requests,
        "timeout_requests": result.timeout_requests,
        "success_rate": (
            result.successful_requests / result.attempted_requests
            if result.attempted_requests
            else 0.0
        ),
        "request_throughput_rps": (
            len(result.records) / result.measured_elapsed_s
            if result.measured_elapsed_s > 0
            else 0.0
        ),
        "errors": result.errors,
        "warnings": result.warnings,
        "kv_cache": kv,
        "buckets": {
            label: summarize_bucket(
                grouped.get(label, []),
                exclude_first_turns,
            )
            for label in bucket_labels(thresholds)
        },
    }


def fmt_cell(value: Any, suffix: str = "") -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.1f}{suffix}"
    return f"{value}{suffix}"


def render_cell_table(summary: dict[str, Any], show_first_turn: bool) -> str:
    cache_available = summary["kv_cache"].get("available", False)
    cache_rate = summary["kv_cache"].get("kv_cache_hit_rate")
    cell_cache_rate = (
        f"{cache_rate:.2%}" if cache_available and cache_rate is not None else "N/A"
    )
    lines = [
        f"\nSystem={fmt_tokens(summary['system_tokens_target'])} | "
        f"CacheMode={summary['cache_mode']} | "
        f"CacheRatio={summary.get('actual_shared_system_ratio', 0):.1%} | "
        f"SharedUsers={summary.get('shared_user_count', 0)}/"
        f"{summary.get('conversation_pool_size', 0)} | "
        f"Concurrency={summary['concurrency']} | "
        f"Requests={summary['successful_requests']}/{summary['attempted_requests']} | "
        f"Success={summary['success_rate']:.2%} | "
        f"Timeouts={summary['timeout_requests']} | "
        f"Measured={summary['measured_elapsed_s']:.1f}s | "
        f"RPS={summary['request_throughput_rps']:.2f} | "
        f"CellKV_HitRate={cell_cache_rate} | "
        f"CellKV_Coverage={summary['kv_cache'].get('observed_waves', 0)}/"
        f"{summary['kv_cache'].get('attempted_waves', 0)} | "
        f"Errors={len(summary['errors'])} | "
        f"Warnings={len(summary['warnings'])}"
    ]
    headers = [
        "Bucket",
        "#Req",
        "PromptTok_mean",
        "PromptTok_min",
        "PromptTok_max",
        "GeneratedTTFT_P95",
        "ContentTTFT_P95",
        "SteadyGeneratedTTFT_P95",
        "TotalTime_P95",
        "GenerationTime_P95",
        "CompletionTok_mean",
        "ReasoningTok_mean",
        "CompletionTPS_mean",
        "ContentTPS_mean",
    ]
    if show_first_turn:
        headers[2:2] = ["#First"]
        headers[7:7] = ["FirstGeneratedTTFT_P95"]
    rows = []
    for label, bucket in summary["buckets"].items():
        row = [
            label,
            str(bucket["count"]),
            fmt_cell(bucket["prompt_tokens_mean"]),
            fmt_cell(bucket["prompt_tokens_min"]),
            fmt_cell(bucket["prompt_tokens_max"]),
            fmt_cell(bucket["generated_ttft_p95_ms"]),
            fmt_cell(bucket["content_ttft_p95_ms"]),
            fmt_cell(bucket["steady_generated_ttft_p95_ms"]),
            fmt_cell(bucket["total_time_p95_ms"]),
            fmt_cell(bucket["generation_time_p95_ms"]),
            fmt_cell(bucket["completion_tokens_mean"]),
            fmt_cell(bucket["reasoning_tokens_mean"]),
            fmt_cell(bucket["completion_tps_mean"]),
            fmt_cell(bucket["content_tps_mean"]),
        ]
        if show_first_turn:
            row[2:2] = [str(bucket["first_turn_count"])]
            row[7:7] = [fmt_cell(bucket["first_turn_generated_ttft_p95_ms"])]
        rows.append(row)

    widths = [
        max(len(headers[idx]), *(len(row[idx]) for row in rows))
        for idx in range(len(headers))
    ]
    lines.append(
        " | ".join(headers[idx].rjust(widths[idx]) for idx in range(len(headers)))
    )
    lines.append("-+-".join("-" * width for width in widths))
    for row in rows:
        lines.append(" | ".join(row[idx].rjust(widths[idx]) for idx in range(len(row))))
    return "\n".join(lines)


def print_cell_table(summary: dict[str, Any], show_first_turn: bool) -> None:
    print(render_cell_table(summary, show_first_turn))


def render_benchmark_log(results: dict[str, Any], show_first_turn: bool) -> str:
    config = results["config"]
    metrics_state = "enabled" if config.get("metrics_enabled") else "disabled"
    chunks = [
        "Token Cache Benchmark",
        (
            f"StartedUTC={config.get('started_at_utc', '-')} | "
            f"Model={config.get('model_name', '-')} | "
            f"Endpoint={config.get('chat_completions_url', '-')} | "
            f"Network={config.get('network_label', 'unspecified')}"
        ),
        (
            f"Tokenizer={config.get('tokenizer_path', '-')} | "
            f"TokenizerRevision={config.get('tokenizer_revision') or '-'} | "
            f"Seed={config.get('seed', '-')} | "
            f"MaxOutputTokens={config.get('max_output_tokens', '-')} | "
            f"Metrics={metrics_state} | "
            f"SourceCommit={config.get('source_git_commit') or '-'}"
        ),
    ]
    for summary in results["summaries"]:
        chunks.append(
            f"Running system={fmt_tokens(summary['system_tokens_target'])}, "
            f"cache_mode={summary['cache_mode']}, "
            f"cache_ratio={summary.get('shared_system_ratio', 0):.0%}, "
            f"concurrency={summary['concurrency']}"
        )
        chunks.append(render_cell_table(summary, show_first_turn).lstrip("\n"))
        for label, messages in (
            ("Warnings", summary["warnings"]),
            ("Errors", summary["errors"]),
        ):
            if not messages:
                continue
            shown = messages[:20]
            lines = [f"{label}:", *(f"- {message}" for message in shown)]
            if len(messages) > len(shown):
                lines.append(f"- ... {len(messages) - len(shown)} more in JSON report")
            chunks.append("\n".join(lines))
    return "\n\n".join(chunks) + "\n"


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(content, encoding="utf-8")
    temporary_path.replace(path)


def write_results_checkpoint(
    output_path: Path,
    results: dict[str, Any],
    show_first_turn: bool,
) -> None:
    atomic_write_text(
        output_path,
        json.dumps(results, ensure_ascii=False, indent=2),
    )
    atomic_write_text(
        output_path.with_suffix(".log"),
        render_benchmark_log(results, show_first_turn),
    )


def record_to_json(record: RequestRecord) -> dict[str, Any]:
    completion_tps = (
        (record.completion_tokens - record.first_generated_chunk_tokens)
        * 1000
        / record.generation_time_ms
        if record.completion_tokens > record.first_generated_chunk_tokens
        and record.generation_time_ms > 0
        else None
    )
    content_tps = (
        (record.content_tokens - record.first_content_chunk_tokens)
        * 1000
        / record.content_generation_time_ms
        if record.content_tokens > record.first_content_chunk_tokens
        and record.content_generation_time_ms is not None
        and record.content_generation_time_ms > 0
        else None
    )
    return {
        "system_tokens_target": record.system_tokens_target,
        "cache_mode": record.cache_mode,
        "concurrency": record.concurrency,
        "shared_system_ratio": record.shared_system_ratio,
        "uses_shared_system_prompt": record.uses_shared_system_prompt,
        "virtual_user_id": record.virtual_user_id,
        "turn_index": record.turn_index,
        "is_first_turn": record.is_first_turn,
        "bucket": record.bucket,
        "prompt_tokens": record.prompt_tokens,
        "generated_ttft_ms": round(record.generated_ttft_ms, 3),
        "content_ttft_ms": (
            round(record.content_ttft_ms, 3)
            if record.content_ttft_ms is not None
            else None
        ),
        "total_time_ms": round(record.total_time_ms, 3),
        "generation_time_ms": round(record.generation_time_ms, 3),
        "content_generation_time_ms": (
            round(record.content_generation_time_ms, 3)
            if record.content_generation_time_ms is not None
            else None
        ),
        "completion_tokens": record.completion_tokens,
        "content_tokens": record.content_tokens,
        "reasoning_tokens": record.reasoning_tokens,
        "first_generated_chunk_tokens": record.first_generated_chunk_tokens,
        "first_content_chunk_tokens": record.first_content_chunk_tokens,
        "token_count_source": record.token_count_source,
        "reasoning_token_source": record.reasoning_token_source,
        "completion_tps": (
            round(completion_tps, 3) if completion_tps is not None else None
        ),
        "content_tps": round(content_tps, 3) if content_tps is not None else None,
        "timestamp": record.timestamp,
    }


def validate_args(args: argparse.Namespace) -> None:
    thresholds = args.prompt_token_buckets
    if len(set(thresholds)) != len(thresholds) or thresholds != sorted(thresholds):
        raise ValueError("--prompt-token-buckets must be strictly increasing")
    if any(threshold <= BUCKET_FLOOR for threshold in thresholds):
        raise ValueError(
            f"--prompt-token-buckets must all be greater than {BUCKET_FLOOR}"
        )
    if args.min_samples_per_bucket <= 0:
        raise ValueError("--min-samples-per-bucket must be positive")
    if args.max_prompt_tokens <= 0:
        raise ValueError("--max-prompt-tokens must be positive")
    for system_tokens in args.system_prompt_tokens:
        if not target_bucket_labels(system_tokens, thresholds, args.max_prompt_tokens):
            raise ValueError(
                "no reachable measured prompt bucket for system prompt target "
                f"{system_tokens}"
            )
    if args.prompt_warmup < 0:
        raise ValueError("--prompt-warmup cannot be negative")
    if args.exclude_first_turns < 0:
        raise ValueError("--exclude-first-turns cannot be negative")
    if args.conversation_pool_size < 0:
        raise ValueError("--conversation-pool-size cannot be negative")
    if args.conversation_pool_multiplier <= 0:
        raise ValueError("--conversation-pool-multiplier must be positive")
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")
    if args.max_output_tokens is not None and args.max_output_tokens <= 0:
        raise ValueError("--max-output-tokens must be positive")
    if args.max_consecutive_failed_waves <= 0:
        raise ValueError("--max-consecutive-failed-waves must be positive")
    build_chat_completions_url(args.base_url)
    if not args.disable_metrics:
        normalize_metrics_url(args.base_url, args.metrics_url)
    reserved = RESERVED_REQUEST_FIELDS & args.request_extra_json.keys()
    if reserved:
        raise ValueError(
            "--request-extra-json cannot override reserved fields: "
            + ", ".join(sorted(reserved))
        )


async def run_benchmark(
    args: argparse.Namespace,
    on_checkpoint: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    validate_args(args)
    chat_url = build_chat_completions_url(args.base_url)
    resolved_metrics_url = (
        None
        if args.disable_metrics
        else normalize_metrics_url(args.base_url, args.metrics_url)
    )
    api_key = os.environ.get(args.api_key_env) if args.api_key_env else None
    api_key = api_key or None
    request_headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        trust_remote_code=args.trust_remote_code,
    )
    thresholds = args.prompt_token_buckets
    run_id = str(time.time_ns())
    all_summaries: list[dict[str, Any]] = []
    all_records: list[dict[str, Any]] = []
    results = {
        "config": {
            "base_url": redact_url_for_report(args.base_url),
            "chat_completions_url": redact_url_for_report(chat_url),
            "metrics_url": redact_url_for_report(resolved_metrics_url),
            "metrics_enabled": not args.disable_metrics,
            "api_key_env": args.api_key_env,
            "bearer_auth_enabled": bool(api_key),
            "request_extra_keys": sorted(args.request_extra_json),
            "network_label": args.network_label,
            "model_name": args.model_name,
            "tokenizer_path": args.tokenizer_path,
            "trust_remote_code": args.trust_remote_code,
            "system_prompt_tokens": args.system_prompt_tokens,
            "system_cache_modes": args.system_cache_modes,
            "concurrency": args.concurrency,
            "prompt_token_buckets": thresholds,
            "min_samples_per_bucket": args.min_samples_per_bucket,
            "max_prompt_tokens": args.max_prompt_tokens,
            "max_turns_per_user": args.max_turns_per_user,
            "prompt_warmup": args.prompt_warmup,
            "exclude_first_turns": args.exclude_first_turns,
            "show_first_turn_metrics": args.show_first_turn_metrics,
            "conversation_pool_size": args.conversation_pool_size,
            "conversation_pool_multiplier": args.conversation_pool_multiplier,
            "shared_system_ratio": args.shared_system_ratio,
            "run_id": run_id,
            "timeout": args.timeout,
            "temperature": args.temperature,
            "max_output_tokens": args.max_output_tokens,
            "max_consecutive_failed_waves": args.max_consecutive_failed_waves,
            "seed": args.seed,
            "started_at_utc": datetime.now(UTC).isoformat(),
            "source_git_commit": source_git_commit(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "package_versions": {
                "aiohttp": installed_version("aiohttp"),
                "transformers": installed_version("transformers"),
            },
            "tokenizer_revision": tokenizer_revision(tokenizer),
        },
        "summaries": all_summaries,
        "records": all_records,
    }

    for cache_mode in args.system_cache_modes:
        for system_tokens in args.system_prompt_tokens:
            for shared_system_ratio in effective_shared_system_ratios(
                cache_mode, args.shared_system_ratio
            ):
                for concurrency in args.concurrency:
                    print(
                        f"\nRunning system={fmt_tokens(system_tokens)}, "
                        f"cache_mode={cache_mode}, "
                        f"cache_ratio={shared_system_ratio:.0%}, "
                        f"concurrency={concurrency}"
                    )
                    result = await run_cell(
                        tokenizer=tokenizer,
                        system_tokens_target=system_tokens,
                        cache_mode=cache_mode,
                        concurrency=concurrency,
                        base_url=chat_url,
                        model_name=args.model_name,
                        thresholds=thresholds,
                        min_samples_per_bucket=args.min_samples_per_bucket,
                        max_prompt_tokens=args.max_prompt_tokens,
                        max_turns_per_user=args.max_turns_per_user,
                        prompt_warmup=args.prompt_warmup,
                        conversation_pool_size=max(
                            args.conversation_pool_size,
                            concurrency * args.conversation_pool_multiplier,
                        ),
                        shared_system_ratio=shared_system_ratio,
                        timeout_s=args.timeout,
                        seed=(
                            args.seed
                            + system_tokens
                            + concurrency
                            + int(round(shared_system_ratio * 100))
                        ),
                        run_id=run_id,
                        temperature=args.temperature,
                        max_output_tokens=args.max_output_tokens,
                        request_headers=request_headers,
                        request_extra=args.request_extra_json,
                        metrics_url=resolved_metrics_url,
                        disable_metrics=args.disable_metrics,
                        max_consecutive_failed_waves=(
                            args.max_consecutive_failed_waves
                        ),
                    )
                    summary = summarize_cell(
                        result,
                        thresholds,
                        args.exclude_first_turns,
                    )
                    all_summaries.append(summary)
                    all_records.extend(
                        record_to_json(record) for record in result.records
                    )
                    if on_checkpoint is not None:
                        on_checkpoint(results)
                    print_cell_table(summary, args.show_first_turn_metrics)

    return results


def parse_cache_modes(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    valid = {"shared_system", "isolated_system", "mixed_system"}
    unknown = [value for value in values if value not in valid]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown cache modes: {', '.join(unknown)}")
    if not values:
        raise argparse.ArgumentTypeError("cache modes cannot be empty")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Token-bucket benchmark with virtual multi-turn users and controlled "
            "cache sharing."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:9019")
    parser.add_argument("--model-name", default="model")
    parser.add_argument(
        "--api-key-env",
        default=DEFAULT_API_KEY_ENV,
        help=(
            "Environment variable containing the Bearer token. The value is never "
            "written to reports."
        ),
    )
    parser.add_argument(
        "--request-extra-json",
        type=parse_request_extra_json,
        default={},
        help=(
            "Extra JSON object merged into each request. Reserved benchmark fields "
            "cannot be overridden; reports store key names only."
        ),
    )
    parser.add_argument(
        "--metrics-url",
        default=None,
        help="Full metrics endpoint URL. Omit to derive <server-root>/metrics.",
    )
    parser.add_argument(
        "--disable-metrics",
        action="store_true",
        help="Do not query server-global vLLM metrics.",
    )
    parser.add_argument(
        "--network-label",
        default="unspecified",
        help="Report-only label such as direct, local, or company-vpn.",
    )
    parser.add_argument(
        "--tokenizer-path",
        required=True,
        help="Local tokenizer-only directory or Hugging Face model repository ID.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow a tokenizer repository to execute its custom Python code.",
    )
    parser.add_argument(
        "--system-prompt-tokens",
        type=parse_int_list,
        default=parse_int_list(DEFAULT_SYSTEM_TOKENS),
    )
    parser.add_argument(
        "--system-cache-modes",
        type=parse_cache_modes,
        default=parse_cache_modes("shared_system"),
        help="Comma-separated cache modes: shared_system,isolated_system,mixed_system",
    )
    parser.add_argument(
        "--concurrency",
        type=parse_int_list,
        default=parse_int_list(DEFAULT_CONCURRENCY),
    )
    parser.add_argument(
        "--prompt-token-buckets",
        type=parse_int_list,
        default=parse_int_list(DEFAULT_BUCKETS),
        help="Comma-separated ascending bucket thresholds.",
    )
    parser.add_argument("--min-samples-per-bucket", type=int, default=20)
    parser.add_argument("--max-prompt-tokens", type=int, default=32256)
    parser.add_argument(
        "--max-turns-per-user",
        type=int,
        default=320,
        help="Maximum request attempts per virtual user.",
    )
    parser.add_argument(
        "--conversation-pool-size",
        type=int,
        default=0,
        help=(
            "Fixed number of maintained conversations. 0 means use concurrency "
            "* multiplier."
        ),
    )
    parser.add_argument(
        "--conversation-pool-multiplier",
        type=int,
        default=1,
        help="Conversation pool size multiplier when --conversation-pool-size is 0.",
    )
    parser.add_argument(
        "--shared-system-ratio",
        type=parse_ratio_list,
        default=parse_ratio_list("0.7"),
        help=(
            "Comma-separated cache-sharing ratios for mixed_system mode. "
            "Each value is the fraction of the conversation pool that shares one "
            "system prompt (0=all isolated, 1=all shared)."
        ),
    )
    parser.add_argument(
        "--prompt-warmup",
        type=int,
        default=1,
        help=(
            "Warmup requests before each measured cell. These requests are not "
            "included in metrics."
        ),
    )
    parser.add_argument(
        "--exclude-first-turns",
        type=int,
        default=1,
        help="Exclude the first N turns of each conversation from SteadyTTFT metrics.",
    )
    parser.add_argument(
        "--show-first-turn-metrics",
        action="store_true",
        help="Show #First and FirstTTFT_P95 columns in the console table.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=900,
        help="Total timeout in seconds for one complete streaming request.",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="Maximum generated tokens per request. Omit to use the server default.",
    )
    parser.add_argument(
        "--max-consecutive-failed-waves",
        type=int,
        default=3,
        help="Stop a cell after this many consecutive waves with no successes.",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260601)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    log_path = output_path.with_suffix(".log")
    asyncio.run(
        run_benchmark(
            args,
            lambda results: write_results_checkpoint(
                output_path,
                results,
                args.show_first_turn_metrics,
            ),
        )
    )
    print(f"\nSaved results: {output_path}")
    print(f"Saved text log: {log_path}")


if __name__ == "__main__":
    main()
