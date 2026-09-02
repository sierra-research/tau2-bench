#!/usr/bin/env python3
"""
Run audio-native evaluations across providers and speech complexities.

Provider syntax: "provider", "provider:model", or "provider:model:reasoning"
  - openai                           (default model, no reasoning)
  - openai:gpt-realtime-1.5          (specific model)
  - openai:gpt-realtime-1.5:high     (model + reasoning effort)
  - openai:pine-voice-preview        (Pine's OpenAI-compatible model)
  - livekit                           (default cascaded config)
  - livekit::openai-thinking          (default model + cascaded config)

Pine models use the normal OpenAI provider syntax. Set ``PINE_API_KEY`` and
``PINE_REALTIME_BASE_URL``; the OpenAI realtime provider selects them
automatically for model names beginning with ``pine-``.

Usage:
    python -m experiments.tau_voice.run_multiple --providers openai,gemini --save-to data/exp/my_run
    python -m experiments.tau_voice.run_multiple --providers openai:gpt-realtime-1.5:high --save-to data/exp/run --num-tasks 5
    python -m experiments.tau_voice.run_multiple --providers openai:pine-voice-preview --save-to data/exp/pine
    python -m experiments.tau_voice.run_multiple --providers livekit,livekit::openai-thinking --save-to data/exp/run

By default the grid runs sequentially, one `tau2 run` subprocess per combo.
With --workers N the combos are registered as producers in one controller and
run concurrently across N worker processes (see docs/designs/parallel-runner.md):

    python -m experiments.tau_voice.run_multiple --providers openai,gemini \\
        --save-to data/exp/run --workers 4
"""

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from tau2.config import DEFAULT_AUDIO_NATIVE_MODELS, DEFAULT_LLM_USER, DEFAULT_SEED

DEFAULT_DOMAINS = ["airline", "retail"]
DEFAULT_COMPLEXITIES = ["control", "regular"]
DEFAULT_PROVIDER_LIMITS = "gemini=40,openai=40,xai=40,livekit=10,nova=5,qwen=5"


@dataclass
class ProviderSpec:
    provider: str
    model: str
    reasoning_effort: Optional[str] = None
    cascaded_config: Optional[str] = None
    display_name: str = ""

    def __post_init__(self):
        if not self.display_name:
            self.display_name = self.provider


def parse_provider(spec: str) -> ProviderSpec:
    """Parse provider spec: 'provider', 'provider:model', or 'provider:model:qualifier'.

    For livekit, the third field is a cascaded config name (e.g. 'openai-thinking').
    For all other providers, the third field is reasoning effort (e.g. 'high').
    """
    parts = spec.split(":")
    provider = parts[0]
    model = DEFAULT_AUDIO_NATIVE_MODELS.get(provider, "dummy")

    if len(parts) == 1:
        return ProviderSpec(provider=provider, model=model, display_name=spec)
    elif len(parts) == 2:
        if parts[1]:
            model = parts[1]
        return ProviderSpec(provider=provider, model=model, display_name=spec)
    elif len(parts) == 3:
        if parts[1]:
            model = parts[1]
        qualifier = parts[2]
        if provider == "livekit":
            return ProviderSpec(
                provider=provider,
                model=model,
                cascaded_config=qualifier,
                display_name=spec,
            )
        else:
            return ProviderSpec(
                provider=provider,
                model=model,
                reasoning_effort=qualifier,
                display_name=spec,
            )
    else:
        raise ValueError(f"Invalid provider spec: {spec}")


def build_command(
    domain: str,
    spec: ProviderSpec,
    complexity: str,
    save_to: str,
    *,
    num_tasks: int | None = None,
    seed: int = DEFAULT_SEED,
    user_llm: str = DEFAULT_LLM_USER,
    user_llm_args: dict | None = None,
    max_concurrency: int = 8,
    max_steps_seconds: int | None = None,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "tau2.cli",
        "run",
        "--domain",
        domain,
        "--audio-native",
        "--audio-native-provider",
        spec.provider,
        "--audio-native-model",
        spec.model,
        "--speech-complexity",
        complexity,
        "--seed",
        str(seed),
        "--user-llm",
        user_llm,
        "--max-concurrency",
        str(max_concurrency),
        "--verbose-logs",
        "--auto-review",
        "--auto-resume",
        "--save-to",
        save_to,
    ]
    if user_llm_args is not None:
        cmd.extend(["--user-llm-args", json.dumps(user_llm_args)])
    if spec.reasoning_effort is not None:
        cmd.extend(["--reasoning-effort", spec.reasoning_effort])
    if spec.cascaded_config is not None:
        cmd.extend(["--cascaded-config", spec.cascaded_config])
    if num_tasks is not None:
        cmd.extend(["--num-tasks", str(num_tasks)])
    if max_steps_seconds is not None:
        cmd.extend(["--max-steps-seconds", str(max_steps_seconds)])
    return cmd


def build_config(
    domain: str,
    spec: ProviderSpec,
    complexity: str,
    save_to: str,
    *,
    num_tasks: int | None = None,
    seed: int = DEFAULT_SEED,
    user_llm: str = DEFAULT_LLM_USER,
    user_llm_args: dict | None = None,
    max_concurrency: int = 8,
    max_steps_seconds: int | None = None,
):
    """The same run build_command() shells out for, as a VoiceRunConfig —
    used by --workers mode to register combos with one controller."""
    from tau2.data_model.simulation import AudioNativeConfig, VoiceRunConfig

    audio_kwargs = dict(
        provider=spec.provider,
        model=spec.model,
        cascaded_config_name=spec.cascaded_config,
        reasoning_effort=spec.reasoning_effort,
    )
    if max_steps_seconds is not None:
        audio_kwargs["max_steps_seconds"] = max_steps_seconds

    config_kwargs = dict(
        domain=domain,
        num_tasks=num_tasks,
        llm_user=user_llm,
        seed=seed,
        max_concurrency=max_concurrency,
        verbose_logs=True,
        auto_review=True,
        auto_resume=True,
        save_to=save_to,
        audio_native_config=AudioNativeConfig(**audio_kwargs),
        speech_complexity=complexity,
    )
    if user_llm_args is not None:
        config_kwargs["llm_args_user"] = user_llm_args
    return VoiceRunConfig(**config_kwargs)


def main():
    parser = argparse.ArgumentParser(
        description="Run audio-native evals across providers and speech complexities."
    )
    parser.add_argument(
        "--providers",
        type=str,
        required=True,
        help="Comma-separated provider specs (e.g. openai,openai:model:high,livekit::openai-thinking)",
    )
    parser.add_argument(
        "--domains",
        type=str,
        default=",".join(DEFAULT_DOMAINS),
        help=f"Comma-separated domains. Default: {','.join(DEFAULT_DOMAINS)}",
    )
    parser.add_argument(
        "--complexities",
        type=str,
        default=",".join(DEFAULT_COMPLEXITIES),
        help=f"Comma-separated speech complexities. Default: {','.join(DEFAULT_COMPLEXITIES)}",
    )
    parser.add_argument("--num-tasks", type=int, default=None)
    parser.add_argument(
        "--max-steps-seconds",
        type=int,
        default=None,
        help="Cap conversation duration in seconds (passed through to tau2 run).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--user-llm", type=str, default=DEFAULT_LLM_USER)
    parser.add_argument(
        "--user-llm-args",
        type=json.loads,
        default=None,
        help="JSON object passed to the user simulator model.",
    )
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument(
        "--save-to",
        type=str,
        required=True,
        help="Base directory for results (e.g. data/exp/my_run)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Run all combos concurrently under one controller with N worker "
        "processes (0 = sequential subprocess per combo, the default).",
    )
    parser.add_argument(
        "--provider-limit",
        type=str,
        default=DEFAULT_PROVIDER_LIMITS,
        help="Per-provider concurrency caps with --workers. "
        f"Default: {DEFAULT_PROVIDER_LIMITS}",
    )
    args = parser.parse_args()

    specs = [parse_provider(p.strip()) for p in args.providers.split(",")]
    domains = [d.strip() for d in args.domains.split(",")]
    complexities = [c.strip() for c in args.complexities.split(",")]

    base_dir = Path(args.save_to).resolve()

    combos = [
        (domain, spec, complexity)
        for domain in domains
        for spec in specs
        for complexity in complexities
    ]
    total = len(combos)

    combo_kwargs = dict(
        num_tasks=args.num_tasks,
        seed=args.seed,
        user_llm=args.user_llm,
        user_llm_args=args.user_llm_args,
        max_concurrency=args.max_concurrency,
        max_steps_seconds=args.max_steps_seconds,
    )

    def combo_save_to(domain, spec, complexity) -> str:
        run_name = f"{domain}_{complexity}_{spec.display_name}".replace(":", "_")
        return str(base_dir / run_name)

    if args.workers > 0:
        from tau2.runner.batch import run_domains
        from tau2.runner.work import parse_provider_limits

        print(
            f"Running {total} combinations concurrently -> {base_dir} "
            f"({args.workers} workers x {args.max_concurrency} slots)\n"
        )
        configs = [
            build_config(
                domain,
                spec,
                complexity,
                combo_save_to(domain, spec, complexity),
                **combo_kwargs,
            )
            for domain, spec, complexity in combos
        ]
        run_domains(
            configs,
            workers=args.workers,
            provider_limits=parse_provider_limits(args.provider_limit),
        )
        print(f"Done. Results in {base_dir}")
        return

    print(f"Running {total} combinations -> {base_dir}\n")

    for i, (domain, spec, complexity) in enumerate(combos, 1):
        save_to = combo_save_to(domain, spec, complexity)

        print(f"[{i}/{total}] {Path(save_to).name}")
        cmd = build_command(domain, spec, complexity, save_to, **combo_kwargs)
        print(f"  $ {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"  WARNING: exit code {result.returncode}")
        print()

    print(f"Done. Results in {base_dir}")


if __name__ == "__main__":
    sys.exit(main())
