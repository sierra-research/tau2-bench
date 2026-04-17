#!/usr/bin/env python3
"""
Run audio-native evaluations across providers and speech complexities.

Provider syntax: "provider", "provider:model", or "provider:model:reasoning"
  - openai                           (default model, no reasoning)
  - openai:gpt-realtime-1.5          (specific model)
  - openai:gpt-realtime-1.5:high     (model + reasoning effort)
  - livekit                           (default cascaded config)
  - livekit::openai-thinking          (default model + cascaded config)

Usage:
    python -m experiments.tau_voice.run_multiple --providers openai,gemini --save-to data/exp/my_run
    python -m experiments.tau_voice.run_multiple --providers openai:gpt-realtime-1.5:high --save-to data/exp/run --num-tasks 5
    python -m experiments.tau_voice.run_multiple --providers livekit,livekit::openai-thinking --save-to data/exp/run
"""

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from tau2.config import DEFAULT_AUDIO_NATIVE_MODELS, DEFAULT_LLM_USER, DEFAULT_SEED

DEFAULT_DOMAINS = ["airline", "retail"]
DEFAULT_COMPLEXITIES = ["control", "regular"]


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
                provider=provider, model=model,
                cascaded_config=qualifier, display_name=spec,
            )
        else:
            return ProviderSpec(
                provider=provider, model=model,
                reasoning_effort=qualifier, display_name=spec,
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
    max_concurrency: int = 8,
) -> list[str]:
    cmd = [
        "uv", "run", "tau2", "run",
        "--domain", domain,
        "--audio-native",
        "--audio-native-provider", spec.provider,
        "--audio-native-model", spec.model,
        "--speech-complexity", complexity,
        "--seed", str(seed),
        "--user-llm", user_llm,
        "--max-concurrency", str(max_concurrency),
        "--verbose-logs",
        "--auto-review",
        "--auto-resume",
        "--save-to", save_to,
    ]
    if spec.reasoning_effort is not None:
        cmd.extend(["--reasoning-effort", spec.reasoning_effort])
    if spec.cascaded_config is not None:
        cmd.extend(["--cascaded-config", spec.cascaded_config])
    if num_tasks is not None:
        cmd.extend(["--num-tasks", str(num_tasks)])
    return cmd


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
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--user-llm", type=str, default=DEFAULT_LLM_USER)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument(
        "--save-to",
        type=str,
        required=True,
        help="Base directory for results (e.g. data/exp/my_run)",
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

    print(f"Running {total} combinations -> {base_dir}\n")

    for i, (domain, spec, complexity) in enumerate(combos, 1):
        run_name = f"{domain}_{complexity}_{spec.display_name}".replace(":", "_")
        save_to = str(base_dir / run_name)

        print(f"[{i}/{total}] {run_name}")
        cmd = build_command(
            domain, spec, complexity, save_to,
            num_tasks=args.num_tasks,
            seed=args.seed,
            user_llm=args.user_llm,
            max_concurrency=args.max_concurrency,
        )
        print(f"  $ {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"  WARNING: exit code {result.returncode}")
        print()

    print(f"Done. Results in {base_dir}")


if __name__ == "__main__":
    sys.exit(main())
