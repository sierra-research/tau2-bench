#!/usr/bin/env python3
"""
Run audio-native evaluations across providers and speech complexities.

Usage:
    python -m experiments.tau_voice.run_multiple --providers openai,gemini,xai --save-to data/exp/my_run
    python -m experiments.tau_voice.run_multiple --providers openai --save-to data/exp/my_run --num-tasks 5
    python -m experiments.tau_voice.run_multiple --providers openai,gemini --save-to data/exp/my_run --domains airline --complexities control
"""

import argparse
import subprocess
import sys
from pathlib import Path

from tau2.config import DEFAULT_AUDIO_NATIVE_MODELS, DEFAULT_LLM_USER, DEFAULT_SEED

DEFAULT_DOMAINS = ["airline", "retail"]
DEFAULT_COMPLEXITIES = ["control", "regular"]

# Virtual providers that map to a real provider + cascaded config
VIRTUAL_PROVIDERS = {
    "livekit-thinking": ("livekit", "openai-thinking"),
}


def build_command(
    domain: str,
    provider: str,
    model: str,
    complexity: str,
    save_to: str,
    *,
    cascaded_config: str | None = None,
    num_tasks: int | None = None,
    seed: int = DEFAULT_SEED,
    user_llm: str = DEFAULT_LLM_USER,
    max_concurrency: int = 8,
) -> list[str]:
    cmd = [
        "uv",
        "run",
        "tau2",
        "run",
        "--domain",
        domain,
        "--audio-native",
        "--audio-native-provider",
        provider,
        "--audio-native-model",
        model,
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
    if cascaded_config is not None:
        cmd.extend(["--cascaded-config", cascaded_config])
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
        help="Comma-separated providers (e.g. openai,gemini,xai,livekit,livekit-thinking)",
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

    providers = [p.strip() for p in args.providers.split(",")]
    domains = [d.strip() for d in args.domains.split(",")]
    complexities = [c.strip() for c in args.complexities.split(",")]

    # Resolve provider -> (real_provider, model, cascaded_config, display_name)
    provider_entries = []
    for p in providers:
        if p in VIRTUAL_PROVIDERS:
            real_provider, cascaded = VIRTUAL_PROVIDERS[p]
            model = DEFAULT_AUDIO_NATIVE_MODELS[real_provider]
            provider_entries.append((real_provider, model, cascaded, p))
        elif ":" in p:
            prov, model = p.split(":", 1)
            provider_entries.append((prov, model, None, p))
        else:
            provider_entries.append((p, DEFAULT_AUDIO_NATIVE_MODELS[p], None, p))

    base_dir = Path(args.save_to).resolve()

    combos = [
        (domain, prov, model, cascaded, display, complexity)
        for domain in domains
        for prov, model, cascaded, display in provider_entries
        for complexity in complexities
    ]
    total = len(combos)

    print(f"Running {total} combinations -> {base_dir}\n")

    for i, (domain, provider, model, cascaded, display, complexity) in enumerate(
        combos, 1
    ):
        run_name = f"{domain}_{complexity}_{display}_{model}"
        save_to = str(base_dir / run_name)

        print(f"[{i}/{total}] {run_name}")
        cmd = build_command(
            domain,
            provider,
            model,
            complexity,
            save_to,
            cascaded_config=cascaded,
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
