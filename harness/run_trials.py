#!/usr/bin/env python3
"""Run the v2 CRM evaluation one trial at a time.

Each trial is isolated:

1. Reset Twenty and verify the clean state.
2. Seed the task fixtures through MCP.
3. Mirror the seeded records into the local CRMDB through initial_state.
4. Run one tau2 simulation with the v2 task.
5. Save the full simulation JSON plus a run metadata file.

This avoids the shared-state contamination in the earlier run.

Example:
    python run_trials.py \
      --repo /workspace/23-tau2-bench-copy \
      --models Qwen3.5-9B,GPT-OSS-20B \
      --user-model Qwen3.5-9B \
      --agent-args '{"api_base":"http://127.0.0.1:8080/v1","api_key":"local"}' \
      --user-args '{"api_base":"http://127.0.0.1:8080/v1","api_key":"local"}' \
      --trials 8
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from reset_twenty import reset as reset_twenty  # noqa: E402
from seed_twenty import seed_task  # noqa: E402
from twenty_mcp import MCPClient, get_credentials  # noqa: E402


def safe_name(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in value)


def parse_json_arg(value: str | None) -> dict:
    if not value:
        return {}
    return json.loads(value)


def load_env_file(path: Path) -> None:
    """Load KEY=VALUE pairs without overriding the current environment."""
    if not path.exists():
        return

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)


def first_env(*keys: str) -> str | None:
    for key in keys:
        value = os.environ.get(key)
        if value:
            return value
    return None


PROVIDER_KEY_ENV = {
    "deepseek": "DEEPSEEK_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "together_ai": "TOGETHER_API_KEY",
}

SECRET_KEY_NAMES = {
    "api_key",
    "apikey",
    "key",
    "token",
    "access_token",
    "auth_token",
    "secret",
    "client_secret",
    "authorization",
    "password",
}
SECRET_KEY_SUFFIXES = ("_api_key", "_key", "_token", "_secret", "_password")


def is_secret_name(name: str) -> bool:
    lowered = name.lower()
    return lowered in SECRET_KEY_NAMES or lowered.endswith(SECRET_KEY_SUFFIXES)

# Local llama.cpp models are addressed with the openai/ prefix. Any other
# provider prefix is treated as a remote API and left to LiteLLM.
LOCAL_MODEL_PREFIX = "openai/"


def model_provider(model: str) -> str:
    return model.split("/", 1)[0] if "/" in model else ""


def redact(value, key: str = ""):
    """Recursively mask secret-looking values before they are written to disk."""
    if isinstance(value, dict):
        return {k: redact(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v, key) for v in value]
    if is_secret_name(key):
        if isinstance(value, str) and value in ("", "no_key_required", "local", "***"):
            return value
        return "***" if value else value
    return value


def scrub(text: str) -> str:
    """Remove any known provider key from text destined for logs."""
    for name in set(PROVIDER_KEY_ENV.values()):
        value = os.environ.get(name)
        if value and len(value) >= 8:
            text = text.replace(value, "***")
    return text


def apply_base_urls(
    models: list[str],
    user_model: str,
    agent_args: dict,
    user_args: dict,
    env_file: Path,
    agent_mode: str = "auto",
) -> None:
    """Fill api_base/api_key for local models; leave remote models to LiteLLM.

    Local llama.cpp agents keep the openai/ prefix and are filled from .env as
    before. Provider-prefixed models (e.g. deepseek/deepseek-chat) keep no
    api_base/api_key here; LiteLLM reads the provider key from the environment.
    """
    explicit_base = bool(agent_args.get("api_base"))
    if agent_mode == "local":
        local_agent = True
    elif agent_mode == "remote":
        local_agent = False
    else:
        local_agent = not explicit_base and all(
            model.startswith(LOCAL_MODEL_PREFIX) for model in models
        )

    if not explicit_base:
        if local_agent:
            if models and all(model == user_model for model in models):
                base = first_env("LLAMA_USER_BASE_URL", "OPENAI_BASE_URL")
            else:
                base = first_env("LLAMA_AGENT_BASE_URL")
            if not base:
                raise SystemExit(
                    "No agent api_base. Pass it in --agent-args or add "
                    f"LLAMA_AGENT_BASE_URL to {env_file}"
                )
            agent_args["api_base"] = base
        else:
            provider = model_provider(models[0]) if models else ""
            env_name = PROVIDER_KEY_ENV.get(provider)
            if env_name and not os.environ.get(env_name):
                raise SystemExit(
                    f"Remote agent model {models[0]!r} needs {env_name}. "
                    f"Add it to {env_file} or the environment."
                )

    if not user_args.get("api_base"):
        base = first_env("LLAMA_USER_BASE_URL", "OPENAI_BASE_URL")
        if not base:
            raise SystemExit(
                "No user api_base. Pass it in --user-args or add "
                f"LLAMA_USER_BASE_URL to {env_file}"
            )
        user_args["api_base"] = base

    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        if local_agent:
            agent_args.setdefault("api_key", api_key)
        user_args.setdefault("api_key", api_key)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path, help="tau2-bench-copy repo root")
    parser.add_argument("--tasks", type=Path, default=HERE / "tasks_v2.json")
    parser.add_argument("--fixtures", type=Path, default=HERE / "fixtures_v2.json")
    parser.add_argument("--out", type=Path, default=HERE / "runs")
    parser.add_argument("--models", required=True, help="comma-separated agent model ids")
    parser.add_argument("--user-model", required=True)
    parser.add_argument("--agent-args", default=None, help="JSON dict for llm_args_agent")
    parser.add_argument("--user-args", default=None, help="JSON dict for llm_args_user")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--agent-local",
        action="store_const",
        const="local",
        dest="agent_mode",
        help="force local llama.cpp endpoint (ignore model prefix)",
    )
    mode.add_argument(
        "--agent-remote",
        action="store_const",
        const="remote",
        dest="agent_mode",
        help="force remote provider endpoint (ignore model prefix)",
    )
    parser.set_defaults(agent_mode="auto")
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--task-ids", default=None, help="comma-separated task ids")
    parser.add_argument("--seed-base", type=int, default=4242)
    parser.add_argument("--user-seed-base", type=int, default=9242)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--url", default=None)
    parser.add_argument("--token", default=None)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip trials whose simulation.json already exists (resume a stopped run)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "skip tasks that already have all trials; re-run an incomplete "
            "task from scratch (clears its folder)"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    repo = args.repo.resolve()
    env_file = repo / ".env"
    load_env_file(env_file)

    tasks = {str(t["id"]): t for t in json.loads(args.tasks.read_text())}
    fixtures = json.loads(args.fixtures.read_text())
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    task_ids = (
        [t.strip() for t in args.task_ids.split(",") if t.strip()]
        if args.task_ids
        else sorted(tasks)
    )

    agent_args = parse_json_arg(args.agent_args)
    user_args = parse_json_arg(args.user_args)
    if not args.dry_run:
        apply_base_urls(
            models,
            args.user_model,
            agent_args,
            user_args,
            env_file,
            agent_mode=args.agent_mode,
        )

    provider = model_provider(models[0]) or "local"
    print(f"Tasks: {task_ids}")
    print(f"Models: {models}")
    print(f"Agent mode: {args.agent_mode} (provider={provider})")
    print(f"Trials per task: {args.trials}")
    print(f"Agent api_base: {agent_args.get('api_base') or '<provider default>'}")
    print(f"User api_base: {user_args.get('api_base')}")

    if args.dry_run:
        for model in models:
            for task_id in task_ids:
                for trial in range(args.trials):
                    print(f"  {model} / T{task_id} / trial {trial + 1}")
        return

    sys.path.insert(0, str(repo / "src"))

    from tau2.data_model.simulation import TextRunConfig  # noqa: E402
    from tau2.data_model.tasks import Task  # noqa: E402
    from tau2.runner.batch import run_single_task  # noqa: E402

    mcp_url, mcp_token = get_credentials(repo, args.url, args.token)
    client = MCPClient(mcp_url, mcp_token)

    args.out.mkdir(parents=True, exist_ok=True)
    failures = 0

    for model in models:
        for task_id in task_ids:
            task_dir = args.out / safe_name(model) / f"task_{task_id}"
            if args.resume and task_dir.exists():
                done = len(list(task_dir.glob("trial_*/simulation.json")))
                if done >= args.trials:
                    print(
                        f"skip complete task {model} | T{task_id} "
                        f"({done}/{args.trials})"
                    )
                    continue
                print(
                    f"incomplete task {model} | T{task_id} "
                    f"({done}/{args.trials}); rerunning from scratch"
                )
                shutil.rmtree(task_dir)

            for trial in range(args.trials):
                seed = args.seed_base + trial
                user_seed = args.user_seed_base + trial
                run_dir = args.out / safe_name(model) / f"task_{task_id}" / f"trial_{trial + 1:02d}"
                run_dir.mkdir(parents=True, exist_ok=True)

                if args.skip_existing and (run_dir / "simulation.json").exists():
                    print(
                        f"skip existing {model} | T{task_id} | "
                        f"trial {trial + 1}/{args.trials}"
                    )
                    continue

                print(f"\n=== {model} | T{task_id} | trial {trial + 1}/{args.trials} ===")

                try:
                    try:
                        before = client.clean_counts()
                    except Exception as exc:
                        print("Pre-trial count failed; forcing reset:", exc)
                        before = {
                            "tasks": -1,
                            "opportunities": -1,
                            "people": -1,
                            "companies": -1,
                        }
                    if any(before.values()):
                        print("State is not clean before trial:", before)
                        if reset_twenty(
                            repo,
                            args.url,
                            args.token,
                            known_before=before,
                        ) != 0:
                            raise RuntimeError(
                                f"Reset reported a dirty state; before={before}"
                            )

                    agent_data = seed_task(client, fixtures[task_id])
                    expected = {
                        "companies": len(fixtures[task_id].get("companies", [])),
                        "people": len(fixtures[task_id].get("people", [])),
                        "opportunities": len(
                            fixtures[task_id].get("opportunities", [])
                        ),
                        "tasks": len(fixtures[task_id].get("tasks", [])),
                    }
                    seeded = client.clean_counts()
                    if seeded != expected:
                        for object_name in (
                            "companies",
                            "people",
                            "opportunities",
                            "tasks",
                        ):
                            try:
                                raw = client.call(
                                    f"find_many_{object_name}", {"limit": 5}
                                )
                                print(
                                    f"RAW find_many_{object_name}: {str(raw)[:1000]}"
                                )
                            except Exception as exc:
                                print(
                                    f"RAW find_many_{object_name} ERROR: {exc}"
                                )
                        raise RuntimeError(
                            f"Seed count mismatch; got {seeded}, expected {expected}"
                        )

                    task_dict = copy.deepcopy(tasks[task_id])
                    task_dict["initial_state"] = {
                        "initialization_data": {"agent_data": agent_data}
                    }
                    task = Task.model_validate(task_dict)

                    agent_llm_args = {"seed": seed, **agent_args}
                    user_llm_args = {"seed": user_seed, **user_args}
                    config = TextRunConfig(
                        domain="crm",
                        agent="llm_agent",
                        llm_agent=model,
                        llm_args_agent=agent_llm_args,
                        user="user_simulator",
                        llm_user=args.user_model,
                        llm_args_user=user_llm_args,
                        max_steps=args.max_steps,
                        num_trials=1,
                        max_concurrency=1,
                        seed=seed,
                    )

                    simulation = run_single_task(
                        config,
                        task,
                        seed=seed,
                        save_dir=run_dir,
                        verbose_logs=True,
                    )

                    (run_dir / "simulation.json").write_text(
                        json.dumps(simulation.model_dump(mode="json"), indent=1)
                    )
                    provider = model_provider(model) or "local"
                    key_env = PROVIDER_KEY_ENV.get(provider)
                    key_present = bool(
                        (os.environ.get(key_env) if key_env else None)
                        or os.environ.get("OPENAI_API_KEY")
                    )
                    (run_dir / "meta.json").write_text(
                        json.dumps(
                            {
                                "model": model,
                                "user_model": args.user_model,
                                "provider": provider,
                                "api_key_present": key_present,
                                "timestamp_utc": datetime.now(
                                    timezone.utc
                                ).isoformat(),
                                "task_id": task_id,
                                "trial": trial + 1,
                                "seed": seed,
                                "user_seed": user_seed,
                                "agent_args": redact(agent_llm_args),
                                "user_args": redact(user_llm_args),
                                "fixtures": agent_data,
                            },
                            indent=1,
                        )
                    )

                    reward = None
                    if getattr(simulation, "reward_info", None) is not None:
                        reward = simulation.reward_info.reward
                    print(f"reward: {reward}")

                except Exception as exc:
                    failures += 1
                    print(f"FAILED: {scrub(str(exc))}")
                    (run_dir / "error.txt").write_text(
                        scrub(f"{exc}\n\n{traceback.format_exc()}")
                    )

    print(f"\nDone. Failures: {failures}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
