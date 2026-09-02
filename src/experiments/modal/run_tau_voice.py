"""Run configurable tau-voice experiment grids on Modal.

The launcher is intentionally provider-agnostic. It delegates experiment
configuration to :mod:`experiments.tau_voice.run_multiple` and uses a Modal
Volume only for durable results.

Resource names are configured when the module is loaded:

* ``TAU2_MODAL_APP_NAME`` (default: ``tau2-experiments``)
* ``TAU2_MODAL_VOLUME_NAME`` (default: ``tau2-experiment-results``)
* ``TAU2_MODAL_SECRET_NAMES`` (defaults to evaluation and review secrets)

Example::

    modal run --detach --env tau-bench \
      src/experiments/modal/run_tau_voice.py::run \
      --providers openai:gpt-realtime \
      --domains retail,airline \
      --complexities regular \
      --result-root /results/my-experiment
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import modal

APP_NAME = os.environ.get("TAU2_MODAL_APP_NAME", "tau2-experiments")
VOLUME_NAME = os.environ.get("TAU2_MODAL_VOLUME_NAME", "tau2-experiment-results")
SECRET_NAMES = tuple(
    name.strip()
    for name in os.environ.get(
        "TAU2_MODAL_SECRET_NAMES",
        "tau2-experiment-keys,tau2-experiment-review-key",
    ).split(",")
    if name.strip()
)

if len(SECRET_NAMES) != 2:
    raise ValueError(
        "TAU2_MODAL_SECRET_NAMES must contain exactly two Secret names: "
        "evaluation and review"
    )

REMOTE_REPO = Path("/root/tau2")
RESULTS_ROOT = Path("/results")


def _find_repo_root() -> Path:
    """Find the local checkout while remaining importable in Modal containers."""
    source = Path(__file__).resolve()
    for parent in source.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return REMOTE_REPO


REPO_ROOT = _find_repo_root()

app = modal.App(APP_NAME)
results_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
secrets = [modal.Secret.from_name(name) for name in SECRET_NAMES]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(
        "build-essential",
        "ffmpeg",
        "git",
        "libsndfile1",
        "portaudio19-dev",
    )
    .pip_install_from_pyproject(
        str(REPO_ROOT / "pyproject.toml"),
        optional_dependencies=["voice"],
    )
    .env(
        {
            "PYTHONPATH": str(REMOTE_REPO / "src"),
            "PYTHONUNBUFFERED": "1",
        }
    )
    .add_local_dir(
        str(REPO_ROOT),
        remote_path=str(REMOTE_REPO),
        ignore=[
            ".env",
            ".git/**",
            ".venv/**",
            ".pytest_cache/**",
            ".ruff_cache/**",
            "**/__pycache__/**",
            "data/simulations/**",
            "web/leaderboard/node_modules/**",
        ],
    )
)


def _csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _result_root(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or RESULTS_ROOT not in path.parents:
        raise ValueError(f"result_root must be a child of {RESULTS_ROOT}")
    return path


def _command(
    *,
    result_root: Path,
    providers: str,
    domains: str,
    complexities: str,
    user_llm: str,
    user_llm_args: str | None,
    max_concurrency: int,
    num_tasks: int | None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "experiments.tau_voice.run_multiple",
        "--providers",
        providers,
        "--domains",
        domains,
        "--complexities",
        complexities,
        "--user-llm",
        user_llm,
        "--max-concurrency",
        str(max_concurrency),
        "--save-to",
        str(result_root),
    ]
    if user_llm_args:
        parsed = json.loads(user_llm_args)
        if not isinstance(parsed, dict):
            raise ValueError("user_llm_args must decode to a JSON object")
        command.extend(["--user-llm-args", json.dumps(parsed)])
    if num_tasks is not None:
        command.extend(["--num-tasks", str(num_tasks)])
    return command


def _validate_environment(required_env: str) -> None:
    missing = [name for name in _csv(required_env) if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            "Missing required keys in the configured Modal Secrets: "
            + ", ".join(missing)
        )


@app.function(
    image=image,
    secrets=secrets,
    volumes={str(RESULTS_ROOT): results_volume},
    cpu=8,
    memory=32768,
    timeout=24 * 60 * 60,
)
def run_experiment(
    result_root: str,
    providers: str,
    domains: str,
    complexities: str,
    user_llm: str,
    user_llm_args: str | None,
    max_concurrency: int,
    num_tasks: int | None,
    required_env: str,
    resume: bool,
) -> None:
    """Run or resume a tau-voice experiment grid."""
    _validate_environment(required_env)
    save_to = _result_root(result_root)
    if save_to.exists() and any(save_to.iterdir()) and not resume:
        raise RuntimeError(
            f"Refusing to reuse existing result directory {save_to}; pass --resume "
            "only when intentionally continuing that experiment."
        )

    save_to.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        _command(
            result_root=save_to,
            providers=providers,
            domains=domains,
            complexities=complexities,
            user_llm=user_llm,
            user_llm_args=user_llm_args,
            max_concurrency=max_concurrency,
            num_tasks=num_tasks,
        ),
        cwd=REMOTE_REPO,
        check=True,
        env=os.environ.copy(),
    )
    results_volume.commit()
    print(f"Experiment complete; committed {save_to} to {VOLUME_NAME}", flush=True)


@app.function(
    image=modal.Image.debian_slim(python_version="3.12"),
    volumes={str(RESULTS_ROOT): results_volume},
    timeout=10 * 60,
)
def show_status(result_root: str) -> None:
    """Print a provider-agnostic summary of result checkpoints."""
    results_volume.reload()
    root = _result_root(result_root)
    result_files = sorted(root.glob("*/results.json"))
    if not result_files:
        print(f"No result checkpoints found under {root}")
        return

    for path in result_files:
        data = json.loads(path.read_text())
        simulations = data.get("simulation_index", [])
        infrastructure = [
            item
            for item in simulations
            if item.get("termination_reason") == "infrastructure_error"
        ]
        valid = len(simulations) - len(infrastructure)
        passes = sum(
            item.get("reward") == 1
            and item.get("termination_reason") != "infrastructure_error"
            for item in simulations
        )
        rate = 100 * passes / valid if valid else 0.0
        print(
            f"{path.parent.name}: valid={valid} passes={passes} "
            f"rate={rate:.2f}% infrastructure={len(infrastructure)}"
        )


@app.local_entrypoint()
def run(
    result_root: str,
    providers: str,
    domains: str = "airline,retail",
    complexities: str = "control,regular",
    user_llm: str = "gpt-4.1-2025-04-14",
    user_llm_args: str | None = None,
    max_concurrency: int = 8,
    num_tasks: int | None = None,
    required_env: str = "",
    resume: bool = False,
) -> None:
    """Launch a configurable experiment in the private Modal environment."""
    run_experiment.remote(
        result_root=result_root,
        providers=providers,
        domains=domains,
        complexities=complexities,
        user_llm=user_llm,
        user_llm_args=user_llm_args,
        max_concurrency=max_concurrency,
        num_tasks=num_tasks,
        required_env=required_env,
        resume=resume,
    )


@app.local_entrypoint()
def status(result_root: str) -> None:
    """Show result checkpoint progress without reading any secrets."""
    show_status.remote(result_root)
