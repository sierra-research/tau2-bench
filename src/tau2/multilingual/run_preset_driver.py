# Copyright Sierra
"""``tau2 run-preset`` engine — execute multilingual run presets and provider
matrices. (The argparse surface and mode dispatch live in
``tau2.multilingual.cli``, the package-owned CLI registration module.)

Two modes, one verb:

- **Preset mode** (``tau2 run-preset <preset> --stage smoke|full``): run one
  named preset (generated from each language pack's ``experiment`` block; see
  ``tau2.multilingual.run_presets``), one ``tau2 run`` subprocess per arm,
  sequentially, with the arm's environment overrides applied (the English
  baseline preset forces the LLM communicate judge so its metric matches the
  localized arms). Stages: ``smoke`` = 1 task per arm (cheap pipeline sanity —
  native-script turns, backchannels, metadata, evaluator — before spending on
  the full run); ``full`` = all tasks per arm.

- **Matrix mode** (``tau2 run-preset --all-languages --providers openai,gemini``):
  the cross-provider language matrix. One lane per provider runs concurrently;
  within a lane the languages run sequentially, each as a full ``tau2 run``
  over the language's registered task set (``<domain>_<lang>``) with the
  language-code persona. Results land under
  ``data/simulations/<base>/<provider>/<lang>/``; each cell's output is teed to
  ``<cell>.driver.log`` next to its results dir. OpenAI pins ``gpt-realtime-2``;
  Gemini uses its default model. ``--auto-resume`` on every cell means a rerun
  of the same command fills infra-error gaps instead of redoing work.

Preset results land under ``data/simulations/<preset>/<stage>_<arm>/``.
Requires the same API keys as any audio-native run (OPENAI_API_KEY,
ELEVENLABS_API_KEY, plus the user-sim/judge LLM key).
"""

from __future__ import annotations

import datetime
import json
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from tau2.config import (
    DEFAULT_MATRIX_MAX_STEPS_SECONDS,
    DEFAULT_MATRIX_SEED,
    DEFAULT_MATRIX_SPEECH_COMPLEXITY,
    DEFAULT_MULTILINGUAL_DOMAIN,
    DEFAULT_MULTILINGUAL_PROVIDERS,
    DEFAULT_MULTILINGUAL_RUN_CONCURRENCY,
    DEFAULT_MULTILINGUAL_VOICE_TIMEOUT_SECONDS,
    resolve_audio_native_reasoning_effort,
)
from tau2.utils import DATA_DIR

# Matrix-mode matched conditions (from the airline multilingual matrix runs):
# gpt-realtime-2 pinned for OpenAI (Gemini uses its provider default), plus
# the shared DEFAULT_MATRIX_* conditions from ``tau2.config``.
MATRIX_OPENAI_MODEL = "gpt-realtime-2"


# =============================================================================
# Shared subprocess plumbing
# =============================================================================


def _subprocess_env(extra_env: dict) -> dict:
    """Env for a ``tau2 run`` subprocess, with the running interpreter's bin dir
    on PATH.

    The commands shell out to a bare ``tau2``; if the driver is launched by a
    venv interpreter whose bin dir isn't on PATH, that bare ``tau2`` would not
    resolve and the run dies with ``FileNotFoundError``. Prepending
    ``dirname(sys.executable)`` makes the sibling ``tau2`` console-script
    resolve regardless of activation.
    """
    env = {**os.environ, **extra_env}
    bin_dir = os.path.dirname(sys.executable)
    env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
    return env


def _run_teed(command: list[str], env: dict) -> subprocess.CompletedProcess:
    """Run one ``tau2 run``, teeing its stderr so a tail is available.

    stderr is both streamed live (so a long run isn't silent) and captured, so
    that on a non-zero exit we can echo the last lines inline — distinguishing a
    real defect from a flake without forcing a manual re-run (see
    ``_report_arm_failure``). stdout is left attached to the terminal unchanged.
    """
    tail: list[str] = []
    proc = subprocess.Popen(command, env=env, stderr=subprocess.PIPE, text=True)
    assert proc.stderr is not None
    for line in proc.stderr:
        sys.stderr.write(line)
        tail.append(line)
    proc.wait()
    return subprocess.CompletedProcess(
        command, proc.returncode, stdout=None, stderr="".join(tail)
    )


def _has_clean_results(save_to: str) -> bool:
    """True iff the run's results.json exists and recorded zero sim errors.

    A non-zero exit from ``tau2 run`` is sometimes spurious: a fully-persisted,
    error-free sim has been observed to exit non-zero while an identical re-run
    exits 0. Detecting a clean persisted result lets us downgrade such a failure
    to a warning instead of a hard FAILED.
    """
    # Deliberately lazy: the data model is heavy and this path only runs after
    # a subprocess already failed.
    from pydantic import ValidationError

    from tau2.data_model.simulation import SimulationIndexEntry

    results_path = DATA_DIR / "simulations" / save_to / "results.json"
    if not results_path.exists():
        return False
    try:
        payload = json.loads(results_path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    if not isinstance(payload, dict):
        return False
    # Voice runs use the dir format: metadata excludes "simulations" but carries
    # a lightweight "simulation_index" with per-sim termination_reason. Fall back
    # to the inline "simulations" list for the single-file format (a superset of
    # the index-entry fields, so the same typed view applies).
    raw_sims = payload.get("simulation_index") or payload.get("simulations") or []
    try:
        sims = [SimulationIndexEntry.model_validate(sim) for sim in raw_sims]
    except ValidationError:
        # Mid-write/partial entries mean we cannot vouch for the run.
        return False
    if not sims:
        return False
    # A run is "clean" only if EVERY persisted sim reached a non-error terminal
    # state. Besides too_many_errors, an infrastructure_error sim (or a missing
    # reason from a partial/mid-run write) also means the non-zero exit may be
    # real — don't downgrade those to a spurious-flake note. (We can't correlate
    # this file to the specific failed subprocess, so keep the check strict.)
    bad_reasons = {"too_many_errors", "infrastructure_error", None, ""}
    return all(sim.termination_reason not in bad_reasons for sim in sims)


def _report_failure(
    label: str, save_to: str, result: subprocess.CompletedProcess
) -> None:
    """Surface a tail of the failed run's stderr (and note clean-results flakes)."""
    tail_lines = (result.stderr or "").splitlines()
    if tail_lines:
        shown = tail_lines[-20:]
        print(f"--- last {len(shown)} stderr line(s) for '{label}': ---")
        for line in shown:
            print(f"    {line}")
    else:
        print(f"    (no stderr captured for '{label}')")
    if _has_clean_results(save_to):
        print(
            f"    NOTE: '{label}' has a persisted results.json with 0 "
            "errors despite the non-zero exit — likely a spurious exit code, "
            "not a real defect. Inspect the results before treating as a failure."
        )


# =============================================================================
# Preset mode
# =============================================================================


def run_preset(
    preset_name: str,
    *,
    stage: str = "smoke",
    arm: Optional[str] = None,
    num_trials: int = 1,
    max_concurrency: int = DEFAULT_MULTILINGUAL_RUN_CONCURRENCY,
    dry_run: bool = False,
    extra_args: Optional[list[str]] = None,
) -> int:
    """Run one preset's arms sequentially; returns a process exit code."""
    # Deliberately lazy: building the preset table loads every language pack.
    from tau2.multilingual.run_presets import get_run_preset, get_run_presets

    try:
        preset = get_run_preset(preset_name)
    except ValueError:
        available = sorted(get_run_presets())
        print(f"FAILED — unknown run preset '{preset_name}'. Available presets:")
        for name in available:
            print(f"  - {name}")
        if not available:
            print("  (none — no registered pack has a usable experiment block)")
        return 1
    arms = [preset.get_arm(arm)] if arm else preset.arms

    failures: list[str] = []
    for preset_arm in arms:
        command = preset.cli_command(
            preset_arm,
            stage=stage,
            num_trials=num_trials,
            max_concurrency=max_concurrency,
            # --auto-resume for EVERY stage (smoke included): a stale
            # data/simulations/<preset>/<stage>_<arm>/results.json otherwise
            # triggers an interactive "resume? (y/n)" prompt inside `tau2 run`,
            # which crashes with EOFError when there's no stdin (the common
            # cause of english-baseline smoke-arm failures). The adaptive
            # full-stage executor already injects --auto-resume; this matches it.
            extra_args=[*(extra_args or []), "--auto-resume"],
        )
        env_prefix = " ".join(f"{k}={v}" for k, v in preset_arm.env.items())
        printable = f"{env_prefix} {shlex.join(command)}".strip()
        print(f"\n=== [{preset.name}/{stage}] arm: {preset_arm.name} ===")
        print(printable)
        if dry_run:
            continue

        result = _run_teed(command, _subprocess_env(preset_arm.env))
        if result.returncode != 0:
            print(f"!!! arm '{preset_arm.name}' exited with code {result.returncode}")
            _report_failure(preset_arm.name, preset.save_to(preset_arm, stage), result)
            failures.append(preset_arm.name)
            # Keep going: the remaining arms are independent experiments and
            # their results are still useful for debugging the failed one.

    if failures:
        print(f"\nFAILED arms: {failures}")
        return 1
    if not dry_run:
        print(f"\nAll arms completed. Results: data/simulations/{preset.name}/")
    return 0


# =============================================================================
# Matrix mode (langs x providers)
# =============================================================================


def matrix_languages() -> list[str]:
    """Every registered language pack except the English baseline."""
    # Deliberately lazy: loads every language pack.
    from tau2.multilingual.loader import load_language_packs
    from tau2.multilingual.registry import list_language_packs

    load_language_packs()
    return sorted(lang for lang in list_language_packs() if lang != "en")


def matrix_cell_command(
    *,
    domain: str,
    lang: str,
    provider: str,
    save_to: str,
    num_trials: int,
    max_concurrency: int,
    num_tasks: Optional[int] = None,
    extra_args: Optional[list[str]] = None,
) -> list[str]:
    """The ``tau2 run`` argv for one matrix cell (one lang on one provider).

    The task set is resolved through the SAME identity-variant selection preset
    generation uses (``run_presets.resolve_main_task_suffix``): when a
    ``<domain>_<lang>_identity`` set exists (and the pack does not opt out via
    ``experiment.include_identity_arm: false``), the matrix runs it too.
    """
    # Deliberately lazy: resolving the task set loads the language packs.
    from tau2.multilingual.run_presets import matrix_task_set_name

    command = [
        "tau2",
        "run",
        "--domain",
        domain,
        "--task-set-name",
        matrix_task_set_name(domain, lang),
        "--audio-native",
        "--audio-native-provider",
        provider,
    ]
    if provider == "openai":
        command += ["--audio-native-model", MATRIX_OPENAI_MODEL]
    # Pinned explicitly (not left to the provider default) so the argv itself
    # names the arm — see AGENTS.md "Pin --reasoning-effort".
    command += [
        "--reasoning-effort",
        resolve_audio_native_reasoning_effort(provider, None).value,
    ]
    command += [
        "--speech-complexity",
        DEFAULT_MATRIX_SPEECH_COMPLEXITY,
        "--user-persona-id",
        lang,
        "--seed",
        str(DEFAULT_MATRIX_SEED),
        "--num-trials",
        str(num_trials),
        "--max-concurrency",
        str(max_concurrency),
        "--save-to",
        save_to,
        "--max-steps-seconds",
        str(DEFAULT_MATRIX_MAX_STEPS_SECONDS),
        "--timeout",
        str(DEFAULT_MULTILINGUAL_VOICE_TIMEOUT_SECONDS),
        "--verbose-logs",
        "--auto-resume",
    ]
    if num_tasks is not None:
        command += ["--num-tasks", str(num_tasks)]
    if extra_args:
        command += extra_args
    return command


def _run_matrix_lane(
    provider: str,
    langs: list[str],
    *,
    domain: str,
    base: str,
    num_trials: int,
    max_concurrency: int,
    num_tasks: Optional[int],
    extra_args: Optional[list[str]],
) -> list[str]:
    """One provider lane: its languages, sequentially. Returns failed cells."""
    failures: list[str] = []
    for lang in langs:
        save_to = f"{base}/{provider}/{lang}"
        command = matrix_cell_command(
            domain=domain,
            lang=lang,
            provider=provider,
            save_to=save_to,
            num_trials=num_trials,
            max_concurrency=max_concurrency,
            num_tasks=num_tasks,
            extra_args=extra_args,
        )
        log_path = DATA_DIR / "simulations" / f"{save_to}.driver.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{provider}] START {save_to} (log: {log_path})", flush=True)
        # Cells log to a file (lanes run concurrently — interleaved live output
        # would be unreadable); the tail is echoed on failure.
        with open(log_path, "w") as log:
            proc = subprocess.run(
                command,
                env=_subprocess_env({}),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if proc.returncode != 0:
            print(f"[{provider}] !!! {save_to} exited {proc.returncode}")
            tail = log_path.read_text().splitlines()[-20:]
            _report_failure(
                save_to,
                save_to,
                subprocess.CompletedProcess(
                    command, proc.returncode, None, "\n".join(tail)
                ),
            )
            failures.append(save_to)
        else:
            print(f"[{provider}] DONE  {save_to}", flush=True)
    return failures


def run_matrix(
    *,
    domain: str = DEFAULT_MULTILINGUAL_DOMAIN,
    providers: Optional[list[str]] = None,
    lanes: Optional[int] = None,
    num_tasks: Optional[int] = None,
    num_trials: int = 1,
    max_concurrency: int = DEFAULT_MULTILINGUAL_RUN_CONCURRENCY,
    base: Optional[str] = None,
    dry_run: bool = False,
    extra_args: Optional[list[str]] = None,
) -> int:
    """The langs x providers matrix; returns a process exit code."""
    providers = providers or list(DEFAULT_MULTILINGUAL_PROVIDERS)
    base = base or f"{domain}_matrix_{datetime.date.today().isoformat()}"
    langs = matrix_languages()

    print(
        f"=== MATRIX: {domain} voice | providers={','.join(providers)} | "
        f"langs={' '.join(langs)} | conc={max_concurrency} | "
        f"complexity={DEFAULT_MATRIX_SPEECH_COMPLEXITY} | "
        f"tasks={'all' if num_tasks is None else num_tasks} ==="
    )

    if dry_run:
        for provider in providers:
            for lang in langs:
                command = matrix_cell_command(
                    domain=domain,
                    lang=lang,
                    provider=provider,
                    save_to=f"{base}/{provider}/{lang}",
                    num_trials=num_trials,
                    max_concurrency=max_concurrency,
                    num_tasks=num_tasks,
                    extra_args=extra_args,
                )
                print(shlex.join(command))
        return 0

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=lanes or len(providers)) as pool:
        lane_futures = {
            provider: pool.submit(
                _run_matrix_lane,
                provider,
                langs,
                domain=domain,
                base=base,
                num_trials=num_trials,
                max_concurrency=max_concurrency,
                num_tasks=num_tasks,
                extra_args=extra_args,
            )
            for provider in providers
        }
        for provider, future in lane_futures.items():
            failures.extend(future.result())
            print(f"=== {provider.upper()} LANE DONE ===")

    if failures:
        print(f"\nFAILED cells: {failures}")
        return 1
    print(f"\nAll cells completed. Results: data/simulations/{base}/")
    return 0
