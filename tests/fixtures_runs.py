# Copyright Sierra
"""Shared synthetic-Hindi-run builders for the annotation and judges suites.

Both suites need the same thing: a real run on disk (written via
``Results.save``) whose sims carry COMPLETE stored nativeness verdicts, so
every judge path runs with ``reuse_existing`` and zero LLM calls. The two
suites differ only in the shape of the stored verdicts:

- annotation plants a severity-3 FAIL (``fail_factor``) among severity-2
  PASSes — "the judge found something" for packet/sheet exports;
- judges plants a DEFERRED/ERROR gap (``gap_factor``/``gap_outcome``) among
  severity-3 PASSes with recorded evidence — rejudge reuse-gating.

``hi_factor_checks`` carries both knobs; each suite's conftest wraps it (and
``hi_sim``) with its own defaults and judge model.
"""

from pathlib import Path
from typing import Literal, Optional

from tau2.data_model.message import AssistantMessage
from tau2.data_model.simulation import (
    AgentInfo,
    AudioNativeConfig,
    Info,
    JudgeOutcome,
    NativenessFactorCheck,
    NativenessInfo,
    Results,
    RewardInfo,
    SimulationRun,
    TerminationReason,
    UserInfo,
)
from tau2.data_model.tasks import StructuredUserInstructions, Task, UserScenario
from tau2.data_model.voice import VoiceSettings
from tau2.environment.environment import EnvironmentInfo
from tau2.judges.export import judge_factor_ids
from tau2.judges.nativeness.harness import NATIVENESS_RUBRIC_VERSION
from tau2.judges.nativeness.judge import NATIVENESS_JUDGE_PROMPT_VERSION

HI_TRANSCRIPT = "आपका कोड JMO1MG है और आपकी flight कल सुबह है"


def hi_factor_checks(
    *,
    fail_factor: Optional[str] = None,
    gap_factor: Optional[str] = None,
    gap_outcome: JudgeOutcome = JudgeOutcome.DEFERRED,
    pass_severity: int = 2,
    pass_evidence: Optional[str] = None,
) -> list[NativenessFactorCheck]:
    """A COMPLETE set of hi judge-factor checks with two independent knobs.

    ``fail_factor`` plants a severity-3 FAIL with evidence and quote;
    ``gap_factor`` records ``gap_outcome`` (e.g. DEFERRED/ERROR) with no
    evidence, opening a gap. Every other factor PASSes at ``pass_severity``
    with ``pass_evidence``.
    """
    checks = []
    for factor_id in sorted(judge_factor_ids("hi")):
        if factor_id == fail_factor:
            checks.append(
                NativenessFactorCheck(
                    id=factor_id,
                    category="register",
                    severity=3,
                    outcome=JudgeOutcome.FAIL,
                    evidence="used singular agreement with आप",
                    quote="आप तैयार है",
                )
            )
        elif factor_id == gap_factor:
            checks.append(
                NativenessFactorCheck(
                    id=factor_id,
                    category="register",
                    severity=pass_severity,
                    outcome=gap_outcome,
                )
            )
        else:
            checks.append(
                NativenessFactorCheck(
                    id=factor_id,
                    category="register",
                    severity=pass_severity,
                    outcome=JudgeOutcome.PASS,
                    evidence=pass_evidence,
                )
            )
    return checks


def hi_sim(
    sim_id: str,
    task_id: str,
    *,
    checks: list[NativenessFactorCheck],
    judge_model: str,
    reward: float = 1.0,
    trial: int = 0,
    language: str = "hi",
    judge_prompt_version: Optional[str] = NATIVENESS_JUDGE_PROMPT_VERSION,
) -> SimulationRun:
    """A single-message Hindi sim carrying the given stored verdicts.

    Stored verdicts default to the CURRENT judge prompt version so they read
    as reusable; pass an older ``judge_prompt_version`` to make them stale.
    """
    return SimulationRun(
        id=sim_id,
        task_id=task_id,
        trial=trial,
        start_time="2026-01-01T00:00:00",
        end_time="2026-01-01T00:05:00",
        duration=300.0,
        termination_reason=TerminationReason.AGENT_STOP,
        reward_info=RewardInfo(reward=reward),
        messages=[AssistantMessage(role="assistant", content=HI_TRANSCRIPT)],
        nativeness_info=NativenessInfo(
            score=0.5,
            factor_checks=checks,
            language=language,
            script="deva",
            rubric_version=NATIVENESS_RUBRIC_VERSION,
            judge_model=judge_model,
            judge_prompt_version=judge_prompt_version,
        ),
    )


def make_task(task_id: str) -> Task:
    return Task(
        id=task_id,
        user_scenario=UserScenario(
            instructions=StructuredUserInstructions(
                domain="airline",
                task_instructions=f"Synthetic task {task_id}.",
                reason_for_call="Synthetic.",
            )
        ),
    )


def make_hi_results(
    tmp_path: Path,
    sims: list[SimulationRun],
    *,
    format: Literal["json", "dir"] = "dir",
    name: str = "hi_run",
    tasks: Optional[list[Task]] = None,
    num_trials: int = 1,
    voice_settings: Optional[VoiceSettings] = None,
    domain: str = "airline",
    audio_native_config: Optional[AudioNativeConfig] = None,
    agent_llm: str = "fake-llm",
    agent_llm_args: Optional[dict] = None,
) -> Path:
    """Write a Hindi run to disk in the requested format; returns its path.

    ``tasks`` defaults to one synthetic task per distinct ``task_id`` in
    ``sims``. Pass ``voice_settings`` to represent a VOICE run: real voice runs
    always carry a ``VoiceSettings`` block, and a default-None fixture hid a
    crash in language derivation that only fired once it was populated.

    ``audio_native_config`` is what makes the run AUDIO-NATIVE, and with it the
    provider/model/reasoning-effort triple that identifies the arm. Leave it
    None for a text run; sims that set ``agent_provider`` without it describe a
    run whose arm cannot be identified, which feature extraction refuses.
    """
    results = Results(
        info=Info(
            git_commit="0" * 40,
            num_trials=num_trials,
            max_steps=50,
            max_errors=10,
            user_info=UserInfo(
                implementation="fake_user_simulator",
                llm="fake-llm",
                voice_settings=voice_settings,
            ),
            agent_info=AgentInfo(
                implementation="fake_llm_agent",
                llm=agent_llm,
                llm_args=agent_llm_args,
            ),
            environment_info=EnvironmentInfo(domain_name=domain, policy="synthetic"),
            audio_native_config=audio_native_config,
        ),
        tasks=(
            tasks
            if tasks is not None
            else [make_task(tid) for tid in sorted({s.task_id for s in sims})]
        ),
        simulations=sims,
    )
    if format == "dir":
        run_dir = tmp_path / name
        run_dir.mkdir(parents=True, exist_ok=True)
        results.save(run_dir, format="dir")
        return run_dir
    path = tmp_path / f"{name}.json"
    results.save(path, format="json")
    return path
