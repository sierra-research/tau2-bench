# Copyright Sierra
"""Tests for ``tau2 run refill``.

The verb exists because refilling a simulation used to mean retyping the run's
flags, and a flag left off does not fail — it writes a call the rest of the
directory disagrees with. So the load-bearing claims here are:

- the config the refill runs on comes back from the result, field for field
  (``test_round_trip_*``), and a run knob that stops being recorded shows up as
  a failure rather than as a silent default;
- nothing is deleted until every reason to refuse has been checked;
- the cells deleted are exactly the cells requested — not the cross product of
  the task ids and the trials.
"""

from __future__ import annotations

import argparse
import json
from functools import lru_cache

import pytest

from tau2.config import SUBSET_AUTO
from tau2.data_model.simulation import (
    AudioNativeConfig,
    Info,
    NativenessJudgeSettings,
    Results,
    RewardInfo,
    Score,
    SimulationRun,
    TerminationReason,
    TextRunConfig,
    VoiceRunConfig,
)
from tau2.data_model.tasks import Task
from tau2.data_model.voice import SpeechEnvironment
from tau2.environment.environment import EnvironmentInfo
from tau2.multilingual.registry import PersonaResolutionError
from tau2.runner import refill as refill_module
from tau2.runner.helpers import get_info, trial_seeds
from tau2.runner.refill import (
    RefillError,
    RefillOverrides,
    execute_refill,
    plan_refill,
    reconstruct_config,
    resolve_run_dir,
)

# Fields of a RunConfig that a result deliberately does NOT record, so a
# rebuilt config cannot match the original on them:
#
# - task selection: the refill picks it (subset name, or the ids the
#   checkpoint holds), see reconstruct_config;
# - targeting: where to write and whether to resume — the refill IS a resume
#   into a given directory;
# - execution: retries, concurrency, log level. They change how the run goes,
#   not what it produces;
# - debug artifacts: audio taps and per-tick dumps;
# - dead config: text_streaming_config and *_voice_settings are declared on
#   the config and read by nothing.
#
# Anything else that differs is a run knob the results stopped recording, and
# the round-trip tests below fail on it.
UNRECONSTRUCTED_FIELDS = {
    "task_ids",
    "num_tasks",
    "task_subset",
    "save_to",
    "auto_resume",
    "max_concurrency",
    "log_level",
    "max_retries",
    "retry_delay",
    "redo_stale_tasks",
    "is_remote",
    "audio_debug",
    "audio_taps",
    "text_streaming_config",
    "agent_voice_settings",
    "user_voice_settings",
}


def _mock_task_ids() -> tuple[str, str]:
    """Two real ids from the `mock` task set.

    Fixture tasks cannot be invented: planning loads the config's task set and
    refuses to delete anything when it does not hold the directory's tasks.
    """
    from tau2.registry import registry

    tasks = registry.get_tasks_loader("mock")(task_split_name=None)
    return tasks[0].id, tasks[1].id


T0, T1 = _mock_task_ids()


def _telecom_ko_task_ids() -> list[str]:
    """Two real ids from the Korean identity-swap telecom set.

    Localized ids are the case the task-set check exists for: they belong to no
    English task set, so a directory that lost its task set name cannot be
    replayed on the domain's own.
    """
    from tau2.registry import registry

    tasks = registry.get_tasks_loader("telecom_ko_identity")(task_split_name=None)
    return [task.id for task in tasks[:2]]


def _ko_task_id() -> str:
    """A real id from the Korean telecom set, which imposes a Korean caller."""
    from tau2.registry import registry

    return registry.get_tasks_loader("telecom_ko")(task_split_name=None)[0].id


@lru_cache(maxsize=1)
def _task_index() -> dict[str, Task]:
    from tau2.registry import registry

    index: dict[str, Task] = {}
    for name in ("mock", "telecom_ko", "telecom_ko_identity"):
        for task in registry.get_tasks_loader(name)(task_split_name=None):
            index.setdefault(task.id, task)
    return index


def _make_task(task_id: str) -> Task:
    """The task as its task set defines it.

    Not invented text: planning compares the checkpoint's tasks to the loaded
    ones (see check_task_drift), so a made-up task is a directory whose tasks
    were rewritten after the run — which is a refusal, not a fixture.
    """
    return _task_index()[task_id].model_copy(deep=True)


def _make_sim(
    task_id: str,
    *,
    trial: int = 0,
    seed: int = 0,
    reward: float = 1.0,
    termination_reason: TerminationReason = TerminationReason.USER_STOP,
) -> SimulationRun:
    return SimulationRun(
        id=f"sim-{task_id}-{trial}",
        task_id=task_id,
        trial=trial,
        seed=seed,
        start_time="2026-01-01T00:00:00",
        end_time="2026-01-01T00:01:00",
        duration=60.0,
        termination_reason=termination_reason,
        reward_info=RewardInfo(reward=reward),
        messages=[],
    )


def _ko_sim(task_id: str, *, persona_id: str, **kwargs) -> SimulationRun:
    """A call whose speech environment records the pack persona that spoke it.

    That record is the only evidence of the caller in a directory written
    before Info.user_persona_id existed.
    """
    sim = _make_sim(task_id, **kwargs)
    sim.speech_environment = SpeechEnvironment(
        persona_name=persona_id, persona_id=persona_id, language="ko"
    )
    return sim


def _voice_config(**overrides) -> VoiceRunConfig:
    kwargs = dict(
        domain="mock",
        task_set_name="mock",
        audio_native_config=AudioNativeConfig(
            provider="openai", model="gpt-realtime-2", max_steps_seconds=420
        ),
        speech_complexity="control",
        user_persona_id="hi",
        llm_user="gpt-5.4-mini",
        num_trials=2,
        seed=17,
        timeout=1200.0,
        verbose_logs=True,
        scores={Score.REWARD},
        max_errors=7,
        hallucination_retries=0,
    )
    kwargs.update(overrides)
    return VoiceRunConfig(**kwargs)


def _text_config(**overrides) -> TextRunConfig:
    kwargs = dict(
        domain="mock",
        task_set_name="mock",
        agent="llm_agent",
        llm_agent="gpt-5.4-mini",
        user="user_simulator",
        llm_user="gpt-5.4-mini",
        num_trials=1,
        seed=3,
        max_steps=42,
        timeout=300.0,
        enforce_communication_protocol=True,
        auto_review=True,
        review_mode="user",
        review_model="gpt-5.4-mini",
    )
    kwargs.update(overrides)
    return TextRunConfig(**kwargs)


def _write_run(tmp_path, config, sims, *, name="run", tasks=None, fmt="dir"):
    """A run directory on disk, with an Info built from ``config`` by get_info.

    Built through ``get_info`` rather than by hand so the fixture cannot drift
    from what a real run writes.
    """
    tasks = tasks or [
        _make_task(task_id) for task_id in dict.fromkeys(s.task_id for s in sims)
    ]
    results = Results(info=get_info(config), tasks=tasks, simulations=sims)
    run_dir = tmp_path / name
    run_dir.mkdir(parents=True, exist_ok=True)
    results.save(run_dir / "results.json", format=fmt)
    return run_dir


def _differing_fields(original, rebuilt) -> set[str]:
    left = original.model_dump()
    right = rebuilt.model_dump()
    assert set(left) == set(right), "text and voice configs cannot be compared"
    return {key for key in left if left[key] != right[key]}


class TestConfigRoundTrip:
    """A config written to a result and read back must be the same run."""

    def test_the_exclusion_list_names_real_config_fields(self):
        # A stale name here would silently widen what the round-trip tests
        # tolerate.
        fields = set(TextRunConfig.model_fields) | set(VoiceRunConfig.model_fields)
        assert UNRECONSTRUCTED_FIELDS <= fields

    def test_voice_config_survives_the_round_trip(self):
        original = _voice_config()

        rebuilt = reconstruct_config(
            get_info(original), run_dir="/tmp/run", task_ids=[T0]
        )

        assert _differing_fields(original, rebuilt.config) <= UNRECONSTRUCTED_FIELDS
        assert rebuilt.unrecorded == []
        assert rebuilt.config.audio_native_config.max_steps_seconds == 420
        assert rebuilt.config.user_persona_id == "hi"
        assert rebuilt.config.speech_complexity == "control"
        assert rebuilt.config.timeout == 1200.0
        assert rebuilt.config.seed == 17
        assert rebuilt.config.num_trials == 2

    def test_text_config_survives_the_round_trip(self):
        original = _text_config()

        rebuilt = reconstruct_config(
            get_info(original), run_dir="/tmp/run", task_ids=[T0]
        )

        assert _differing_fields(original, rebuilt.config) <= UNRECONSTRUCTED_FIELDS
        # A text run has no caller persona, and a stored None cannot be told
        # apart from a results file predating the field — so it is reported.
        assert rebuilt.unrecorded == ["user_persona_id"]
        assert rebuilt.config.max_steps == 42
        assert rebuilt.config.enforce_communication_protocol is True
        assert rebuilt.config.review_mode == "user"

    def test_the_rebuilt_config_targets_the_run_directory_and_resumes(self):
        rebuilt = reconstruct_config(
            get_info(_voice_config()), run_dir="/data/elsewhere/run", task_ids=[T0]
        )

        # An absolute save_to is what lets a refill work on a directory that
        # was moved out of data/simulations.
        assert rebuilt.config.save_to == "/data/elsewhere/run"
        assert rebuilt.config.auto_resume is True

    def test_a_subset_run_is_rebuilt_on_the_subset_not_on_task_ids(self):
        info = get_info(_voice_config())
        info.task_subset = _subset_info()

        rebuilt = reconstruct_config(info, run_dir="/tmp/run", task_ids=[T0, T1])

        assert rebuilt.config.task_subset == "telecom_50"
        assert rebuilt.config.task_ids is None

    def test_a_run_without_a_subset_is_pinned_to_the_recorded_task_ids(self):
        rebuilt = reconstruct_config(
            get_info(_voice_config()), run_dir="/tmp/run", task_ids=[T0, T1]
        )

        assert rebuilt.config.task_ids == [T0, T1]

    def test_results_with_no_user_model_cannot_be_rebuilt(self):
        info = get_info(_text_config())
        info.user_info.llm = None

        with pytest.raises(RefillError, match="no user simulator model"):
            reconstruct_config(info, run_dir="/tmp/run", task_ids=[T0])


def _subset_info():
    from tau2.data_model.simulation import TaskSubsetInfo

    return TaskSubsetInfo(
        name="telecom_50",
        size=50,
        frame_task_set="telecom",
        frame_size=114,
        frame_digest="deadbeef",
        strategy="stratified",
        seed=7,
    )


def _legacy_info() -> Info:
    """An Info as written before the run config was recorded in full."""
    return Info(
        git_commit="0" * 40,
        num_trials=1,
        max_steps=100,
        max_errors=10,
        user_info={"implementation": "user_simulator", "llm": "gpt-5.4-mini"},
        agent_info={"implementation": "llm_agent", "llm": "gpt-5.4-mini"},
        environment_info=EnvironmentInfo(domain_name="mock", policy="p"),
    )


def _legacy_voice_info() -> Info:
    """A pre-recording Info for a voice run — audio config, nothing else."""
    info = _legacy_info()
    info.audio_native_config = AudioNativeConfig(
        provider="openai", model="gpt-realtime-2", max_steps_seconds=420
    )
    return info


class TestUnrecordedFields:
    """A knob the checkpoint never recorded is reported, never assumed."""

    def test_legacy_results_name_every_field_they_do_not_carry(self):
        rebuilt = reconstruct_config(_legacy_info(), run_dir="/tmp/run", task_ids=[T0])

        assert set(rebuilt.unrecorded) == {
            "user_persona_id",
            "timeout",
            "scores",
            "nativeness_judge",
            "quality_judge",
            "communicate_judge_mode",
            "task_split_name",
            "hallucination_retries",
            "auto_review",
            "review_mode",
            "review_model",
            "verbose_logs",
            "enforce_communication_protocol",
        }

    def test_an_override_fills_an_unrecorded_field(self):
        rebuilt = reconstruct_config(
            _legacy_info(),
            run_dir="/tmp/run",
            task_ids=[T0],
            overrides=RefillOverrides(verbose_logs=True, timeout=900.0),
        )

        assert rebuilt.config.verbose_logs is True
        assert rebuilt.config.timeout == 900.0
        # Still reported: the run itself never said so.
        assert "verbose_logs" in rebuilt.unrecorded

    def test_an_override_contradicting_the_record_is_refused(self):
        info = get_info(_voice_config(timeout=1200.0))

        with pytest.raises(RefillError, match="records timeout=1200"):
            reconstruct_config(
                info,
                run_dir="/tmp/run",
                task_ids=[T0],
                overrides=RefillOverrides(timeout=60.0),
            )

    def test_an_override_agreeing_with_the_record_is_accepted(self):
        rebuilt = reconstruct_config(
            get_info(_voice_config(verbose_logs=True)),
            run_dir="/tmp/run",
            task_ids=[T0],
            overrides=RefillOverrides(verbose_logs=True),
        )

        assert rebuilt.config.verbose_logs is True
        assert rebuilt.unrecorded == []

    def test_a_run_with_no_wallclock_guard_does_not_get_one_back(self):
        # `--timeout 0` opts out of the guard, and RunConfig carries that as
        # None — the same shape as a field a results file predates. Info
        # records the opt-out as 0 so the two are distinguishable, and a
        # refill of a run that had no cap must not install one: the cap would
        # end calls the original was free to finish.
        info = get_info(_voice_config(timeout=None))

        rebuilt = reconstruct_config(info, run_dir="/tmp/run", task_ids=[T0])

        assert info.timeout == 0.0
        assert rebuilt.config.timeout is None
        assert "timeout" not in rebuilt.unrecorded

    def test_the_timeout_escape_can_say_the_run_had_none(self):
        rebuilt = reconstruct_config(
            _legacy_info(),
            run_dir="/tmp/run",
            task_ids=[T0],
            overrides=RefillOverrides(timeout=0),
        )

        # 0 reaches the config as None (no guard), not as an instant timeout.
        assert rebuilt.config.timeout is None
        assert "timeout" in rebuilt.unrecorded

    def test_scores_all_covers_delivery_only_for_a_voice_directory(self):
        # 'all' is modality-dependent, so it is parsed against the results
        # rather than at argument-parse time, where the modality is unknown.
        voice = reconstruct_config(
            _legacy_voice_info(),
            run_dir="/tmp/run",
            task_ids=[T0],
            overrides=RefillOverrides(scores_spec="all"),
        )
        text = reconstruct_config(
            _legacy_info(),
            run_dir="/tmp/run",
            task_ids=[T0],
            overrides=RefillOverrides(scores_spec="all"),
        )

        assert Score.DELIVERY in voice.config.scores
        assert Score.DELIVERY not in text.config.scores

    def test_an_unknown_score_is_a_refill_error_not_a_traceback(self):
        with pytest.raises(RefillError, match="unknown score 'nativness'"):
            reconstruct_config(
                _legacy_info(),
                run_dir="/tmp/run",
                task_ids=[T0],
                overrides=RefillOverrides(scores_spec="nativness"),
            )

    def test_the_judge_escape_agrees_on_the_flag_not_the_whole_settings(self):
        # --no-nativeness-llm-judge says the judge did not run; it says nothing
        # about which model would have judged. A run recorded with a
        # non-default judge model agrees with the flag and keeps its model.
        info = get_info(
            _voice_config(
                nativeness_judge=NativenessJudgeSettings(
                    llm_judge=False, model="gpt-5.4-mini"
                )
            )
        )

        rebuilt = reconstruct_config(
            info,
            run_dir="/tmp/run",
            task_ids=[T0],
            overrides=RefillOverrides(nativeness_llm_judge=False),
        )

        assert rebuilt.config.nativeness_judge.model == "gpt-5.4-mini"
        assert rebuilt.config.nativeness_judge.llm_judge is False

    def test_the_judge_escape_contradicting_the_record_is_still_refused(self):
        info = get_info(
            _voice_config(nativeness_judge=NativenessJudgeSettings(llm_judge=True))
        )

        with pytest.raises(RefillError, match="records nativeness llm_judge=True"):
            reconstruct_config(
                info,
                run_dir="/tmp/run",
                task_ids=[T0],
                overrides=RefillOverrides(nativeness_llm_judge=False),
            )

    def test_max_concurrency_is_always_the_operators(self):
        rebuilt = reconstruct_config(
            get_info(_voice_config(max_concurrency=10)),
            run_dir="/tmp/run",
            task_ids=[T0],
            overrides=RefillOverrides(max_concurrency=3),
        )

        assert rebuilt.config.max_concurrency == 3
        assert "max_concurrency" not in rebuilt.unrecorded


class TestPlan:
    """What a refill would do, worked out before anything is written."""

    def _run_with_two_trials(self, tmp_path):
        config = _voice_config(num_trials=2, user_persona_id=None)
        seeds = trial_seeds(config.seed, 2)
        sims = [
            _make_sim(task_id, trial=trial, seed=seeds[trial])
            for trial in (0, 1)
            for task_id in (T0, T1)
        ]
        return config, _write_run(tmp_path, config, sims)

    def test_the_plan_names_the_cells_it_would_delete(self, tmp_path):
        _, run_dir = self._run_with_two_trials(tmp_path)

        plan = plan_refill(run_dir, task_ids=[T0], trials=[1])

        assert [(c.task_id, c.trial) for c in plan.drop] == [(T0, 1)]
        assert plan.drop[0].reward == 1.0
        assert plan.total_to_run == 1

    def test_all_trials_of_a_task_by_default(self, tmp_path):
        _, run_dir = self._run_with_two_trials(tmp_path)

        plan = plan_refill(run_dir, task_ids=[T0])

        assert sorted(c.trial for c in plan.drop) == [0, 1]

    def test_a_cell_the_run_never_wrote_is_planned_as_a_run_not_a_delete(
        self, tmp_path
    ):
        config = _voice_config(num_trials=1, user_persona_id=None)
        seeds = trial_seeds(config.seed, 1)
        run_dir = _write_run(
            tmp_path,
            config,
            [_make_sim(T0, seed=seeds[0])],
            tasks=[_make_task(T0), _make_task(T1)],
        )

        plan = plan_refill(run_dir, task_ids=[T1])

        assert plan.drop == []
        assert [(c.task_id, c.trial) for c in plan.requested_empty] == [(T1, 0)]

    def test_pre_existing_gaps_elsewhere_are_reported_as_spend(self, tmp_path):
        config = _voice_config(num_trials=1, user_persona_id=None)
        seeds = trial_seeds(config.seed, 1)
        run_dir = _write_run(
            tmp_path,
            config,
            [_make_sim(T0, seed=seeds[0])],
            tasks=[_make_task(T0), _make_task(T1)],
        )

        plan = plan_refill(run_dir, task_ids=[T0])

        # The resume fills every gap it finds, so t1 runs too — and saying so
        # up front is the difference between a 1-call refill and a surprise.
        assert [(c.task_id, c.trial) for c in plan.other_gaps] == [(T1, 0)]
        assert plan.total_to_run == 2

    def test_a_stored_seed_the_config_does_not_derive_is_flagged(self, tmp_path):
        config = _voice_config(num_trials=1, user_persona_id=None)
        run_dir = _write_run(tmp_path, config, [_make_sim(T0, seed=999_999)])

        plan = plan_refill(run_dir, task_ids=[T0])

        assert [c.task_id for c in plan.seed_mismatch] == [T0]

    def test_an_unknown_task_id_is_refused(self, tmp_path):
        _, run_dir = self._run_with_two_trials(tmp_path)

        with pytest.raises(RefillError, match="does not hold task"):
            plan_refill(run_dir, task_ids=["nope"])

    def test_a_trial_outside_the_grid_is_refused(self, tmp_path):
        _, run_dir = self._run_with_two_trials(tmp_path)

        with pytest.raises(RefillError, match="outside the run's grid"):
            plan_refill(run_dir, task_ids=[T0], trials=[2])

    def test_a_directory_with_no_results_is_refused(self, tmp_path):
        (tmp_path / "empty").mkdir()

        with pytest.raises(RefillError, match="no results.json"):
            plan_refill(tmp_path / "empty", task_ids=[T0])

    def test_planning_leaves_the_global_rng_where_it_found_it(self, tmp_path):
        # Deriving the run's trial seeds reseeds the global RNG, which is how
        # the runner does it. A plan is not a run: whatever draws next must not
        # change because someone asked what a refill would do.
        import random

        _, run_dir = self._run_with_two_trials(tmp_path)
        random.seed(1234)
        expected = [random.random() for _ in range(3)]

        random.seed(1234)
        plan_refill(run_dir, task_ids=[T0])

        assert [random.random() for _ in range(3)] == expected

    def test_planning_never_touches_the_directory(self, tmp_path):
        _, run_dir = self._run_with_two_trials(tmp_path)
        before = sorted(p.name for p in (run_dir / "simulations").glob("*.json"))

        plan_refill(run_dir, task_ids=[T0])

        assert (
            sorted(p.name for p in (run_dir / "simulations").glob("*.json")) == before
        )


class TestCallerPersonaGuard:
    """The guard that made this verb necessary, applied before the delete."""

    def test_a_localized_run_that_lost_its_persona_dies_before_deleting(self, tmp_path):
        # A Korean task set with no --user-persona-id: exactly the state that
        # put five preference-pool calls in a stock English voice.
        task_id = _ko_task_id()
        config = _voice_config(
            domain="telecom", task_set_name="telecom_ko", user_persona_id=None
        )
        run_dir = _write_run(
            tmp_path,
            config,
            [_make_sim(task_id, seed=trial_seeds(config.seed, 2)[0])],
        )

        with pytest.raises(PersonaResolutionError):
            plan_refill(run_dir, task_ids=[task_id])

        # And the simulation it was asked to replace is still there.
        assert list((run_dir / "simulations").glob("*.json"))

    def test_a_pre_recording_run_is_refused_until_its_caller_is_named(self, tmp_path):
        # Every directory written before Info.user_persona_id existed — which
        # is every directory of the preference pool. The calls themselves
        # still say who spoke them, so the refill refuses rather than
        # defaulting to a stock English voice.
        run_dir = _write_run(
            tmp_path,
            _voice_config(user_persona_id=None),
            [_ko_sim(T0, persona_id="soyeon_ko_v1")],
        )

        with pytest.raises(RefillError, match="record no --user-persona-id"):
            plan_refill(run_dir, task_ids=[T0])

        plan = plan_refill(
            run_dir,
            task_ids=[T0],
            overrides=RefillOverrides(user_persona_id="ko"),
        )
        assert plan.config.user_persona_id == "ko"

    def test_a_caller_of_the_wrong_language_is_refused(self, tmp_path):
        run_dir = _write_run(
            tmp_path,
            _voice_config(user_persona_id=None),
            [_ko_sim(T0, persona_id="soyeon_ko_v1")],
        )

        with pytest.raises(RefillError, match="resolves to language 'hi'"):
            plan_refill(
                run_dir,
                task_ids=[T0],
                overrides=RefillOverrides(user_persona_id="hi"),
            )

    def test_pinning_one_speaker_over_a_per_task_assignment_is_refused(self, tmp_path):
        run_dir = _write_run(
            tmp_path,
            _voice_config(user_persona_id=None, num_trials=1),
            [
                _ko_sim(T0, persona_id="soyeon_ko_v1"),
                _ko_sim(T1, persona_id="jihun_ko_v1"),
            ],
        )

        with pytest.raises(RefillError, match="pins one speaker"):
            plan_refill(
                run_dir,
                task_ids=[T0],
                overrides=RefillOverrides(user_persona_id="soyeon_ko_v1"),
            )

    def test_a_localized_run_that_kept_its_persona_plans_normally(self, tmp_path):
        task_id = _ko_task_id()
        config = _voice_config(
            domain="telecom", task_set_name="telecom_ko", user_persona_id="ko"
        )
        run_dir = _write_run(
            tmp_path,
            config,
            [_make_sim(task_id, seed=trial_seeds(config.seed, 2)[0])],
        )

        plan = plan_refill(run_dir, task_ids=[task_id], trials=[0])

        assert plan.config.user_persona_id == "ko"
        assert [c.task_id for c in plan.drop] == [task_id]


def _forget_info_fields(run_dir, *fields) -> None:
    """Strip fields from a written results.json.

    What a directory from before those fields were recorded looks like — the
    whole preference pool, in the case of task_set_name.
    """
    path = run_dir / "results.json"
    payload = json.loads(path.read_text())
    for field in fields:
        payload["info"].pop(field, None)
    path.write_text(json.dumps(payload))


class TestTaskSetRecovery:
    """The tasks a refill would load are checked while the originals exist.

    A directory whose task set is not recorded rebuilds onto the domain's own
    English set, which does not hold its tasks. Left to the runner that is a
    ValueError *after* the delete, with the simulation gone.
    """

    def _ko_run(self, tmp_path):
        task_ids = _telecom_ko_task_ids()
        config = _voice_config(
            domain="telecom",
            task_set_name="telecom_ko_identity",
            user_persona_id="ko",
            num_trials=1,
        )
        seed = trial_seeds(config.seed, 1)[0]
        run_dir = _write_run(
            tmp_path,
            config,
            [_make_sim(task_id, seed=seed) for task_id in task_ids],
        )
        return run_dir, task_ids

    def test_a_directory_that_lost_its_task_set_is_refused_before_deleting(
        self, tmp_path
    ):
        run_dir, task_ids = self._ko_run(tmp_path)
        _forget_info_fields(run_dir, "task_set_name")

        with pytest.raises(RefillError, match="--task-set-name"):
            plan_refill(run_dir, task_ids=[task_ids[0]])

        assert len(list((run_dir / "simulations").glob("*.json"))) == 2

    def test_the_refusal_names_the_task_set_that_holds_the_tasks(self, tmp_path):
        run_dir, task_ids = self._ko_run(tmp_path)
        _forget_info_fields(run_dir, "task_set_name")

        with pytest.raises(RefillError, match="telecom_ko_identity"):
            plan_refill(run_dir, task_ids=[task_ids[0]])

    def test_naming_the_task_set_makes_a_legacy_directory_refillable(self, tmp_path):
        run_dir, task_ids = self._ko_run(tmp_path)
        _forget_info_fields(run_dir, "task_set_name")

        plan = plan_refill(
            run_dir,
            task_ids=[task_ids[0]],
            overrides=RefillOverrides(task_set_name="telecom_ko_identity"),
        )

        assert plan.config.task_set_name == "telecom_ko_identity"
        assert [cell.task_id for cell in plan.drop] == [task_ids[0]]

    def test_a_task_set_contradicting_the_record_is_refused(self, tmp_path):
        run_dir, task_ids = self._ko_run(tmp_path)

        with pytest.raises(RefillError, match="records task_set_name"):
            plan_refill(
                run_dir,
                task_ids=[task_ids[0]],
                overrides=RefillOverrides(task_set_name="telecom_en"),
            )

    def test_a_task_rewritten_since_the_run_is_refused_before_deleting(self, tmp_path):
        # try_resume aborts when a checkpoint task's text no longer matches the
        # task set — but it does that inside run_domain, which a refill reaches
        # only after deleting. Same comparison, run while the simulations still
        # exist.
        run_dir, task_ids = self._ko_run(tmp_path)
        path = run_dir / "results.json"
        payload = json.loads(path.read_text())
        payload["tasks"][0]["user_scenario"]["instructions"] = "something else"
        path.write_text(json.dumps(payload))

        with pytest.raises(RefillError, match="changed since this directory"):
            plan_refill(run_dir, task_ids=[task_ids[0]])

        assert len(list((run_dir / "simulations").glob("*.json"))) == 2

    def test_an_unregistered_task_set_is_refused_by_name(self, tmp_path):
        run_dir, task_ids = self._ko_run(tmp_path)
        _forget_info_fields(run_dir, "task_set_name")

        with pytest.raises(RefillError, match="no task set 'telecom_kr' is registered"):
            plan_refill(
                run_dir,
                task_ids=[task_ids[0]],
                overrides=RefillOverrides(task_set_name="telecom_kr"),
            )

    def test_a_recorded_task_set_is_not_reported_as_unrecorded(self, tmp_path):
        run_dir, task_ids = self._ko_run(tmp_path)

        plan = plan_refill(run_dir, task_ids=[task_ids[0]])

        assert "task_set_name" not in plan.unrecorded
        assert plan.selection_note is None

    def test_a_subset_that_no_longer_selects_the_tasks_falls_back_to_ids(
        self, tmp_path
    ):
        # A subset recorded as provenance for a run that was really a prefix of
        # the file: replaying the subset would run 50 tasks this directory
        # never held. The ids it does hold are unambiguous, so they win.
        run_dir, task_ids = self._ko_run(tmp_path)
        path = run_dir / "results.json"
        payload = json.loads(path.read_text())
        payload["info"]["task_subset"] = _subset_info().model_dump()
        path.write_text(json.dumps(payload))

        plan = plan_refill(run_dir, task_ids=[task_ids[0]])

        assert plan.config.task_ids == task_ids
        assert plan.config.task_subset == SUBSET_AUTO
        assert "no longer selects" in plan.selection_note
        assert plan.total_to_run == 1


class TestExecute:
    """Deleting and re-running, with the runner itself stubbed out."""

    def _run_dir(self, tmp_path, num_trials=2):
        config = _voice_config(num_trials=num_trials, user_persona_id=None)
        seeds = trial_seeds(config.seed, num_trials)
        sims = [
            _make_sim(task_id, trial=trial, seed=seeds[trial])
            for trial in range(num_trials)
            for task_id in (T0, T1)
        ]
        return _write_run(tmp_path, config, sims)

    def _stub_run_domain(self, monkeypatch, run_dir, refill_with=None):
        """Stand in for the runner: record the config, write back the gaps."""
        seen = {}

        def fake_run_domain(config):
            seen["config"] = config
            results = Results.load(run_dir / "results.json")
            held = {(s.trial, s.task_id) for s in results.simulations}
            for trial in range(config.num_trials):
                for task in results.tasks:
                    if (trial, task.id) in held:
                        continue
                    results.simulations.append(
                        _make_sim(
                            task.id,
                            trial=trial,
                            seed=trial_seeds(config.seed, config.num_trials)[trial],
                            reward=0.0,
                            termination_reason=(
                                refill_with or TerminationReason.AGENT_STOP
                            ),
                        )
                    )
            results.save(run_dir / "results.json", format="dir")

        monkeypatch.setattr(
            "tau2.runner.batch.run_domain", fake_run_domain, raising=True
        )
        return seen

    def test_only_the_requested_cells_are_deleted(self, tmp_path, monkeypatch):
        run_dir = self._run_dir(tmp_path)
        plan = plan_refill(run_dir, task_ids=[T0, T1], trials=[0])
        deleted = []

        def fake_run_domain(config):
            # Snapshot taken between the delete and the re-run: trial 1 must
            # still be on disk. Dropping by (every task id) x (every trial)
            # would have taken it too.
            results = Results.load(run_dir / "results.json")
            deleted.extend(sorted((s.trial, s.task_id) for s in results.simulations))

        monkeypatch.setattr(
            "tau2.runner.batch.run_domain", fake_run_domain, raising=True
        )
        execute_refill(plan)

        assert deleted == [(1, T0), (1, T1)]

    def test_a_refilled_cell_is_reported_with_its_new_outcome(
        self, tmp_path, monkeypatch
    ):
        run_dir = self._run_dir(tmp_path, num_trials=1)
        plan = plan_refill(run_dir, task_ids=[T0])
        self._stub_run_domain(monkeypatch, run_dir)

        report = execute_refill(plan)

        assert [(c.task_id, c.trial) for c in report.filled] == [(T0, 0)]
        assert report.filled[0].termination_reason == TerminationReason.AGENT_STOP
        assert report.filled[0].reward == 0.0
        assert report.missing == []

    def test_a_cell_that_did_not_come_back_is_reported_missing(
        self, tmp_path, monkeypatch
    ):
        run_dir = self._run_dir(tmp_path, num_trials=1)
        plan = plan_refill(run_dir, task_ids=[T0])
        monkeypatch.setattr(
            "tau2.runner.batch.run_domain", lambda config: None, raising=True
        )

        report = execute_refill(plan)

        assert [(c.task_id, c.trial) for c in report.missing] == [(T0, 0)]
        assert report.filled == []

    def test_the_run_is_handed_the_reconstructed_config(self, tmp_path, monkeypatch):
        run_dir = self._run_dir(tmp_path, num_trials=1)
        plan = plan_refill(run_dir, task_ids=[T0])
        seen = self._stub_run_domain(monkeypatch, run_dir)

        execute_refill(plan)

        config = seen["config"]
        assert config.save_to == str(run_dir)
        assert config.auto_resume is True
        assert config.seed == 17
        assert config.audio_native_config.provider == "openai"


class TestResolveRunDir:
    def test_accepts_the_directory_the_results_file_or_neither(self, tmp_path):
        run_dir = tmp_path / "run"
        (run_dir).mkdir()
        (run_dir / "results.json").write_text("{}")

        assert resolve_run_dir(run_dir) == run_dir
        assert resolve_run_dir(run_dir / "results.json") == run_dir
        with pytest.raises(RefillError, match="no run directory"):
            resolve_run_dir(tmp_path / "missing")

    def test_falls_back_to_the_save_to_name_under_data_simulations(
        self, tmp_path, monkeypatch
    ):
        simulations = tmp_path / "simulations"
        (simulations / "my_cell").mkdir(parents=True)
        monkeypatch.setattr(refill_module, "DATA_DIR", tmp_path)

        assert resolve_run_dir("my_cell") == simulations / "my_cell"


class TestCliWiring:
    """`tau2 run refill` is reachable, and plain `tau2 run` is untouched.

    Driven through the real entry point, because the thing that could break is
    the registration itself: `refill` is a sub-verb of `run`, and `run` is a
    parser full of optional flags.
    """

    def _parser(self):
        from tau2.runner.cli import add_refill_args

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="run_command")
        add_refill_args(subparsers.add_parser("refill"))
        return parser

    def test_refill_args_parse(self):
        args = self._parser().parse_args(
            ["refill", "--save-to", "cell", "--task-ids", "a", "b", "--trials", "0"]
        )

        assert args.save_to == "cell"
        assert args.task_ids == ["a", "b"]
        assert args.trials == [0]
        assert args.func.__name__ == "run_refill"

    def test_save_to_and_task_ids_are_required(self):
        with pytest.raises(SystemExit):
            self._parser().parse_args(["refill", "--save-to", "cell"])

    def test_tau2_run_refill_dry_runs_through_the_entry_point(
        self, tmp_path, monkeypatch, capsys
    ):
        from tau2 import cli

        config = _voice_config(num_trials=1, user_persona_id=None)
        run_dir = _write_run(
            tmp_path, config, [_make_sim(T0, seed=trial_seeds(config.seed, 1)[0])]
        )
        monkeypatch.setattr(
            "sys.argv",
            [
                "tau2",
                "run",
                "refill",
                "--save-to",
                str(run_dir),
                "--task-ids",
                T0,
                "--dry-run",
            ],
        )

        cli.main()

        out = capsys.readouterr().out
        assert "deleting 1 simulation(s)" in out
        assert "nothing deleted" in out

    def test_plain_tau2_run_still_dispatches_to_the_runner(self, monkeypatch):
        from tau2 import cli

        seen = {}

        def fake_run_domain(config):
            seen["c"] = config
            return Results(info=get_info(config), tasks=[], simulations=[])

        monkeypatch.setattr(cli, "run_domain", fake_run_domain)
        monkeypatch.setattr(
            "sys.argv", ["tau2", "run", "--domain", "mock", "--num-tasks", "1"]
        )

        cli.main()

        assert seen["c"].domain == "mock"
        assert seen["c"].num_tasks == 1
