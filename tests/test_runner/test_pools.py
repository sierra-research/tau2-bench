# Copyright Sierra
"""Tests for declarative run pools and their thin controller adapter."""

from __future__ import annotations

import argparse
import json

import pytest

from tau2.config import DEFAULT_AUDIO_NATIVE_MODELS
from tau2.data_model.simulation import TextRunConfig, VoiceRunConfig
from tau2.runner.cli import add_pool_args
from tau2.runner.pool_driver import format_matrix, run_pool
from tau2.runner.pools import (
    DEFAULT_POOL_WORKERS,
    POOLS,
    PoolArm,
    PoolSpec,
    TaskSelection,
    census_cell,
    census_pool,
    get_pool,
    list_pools,
)

TINY = PoolSpec(
    name="tiny",
    description="two languages, one arm",
    domain="telecom",
    task_sets={"en": "telecom_en", "es": "telecom_es_identity"},
    roots={"en": "tiny_english", "es": "tiny_spanish"},
    arms=(PoolArm(provider="gemini", reasoning_effort="high"),),
    tasks=TaskSelection(kind="prefix", num_tasks=2),
    num_trials=2,
    communicate_judge_mode="llm",
)


@pytest.fixture
def pool_data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("tau2.runner.pools.DATA_DIR", tmp_path)
    return tmp_path


def _write_cell(tmp_path, spec, language, arm, entries):
    cell = tmp_path / "simulations" / spec.save_to(language, arm)
    cell.mkdir(parents=True)
    (cell / "results.json").write_text(json.dumps({"simulation_index": entries}))
    return cell


class TestRegisteredPools:
    # Every pool is a fixed full grid over its own frozen selection: the
    # factorial and xai_v2 frames run 50 tasks, the romanization-2x2 ablation
    # pools the retail_30 prefix. The size is asserted per pool rather than
    # read back off the spec, so a selection that silently changes fails here.
    @pytest.mark.parametrize(
        "name,tasks_per_cell",
        [
            (
                name,
                30
                if name.startswith(("retail_dbscript", "retail_unlocalized"))
                else 50,
            )
            for name in list_pools()
        ],
    )
    def test_every_registered_cell_is_a_fixed_full_grid_run(self, name, tasks_per_cell):
        spec = get_pool(name)
        assert spec.target_per_cell == tasks_per_cell * spec.num_trials
        assert spec.num_trials in (1, 2)
        assert spec.total_target == spec.target_per_cell * len(spec.languages) * len(
            spec.arms
        )
        assert set(spec.task_sets) == set(spec.roots)

    def test_expected_registry_shapes(self):
        assert set(POOLS) == {
            "airline_v1",
            "airline_xai_v2",
            "banking_text_en_v1",
            "multilingual_text_airline_v1",
            "multilingual_text_retail_v1",
            "multilingual_text_telecom_v1",
            "preference_v1",
            "preference_v1_prefix",
            "retail_dbscript_v1",
            "retail_unlocalized_v1",
            "retail_v1",
            "retail_xai_v1",
            "retail_xai_v2",
            "telecom_xai_v2",
        }
        assert get_pool("preference_v1").total_target == 2400
        assert get_pool("preference_v1_prefix").total_target == 1200
        assert get_pool("banking_text_en_v1").total_target == 400
        assert get_pool("multilingual_text_airline_v1").total_target == 600
        assert get_pool("multilingual_text_retail_v1").total_target == 600
        assert get_pool("multilingual_text_telecom_v1").total_target == 600
        # The xai_v2 redo column: 1 trial x 50 tasks x 6 languages per domain.
        for name in ("airline_xai_v2", "retail_xai_v2", "telecom_xai_v2"):
            spec = get_pool(name)
            assert spec.num_trials == 1
            assert spec.total_target == 300
            assert [arm.provider for arm in spec.arms] == ["xai"]
        # The romanization x localization 2x2 ablation pools: hi/zh only, the
        # two strong arms, 1 trial on the frozen retail_30 prefix (120 each).
        for name in ("retail_dbscript_v1", "retail_unlocalized_v1"):
            spec = get_pool(name)
            assert spec.num_trials == 1
            assert spec.total_target == 120
            assert set(spec.languages) == {"hi", "zh"}
            assert [arm.key for arm in spec.arms] == ["openai/xhigh", "gemini/high"]
            assert spec.tasks == TaskSelection(kind="subset", subset="retail_30")
            # Pinned identical to retail_v1 in everything but the ablated
            # condition: same seed, personas, judge, and (default) ceilings.
            retail = get_pool("retail_v1")
            assert spec.seed == retail.seed
            assert spec.use_personas == retail.use_personas
            assert spec.communicate_judge_mode == retail.communicate_judge_mode
            assert spec.max_steps_seconds == retail.max_steps_seconds
        # The two quadrants run DIFFERENT task sets — that difference IS the
        # localization axis of the 2x2.
        assert get_pool("retail_dbscript_v1").task_sets == {
            "hi": "retail_hi_identity_native",
            "zh": "retail_zh_identity_native",
        }
        assert get_pool("retail_unlocalized_v1").task_sets == {
            "hi": "retail_hi",
            "zh": "retail_zh",
        }

    def test_xai_v2_matches_its_sibling_frame_conditions(self):
        # The provider contrast only reads if the xai_v2 cells run the SAME
        # task population and judge as the openai/gemini frame they sit
        # beside. Ceilings are the deliberate exception: all three xai_v2
        # pools run at the current 2400s default (owner's call, 2026-08-27),
        # even where the sibling's collected cells ran at 1200s.
        for xai_name, sibling_name in (
            ("airline_xai_v2", "airline_v1"),
            ("retail_xai_v2", "retail_v1"),
            ("telecom_xai_v2", "preference_v1"),
        ):
            xai, sib = get_pool(xai_name), get_pool(sibling_name)
            assert xai.tasks == sib.tasks
            assert xai.task_sets == sib.task_sets
            assert xai.max_steps_seconds == 2400
            assert xai.communicate_judge_mode == sib.communicate_judge_mode
            assert set(xai.roots.values()).isdisjoint(sib.roots.values())

    def test_task_population_is_part_of_pool_identity(self):
        stratified = get_pool("preference_v1")
        prefix = get_pool("preference_v1_prefix")
        assert stratified.tasks == TaskSelection(kind="subset", subset="telecom_50")
        assert prefix.tasks == TaskSelection(kind="prefix", num_tasks=50)
        assert set(stratified.roots.values()).isdisjoint(prefix.roots.values())

    def test_pool_roots_are_not_shared(self):
        roots = [root for spec in POOLS.values() for root in spec.roots.values()]
        assert len(roots) == len(set(roots))

    def test_unknown_pool_lists_registered_names(self):
        with pytest.raises(SystemExit, match="Registered:.*preference_v1"):
            get_pool("missing")


class TestTypedCellConfigs:
    def test_voice_config_carries_every_pool_condition(self):
        arm = TINY.arms[0]
        config = TINY.cell_config("es", arm, redo_stale_tasks=True)

        assert isinstance(config, VoiceRunConfig)
        assert config.domain == "telecom"
        assert config.task_set_name == "telecom_es_identity"
        assert config.num_tasks == 2
        assert config.user_persona_id == "es"
        assert config.seed == TINY.seed
        assert config.num_trials == 2
        assert config.max_concurrency == TINY.max_concurrency
        assert config.timeout == TINY.timeout_seconds
        assert config.verbose_logs and config.auto_resume
        assert config.redo_stale_tasks
        assert config.communicate_judge_mode == "llm"
        assert config.save_to == TINY.save_to("es", arm)
        assert config.audio_native_config.provider == "gemini"
        assert config.audio_native_config.model == DEFAULT_AUDIO_NATIVE_MODELS["gemini"]
        assert config.audio_native_config.reasoning_effort == "high"
        assert config.audio_native_config.max_steps_seconds == TINY.max_steps_seconds

    def test_subset_selection_is_typed_not_spelled_as_cli_args(self):
        spec = get_pool("preference_v1")
        config = spec.cell_config("en", spec.arms[0])
        assert config.task_subset == "telecom_50"
        assert config.num_tasks is None

    def test_xai_paper_pools_keep_the_collected_model_pin(self):
        """A moving provider default must not rewrite a frozen paper condition."""
        for pool_name in ("airline_xai_v2", "retail_xai_v2", "telecom_xai_v2"):
            spec = get_pool(pool_name)
            arm = spec.arms[0]
            config = spec.cell_config("en", arm)

            assert config.audio_native_config.model == "grok-voice-think-fast-1.0"

    def test_text_config_carries_model_effort_and_dir_format(self):
        spec = get_pool("banking_text_en_v1")
        arm = spec.arms[0]
        config = spec.cell_config("en", arm)

        assert isinstance(config, TextRunConfig)
        assert config.llm_agent == arm.agent_llm
        assert config.llm_args_agent == {"reasoning_effort": arm.reasoning_effort}
        assert config.results_format == "dir"
        assert config.max_steps == spec.max_steps
        assert config.user_persona_id is None
        assert config.communicate_judge_mode == "auto"

    @pytest.mark.parametrize(
        ("pool_name", "selection"),
        [
            (
                "multilingual_text_airline_v1",
                TaskSelection(kind="prefix", num_tasks=50),
            ),
            (
                "multilingual_text_retail_v1",
                TaskSelection(kind="subset", subset="retail_50"),
            ),
            (
                "multilingual_text_telecom_v1",
                TaskSelection(kind="subset", subset="telecom_50"),
            ),
        ],
    )
    def test_multilingual_text_pools_match_voice_frames(self, pool_name, selection):
        spec = get_pool(pool_name)

        assert spec.modality == "text"
        assert spec.tasks == selection
        assert spec.num_trials == 1
        assert spec.communicate_judge_mode == "llm"
        assert [arm.agent_llm for arm in spec.arms] == [
            "gpt-5.5",
            "gemini/gemini-3.1-pro-preview",
        ]
        assert [arm.reasoning_effort for arm in spec.arms] == ["xhigh", "high"]
        assert spec.total_target == 600

    def test_modality_and_arms_must_agree(self):
        base = TINY.model_dump(exclude={"arms", "modality"})
        with pytest.raises(ValueError, match="has no agent_llm"):
            PoolSpec(
                **base,
                modality="text",
                arms=(PoolArm(provider="gpt", reasoning_effort="high"),),
            )
        with pytest.raises(ValueError, match="sets agent_llm"):
            PoolSpec(
                **base,
                modality="voice",
                arms=(
                    PoolArm(
                        provider="gpt",
                        reasoning_effort="high",
                        agent_llm="gpt-5.5",
                    ),
                ),
            )

    def test_selection_requires_its_payload(self):
        with pytest.raises(ValueError, match="subset name"):
            TaskSelection(kind="subset")
        with pytest.raises(ValueError, match="num_tasks"):
            TaskSelection(kind="prefix")


class TestCensusAndStatus:
    def test_counts_distinct_usable_pairs_and_exceptions(self, pool_data_dir):
        arm = TINY.arms[0]
        _write_cell(
            pool_data_dir,
            TINY,
            "en",
            arm,
            [
                {"id": "a", "task_id": "1", "trial": 0},
                {"id": "dup", "task_id": "1", "trial": 0},
                {"id": "b", "task_id": "2", "trial": 0},
                {
                    "id": "infra",
                    "task_id": "2",
                    "trial": 1,
                    "termination_reason": "infrastructure_error",
                },
            ],
        )
        census = census_cell(TINY, "en", arm)
        assert census.usable == 2
        assert census.tasks == 2
        assert census.duplicates == 1
        assert census.infrastructure_errors == 1
        assert census.entries == 4

    def test_missing_or_half_written_cells_report_zero(self, pool_data_dir):
        arm = TINY.arms[0]
        assert census_cell(TINY, "en", arm).usable == 0
        cell = pool_data_dir / "simulations" / TINY.save_to("en", arm)
        cell.mkdir(parents=True)
        (cell / "results.json").write_text("{")
        assert census_cell(TINY, "en", arm).usable == 0

    def test_matrix_reports_short_cells_and_recorded_failures(self, pool_data_dir):
        arm = TINY.arms[0]
        _write_cell(
            pool_data_dir,
            TINY,
            "en",
            arm,
            [
                {"id": "a", "task_id": "1", "trial": 0},
                {
                    "id": "infra",
                    "task_id": "2",
                    "trial": 0,
                    "termination_reason": "infrastructure_error",
                },
            ],
        )
        table = format_matrix(TINY, census_pool(TINY))
        assert "1 / 8 usable" in table
        assert "1 infrastructure errors" in table
        assert "en/gemini/high:3" in table
        assert "es/gemini/high:4" in table


class TestPoolDriver:
    def test_registers_all_selected_runs_with_one_controller(
        self, pool_data_dir, monkeypatch
    ):
        calls = []

        def fake_run_domains(configs, **kwargs):
            calls.append((configs, kwargs))
            return {}

        monkeypatch.setattr("tau2.runner.batch.run_domains", fake_run_domains)
        code = run_pool(
            TINY,
            workers=3,
            provider_limits={"gemini": 7},
            redo_stale_tasks=True,
        )

        assert code == 0
        assert len(calls) == 1
        configs, kwargs = calls[0]
        assert [config.user_persona_id for config in configs] == ["en", "es"]
        assert all(config.redo_stale_tasks for config in configs)
        assert kwargs == {"workers": 3, "provider_limits": {"gemini": 7}}

    def test_language_and_cell_filters_limit_the_registered_configs(
        self, pool_data_dir, monkeypatch
    ):
        captured = []
        monkeypatch.setattr(
            "tau2.runner.batch.run_domains",
            lambda configs, **kwargs: captured.extend(configs),
        )
        run_pool(TINY, languages=["es"], workers=1)
        assert [config.user_persona_id for config in captured] == ["es"]

        captured.clear()
        run_pool(TINY, only=["en/gemini/high"], workers=1)
        assert [config.user_persona_id for config in captured] == ["en"]

    def test_dry_run_builds_configs_without_starting_controller(self, monkeypatch):
        def unexpected(*args, **kwargs):
            raise AssertionError("controller should not run")

        monkeypatch.setattr("tau2.runner.batch.run_domains", unexpected)
        assert run_pool(TINY, workers=2, dry_run=True) == 0

    def test_selecting_nothing_is_an_error(self):
        with pytest.raises(SystemExit, match="no cells selected"):
            run_pool(TINY, languages=["ko"], workers=1, dry_run=True)


class TestCliSurface:
    def _parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        add_pool_args(parser)
        return parser

    def test_run_uses_shared_runner_controls(self):
        args = self._parser().parse_args(
            [
                "run",
                "preference_v1",
                "--only",
                "zh/gemini/high",
                "--workers",
                "6",
                "--provider-limit",
                "gemini=20",
                "--dry-run",
            ]
        )
        assert args.only == "zh/gemini/high"
        assert args.workers == 6
        assert args.provider_limit == "gemini=20"
        assert args.dry_run

    def test_default_worker_count_matches_collected_pool_envelope(self):
        args = self._parser().parse_args(["run", "preference_v1"])
        assert args.workers == DEFAULT_POOL_WORKERS

    @pytest.mark.parametrize(
        "removed",
        ["--budget", "--attempts", "--rounds", "--poll-seconds", "--dedup"],
    )
    def test_old_scheduler_flags_are_removed(self, removed):
        with pytest.raises(SystemExit):
            self._parser().parse_args(["run", "preference_v1", removed, "1"])

    def test_repair_ceiling_surface(self):
        args = self._parser().parse_args(
            [
                "repair-ceiling",
                "--save-to",
                "airline_v1_portuguese_airline/pt_airline_gemini_high",
                "--max-steps-seconds",
                "2400",
                "--dry-run",
            ]
        )
        assert args.save_to.endswith("pt_airline_gemini_high")
        assert args.max_steps_seconds == 2400
        assert args.dry_run
        # Both the target and the ceiling are required: a repair verb with a
        # default would rewrite headers to a value nobody stated.
        with pytest.raises(SystemExit):
            self._parser().parse_args(["repair-ceiling", "--save-to", "x"])
        with pytest.raises(SystemExit):
            self._parser().parse_args(["repair-ceiling", "--max-steps-seconds", "1"])

    def test_status_list_and_required_subcommand(self):
        assert self._parser().parse_args(["status", "preference_v1"]).pool
        assert self._parser().parse_args(["list"]).func is not None
        with pytest.raises(SystemExit):
            self._parser().parse_args([])
