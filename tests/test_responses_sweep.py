import random

import experiments.hyperparam.responses_sweep as responses_sweep
from experiments.hyperparam.responses_sweep import (
    DEFAULT_BASELINE_REASONING,
    DEFAULT_BASELINE_SERVICE_TIER,
    DEFAULT_BASELINE_VERBOSITY,
    DEFAULT_BASELINE_WEB_SEARCH,
    RunMode,
    SweepPoint,
    SweepShape,
    _expected_simulation_keys,
    build_agent_llm_args,
    build_known_variant_points,
    build_sweep_points,
    canonicalize_response_repair_results,
    estimate_reasoning_web_search_cost_usd,
    estimate_token_cost_usd,
    log_responses_dry_run_plan,
    make_sweep_run_config,
    validate_reasoning_efforts_for_model,
    validate_response_resume_state,
)
from tau2.data_model.simulation import (
    Info,
    Results,
    RewardInfo,
    SimulationRun,
    TerminationReason,
    UserInfo,
)
from tau2.data_model.tasks import EvaluationCriteria, Task, UserScenario
from tau2.environment.environment import EnvironmentInfo


def _make_info() -> Info:
    return Info(
        git_commit="abc123",
        num_trials=1,
        max_steps=100,
        max_errors=10,
        user_info=UserInfo(implementation="user_simulator"),
        agent_info={"implementation": "llm_agent"},
        environment_info=EnvironmentInfo(domain_name="mock", policy="test policy"),
    )


def _make_task(task_id: str) -> Task:
    return Task(
        id=task_id,
        user_scenario=UserScenario(instructions="test instruction"),
        evaluation_criteria=EvaluationCriteria(),
    )


def _make_sim(
    task_id: str,
    *,
    sim_id: str | None = None,
    trial: int = 0,
    seed: int = 42,
    termination_reason: TerminationReason = TerminationReason.USER_STOP,
) -> SimulationRun:
    return SimulationRun(
        id=sim_id or f"sim-{task_id}",
        task_id=task_id,
        start_time="2026-01-01T00:00:00",
        end_time="2026-01-01T00:01:00",
        duration=60.0,
        termination_reason=termination_reason,
        reward_info=RewardInfo(reward=1.0),
        messages=[],
        trial=trial,
        seed=seed,
    )


def test_build_sweep_points_grid():
    points = build_sweep_points(
        SweepShape.GRID,
        reasoning_efforts=["low", "medium"],
        verbosities=["low", "medium"],
        web_search_modes=["off", "auto"],
        service_tiers=["default", "priority"],
    )
    assert len(points) == 16
    assert SweepPoint("low", "low", "off", "default") in points
    assert SweepPoint("medium", "medium", "auto", "priority") in points


def test_build_sweep_points_ofat_uses_medium_baseline():
    points = build_sweep_points(
        SweepShape.OFAT,
        reasoning_efforts=["minimal", DEFAULT_BASELINE_REASONING],
        verbosities=["low", DEFAULT_BASELINE_VERBOSITY],
        web_search_modes=[DEFAULT_BASELINE_WEB_SEARCH, "required"],
        service_tiers=[DEFAULT_BASELINE_SERVICE_TIER, "priority"],
    )
    assert points == [
        SweepPoint(
            reasoning_effort=DEFAULT_BASELINE_REASONING,
            verbosity=DEFAULT_BASELINE_VERBOSITY,
            web_search_mode=DEFAULT_BASELINE_WEB_SEARCH,
            service_tier=DEFAULT_BASELINE_SERVICE_TIER,
        ),
        SweepPoint(
            reasoning_effort="minimal",
            verbosity=DEFAULT_BASELINE_VERBOSITY,
            web_search_mode=DEFAULT_BASELINE_WEB_SEARCH,
            service_tier=DEFAULT_BASELINE_SERVICE_TIER,
        ),
        SweepPoint(
            reasoning_effort=DEFAULT_BASELINE_REASONING,
            verbosity="low",
            web_search_mode=DEFAULT_BASELINE_WEB_SEARCH,
            service_tier=DEFAULT_BASELINE_SERVICE_TIER,
        ),
        SweepPoint(
            reasoning_effort=DEFAULT_BASELINE_REASONING,
            verbosity=DEFAULT_BASELINE_VERBOSITY,
            web_search_mode="required",
            service_tier=DEFAULT_BASELINE_SERVICE_TIER,
        ),
        SweepPoint(
            reasoning_effort=DEFAULT_BASELINE_REASONING,
            verbosity=DEFAULT_BASELINE_VERBOSITY,
            web_search_mode=DEFAULT_BASELINE_WEB_SEARCH,
            service_tier="priority",
        ),
    ]


def test_build_agent_llm_args_adds_web_search_config_only_when_enabled():
    base_args = {"temperature": 0.0}
    off_args = build_agent_llm_args(
        base_args,
        SweepPoint("medium", "medium", "off", "default"),
        web_search_context_size="high",
        web_search_allowed_domains=["openai.com"],
    )
    assert off_args == {
        "temperature": 0.0,
        "reasoning_effort": "medium",
        "verbosity": "medium",
        "web_search_mode": "off",
        "service_tier": "default",
    }

    on_args = build_agent_llm_args(
        base_args,
        SweepPoint("high", "low", "auto", "priority"),
        web_search_context_size="high",
        web_search_allowed_domains=["openai.com"],
    )
    assert on_args == {
        "temperature": 0.0,
        "reasoning_effort": "high",
        "verbosity": "low",
        "web_search_mode": "auto",
        "service_tier": "priority",
        "web_search_context_size": "high",
        "web_search_filters": {"allowed_domains": ["openai.com"]},
    }


def test_build_agent_llm_args_adds_transport_and_parallel_tool_calls():
    args = build_agent_llm_args(
        {"temperature": 0.0},
        SweepPoint(
            "medium",
            "medium",
            "off",
            "default",
            responses_transport="websocket",
            parallel_tool_calls=False,
        ),
    )

    assert args["responses_transport"] == "websocket"
    assert args["parallel_tool_calls"] is False


def test_known_variant_suite_preserves_baseline_name_and_adds_variants():
    points = build_known_variant_points()

    assert points[0] == SweepPoint("medium", "medium", "off", "default")
    assert any(point.llm == "gpt-5.5" and point.reasoning_effort == "low" for point in points)
    assert any(point.parallel_tool_calls is False for point in points)
    assert any(point.responses_transport == "websocket" for point in points)


def test_estimate_token_cost_uses_priority_rate_card():
    cost = estimate_token_cost_usd(
        model="gpt-5.4-mini",
        service_tier="priority",
        prompt_tokens=1_000_000,
        cached_tokens=100_000,
        completion_tokens=500_000,
    )
    assert cost == 5.865


def test_estimate_token_cost_uses_gpt_55_priority_rate_card():
    cost = estimate_token_cost_usd(
        model="gpt-5.5",
        service_tier="priority",
        prompt_tokens=1_000_000,
        cached_tokens=100_000,
        completion_tokens=500_000,
    )
    assert cost == 48.875


def test_estimate_reasoning_web_search_cost_usd():
    assert estimate_reasoning_web_search_cost_usd("gpt-5.4-mini", 3) == 0.03
    assert estimate_reasoning_web_search_cost_usd("gpt-4.1", 3) == 0.0


def test_validate_reasoning_efforts_for_gpt_54():
    validate_reasoning_efforts_for_model("gpt-5.4-mini", ["none", "low", "xhigh"])

    try:
        validate_reasoning_efforts_for_model("gpt-5.4-mini", ["minimal"])
    except ValueError as exc:
        assert "does not support reasoning efforts" in str(exc)
    else:
        raise AssertionError("Expected unsupported reasoning effort to raise")


def test_validate_reasoning_efforts_for_gpt_55():
    validate_reasoning_efforts_for_model("gpt-5.5", ["none", "low", "xhigh"])


def test_make_sweep_run_config_applies_domain_task_split_override():
    config, spec = make_sweep_run_config(
        exp_name="exp",
        llm="gpt-5.4-mini",
        domain="telecom",
        mode=RunMode.DEFAULT,
        point=SweepPoint("medium", "medium", "off", "default"),
        llm_user="gpt-4.1-2025-04-14",
        llm_user_args={"temperature": 0.0},
        base_agent_llm_args={"temperature": 0.0},
        seed=300,
        max_steps=200,
        max_duration_seconds=900,
        max_errors=10,
        max_concurrency=5,
        num_trials=1,
        num_tasks=None,
        task_ids=None,
        task_split_name="base",
        domain_task_splits={"telecom": "full"},
        auto_resume=True,
        web_search_context_size="medium",
        web_search_allowed_domains=None,
        web_search_user_location=None,
    )

    assert config.task_split_name == "full"
    assert config.timeout == 900
    assert spec.domain == "telecom"


def test_make_sweep_run_config_preserves_baseline_cache_name():
    _, spec = make_sweep_run_config(
        exp_name="exp",
        llm="gpt-5.4-mini",
        domain="telecom",
        mode=RunMode.DEFAULT,
        point=SweepPoint("medium", "medium", "off", "default"),
        llm_user="gpt-4.1-2025-04-14",
        llm_user_args={"temperature": 0.0},
        base_agent_llm_args={"temperature": 0.0},
        seed=300,
        max_steps=100,
        max_duration_seconds=900,
        max_errors=10,
        max_concurrency=5,
        num_trials=1,
        num_tasks=None,
        task_ids=None,
        task_split_name="base",
        domain_task_splits=None,
        auto_resume=True,
        web_search_context_size="medium",
        web_search_allowed_domains=None,
        web_search_user_location=None,
    )

    assert spec.name == (
        "model=gpt-5.4-mini__domain=telecom__mode=default__"
        "reasoning=medium__verbosity=medium__web=off__service=default"
    )


def test_make_sweep_run_config_names_known_variant_fields():
    _, spec = make_sweep_run_config(
        exp_name="exp",
        llm="gpt-5.4-mini",
        domain="telecom",
        mode=RunMode.DEFAULT,
        point=SweepPoint(
            "medium",
            "medium",
            "off",
            "default",
            responses_transport="websocket",
            parallel_tool_calls=True,
            variant="websocket_parallel",
        ),
        llm_user="gpt-4.1-2025-04-14",
        llm_user_args={"temperature": 0.0},
        base_agent_llm_args={"temperature": 0.0},
        seed=300,
        max_steps=100,
        max_duration_seconds=900,
        max_errors=10,
        max_concurrency=5,
        num_trials=1,
        num_tasks=None,
        task_ids=None,
        task_split_name="base",
        domain_task_splits=None,
        auto_resume=True,
        web_search_context_size="medium",
        web_search_allowed_domains=None,
        web_search_user_location=None,
    )

    assert "variant=websocket_parallel" in spec.name
    assert "transport=websocket" in spec.name
    assert "parallel=true" in spec.name


def test_expected_simulation_keys_match_tau2_seed_schedule():
    config, _ = make_sweep_run_config(
        exp_name="exp",
        llm="gpt-5.4-mini",
        domain="airline",
        mode=RunMode.DEFAULT,
        point=SweepPoint("medium", "medium", "off", "default"),
        llm_user="gpt-4.1-2025-04-14",
        llm_user_args={"temperature": 0.0},
        base_agent_llm_args={"temperature": 0.0},
        seed=300,
        max_steps=200,
        max_duration_seconds=900,
        max_errors=10,
        max_concurrency=5,
        num_trials=2,
        num_tasks=1,
        task_ids=None,
        task_split_name="base",
        domain_task_splits=None,
        auto_resume=True,
        web_search_context_size="medium",
        web_search_allowed_domains=None,
        web_search_user_location=None,
    )

    rng = random.Random(300)
    expected_seeds = [rng.randint(0, 1000000) for _ in range(2)]
    keys = _expected_simulation_keys(config)

    assert len(keys) == 2
    assert {key[0] for key in keys} == {0, 1}
    assert {key[2] for key in keys} == set(expected_seeds)


def test_canonicalize_response_repair_results_installs_resume_checkpoint(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(responses_sweep, "DATA_DIR", tmp_path)

    _, target_spec = make_sweep_run_config(
        exp_name="base-exp",
        llm="gpt-5.4-mini",
        domain="airline",
        mode=RunMode.DEFAULT,
        point=SweepPoint(
            "medium",
            "medium",
            "off",
            "default",
            responses_transport="websocket",
            variant="websocket",
        ),
        llm_user="gpt-4.1-2025-04-14",
        llm_user_args={"temperature": 0.0},
        base_agent_llm_args={"temperature": 0.0},
        seed=300,
        max_steps=100,
        max_duration_seconds=900,
        max_errors=10,
        max_concurrency=5,
        num_trials=1,
        num_tasks=None,
        task_ids=None,
        task_split_name="base",
        domain_task_splits=None,
        auto_resume=True,
        web_search_context_size="medium",
        web_search_allowed_domains=None,
        web_search_user_location=None,
    )
    _, repair_spec = make_sweep_run_config(
        exp_name="repair-exp",
        llm="gpt-5.4-mini",
        domain="airline",
        mode=RunMode.DEFAULT,
        point=SweepPoint(
            "medium",
            "medium",
            "off",
            "default",
            responses_transport="websocket",
        ),
        llm_user="gpt-4.1-2025-04-14",
        llm_user_args={"temperature": 0.0},
        base_agent_llm_args={"temperature": 0.0},
        seed=300,
        max_steps=100,
        max_duration_seconds=900,
        max_errors=10,
        max_concurrency=5,
        num_trials=1,
        num_tasks=None,
        task_ids=None,
        task_split_name="base",
        domain_task_splits=None,
        auto_resume=True,
        web_search_context_size="medium",
        web_search_allowed_domains=None,
        web_search_user_location=None,
    )

    base_exp = tmp_path / "exp" / "responses" / "base-exp"
    repair_exp = tmp_path / "exp" / "responses" / "repair-exp"
    base_exp.mkdir(parents=True)
    repair_exp.mkdir(parents=True)
    (base_exp / "manifest.json").write_text(
        responses_sweep.json.dumps({"configs": [responses_sweep.asdict(target_spec)]})
    )

    results = Results(
        info=_make_info(),
        tasks=[_make_task("t0")],
        simulations=[_make_sim("t0")],
    )
    repair_raw_path = repair_exp / "raw" / f"{repair_spec.name}.json"
    repair_sim_path = (
        tmp_path / "simulations" / repair_spec.simulation_save_to / "results.json"
    )
    results.save(repair_raw_path, format="json")
    results.save(repair_sim_path, format="json")
    summary_row, simulation_rows = responses_sweep._summarize_results(
        results,
        repair_spec,
        raw_result_path=repair_raw_path,
        simulation_result_path=repair_sim_path,
    )
    responses_sweep._write_csv_atomic([summary_row], repair_exp / "results.csv")
    responses_sweep._write_csv_atomic(simulation_rows, repair_exp / "simulations.csv")

    result = canonicalize_response_repair_results(
        exp_dir="base-exp",
        repair_exp_dirs=["repair-exp"],
    )

    target_sim_path = (
        tmp_path / "simulations" / target_spec.simulation_save_to / "results.json"
    )
    target_raw_path = base_exp / "raw" / f"{target_spec.name}.json"
    assert target_spec.name in result["canonicalized"]
    assert target_sim_path.exists()
    assert target_raw_path.exists()
    assert validate_response_resume_state("base-exp") == []

    base_rows = responses_sweep._read_csv_rows(base_exp / "results.csv")
    assert base_rows[0]["name"] == target_spec.name
    assert base_rows[0]["simulation_result_path"] == str(target_sim_path)


def test_validate_response_resume_state_flags_noncanonical_path(monkeypatch, tmp_path):
    monkeypatch.setattr(responses_sweep, "DATA_DIR", tmp_path)
    exp = tmp_path / "exp" / "responses" / "base-exp"
    exp.mkdir(parents=True)
    responses_sweep._write_csv_atomic(
        [
            {
                "name": "config",
                "simulation_save_to": "exp/responses/base-exp/runs/config",
                "simulation_result_path": str(tmp_path / "other" / "results.json"),
            }
        ],
        exp / "results.csv",
    )

    problems = validate_response_resume_state("base-exp")

    assert len(problems) == 2
    assert "missing checkpoint" in problems[0]
    assert "simulation_result_path points at" in problems[1]


def test_dry_run_plan_reports_only_missing_checkpoint_tasks(monkeypatch, tmp_path):
    monkeypatch.setattr(responses_sweep, "DATA_DIR", tmp_path)
    config, spec = make_sweep_run_config(
        exp_name="dry-run-exp",
        llm="gpt-5.4-mini",
        domain="airline",
        mode=RunMode.DEFAULT,
        point=SweepPoint("medium", "medium", "off", "default"),
        llm_user="gpt-4.1-2025-04-14",
        llm_user_args={"temperature": 0.0},
        base_agent_llm_args={"temperature": 0.0},
        seed=300,
        max_steps=100,
        max_duration_seconds=900,
        max_errors=10,
        max_concurrency=5,
        num_trials=1,
        num_tasks=1,
        task_ids=None,
        task_split_name="base",
        domain_task_splits=None,
        auto_resume=True,
        web_search_context_size="medium",
        web_search_allowed_domains=None,
        web_search_user_location=None,
    )
    trial, task_id, seed = next(iter(_expected_simulation_keys(config)))

    plan = log_responses_dry_run_plan(
        [(config, spec)],
        auto_resume=True,
        reuse_from_exp_dirs=None,
    )

    assert plan["total"] == 1
    assert plan["configs"][0]["tasks"] == [
        {"trial": trial, "task_id": task_id, "seed": seed}
    ]

    checkpoint = tmp_path / "simulations" / spec.simulation_save_to / "results.json"
    Results(
        info=_make_info(),
        tasks=[_make_task(str(task_id))],
        simulations=[
            _make_sim(task_id, trial=trial, seed=seed),
        ],
    ).save(checkpoint, format="json")

    plan = log_responses_dry_run_plan(
        [(config, spec)],
        auto_resume=True,
        reuse_from_exp_dirs=None,
    )

    assert plan == {"total": 0, "configs": []}
