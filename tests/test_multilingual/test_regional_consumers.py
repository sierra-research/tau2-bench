"""Live, saved and reviewed runs select the same complete scenario binding."""

import json
import shutil
from pathlib import Path

import pytest
import yaml

from tau2.data_model.language import CallContext, LanguageDirective
from tau2.data_model.run_config import AudioNativeConfig
from tau2.data_model.semantic_communication import CommunicateScoring
from tau2.data_model.simulation import (
    Results,
    RewardInfo,
    SimulationRun,
    TerminationReason,
    TextRunConfig,
    VoiceRunConfig,
)
from tau2.data_model.tasks import Task
from tau2.environment.regional import REGIONAL_RUNTIME_VERSION
from tau2.multilingual.regional import PublishedRegionalBundle, canonical_scenario
from tau2.multilingual.runtime import load_language
from tau2.runner.batch import _load_run_tasks
from tau2.runner.build import _build_env_kwargs, build_environment
from tau2.runner.helpers import get_info


@pytest.mark.parametrize("change", ["database", "user_database", "policy", "runtime"])
def test_english_control_pins_complete_environment_and_rejects_drift(
    tmp_path, monkeypatch, change
):
    import tau2.utils
    from tau2.multilingual import regional
    from tau2.multilingual.runtime import (
        pin_language_artifacts,
        validate_language_resume,
    )
    from tau2.scripts.evaluate_trajectories import compute_simulation_rewards

    source_root = tau2.utils.DATA_DIR
    for relative in ("tau2/multilingual/en", "tau2/domains/telecom"):
        shutil.copytree(source_root / relative, tmp_path / relative)
    monkeypatch.setattr(tau2.utils, "DATA_DIR", tmp_path)
    config = TextRunConfig(domain="telecom", language="en", num_tasks=1)
    tasks = _load_run_tasks(config)
    before = get_info(config, tasks=tasks)
    pins = before.language_artifacts
    assert pins.scenario_mode == "pack"
    assert pins.scenario_sha256 and pins.runtime_rules_sha256
    assert pins.localization_map_sha256 is None
    results = Results(info=before, tasks=tasks, simulations=[])
    # The unchanged installed snapshot is sufficient for saved evaluation.
    assert compute_simulation_rewards(results).info == before

    directory = tmp_path / "tau2/domains/telecom"
    if change in {"database", "user_database"}:
        name, old, new = (
            ("db.toml", "price_per_month = 40.0", "price_per_month = 41.0")
            if change == "database"
            else ("user_db.toml", "battery_level = 80", "battery_level = 81")
        )
        path = directory / name
        assert old in path.read_text()
        path.write_text(path.read_text().replace(old, new, 1))
    elif change == "policy":
        path = directory / "main_policy.md"
        path.write_text(path.read_text() + "\nChanged canonical policy.\n")
    else:
        clock_source = Path(regional.__file__)
        original = Path.read_bytes
        old, new = b"datetime(2025, 2, 25, 12, 8)", b"datetime(2025, 2, 26, 12, 8)"
        assert old in original(clock_source)

        def changed_clock_bytes(path):
            content = original(path)
            return content.replace(old, new) if path == clock_source else content

        monkeypatch.setattr(Path, "read_bytes", changed_clock_bytes)

    after_config = config.model_copy(update={"language_artifacts": None})
    after = get_info(after_config, tasks=tasks)
    assert after.language_artifacts.pack_sha256 == pins.pack_sha256
    assert after.language_artifacts.tasks_sha256 == pins.tasks_sha256
    assert after.language_artifacts.metadata_sha256 == pins.metadata_sha256
    if change == "runtime":
        assert after.language_artifacts.scenario_sha256 == pins.scenario_sha256
        assert (
            after.language_artifacts.runtime_rules_version == pins.runtime_rules_version
        )
        assert (
            after.language_artifacts.runtime_rules_sha256 != pins.runtime_rules_sha256
        )
    else:
        assert after.language_artifacts.scenario_sha256 != pins.scenario_sha256
    with pytest.raises(ValueError, match="artifacts changed"):
        pin_language_artifacts(config)
    with pytest.raises(ValueError, match="resume|conditions"):
        validate_language_resume(before, after)
    with pytest.raises(ValueError, match="artifacts differ"):
        compute_simulation_rewards(results)


@pytest.fixture
def regional_retail(tmp_path, monkeypatch):
    import tau2.utils

    source_root = tau2.utils.DATA_DIR
    english = load_language("en", "retail", mode="voice")
    first = english.tasks[0]
    metadata = english.metadata[first.id]
    scenario = canonical_scenario("retail", [first], source_root)
    original_name = next(iter(scenario.database["users"].values()))["name"][
        "first_name"
    ]
    next(iter(scenario.database["users"].values()))["name"]["first_name"] = (
        "RegionalFixture"
    )
    scenario.tasks = [
        Task(id=first.id, user_scenario={"instructions": "Regional fixture caller."})
    ]
    scenario.policies = {"policy": "Regional fixture policy in KRW."}
    pack = english.pack.model_copy(deep=True)
    pack.language = "xx"
    pack.agent_language_clause = "Speak Example."
    pack.personas = {
        name: persona
        for name, persona in list(pack.personas.items())
        if metadata.caller_gender is None or persona.gender == metadata.caller_gender
    }
    selected_persona = next(iter(pack.personas))
    pack.personas = {
        selected_persona: pack.personas[selected_persona].model_copy(
            update={"language_directive": LanguageDirective(origin="Example region")}
        )
    }
    metadata = metadata.model_copy(update={"persona_id": selected_persona})
    binding = pack.domains["retail"].model_copy(
        update={
            "source": "file",
            "canonical_scenario": False,
            "tasks_file": None,
            "task_metadata_file": "metadata.json",
            "regional_bundle_file": "regional.json",
        }
    )
    pack.domains = {"retail": binding}
    pack.default_binding = None
    bundle = PublishedRegionalBundle(
        **scenario.model_dump(),
        revision="fixture",
        settings={
            "language": "xx",
            "country_code": "KR",
            "currency_code": "KRW",
            "currency_scale": "1000",
            "time_zone": "Asia/Seoul",
            "name_order": "family_first",
        },
        localization_map_sha256="a" * 64,
        source_document_sha256={},
        metadata={first.id: metadata},
        runtime_rules_version=REGIONAL_RUNTIME_VERSION,
    )
    root = tmp_path / "tau2/multilingual/xx"
    root.mkdir(parents=True)
    (root / "pack.yaml").write_text(
        yaml.safe_dump(pack.model_dump(mode="json"), allow_unicode=True)
    )
    (root / "metadata.json").write_text(
        json.dumps({first.id: metadata.model_dump(mode="json")})
    )
    (root / "regional.json").write_text(bundle.model_dump_json())
    shutil.copytree(
        source_root / "tau2/domains/retail", tmp_path / "tau2/domains/retail"
    )
    shutil.copytree(
        source_root / "tau2/multilingual/en", tmp_path / "tau2/multilingual/en"
    )
    monkeypatch.setattr(tau2.utils, "DATA_DIR", tmp_path)
    return root, original_name


@pytest.mark.parametrize("localize_db", [True, False])
def test_live_replay_and_prompt_review_share_scenario_and_detect_drift(
    regional_retail, localize_db, monkeypatch
):
    from tau2.multilingual.prompt_review import render_review
    from tau2.scripts import evaluate_trajectories as replay

    root, original_name = regional_retail
    config = VoiceRunConfig(
        domain="retail",
        language="xx",
        localize_db=localize_db,
        audio_native_config=AudioNativeConfig(),
    )
    tasks = _load_run_tasks(config)
    assert len(tasks) == 1
    expected_name = "RegionalFixture" if localize_db else original_name
    from tau2.multilingual.runtime import task_context

    environment = build_environment(
        "retail",
        env_kwargs=_build_env_kwargs(
            config, tasks[0], language_context=task_context(config, tasks[0])
        ),
    )
    assert (
        next(iter(environment.tools.db.users.values())).name.first_name == expected_name
    )
    assert ("Regional fixture" in str(tasks[0].user_scenario)) == localize_db
    info = get_info(config, tasks=tasks)
    assert info.localize_db == localize_db
    assert info.language == "xx"
    assert info.language_artifacts == config.language_artifacts
    assert info.environment_info.policy == environment.get_policy()
    assert config.language_artifacts.scenario_mode == (
        "regional" if localize_db else "canonical"
    )
    simulation = SimulationRun(
        id="fixture",
        task_id=tasks[0].id,
        start_time="2026-01-01T00:00:00",
        end_time="2026-01-01T00:01:00",
        duration=60,
        termination_reason=TerminationReason.USER_STOP,
        messages=[],
        language_context=info.language_tasks[tasks[0].id],
    )
    results = Results.model_validate_json(
        Results(info=info, tasks=tasks, simulations=[simulation]).model_dump_json()
    )
    seen = []

    def evaluate(**kwargs):
        replay_environment = build_environment(
            "retail", env_kwargs=kwargs["env_kwargs"]
        )
        seen.append(replay_environment)
        return RewardInfo(reward=1)

    monkeypatch.setattr(replay, "evaluate_simulation", evaluate)
    rescored = replay.compute_simulation_rewards(results)
    assert rescored.simulations[0].reward_info.reward == 1
    assert seen[0].tools.db == environment.tools.db
    assert seen[0].get_policy() == environment.get_policy()
    report = render_review(languages=["xx"], domains=["retail"], template=config)
    assert len(report.pairs) == 1
    pair = report.pairs[0]
    assert pair.config.localize_db == localize_db
    assert pair.language_context.artifact_pins == info.language_artifacts
    from tau2.multilingual.agent_guidance import localized_transfer_policy

    assert (
        localized_transfer_policy(environment.get_policy(), "retail", config.language)
        in pair.agent_prompt
    )
    assert "Speak Example." in pair.agent_prompt
    assert str(tasks[0].user_scenario) in pair.customer_prompt
    if localize_db:
        data = json.loads((root / "regional.json").read_text())
        data["policies"]["policy"] += " Changed."
        (root / "regional.json").write_text(json.dumps(data))
    else:
        policy = root.parents[1] / "domains/retail/policy.md"
        policy.write_text(policy.read_text() + "\nChanged canonical policy.")
    with pytest.raises(ValueError, match="artifacts|binding"):
        replay.compute_simulation_rewards(results)
    assert len(seen) == 1
    standalone = config.model_copy(update={"language_artifacts": None})
    with pytest.raises(ValueError, match="binding"):
        _build_env_kwargs(
            standalone, tasks[0], language_context=simulation.language_context
        )


def test_mode_is_a_recorded_condition_and_resume_cannot_switch_scenarios(
    regional_retail,
):
    from tau2.multilingual.runtime import validate_language_resume

    regional = VoiceRunConfig(
        domain="retail", language="xx", audio_native_config=AudioNativeConfig()
    )
    canonical = regional.model_copy(update={"localize_db": False}, deep=True)
    assert regional.localize_db is None and regional.effective_localize_db
    before = get_info(regional, tasks=_load_run_tasks(regional))
    after = get_info(canonical, tasks=_load_run_tasks(canonical))
    assert before.localize_db is None and after.localize_db is False
    assert before.language_artifacts.scenario_mode == "regional"
    assert after.language_artifacts.scenario_mode == "canonical"
    with pytest.raises(ValueError, match="resume|conditions"):
        validate_language_resume(before, after)


@pytest.mark.parametrize("localize_db", [True, False])
def test_direct_simulation_reopens_pinned_scenario_before_running(
    regional_retail, localize_db, monkeypatch
):
    from types import SimpleNamespace

    from tau2.multilingual.runtime import language_environment_kwargs, resolve_task
    from tau2.runner import simulation as running

    loaded = load_language("xx", "retail", localize_db=localize_db, mode="voice")
    task = loaded.tasks[0]
    config = VoiceRunConfig(
        domain="retail",
        language="xx",
        localize_db=localize_db,
        audio_native_config=AudioNativeConfig(),
    )
    context = resolve_task(loaded, task, config.seed)
    environment = build_environment(
        "retail", env_kwargs=_build_env_kwargs(config, task, language_context=context)
    )
    calls = []

    def run():
        calls.append("run")
        return SimulationRun(
            id="fixture",
            task_id=task.id,
            start_time="2026-01-01T00:00:00",
            end_time="2026-01-01T00:01:00",
            duration=60,
            termination_reason=TerminationReason.USER_STOP,
            messages=[],
        )

    def evaluate(**kwargs):
        actual = build_environment("retail", env_kwargs=kwargs["env_kwargs"])
        assert actual.tools.db == environment.tools.db
        assert actual.tools.db is not environment.tools.db
        assert actual.get_policy() == environment.get_policy()
        return RewardInfo(reward=1)

    monkeypatch.setattr(running, "evaluate_simulation", evaluate)
    orchestrator = SimpleNamespace(
        task=task,
        environment=environment,
        call_context=CallContext(
            language=context, communicate_scoring=CommunicateScoring()
        ),
        run=run,
    )
    wrong_database = language_environment_kwargs(loaded)
    next(iter(wrong_database["db"].users.values())).name.first_name = "Wrong caller"
    for overrides in ({}, wrong_database):
        with pytest.raises(ValueError, match="Explicit evaluation environment differs"):
            running.run_simulation(orchestrator, env_kwargs=overrides)
    assert calls == []
    assert running.run_simulation(orchestrator).reward_info.reward == 1
    assert (
        running.run_simulation(
            orchestrator, env_kwargs=language_environment_kwargs(loaded)
        ).reward_info.reward
        == 1
    )
    orchestrator.task = task.model_copy(deep=True)
    orchestrator.task.user_scenario.instructions = "Changed caller task."
    with pytest.raises(ValueError, match="task differs"):
        running.run_simulation(orchestrator)
    assert calls == ["run", "run"]


@pytest.mark.parametrize(
    ("domain", "extension", "task_id"),
    [
        ("banking_knowledge", "banking", "task_002_hi"),
        ("intake", "intake", "intake_person_names_easy_01_hi"),
    ],
)
def test_regional_evaluation_keeps_domain_constructor_settings(
    domain, extension, task_id, monkeypatch
):
    from types import SimpleNamespace

    from tau2.data_model.elicitation import ElicitationStrategy
    from tau2.multilingual.runtime import (
        TaskSelection,
        language_environment_kwargs,
        resolve_task,
    )
    from tau2.runner import simulation as running

    selection = TaskSelection(
        task_split_name="base",
        task_kwargs={"strategy": "agent_directed"} if domain == "intake" else {},
    )
    loaded = load_language(
        "hi", domain, mode="voice", selection=selection, pack_extension=extension
    )
    task = next(task for task in loaded.tasks if task.id == task_id)
    context = resolve_task(loaded, task, 42)
    kwargs = language_environment_kwargs(loaded)
    if domain == "banking_knowledge":
        kwargs.update(retrieval_variant="bm25", task=task, read_log_allowlist=set())
    else:
        kwargs.update(
            task=task, strategy=ElicitationStrategy.AGENT_DIRECTED, channel="voice"
        )
    seen = []

    def evaluate(**arguments):
        seen.append(arguments["env_kwargs"])
        return RewardInfo(reward=1)

    monkeypatch.setattr(running, "evaluate_simulation", evaluate)
    orchestrator = SimpleNamespace(
        task=task,
        environment=SimpleNamespace(
            get_domain_name=lambda: domain, get_policy=lambda: "policy"
        ),
        call_context=CallContext(
            language=context, communicate_scoring=CommunicateScoring()
        ),
        run=lambda: SimulationRun(
            id="fixture",
            task_id=task.id,
            start_time="2026-01-01T00:00:00",
            end_time="2026-01-01T00:01:00",
            duration=60,
            termination_reason=TerminationReason.USER_STOP,
            messages=[],
        ),
    )
    assert (
        running.run_simulation(orchestrator, env_kwargs=kwargs).reward_info.reward == 1
    )
    assert seen[0]["task"] == task
    assert seen[0]["db"] == kwargs["db"]
    assert seen[0]["db"] is not kwargs["db"]

    wrong_task = task.model_copy(deep=True)
    wrong_task.description = "Different task"
    with pytest.raises(ValueError, match="Evaluation task differs"):
        running.run_simulation(orchestrator, env_kwargs={**kwargs, "task": wrong_task})
