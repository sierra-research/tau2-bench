"""Offline integration coverage for runner, configuration, and checkpoints."""

import json
import os
import subprocess
import sys
from collections import deque
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from tau2.data_model.elicitation import ElicitationStrategy
from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage
from tau2.data_model.run_config import AudioNativeConfig
from tau2.data_model.simulation import (
    Results,
    SimulationRun,
    TerminationReason,
    TextRunConfig,
    VoiceRunConfig,
)
from tau2.domains.intake.agent_directed import get_tasks
from tau2.domains.intake.environment import load_frozen_tasks as scaffolded_tasks
from tau2.domains.intake.folds import SCORING_VERSION
from tau2.domains.intake.spell_events import SPELL_EVENT_DETECTION_VERSION
from tau2.evaluator.evaluator import EvaluationType
from tau2.multilingual.runtime import load_pack, task_context
from tau2.registry import registry
from tau2.runner.batch import (
    _load_run_tasks,
    make_voice_run_settings,
    run_single_task,
)
from tau2.runner.build import _build_env_kwargs
from tau2.runner.checkpoint import try_resume
from tau2.runner.elicitation import (
    sample_run_complication,
    user_prompt_task,
)
from tau2.runner.helpers import get_info


def _voice_config(**kwargs):
    return VoiceRunConfig(
        domain="intake",
        audio_native_config=AudioNativeConfig(),
        **kwargs,
    )


@pytest.mark.parametrize("domain", ["mock", "airline", "retail", "telecom"])
@pytest.mark.parametrize("voice", [False, True])
def test_existing_domains_are_noops_at_each_elicitation_seam(domain, voice):
    config = (
        VoiceRunConfig(domain=domain, audio_native_config=AudioNativeConfig())
        if voice
        else TextRunConfig(domain=domain)
    )
    tasks = registry.get_tasks_loader(domain)()
    task = tasks[0]
    assert sample_run_complication(config, task) is None
    assert user_prompt_task(config, task) is task
    env_kwargs = _build_env_kwargs(
        config, task, language_context=task_context(config, task)
    )
    assert "channel" not in env_kwargs
    assert "task" not in env_kwargs
    info = get_info(config)
    assert info.elicitation is None
    assert info.elicitation_scoring_version is None
    assert info.elicitation_spell_detection_version is None


@pytest.mark.parametrize("domain", ["mock", "airline", "retail", "telecom"])
def test_other_domains_keep_their_original_agent_policy(domain):
    from tau2.runner.build import build_text_orchestrator

    task = registry.get_tasks_loader(domain)()[0]
    orchestrator = build_text_orchestrator(TextRunConfig(domain=domain), task)
    expected_policy = registry.get_env_constructor(domain)().get_policy()
    assert orchestrator.environment.get_policy() == expected_policy
    # English runs carry the English pack's agent language clause, like any language.
    assert orchestrator.agent.domain_policy.startswith(expected_policy)
    assert orchestrator.agent.domain_policy.rstrip().endswith(
        load_pack("en").agent_language_clause
    )
    assert "# Callback order" not in orchestrator.agent.system_prompt


def test_text_agent_has_the_task_order_before_initialization():
    from tau2.domains.intake.data_model import CallbackOrder, IntakeDB
    from tau2.runner.build import build_text_orchestrator

    config = TextRunConfig(domain="intake", elicitation={"strategy": "scaffolded"})
    tasks = scaffolded_tasks()[:2]
    policies = []
    for task in tasks:
        original = task.model_dump_json()
        orchestrator = build_text_orchestrator(config, task)
        policy = orchestrator.environment.get_policy()
        assert policy in orchestrator.agent.system_prompt
        assert orchestrator.environment.tools.db == IntakeDB()
        order = CallbackOrder.model_validate_json(
            policy.split("# Callback order\n\n", 1)[1]
        )
        golden = task.evaluation_criteria.actions[0].arguments
        assert order.record_id == golden["record_id"]
        assert [field.name for field in order.missing_fields] == list(golden["fields"])
        assert not any(value in policy for value in golden["fields"].values())
        assert task.model_dump_json() == original
        policies.append(policy)
    assert policies[0] != policies[1]


@pytest.mark.parametrize("strategy", list(ElicitationStrategy))
@pytest.mark.parametrize("rate", [0, 1, None])
def test_strategy_and_rate_keep_both_tiers_in_the_full_frozen_frame(strategy, rate):
    settings = {"strategy": strategy, "rate": rate}
    full = _load_run_tasks(_voice_config(elicitation=settings))
    limited = _load_run_tasks(_voice_config(elicitation=settings, num_tasks=3))
    assert len(full) == 200
    assert sum("_hard_" in task.id for task in full) == 100
    assert sum("_easy_" in task.id for task in full) == 100
    assert [task.id for task in limited] == [task.id for task in full[:3]]
    selected_ids = [
        next(task.id for task in full if f"_{tier}_" in task.id)
        for tier in ("easy", "hard")
    ]
    selected = _load_run_tasks(
        _voice_config(elicitation=settings, task_ids=selected_ids)
    )
    assert {task.id for task in selected} == set(selected_ids)


@pytest.mark.parametrize("strategy", list(ElicitationStrategy))
def test_elicitation_task_set_must_belong_to_intake(strategy):
    config = VoiceRunConfig(
        domain="intake",
        task_set_name="intake",
        elicitation={"strategy": strategy},
        audio_native_config=AudioNativeConfig(),
    )
    assert _load_run_tasks(config) == registry.get_tasks_loader("intake")(
        strategy=strategy
    )
    mismatched = config.model_copy(update={"task_set_name": "mock"})
    with pytest.raises(ValueError, match="must match selected domain"):
        _load_run_tasks(mismatched)


def test_worker_roundtrip_replays_the_same_prompt_stimuli_without_task_mutation():
    config = _voice_config(seed=17)
    worker_config = VoiceRunConfig.model_validate_json(config.model_dump_json())
    for task in get_tasks():
        original = task.model_dump_json()
        sampled = sample_run_complication(config, task)
        assert sample_run_complication(worker_config, task) == sampled
        prompted = user_prompt_task(config, task)
        original_instructions = task.user_scenario.instructions.task_instructions
        if sampled is not None and sampled.injected:
            assert prompted.user_scenario.instructions.task_instructions == (
                original_instructions + "\n\n" + sampled.line
            )
        else:
            assert prompted is task
        assert task.model_dump_json() == original


def test_resolved_settings_and_scoring_are_recorded():
    config = _voice_config(seed=81, elicitation={"rate": 0.5})
    info = get_info(config)
    assert info.elicitation == config.elicitation
    assert info.elicitation_scoring_version == SCORING_VERSION
    assert info.elicitation_spell_detection_version == SPELL_EVENT_DETECTION_VERSION
    assert info.seed == 81
    assert info.audio_native_config == config.audio_native_config
    assert info.user_info.llm == config.llm_user
    assert info.user_info.llm_args == config.llm_args_user
    assert info.environment_info.domain_name == "intake"


@pytest.mark.parametrize("voice", [False, True])
def test_scaffolded_runs_do_not_record_agent_directed_spell_detection(voice):
    config = (
        VoiceRunConfig(
            domain="intake",
            elicitation={"strategy": "scaffolded"},
            audio_native_config=AudioNativeConfig(),
        )
        if voice
        else TextRunConfig(domain="intake", elicitation={"strategy": "scaffolded"})
    )
    assert get_info(config).elicitation_spell_detection_version is None


@pytest.mark.parametrize(
    "condition",
    [
        "rate",
        "strategy",
        "scoring",
        "prompt",
        "catalog",
        "spell_detection",
        "seed",
        "user_args",
        "audio_native",
    ],
)
def test_resume_refuses_mixed_elicitation_conditions(tmp_path, condition):
    config = _voice_config(seed=81)
    info = get_info(config)
    task = get_tasks()[0]
    path = tmp_path / "results.json"
    Results(info=info, tasks=[task], simulations=[]).save(path)
    changed = info.model_copy(deep=True)
    if condition == "rate":
        changed.elicitation.rate = 0
    elif condition == "strategy":
        changed.elicitation.strategy = ElicitationStrategy.SCAFFOLDED
    elif condition == "scoring":
        changed.elicitation_scoring_version = "future"
    elif condition == "prompt":
        changed.elicitation_prompt_version = "future"
    elif condition == "catalog":
        changed.elicitation_catalog_version = "future"
    elif condition == "spell_detection":
        changed.elicitation_spell_detection_version = "future"
    elif condition == "seed":
        changed.seed += 1
    elif condition == "user_args":
        changed.user_info.llm_args["temperature"] = 0.5
    else:
        changed.audio_native_config.reasoning_effort = "high"
    # The language run-condition check may refuse first; either refusal is correct.
    with pytest.raises(ValueError, match="[Ee]licitation|run conditions changed"):
        try_resume(
            path,
            Results(info=changed, tasks=[task], simulations=[]),
            tasks=[task],
            num_trials=1,
            auto_resume=True,
        )


def test_unchanged_resume_keeps_recorded_stimulus_and_skips_completed_trial(tmp_path):
    config = _voice_config(seed=81)
    task = get_tasks()[0]
    simulation = SimulationRun(
        id="completed",
        task_id=task.id,
        trial=0,
        seed=4,
        start_time="2026-09-17T00:00:00",
        end_time="2026-09-17T00:00:01",
        duration=1,
        termination_reason=TerminationReason.AGENT_STOP,
        messages=[],
        complication=sample_run_complication(config, task),
    )
    info = get_info(config)
    path = tmp_path / "results.json"
    Results(info=info, tasks=[task], simulations=[simulation]).save(path)
    result, done, _ = try_resume(
        path,
        Results(info=info, tasks=[task], simulations=[]),
        tasks=[task],
        num_trials=1,
        auto_resume=True,
    )
    assert done == {(0, task.id, 4)}
    assert result.simulations[0].complication == simulation.complication


def test_voice_resume_accepts_settings_after_checkpoint_json_round_trip(tmp_path):
    config = _voice_config(seed=81)
    task = get_tasks()[0]
    voice_settings, persona_config = make_voice_run_settings(config)
    info = get_info(
        config,
        user_voice_settings=voice_settings,
        user_persona_config=persona_config,
    )
    path = tmp_path / "results.json"
    Results(info=info, tasks=[task], simulations=[]).save(path)

    resumed, done, _ = try_resume(
        path,
        Results(info=info, tasks=[task], simulations=[]),
        tasks=[task],
        num_trials=1,
        auto_resume=True,
    )

    assert done == set()
    assert resumed.info.user_info.voice_settings is not None


def test_text_runner_executes_real_tools_evaluation_and_stimulus_recording(monkeypatch):
    """Mock LLM replies only; use the real agents, orchestrator, tools and reward."""
    from tau2.agent import llm_agent
    from tau2.user import user_simulator

    task = scaffolded_tasks()[0]
    original = task.model_dump_json()
    action = task.evaluation_criteria.actions[0]
    field, value = next(iter(action.arguments["fields"].items()))

    def tool_message(name, arguments):
        return AssistantMessage(
            role="assistant",
            content=None,
            cost=0,
            tool_calls=[ToolCall(id=name, name=name, arguments=arguments)],
        )

    agent_replies = deque(
        [
            AssistantMessage(
                role="assistant", content=f"What is your {field}?", cost=0
            ),
            tool_message("log_capture", {"field_name": field, "value": value}),
            tool_message(action.name, deepcopy(action.arguments)),
            tool_message("end_call", {}),
        ]
    )
    user_replies = deque(
        [
            AssistantMessage(role="assistant", content="Hello?", cost=0),
            tool_message("get_entity", {"field": field}),
            AssistantMessage(role="assistant", content=value, cost=0),
        ]
    )
    observed_user_prompts = []
    observed_agent_prompts = []

    def agent_generate(**kwargs):
        from tau2.utils.utils import get_now

        observed_agent_prompts.append(kwargs["messages"][0].content)
        message = agent_replies.popleft()
        message.timestamp = get_now()
        return message

    def user_generate(**kwargs):
        observed_user_prompts.append(kwargs["messages"][0].content)
        return user_replies.popleft()

    monkeypatch.setattr(llm_agent, "generate", agent_generate)
    monkeypatch.setattr(user_simulator, "generate", user_generate)
    config = TextRunConfig(
        domain="intake", seed=17, elicitation={"strategy": "scaffolded", "rate": 1}
    )
    simulation = run_single_task(
        config, task, seed=5, evaluation_type=EvaluationType.ENV
    )
    assert simulation.reward_info.reward == 1
    assert simulation.termination_reason == TerminationReason.AGENT_STOP
    assert simulation.messages[0].content == task.agent_opener
    assert simulation.complication == sample_run_complication(config, task)
    if simulation.complication is not None and simulation.complication.injected:
        assert simulation.complication.line in observed_user_prompts[0]
    assert task.model_dump_json() == original
    assert simulation.policy in observed_agent_prompts[0]
    assert "# Callback order\n\n" in observed_agent_prompts[0]
    assert action.arguments["record_id"] in observed_agent_prompts[0]
    assert value not in observed_agent_prompts[0]
    assert not any(
        call.name == "get_callback_order"
        for message in simulation.messages
        for call in (getattr(message, "tool_calls", None) or [])
    )
    assert not agent_replies and not user_replies


def test_mixed_submit_and_stop_is_a_scored_protocol_failure(monkeypatch):
    """Reject the mixed turn without executing or replaying its write action."""
    from tau2.agent import llm_agent
    from tau2.runner import batch
    from tau2.user import user_simulator
    from tau2.utils.utils import get_now

    task = scaffolded_tasks()[0]
    original = task.model_dump_json()
    action = task.evaluation_criteria.actions[0]
    orchestrators = []
    original_builder = batch.build_orchestrator

    def capture_orchestrator(*args, **kwargs):
        orchestrator = original_builder(*args, **kwargs)
        orchestrators.append(orchestrator)
        return orchestrator

    def agent_generate(**kwargs):
        return AssistantMessage(
            role="assistant",
            timestamp=get_now(),
            cost=0,
            tool_calls=[
                ToolCall(
                    id="submit", name=action.name, arguments=deepcopy(action.arguments)
                ),
                ToolCall(id="stop", name="end_call", arguments={}),
            ],
        )

    monkeypatch.setattr(batch, "build_orchestrator", capture_orchestrator)
    monkeypatch.setattr(llm_agent, "generate", agent_generate)
    monkeypatch.setattr(
        user_simulator,
        "generate",
        lambda **kwargs: AssistantMessage(
            role="assistant",
            content="Hello?",
            cost=0,
        ),
    )
    config = TextRunConfig(
        domain="intake", elicitation={"strategy": "scaffolded", "rate": 0}
    )
    simulation = run_single_task(config, task, evaluation_type=EvaluationType.ENV)
    assert simulation.termination_reason == TerminationReason.AGENT_STOP
    assert simulation.reward_info.reward == 0
    responses = [
        message for message in simulation.messages if isinstance(message, ToolMessage)
    ]
    assert {response.id for response in responses} == {"submit", "stop"}
    assert all(
        response.error and response.execution_status == "rejected"
        for response in responses
    )
    assert (
        len(orchestrators[0].environment.tools.db.intake_records[0].missing_fields) == 1
    )
    assert task.model_dump_json() == original


@pytest.mark.parametrize("strategy", [None, "scaffolded"])
def test_cli_strategy_reaches_live_orchestrator_with_main_provider_settings(
    monkeypatch,
    strategy,
):
    """Build both real participants; no connection, audio generation or LLM call."""
    pytest.importorskip("elevenlabs")
    pytest.importorskip("aiortc")
    from tau2 import cli
    from tau2.data_model.live_config import LiveConfig
    from tau2.domains.intake.agent_directed import AGENT_DIRECTED_USER_TASK_INSTRUCTIONS
    from tau2.domains.intake.call_frame import USER_TASK_INSTRUCTIONS
    from tau2.domains.intake.data_model import CallbackOrder, IntakeDB
    from tau2.runner.build import build_voice_orchestrator
    from tau2.voice.audio_native.openai.live_provider import OpenAILiveProvider

    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-only")
    live = LiveConfig(backend_model="gpt-6-astra")
    captured = []
    monkeypatch.setattr(cli, "run_domain", captured.append)
    invocation = [
        "tau2",
        "run",
        "--domain",
        "intake",
        "--audio-native",
        "--benchmark",
        "none",
        "--speech-complexity",
        "control",
        "--audio-native-provider",
        "openai_live",
        "--audio-native-model",
        "gpt-live-1",
        "--reasoning-effort",
        "medium",
        "--live-config",
        live.model_dump_json(),
    ]
    if strategy is not None:
        invocation += ["--elicitation-strategy", strategy]
    monkeypatch.setattr(sys, "argv", invocation)
    cli.main()
    config = captured[0]
    agent_directed = strategy is None
    expected_strategy = (
        ElicitationStrategy.AGENT_DIRECTED
        if agent_directed
        else ElicitationStrategy.SCAFFOLDED
    )
    assert config.elicitation.strategy == expected_strategy
    assert config.elicitation.rate == 1
    task = _load_run_tasks(config)[0]
    expected_instructions = (
        AGENT_DIRECTED_USER_TASK_INSTRUCTIONS
        if agent_directed
        else USER_TASK_INSTRUCTIONS
    )
    assert task.user_scenario.instructions.task_instructions == expected_instructions
    orchestrator = build_voice_orchestrator(config, task, seed=0)
    try:
        assert orchestrator.user.realtime_generation is True
        assert (orchestrator.user.spell_event_detector is not None) == agent_directed
        assert orchestrator.agent.live_config == live
        assert "end_call" in orchestrator.agent.additional_stop_tool_names
        prompt = orchestrator.agent._build_system_prompt()
        assert ("The record must end up correct." in prompt) == agent_directed
        user_tool_names = {
            tool.name for tool in orchestrator.environment.get_user_tools()
        }
        assert ("note_readback" in user_tool_names) == agent_directed
        assert (
            "offer to spell it out letter by letter" in orchestrator.user.system_prompt
        ) == (not agent_directed)
        assert expected_instructions in orchestrator.user.system_prompt
        submit = next(
            tool
            for tool in orchestrator.environment.get_tools()
            if tool.name == "submit_fields"
        )
        assert (
            "confirmed_with_user"
            in submit.openai_schema["function"]["parameters"]["properties"]
        ) == (not agent_directed)
        policy = orchestrator.environment.get_policy()
        assert policy in prompt
        order = CallbackOrder.model_validate_json(
            policy.split("# Callback order\n\n", 1)[1]
        )
        golden = task.evaluation_criteria.actions[0].arguments
        assert order.record_id == golden["record_id"]
        assert [field.name for field in order.missing_fields] == list(golden["fields"])
        assert not any(value in prompt for value in golden["fields"].values())
        assert orchestrator.environment.tools.db == IntakeDB()
        provider = OpenAILiveProvider(
            model=orchestrator.agent.model,
            config=orchestrator.agent.live_config,
            reasoning_effort=orchestrator.agent.reasoning_effort,
            api_key="offline-test-only",
        )
        session = provider._build_session(prompt, orchestrator.agent.tools)
        backend = session["delegation"]["responses"]
        assert session["model"] == "gpt-live-1"
        assert backend["model"] == "gpt-6-astra"
        assert backend["reasoning"] == {"effort": "medium"}
        assert "submit_fields" in {tool["name"] for tool in backend["tools"]}
        assert "get_callback_order" not in {tool["name"] for tool in backend["tools"]}
        assert orchestrator.task is task
    finally:
        orchestrator.user._user_executor.shutdown(wait=True)


def test_elicitation_voice_draw_is_stable_across_python_worker_hash_seeds():
    """Exercise the real builder up to sampling in independent worker interpreters."""
    script = """
import json
from tau2.data_model.simulation import AudioNativeConfig, VoiceRunConfig
from tau2.multilingual.runtime import task_context
from tau2.registry import registry
from tau2.runner import build

class Captured(Exception):
    pass

draws = {}
def capture(*, domain, task_id, task_seed, **kwargs):
    draws[domain] = {"seed": task_seed}
    raise Captured

build.get_or_load_task_voice_config = capture
for domain in ("intake", "mock"):
    environment = registry.get_env_constructor(domain)()
    task = registry.get_tasks_loader(domain)()[0]
    try:
        config = VoiceRunConfig(
            domain=domain, seed=7, audio_native_config=AudioNativeConfig()
        )
        build.build_voice_user(
            environment,
            task,
            config.audio_native_config,
            language_context=task_context(config, task),
            seed=7,
        )
    except Captured:
        pass
print(json.dumps(draws))
"""
    outputs = []
    for hash_seed in ("1", "999"):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = hash_seed
        process = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parents[2],
            env=environment,
        )
        outputs.append(json.loads(process.stdout.strip().splitlines()[-1]))
    # Every domain (English included) draws from the canonical task identity.
    for domain in ("intake", "mock"):
        assert outputs[0][domain]["seed"] == outputs[1][domain]["seed"]


@pytest.mark.parametrize("voice", [False, True])
def test_cli_preserves_other_domain_defaults(monkeypatch, voice):
    from tau2 import cli
    from tau2.config import (
        DEFAULT_LLM_ARGS_USER,
        DEFAULT_LLM_USER,
        DEFAULT_MAX_STEPS_SECONDS,
    )

    captured = []
    monkeypatch.setattr(cli, "run_domain", captured.append)
    monkeypatch.setattr(
        sys,
        "argv",
        ["tau2", "run", "--domain", "retail"]
        + (["--audio-native", "--benchmark", "none"] if voice else []),
    )
    cli.main()
    config = captured[0]
    assert config.elicitation is None
    assert config.llm_user == DEFAULT_LLM_USER
    assert config.llm_args_user == DEFAULT_LLM_ARGS_USER
    if voice:
        assert config.audio_native_config.max_steps_seconds == DEFAULT_MAX_STEPS_SECONDS


def test_cli_resolves_elicitation_defaults_and_explicit_overrides(monkeypatch):
    from tau2 import cli

    captured = []
    monkeypatch.setattr(cli, "run_domain", captured.append)
    base = [
        "tau2",
        "run",
        "--domain",
        "intake",
        "--audio-native",
        "--benchmark",
        "none",
    ]
    monkeypatch.setattr(sys, "argv", base)
    cli.main()
    assert captured[-1].elicitation.rate == 1
    assert captured[-1].elicitation.strategy == ElicitationStrategy.AGENT_DIRECTED
    assert captured[-1].llm_user == "gpt-5.5"
    assert captured[-1].audio_native_config.max_steps_seconds == 240
    monkeypatch.setattr(
        sys,
        "argv",
        base
        + [
            "--elicitation-strategy",
            "scaffolded",
            "--complication-rate",
            "0",
            "--user-llm",
            "explicit-user",
            "--user-llm-args",
            "{}",
            "--max-steps-seconds",
            "90",
        ],
    )
    cli.main()
    config = captured[-1]
    assert config.elicitation.strategy == ElicitationStrategy.SCAFFOLDED
    assert config.elicitation.rate == 0
    assert config.llm_user == "explicit-user"
    assert config.llm_args_user == {}
    assert config.audio_native_config.max_steps_seconds == 90


def test_cli_rejects_removed_complication_profile(monkeypatch):
    from tau2 import cli

    def refuse_run(config):
        pytest.fail("An unsupported complication profile must fail before running")

    monkeypatch.setattr(cli, "run_domain", refuse_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tau2",
            "run",
            "--domain",
            "intake",
            "--audio-native",
            "--complication-profile",
            "hard",
        ],
    )
    with pytest.raises(SystemExit):
        cli.main()


@pytest.mark.parametrize(
    "task_opener,expected", [("Task opening", "Task opening"), (None, "Pack greeting")]
)
@pytest.mark.parametrize("audio_factory", [False, True])
def test_voice_task_opener_precedes_pack_greeting(task_opener, expected, audio_factory):
    from types import SimpleNamespace

    from tau2.orchestrator.full_duplex_orchestrator import FullDuplexOrchestrator

    orchestrator = object.__new__(FullDuplexOrchestrator)
    orchestrator.task = SimpleNamespace(agent_opener=task_opener)
    orchestrator.call_context = SimpleNamespace(
        language=SimpleNamespace(agent_greeting="Pack greeting")
    )
    orchestrator.agent = SimpleNamespace(
        get_init_state=lambda message_history: message_history
    )
    if audio_factory:
        orchestrator.agent.create_initial_message = (
            lambda content="Default": AssistantMessage(
                role="assistant", content=content
            )
        )
    orchestrator.user = SimpleNamespace(
        get_init_state=lambda: None,
        get_next_chunk=lambda state, participant_chunk: (None, state),
    )
    orchestrator.tick_duration_seconds = 0.1
    orchestrator._initialize_full_duplex()
    assert orchestrator.ticks[0].agent_chunk.content == expected
    assert orchestrator.initial_agent_message.content == "Pack greeting"


@pytest.mark.parametrize(
    "task_opener,expected", [("Task opening", "Task opening"), (None, "Pack greeting")]
)
def test_text_task_opener_precedes_pack_greeting(task_opener, expected):
    from tau2.runner.build import build_text_orchestrator

    bound = registry.get_tasks_loader("mock")()[0]
    orchestrator = build_text_orchestrator(TextRunConfig(domain="mock"), bound)
    orchestrator.task = bound.model_copy(
        update={"initial_state": None, "agent_opener": task_opener}
    )
    orchestrator.call_context = SimpleNamespace(
        language=SimpleNamespace(agent_greeting="Pack greeting")
    )
    orchestrator.initialize()
    assert orchestrator.trajectory[0].content == expected
    assert orchestrator.initial_agent_message.content == "Pack greeting"


def test_language_voice_builder_preserves_explicit_zero_seed(monkeypatch):
    from types import SimpleNamespace

    from tau2.data_model.tasks import Task
    from tau2.multilingual import runtime
    from tau2.runner import build

    context = SimpleNamespace(
        language="en",
        persona=object(),
        speech_insert_events_per_minute=1.0,
        agent_greeting="Fixture greeting",
    )
    monkeypatch.setattr(runtime, "task_context", lambda *_, **__: context)
    monkeypatch.setattr(runtime, "agent_language_guidance", lambda *_, **__: "")
    monkeypatch.setattr(
        build,
        "call_context",
        lambda *_: SimpleNamespace(agent_voice_gender=None),
    )
    monkeypatch.setattr(build, "build_environment", lambda *a, **kw: object())
    monkeypatch.setattr(build, "_build_env_kwargs", lambda *a, **kw: {})
    monkeypatch.setattr(build, "build_agent", lambda *a, **kw: object())
    observed = []

    def voice_user(*args, **kwargs):
        observed.append(kwargs)
        return object()

    monkeypatch.setattr(build, "build_voice_user", voice_user)
    monkeypatch.setattr(
        build, "FullDuplexOrchestrator", lambda **kw: SimpleNamespace(**kw)
    )
    config = VoiceRunConfig(
        domain="retail", seed=0, audio_native_config=AudioNativeConfig()
    )
    task = Task(id="fixture", user_scenario={"instructions": "Fixture."})
    result = build.build_voice_orchestrator(config, task, seed=0)
    assert observed[0]["language_context"] is context
    assert observed[0]["seed"] == result.seed == 0
