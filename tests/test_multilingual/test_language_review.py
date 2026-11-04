"""Offline review boundaries use the recorded caller's complete language prompt."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tau2.data_model.language import (
    CallContext,
    LanguageDirective,
    LanguageTaskInfo,
    PersonaVoice,
    RuntimePersona,
)
from tau2.data_model.run_config import AudioNativeConfig
from tau2.data_model.semantic_communication import CommunicateScoring
from tau2.data_model.simulation import (
    AgentInfo,
    AuthenticationClassification,
    Info,
    Results,
    Review,
    SimulationRun,
    TerminationReason,
    UserInfo,
    UserOnlyReview,
    VoiceRunConfig,
)
from tau2.data_model.tasks import Task
from tau2.environment.environment import EnvironmentInfo
from tau2.user.guidelines import (
    finalize_system_prompt,
    language_reminder,
    render_simulation_guidelines,
)


def expected_review_guidelines(persona, *, use_tools):
    """The recorded caller's scaffold with its behavior and reminder, no scenario."""
    rendered = render_simulation_guidelines(persona, use_tools=use_tools).replace(
        "<PERSONA_GUIDELINES>", f"\n\n{persona.to_guidelines_text(persona.mode)}\n"
    )
    return finalize_system_prompt(persona, rendered)


@pytest.fixture(params=[False, True])
def recorded(request):
    voice = request.param
    content = json.loads(
        (Path(__file__).parent / "fixtures/reviewed_prompt_content.json").read_text()
    )["es"]
    mode = "voice" if voice else "text"
    persona = RuntimePersona(
        persona_id="speaker",
        mode=mode,
        gender="male",
        locale="Fixture",
        variety="Fixture",
        language="es",
        display_language="Spanish",
        gender_guidance=content["prompt_content"]["gender_guidance"],
        language_directive=LanguageDirective(origin="Fixture region"),
        pragmatics_clauses=["Recorded persona behavior"],
        mode_guidelines=f"Recorded {mode} domain guidance",
        spellout_section=content["spellout_section"] if voice else "",
        voice=PersonaVoice(
            voice_id="fixture",
            backchannel_phrases=["Fixture"],
            voice_examples=content["prompt_content"]["voice_examples"],
        )
        if voice
        else None,
    )
    task = Task(id="recorded-task", user_scenario={"instructions": "Recorded scenario"})
    context = LanguageTaskInfo(
        language="es",
        mode=mode,
        cross_lingual=True,
        pack_revision="fixture",
        pack_sha256="fixture",
        domain_revision="fixture",
        tasks_sha256="fixture",
        metadata_sha256="fixture",
        task_sha256=hashlib.sha256(
            json.dumps(
                task.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest(),
        canonical_task_id="canonical",
        agent_greeting="Fixture greeting",
        agent_language_clause="Fixture language",
        speech_insert_events_per_minute=1 if voice else None,
        backchannel_level="medium" if voice else None,
        backchannel_prompt_version="fixture" if voice else None,
        caller_gender="male",
        persona=persona,
        guideline_template_version="fixture",
    )
    info = Info(
        git_commit="fixture",
        num_trials=1,
        max_steps=100,
        max_errors=1,
        user_info=UserInfo(implementation="user_simulator"),
        agent_info=AgentInfo(implementation="fixture"),
        environment_info=EnvironmentInfo(
            domain_name="telecom", policy="Recorded policy"
        ),
        language="es",
        language_tasks={task.id: context},
        audio_native_config=AudioNativeConfig() if voice else None,
    )
    sim = SimulationRun(
        id="call",
        task_id=task.id,
        trial=0,
        start_time="a",
        end_time="b",
        duration=1,
        termination_reason=TerminationReason.USER_STOP,
        messages=[],
        language_context=context,
    )
    return Results(info=info, tasks=[task], simulations=[sim])


@pytest.mark.parametrize("mode", ["full", "user"])
@pytest.mark.parametrize("tools", [False, True])
def test_posthoc_review_resolves_owned_guidelines(recorded, monkeypatch, mode, tools):
    from tau2.multilingual import runtime
    from tau2.registry import registry
    from tau2.scripts import review_conversation

    # Per-call context wins over a different run-level assignment snapshot.
    other = recorded.info.language_tasks[recorded.tasks[0].id].model_copy(deep=True)
    other.persona.pragmatics_clauses = ["Different run snapshot behavior"]
    recorded.info.language_tasks[recorded.tasks[0].id] = other

    monkeypatch.setattr(
        runtime,
        "load_language",
        lambda *a: pytest.fail("Do not reload published packs"),
    )
    monkeypatch.setattr(
        registry,
        "get_env_constructor",
        lambda _: lambda **kw: SimpleNamespace(
            user_tools=object() if tools else None,
            get_user_tools=lambda **kw: [object()] if tools else [],
        ),
    )
    captured = []

    def judge(**kwargs):
        captured.append(kwargs["user_info"])
        if mode == "full":
            return Review(has_errors=False), AuthenticationClassification(
                status="not_needed"
            )
        return UserOnlyReview(has_errors=False), None

    monkeypatch.setattr(review_conversation, "review_simulation", judge)
    review = (
        review_conversation.review_simulation_full
        if mode == "full"
        else review_conversation.review_simulation_user
    )
    review(recorded.simulations[0], recorded.tasks[0], recorded)
    info = captured[0]
    persona = recorded.simulations[0].language_context.persona
    prompt_mode = "voice" if recorded.info.audio_native_config else "text"
    assert info.global_simulation_guidelines == expected_review_guidelines(
        persona, use_tools=tools
    )
    assert "Recorded scenario" not in info.global_simulation_guidelines
    assert recorded.info.user_info.global_simulation_guidelines is None
    assert info.persona_config == persona
    rendered = info.global_simulation_guidelines
    selected_guidance = f"Recorded {prompt_mode} domain guidance"
    other_mode = "text" if prompt_mode == "voice" else "voice"
    for clause in (
        persona.gender_guidance,
        "Recorded persona behavior",
        selected_guidance,
    ):
        assert rendered.count(clause) == 1
    assert "Different run snapshot behavior" not in rendered
    assert f"Recorded {other_mode} domain guidance" not in rendered
    if prompt_mode == "text":
        from tau2.user.user_simulator import UserSimulator

        monkeypatch.setattr(
            "tau2.user.user_simulator.generate",
            lambda *args, **kwargs: pytest.fail("Prompt construction called a model"),
        )
        actual_user = UserSimulator(
            llm="offline-fixture",
            instructions=recorded.tasks[0].user_scenario.instructions,
            tools=[] if tools else None,
            persona_config=persona,
        )
        # Review excludes the task scenario, but its instructions are exactly
        # the prefix that the real user received before that scenario/reminder.
        reviewed_prefix = rendered.removesuffix("\n\n" + language_reminder(persona))
        assert actual_user.system_prompt.startswith(reviewed_prefix)
        for clause in (
            persona.gender_guidance,
            "Recorded persona behavior",
            selected_guidance,
        ):
            assert actual_user.system_prompt.count(clause) == 1


@pytest.mark.parametrize("defect", ["missing", "wrong-task"])
def test_language_review_never_falls_back_to_english(recorded, monkeypatch, defect):
    from tau2.multilingual.runtime import recorded_review_user_info

    if defect == "missing":
        recorded.info.language_tasks = {}
        context = None
    else:
        context = recorded.simulations[0].language_context
        context.task_sha256 = "different"
    with pytest.raises(ValueError, match="language.*context"):
        recorded_review_user_info(recorded.info, recorded.tasks[0], context=context)


def test_auto_review_uses_actual_context_and_tools(recorded, monkeypatch):
    from tau2.evaluator import reviewer
    from tau2.runner import batch

    captured = []
    monkeypatch.setattr(
        reviewer,
        "review_simulation",
        lambda **kw: (
            captured.append(kw["user_info"]) or UserOnlyReview(has_errors=False),
            None,
        ),
    )
    monkeypatch.setattr(
        "tau2.user.guidelines.default_persona",
        lambda *a: pytest.fail("English fallback"),
    )
    context = recorded.simulations[0].language_context
    built = SimpleNamespace(
        call_context=CallContext(
            language=context, communicate_scoring=CommunicateScoring()
        ),
        user=SimpleNamespace(tools=[object()]),
        environment=SimpleNamespace(get_policy=lambda: "Recorded policy"),
    )
    monkeypatch.setattr(batch, "build_orchestrator", lambda *a, **kw: built)
    # This review unit uses a fully recorded synthetic orchestrator/context,
    # so its evaluator environment is supplied by the same fixture.
    monkeypatch.setattr(batch, "_build_env_kwargs", lambda *a, **kw: {})
    monkeypatch.setattr(
        batch, "run_simulation", lambda *a, **kw: recorded.simulations[0]
    )
    from tau2.data_model.simulation import TextRunConfig

    config = (
        VoiceRunConfig(
            domain="telecom", language="es", audio_native_config=AudioNativeConfig()
        )
        if recorded.info.audio_native_config
        else TextRunConfig(domain="telecom", language="es")
    )
    batch.run_single_task(
        config, recorded.tasks[0], seed=0, auto_review=True, review_mode="user"
    )
    prompt_mode = "voice" if recorded.info.audio_native_config else "text"
    assert captured[0].global_simulation_guidelines == expected_review_guidelines(
        context.persona, use_tools=True
    )
    assert language_reminder(context.persona) in (
        captured[0].global_simulation_guidelines
    )
    assert captured[0].persona_config == context.persona
    rendered = captured[0].global_simulation_guidelines
    assert rendered.count(context.persona.gender_guidance) == 1
    assert rendered.count("Recorded persona behavior") == 1
    assert rendered.count(f"Recorded {prompt_mode} domain guidance") == 1
    other_mode = "text" if prompt_mode == "voice" else "voice"
    assert f"Recorded {other_mode} domain guidance" not in rendered


def test_explicitly_disabled_tools_and_recorded_mapping_need_no_environment(
    recorded, monkeypatch
):
    from tau2.multilingual.runtime import recorded_review_user_info
    from tau2.registry import registry

    task = recorded.tasks[0]
    task.user_tools = []
    context = recorded.info.language_tasks[task.id]
    context.task_sha256 = hashlib.sha256(
        json.dumps(
            task.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    monkeypatch.setattr(
        registry,
        "get_env_constructor",
        lambda *a: pytest.fail("Explicitly no user tools"),
    )
    result = recorded_review_user_info(recorded.info, task)
    assert result.global_simulation_guidelines == expected_review_guidelines(
        context.persona, use_tools=False
    )


@pytest.mark.parametrize("override", ["ordinary", "elicitation"])
def test_non_language_and_elicitation_review_are_unchanged(
    recorded, monkeypatch, override
):
    from tau2.data_model.elicitation import ElicitationConfig
    from tau2.multilingual.runtime import recorded_review_user_info
    from tau2.registry import registry

    if override == "ordinary":
        # Results recorded before every run resolved a language context.
        recorded.info.language = None
        recorded.info.language_tasks = {}
    else:
        recorded.info.elicitation = ElicitationConfig()
    recorded.info.user_info.global_simulation_guidelines = (
        "Recorded ordinary/domain override"
    )
    monkeypatch.setattr(
        registry, "get_env_constructor", lambda *a: pytest.fail("Not a language review")
    )
    assert (
        recorded_review_user_info(recorded.info, recorded.tasks[0])
        is recorded.info.user_info
    )
