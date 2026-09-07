"""The intake_free (free-strategy) domain variant.

Covers, per docs/designs/intake-free-strategy.md:

- the variant tool surface: submit_fields drops confirmed_with_user from the
  SCHEMA entirely; the other agent tools are inherited unchanged;
- the new silent user-side measurement tools (note_spell_request,
  note_readback) and their invisibility to the agent-visible tool surface;
- the voice-only environment (channel="text" fails loud) and the new policy
  file (no spell/read-back mandates; the frozen policy files untouched);
- the in-memory task rewrite: swapped caller instructions with a drift
  guard, golden submit_fields args stripped, everything else byte-equal to
  the frozen set — which itself stays untouched on disk;
- golden replay through the evaluator: the rewritten golden actions score
  reward 1.0 against the variant environment;
- prompt rendering through the same seams the prompt-bed review packet uses
  (user_prompt_task + build_voice_user).
"""

import json

import pytest

from tau2.data_model.message import AssistantMessage, Message, ToolCall
from tau2.data_model.tasks import Task
from tau2.domains.intake.call_frame import USER_TASK_INSTRUCTIONS
from tau2.domains.intake.environment import get_tasks as intake_get_tasks
from tau2.domains.intake.free_strategy import (
    FREE_STRATEGY_POLICY_PATH,
    FREE_STRATEGY_USER_TASK_INSTRUCTIONS,
    FreeStrategyIntakeUserTools,
    _free_strategy_task_variant,
    get_environment,
    get_tasks,
    get_tasks_split,
)
from tau2.domains.intake.user_data_model import IntakeUserDB
from tau2.domains.intake.utils import (
    INTAKE_MAIN_POLICY_PATH,
    INTAKE_TEXT_POLICY_PATH,
)
from tau2.evaluator.evaluator_env import EnvironmentEvaluator

AGENT_TOOLS = {
    "get_callback_order",
    "log_capture",
    "submit_fields",
    "get_today",
    "end_call",
}
NOTE_TOOLS = {"note_spell_request", "note_readback"}


@pytest.fixture(scope="module")
def env():
    return get_environment()


@pytest.fixture(scope="module")
def tasks() -> list[Task]:
    return get_tasks()


@pytest.fixture(scope="module")
def canonical_tasks() -> list[Task]:
    return intake_get_tasks()


# ---------------------------------------------------------------------------
# Tool surface
# ---------------------------------------------------------------------------


def test_agent_tool_surface_unchanged_names(env):
    assert {t.name for t in env.get_tools()} == AGENT_TOOLS


def test_submit_fields_schema_drops_confirmed_with_user(env):
    tool = next(t for t in env.get_tools() if t.name == "submit_fields")
    params = tool.openai_schema["function"]["parameters"]
    assert set(params["properties"]) == {"record_id", "fields"}
    assert set(params["required"]) == {"record_id", "fields"}
    assert "confirmed_with_user" not in json.dumps(tool.openai_schema)


def test_user_tool_surface_adds_measurement_notes(env):
    assert {t.name for t in env.get_user_tools()} == {"get_entity", *NOTE_TOOLS}


def test_note_tools_never_on_agent_surface(env):
    """User-side measurement tools are invisible to the agent: absent from
    the agent tool list and from the assistant tool descriptions."""
    assert NOTE_TOOLS.isdisjoint({t.name for t in env.get_tools()})
    agent_desc = env.get_tools_description("assistant")
    for name in NOTE_TOOLS:
        assert name not in agent_desc
    user_desc = env.get_tools_description("user")
    for name in NOTE_TOOLS:
        assert name in user_desc


def test_note_tool_calls_route_as_user_requestor_only(env):
    """The orchestrator routes tool calls by requestor; a note call resolves
    through the USER toolkit and the agent toolkit has no such tool."""
    assert env.user_tools.has_tool("note_spell_request")
    assert env.user_tools.has_tool("note_readback")
    assert not env.tools.has_tool("note_spell_request")
    assert not env.tools.has_tool("note_readback")


def test_submit_fields_writes_like_canonical(env, tasks):
    """The variant write folds and fills exactly like the canonical desk."""
    task = tasks[0]
    init = task.initial_state
    env2 = get_environment()
    env2.set_state(
        initialization_data=init.initialization_data if init else None,
        initialization_actions=init.initialization_actions if init else None,
        message_history=[],
    )
    record_id, fields = _golden_submission(task)
    result = env2.use_tool("submit_fields", record_id=record_id, fields=fields)
    assert "Fields submitted" in result
    with pytest.raises(Exception, match="no missing fields"):
        env2.use_tool("submit_fields", record_id=record_id, fields=fields)


# ---------------------------------------------------------------------------
# User-side note tools
# ---------------------------------------------------------------------------


def _user_tools() -> FreeStrategyIntakeUserTools:
    tools = FreeStrategyIntakeUserTools(IntakeUserDB())
    tools.set_entities({"contact_phone": "555-014-2233"})
    return tools


def test_note_spell_request_known_field():
    assert (
        _user_tools().note_spell_request("contact_phone")
        == "Noted spell request for contact_phone."
    )


def test_note_readback_records_affirmation_state():
    tools = _user_tools()
    assert "affirmed" in tools.note_readback("contact_phone", i_affirmed=True)
    assert "corrected" in tools.note_readback("contact_phone", i_affirmed=False)


@pytest.mark.parametrize("tool", ["note_spell_request", "note_readback"])
def test_note_tools_fail_loud_on_unknown_field(tool):
    tools = _user_tools()
    kwargs = {"field": "shoe_size"}
    if tool == "note_readback":
        kwargs["i_affirmed"] = True
    with pytest.raises(ValueError, match="contact_phone"):
        getattr(tools, tool)(**kwargs)


# ---------------------------------------------------------------------------
# Environment: policy + voice-only
# ---------------------------------------------------------------------------


def test_environment_is_voice_only():
    with pytest.raises(ValueError, match="voice-only"):
        get_environment(channel="text")


def test_policy_is_the_new_file_with_no_protocol_mandates(env):
    from tau2.domains.intake.free_strategy import POLICY_VERIFICATION_SENTENCE

    policy = env.get_policy()
    assert policy == FREE_STRATEGY_POLICY_PATH.read_text()
    # The canonical (1.2.0) policy names verification once, with spelling as
    # a worked example, inside its one verification sentence — and NOWHERE
    # else: outside that sentence the rigid-protocol machinery must be
    # absent (no spelling mandate, no read-backs, no per-entity
    # confirmation, no attestation).
    assert policy.count(POLICY_VERIFICATION_SENTENCE) == 1
    stripped = policy.replace(POLICY_VERIFICATION_SENTENCE, "", 1)
    for banned in (
        "spell",
        "Spell",
        "verify",
        "read back",
        "read-back",
        "read it back",
        "confirmed_with_user",
        "Pinning the written form",
        "explicit yes",
    ):
        assert banned not in stripped, banned
    # The process rules the arm keeps.
    assert "log_capture" in policy
    assert "exactly one `submit_fields` call" in policy
    assert "never include `submit_fields`" in policy
    assert "submit nothing" in policy


def test_frozen_policy_files_untouched():
    """The canonical policies still carry the protocol (guard against an
    accidental edit of the frozen files while building the variant)."""
    main = INTAKE_MAIN_POLICY_PATH.read_text()
    assert "Pinning the written form" in main
    assert "letter by letter" in main
    assert "confirmed" in INTAKE_TEXT_POLICY_PATH.read_text()


# ---------------------------------------------------------------------------
# Task rewrite
# ---------------------------------------------------------------------------


def test_variant_swaps_instructions_on_every_task(tasks, canonical_tasks):
    assert len(tasks) == len(canonical_tasks) == 200
    for task in tasks:
        instructions = task.user_scenario.instructions
        assert instructions.task_instructions == FREE_STRATEGY_USER_TASK_INSTRUCTIONS


def test_variant_preserves_everything_else(tasks, canonical_tasks):
    for variant, canonical in zip(tasks, canonical_tasks):
        assert variant.id == canonical.id
        assert variant.agent_opener == canonical.agent_opener
        assert variant.user_scenario.persona == canonical.user_scenario.persona
        v_instr = variant.user_scenario.instructions
        c_instr = canonical.user_scenario.instructions
        assert v_instr.known_info == c_instr.known_info
        assert v_instr.unknown_info == c_instr.unknown_info
        assert v_instr.reason_for_call == c_instr.reason_for_call
        assert (
            variant.initial_state.model_dump() == canonical.initial_state.model_dump()
        )


def test_variant_strips_confirmed_with_user_from_golden_actions(tasks, canonical_tasks):
    for variant, canonical in zip(tasks, canonical_tasks):
        for action in variant.evaluation_criteria.actions or []:
            if action.name == "submit_fields":
                assert "confirmed_with_user" not in action.arguments
        # The canonical set still carries the attestation (deep-copy guard:
        # the rewrite must never mutate the shared loader's objects).
        for action in canonical.evaluation_criteria.actions or []:
            if action.name == "submit_fields":
                assert "confirmed_with_user" in action.arguments


def test_variant_instructions_carry_the_free_strategy_contract():
    text = FREE_STRATEGY_USER_TASK_INSTRUCTIONS
    assert "never volunteer a spelling" in text
    assert "note_spell_request" in text
    assert "note_readback" in text
    assert "i_affirmed" in text
    # The canonical grounding rules survive the swap.
    assert 'exactly "Hello?"' in text
    assert "get_entity" in text


def test_drift_guard_fails_loud_on_changed_frozen_instructions(canonical_tasks):
    drifted = canonical_tasks[0].model_copy(deep=True)
    drifted.user_scenario.instructions.task_instructions = "something else"
    with pytest.raises(ValueError, match="drifted"):
        _free_strategy_task_variant(drifted)


def test_splits_are_shared_with_canonical(tasks):
    splits = get_tasks_split()
    assert set(splits["base"]) == {t.id for t in tasks}


def test_canonical_frame_still_matches_frozen_set(canonical_tasks):
    """The drift guard's premise: every frozen task carries the canonical
    call-frame instructions verbatim."""
    for task in canonical_tasks:
        assert (
            task.user_scenario.instructions.task_instructions == USER_TASK_INSTRUCTIONS
        )


# ---------------------------------------------------------------------------
# Golden replay (reward plumbing)
# ---------------------------------------------------------------------------


def _golden_submission(task: Task) -> tuple[str, dict]:
    action = next(
        a for a in task.evaluation_criteria.actions if a.name == "submit_fields"
    )
    return action.arguments["record_id"], dict(action.arguments["fields"])


def _replay_reward(task: Task, actions: list[tuple[str, str, dict]]) -> float:
    env = get_environment()
    init = task.initial_state
    env.set_state(
        initialization_data=init.initialization_data if init else None,
        initialization_actions=init.initialization_actions if init else None,
        message_history=[],
    )
    messages: list[Message] = []
    for index, (requestor, name, arguments) in enumerate(actions):
        tool_call = ToolCall(
            id=f"sim_{index}", name=name, arguments=arguments, requestor=requestor
        )
        messages.append(
            AssistantMessage(role="assistant", content=None, tool_calls=[tool_call])
        )
        messages.append(env.get_response(tool_call))
    return EnvironmentEvaluator.calculate_reward(
        environment_constructor=get_environment,
        task=task,
        full_trajectory=messages,
    ).reward


@pytest.mark.parametrize("index", [0, 42, 199])
def test_golden_replay_scores_one(tasks, index):
    task = tasks[index]
    actions = [
        (a.requestor, a.name, json.loads(json.dumps(a.arguments)))
        for a in task.evaluation_criteria.actions
    ]
    assert _replay_reward(task, actions) == 1.0


def test_no_submission_scores_zero(tasks):
    assert _replay_reward(tasks[0], []) == 0.0


# ---------------------------------------------------------------------------
# Prompt rendering through the packet seams
# ---------------------------------------------------------------------------


def test_user_prompt_task_renders_variant_instructions(tasks):
    """user_prompt_task is the seam the prompt-bed review packet renders
    through; the variant instructions must survive it verbatim."""
    from tau2.data_model.simulation import AudioNativeConfig, VoiceRunConfig
    from tau2.runner.build import user_prompt_task

    config = VoiceRunConfig(
        domain="intake_free",
        task_set_name="intake_free",
        audio_native_config=AudioNativeConfig(),
        seed=42,
        # Complications default-armed: rate 0 keeps this render clean and
        # deterministic for the assertion below.
        complication_rate=0.0,
    )
    task = user_prompt_task(config, tasks[0], None)
    assert (
        task.user_scenario.instructions.task_instructions
        == FREE_STRATEGY_USER_TASK_INSTRUCTIONS
    )


def test_voice_user_system_prompt_carries_variant_frame(tasks):
    """Build the voice user simulator exactly as the runner (and the
    prompt-bed packet's _build_runtime_user) does, and check the rendered
    system prompt: variant instructions in, note tools available, no
    volunteered-spelling frame."""
    from tau2.data_model.simulation import AudioNativeConfig, VoiceRunConfig
    from tau2.runner.build import build_voice_user, user_prompt_task

    env = get_environment()
    config = VoiceRunConfig(
        domain="intake_free",
        task_set_name="intake_free",
        audio_native_config=AudioNativeConfig(),
        seed=42,
        complication_rate=0.0,
    )
    task = user_prompt_task(config, tasks[0], None)
    user = build_voice_user(
        env,
        task,
        config.audio_native_config,
        seed=42,
        persona_seed=42,
        domain="intake_free",
    )
    prompt = user.system_prompt
    assert "note_spell_request" in prompt
    assert "never volunteer a spelling or a written form" in prompt
    assert USER_TASK_INSTRUCTIONS not in prompt
    # The shared outbound guidelines' spell-offer nudge is stripped for the
    # spell-protocol-free domain (resolved Q3).
    assert "offer to spell it out letter by letter" not in prompt
    tool_names = {t.name for t in user.tools}
    assert {"get_entity", *NOTE_TOOLS} <= tool_names


def test_protocol_era_intake_caller_keeps_spell_offer_nudge(canonical_tasks):
    """The scoped override must not leak: the canonical intake caller still
    gets the shared outbound guidelines verbatim, nudge included."""
    from tau2.data_model.simulation import AudioNativeConfig, VoiceRunConfig
    from tau2.domains.intake.environment import (
        get_environment as canonical_get_environment,
    )
    from tau2.runner.build import build_voice_user, user_prompt_task

    env = canonical_get_environment()
    config = VoiceRunConfig(
        domain="intake",
        task_set_name="intake",
        audio_native_config=AudioNativeConfig(),
        seed=42,
        complication_rate=0.0,
    )
    task = user_prompt_task(config, canonical_tasks[0], None)
    user = build_voice_user(
        env,
        task,
        config.audio_native_config,
        seed=42,
        persona_seed=42,
        domain="intake",
    )
    assert "offer to spell it out letter by letter" in user.system_prompt


def test_strip_outbound_spell_offer_nudge_is_exact_and_fails_loud():
    from tau2.user.user_simulator import (
        OUTBOUND_SPELL_OFFER_NUDGE_LINE,
        CallDirection,
        get_global_user_sim_guidelines_voice,
        strip_outbound_spell_offer_nudge,
    )

    shared = get_global_user_sim_guidelines_voice(
        use_tools=True, direction=CallDirection.OUTBOUND
    )
    stripped = strip_outbound_spell_offer_nudge(shared)
    assert OUTBOUND_SPELL_OFFER_NUDGE_LINE not in stripped
    assert stripped == shared.replace(OUTBOUND_SPELL_OFFER_NUDGE_LINE, "")
    # Drift guard: stripping guidelines that no longer carry the exact line
    # must fail loud, never silently no-op.
    with pytest.raises(ValueError, match="retune"):
        strip_outbound_spell_offer_nudge(stripped)


def test_spell_event_detector_wired_for_intake_free_only(tasks):
    from tau2.data_model.simulation import (
        AudioNativeConfig,
        TextRunConfig,
        VoiceRunConfig,
    )
    from tau2.runner.complications import run_spell_event_detector

    voice_config = VoiceRunConfig(
        domain="intake_free",
        audio_native_config=AudioNativeConfig(),
        seed=42,
    )
    detector = run_spell_event_detector(voice_config, tasks[0])
    assert detector is not None
    assert detector.detect("") == []

    # Canonical intake stamps nothing — byte-identical runs.
    canonical_config = TextRunConfig(domain="intake", seed=42, workers=0)
    assert run_spell_event_detector(canonical_config, tasks[0]) is None


# ---------------------------------------------------------------------------
# Variant complication-density default (owner decision 2026-09-01)
# ---------------------------------------------------------------------------


def test_variant_pins_complication_rate_one():
    """Closed set: intake_free is the only domain with a pinned default rate
    (1.0 — every call draws, --complication-rate 1.0 semantics). The shared
    intake domain keeps its profile rates verbatim."""
    from tau2.config import DOMAIN_COMPLICATION_RATE

    assert DOMAIN_COMPLICATION_RATE == {"intake_free": 1.0}


def test_rate_one_density_over_the_variant_set(tasks):
    """At the pinned default the draw triggers on every fully-feasible task;
    constrained tasks keep the mass-to-none discount, so density over the
    frozen set is high but below 1 (144/200 at seed 42, catalog 2.4.0)."""
    from tau2.data_model.simulation import AudioNativeConfig, VoiceRunConfig
    from tau2.runner.complications import sample_run_complication

    config = VoiceRunConfig(
        domain="intake_free",
        task_set_name="intake_free",
        audio_native_config=AudioNativeConfig(),
        seed=42,
        complication_rate=1.0,
    )
    triggered = sum(
        1 for task in tasks if sample_run_complication(config, task) is not None
    )
    assert triggered / len(tasks) >= 0.65, triggered


def test_rate_one_renders_the_line_through_the_packet_seam(tasks):
    """user_prompt_task at the variant default injects the sampled line for
    injected kinds — the same seam the prompt-bed review packet renders
    through, so the owner reviews full-density prompts."""
    from tau2.data_model.simulation import AudioNativeConfig, VoiceRunConfig
    from tau2.runner.build import user_prompt_task
    from tau2.runner.complications import sample_run_complication

    config = VoiceRunConfig(
        domain="intake_free",
        task_set_name="intake_free",
        audio_native_config=AudioNativeConfig(),
        seed=42,
        complication_rate=1.0,
    )
    task = next(
        t
        for t in tasks
        if (s := sample_run_complication(config, t)) is not None and s.injected
    )
    sampled = sample_run_complication(config, task)
    prompt_task = user_prompt_task(config, task, None)
    instructions = prompt_task.user_scenario.instructions.task_instructions
    assert "Scripted complication" in instructions
    assert sampled.line in instructions


# ---------------------------------------------------------------------------
# Agent voice prompt: no prescribed elicitation strategy (resolved Q3)
# ---------------------------------------------------------------------------

# The shared section both spell-protocol-free instruction variants must drop,
# byte-for-byte. If the shared instructions are ever retuned, this literal
# (and the *_SPELL_PROTOCOL_FREE constants) must be retuned with them — the
# equality guards below fail loud on any drift in either direction.
_SPELL_PROTOCOL_SECTION = """# User authentication and user information collection

1. When collecting customer information (e.g. names, emails, IDs), ask the customer to spell it out letter by letter (e.g. "J, O, H, N") to ensure you have the correct information and accomodate for customer audio being unclear or background noise.

2. If authenticating the user fails based on user provided information, ALWAYS explicitly ask the customer to SPELL THINGS OUT or provide information LETTER BY LETTER (e.g. "first name J, O, H, N last name S, M, I, T, H")."""


def test_spell_protocol_free_instructions_are_shared_minus_the_section():
    from tau2.agent.discrete_time_audio_native_agent import (
        AUDIO_NATIVE_VOICE_INSTRUCTION,
        AUDIO_NATIVE_VOICE_INSTRUCTION_SPELL_PROTOCOL_FREE,
        CASCADED_MODEL_INSTRUCTION,
        CASCADED_MODEL_INSTRUCTION_SPELL_PROTOCOL_FREE,
    )

    assert AUDIO_NATIVE_VOICE_INSTRUCTION == (
        AUDIO_NATIVE_VOICE_INSTRUCTION_SPELL_PROTOCOL_FREE
        + "\n\n"
        + _SPELL_PROTOCOL_SECTION
    )
    assert CASCADED_MODEL_INSTRUCTION == (
        CASCADED_MODEL_INSTRUCTION_SPELL_PROTOCOL_FREE
        + "\n\n"
        + _SPELL_PROTOCOL_SECTION
    )


def _rendered_agent_system_prompt(environment) -> str:
    """Render the audio-native system prompt through the real seam
    (build_agent -> factory -> _build_system_prompt), exactly as a run
    builds it."""
    from tau2.data_model.simulation import AudioNativeConfig
    from tau2.runner.build import build_agent

    agent = build_agent(
        "discrete_time_audio_native_agent",
        environment,
        audio_native_config=AudioNativeConfig(
            provider="openai", reasoning_effort="xhigh"
        ),
    )
    return agent._build_system_prompt()


def test_intake_free_agent_prompt_has_no_spell_nudge(env):
    prompt = _rendered_agent_system_prompt(env)
    assert "spell it out letter by letter" not in prompt
    assert "SPELL THINGS OUT" not in prompt
    assert "User authentication and user information collection" not in prompt
    # The variant policy rides in the same prompt.
    assert "The record must end up correct." in prompt


def test_protocol_era_intake_agent_prompt_keeps_spell_nudge():
    from tau2.domains.intake.environment import (
        get_environment as canonical_get_environment,
    )

    prompt = _rendered_agent_system_prompt(canonical_get_environment())
    assert "spell it out letter by letter" in prompt
    assert "SPELL THINGS OUT" in prompt


# ---------------------------------------------------------------------------
# Canonical policy: verification sentence guard (no switches)
# ---------------------------------------------------------------------------

# The canonical policy's verification sentence (FREE_STRATEGY_POLICY_VERSION
# 1.2.0) and its anchor, byte-for-byte — the equality guards below fail loud
# on any rewording. There is exactly ONE mode-B policy: the ablation knob
# was measured once and removed by owner decision (see the design doc §3a).
_VERIFICATION_SENTENCE = (
    "Getting each value exactly right is what matters: if you are not sure "
    "you heard a value exactly, verify it with the person before recording "
    "it — for example by asking them to spell it letter by letter."
)
_VERIFICATION_ANCHOR = "How you get each value right is up to you."


def test_canonical_policy_version_and_sentence():
    from tau2.domains.intake.free_strategy import (
        FREE_STRATEGY_POLICY_VERSION,
        POLICY_VERIFICATION_SENTENCE,
    )

    assert FREE_STRATEGY_POLICY_VERSION == "1.2.0"
    assert POLICY_VERIFICATION_SENTENCE == _VERIFICATION_SENTENCE


def test_canonical_policy_carries_verification_sentence_once_anchored(env):
    """The one mode-B policy: the file verbatim, carrying the verification
    sentence exactly once, right after the anchor — asserted through the
    real agent render seam."""
    prompt = _rendered_agent_system_prompt(env)
    assert prompt.count(_VERIFICATION_SENTENCE) == 1
    assert f"{_VERIFICATION_ANCHOR} {_VERIFICATION_SENTENCE}" in prompt
    assert env.get_policy() == FREE_STRATEGY_POLICY_PATH.read_text()


def test_policy_load_fails_loud_on_verification_sentence_drift(tmp_path):
    """load_free_strategy_policy guards the versioned prose: a policy file
    without the anchored verification sentence must never load silently."""
    from unittest import mock

    from tau2.domains.intake import free_strategy

    drifted = tmp_path / "free_strategy_policy.md"
    drifted.write_text(
        FREE_STRATEGY_POLICY_PATH.read_text().replace(
            f" {_VERIFICATION_SENTENCE}", "", 1
        )
    )
    with mock.patch.object(free_strategy, "FREE_STRATEGY_POLICY_PATH", drifted):
        with pytest.raises(ValueError, match="drifted"):
            free_strategy.load_free_strategy_policy()
