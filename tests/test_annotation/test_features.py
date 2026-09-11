# Copyright Sierra
"""Tier-1 feature extraction on constructed tick fixtures (known timestamps
-> known latency/silence/interruption values) + table provenance."""

import importlib.util
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest
from fixtures_runs import make_hi_results

from tau2.annotation.features import (
    AUTH_TOOLS_BY_DOMAIN,
    BACKCHANNEL_REFERENCE_PER_MIN,
    EXTRACTED_FEATURES,
    FEATURE_EXTRACTOR_VERSION,
    JUDGED_FEATURE_SLOTS,
    Arm,
    AuthLookupOutcome,
    CallFeatures,
    FeatureExtractionConfig,
    build_feature_table,
    classify_auth_lookup,
    export_feature_table,
    extract_call_features,
    load_feature_table,
    resolve_backchannel_reference_per_min,
)
from tau2.backchannel import BackchannelLevel
from tau2.config import ReasoningEffort
from tau2.data_model.message import (
    AssistantMessage,
    Tick,
    ToolCall,
    ToolMessage,
    TurnTakingAction,
    UserMessage,
)
from tau2.data_model.simulation import (
    AudioNativeConfig,
    NativenessInfo,
    ReasoningEffortBackfill,
    ReasoningEffortSource,
    RewardInfo,
    SimulationRun,
    TerminationReason,
)
from tau2.data_model.voice import VoiceSettings

DUR = 0.2  # tick clock used by every fixture

#: The audio-native block a real pinned run carries. Every voice fixture below
#: passes one: a run with a provider but no such block is exactly the artifact
#: extraction now refuses, and ``test_extraction_refuses_*`` owns that case.
OPENAI_XHIGH = AudioNativeConfig(
    provider="openai",
    model="gpt-realtime-2",
    reasoning_effort=ReasoningEffort.XHIGH,
)
#: What the runner records in ``agent_info.llm`` for that arm — already
#: provider-prefixed, as real runs are.
OPENAI_LLM = "openai:gpt-realtime-2"


def make_tick(
    i: int,
    *,
    agent_speech: bool = False,
    user_speech: bool = False,
    user_action: Optional[str] = None,
    agent_tool_calls: Optional[list[ToolCall]] = None,
    agent_tool_results: Optional[list[ToolMessage]] = None,
) -> Tick:
    agent_chunk = (
        AssistantMessage.voice(content="a", is_audio=False, contains_speech=True)
        if agent_speech
        else None
    )
    user_chunk = None
    if user_speech or user_action:
        user_chunk = UserMessage.voice(
            content="u" if user_speech else None,
            is_audio=False,
            contains_speech=user_speech,
            turn_taking_action=(
                TurnTakingAction(action=user_action) if user_action else None
            ),
        )
    return Tick(
        tick_id=i,
        timestamp=f"2026-01-01T00:00:{i:02d}",
        tick_duration_seconds=DUR,
        agent_chunk=agent_chunk,
        user_chunk=user_chunk,
        agent_tool_calls=agent_tool_calls or [],
        agent_tool_results=agent_tool_results or [],
    )


def ticks_from_pattern(
    pattern: str,
    *,
    bc_ticks: Optional[set[int]] = None,
) -> list[Tick]:
    """Build ticks from a compact per-tick pattern string:
    'a' agent speech, 'u' user speech, 'b' both, '.' silence."""
    ticks = []
    for i, ch in enumerate(pattern):
        ticks.append(
            make_tick(
                i,
                agent_speech=ch in "ab",
                user_speech=ch in "ub",
                user_action=("backchannel" if bc_ticks and i in bc_ticks else None),
            )
        )
    return ticks


def make_sim(
    ticks: Optional[list[Tick]],
    *,
    sim_id: str = "sim1",
    termination: TerminationReason = TerminationReason.USER_STOP,
) -> SimulationRun:
    return SimulationRun(
        id=sim_id,
        task_id="task_1",
        trial=0,
        start_time="2026-01-01T00:00:00",
        end_time="2026-01-01T00:05:00",
        duration=300.0,
        termination_reason=termination,
        reward_info=RewardInfo(reward=1.0),
        ticks=ticks,
        mode="full_duplex",
    )


def features_for(pattern: str, *, language: str = "en", **kwargs) -> CallFeatures:
    sim = make_sim(ticks_from_pattern(pattern, **kwargs))
    return extract_call_features(sim, domain="telecom", language=language)


@pytest.fixture
def packs_at_levels(monkeypatch):
    """Declare a synthetic ``BackchannelLevel`` per language code.

    Every shipped pack sits at MEDIUM since 2026-07-26 (one density knob, one
    setting — per-language tuning was guesswork, and the split let a
    cross-language comparison read a difference that came from the knob). The
    per-language reference machinery still has to work the day a pack declares
    something else, so the tests that exercise it stub the registry instead of
    asserting against pack data that is now uniform. Codes absent from the map
    resolve as having no pack at all.
    """

    def apply(levels: dict[str, BackchannelLevel]) -> None:
        def fake_get_language_pack(code: str):
            level = levels.get(code)
            return None if level is None else SimpleNamespace(backchannel_level=level)

        monkeypatch.setattr(
            "tau2.multilingual.registry.get_language_pack", fake_get_language_pack
        )

    return apply


# ---------------------------------------------------------------------------
# Latency & exchanges
# ---------------------------------------------------------------------------


def test_response_latency_and_turn_count():
    #        0         1         2         3
    #        0123456789012345678901234567890123456
    pattern = "uuuuu.....aaaaaaaaaa.....uuuuu..aaaa"
    f = features_for(pattern)
    # Exchange 1: last user tick 4 -> agent onset 10 = 6 ticks = 1.2 s.
    # Exchange 2: last user tick 29 -> agent onset 32 = 3 ticks = 0.6 s.
    assert f.turn_count == 2.0
    assert f.response_latency_p50 == pytest.approx(0.9)
    assert f.response_latency_p90 == pytest.approx(0.6 + (1.2 - 0.6) * 0.9)


def test_agent_pause_is_not_an_exchange():
    # Agent pauses mid-turn with no user speech in the gap: the second agent
    # onset is a continuation, not a response.
    f = features_for("uuu..aaaa..aaaa")
    assert f.turn_count == 1.0
    # Last user tick 2 -> agent onset 5 = 3 ticks = 0.6 s.
    assert f.response_latency_p50 == pytest.approx(0.6)


def test_backchannel_does_not_initiate_an_exchange():
    # User backchannels during an agent pause; the agent resuming is not a
    # response to it.
    f = features_for("uu..aaaa.uu.aaaa", bc_ticks={9})
    assert f.turn_count == 1.0
    assert f.backchannel_count == 1.0


# ---------------------------------------------------------------------------
# Silence & overlap
# ---------------------------------------------------------------------------


def test_long_silence_and_dead_air():
    pattern = "uuu" + "." * 18 + "aaaaa"  # 3.6 s mutual silence
    f = features_for(pattern)
    assert f.long_silence_count == 1.0
    assert f.dead_air_ratio == pytest.approx(18 / 26)
    assert f.time_to_resolution == pytest.approx(26 * DUR)


def test_short_silences_do_not_count_as_long():
    f = features_for("uuuuu.....aaaaaaaaaa")  # 1.0 s silence only
    assert f.long_silence_count == 0.0


def test_overlap_time_total():
    f = features_for("aaaabbbaaa")  # 3 ticks of simultaneous speech
    assert f.overlap_time_total == pytest.approx(3 * DUR)


# ---------------------------------------------------------------------------
# Interruptions & backchannels
# ---------------------------------------------------------------------------


def test_user_barge_in_counted():
    # User starts speaking while the agent holds the floor.
    f = features_for("aaaabbbbaa")
    assert f.user_barge_in_count == 1.0
    assert f.agent_interruption_count == 0.0


def test_agent_interruption_counted():
    # Agent starts speaking while the user holds the floor.
    f = features_for("uuuubbbbuu")
    assert f.agent_interruption_count == 1.0
    assert f.user_barge_in_count == 0.0


def test_backchannel_overlap_is_not_a_barge_in():
    # The user's overlapping speech is a marked backchannel utterance.
    f = features_for("aaaabbaaaa", bc_ticks={4})
    assert f.user_barge_in_count == 0.0
    assert f.backchannel_count == 1.0


def test_backchannel_density_deviation():
    # 20 ticks = 4 s; one backchannel -> 15 bc/min; en is a medium-level pack,
    # reference 1.0 -> 14.0.
    f = features_for("a" * 20, bc_ticks={10})
    assert f.backchannel_count == 1.0
    assert f.backchannel_density_deviation == pytest.approx(14.0)


def test_backchannel_reference_follows_the_pack_level(packs_at_levels):
    """The reference rate comes from the pack's declared level, not a global
    constant: two languages score the same audio differently when their packs
    differ. (Pins the mechanism, not today's uniformly-MEDIUM pack data.)"""
    packs_at_levels(
        {
            "ko": BackchannelLevel.HIGH,
            "de": BackchannelLevel.LOW,
            "es": BackchannelLevel.MEDIUM,
        }
    )
    cfg = FeatureExtractionConfig()
    assert resolve_backchannel_reference_per_min("ko", cfg) == pytest.approx(
        BACKCHANNEL_REFERENCE_PER_MIN[BackchannelLevel.HIGH]
    )
    assert resolve_backchannel_reference_per_min("de", cfg) == pytest.approx(
        BACKCHANNEL_REFERENCE_PER_MIN[BackchannelLevel.LOW]
    )
    assert resolve_backchannel_reference_per_min("es", cfg) == pytest.approx(
        BACKCHANNEL_REFERENCE_PER_MIN[BackchannelLevel.MEDIUM]
    )
    # Case-insensitive on the ISO code.
    assert resolve_backchannel_reference_per_min("KO", cfg) == pytest.approx(
        resolve_backchannel_reference_per_min("ko", cfg)
    )


def test_backchannel_reference_falls_back_without_a_pack():
    cfg = FeatureExtractionConfig()
    # No pack for this code -> the configured single reference.
    assert resolve_backchannel_reference_per_min("xx", cfg) == pytest.approx(
        cfg.human_backchannel_per_min
    )
    # ...and the fallback is configurable.
    custom = FeatureExtractionConfig(human_backchannel_per_min=2.5)
    assert resolve_backchannel_reference_per_min("xx", custom) == pytest.approx(2.5)


def test_high_density_language_deviates_less_at_the_same_rate(packs_at_levels):
    """A native-rate call in a high-density language is NOT scored as the
    defect a global reference made it: same ticks, the high-density side
    deviates less (and a silent listener deviates MORE there, which the old
    constant could not express). Levels are stubbed — every shipped pack is
    MEDIUM — so this pins the scoring, not the pack data."""
    packs_at_levels({"ko": BackchannelLevel.HIGH, "en": BackchannelLevel.MEDIUM})
    # 300 ticks = 60 s; 2 backchannels -> 2.0 bc/min.
    pattern = "a" * 300
    bc = {100, 200}
    en = features_for(pattern, bc_ticks=bc, language="en")
    ko = features_for(pattern, bc_ticks=bc, language="ko")
    assert en.backchannel_density_deviation == pytest.approx(1.0)  # |2.0 - 1.0|
    assert ko.backchannel_density_deviation == pytest.approx(0.3)  # |2.0 - 1.7|
    assert ko.backchannel_density_deviation < en.backchannel_density_deviation
    # Under-backchanneling: silence costs more in a high-density language.
    en_silent = features_for(pattern, language="en")
    ko_silent = features_for(pattern, language="ko")
    assert en_silent.backchannel_density_deviation == pytest.approx(1.0)
    assert ko_silent.backchannel_density_deviation == pytest.approx(1.7)
    assert (
        ko_silent.backchannel_density_deviation
        > en_silent.backchannel_density_deviation
    )


def test_backchannel_reference_uses_the_config_level_map(packs_at_levels):
    packs_at_levels({"ko": BackchannelLevel.HIGH})
    cfg = FeatureExtractionConfig(
        backchannel_per_min_by_level={
            BackchannelLevel.LOW: 0.1,
            BackchannelLevel.MEDIUM: 0.2,
            BackchannelLevel.HIGH: 9.0,
        }
    )
    assert resolve_backchannel_reference_per_min("ko", cfg) == pytest.approx(9.0)
    # Values are recorded in the table's provenance (the config is serialized).
    dumped = cfg.model_dump(mode="json")["backchannel_per_min_by_level"]
    assert dumped == {"low": 0.1, "medium": 0.2, "high": 9.0}


def test_partial_level_map_is_rejected():
    """A map missing a level would silently score those languages against the
    no-pack fallback."""
    with pytest.raises(ValueError, match="high"):
        FeatureExtractionConfig(
            backchannel_per_min_by_level={
                BackchannelLevel.LOW: 0.1,
                BackchannelLevel.MEDIUM: 0.2,
            }
        )


# ---------------------------------------------------------------------------
# Speech economy
# ---------------------------------------------------------------------------


def test_agent_talk_ratio_and_turn_merging():
    # Two agent segments split by a 0.4 s pause with no user speech merge
    # into ONE 4.0 s turn; the user speaks 1.0 s.
    pattern = "aaaaaaaaaa..aaaaaaaa.....uuuuu"
    f = features_for(pattern)
    assert f.agent_talk_ratio == pytest.approx(3.6 / 4.6)
    assert f.agent_turn_len_p90 == pytest.approx(20 * DUR)


def test_agent_turns_split_by_user_floor():
    # A floor-taking user utterance in the gap splits the agent turns.
    pattern = "aaaa.uu.aaaa"
    f = features_for(pattern)
    assert f.agent_turn_len_p90 == pytest.approx(4 * DUR)


# ---------------------------------------------------------------------------
# Tool flow / auth
# ---------------------------------------------------------------------------


def auth_ticks() -> list[Tick]:
    ticks = ticks_from_pattern("u" * 20)
    ticks[5] = make_tick(
        5,
        user_speech=True,
        agent_tool_calls=[
            ToolCall(
                id="c1", name="get_customer_by_name", arguments={"full_name": "Ana"}
            )
        ],
    )
    ticks[6] = make_tick(
        6,
        user_speech=True,
        agent_tool_results=[
            ToolMessage(id="c1", role="tool", content="[]", error=False)
        ],
    )
    ticks[10] = make_tick(
        10,
        user_speech=True,
        agent_tool_calls=[
            ToolCall(
                id="c2", name="get_customer_by_phone", arguments={"phone_number": "555"}
            )
        ],
    )
    ticks[12] = make_tick(
        12,
        user_speech=True,
        agent_tool_results=[
            ToolMessage(
                id="c2", role="tool", content='{"customer_id": "C1"}', error=False
            )
        ],
    )
    ticks[14] = make_tick(
        14,
        user_speech=True,
        agent_tool_calls=[
            ToolCall(id="c3", name="suspend_line", arguments={"line_id": "L1"})
        ],
    )
    ticks[15] = make_tick(
        15,
        user_speech=True,
        agent_tool_results=[
            ToolMessage(id="c3", role="tool", content="boom", error=True)
        ],
    )
    ticks[16] = make_tick(
        16,
        user_speech=True,
        agent_tool_calls=[
            ToolCall(
                id="c4", name="get_customer_by_phone", arguments={"phone_number": "555"}
            )
        ],
    )
    return ticks


def test_auth_and_tool_features():
    sim = make_sim(auth_ticks())
    f = extract_call_features(sim, domain="telecom", language="en")
    # Empty-list name lookup is NOT a successful identification; the phone
    # lookup at tick 12 is. Attempts include the successful call.
    assert f.time_to_auth == pytest.approx(12 * DUR)
    assert f.auth_attempt_count == 2.0
    assert f.auth_arg_mismatch == 1.0  # the empty-list name lookup
    assert f.tool_error_count == 1.0
    # c4 repeats c2 (same tool, identical arguments).
    assert f.repeated_tool_call_count == 1.0


def test_auth_none_when_never_successful():
    ticks = ticks_from_pattern("u" * 8)
    ticks[2] = make_tick(
        2,
        user_speech=True,
        agent_tool_calls=[
            ToolCall(id="c1", name="get_customer_by_name", arguments={"full_name": "X"})
        ],
    )
    ticks[3] = make_tick(
        3,
        user_speech=True,
        agent_tool_results=[
            ToolMessage(id="c1", role="tool", content="[]", error=False)
        ],
    )
    f = extract_call_features(make_sim(ticks), domain="telecom", language="en")
    assert f.time_to_auth is None
    assert f.auth_attempt_count == 1.0


def test_unknown_domain_has_no_auth_features():
    """A domain outside the seam gets null auth features — but the domain used
    here has to be one that really is outside it. ``retail`` used to stand in
    for "unregistered" and is now a pool domain with its own auth tools."""
    f = extract_call_features(make_sim(auth_ticks()), domain="mock", language="en")
    assert f.time_to_auth is None
    assert f.auth_attempt_count is None
    assert f.auth_arg_mismatch is None
    assert f.tool_error_count == 1.0  # tool friction still computed


# ---------------------------------------------------------------------------
# No-match classification: the two transports
# ---------------------------------------------------------------------------


def test_an_empty_non_error_result_is_a_no_match():
    for content in ("[]", "", "null", "None", "  []  "):
        assert classify_auth_lookup(content, False) is AuthLookupOutcome.NO_MATCH
    assert classify_auth_lookup(None, False) is AuthLookupOutcome.NO_MATCH


def test_a_no_records_prose_result_is_a_no_match():
    """The third transport: banking's finders miss with a NON-error prose
    result. Anchored at the start — a successful lookup opens with 'Found N
    record(s)' and must classify IDENTIFIED even though it mentions records."""
    miss = "No records found in 'users'."
    assert classify_auth_lookup(miss, False) is AuthLookupOutcome.NO_MATCH
    hit = "Found 1 record(s) in 'users':\n1. Record ID: u1\n   name: A"
    assert classify_auth_lookup(hit, False) is AuthLookupOutcome.IDENTIFIED


def test_a_not_found_error_is_a_no_match_not_a_tool_failure():
    """The transport 2.0.0 missed. Five of the six registered auth tools
    signal no-match by raising, so reading only the empty-result transport
    scored zero mismatches for all of airline, all of retail, and two thirds
    of telecom."""
    for content in (
        "Error: User not found",
        "Error: Customer with ID C9 not found",
        "Error: Customer with phone number +1-555 not found",
    ):
        assert classify_auth_lookup(content, True) is AuthLookupOutcome.NO_MATCH


def test_a_real_tool_failure_is_not_counted_as_a_mismatch():
    """A harness fault carries no evidence about what the agent heard, so it
    must not enter the mis-hearing feature."""
    for content in (
        "Error: missing 1 required positional argument: 'dob'",
        "Error: connection reset",
    ):
        assert classify_auth_lookup(content, True) is AuthLookupOutcome.TOOL_FAILURE


def test_a_populated_result_identifies_someone():
    assert (
        classify_auth_lookup('{"customer_id": "C1"}', False)
        is AuthLookupOutcome.IDENTIFIED
    )
    assert classify_auth_lookup("sara_doe_496", False) is AuthLookupOutcome.IDENTIFIED


def test_every_registered_auth_tool_signals_no_match_recognizably():
    """The guard that keeps ``_NOT_FOUND_ERROR`` honest.

    Calls every tool in ``AUTH_TOOLS_BY_DOMAIN`` against its REAL domain
    environment with arguments matching no record, and asserts the result
    classifies as NO_MATCH. A domain that reworded its not-found, or a tool
    whose signature moved, fails here — instead of silently zeroing
    ``auth_arg_mismatch``, which is a fit feature and would show up as an
    agent that never mishears anything.
    """
    from tau2.registry import registry

    # What "matches nothing" is for each registered auth tool. Declared, not
    # derived: a fuzzed argument that accidentally matched a record would make
    # this test pass for the wrong reason.
    no_match_args: dict[str, dict[str, str]] = {
        "get_customer_by_phone": {"phone_number": "+1-000-000-0000"},
        "get_customer_by_name": {"full_name": "Nobody Atall", "dob": "1900-01-01"},
        "get_customer_by_id": {"customer_id": "NO_SUCH_CUSTOMER"},
        "get_user_details": {"user_id": "no_such_user_000"},
        "find_user_id_by_name_zip": {
            "first_name": "Nobody",
            "last_name": "Atall",
            "zip": "00000",
        },
        "find_user_id_by_email": {"email": "nobody.atall@example.invalid"},
        "get_user_information_by_id": {"user_id": "no_such_user_000"},
        "get_user_information_by_name": {"customer_name": "Nobody Atall"},
        "get_user_information_by_email": {"email": "nobody.atall@example.invalid"},
    }
    registered = {t for tools in AUTH_TOOLS_BY_DOMAIN.values() for t in tools}
    assert registered == set(no_match_args), (
        "every registered auth tool needs a no-match argument set here"
    )

    for domain, tools in sorted(AUTH_TOOLS_BY_DOMAIN.items()):
        if (
            domain == "banking_knowledge"
            and importlib.util.find_spec("rank_bm25") is None
        ):
            continue
        env = registry.get_env_constructor(domain)()
        for name in sorted(tools):
            result = env.get_response(
                ToolCall(
                    id="probe",
                    name=name,
                    requestor="assistant",
                    arguments=no_match_args[name],
                )
            )
            assert (
                classify_auth_lookup(result.content, result.error)
                is AuthLookupOutcome.NO_MATCH
            ), f"{domain}.{name} -> {result.error=} {result.content!r}"


# ---------------------------------------------------------------------------
# Retail: prose name+zip and spelled-aloud email
# ---------------------------------------------------------------------------


def retail_lookup_ticks(
    tool: str, arguments: dict, content: str, *, error: bool
) -> list[Tick]:
    """A retail call whose single auth lookup returns ``content``."""
    ticks = ticks_from_pattern("u" * 8)
    ticks[2] = make_tick(
        2,
        user_speech=True,
        agent_tool_calls=[ToolCall(id="c1", name=tool, arguments=arguments)],
    )
    ticks[4] = make_tick(
        4,
        user_speech=True,
        agent_tool_results=[
            ToolMessage(id="c1", role="tool", content=content, error=error)
        ],
    )
    return ticks


def test_retail_yields_a_non_null_auth_arg_mismatch():
    """Retail is a pool domain, so its rows must carry the fit's twelfth
    feature rather than dropping out of the fit entirely."""
    ticks = retail_lookup_ticks(
        "find_user_id_by_name_zip",
        {"first_name": "Sara", "last_name": "Doe", "zip": "12345"},
        "sara_doe_496",
        error=False,
    )
    f = extract_call_features(make_sim(ticks), domain="retail", language="en")
    assert f.auth_arg_mismatch == 0.0
    assert f.auth_attempt_count == 1.0
    assert f.time_to_auth == pytest.approx(4 * DUR)


def test_a_mis_heard_prose_name_zip_counts_as_a_mismatch():
    """The highest-value case in the feature. The caller recites name and zip
    aloud; a zip digit heard wrong comes back as ``Error: User not found``,
    which is a mis-hearing, not a tool failure — and must not be scored as a
    successful identification."""
    ticks = retail_lookup_ticks(
        "find_user_id_by_name_zip",
        {"first_name": "Sara", "last_name": "Doe", "zip": "12346"},
        "Error: User not found",
        error=True,
    )
    f = extract_call_features(make_sim(ticks), domain="retail", language="en")
    assert f.auth_arg_mismatch == 1.0
    assert f.time_to_auth is None  # never identified
    assert f.auth_attempt_count == 1.0


def test_the_diacritic_fold_does_not_defeat_the_mismatch_counter():
    """PR #533 folded case and diacritics in ``find_user_id_by_name_zip`` so a
    spelled-aloud 'Vazquez' still finds 'Vázquez'. The fold makes the NAME
    forgiving; the zip is still the verification factor, so a wrong zip is
    still a not-found and still counts. Asserted against the real retail
    environment, not a hand-written result string."""
    from tau2.registry import registry

    env = registry.get_env_constructor("retail")()
    folded_name = env.get_response(
        ToolCall(
            id="c1",
            name="find_user_id_by_name_zip",
            requestor="assistant",
            arguments={"first_name": "Nobody", "last_name": "Atall", "zip": "00000"},
        )
    )
    # The fold cannot rescue a caller who matches no record at all.
    assert (
        classify_auth_lookup(folded_name.content, folded_name.error)
        is AuthLookupOutcome.NO_MATCH
    )


def test_a_mis_spelled_email_counts_as_a_mismatch():
    """Spelling an address aloud is the hardest spoken-auth case in the
    benchmark, and its failure is a clean not-found."""
    ticks = retail_lookup_ticks(
        "find_user_id_by_email",
        {"email": "sara.do@example.com"},
        "Error: User not found",
        error=True,
    )
    f = extract_call_features(make_sim(ticks), domain="retail", language="en")
    assert f.auth_arg_mismatch == 1.0
    assert f.time_to_auth is None


def test_retail_get_user_details_is_not_an_auth_tool():
    """Retail's policy requires locating the user id via email or name+zip
    "even when the user already provides the user id", so the lookup that
    CONSUMES an id authenticates nobody. Counting it would book auth as
    succeeding before it had."""
    assert "get_user_details" not in AUTH_TOOLS_BY_DOMAIN["retail"]
    ticks = retail_lookup_ticks(
        "get_user_details",
        {"user_id": "sara_doe_496"},
        '{"user_id": "sara_doe_496"}',
        error=False,
    )
    f = extract_call_features(make_sim(ticks), domain="retail", language="en")
    assert f.auth_attempt_count == 0.0
    assert f.auth_arg_mismatch == 0.0
    assert f.time_to_auth is None


def test_tickless_retail_sim_still_counts_the_mismatch():
    """The half-duplex path has its own copy of the auth accounting; a
    transport fixed in one and not the other is the bug this guards."""
    sim = make_sim(None)
    sim.messages = [
        AssistantMessage(
            role="assistant",
            content="Let me look you up.",
            tool_calls=[
                ToolCall(
                    id="c1",
                    name="find_user_id_by_email",
                    arguments={"email": "wrong@example.com"},
                )
            ],
        ),
        ToolMessage(id="c1", role="tool", content="Error: User not found", error=True),
    ]
    f = extract_call_features(sim, domain="retail", language="en")
    assert f.auth_arg_mismatch == 1.0
    assert f.auth_attempt_count == 1.0


# ---------------------------------------------------------------------------
# Tick-less sims (half-duplex fallback)
# ---------------------------------------------------------------------------


def test_tickless_sim_gets_counts_only():
    sim = make_sim(None)
    sim.messages = [
        AssistantMessage(
            role="assistant",
            content="Let me look you up.",
            tool_calls=[
                ToolCall(
                    id="c1", name="get_customer_by_id", arguments={"customer_id": "C1"}
                )
            ],
        ),
        ToolMessage(id="c1", role="tool", content='{"customer_id": "C1"}', error=False),
    ]
    f = extract_call_features(sim, domain="telecom", language="en")
    assert f.auth_attempt_count == 1.0
    assert f.tool_error_count == 0.0
    assert f.response_latency_p50 is None
    assert f.time_to_auth is None
    assert f.time_to_resolution is None


# ---------------------------------------------------------------------------
# Schema coverage & enumerations
# ---------------------------------------------------------------------------


def test_feature_enumerations_cover_the_model_exactly():
    declared = set(EXTRACTED_FEATURES) | set(JUDGED_FEATURE_SLOTS)
    assert declared == set(CallFeatures.model_fields)
    assert set(EXTRACTED_FEATURES).isdisjoint(JUDGED_FEATURE_SLOTS)


def test_unjudged_slots_are_named_not_silently_empty():
    """``judged_phrasing`` and ``judged_repetition`` have no judge yet; the
    gap is declared rather than discovered downstream."""
    from tau2.annotation.features import UNJUDGED_SLOTS

    assert set(UNJUDGED_SLOTS) <= set(JUDGED_FEATURE_SLOTS)


def test_there_is_no_hand_picked_steer_set_left():
    """v7 draws uniformly at random inside a stratum, so no feature has any
    say in selection. A re-introduced steer set would be a silent return to
    feature-dependent inclusion probabilities — the thing the rewrite removed.
    """
    import tau2.annotation.features as features_module

    assert not hasattr(features_module, "SAMPLING_FEATURES")
    assert not hasattr(features_module, "SAMPLING_EXCLUSIONS")


def test_backchannel_reference_covers_every_level():
    """Closed catalog: a new density level cannot ship without a reference
    rate, and the default config map mirrors the catalog exactly."""
    assert set(BACKCHANNEL_REFERENCE_PER_MIN) == set(BackchannelLevel)
    assert FeatureExtractionConfig().backchannel_per_min_by_level == (
        BACKCHANNEL_REFERENCE_PER_MIN
    )
    # Monotone in density: more expected continuers as the level rises.
    assert (
        BACKCHANNEL_REFERENCE_PER_MIN[BackchannelLevel.LOW]
        < BACKCHANNEL_REFERENCE_PER_MIN[BackchannelLevel.MEDIUM]
        < BACKCHANNEL_REFERENCE_PER_MIN[BackchannelLevel.HIGH]
    )


def test_every_pack_level_resolves_to_a_reference():
    """Every registered pack's declared level has a rate — no live language
    silently falls back to the global constant."""
    from tau2.multilingual.registry import get_language_pack, list_language_packs

    cfg = FeatureExtractionConfig()
    for language in list_language_packs():
        pack = get_language_pack(language)
        if pack.backchannel_level is None:
            continue
        assert resolve_backchannel_reference_per_min(language, cfg) == pytest.approx(
            BACKCHANNEL_REFERENCE_PER_MIN[pack.backchannel_level]
        ), language


def test_auth_tool_seam_covers_study_domains():
    """Every POOL domain — the three voice domains plus banking, the text
    pool's domain: a domain missing here carries a null ``auth_arg_mismatch``
    and drops out of its target's fit entirely."""
    for domain in ("telecom", "airline", "retail", "banking_knowledge"):
        assert domain in AUTH_TOOLS_BY_DOMAIN, domain
    assert "get_customer_by_name" in AUTH_TOOLS_BY_DOMAIN["telecom"]
    assert "get_customer_by_phone" in AUTH_TOOLS_BY_DOMAIN["telecom"]
    assert AUTH_TOOLS_BY_DOMAIN["retail"] == frozenset(
        {"find_user_id_by_name_zip", "find_user_id_by_email"}
    )


def test_every_registered_auth_tool_exists_in_its_domain():
    """A typo in the seam is a permanently-zero fit feature with no error
    anywhere — the map names tools, and nothing but this checks they are real."""
    from tau2.registry import registry

    for domain, tools in sorted(AUTH_TOOLS_BY_DOMAIN.items()):
        if (
            domain == "banking_knowledge"
            and importlib.util.find_spec("rank_bm25") is None
        ):
            continue
        available = {
            tool.name for tool in registry.get_env_constructor(domain)().get_tools()
        }
        for name in sorted(tools):
            assert name in available, f"{domain} has no tool '{name}'"


# ---------------------------------------------------------------------------
# Table build: gate, identity, provenance
# ---------------------------------------------------------------------------


def voice_sim(sim_id: str, *, trial: int, provider: str, pattern: str) -> SimulationRun:
    sim = make_sim(ticks_from_pattern(pattern), sim_id=sim_id)
    sim.trial = trial
    sim.agent_provider = provider
    sim.nativeness_info = NativenessInfo(
        score=0.5, factor_checks=[], language="hi", script="deva"
    )
    return sim


def voice_run(tmp_path: Path, sims: list[SimulationRun], **kwargs) -> Path:
    """A run on disk with a pinned audio-native block, i.e. an IDENTIFIABLE arm."""
    kwargs.setdefault("audio_native_config", OPENAI_XHIGH)
    kwargs.setdefault("agent_llm", OPENAI_LLM)
    return make_hi_results(tmp_path, sims, **kwargs)


def test_build_feature_table_rows_and_provenance(tmp_path: Path):
    sims = [
        voice_sim("s1", trial=0, provider="openai", pattern="uuu...aaaa"),
        voice_sim("s2", trial=1, provider="openai", pattern="uuu.......aaaa"),
        make_sim(None, sim_id="s3", termination=TerminationReason.AGENT_ERROR),
    ]
    run_dir = voice_run(tmp_path, sims, num_trials=2)
    table = build_feature_table([run_dir])
    assert table.extractor_version == FEATURE_EXTRACTOR_VERSION
    assert len(table.rows) == 3
    by_id = {r.sim_id: r for r in table.rows}
    assert by_id["s1"].language == "hi"
    assert by_id["s1"].provider == "openai"
    assert by_id["s1"].gate_passed
    assert by_id["s1"].features.response_latency_p50 == pytest.approx(0.8)
    assert by_id["s2"].features.response_latency_p50 == pytest.approx(1.6)
    # Errored + tick-less sim fails the gate with explicit reasons.
    assert not by_id["s3"].gate_passed
    assert "termination:agent_error" in by_id["s3"].gate_reasons
    assert "no_ticks" in by_id["s3"].gate_reasons
    assert table.source_runs[0].n_calls == 3


# ---------------------------------------------------------------------------
# Reasoning effort: part of the arm, never defaulted
# ---------------------------------------------------------------------------


def test_rows_and_source_runs_carry_the_runs_reasoning_effort(tmp_path: Path):
    run_dir = voice_run(
        tmp_path, [voice_sim("s1", trial=0, provider="openai", pattern="uuu...aaaa")]
    )
    table = build_feature_table([run_dir])
    assert table.rows[0].reasoning_effort is ReasoningEffort.XHIGH
    assert table.source_runs[0].reasoning_effort is ReasoningEffort.XHIGH
    assert Arm.of(table.rows[0]).label == "openai:gpt-realtime-2@xhigh"


def test_the_two_effort_cells_of_one_model_are_two_arms(tmp_path: Path):
    """The English pool is a 2x2 of provider x effort, and both openai cells
    report the same ``agent_model``. If effort were not part of the identity,
    an xhigh-vs-minimal pair would be typed a stochastic rerun of itself."""
    arms = []
    for effort in (ReasoningEffort.XHIGH, ReasoningEffort.MINIMAL):
        run_dir = voice_run(
            tmp_path,
            [voice_sim("s1", trial=0, provider="openai", pattern="uuu...aaaa")],
            name=f"run_{effort.value}",
            audio_native_config=AudioNativeConfig(
                provider="openai", model="gpt-realtime-2", reasoning_effort=effort
            ),
        )
        table = build_feature_table([run_dir])
        arms.append(Arm.of(table.rows[0]))
    assert arms[0] != arms[1]
    assert arms[0].agent_model == arms[1].agent_model  # the collision itself
    assert {a.label for a in arms} == {
        "openai:gpt-realtime-2@xhigh",
        "openai:gpt-realtime-2@minimal",
    }


def test_extraction_refuses_a_run_whose_effort_was_never_recorded(tmp_path: Path):
    """Runs written before the field became eager stored ``null``; the loader
    resolves it to the provider default and marks the source ``inferred``. That
    resolved value must NOT reach a row — read as real it would merge an
    unpinned arm into a pinned one."""
    unrecorded = AudioNativeConfig(
        provider="openai", model="gpt-realtime-2", reasoning_effort=ReasoningEffort.HIGH
    )
    unrecorded = unrecorded.model_copy(
        update={"reasoning_effort_source": ReasoningEffortSource.INFERRED}
    )
    assert unrecorded.reasoning_effort_backfill is None
    run_dir = voice_run(
        tmp_path,
        [voice_sim("s1", trial=0, provider="openai", pattern="uuu...aaaa")],
        audio_native_config=unrecorded,
    )
    with pytest.raises(ValueError, match="never backfilled"):
        build_feature_table([run_dir])


def test_extraction_accepts_a_backfilled_effort(tmp_path: Path):
    """Backfilled is ``inferred`` WITH a provenance stamp — written by the
    since-retired backfill verb, and still trustworthy."""
    backfilled = AudioNativeConfig(
        provider="openai", model="gpt-realtime-2", reasoning_effort=ReasoningEffort.HIGH
    ).model_copy(
        update={
            "reasoning_effort_source": ReasoningEffortSource.INFERRED,
            "reasoning_effort_backfill": ReasoningEffortBackfill(
                backfilled_at="2026-07-26T00:00:00",
                git_commit="0" * 40,
                tool="tau2 backfill-reasoning-effort",
                basis="provider default table",
            ),
        }
    )
    run_dir = voice_run(
        tmp_path,
        [voice_sim("s1", trial=0, provider="openai", pattern="uuu...aaaa")],
        audio_native_config=backfilled,
    )
    table = build_feature_table([run_dir])
    assert table.rows[0].reasoning_effort is ReasoningEffort.HIGH


def test_extraction_refuses_a_provider_with_no_audio_native_block(tmp_path: Path):
    """Same failure, different shape: the sim says it was audio-native but the
    run recorded no config, so the arm is unidentifiable."""
    run_dir = make_hi_results(
        tmp_path, [voice_sim("s1", trial=0, provider="openai", pattern="uuu...aaaa")]
    )
    with pytest.raises(ValueError, match="never backfilled"):
        build_feature_table([run_dir])


def test_a_text_run_has_no_effort_and_that_is_not_a_gap(tmp_path: Path):
    """No provider and no audio-native block: there was no knob to record."""
    run_dir = make_hi_results(tmp_path, [make_sim(ticks_from_pattern("uuu...aaaa"))])
    table = build_feature_table([run_dir])
    assert table.rows[0].reasoning_effort is None
    assert table.rows[0].provider == ""
    assert Arm.of(table.rows[0]).label == "fake-llm"


def test_language_falls_back_to_task_suffix_on_voice_runs(tmp_path: Path):
    """A voice run with no stored judge verdicts still resolves its language.

    English runs are never judged, so they carry no ``nativeness_info``, and
    the run-level ``VoiceSettings`` holds no language at all — the localized
    task-set suffix on the task id is the only deterministic source. Populating
    ``voice_settings`` is the point of the fixture here: it is what real runs
    look like, and what a None-valued fixture hid.
    """
    sim = voice_sim("s1", trial=0, provider="openai", pattern="uuu...aaaa")
    sim.nativeness_info = None  # English runs are never judged
    run_dir = voice_run(
        tmp_path, [sim], voice_settings=VoiceSettings(), domain="telecom"
    )
    table = build_feature_table([run_dir])
    # Base (un-localized) task id => English.
    assert table.rows[0].language == "en"


def test_language_prefers_stored_judge_language(tmp_path: Path):
    """When the judge ran, its recorded language wins over any id inference."""
    sim = voice_sim("s1", trial=0, provider="openai", pattern="uuu...aaaa")
    run_dir = voice_run(tmp_path, [sim], voice_settings=VoiceSettings())
    table = build_feature_table([run_dir])
    assert table.rows[0].language == "hi"
    assert table.source_runs[0].domain == "airline"


def test_table_id_is_content_derived(tmp_path: Path):
    sims = [voice_sim("s1", trial=0, provider="openai", pattern="uuu...aaaa")]
    run_dir = voice_run(tmp_path, sims)
    first = build_feature_table([run_dir])
    second = build_feature_table([run_dir])
    assert first.table_id == second.table_id
    different = build_feature_table(
        [run_dir], FeatureExtractionConfig(long_silence_threshold_s=9.9)
    )
    assert different.table_id != first.table_id


def test_export_and_load_round_trip(tmp_path: Path):
    sims = [voice_sim("s1", trial=0, provider="openai", pattern="uuu...aaaa")]
    run_dir = voice_run(tmp_path, sims)
    out = tmp_path / "table.json"
    path = export_feature_table([run_dir], out)
    loaded = load_feature_table(path)
    assert loaded.rows[0].sim_id == "s1"
    assert loaded.table_id == build_feature_table([run_dir]).table_id


def test_export_refuses_empty_table(tmp_path: Path):
    sims = [voice_sim("s1", trial=0, provider="openai", pattern="uuu...aaaa")]
    run_dir = voice_run(tmp_path, sims)
    with pytest.raises(ValueError, match="no calls extracted"):
        export_feature_table(
            [run_dir],
            tmp_path / "t.json",
            FeatureExtractionConfig(langs=["zz"]),
        )


# ---------------------------------------------------------------------------
# Integration smoke on real saved simulations (skips when absent)
# ---------------------------------------------------------------------------


def test_extractor_smoke_on_real_run():
    sims_root = Path(__file__).resolve().parents[2] / "data" / "simulations"
    results_files = (
        sorted(sims_root.rglob("results.json")) if sims_root.exists() else []
    )
    if not results_files:
        pytest.skip("no saved simulations under data/simulations")
    table = build_feature_table([results_files[0]], FeatureExtractionConfig(max_sims=2))
    assert table.rows
    for row in table.rows:
        assert row.sim_id
        assert row.features is not None


def test_a_consolidation_sidecar_is_pinned_by_content(tmp_path: Path):
    """A consolidated arm's sidecar is the only record of which source runs it
    was assembled from, so the table pins its content, not just its path."""
    run_dir = voice_run(
        tmp_path, [voice_sim("s1", trial=0, provider="openai", pattern="uuu...aaaa")]
    )
    assert build_feature_table([run_dir]).source_runs[0].consolidation_id == ""

    sidecar = run_dir / "consolidation.json"
    sidecar.write_text('{"sources": ["run_a", "run_b"]}')
    pinned = build_feature_table([run_dir]).source_runs[0].consolidation_id
    assert re.fullmatch(r"[0-9a-f]{12}", pinned)

    sidecar.write_text('{"sources": ["run_a", "run_c"]}')
    assert build_feature_table([run_dir]).source_runs[0].consolidation_id != pinned


# ---------------------------------------------------------------------------
# Text mode: modality, message-based gate, text-economy features
# ---------------------------------------------------------------------------


def text_sim(
    sim_id: str = "t1",
    *,
    termination: TerminationReason = TerminationReason.USER_STOP,
    messages: Optional[list] = None,
) -> SimulationRun:
    sim = make_sim(None, sim_id=sim_id, termination=termination)
    sim.mode = "half_duplex"
    sim.messages = (
        messages
        if messages is not None
        else [
            UserMessage(role="user", content="Hi, I want to close my savings account."),
            AssistantMessage(
                role="assistant", content="Sure - can you confirm your date of birth?"
            ),
        ]
    )
    return sim


def text_run(
    tmp_path: Path,
    sims: list[SimulationRun],
    *,
    name: str = "text_run",
    effort: Optional[str] = None,
    agent_llm: str = "gpt-5.5",
) -> Path:
    """A run with NO audio-native block: a half-duplex text run. The effort,
    when pinned, lives where the runner records it — agent llm_args."""
    return make_hi_results(
        tmp_path,
        sims,
        name=name,
        agent_llm=agent_llm,
        agent_llm_args=({"reasoning_effort": effort} if effort else None),
    )


def test_text_rows_pass_the_gate_without_ticks(tmp_path: Path):
    """The headline fix: a tick-less sim is a broken RECORDING only when the
    run promised audio. A text run's transcript is its messages."""
    run_dir = text_run(tmp_path, [text_sim()])
    table = build_feature_table([run_dir])
    row = table.rows[0]
    assert row.modality == "text"
    assert row.gate_passed
    assert row.gate_reasons == []
    assert table.source_runs[0].modality == "text"


def test_a_text_sim_without_messages_fails_the_gate(tmp_path: Path):
    run_dir = text_run(tmp_path, [text_sim(messages=[])])
    row = build_feature_table([run_dir]).rows[0]
    assert not row.gate_passed
    assert "no_messages" in row.gate_reasons
    assert "no_ticks" not in row.gate_reasons


def test_a_broken_text_termination_still_fails_the_gate(tmp_path: Path):
    run_dir = text_run(tmp_path, [text_sim(termination=TerminationReason.AGENT_ERROR)])
    row = build_feature_table([run_dir]).rows[0]
    assert not row.gate_passed
    assert row.gate_reasons == ["termination:agent_error"]


def test_voice_rows_carry_the_voice_modality(tmp_path: Path):
    run_dir = voice_run(
        tmp_path, [voice_sim("s1", trial=0, provider="openai", pattern="uuu...aaaa")]
    )
    table = build_feature_table([run_dir])
    assert table.rows[0].modality == "voice"
    assert table.source_runs[0].modality == "voice"


def test_the_two_effort_cells_of_one_text_model_are_two_arms(tmp_path: Path):
    """The banking text pool is a 2x2 of model x effort; both gpt-5.5 cells
    report the same ``agent_model``, so — exactly as for voice — the pinned
    effort in llm_args is what keeps xhigh-vs-none a systematic contrast."""
    arms = []
    for effort in ("xhigh", "none"):
        run_dir = text_run(tmp_path, [text_sim()], name=f"run_{effort}", effort=effort)
        table = build_feature_table([run_dir])
        arms.append(Arm.of(table.rows[0]))
        assert table.source_runs[0].reasoning_effort is ReasoningEffort(effort)
    assert arms[0] != arms[1]
    assert arms[0].agent_model == arms[1].agent_model
    assert {a.label for a in arms} == {"gpt-5.5@xhigh", "gpt-5.5@none"}


def test_a_text_run_pinning_an_unknown_effort_is_refused(tmp_path: Path):
    """A pinned level the enum does not know is arm identity that cannot be
    represented — dropping it would merge arms, so extraction refuses."""
    run_dir = text_run(tmp_path, [text_sim()], effort="hyperspeed")
    with pytest.raises(ValueError, match="not a known level"):
        build_feature_table([run_dir])


def test_text_economy_features_measure_the_transcript():
    user = "Hi, I want to close my savings account."
    short = "Sure - can you confirm your date of birth?"
    formatted = "Here are your accounts:\n- Savings (1234)\n- Checking (5678)"
    sim = text_sim(
        messages=[
            UserMessage(role="user", content=user),
            AssistantMessage(role="assistant", content=short),
            # Tool-call-only turn: plumbing the user never sees, not a turn.
            AssistantMessage(
                role="assistant",
                content=None,
                tool_calls=[ToolCall(id="c1", name="get_accounts", arguments={})],
            ),
            ToolMessage(id="c1", role="tool", content="[...]", error=False),
            AssistantMessage(role="assistant", content=formatted),
        ]
    )
    f = extract_call_features(sim, domain="banking_knowledge", language="en")
    assert f.turn_count == 2.0
    lengths = sorted([len(short), len(formatted)])
    expected_p90 = lengths[0] + (lengths[1] - lengths[0]) * 0.9
    assert f.agent_msg_len_p90 == pytest.approx(expected_p90)
    agent_chars = len(short) + len(formatted)
    assert f.agent_char_ratio == pytest.approx(agent_chars / (agent_chars + len(user)))
    # 4 non-empty agent lines, 2 of them bullets.
    assert f.formatting_density == pytest.approx(0.5)
    # The perceptual-clock columns stay None: there is no clock to measure on.
    assert f.response_latency_p50 is None
    assert f.agent_talk_ratio is None


def test_plain_prose_has_zero_formatting_density():
    sim = text_sim()
    f = extract_call_features(sim, domain="banking_knowledge", language="en")
    assert f.formatting_density == 0.0
    assert f.turn_count == 1.0


def test_formatting_detector_recognizes_each_block_construct():
    constructs = [
        "- a bullet",
        "* another bullet",
        "1. a numbered item",
        "2) another numbered item",
        "# a heading",
        "| a | table |",
        "```",
        "> a quote",
    ]
    sim = text_sim(
        messages=[AssistantMessage(role="assistant", content="\n".join(constructs))]
    )
    f = extract_call_features(sim, domain="banking_knowledge", language="en")
    assert f.formatting_density == 1.0


def test_an_empty_transcript_yields_no_economy_features():
    sim = text_sim(messages=[])
    f = extract_call_features(sim, domain="banking_knowledge", language="en")
    assert f.turn_count is None
    assert f.agent_msg_len_p90 is None
    assert f.agent_char_ratio is None
    assert f.formatting_density is None
