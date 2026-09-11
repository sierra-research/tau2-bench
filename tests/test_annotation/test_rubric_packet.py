# Copyright Sierra
"""Combined rubric packet: catalogs, CSV contract, and offline build."""

import wave
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tau2.annotation.artifacts import ArtifactManifest, AudioQualityEntry
from tau2.annotation.models import (
    OVERALL_INTERACTION_QUALITY_FACTOR_ID,
    OVERALL_NATIVENESS_FACTOR_ID,
    RubricAnnotationRow,
)
from tau2.annotation.packets.audio_quality import AUDIO_QUALITY_DIMENSIONS
from tau2.annotation.packets.forms import (
    AUDIO_QUALITY_KIND,
    RUBRIC_PACKET_KIND,
    match_browser_kind,
)
from tau2.annotation.packets.rubric import (
    RubricPacketOptions,
    audio_sections,
    build_rubric_packet,
    interaction_sections,
    nativeness_sections_for,
    rubric_sections_for,
)
from test_annotation.conftest import make_packet_results_dir, packet_config


def _valid_stereo_wav(path: Path) -> bytes:
    frames = (b"\x10\x00\x20\x00") * 800
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(2)
        audio.setsampwidth(2)
        audio.setframerate(8000)
        audio.writeframes(frames)
    return path.read_bytes()


def _source_manifest(tmp_path: Path, results: Path) -> Path:
    path = tmp_path / "source" / "manifest.json"
    path.parent.mkdir()
    agent_clip = path.parent / "clips" / "clip_001.wav"
    agent_clip.parent.mkdir()
    frames = b"\x30\x00" * 400
    with wave.open(str(agent_clip), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(8000)
        audio.writeframes(frames)
    manifest = ArtifactManifest(
        kind=AUDIO_QUALITY_KIND,
        batch_id="source-batch",
        batch_name="source",
        created_at=datetime.now(timezone.utc).isoformat(),
        git_sha="deadbeef",
        language="es",
        form="audio_quality",
        provenance={},
        files=[],
        audio_quality_entries=[
            AudioQualityEntry(
                clip_id="clip_001",
                sim_id="s1",
                task_id="t1",
                trial=0,
                language="es",
                domain="airline",
                provider="openai",
                agent_model="openai:test",
                reasoning_effort="minimal",
                results_path=str(results / "results.json"),
                audio_file="clips/clip_001.wav",
            )
        ],
    )
    path.write_text(manifest.model_dump_json(indent=2))
    return path


def test_combined_rubric_has_three_sections_with_one_canonical_audio_catalog():
    interaction = interaction_sections()
    nativeness = nativeness_sections_for("es")
    audio = audio_sections()
    combined = rubric_sections_for("es")

    assert [section.id for section in combined] == [
        "interaction",
        "nativeness",
        "audio",
    ]
    assert [len(section.questions) for section in combined] == [6, 3, 11]
    assert [question.id for question in audio[0].questions] == [
        "intonation",
        *[
            dimension.id
            for dimension in AUDIO_QUALITY_DIMENSIONS
            if dimension.id != "intonation"
        ],
    ]
    assert all(
        question.answer_kind == "audio_severity" for question in audio[0].questions
    )

    interaction_ids = [question.id for question in interaction[0].questions]
    nativeness_ids = [question.id for question in nativeness[0].questions]
    assert len(interaction_ids) == len(set(interaction_ids)) == 6
    assert len(nativeness_ids) == len(set(nativeness_ids)) == 3
    assert interaction_ids[0] == OVERALL_INTERACTION_QUALITY_FACTOR_ID
    assert nativeness_ids[0] == OVERALL_NATIVENESS_FACTOR_ID
    assert "register_formality" not in nativeness_ids
    assert "responsiveness" not in interaction_ids
    assert "yielding" not in interaction_ids
    assert "backchannel_selectivity" not in interaction_ids
    assert "redundant_successful_tool_call" not in interaction_ids
    assert "unnecessary_tool_call" in interaction_ids
    assert "regional_consistency" not in nativeness_ids
    assert "translationese" not in nativeness_ids
    assert "verb_morphology" not in nativeness_ids

    interaction_score = interaction[0].questions[0]
    assert interaction_score.answer_kind == "likert4"
    assert [anchor.value for anchor in interaction_score.score_anchors] == [
        "1",
        "2",
        "3",
        "4",
    ]
    assert interaction_score.score_anchors[0].label == "Incredibly frustrating"
    assert interaction_score.score_anchors[-1].label == "Pleasant"
    nativeness_score = nativeness[0].questions[0]
    assert nativeness_score.answer_kind == "likert4"
    assert nativeness_score.score_anchors[0].label == "Clearly non-native"
    assert nativeness_score.score_anchors[-1].label == "Native-speaker-like"

    auth = next(
        question
        for question in interaction[0].questions
        if question.id == "auth_arg_mismatch"
    )
    assert "If the caller gave the wrong name, do not count it" in auth.guidance
    assert "transcription or carryover errors only" in auth.guidance
    tool_error = next(
        question
        for question in interaction[0].questions
        if question.id == "agent_caused_tool_error"
    )
    wrong_parameters = next(
        question
        for question in interaction[0].questions
        if question.id == "incorrect_tool_parameters"
    )
    assert "only when the tool returned an error" in tool_error.guidance
    assert "tool call that can succeed but contains wrong details" in (
        wrong_parameters.guidance
    )
    unnecessary = next(
        question
        for question in interaction[0].questions
        if question.id == "unnecessary_tool_call"
    )
    assert "same correct outcome without that tool call" in unnecessary.guidance
    assert (
        "legitimate attempts to find the caller’s order is not" in unnecessary.guidance
    )


def test_packet_uses_the_combined_spanish_naturalness_rubric():
    nativeness = nativeness_sections_for("es")[0]
    naturalness = next(
        question
        for question in nativeness.questions
        if question.id == "natural_word_choice"
    )
    assert naturalness.question == (
        "This is a customer-service call in contemporary Spanish as spoken in "
        "Madrid, Spain. Does this utterance use natural Madrid Spanish without "
        "translationese or unnatural word choice, and with correct verb forms "
        "and basic function words?"
    )
    assert naturalness.guidance.startswith(
        "Combined utterance naturalness for Madrid Spanish. Good: Natural "
        "conversational Madrid Spanish"
    )
    assert "written Spanish diacritics" in naturalness.guidance
    assert "non-Madrid regional varieties" in naturalness.guidance
    assert naturalness.answer_kind == "nativeness"
    assert naturalness.evaluation_level == "utterance"
    assert "Standard:" not in naturalness.guidance


def test_korean_name_address_packet_guidance_keeps_hybrid_boundaries():
    questions = {
        question.id: question for question in nativeness_sections_for("ko")[0].questions
    }
    guidance = questions["name_address_conventions"].guidance
    assert "민준 김" in guidance
    assert "김 씨" in guidance
    assert "foreign/source name" in guidance
    assert "spacing before 님" in guidance


def test_combined_rubric_row_validates_section_specific_answers():
    binary = RubricAnnotationRow(
        clip_id="clip_001",
        language="es",
        section="interaction",
        factor_id="unnecessary_tool_call",
        question="Did the agent make an unnecessary tool call?",
        answer="No",
        notes="Uncaptured regional phrasing issue.",
        completed=True,
    )
    assert RubricAnnotationRow.from_cells(binary.to_cells()) == binary
    assert match_browser_kind(set(RubricAnnotationRow.headers())) == RUBRIC_PACKET_KIND

    audio = RubricAnnotationRow(
        clip_id="clip_001",
        language="es",
        section="audio",
        factor_id="intonation",
        question="Intonation",
        answer="2",
        audio_consulted=True,
        completed=True,
    )
    assert RubricAnnotationRow.from_cells(audio.to_cells()) == audio

    interaction_score = RubricAnnotationRow(
        clip_id="clip_001",
        language="es",
        section="interaction",
        factor_id=OVERALL_INTERACTION_QUALITY_FACTOR_ID,
        question="Overall, how was the interaction quality?",
        answer="1",
        completed=True,
    )
    assert RubricAnnotationRow.from_cells(interaction_score.to_cells()) == (
        interaction_score
    )
    nativeness_score = RubricAnnotationRow(
        clip_id="clip_001",
        language="es",
        section="nativeness",
        factor_id=OVERALL_NATIVENESS_FACTOR_ID,
        question="Overall, how native-like was the agent's Spanish?",
        answer="4",
        completed=True,
    )
    assert RubricAnnotationRow.from_cells(nativeness_score.to_cells()) == (
        nativeness_score
    )

    with pytest.raises(ValueError, match="no answer"):
        RubricAnnotationRow(question="Question", completed=True)
    with pytest.raises(ValueError, match="has not consulted audio"):
        RubricAnnotationRow(
            section="audio",
            question="Intonation",
            answer="2",
            completed=True,
        )
    with pytest.raises(ValueError, match="non-audio answer"):
        RubricAnnotationRow(
            section="audio",
            question="Intonation",
            answer="Yes",
            audio_consulted=True,
            completed=True,
        )
    with pytest.raises(ValueError, match="needs a 1-4 answer"):
        RubricAnnotationRow(
            section="interaction",
            factor_id=OVERALL_INTERACTION_QUALITY_FACTOR_ID,
            question="Overall interaction quality",
            answer="Yes",
            completed=True,
        )
    with pytest.raises(ValueError, match="belongs to the nativeness section"):
        RubricAnnotationRow(
            section="interaction",
            factor_id=OVERALL_NATIVENESS_FACTOR_ID,
            question="Overall nativeness",
            answer="4",
            completed=True,
        )


def test_rubric_row_severity_turns_and_phase_round_trip():
    row = RubricAnnotationRow(
        clip_id="clip_001",
        language="es",
        section="semantic",
        factor_id="conciseness",
        question="Were the agent's spoken turns too long?",
        answer="Yes",
        severity="major",
        selected_turn_indices=[3, 1, 3],
        phase="pre_reveal",
        completed=True,
    )
    # Indices deduplicate and sort; the cell renders comma-joined and
    # round-trips losslessly through the sanctioned CSV path.
    assert row.selected_turn_indices == [1, 3]
    cells = row.to_cells()
    assert cells["selected_turn_indices"] == "1, 3"
    assert cells["severity"] == "major"
    assert cells["phase"] == "pre_reveal"
    assert RubricAnnotationRow.from_cells(cells) == row
    # A bracketed JSON-array form is absorbed too.
    cells["selected_turn_indices"] = "[1, 3]"
    assert RubricAnnotationRow.from_cells(cells) == row

    # Severity and turns are violation payloads: they require a YES answer.
    with pytest.raises(ValueError, match=r"yes \(violation\) answer"):
        RubricAnnotationRow(
            section="semantic",
            factor_id="conciseness",
            question="q",
            answer="No",
            selected_turn_indices=[1],
        )
    with pytest.raises(ValueError, match=r"yes \(violation\) answer"):
        RubricAnnotationRow(
            section="semantic",
            factor_id="conciseness",
            question="q",
            answer="No",
            severity="minor",
        )


def test_legacy_rubric_csv_headers_without_new_columns_still_dispatch():
    legacy = set(RubricAnnotationRow.headers()) - {
        "severity",
        "selected_turn_indices",
        "phase",
    }
    assert match_browser_kind(legacy) == RUBRIC_PACKET_KIND
    # A partial absence matches too; unrelated header drift stays loud.
    assert match_browser_kind(set(RubricAnnotationRow.headers()) - {"phase"}) == (
        RUBRIC_PACKET_KIND
    )
    assert match_browser_kind(legacy - {"answer"}) is None
    assert match_browser_kind(legacy | {"mystery"}) is None


def test_build_combined_packet_copies_both_modes_and_renders_split_layout(
    tmp_path, monkeypatch
):
    results = make_packet_results_dir(tmp_path / "run")
    source_audio = results / "tasks" / "task_t1" / "sim_s1" / "audio" / "both.wav"
    expected_full_audio = _valid_stereo_wav(source_audio)
    source = _source_manifest(tmp_path, results)
    expected_agent_audio = source.parent / "clips" / "clip_001.wav"
    out = tmp_path / "packet"

    monkeypatch.setattr(
        "tau2.annotation.packets.rubric.caller_gender_for_task",
        lambda task_id, language, domain: "male",
    )
    manifest_path = build_rubric_packet(
        RubricPacketOptions(
            source_manifest=source,
            batch_name="spanish_combined",
            out_dir=out,
            emit_zip=False,
        )
    )

    assert (out / "calls" / "clip_001_full.wav").read_bytes() == expected_full_audio
    assert (
        out / "calls" / "clip_001_agent.wav"
    ).read_bytes() == expected_agent_audio.read_bytes()
    manifest = ArtifactManifest.model_validate_json(manifest_path.read_text())
    assert manifest.kind == RUBRIC_PACKET_KIND
    entry = manifest.rubric_entries[0]
    assert entry.sim_id == "s1"
    assert entry.full_audio_file == "calls/clip_001_full.wav"
    assert entry.agent_audio_file == "calls/clip_001_agent.wav"
    assert entry.agent_gender == "female"
    assert entry.caller_gender == "male"
    assert "No machine judge verdicts" in manifest.provenance["judge_policy"]

    html = (out / "index.html").read_text()
    config = packet_config(out / "index.html")
    assert len(config["questions"]) == 20
    assert [question["section"] for question in config["questions"]].count(
        "interaction"
    ) == 6
    assert [question["section"] for question in config["questions"]].count(
        "nativeness"
    ) == 3
    assert [question["section"] for question in config["questions"]].count(
        "audio"
    ) == 11
    assert config["contexts"]["clip_001"] == {
        "agent_gender": "female",
        "caller_gender": "male",
    }
    assert config["calls"]["clip_001"]["simulation_id"] == "s1"
    assert config["calls"]["clip_001"]["task_id"] == "t1"
    assert config["calls"]["clip_001"]["agent_turns"][0]["turn_id"] == (
        "agent-turn-000"
    )
    assert config["packet_version"] == "combined-rubric-packet-v16"
    assert "Full conversation" in html and "Agent only" in html
    assert "Overall + interaction + audio CSV" in html and "Nativeness CSV" in html
    assert 'data-native-label="violation"' in html
    # Every answer button family needs a visible pressed state — a click with
    # no visual feedback reads as a dead control (owner-reported on Pass).
    for label in ("pass", "violation", "not_applicable"):
        assert (
            f'.rp-options button[aria-pressed="true"][data-native-label="{label}"]'
            in html
        )
    call_nativeness_count = sum(
        question["answer_kind"] == "nativeness"
        and question["evaluation_level"] == "call"
        for question in config["questions"]
    )
    assert html.count("<select data-native-severity>") == call_nativeness_count
    assert "data-agent-turn" in html
    # EVERY utterance-level question (any answer kind) renders the
    # turn-selection input; nativeness questions always carry the hidden
    # input (their call-level detail block includes it), so the total is
    # nativeness questions + non-nativeness utterance questions.
    utterance_count = sum(
        question["evaluation_level"] == "utterance" for question in config["questions"]
    )
    nativeness_count = sum(
        question["answer_kind"] == "nativeness" for question in config["questions"]
    )
    utterance_non_nativeness = sum(
        question["evaluation_level"] == "utterance"
        and question["answer_kind"] != "nativeness"
        for question in config["questions"]
    )
    assert utterance_non_nativeness > 0
    assert html.count(
        '<input type="hidden" data-selected-turn-indices value="[]">'
    ) == (nativeness_count + utterance_non_nativeness)
    assert html.count('<span class="rp-level-badge">utterance</span>') == (
        utterance_count
    )
    # Semantic-source questions carry the minor/major severity control.
    severity_count = sum(
        question["severity_scale"] == "minor_major" for question in config["questions"]
    )
    assert severity_count == 2  # unnecessary_repetition + unnecessary_tool_call
    assert html.count('<button type="button" data-severity="major"') == severity_count
    # A blind rubric packet renders NO two-phase machinery (the shared JS/CSS
    # carry the inert hooks, so assert on the rendered markup).
    assert "data-lock-blind disabled" not in html
    assert ' judge-phased"' not in html
    assert config["judge_visible"] is False
    assert (
        "if (interactionSection.open) setEvidenceMode(card, 'full_conversation')"
        in (html)
    )
    assert "if (audioSection.open) setEvidenceMode(card, 'agent_only')" in html
    assert "nativenessSection.addEventListener" not in html
    assert "<th>Tick</th>" not in html
    assert "rp-workspace" in html
    assert "rp-evidence-pane" in html and "rp-annotations" in html
    assert "Required for the audio judge section only — optional for the rest." in html
    assert "Interaction and workflow quality" in html
    assert "Spanish language nativeness" in html
    assert (
        sum(question["answer_kind"] == "likert4" for question in config["questions"])
        == 2
    )
    assert 'data-answer-kind="likert4"' in html
    assert "Incredibly frustrating" in html and "Pleasant" in html
    assert "Clearly non-native" in html and "Native-speaker-like" in html
    assert "question.answer_kind !== 'nativeness'" in html
    assert "question.answer_kind === 'nativeness'" in html
    assert "Audio quality" in html
    assert "Anything the metrics missed?" in html
    assert "especially a nativeness issue" in html
    assert "find_reservation" in html
    assert "Agent gender" in html and "Caller gender" in html
    assert "I reviewed the complete agent transcript" not in html
    assert "I consulted the audio" not in html
    assert "openai:test" not in html
    assert "quality_info" not in html
