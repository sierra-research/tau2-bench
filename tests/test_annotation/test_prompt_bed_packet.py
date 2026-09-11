# Copyright Sierra
"""Prompt + bed review packet: row contract, ingest dispatch, and a REAL
build for Spanish (prompts rendered through the actual runtime user-sim
build path; beds from a fake locale tree so the test copies bytes, not the
190MB of real audio; voice samples through a fake TTS renderer so no
ElevenLabs call happens)."""

import csv
import html
import json
import re

import pytest

from tau2 import voice_config
from tau2.annotation.artifacts import CSV_ENCODING, ArtifactManifest, armor_cell
from tau2.annotation.models import (
    PromptBedItemKind,
    PromptBedRow,
    PromptBedVerdict,
)
from tau2.annotation.packets.forms import PROMPT_BED_KIND, ingest_browser_csv
from tau2.annotation.packets.prompt_bed import (
    CONVERSATION_HISTORY_PLACEHOLDER,
    PROMPT_CATEGORIES,
    PromptBedPacketOptions,
    build_prompt_bed_packet,
)
from tau2.multilingual.factory.paths import (
    OUTDOOR_BED_BASENAME,
    SHARED_OFFICE_BED_BASENAME,
    TV_KITCHEN_BED_BASENAME,
)
from test_annotation.conftest import packet_config

LOCALES = [
    "ar_EG",
    "de_DE",
    "en_US",
    "es_ES",
    "fr_FR",
    "hi_IN",
    "it_IT",
    "ja_JP",
    "ko_KR",
    "nl_NL",
    "pl_PL",
    "pt_BR",
    "ro_RO",
    "ru_RU",
    "tr_TR",
    "vi_VN",
    "zh_CN",
]


# ---------------------------------------------------------------------------
# Row model + browser-CSV ingest dispatch
# ---------------------------------------------------------------------------


def test_prompt_bed_row_round_trip():
    row = PromptBedRow(
        row_kind=PromptBedItemKind.BED,
        batch="prompt_bed_review_es",
        rater="Ana",
        language="es",
        item_id="bed_es_ES_outdoor",
        locale="es_ES",
        bed_type="outdoor",
        verdict=PromptBedVerdict.ISSUE,
        notes="loop point audible at ~30s",
        completed=True,
        created_at="2026-07-19T00:00:00Z",
    )
    assert PromptBedRow.from_cells(row.to_cells()) == row


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("OK", PromptBedVerdict.OK),
        ("ok", PromptBedVerdict.OK),
        ("✅ yes", PromptBedVerdict.OK),
        ("issue", PromptBedVerdict.ISSUE),
        ("Issues found", PromptBedVerdict.ISSUE),
        ("Sounds off", PromptBedVerdict.ISSUE),
        ("", None),
    ],
)
def test_verdict_normalizer_absorbs_radio_labels(raw, expected):
    assert PromptBedRow.model_validate({"verdict": raw}).verdict is expected


def test_verdict_normalizer_junk_is_loud():
    with pytest.raises(ValueError, match="unrecognized prompt/bed verdict"):
        PromptBedRow.model_validate({"verdict": "meh"})


def test_ingest_browser_csv_dispatches_prompt_bed(tmp_path):
    rows = [
        PromptBedRow(
            row_kind=PromptBedItemKind.PROMPT,
            batch="prompt_bed_review_es",
            rater="Ana",
            language="es",
            item_id="system_prompt_english",
            verdict=PromptBedVerdict.OK,
            completed=True,
        ),
        PromptBedRow(
            row_kind=PromptBedItemKind.BED,
            batch="prompt_bed_review_es",
            rater="Ana",
            language="es",
            item_id="bed_shared_office",
            locale="shared",
            bed_type="shared_office",
            verdict=PromptBedVerdict.ISSUE,
            notes="=SUM(A1) chatter too loud",
        ),
    ]
    path = tmp_path / "prompt_bed_review_es_Ana.csv"
    with open(path, "w", newline="", encoding=CSV_ENCODING) as fp:
        writer = csv.DictWriter(fp, fieldnames=PromptBedRow.headers())
        writer.writeheader()
        for row in rows:
            writer.writerow({h: armor_cell(v) for h, v in row.to_cells().items()})

    browser = ingest_browser_csv(path)
    assert browser.kind == PROMPT_BED_KIND
    assert browser.form is None
    assert browser.completed == 1
    assert browser.rows == rows


# ---------------------------------------------------------------------------
# The REAL build (es), with a fake bed tree
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_beds(tmp_path, monkeypatch):
    root = tmp_path / "continuous"
    for i, locale in enumerate(LOCALES):
        d = root / locale
        d.mkdir(parents=True)
        (d / OUTDOOR_BED_BASENAME).write_bytes(f"outdoor-{i}".encode())
        (d / TV_KITCHEN_BED_BASENAME).write_bytes(f"tv-{i}".encode())
    (root / SHARED_OFFICE_BED_BASENAME).write_bytes(b"office")
    monkeypatch.setattr(voice_config, "BACKGROUND_NOISE_CONTINUOUS_DIR", root)
    return root


@pytest.fixture(scope="module")
def es_task():
    from tau2.multilingual.run_presets import matrix_task_set_name
    from tau2.registry import registry

    task_set_name = matrix_task_set_name("airline", "es")
    return registry.get_tasks_loader(task_set_name)()[0]


def fake_render_voice_samples(language, *, out_dir, label="pinned", **kwargs):
    """Stand-in for the ElevenLabs render: same filenames and result shape,
    fake bytes."""
    from tau2.multilingual.factory.voice_samples import VoiceSampleResult
    from tau2.multilingual.registry import get_language_pack

    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for pid, persona in sorted(get_language_pack(language).personas.items()):
        gender = persona.tags.get("gender", "unknown")
        path = out_dir / f"{language}_{pid}_{gender}_{label}.mp3"
        path.write_bytes(f"mp3-{pid}".encode())
        results.append(
            VoiceSampleResult(
                language=language, label=f"{pid}_{label}", voice_id="fake", path=path
            )
        )
    return results


@pytest.fixture(scope="module")
def built_packet(tmp_path_factory):
    """One real es build shared by the assertions below (construction is
    offline but not free — build once)."""
    from tau2.multilingual.factory import voice_samples as voice_samples_module

    out_root = tmp_path_factory.mktemp("prompt_bed") / "packets"
    root = tmp_path_factory.mktemp("beds") / "continuous"
    for i, locale in enumerate(LOCALES):
        d = root / locale
        d.mkdir(parents=True)
        (d / OUTDOOR_BED_BASENAME).write_bytes(f"outdoor-{i}".encode())
        (d / TV_KITCHEN_BED_BASENAME).write_bytes(f"tv-{i}".encode())
    (root / SHARED_OFFICE_BED_BASENAME).write_bytes(b"office")
    original = voice_config.BACKGROUND_NOISE_CONTINUOUS_DIR
    original_render = voice_samples_module.render_voice_samples
    voice_config.BACKGROUND_NOISE_CONTINUOUS_DIR = root
    voice_samples_module.render_voice_samples = fake_render_voice_samples
    try:
        manifest_path = build_prompt_bed_packet(
            PromptBedPacketOptions(language="es", out_dir=out_root)
        )
    finally:
        voice_config.BACKGROUND_NOISE_CONTINUOUS_DIR = original
        voice_samples_module.render_voice_samples = original_render
    return manifest_path


def test_packet_folder_contents(built_packet):
    packet_dir = built_packet.parent
    assert packet_dir.name == "es"
    assert (packet_dir / "index.html").is_file()
    wavs = sorted(p for p in (packet_dir / "beds").rglob("*.wav"))
    assert len(wavs) == 35  # 17 locales x 2 beds + the shared office bed
    assert (packet_dir / "beds" / SHARED_OFFICE_BED_BASENAME).is_file()
    assert (packet_dir / "beds" / "es_ES" / OUTDOOR_BED_BASENAME).is_file()
    mp3s = sorted(p.name for p in (packet_dir / "voices").glob("*.mp3"))
    assert mp3s == [
        "es_alejandro_es_v1_male_packet.mp3",
        "es_camila_es_v1_female_packet.mp3",
    ]


def test_page_carries_real_rendered_prompts(built_packet):
    page = (built_packet.parent / "index.html").read_text()
    user_part, rest = page.split('id="section-backchannel_decision_prompt"', 1)

    # The one arm runs: versioned directive + localization glossary + a
    # register pragmatics clause, all inside the user-sim system prompt.
    assert "## LANGUAGE OF THE CALL (MANDATORY)" in user_part
    assert "## LANGUAGE LOCALIZATION (Spanish)" in user_part
    assert "Peninsular" in user_part  # camila_es_v1 pragmatics clause

    # The packet's domain glossary really reaches the prompt. Asserted against
    # the pack's own value, not a literal: the natives are machine-drafted and
    # a re-draft (`draft-localization --force`) legitimately rewords them.
    from tau2.multilingual.registry import get_language_pack

    glossary = get_language_pack("es").localization.domain_glossaries["airline"]
    user_id = next(g for g in glossary if g.term_id == "user_id")
    assert user_id.native in user_part

    # One user-sim prompt on the page, so exactly one directive render.
    assert page.count("## LANGUAGE OF THE CALL (MANDATORY)") == 1

    # Backchannel prompt rendered with the placeholder slot; greeting verbatim.
    assert CONVERSATION_HISTORY_PLACEHOLDER in rest
    assert "¿En qué puedo ayudarle?" in rest


def test_scenario_prose_is_english_with_localized_entities(built_packet, es_task):
    """The rendered prompt carries the ENGLISH task instructions — the
    localized prose is the retired native arm's, and no longer reaches a
    run."""
    page = html.unescape((built_packet.parent / "index.html").read_text())
    user_part = page.split('id="section-backchannel_decision_prompt"', 1)[0]
    instructions = es_task.user_scenario.instructions
    fragment = str(instructions.task_instructions)[:60]
    assert fragment, "es task 0 has no task_instructions to check against"
    assert fragment not in user_part


def test_judge_and_factory_sections_render_real_pack_data(built_packet):
    page = html.unescape((built_packet.parent / "index.html").read_text())

    # Agent-side: the voice agent prompt ends with the pack's language clause.
    from tau2.multilingual.registry import get_language_pack

    pack = get_language_pack("es")
    agent_part = page.split('id="section-agent_system_prompt_voice"', 1)[1]
    assert pack.agent_language_clause.strip()[:40] in agent_part
    assert "originally from Madrid, Spain" in agent_part
    assert "speaks Spanish — expect Peninsular Spanish (Spain)" in agent_part

    # Nativeness judge: enabled factor ids appear in the exact runtime groups.
    nativeness_part = page.split('id="section-nativeness_judge_user_prompts"', 1)[
        1
    ].split("</section>", 1)[0]
    from tau2.judges.nativeness.factors import judge_factors_for

    runtime_factors = judge_factors_for("es")
    for factor in (f for f in runtime_factors if f.enabled and f.type == "judge"):
        assert f'"factor_id": "{factor.id}"' in nativeness_part
    for factor in (f for f in runtime_factors if f.type != "judge"):
        assert f'"factor_id": "{factor.id}"' not in nativeness_part
    assert "CALL-LEVEL BATCH" not in nativeness_part
    assert nativeness_part.count("UTTERANCE-LEVEL SHARED BATCH") == 1
    isolated_heading = "UTTERANCE-LEVEL ISOLATED — natural_word_choice"
    assert nativeness_part.count(isolated_heading) == 1
    isolated_part, shared_part = nativeness_part.split(
        "UTTERANCE-LEVEL SHARED BATCH", 1
    )
    assert '"factor_id": "natural_word_choice"' in isolated_part
    assert '"factor_id": "natural_word_choice"' not in shared_part
    for factor in (
        factor
        for factor in runtime_factors
        if factor.enabled
        and factor.type == "judge"
        and factor.id != "natural_word_choice"
    ):
        assert f'"factor_id": "{factor.id}"' in shared_part
        assert f'"factor_id": "{factor.id}"' not in isolated_part

    # Delivery judge: the pack-mode rubric rendered for Spanish.
    assert "Language-specific delivery rubric for Spanish (es)" in page
    # Communicate judge: the language addendum rendered with the code.
    assert "ISO 639-1 code 'es'" in page

    # Factory: translation guidance filled (no leftover template slot). The
    # voice-design prompts are deliberately ABSENT — the voices section
    # carries rendered audio instead.
    assert "{translation_guidance}" not in page
    for persona in pack.personas.values():
        assert persona.tts_voice_prompt.strip()[:40] not in page


def test_identity_swap_section_samples_the_real_map(built_packet):
    """The runtime-user group carries sample rows of the es identity map —
    English caller → localized name/email/gender, straight from the file."""
    from tau2.multilingual.factory.entity_localization import identity_map_path

    page = html.unescape((built_packet.parent / "index.html").read_text())
    section = page.split('id="section-identity_swap_examples"', 1)[1].split(
        "</section>", 1
    )[0]
    identity_map = json.loads(identity_map_path("es", "airline").read_text())
    for caller_key, identity in sorted(identity_map.items())[:8]:
        assert caller_key in section
        assert f"{identity['first_name']} {identity['last_name']}" in section
        assert identity["email"] in section
        # Airline localizes addresses too — the city rides along.
        assert identity["address"]["city"] in section


def test_language_without_identity_map_omits_the_section():
    from tau2.annotation.packets.prompt_bed import _identity_swap_section

    assert _identity_swap_section("en", "telecom") is None
    section = _identity_swap_section("es", "airline")
    assert section is not None and section.category == "runtime_user"


def test_packet_config_and_manifest(built_packet):
    cfg = packet_config(built_packet.parent / "index.html")
    assert cfg["kind"] == PROMPT_BED_KIND
    assert cfg["language"] == "es"
    assert cfg["csv_headers"] == PromptBedRow.headers()
    prompt_items = [i for i in cfg["items"] if i["row_kind"] == "prompt"]
    voice_items = [i for i in cfg["items"] if i["row_kind"] == "voice"]
    bed_items = [i for i in cfg["items"] if i["row_kind"] == "bed"]
    prompt_ids = [i["item_id"] for i in prompt_items]
    # The EXACT per-language inventory: global (language-independent)
    # prompts are deliberately excluded — they are noise to a native
    # reviewer. Adding/removing a section must update this list.
    assert prompt_ids == [
        "system_prompt_english",
        "backchannel_decision_prompt",
        "agent_greeting",
        "speech_phrases",
        "identity_swap_examples",
        "agent_system_prompt_voice",
        "agent_system_prompt_text",
        "nativeness_judge_user_prompts",
        "delivery_judge_user_prompt",
        "communicate_judge_prompts",
        "nl_assertions_judge_prompts",
        "factory_translator_system_prompt",
        "factory_fixer_system_prompt",
        "factory_verifier_system_prompt",
        "factory_localization_drafting_prompts",
        "factory_autoform_prompts",
        "factory_nuance_candidates_prompt",
    ]
    assert [i["item_id"] for i in voice_items] == [
        "voice_es_alejandro_es_v1",
        "voice_es_camila_es_v1",
    ]
    # Every category renders, and the category map matches the sections.
    by_category = ArtifactManifest.model_validate_json(
        built_packet.read_text()
    ).provenance["sections_by_category"]
    assert set(by_category) == set(PROMPT_CATEGORIES)
    assert all(by_category[k] for k in PROMPT_CATEGORIES)
    assert sum(len(v) for v in by_category.values()) == len(prompt_ids)
    assert len(bed_items) == 35
    assert {i["locale"] for i in bed_items} == {*LOCALES, "shared"}

    manifest = ArtifactManifest.model_validate_json(built_packet.read_text())
    assert manifest.kind == PROMPT_BED_KIND
    assert manifest.language == "es"
    assert manifest.domain == "airline"
    assert manifest.form == "prompt_bed_review"
    assert manifest.batch_id == cfg["batch_id"]
    assert manifest.provenance["task_set_name"] == "airline_es_identity"
    assert manifest.provenance["task_id"] == "0_es_identity"
    # Task 0's caller-gender sidecar pins the female pack persona.
    assert manifest.provenance["persona_id"] == "camila_es_v1"
    assert set(manifest.provenance["prompt_sha256"]) == {
        i["item_id"] for i in prompt_items
    }
    assert set(manifest.provenance["voices"]) == {i["item_id"] for i in voice_items}
    assert len(manifest.provenance["beds"]) == 35


def test_no_voice_samples_builds_offline(fake_beds, tmp_path):
    """voice_samples=False must never touch the TTS renderer: no voices
    section, no voices/ dir, provenance says skipped."""
    from tau2.multilingual.factory import voice_samples as voice_samples_module

    def explode(*args, **kwargs):
        raise AssertionError("TTS renderer called despite --no-voice-samples")

    original = voice_samples_module.render_voice_samples
    voice_samples_module.render_voice_samples = explode
    try:
        manifest_path = build_prompt_bed_packet(
            PromptBedPacketOptions(
                language="es", out_dir=tmp_path / "out", voice_samples=False
            )
        )
    finally:
        voice_samples_module.render_voice_samples = original

    assert not (manifest_path.parent / "voices").exists()
    cfg = packet_config(manifest_path.parent / "index.html")
    assert not [i for i in cfg["items"] if i["row_kind"] == "voice"]
    manifest = ArtifactManifest.model_validate_json(manifest_path.read_text())
    assert manifest.provenance["voices"] == "skipped (--no-voice-samples)"
    assert "Persona voices" not in (manifest_path.parent / "index.html").read_text()


def test_existing_packet_dir_is_loud(built_packet):
    with pytest.raises(FileExistsError, match="already exists"):
        build_prompt_bed_packet(
            PromptBedPacketOptions(language="es", out_dir=built_packet.parent.parent)
        )


def test_unknown_language_is_loud(fake_beds, tmp_path):
    with pytest.raises(ValueError, match="no registered language pack"):
        build_prompt_bed_packet(
            PromptBedPacketOptions(language="xx", out_dir=tmp_path / "out")
        )


def test_missing_bed_is_loud(fake_beds, tmp_path):
    (fake_beds / "es_ES" / OUTDOOR_BED_BASENAME).unlink()
    with pytest.raises(FileNotFoundError, match="es_ES"):
        build_prompt_bed_packet(
            PromptBedPacketOptions(language="es", out_dir=tmp_path / "out")
        )


def test_english_judge_sections_omit_language_addendum():
    """The runtime judges skip the language addendum on en runs — the packet
    must show the prompts an English run actually uses (Bugbot, PR #376)."""
    from tau2.annotation.packets.prompt_bed import _judge_sections
    from tau2.multilingual.registry import get_language_pack

    by_id = {s.item_id: s for s in _judge_sections("en", get_language_pack("en"))}
    for item_id in ("communicate_judge_prompts", "nl_assertions_judge_prompts"):
        assert "ISO 639-1 code" not in by_id[item_id].text  # the addendum body
        assert "English run — no language addendum" in by_id[item_id].text

    es_by_id = {s.item_id: s for s in _judge_sections("es", get_language_pack("es"))}
    assert "SYSTEM PROMPT (with language addendum)" in (
        es_by_id["communicate_judge_prompts"].text
    )


def test_nativeness_prompt_bed_headings_match_runtime_group_order():
    from tau2.annotation.packets.prompt_bed import _judge_sections
    from tau2.multilingual.registry import get_language_pack

    expected = {
        "es": [
            "UTTERANCE-LEVEL ISOLATED — natural_word_choice",
            "UTTERANCE-LEVEL SHARED BATCH",
        ],
        "pt": [
            "CALL-LEVEL BATCH",
            "UTTERANCE-LEVEL SHARED BATCH",
            "UTTERANCE-LEVEL ISOLATED — natural_word_choice",
        ],
    }
    for language, expected_headings in expected.items():
        sections = {
            section.item_id: section
            for section in _judge_sections(language, get_language_pack(language))
        }
        text = sections["nativeness_judge_user_prompts"].text
        assert re.findall(r"───── (.+?) ─────", text) == expected_headings


def test_factory_sections_render_the_packets_domain():
    """The translator/fixer prompts must render the PACKET's domain — a
    telecom packet showing airline example entities under a caption claiming
    'this produced the scenarios you reviewed' is the wrong-content-rendered
    bug class this page exists to prevent."""
    from tau2.annotation.packets.prompt_bed import _factory_sections
    from tau2.multilingual.domain_profiles import get_domain_profile
    from tau2.multilingual.registry import get_language_pack

    pack = get_language_pack("es")
    airline_entities = get_domain_profile("airline").translation_example_entities
    telecom_entities = get_domain_profile("telecom").translation_example_entities

    for domain, wanted, unwanted in [
        ("airline", airline_entities, telecom_entities),
        ("telecom", telecom_entities, airline_entities),
    ]:
        sections = {s.item_id: s.text for s in _factory_sections("es", pack, domain)}
        translator = sections["factory_translator_system_prompt"]
        assert wanted in translator, domain
        assert unwanted not in translator, domain
