"""Banking date speech has a versioned, voice-only regional binding."""

import json
from pathlib import Path

import pytest

from tau2.multilingual.factory.pack_extension import (
    BANKING_DATE_AND_SHIPPING_VERSION,
    BANKING_VOICE_DATE_GUIDANCE_VERSION,
    BANKING_VOICE_DATE_GUIDELINES,
    publish_pack_extension,
)
from tau2.multilingual.runtime import extension_for_domain, load_language, resolve_task
from tau2.user.guidelines import render_simulation_guidelines
from tau2.user.user_simulator import build_user_system_prompt


@pytest.mark.parametrize(
    "language,spoken_order",
    [
        ("hi", "native Indian DD/MM/YYYY order: day, month, year"),
        ("es", "native Spanish DD/MM/YYYY order: day, month, year"),
        ("pt", "native Brazilian DD/MM/YYYY order: day, month, year"),
        ("ko", "native Korean year–month–day order"),
        ("zh", "native Mainland Mandarin year–month–day order"),
    ],
)
def test_banking_date_guidance_reaches_only_voice_call(language, spoken_order):
    old = load_language(
        language, "banking_knowledge", mode="voice", pack_extension="banking"
    )
    new = load_language(
        language, "banking_knowledge", mode="voice", pack_extension="banking-dates-v1"
    )
    assert extension_for_domain(language, "banking_knowledge") == "banking-dates-v1"
    assert new.pack_extension == "banking-dates-v1"
    assert new.pack_sha256 != old.pack_sha256
    assert new.pack.binding("banking_knowledge").revision.endswith(
        BANKING_VOICE_DATE_GUIDANCE_VERSION
    )
    if language not in {"es", "pt", "ko", "hi", "zh"}:
        assert old.tasks_sha256 == new.tasks_sha256
    assert old.metadata_sha256 == new.metadata_sha256
    excluded = {"revision", "knowledge_document_overrides"}
    if language in {"es", "pt", "hi", "zh"}:
        excluded.add("tasks")
    if language in {"ko", "zh"}:
        excluded.update({"database", "tasks", "localization_map_sha256"})
    assert old.scenario.model_dump(exclude=excluded) == new.scenario.model_dump(
        exclude=excluded
    )
    if language not in {"ko", "zh"}:
        assert (
            old.task_index[f"task_018_{language}"]
            == new.task_index[f"task_018_{language}"]
        )

    task = new.task_index[f"task_018_{language}"]
    assert "12/05/1993" in task.user_scenario.instructions
    voice = resolve_task(new, task, seed=0).persona
    assert voice is not None
    guidelines = render_simulation_guidelines(voice, use_tools=False)
    prompt = build_user_system_prompt(
        task.user_scenario.instructions,
        persona_config=voice,
        mode="voice",
        guidelines=guidelines,
    )
    guidance = BANKING_VOICE_DATE_GUIDELINES[language]
    assert spoken_order in prompt
    assert prompt.count(guidance) == 1
    assert "Keep the calendar date unchanged" in prompt
    assert "this takes precedence over the general instruction" in prompt

    text = load_language(
        language, "banking_knowledge", mode="text", pack_extension="banking-dates-v1"
    )
    text_persona = resolve_task(text, text.task_index[task.id], seed=0).persona
    assert text_persona is not None
    assert guidance not in (text_persona.to_guidelines_text(mode="text") or "")
    for other_domain, extension in (("retail", None), ("intake", "intake")):
        other = load_language(
            language, other_domain, mode="voice", pack_extension=extension
        )
        other_persona = resolve_task(other, other.tasks[0], seed=0).persona
        assert other_persona is not None
        assert guidance not in (other_persona.to_guidelines_text(mode="voice") or "")


def test_factory_publishes_fixed_banking_date_guidance(tmp_path):
    root = Path(__file__).resolve().parents[2] / "data/tau2/multilingual_extensions"
    bundle = root / "banking/hi/regional/banking_knowledge.json"
    kwargs = dict(
        name="banking-dates-v1",
        language="hi",
        revision=BANKING_DATE_AND_SHIPPING_VERSION,
        bundle_paths=[bundle],
        output=tmp_path / "banking-dates-v1/hi",
    )
    first = publish_pack_extension(**kwargs)
    assert (
        first.domains["banking_knowledge"].voice_guidelines
        == (BANKING_VOICE_DATE_GUIDELINES["hi"])
    )
    assert first.domains["banking_knowledge"].text_guidelines == ""
    assert first.domains["banking_knowledge"].revision.endswith(
        BANKING_VOICE_DATE_GUIDANCE_VERSION
    )
    second = publish_pack_extension(**kwargs)
    assert first == second

    old = publish_pack_extension(
        name="banking",
        language="hi",
        revision="banking-v1",
        bundle_paths=[bundle],
        output=tmp_path / "banking/hi",
    )
    assert old.domains["banking_knowledge"].voice_guidelines == ""
    assert (
        old.domains["banking_knowledge"].revision
        == json.loads(bundle.read_text())["revision"]
    )
