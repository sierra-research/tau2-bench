# Copyright Sierra
"""Offline end-to-end tests for the reviewed factory nativeness stage."""

import json
from types import SimpleNamespace

import pytest
import yaml

import tau2.multilingual.factory.llm as factory_llm_module
import tau2.multilingual.registry as multilingual_registry
from tau2.judges.nativeness.factors import judge_factors_for
from tau2.multilingual.factory.author_form import FactoryDraftError
from tau2.multilingual.factory.finalize import finalize
from tau2.multilingual.factory.llm import FactoryLLM
from tau2.multilingual.factory.nativeness_drafting import (
    NATIVENESS_AUTHOR_CALL_NAME,
    NATIVENESS_PROMPT_VERSION,
    ReviewStatus,
    _load_draft,
    apply_nativeness_review,
    draft_nativeness_review,
    load_nativeness_review,
    nativeness_catalog_sha256,
    nativeness_prompt_sha256,
)
from tau2.multilingual.factory.state import FactoryProject, FactoryStage, StageStatus
from tau2.multilingual.nativeness_catalog import get_factor_prompt_base
from test_multilingual.factory_testing.toy_language import toy_pack_yaml_dict


def _valid_reply() -> dict:
    return {
        "judge_factors": [
            {
                "factor_id": "regional_consistency",
                "nuance": "Toyland Standard Toylang",
                "target_variety": (
                    "Toyland Standard Toylang (TL; northern and southern "
                    "persona variation allowed)"
                ),
                "agent_rule": (
                    "The agent stays within Toyland Standard Toylang vocabulary."
                ),
                "natural_examples": ["Ik kijk het meteen voor u na, zo."],
                "violation_rule": (
                    "The agent switches to Overseas Toylang words or grammar."
                ),
                "violating_examples": ["Mi checka dat overseas voor jou."],
            },
            {
                "factor_id": "register_formality",
                "nuance": "Toylang service address",
                "agent_rule": ("The agent keeps one service-appropriate address form."),
                "natural_examples": ["Goedan dag zo, hoe kan ik helpen?"],
                "violation_rule": (
                    "The agent switches abruptly to the intimate address form."
                ),
                "violating_examples": ["Hé jij, geef je code."],
            },
            {
                "factor_id": "translationese",
                "nuance": "Natural Toylang phrasing",
                "agent_rule": "The agent uses ordinary spoken Toylang phrasing.",
                "natural_examples": ["Ik kijk het meteen voor u na."],
                "violation_rule": ("The agent copies English word order into Toylang."),
                "violating_examples": ["Ik zal maken een controle voor u."],
            },
        ]
    }


@pytest.fixture
def nativeness_workspace(factory_dir):
    project = FactoryProject(language="tl", script="latn", pack_revision=4)
    project.save()
    data = toy_pack_yaml_dict()
    # A factory draft uses workspace-relative filenames and has no shipped voice
    # pins. Neither distinction matters to LanguagePack validation, but this is
    # the real authoring shape rather than a registered final pack.
    data["guidelines_voice_path"] = "guidelines_draft.md"
    (project.project_dir / "guidelines_draft.md").write_text(
        "# Toylang guidelines\n\n<PERSONA_GUIDELINES>\n"
    )
    for persona in data["personas"].values():
        persona.pop("voice_id", None)
    project.pack_draft_path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
        + "\n# --- nativeness scoring (AUTHOR THIS) ---\n"
        + "# nativeness:\n"
        + "# --- delivery scoring (audio-layer; AUTHOR THIS) ---\n"
        + "# delivery:\n"
    )
    return project


@pytest.fixture
def mocked_generate(monkeypatch):
    calls = []

    def fake_generate(*, model, messages, call_name, **kwargs):
        calls.append(
            {
                "model": model,
                "messages": messages,
                "call_name": call_name,
                "kwargs": kwargs,
            }
        )
        if call_name != NATIVENESS_AUTHOR_CALL_NAME:
            raise AssertionError(f"unexpected generate call: {call_name}")
        return SimpleNamespace(content=json.dumps(_valid_reply(), ensure_ascii=False))

    monkeypatch.setattr(factory_llm_module, "generate", fake_generate)
    return calls


def _approve(project: FactoryProject) -> None:
    path = project.project_dir / "nativeness_review.yaml"
    data = yaml.safe_load(path.read_text())
    data["review_status"] = ReviewStatus.APPROVED.value
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))


def test_offline_generate_review_apply_and_runtime_composition(
    nativeness_workspace, mocked_generate, monkeypatch
):
    project = nativeness_workspace
    original = project.pack_draft_path.read_text()

    drafted = draft_nativeness_review("tl", llm=FactoryLLM())

    assert drafted.written
    assert project.pack_draft_path.read_text() == original  # review pause: no apply
    assert [call["call_name"] for call in mocked_generate] == [
        NATIVENESS_AUTHOR_CALL_NAME
    ]
    call = mocked_generate[0]
    assert call["kwargs"]["reasoning_effort"] == "high"
    system = call["messages"][0].content
    user = call["messages"][1].content
    normalized_system = " ".join(system.split())
    assert "Judge only agent speech" in normalized_system
    assert "Do not include research history" in normalized_system
    assert "CLOSED NATIVENESS CATALOG" in user
    assert "register_formality" in user
    assert "Peninsular Spanish" in normalized_system
    assert '"locale": "TL-NORD"' in user

    packet_text = drafted.review_path.read_text()
    packet = load_nativeness_review("tl")
    assert packet is not None
    assert packet.review_status is ReviewStatus.PENDING_REVIEW
    assert packet.provenance.call_name == NATIVENESS_AUTHOR_CALL_NAME
    assert packet.provenance.prompt_version == NATIVENESS_PROMPT_VERSION
    assert packet.provenance.prompt_sha256 == nativeness_prompt_sha256()
    assert packet.provenance.catalog_sha256 == nativeness_catalog_sha256()
    # The review contract cannot carry or regenerate catalog-owned prompt text.
    assert "canonical_question" not in packet_text
    assert "opportunity:" not in packet_text
    assert "shared_allowed" not in packet_text
    assert "shadow:" not in packet_text

    with pytest.raises(FactoryDraftError, match="pending_review"):
        apply_nativeness_review("tl")
    with pytest.raises(FactoryDraftError, match="has not been applied"):
        finalize("tl")

    _approve(project)
    applied = apply_nativeness_review("tl")
    assert applied.written
    assert applied.factor_ids == [
        "regional_consistency",
        "register_formality",
        "translationese",
    ]
    text = project.pack_draft_path.read_text()
    assert "nativeness scoring (AUTHOR THIS" not in text
    assert "delivery scoring (audio-layer; AUTHOR THIS)" in text
    raw = yaml.safe_load(text)
    rubrics = raw["nativeness"]["judge_factors"]
    assert [rubric["factor_id"] for rubric in rubrics] == applied.factor_ids
    assert all("shadow" not in rubric for rubric in rubrics)  # live by default
    assert all("question" not in rubric for rubric in rubrics)
    assert "Target variety: Toyland Standard Toylang" in rubrics[0]["native_does"]
    assert "Target variety: Toyland Standard Toylang" in rubrics[0]["ai_likely_does"]
    assert "Goedan dag zo" in rubrics[1]["native_does"]
    assert "Hé jij" in rubrics[1]["ai_likely_does"]
    assert FactoryProject.load("tl").stages[FactoryStage.NATIVENESS] == (
        StageStatus.DONE
    )

    # Runtime composition still owns all universal text in the catalog.
    _, _, _, loaded_pack = _load_draft("tl")
    monkeypatch.setattr(
        multilingual_registry,
        "get_language_pack",
        lambda language: loaded_pack if language == "tl" else None,
    )
    factors = {factor.id: factor for factor in judge_factors_for("tl")}
    base = get_factor_prompt_base("register_formality")
    resolved = factors["register_formality"]
    assert resolved.params.question == base.question
    assert resolved.params.opportunity == base.opportunity
    assert resolved.params.shared_allowed == base.allowed
    assert resolved.params.shared_violation == base.violation
    assert "Goedan dag zo" in resolved.params.language_allowed
    assert not resolved.shadow


def test_generation_and_application_are_idempotent_no_touch(
    nativeness_workspace, mocked_generate
):
    project = nativeness_workspace
    first = draft_nativeness_review("tl", llm=FactoryLLM())
    packet_text = first.review_path.read_text()

    second = draft_nativeness_review("tl", llm=FactoryLLM())
    assert not second.written
    assert first.review_path.read_text() == packet_text
    assert len(mocked_generate) == 1

    _approve(project)
    first_apply = apply_nativeness_review("tl")
    applied_text = project.pack_draft_path.read_text()
    revision = FactoryProject.load("tl").pack_revision
    second_apply = apply_nativeness_review("tl")
    assert first_apply.written and not second_apply.written
    assert project.pack_draft_path.read_text() == applied_text
    assert FactoryProject.load("tl").pack_revision == revision


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (
            lambda reply: reply["judge_factors"][1].update(
                factor_id="not_in_the_catalog"
            ),
            "Unknown nativeness judge factor",
        ),
        (
            lambda reply: reply["judge_factors"].append(
                dict(reply["judge_factors"][1])
            ),
            "Duplicate nativeness judge factor",
        ),
    ],
)
def test_unknown_and_duplicate_ids_fail_via_pack_schema(
    nativeness_workspace, monkeypatch, mutate, expected
):
    reply = _valid_reply()
    mutate(reply)

    def fake_generate(**kwargs):
        assert kwargs["call_name"] == NATIVENESS_AUTHOR_CALL_NAME
        return SimpleNamespace(content=json.dumps(reply, ensure_ascii=False))

    monkeypatch.setattr(factory_llm_module, "generate", fake_generate)
    with pytest.raises(FactoryDraftError, match=expected):
        draft_nativeness_review("tl", llm=FactoryLLM())
    assert not (nativeness_workspace.project_dir / "nativeness_review.yaml").exists()


@pytest.mark.parametrize(
    "target_variety, agent_rule",
    [
        (None, "The agent uses the target variety."),
        ("Any internally consistent variety", "The agent stays consistent."),
        ("Peninsular Spanish (Spain)", "The agent may use any regional dialect."),
    ],
)
def test_regional_consistency_rejects_missing_or_generic_target_guidance(
    nativeness_workspace, monkeypatch, target_variety, agent_rule
):
    factor = _valid_reply()["judge_factors"][0]
    factor["agent_rule"] = agent_rule
    if target_variety is None:
        factor.pop("target_variety")
    else:
        factor["target_variety"] = target_variety
    reply = {"judge_factors": [factor]}

    def fake_generate(**kwargs):
        assert kwargs["call_name"] == NATIVENESS_AUTHOR_CALL_NAME
        return SimpleNamespace(content=json.dumps(reply, ensure_ascii=False))

    monkeypatch.setattr(factory_llm_module, "generate", fake_generate)
    with pytest.raises(FactoryDraftError, match="target_variety|exact region"):
        draft_nativeness_review("tl", llm=FactoryLLM())


def test_regional_consistency_is_not_added_when_model_does_not_select_it(
    nativeness_workspace, monkeypatch
):
    reply = _valid_reply()
    reply["judge_factors"] = [
        factor
        for factor in reply["judge_factors"]
        if factor["factor_id"] != "regional_consistency"
    ]

    def fake_generate(**kwargs):
        assert kwargs["call_name"] == NATIVENESS_AUTHOR_CALL_NAME
        return SimpleNamespace(content=json.dumps(reply, ensure_ascii=False))

    monkeypatch.setattr(factory_llm_module, "generate", fake_generate)
    outcome = draft_nativeness_review("tl", llm=FactoryLLM())
    assert [factor.factor_id for factor in outcome.artifact.judge_factors] == [
        "register_formality",
        "translationese",
    ]


def test_stale_review_and_existing_content_are_never_overwritten(
    nativeness_workspace, mocked_generate
):
    project = nativeness_workspace
    outcome = draft_nativeness_review("tl", llm=FactoryLLM())
    packet_before = outcome.review_path.read_text()
    project.pack_draft_path.write_text(
        project.pack_draft_path.read_text().replace("Toylang", "Changedlang", 1)
    )

    with pytest.raises(FactoryDraftError, match="different pack_draft"):
        draft_nativeness_review("tl", llm=FactoryLLM())
    assert outcome.review_path.read_text() == packet_before
    _approve(project)
    with pytest.raises(FactoryDraftError, match="changed after"):
        apply_nativeness_review("tl")
