# Copyright Sierra
"""Text input-style arm: closed catalog, pack schema, prompt wiring, provenance.

Guards the Item-1 contract of docs/designs/text-mode-multilingual.md:

- the style catalog is CLOSED (packs cannot invent a style, native_script
  cannot be declared);
- the directive is a fixed versioned template + pack examples, rendered into
  the TEXT user simulator's system prompt right below the target-language
  directive;
- the default (native_script / no knob) arm renders byte-identical prompts to
  a run without the feature;
- unsupported styles fail the BUILD loudly (never a silent fallback);
- the arm is recorded in results provenance (Info.text_input_style).
"""

import pytest

from tau2.data_model.simulation import TextRunConfig
from tau2.multilingual.registry import get_language_pack, resolve_text_input_style
from tau2.multilingual.schema import (
    TextInputExample,
    TextInputPackConfig,
    TextInputStyleSpec,
)
from tau2.multilingual.text_input_catalog import (
    TextInputStyle,
    render_input_style_directive,
)
from tau2.registry import registry
from tau2.runner.build import build_text_orchestrator
from tau2.runner.helpers import get_info


def _airline_task(task_set: str, task_id: str):
    return next(t for t in registry.get_tasks_loader(task_set)() if t.id == task_id)


class TestClosedCatalog:
    def test_unknown_style_rejected(self):
        with pytest.raises(Exception):
            TextInputStyleSpec.model_validate(
                {
                    "style": "leetspeak",
                    "examples": [
                        {"standard": "a", "typed": "4"},
                        {"standard": "e", "typed": "3"},
                    ],
                }
            )

    def test_native_script_not_declarable(self):
        with pytest.raises(ValueError, match="implicit default"):
            TextInputStyleSpec(
                style=TextInputStyle.NATIVE_SCRIPT,
                examples=[
                    TextInputExample(standard="a", typed="a"),
                    TextInputExample(standard="b", typed="b"),
                ],
            )

    def test_duplicate_styles_rejected(self):
        spec = TextInputStyleSpec(
            style=TextInputStyle.ROMANIZED,
            examples=[
                TextInputExample(standard="अ", typed="a"),
                TextInputExample(standard="ब", typed="ba"),
            ],
        )
        with pytest.raises(ValueError, match="Duplicate text input style"):
            TextInputPackConfig(styles=[spec, spec])

    def test_examples_minimum(self):
        with pytest.raises(Exception):
            TextInputStyleSpec(
                style=TextInputStyle.ROMANIZED,
                examples=[TextInputExample(standard="अ", typed="a")],
            )


class TestPackApplicability:
    """The documented per-language applicability decisions hold in pack data."""

    def test_declared_styles_match_design(self):
        expected = {
            "hi": {TextInputStyle.ROMANIZED, TextInputStyle.CODE_MIXED},
            "es": {TextInputStyle.DIACRITIC_FREE},
            "pt": {TextInputStyle.DIACRITIC_FREE},
        }
        for language, styles in expected.items():
            pack = get_language_pack(language)
            assert pack.text_input is not None, language
            assert {s.style for s in pack.text_input.styles} == styles

    def test_native_script_only_languages(self):
        for language in ("ko", "zh", "en"):
            pack = get_language_pack(language)
            assert pack.text_input is None, (
                f"{language} must stay native_script-only per the design doc"
            )


class TestDirectiveRendering:
    def test_native_script_renders_nothing(self):
        assert (
            render_input_style_directive(TextInputStyle.NATIVE_SCRIPT, "Hindi", [])
            is None
        )
        assert resolve_text_input_style("hi", "native_script") is None

    def test_romanized_directive_carries_pack_examples(self):
        directive = resolve_text_input_style("hi", "romanized")
        assert "ROMANIZED Hindi" in directive
        assert "mera naam Rahul hai" in directive
        assert "overrides the native-orthography instruction" in directive

    def test_diacritic_free_directive(self):
        directive = resolve_text_input_style("es", "diacritic_free")
        assert "NO ACCENTS" in directive
        assert "Cuando sale mi vuelo a Bogota?" in directive

    def test_unsupported_style_raises(self):
        with pytest.raises(ValueError, match="does not support"):
            resolve_text_input_style("ko", "romanized")

    def test_no_language_raises(self):
        with pytest.raises(ValueError, match="requires a language-pack run"):
            resolve_text_input_style(None, "romanized")


class TestOrchestratorWiring:
    def test_hi_romanized_reaches_user_prompt(self):
        config = TextRunConfig(
            domain="airline",
            agent="llm_agent",
            user_persona_id="hi",
            text_input_style="romanized",
        )
        orch = build_text_orchestrator(
            config, _airline_task("airline_hi", "7_hi"), seed=42
        )
        prompt = orch.user.system_prompt
        assert "## HOW YOU TYPE (ROMANIZED Hindi)" in prompt
        assert "mera naam Rahul hai" in prompt
        # The block rides directly below the target-language directive.
        assert prompt.index("LANGUAGE OF THE CALL") < prompt.index("HOW YOU TYPE")

    def test_es_diacritic_free_reaches_user_prompt(self):
        config = TextRunConfig(
            domain="airline",
            agent="llm_agent",
            user_persona_id="es",
            text_input_style="diacritic_free",
        )
        orch = build_text_orchestrator(
            config, _airline_task("airline_es", "7_es"), seed=42
        )
        assert "## HOW YOU TYPE (NO ACCENTS)" in orch.user.system_prompt

    def test_default_arm_is_untouched(self):
        base = TextRunConfig(domain="airline", agent="llm_agent", user_persona_id="hi")
        pinned = TextRunConfig(
            domain="airline",
            agent="llm_agent",
            user_persona_id="hi",
            text_input_style="native_script",
        )
        task = _airline_task("airline_hi", "7_hi")
        assert (
            build_text_orchestrator(base, task, seed=42).user.system_prompt
            == build_text_orchestrator(pinned, task, seed=42).user.system_prompt
        )

    def test_unsupported_style_fails_build(self):
        config = TextRunConfig(
            domain="airline",
            agent="llm_agent",
            user_persona_id="ko",
            text_input_style="romanized",
        )
        with pytest.raises(ValueError, match="does not support"):
            build_text_orchestrator(
                config, _airline_task("airline_ko", "7_ko"), seed=42
            )

    def test_style_without_language_fails_build(self):
        config = TextRunConfig(
            domain="airline", agent="llm_agent", text_input_style="romanized"
        )
        with pytest.raises(ValueError, match="requires a language-pack run"):
            build_text_orchestrator(config, _airline_task("airline", "7"), seed=42)


class TestProvenance:
    def test_config_rejects_unknown_style(self):
        with pytest.raises(ValueError):
            TextRunConfig(domain="airline", text_input_style="leetspeak")

    def test_info_records_style(self):
        config = TextRunConfig(
            domain="airline",
            agent="llm_agent",
            user_persona_id="hi",
            text_input_style="romanized",
        )
        assert get_info(config).text_input_style == "romanized"

    def test_info_none_for_default(self):
        config = TextRunConfig(domain="airline", agent="llm_agent")
        assert get_info(config).text_input_style is None
