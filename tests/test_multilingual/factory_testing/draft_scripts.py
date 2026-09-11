# Copyright Sierra
"""Scripted FakeFactoryLLM responses for one clean toy pack draft.

The toy Toylang pack supplies the creative outputs for the factory's four
drafting calls — Call A (personas JSON), Call B (voice guidelines markdown),
Call C (text guidelines markdown), Call D (localization JSON) — so drafting
suites and the offline e2e pipeline can run ``run_draft`` end to end without
an LLM. ``draft_scripts()`` assembles the full script dict (plus optional
repair-call scripts).
"""

import yaml

from tau2.config import DEFAULT_MULTILINGUAL_DOMAIN
from tau2.multilingual.factory import pack_assembly
from tau2.multilingual.factory.author_form import AuthorForm, extract_fenced_blocks
from tau2.multilingual.factory.drafting import (
    DRAFT_ARTIFACTS_CALL_NAME,
    DRAFT_PERSONAS_CALL_NAME,
    DRAFT_TEXT_ARTIFACTS_CALL_NAME,
    REPAIR_ARTIFACTS_CALL_NAME,
    REPAIR_PERSONAS_CALL_NAME,
    REPAIR_TEXT_ARTIFACTS_CALL_NAME,
)
from tau2.multilingual.factory.localization_drafting import (
    DRAFT_LOCALIZATION_CALL_NAME,
)
from tau2.multilingual.localization_catalog import (
    GUIDELINE_EXAMPLE_CATALOG,
    SYMBOL_CATALOG,
    get_domain_term_catalog,
)
from test_multilingual.factory_testing.toy_language import (
    TOY_GUIDELINES_FILENAME,
    TOY_PACK_DIR,
    toy_pack_yaml_dict,
)

TOY_FORM = """\
# Toylang author form (test fixture)

```yaml
language: tl
display_name: Toylang
script: latn
locale_notes: Synthetic test language; Latin script.
personas:
  - name: Tessa
    sketch: Fast casual 20s speaker with heavy English insertions.
  - name: Orin
    sketch: Slow formal 50s speaker, almost no English.
preferences: Keep everything tiny and deterministic.
# Opt into locale beds so the toy pipeline exercises the acoustic scaffolding
# (the shipped default is off: shared benchmark environments, no presets).
locale_beds: true
```
"""

TOY_FORM_OBJECT = AuthorForm.model_validate(
    yaml.safe_load(extract_fenced_blocks(TOY_FORM)["yaml"][0])
)
# The deterministic acoustic preset ids the assembly scaffolds for 'tl'. The
# scripted personas wire to the first (outdoor) one.
TOY_PRESET_IDS = pack_assembly.acoustic_preset_ids(TOY_FORM_OBJECT)


def guidelines_text() -> str:
    return (TOY_PACK_DIR / TOY_GUIDELINES_FILENAME).read_text()


def toy_personas_for_call_a(pack_data: dict | None = None) -> list[dict]:
    """The persona dicts as Call A would return them (acoustic_preset_id wired
    to a deterministic scaffolded preset id, not the toy pack's own preset)."""
    data = pack_data if pack_data is not None else toy_pack_yaml_dict()
    personas = []
    for persona in data["personas"].values():
        persona = dict(persona)
        persona["acoustic_preset_id"] = TOY_PRESET_IDS[0]
        personas.append(persona)
    return personas


def personas_response(pack_data: dict | None = None) -> dict:
    """Call A's JSON: personas + translation_guidance + agent_greeting."""
    data = pack_data if pack_data is not None else toy_pack_yaml_dict()
    return {
        "personas": toy_personas_for_call_a(data),
        "translation_guidance": data.get("translation_guidance", "Toy guidance."),
        "agent_greeting": data.get("agent_greeting", "Halo!"),
    }


def artifacts_response(pack_data: dict | None = None) -> str:
    """Call B's response: a single markdown block (the guidelines)."""
    return f"```markdown\n{guidelines_text()}```\n"


# Toy TEXT (Call C) guidelines: distinct from the voice guidelines (different
# wording, no spoken-form conventions) so the anti-copy guardrail passes, while
# still carrying the persona slot + a control token.
TOY_TEXT_GUIDELINES = """\
# Text Chat Simulation Guidelines (Toylang)

Tik naturel, zo een echt klant in een chat venster. Volg je task instructies en
skryf Toylang. Skryf getalle in native syfers en tik e-posadresse letterlik.

Use '###STOP###', '###TRANSFER###', '###OUT-OF-SCOPE###' in ASCII when ending
the chat.

<PERSONA_GUIDELINES>
"""


def text_artifacts_response() -> str:
    """Call C's response: a single markdown block (the TEXT guidelines)."""
    return f"```markdown\n{TOY_TEXT_GUIDELINES}```\n"


def localization_response() -> dict:
    """Call D's JSON: a full-coverage toy localization block.

    Built FROM the closed catalogs (not a hand-maintained list) so the fixture
    cannot drift from the coverage contract draft_localization enforces.
    """
    from tau2.multilingual.schema import SpokenValueKind

    return {
        "symbol_readouts": [
            {"symbol": symbol, "spoken": [f"toy-{d.english_name}"]}
            for symbol, d in SYMBOL_CATALOG.items()
        ],
        "date_examples": [
            {"written": "May 17", "spoken": "toy zeventien mei"},
            {"written": "May 21", "spoken": "toy eenentwintig mei"},
        ],
        "number_examples": [
            {"written": "21", "spoken": "toy eenentwintig"},
            {"written": "253 dollars", "spoken": "toy tweehonderd dollar"},
        ],
        "domain_glossary": [
            {"term_id": term_id, "native": f"toy-{d.english}"}
            for term_id, d in get_domain_term_catalog(
                DEFAULT_MULTILINGUAL_DOMAIN
            ).items()
        ],
        "phone_number_readout": {
            "rule": "Toy phones are read in pairs.",
            "examples": [{"written": "06 12 34", "spoken": "toy nul zes twaalf"}],
        },
        "amount_readout": {
            "rule": "Toy amounts put the currency last; the decimal comma "
            "is spoken 'komma'.",
            "examples": [
                {"written": "12,50", "spoken": "toy twaalf komma vijftig"},
                {"written": "253 dollars", "spoken": "toy tweehonderd dollar"},
            ],
        },
        "spelling_alphabet": {
            "rule": "Toy spelling anchors on toy town names.",
            "avoid": "the NATO/anglo 'B for boy' alphabet",
            "anchor_examples": [
                {"written": "A", "spoken": "toy A van Alkmaar"},
                {"written": "B", "spoken": "toy B van Breda"},
                {"written": "C", "spoken": "toy C van Coevorden"},
            ],
        },
        "spoken_value_examples": [
            {
                "kind": kind.value,
                "written": f"toy-{kind.value}-written",
                "spoken": f"toy {kind.value} gesproken",
            }
            for kind in SpokenValueKind
        ],
        "guideline_examples": [
            {
                "kind": kind_id,
                "utterances": [f"toy {kind_id} {i}" for i in range(kind_def.min_items)],
            }
            for kind_id, kind_def in GUIDELINE_EXAMPLE_CATALOG.items()
        ],
    }


def draft_scripts(
    pack_data: dict | None = None,
    *,
    repair_personas: list | None = None,
    repair_artifacts: list | None = None,
    repair_text_artifacts: list | None = None,
) -> dict:
    """A FakeFactoryLLM script dict for one clean four-call draft (+ repairs)."""
    scripts: dict = {
        DRAFT_PERSONAS_CALL_NAME: [personas_response(pack_data)],
        DRAFT_ARTIFACTS_CALL_NAME: [artifacts_response(pack_data)],
        DRAFT_TEXT_ARTIFACTS_CALL_NAME: [text_artifacts_response()],
        DRAFT_LOCALIZATION_CALL_NAME: [localization_response()],
    }
    if repair_personas is not None:
        scripts[REPAIR_PERSONAS_CALL_NAME] = repair_personas
    if repair_artifacts is not None:
        scripts[REPAIR_ARTIFACTS_CALL_NAME] = repair_artifacts
    if repair_text_artifacts is not None:
        scripts[REPAIR_TEXT_ARTIFACTS_CALL_NAME] = repair_text_artifacts
    return scripts
