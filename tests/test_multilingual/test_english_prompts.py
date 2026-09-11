# Copyright Sierra
"""English prompts: the one prompt mode multilingual runs have.

Every multilingual run reads the ENGLISH global guidelines and ENGLISH task
instructions plus a fixed, versioned speak-the-target-language directive.
(The native arm — localized guidelines + translated task instructions — is
retired, and its module was deleted on 2026-08-03.) These
tests guard:

- guidelines + directive selection, text and voice, with pack pragmatics
  injected in both (they describe how to SPEAK);
- the English task-instruction variant: English source prose, entity
  localization preserved (identity-variant caller rename re-applied), and the
  orchestrator/evaluator task untouched;
- the text-orchestrator end-to-end wiring.
"""

import re

import pytest

from tau2.data_model.simulation import TextRunConfig
from tau2.data_model.tasks import Task
from tau2.multilingual.domain_profiles import CallerIdentityKind
from tau2.multilingual.english_prompts import (
    TARGET_LANGUAGE_DIRECTIVE_TEMPLATE_V1,
    TARGET_LANGUAGE_DIRECTIVE_TEMPLATE_V2,
    TARGET_LANGUAGE_DIRECTIVE_TEMPLATE_V3,
    TARGET_LANGUAGE_DIRECTIVE_VERSION,
    TARGET_LANGUAGE_REMINDER_TEMPLATE_V1,
    TARGET_LANGUAGE_REMINDER_VERSION,
    _english_scenario,
    english_user_task_variant,
    render_target_language_directive,
    render_target_language_reminder,
)
from tau2.multilingual.factory.caller_diversity import (
    NAME_AUTH_LINE_CLAUSE_TEMPLATE,
)
from tau2.registry import registry
from tau2.runner.build import build_text_orchestrator
from tau2.user.user_simulator import UserSimulator, get_target_language_directive
from test_multilingual.conftest import first_persona, make_voice_sim

DEVANAGARI = re.compile(r"[ऀ-ॿ]")

# A stable, distinctive phrase of the fixed directive block (bump alongside
# TARGET_LANGUAGE_DIRECTIVE_VERSION if the template is recalibrated).
DIRECTIVE_MARKER = "## LANGUAGE OF THE CALL"


def _airline_hi_task(task_id: str = "7_hi") -> Task:
    return next(t for t in registry.get_tasks_loader("airline_hi")() if t.id == task_id)


def _airline_source_task(task_id: str) -> Task:
    return next(t for t in registry.get_tasks_loader("airline")() if t.id == task_id)


class TestDirective:
    def test_rendered_from_pack_display_name(self):
        directive = render_target_language_directive("hi", "IN-MH")
        assert DIRECTIVE_MARKER in directive
        assert "Hindi" in directive
        assert "###STOP###" in directive  # control tokens stay verbatim

    def test_unregistered_language_fails_loud(self):
        """An unregistered code can name neither variety nor origin — the
        render dies instead of shipping an under-specified directive."""
        with pytest.raises(ValueError, match="no registered"):
            render_target_language_directive("qq", "IN-MH")

    def test_missing_locale_fails_loud(self):
        with pytest.raises(ValueError, match="carries no locale"):
            render_target_language_directive("hi", None)

    def test_unmapped_locale_fails_loud(self):
        with pytest.raises(ValueError, match="no display name"):
            render_target_language_directive("hi", "IN-UNMAPPED")

    def test_helper_gates_on_language(self):
        assert DIRECTIVE_MARKER in get_target_language_directive(first_persona("hi"))
        # English personas (no language) never get a directive.
        from tau2.data_model.persona import PersonaConfig

        assert get_target_language_directive(PersonaConfig()) is None
        # Nor does the ENGLISH language pack's persona, which DOES carry a
        # language — it just speaks the one the prompt is written in.
        assert get_target_language_directive(first_persona("en")) is None

    def test_template_placeholders_are_fixed(self):
        # The templates are fixed prompt material: v1/v2 parameterize the
        # language name only; v3 adds the speaker's origin and the language's
        # pinned variety (tau2.multilingual.varieties). Per-language
        # conventions still live in the pack's localization data, never here.
        for template in (
            TARGET_LANGUAGE_DIRECTIVE_TEMPLATE_V1,
            TARGET_LANGUAGE_DIRECTIVE_TEMPLATE_V2,
        ):
            placeholders = set(re.findall(r"{(\w+)}", template))
            assert placeholders == {"language_name"}
        placeholders = set(
            re.findall(r"{(\w+)}", TARGET_LANGUAGE_DIRECTIVE_TEMPLATE_V3)
        )
        assert placeholders == {"language_name", "origin", "variety_name"}

    def test_directive_lands_at_the_top_of_the_prompt(self):
        """LAYOUT: the directive renders right after the title/role-framing
        intro — before the first behavioral section — in both simulators.
        The language mandate must not trail thousands of tokens of English
        instruction."""
        text_prompt = UserSimulator(
            llm="dummy", instructions="scenario", persona_config=first_persona("es")
        ).system_prompt
        directive_at = text_prompt.index(DIRECTIVE_MARKER)
        first_section_at = text_prompt.index("\n## ")
        assert directive_at <= first_section_at + 1
        assert directive_at < len(text_prompt) * 0.2

    def test_v3_is_current_and_names_origin_and_variety(self):
        """v3 implements the owner ruling that naming the language names the
        location/variety too: the opening sentence states the speaker's
        origin (the resolved persona's locale, human-readable) and the
        language's pinned variety, on top of v2's language-generic rules
        (native orthography, code-switch spelling, persona precedence)."""
        assert TARGET_LANGUAGE_DIRECTIVE_VERSION == "v3"
        directive = render_target_language_directive("es", "ES-MD")
        assert "a native Spanish speaker from Madrid, Spain" in directive
        assert "your Spanish is Peninsular Spanish (Spain)" in directive
        assert "ES-MD" not in directive  # raw locale codes never render
        # v2's generic rules survive verbatim.
        assert "native orthography" in directive
        assert "¿ ¡" in directive  # orthography incl. language-specific marks
        assert "English spelling inside the Spanish sentence" in directive
        assert "never transliterate them phonetically" in directive
        assert "always follow the persona" in directive
        assert "the persona says WHO you are" in directive
        # v1/v2 stay retrievable for provenance, but are no longer rendered.
        assert "native orthography" not in TARGET_LANGUAGE_DIRECTIVE_TEMPLATE_V1
        assert "{origin}" not in TARGET_LANGUAGE_DIRECTIVE_TEMPLATE_V2

    def test_directive_origin_follows_the_resolved_persona(self):
        """The origin is the RESOLVED persona's, not a pack-wide constant:
        the two ko personas render different origins under one variety."""
        seoul = render_target_language_directive("ko", "KR-11")
        busan = render_target_language_directive("ko", "KR-26")
        assert "from Seoul, South Korea" in seoul
        assert "from Busan, South Korea" in busan
        for directive in (seoul, busan):
            assert "standard Seoul Korean" in directive


class TestEndReminder:
    """The fixed, versioned end-of-prompt reminder: a terse recency anchor
    (language mandate + special-token contract) after the scenario block."""

    def test_v1_is_current_and_two_lines(self):
        assert TARGET_LANGUAGE_REMINDER_VERSION == "v1"
        body = TARGET_LANGUAGE_REMINDER_TEMPLATE_V1.split("\n", 1)[1]
        assert len(body.strip().splitlines()) == 2  # terse by contract

    def test_template_parameterizes_language_only(self):
        placeholders = set(re.findall(r"{(\w+)}", TARGET_LANGUAGE_REMINDER_TEMPLATE_V1))
        assert placeholders == {"language_name"}

    def test_rendered_from_pack_display_name(self):
        reminder = render_target_language_reminder("hi")
        assert "## FINAL REMINDER" in reminder
        assert "Hindi" in reminder
        for token in ("###STOP###", "###TRANSFER###", "###OUT-OF-SCOPE###"):
            assert token in reminder
        # Unregistered language falls back to the raw code.
        assert "qq" in render_target_language_reminder("qq")

    def test_reminder_anchors_the_prompt_end_after_the_scenario(self):
        sim = UserSimulator(
            llm="dummy",
            instructions="scenario-sentinel",
            persona_config=first_persona("es"),
        )
        prompt = sim.system_prompt
        reminder = render_target_language_reminder("es")
        assert prompt.rstrip().endswith(reminder)
        assert prompt.index("scenario-sentinel") < prompt.index("## FINAL REMINDER")

    def test_english_persona_gets_no_reminder(self):
        sim = UserSimulator(llm="dummy", instructions="scenario")
        assert "## FINAL REMINDER" not in sim.system_prompt

    def test_english_pack_persona_gets_no_reminder(self):
        """The en pack persona carries language='en', so the reminder used to
        render as 'in English — never in English'. It is suppressed."""
        sim = UserSimulator(
            llm="dummy",
            instructions="scenario",
            persona_config=first_persona("en"),
        )
        assert "## FINAL REMINDER" not in sim.system_prompt


class TestTextPromptAssembly:
    def test_english_guidelines_directive_and_pragmatics(self):
        """English guidelines + directive + pragmatics, never the pack's
        localized guidelines (es ships localized TEXT guidelines, so it is the
        sharp test case)."""
        sim = UserSimulator(
            llm="dummy", instructions="scenario", persona_config=first_persona("es")
        )
        prompt = sim.system_prompt
        assert DIRECTIVE_MARKER in prompt
        assert "Spanish" in prompt
        assert "Chat de Texto" not in prompt  # es localized text guidelines
        # Pack pragmatics (how to SPEAK) are injected too.
        assert "## PERSONA AND LANGUAGE" in prompt

    def test_english_persona_unaffected(self):
        sim = UserSimulator(llm="dummy", instructions="scenario")
        assert DIRECTIVE_MARKER not in sim.system_prompt


class TestVoicePromptAssembly:
    def test_english_guidelines_directive_and_pragmatics(self):
        sim = make_voice_sim(first_persona("hi"))
        guidelines = sim.global_simulation_guidelines
        assert "(Hindi)" not in guidelines  # the hi localized-guidelines marker
        prompt = sim.system_prompt
        assert DIRECTIVE_MARKER in prompt
        assert "Hindi" in prompt
        assert "## PERSONA AND LANGUAGE" in prompt  # pragmatics still injected

    def test_english_persona_unaffected(self):
        sim = make_voice_sim(None)
        assert DIRECTIVE_MARKER not in sim.system_prompt


class TestEnglishTaskVariant:
    def test_localized_task_gets_english_source_scenario(self):
        localized = _airline_hi_task()
        variant = english_user_task_variant(
            localized, domain="airline", task_set_name="airline_hi"
        )
        source = _airline_source_task("7")
        assert str(variant.user_scenario) == str(source.user_scenario)
        assert not DEVANAGARI.search(str(variant.user_scenario))
        # Everything the environment/evaluator reads is the LOCALIZED task's.
        assert variant.id == localized.id
        assert variant.evaluation_criteria == localized.evaluation_criteria
        assert variant.initial_state == localized.initial_state

    def test_entities_survive(self):
        """Plain localized sets keep entities byte-identical to the source
        (the localization invariant), so the English variant must carry every
        concrete value the localized instructions carry."""
        from tau2.multilingual.invariants import extract_values, instruction_text

        localized = _airline_hi_task()
        variant = english_user_task_variant(
            localized, domain="airline", task_set_name="airline_hi"
        )
        variant_text = instruction_text(variant.model_dump())
        for value in extract_values(instruction_text(localized.model_dump())):
            assert value in variant_text

    def test_base_and_unknown_task_sets_pass_through(self):
        task = _airline_source_task("7")
        for task_set_name in (None, "airline", "some_custom_set"):
            assert (
                english_user_task_variant(
                    task, domain="airline", task_set_name=task_set_name
                )
                is task
            )

    def test_localized_task_swapped_without_task_set_name(self):
        """A run handed a localized Task directly (no task_set_name on the
        config) must still get the English swap — the set is inferred from the
        id's ``_<suffix>`` tail. Silent pass-through would keep translated
        instructions under English guidelines and corrupt the arm."""
        localized = _airline_hi_task()
        variant = english_user_task_variant(
            localized, domain="airline", task_set_name=None
        )
        assert not DEVANAGARI.search(str(variant.user_scenario))
        assert variant.id == localized.id

    def test_suffix_inference_prefers_identity_variant(self):
        from tau2.multilingual.english_prompts import _infer_localized_task_set
        from tau2.multilingual.task_sets import discover_localized_task_files

        assert _infer_localized_task_set("7_hi", "airline") == "airline_hi"
        identity_sets = [
            name
            for name in discover_localized_task_files()
            if name.startswith("airline_") and name.endswith("_identity")
        ]
        assert identity_sets, "expected at least one airline identity task set"
        for name in identity_sets:
            suffix = name.removeprefix("airline_")
            assert _infer_localized_task_set(f"7_{suffix}", "airline") == name
        assert _infer_localized_task_set("7", "airline") is None

    def test_task_language_strips_identity_suffix(self):
        """``task_language`` returns the LANGUAGE, not the task-set name, so
        plain and identity variants of a language collapse to one code."""
        from tau2.multilingual.english_prompts import task_language

        assert task_language("7_hi", "airline") == "hi"
        assert task_language("7_hi_identity", "airline") == "hi"
        assert (
            task_language("[service_issue]unseat_sim_card[PERSONA:Easy]_pt", "telecom")
            == "pt"
        )
        # Base English ids belong to no localized set.
        assert task_language("7", "airline") is None

    def test_missing_source_task_fails_loud(self):
        rogue = _airline_hi_task().model_copy(update={"id": "does_not_exist_hi"})
        with pytest.raises(ValueError, match="no English source task"):
            english_user_task_variant(
                rogue, domain="airline", task_set_name="airline_hi"
            )


class TestIdentityRenamePreserved:
    """English instruction text must NOT undo entity localization: the
    identity-variant caller rename (locale name/user_id/email, declared via
    ``initialization_data``) is re-applied to the English source prose."""

    SOURCE = Task.model_validate(
        {
            "id": "1",
            "user_scenario": {
                "instructions": {
                    "domain": "airline",
                    "reason_for_call": "You want to cancel reservation ABC123.",
                    "known_info": "You are Emma Kim. Your user id is "
                    "emma_kim_9957 and your email is emma.kim@example.com.",
                    "task_instructions": "Give your user id emma_kim_9957 only "
                    "when asked. Reservation ABC123 must not be changed.",
                }
            },
        }
    )
    LOCALIZED = Task.model_validate(
        {
            "id": "1_xx_identity",
            "user_scenario": {
                "instructions": {
                    "domain": "airline",
                    "reason_for_call": "LOCALIZED cancel ABC123.",
                    "known_info": "LOCALIZED priya_sharma_1234 "
                    "priya.sharma@example.com Priya Sharma.",
                    "task_instructions": "LOCALIZED priya_sharma_1234 ABC123.",
                }
            },
            "initial_state": {
                "initialization_data": {
                    "agent_data": {
                        "users": {
                            "priya_sharma_1234": {
                                "name": {
                                    "first_name": "Priya",
                                    "last_name": "Sharma",
                                },
                                "email": "priya.sharma@example.com",
                            }
                        }
                    }
                }
            },
        }
    )
    DB = {
        "users": {
            "emma_kim_9957": {
                "name": {"first_name": "Emma", "last_name": "Kim"},
                "email": "emma.kim@example.com",
            }
        }
    }

    def test_rename_applied_to_english_source(self):
        scenario = _english_scenario(
            self.LOCALIZED, self.SOURCE, self.DB, CallerIdentityKind.USER_ID_HANDLE
        )
        text = str(scenario)
        assert "Priya Sharma" in text
        assert "priya_sharma_1234" in text
        assert "priya.sharma@example.com" in text
        assert "emma_kim_9957" not in text
        assert "emma.kim@example.com" not in text
        # Non-identity entities survive verbatim; prose is the English source's.
        assert "ABC123" in text
        assert "You want to cancel" in text
        assert "LOCALIZED" not in text

    def test_plain_localization_returns_source_scenario(self):
        plain = self.LOCALIZED.model_copy(update={"initial_state": None})
        scenario = _english_scenario(
            plain, self.SOURCE, self.DB, CallerIdentityKind.USER_ID_HANDLE
        )
        assert str(scenario) == str(self.SOURCE.user_scenario)


class TestTelecomIdentityRenamePreserved:
    """The STRUCTURED_NAME_PHONE rename (a customers-list patch; the caller
    keeps their customer id and phone number) is re-applied to the English
    source prose exactly like the airline user-id rename."""

    SOURCE = Task.model_validate(
        {
            "id": "[toy]a",
            "user_scenario": {
                "instructions": {
                    "domain": "telecom",
                    "reason_for_call": "Your mobile data is broken.",
                    "known_info": "You are John Smith with phone number 555-123-2002.",
                    "task_instructions": "Follow the agent's guidance.",
                }
            },
            "initial_state": {
                "initialization_actions": [
                    {
                        "env_type": "user",
                        "func_name": "set_user_info",
                        "arguments": {
                            "name": "John Smith",
                            "phone_number": "555-123-2002",
                        },
                    }
                ]
            },
        }
    )
    LOCALIZED = Task.model_validate(
        {
            "id": "[toy]a_xx_identity",
            "user_scenario": {
                "instructions": {
                    "domain": "telecom",
                    "reason_for_call": "LOCALIZED data broken.",
                    "known_info": "LOCALIZED Jorge Gil 555-123-2002.",
                    "task_instructions": "LOCALIZED guidance.",
                }
            },
            "initial_state": {
                "initialization_data": {
                    "agent_data": {
                        "customers": [
                            {
                                "customer_id": "C1001",
                                "full_name": "Jorge Gil",
                                "email": "jorge.gil@example.com",
                                "phone_number": "555-123-2002",
                            }
                        ]
                    }
                },
                "initialization_actions": [
                    {
                        "env_type": "user",
                        "func_name": "set_user_info",
                        "arguments": {
                            "name": "Jorge Gil",
                            "phone_number": "555-123-2002",
                        },
                    }
                ],
            },
        }
    )
    DB = {
        "customers": [
            {
                "customer_id": "C1001",
                "full_name": "John Smith",
                "email": "john.smith@example.com",
                "phone_number": "555-123-2002",
            }
        ]
    }

    def test_rename_applied_to_english_source(self):
        scenario = _english_scenario(
            self.LOCALIZED,
            self.SOURCE,
            self.DB,
            CallerIdentityKind.STRUCTURED_NAME_PHONE,
        )
        text = str(scenario)
        assert "Jorge Gil" in text
        assert "John Smith" not in text
        # Phone number canonical; prose is the English source's.
        assert "555-123-2002" in text
        assert "Your mobile data is broken" in text
        assert "LOCALIZED" not in text


class TestTelecomDiversifiedComposeRename:
    """The compose-bug regression: seed-diversified SOURCE tasks carry their
    own customers patch (a pool caller over the canonical C1001 record), so
    the English-prompt re-rename must baseline on the source's EFFECTIVE
    caller. Resolving from db.toml would derive a John Smith→locale pair — a
    no-op on prose that says Marcus Delgado — and silently ship a
    mixed-identity scenario at RUN time."""

    SOURCE = Task.model_validate(
        {
            "id": "[toy]a",
            "user_scenario": {
                "instructions": {
                    "domain": "telecom",
                    "reason_for_call": "Your mobile data is broken.",
                    "known_info": "You are Marcus Delgado with phone number "
                    "555-123-2002.",
                    "task_instructions": "Follow the agent's guidance.",
                }
            },
            "initial_state": {
                "initialization_data": {
                    "agent_data": {
                        "customers": [
                            {
                                "customer_id": "C1001",
                                "full_name": "Marcus Delgado",
                                "email": "marcus.delgado@example.com",
                                "date_of_birth": "1980-07-09",
                                "phone_number": "555-123-2002",
                            }
                        ]
                    }
                },
                "initialization_actions": [
                    {
                        "env_type": "user",
                        "func_name": "set_user_info",
                        "arguments": {
                            "name": "Marcus Delgado",
                            "phone_number": "555-123-2002",
                        },
                    }
                ],
            },
        }
    )
    LOCALIZED = Task.model_validate(
        {
            "id": "[toy]a_xx_identity",
            "user_scenario": {
                "instructions": {
                    "domain": "telecom",
                    "reason_for_call": "LOCALIZED data broken.",
                    "known_info": "LOCALIZED Jorge Gil 555-123-2002.",
                    "task_instructions": "LOCALIZED guidance.",
                }
            },
            "initial_state": {
                "initialization_data": {
                    "agent_data": {
                        "customers": [
                            {
                                "customer_id": "C1001",
                                "full_name": "Jorge Gil",
                                "email": "jorge.gil@example.com",
                                "date_of_birth": "1980-07-09",
                                "phone_number": "555-123-2002",
                            }
                        ]
                    }
                },
                "initialization_actions": [
                    {
                        "env_type": "user",
                        "func_name": "set_user_info",
                        "arguments": {
                            "name": "Jorge Gil",
                            "phone_number": "555-123-2002",
                        },
                    }
                ],
            },
        }
    )
    DB = {
        "customers": [
            {
                "customer_id": "C1001",
                "full_name": "John Smith",
                "email": "john.smith@example.com",
                "date_of_birth": "1985-06-15",
                "phone_number": "555-123-2002",
            }
        ]
    }

    def test_rename_baselines_on_effective_source_caller(self):
        scenario = _english_scenario(
            self.LOCALIZED,
            self.SOURCE,
            self.DB,
            CallerIdentityKind.STRUCTURED_NAME_PHONE,
        )
        text = str(scenario)
        assert "Jorge Gil" in text
        assert "Marcus Delgado" not in text
        assert "John Smith" not in text
        assert "Your mobile data is broken" in text
        assert "LOCALIZED" not in text

    def test_shipped_es_identity_tasks_present_locale_names_only(self):
        """End-to-end over the committed artifacts: english_user_task_variant
        on es _identity telecom tasks presents the LOCALE caller name and no
        English pool/canonical name."""
        import json as json_mod

        from tau2.multilingual.factory.entity_localization import (
            identity_task_set_path,
        )
        from tau2.multilingual.invariants import caller_set_user_info
        from tau2.multilingual.localize_lib import data_dir

        path = identity_task_set_path("es", "telecom")
        if not path.exists():
            pytest.skip("es telecom identity set not generated")
        seed_path = (
            data_dir() / "tau2" / "domains" / "telecom" / "tasks_multilingual.json"
        )
        english_names = {
            task["id"]: caller_set_user_info(task)["name"]
            for task in json_mod.loads(seed_path.read_text())
        }
        for raw in json_mod.loads(path.read_text())[:6]:
            task = Task.model_validate(raw)
            out = english_user_task_variant(
                task, domain="telecom", task_set_name="telecom_es_identity"
            )
            text = str(out.user_scenario)
            locale_name = caller_set_user_info(raw)["name"]
            source_id = raw["id"].removesuffix("_es_identity")
            assert locale_name in text
            assert english_names[source_id] not in text
            assert "John Smith" not in text


# The name-auth marker, derived from the generator's own template rather than
# retyped: name-auth tasks are exactly those carrying the line-identification
# clause in task_instructions.
NAME_AUTH_CLAUSE_HEAD = NAME_AUTH_LINE_CLAUSE_TEMPLATE.split("{phone_number}")[0]


class TestProfiledSourceResolution:
    """Regression for the wrong-source bug (Jul-21 telecom runs): the English
    swap must read the domain profile's multilingual SOURCE file (telecom: the
    diversified seed split ``tasks_multilingual.json``), never the registry
    default pool — the pool shares task ids with the seed split, so the old
    registry-loader lookup silently resolved every seed scenario to its pool
    ancestor: phone-auth known_info, the name-auth line clause dropped, and
    the user sim reciting the phone number as its authentication credential."""

    def test_telecom_source_is_the_seed_split(self):
        import json as json_mod

        from tau2.multilingual.english_prompts import _load_source_tasks
        from tau2.multilingual.localize_lib import data_dir

        manifest = json_mod.loads(
            (
                data_dir()
                / "tau2"
                / "domains"
                / "telecom"
                / "tasks_multilingual.manifest.json"
            ).read_text()
        )
        tasks = _load_source_tasks("telecom")
        assert [t.id for t in tasks] == manifest["task_ids"]
        # The auth split survives: exactly the manifest's name-auth tasks
        # carry the line-identification clause (the pool ancestor of every
        # one of them has neither that clause nor the DOB known_info).
        name_auth_ids = set(manifest["name_auth_task_ids"])
        assert name_auth_ids
        for task in tasks:
            instructions = task.user_scenario.instructions
            has_clause = NAME_AUTH_CLAUSE_HEAD in (instructions.task_instructions or "")
            assert has_clause == (task.id in name_auth_ids)
            assert instructions.unknown_info is None

    def test_unprofiled_domain_falls_back_to_registry(self):
        from tau2.multilingual.english_prompts import _load_source_tasks

        assert [t.id for t in _load_source_tasks("mock")] == [
            t.id for t in registry.get_tasks_loader("mock")()
        ]

    def test_hi_identity_name_auth_scenario_survives_end_to_end(self):
        """The shipped-artifact regression: a hi _identity name-auth task run
        through the English swap keeps its name+DOB known_info and its
        line-identification clause, and never gains the phone number in
        known_info — the number stays behavioral (which line?), never
        identity material (who are you?)."""
        import json as json_mod

        from tau2.multilingual.factory.entity_localization import (
            identity_task_set_path,
        )
        from tau2.multilingual.invariants import phone_digits

        path = identity_task_set_path("hi", "telecom")
        if not path.exists():
            pytest.skip("hi telecom identity set not generated")
        name_auth = [
            Task.model_validate(raw)
            for raw in json_mod.loads(path.read_text())
            if NAME_AUTH_CLAUSE_HEAD
            in (raw["user_scenario"]["instructions"].get("task_instructions") or "")
        ]
        assert name_auth, "expected name-auth tasks in the hi identity set"
        for task in name_auth[:6]:
            out = english_user_task_variant(
                task, domain="telecom", task_set_name="telecom_hi_identity"
            )
            instructions = out.user_scenario.instructions
            localized = task.user_scenario.instructions
            assert instructions.known_info == localized.known_info
            assert instructions.task_instructions == localized.task_instructions
            assert NAME_AUTH_CLAUSE_HEAD in instructions.task_instructions
            assert phone_digits("555-123-2002") not in phone_digits(
                instructions.known_info
            )


class TestScenarioConsistencyGuard:
    """The English swap must never change what the caller knows. The guard
    turns a wrong-variant source lookup into a failed run instead of a
    silently corrupted arm."""

    def _tasks(self, source_unknown, localized_unknown):
        def make(task_id, unknown):
            return Task.model_validate(
                {
                    "id": task_id,
                    "user_scenario": {
                        "instructions": {
                            "domain": "telecom",
                            "reason_for_call": "Data broken.",
                            "known_info": "You are Jorge Gil.",
                            "unknown_info": unknown,
                            "task_instructions": "Follow guidance.",
                        }
                    },
                }
            )

        return make("[toy]a_xx_identity", localized_unknown), make(
            "[toy]a", source_unknown
        ).user_scenario

    def test_unknown_info_presence_mismatch_raises(self):
        from tau2.multilingual.english_prompts import _check_scenario_consistency

        localized, english = self._tasks(
            source_unknown=None,
            localized_unknown="You do not remember your account PIN.",
        )
        with pytest.raises(ValueError, match="unknown_info mismatch"):
            _check_scenario_consistency(localized, english, "telecom")

    def test_untranslated_domain_requires_byte_equality(self):
        from tau2.multilingual.english_prompts import _check_scenario_consistency

        localized, english = self._tasks(
            source_unknown="You do not remember your account PIN.",
            localized_unknown="You do not remember your account PIN.",
        )
        english.instructions.known_info = "You are Jorge Gil with phone 555."
        with pytest.raises(ValueError, match="known_info mismatch"):
            _check_scenario_consistency(localized, english, "telecom")

    def test_matching_scenarios_pass(self):
        from tau2.multilingual.english_prompts import _check_scenario_consistency

        localized, english = self._tasks(
            source_unknown="You do not remember your account PIN.",
            localized_unknown="You do not remember your account PIN.",
        )
        _check_scenario_consistency(localized, english, "telecom")

    def test_translated_domain_allows_prose_divergence(self, monkeypatch):
        """The retired TRANSLATED regime: byte-inequality is expected there and
        only the presence shape is enforced.

        No shipped domain declares ``tasks_translated`` today — both airline
        and telecom ship English-prose arms — so the branch is exercised
        against a profile copy that does.
        """
        from tau2.multilingual import domain_profiles
        from tau2.multilingual.english_prompts import _check_scenario_consistency

        profile = domain_profiles.get_domain_profile("airline")
        monkeypatch.setitem(
            domain_profiles.DOMAIN_PROFILES,
            "airline",
            profile.model_copy(update={"tasks_translated": True}),
        )
        localized, english = self._tasks(source_unknown=None, localized_unknown=None)
        localized.user_scenario.instructions.known_info = "आप जॉर्ज गिल हैं।"
        _check_scenario_consistency(localized, english, "airline")


class TestTextOrchestratorWiring:
    def _config(self, **overrides):
        return TextRunConfig(
            domain="airline",
            agent="llm_agent",
            user_persona_id="hi",
            task_set_name="airline_hi",
            **overrides,
        )

    def test_english_prompt_end_to_end(self):
        localized = _airline_hi_task()
        orch = build_text_orchestrator(self._config(), localized, seed=42)
        # The user reads English instructions + English guidelines + directive.
        assert not DEVANAGARI.search(orch.user.instructions)
        assert DIRECTIVE_MARKER in orch.user.system_prompt
        # The orchestrator/evaluator task is the LOCALIZED task, untouched:
        # only what the user simulator reads is swapped. (Arm prose is the
        # English source under the standard regime, so "untouched" is asserted
        # on identity — the localized id and the criteria the evaluator scores
        # — not on the script the prose happens to be written in.)
        assert orch.task is not None
        assert orch.task.id == localized.id == "7_hi"
        assert orch.task.evaluation_criteria == localized.evaluation_criteria
        assert orch.task.initial_state == localized.initial_state

    def test_english_baseline_run_unaffected(self):
        config = TextRunConfig(domain="airline", agent="llm_agent")
        task = _airline_source_task("7")
        orch = build_text_orchestrator(config, task, seed=42)
        assert DIRECTIVE_MARKER not in orch.user.system_prompt
        assert orch.user.instructions == str(task.user_scenario)
