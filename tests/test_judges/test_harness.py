# Copyright Sierra
"""Deterministic nativeness harness: checkers + aggregation + applicability."""

import pytest

from tau2.data_model.message import AssistantMessage, Tick
from tau2.data_model.simulation import (
    JudgeOutcome,
    NativenessJudgeSettings,
    SimulationRun,
)
from tau2.data_model.tasks import StructuredUserInstructions, Task, UserScenario
from tau2.judges.nativeness.caller_identity import korean_caller_name
from tau2.judges.nativeness.checkers import (
    CheckerContext,
    check_email_symbol_verbalization,
    check_es_gender_agreement,
    check_es_register_formality,
    check_hi_gender_agreement,
    check_hi_honorific_agreement,
    check_hi_register_formality,
    check_ko_counting_units,
    check_ko_honorific_agreement,
    check_ko_honorific_levels,
    check_ko_name_address_conventions,
    check_pt_gender_agreement,
    check_pt_regional_consistency,
    check_zh_counting_units,
    check_zh_regional_consistency,
    check_zh_register_formality,
    check_zh_script_consistency,
)
from tau2.judges.nativeness.factors import email_symbols_for
from tau2.judges.nativeness.harness import (
    checker_context_for_simulation,
    evaluate_nativeness,
    pinned_agent_greeting,
)

PASS = JudgeOutcome.PASS
FAIL = JudgeOutcome.FAIL
NONE = JudgeOutcome.NO_OPPORTUNITY

NO_JUDGE = NativenessJudgeSettings(llm_judge=False)


def _ko_name():
    name = korean_caller_name("Minjun", "Kim")
    assert name is not None
    return name


def _ctx(text, *, has_email=False, language="hi"):
    return CheckerContext(
        agent_text=text,
        has_email=has_email,
        language=language,
        is_voice=True,
        # email symbols come from the pack (empty for unknown languages).
        email_symbols=email_symbols_for(language),
    )


# --- email_symbol_verbalization --------------------------------------------


@pytest.mark.parametrize(
    "text,language,expected",
    [
        ("jose arroba correo punto es", "es", PASS),  # native tokens
        ("jose dot correo arroba es", "es", FAIL),  # English "dot" leak
        ("wang at qq 点 com", "zh", PASS),  # "at" IS native in zh (艾特/at)
        ("kim dot naver", "ko", FAIL),  # "dot" not native in ko (점)
        (
            "hans dot mueller at the rate gmail",
            "hi",
            PASS,
        ),  # English IS native (Hinglish)
        ("let me pull that up", "zh", NONE),  # email present, no symbol words
        ("foo dot bar", "sw", NONE),  # no seed table for language
    ],
)
def test_email_symbol_verbalization(text, language, expected):
    outcome, _ = check_email_symbol_verbalization(
        _ctx(text, has_email=True, language=language)
    )
    assert outcome == expected


def test_email_no_opportunity_without_email_in_task():
    # "at" appears but the task carries no email -> never fires.
    outcome, _ = check_email_symbol_verbalization(
        _ctx("meet at noon", has_email=False, language="es")
    )
    assert outcome == NONE


def test_email_symbol_verbalization_is_voice_only():
    outcome, _ = check_email_symbol_verbalization(
        CheckerContext(
            agent_text="jose@correo.es",
            has_email=True,
            language="es",
            is_voice=False,
            email_symbols=email_symbols_for("es"),
        )
    )
    assert outcome == NONE


# --- Chinese text script consistency --------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("我来帮您取消订单，App 和 WiFi 都可以继续使用。", PASS),
        ("我來幫您取消訂單，App 和 WiFi 都可以繼續使用。", FAIL),
        ("reservation ID 是 HAT260。", PASS),
    ],
)
def test_zh_script_consistency_is_deterministic_and_allows_latin(text, expected):
    outcome, _ = check_zh_script_consistency(
        CheckerContext(agent_text=text, has_email=False, language="zh")
    )
    assert outcome == expected


def test_zh_script_consistency_is_skipped_for_voice_and_task_literals():
    voice, _ = check_zh_script_consistency(
        CheckerContext(
            agent_text="我會處理這個訂單。",
            has_email=False,
            language="zh",
            is_voice=True,
        )
    )
    supplied_name, _ = check_zh_script_consistency(
        CheckerContext(
            agent_text="您好，張偉。",
            has_email=False,
            language="zh",
            allowed_literals=["張偉"],
        )
    )
    assert voice == NONE
    assert supplied_name == PASS


# --- deterministic customer register ---------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("आप अपना booking ID बताइए।", PASS),
        ("booking ID बता दीजिए।", NONE),
        ("तुम अपना booking ID बताओ।", FAIL),
        ("तू ज़रा रुक।", FAIL),
        ("तुम्हारा नंबर क्या है?", FAIL),
        ("तुझे confirmation मिल जाएगा।", FAIL),
    ],
)
def test_hi_register_rejects_familiar_customer_address(text, expected):
    outcome, _ = check_hi_register_formality(
        CheckerContext(agent_text=text, has_email=False, language="hi")
    )
    assert outcome == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("¿Me puede dar su número de reserva, por favor?", NONE),
        ("¿Me puedes dar tu número de reserva?", PASS),
        ("¿Usted me confirma el número?", PASS),
        ("Te envío el código ahora.", PASS),
        ("Usted puede darme tu número.", FAIL),
        ("Su política permite cambios; te ayudo con el suyo.", PASS),
        ("Voy a comprobar la reserva.", NONE),
    ],
)
def test_es_register_requires_one_consistent_address_family(text, expected):
    outcome, _ = check_es_register_formality(
        CheckerContext(agent_text=text, has_email=False, language="es")
    )
    assert outcome == expected


@pytest.mark.parametrize(
    "text",
    [
        "您好，请问您能提供订单号吗？",
        "请提供订单号。",
        "你好，请问你能提供订单号吗？",
        "您稍等，我会给你确认。",
        "请问你们需要哪一个方案？",
    ],
)
def test_zh_register_is_retired_and_never_scores(text):
    # v4: recall_20 calibration — native raters accepted informal 你-address
    # service calls; the checker's confirmed catches did not justify its
    # false-positive profile, so Mandarin register is never scored.
    outcome, _ = check_zh_register_formality(
        CheckerContext(agent_text=text, has_email=False, language="zh")
    )
    assert outcome == NONE


@pytest.mark.parametrize(
    "turns",
    [
        ["확인해 드리겠습니다.", "잠시만 기다려 주세요."],
        ["어디 보자."],
        ["이름 말해.", "잠깐 기다려."],
        ["알겠어?", "번호 알려줘."],
    ],
)
def test_ko_honorific_levels_is_retired_and_never_scores(turns):
    # v5: owner ruling 2026-09-01 — zero failures in the calibration corpus
    # and every historical firing was a fragment-split false positive, so
    # Korean honorific level is never scored (even on textbook banmal).
    outcome, _ = check_ko_honorific_levels(
        CheckerContext(
            agent_text="\n".join(turns),
            agent_turns=turns,
            has_email=False,
            language="ko",
        )
    )
    assert outcome == NONE


@pytest.mark.parametrize(
    ("checker", "language", "text", "allowed_literals"),
    [
        (check_hi_register_formality, "hi", "तुम", ["तुम"]),
        (check_es_register_formality, "es", "usted", ["USTED"]),
        (check_zh_register_formality, "zh", "你", ["你"]),
        # ko honorific_levels is retired (checker v5) — no literal case.
    ],
)
def test_register_checkers_ignore_task_supplied_literals(
    checker, language, text, allowed_literals
):
    outcome, _ = checker(
        CheckerContext(
            agent_text=text,
            agent_turns=[text],
            has_email=False,
            language=language,
            allowed_literals=allowed_literals,
        )
    )
    assert outcome == NONE


@pytest.mark.parametrize(
    ("checker", "language", "text", "expected"),
    [
        (check_hi_register_formality, "hi", "ग्राहक ने “तुम रुको” कहा।", NONE),
        (
            check_es_register_formality,
            "es",
            'El cliente dijo "tú espera". ¿Usted confirma?',
            PASS,
        ),
        # zh register is retired (checker v4) — no quoted-wording case.
    ],
)
def test_register_checkers_ignore_quoted_customer_wording(
    checker, language, text, expected
):
    outcome, _ = checker(
        CheckerContext(agent_text=text, has_email=False, language=language)
    )
    assert outcome == expected


# --- hybrid gender-agreement prechecks -------------------------------------


@pytest.mark.parametrize(
    ("checker", "language", "text", "agent_gender", "caller_gender"),
    [
        (check_hi_gender_agreement, "hi", "मैं आपकी मदद कर सकता हूँ।", "female", None),
        (check_hi_gender_agreement, "hi", "मैं आपकी मदद कर सकती हूँ।", "male", None),
        (check_hi_gender_agreement, "hi", "आप अभी प्रतीक्षा कर रही हैं।", None, "male"),
        # spoken slash-alternatives fail regardless of any known gender
        (check_hi_gender_agreement, "hi", "मैं सुन रहा/रही हूँ।", None, None),
        (check_hi_gender_agreement, "hi", "मैं बता देता/देती हूँ।", "female", None),
        (check_es_gender_agreement, "es", "Estoy listo para ayudarle.", "female", None),
        (check_es_gender_agreement, "es", "Estoy segura de eso.", "male", None),
        (check_es_gender_agreement, "es", "Gracias, señor.", None, "female"),
        (check_es_gender_agreement, "es", "Señora, deme el código.", None, "male"),
        (check_pt_gender_agreement, "pt", "Estou pronto para ajudar.", "female", None),
        (check_pt_gender_agreement, "pt", "Estou pronta para ajudar.", "male", None),
        (check_pt_gender_agreement, "pt", "Obrigado, senhor.", None, "female"),
        (check_pt_gender_agreement, "pt", "Senhora, informe o código.", None, "male"),
    ],
)
def test_gender_prechecks_fail_only_participant_anchored_contradictions(
    checker, language, text, agent_gender, caller_gender
):
    outcome, evidence = checker(
        CheckerContext(
            agent_text=text,
            agent_turns=[text],
            has_email=False,
            language=language,
            agent_gender=agent_gender,
            caller_gender=caller_gender,
        )
    )

    assert outcome == FAIL
    assert evidence


@pytest.mark.parametrize(
    ("checker", "language", "text", "agent_gender", "caller_gender"),
    [
        (check_hi_gender_agreement, "hi", "मैं आपकी मदद कर सकती हूँ।", "female", None),
        (check_hi_gender_agreement, "hi", "आप अभी प्रतीक्षा कर रही हैं।", None, "female"),
        # v3 carve-out: honorific आप + masculine plural is the unmarked polite
        # form for any addressee — never a checker failure, even for a known
        # female caller.
        (check_hi_gender_agreement, "hi", "आप अभी प्रतीक्षा कर रहे हैं।", None, "female"),
        (
            check_hi_gender_agreement,
            "hi",
            "क्या आप मुझे अपना नंबर बता सकते हैं?",
            None,
            "female",
        ),
        (check_es_gender_agreement, "es", "Estoy lista para ayudarle.", "female", None),
        (check_es_gender_agreement, "es", "Gracias, señora.", None, "female"),
        (check_pt_gender_agreement, "pt", "Estou pronta para ajudar.", "female", None),
        (check_pt_gender_agreement, "pt", "Obrigada, senhora.", None, "female"),
        (check_pt_gender_agreement, "pt", "Obrigado pela espera.", "female", None),
        (check_pt_gender_agreement, "pt", "Obrigada pela espera.", "male", None),
        (check_es_gender_agreement, "es", "Estoy listo para ayudarle.", None, None),
    ],
)
def test_gender_prechecks_defer_correct_ambiguous_and_unanchored_language(
    checker, language, text, agent_gender, caller_gender
):
    outcome, evidence = checker(
        CheckerContext(
            agent_text=text,
            agent_turns=[text],
            has_email=False,
            language=language,
            agent_gender=agent_gender,
            caller_gender=caller_gender,
        )
    )

    assert outcome == NONE
    assert evidence is None


def test_gender_prechecks_ignore_task_literals_and_non_vocative_titles():
    supplied, _ = check_es_gender_agreement(
        CheckerContext(
            agent_text="Estoy listo para ayudarle.",
            agent_turns=["Estoy listo para ayudarle."],
            has_email=False,
            language="es",
            agent_gender="female",
            allowed_literals=["Estoy listo"],
        )
    )
    third_person, _ = check_es_gender_agreement(
        CheckerContext(
            agent_text="La reserva pertenece al señor García.",
            agent_turns=["La reserva pertenece al señor García."],
            has_email=False,
            language="es",
            caller_gender="female",
        )
    )

    assert supplied == third_person == NONE


@pytest.mark.parametrize(
    ("checker", "language", "user_text"),
    [
        (check_hi_gender_agreement, "hi", "मैं मदद कर सकता हूँ।"),
        (check_es_gender_agreement, "es", "Estoy listo."),
        (check_pt_gender_agreement, "pt", "Estou pronto."),
    ],
)
def test_gender_prechecks_never_score_user_speech(checker, language, user_text):
    outcome, _ = checker(
        CheckerContext(
            agent_text="मैं सुन रहा हूँ" if language == "hi" else "Entendido.",
            agent_turns=["मैं सुन रहा हूँ" if language == "hi" else "Entendido."],
            user_turns=[user_text],
            has_email=False,
            language=language,
            agent_gender="female",
        )
    )

    assert outcome == NONE


OLD_HI_PINNED_GREETING = "नमस्ते! मैं आपकी कैसे मदद कर सकता हूँ?"


@pytest.mark.parametrize(
    ("checker", "language", "greeting", "agent_gender"),
    [
        # The historical hi greeting is itself masculine — the exact artifact
        # the v2 exclusion removes (a known-female agent contradicted by our
        # own pinned line).
        (check_hi_gender_agreement, "hi", OLD_HI_PINNED_GREETING, "female"),
        (check_es_gender_agreement, "es", "Estoy listo para ayudarle.", "female"),
        (check_pt_gender_agreement, "pt", "Estou pronto para ajudar.", "female"),
    ],
)
def test_gender_prechecks_never_score_the_pinned_greeting(
    checker, language, greeting, agent_gender
):
    outcome, evidence = checker(
        CheckerContext(
            agent_text=greeting,
            agent_turns=[greeting, "ok"],
            has_email=False,
            language=language,
            agent_gender=agent_gender,
            pinned_greeting=greeting,
        )
    )

    assert outcome == NONE
    assert evidence is None


def test_gender_precheck_still_fails_model_turns_alongside_pinned_greeting():
    model_turn = "मैं आपकी मदद कर सकता हूँ।"
    outcome, evidence = check_hi_gender_agreement(
        CheckerContext(
            agent_text=f"{OLD_HI_PINNED_GREETING}\n{model_turn}",
            agent_turns=[OLD_HI_PINNED_GREETING, model_turn],
            has_email=False,
            language="hi",
            agent_gender="female",
            pinned_greeting=OLD_HI_PINNED_GREETING,
        )
    )

    assert outcome == FAIL
    assert evidence is not None and evidence.startswith("gender-v3:")


def test_gender_precheck_pinned_exclusion_leaves_other_clauses_untouched():
    # Feminine-with-male-agent and caller-side contradictions in ordinary
    # model turns still fail when a (different) pinned greeting is present.
    feminine, _ = check_hi_gender_agreement(
        CheckerContext(
            agent_text="मैं आपकी मदद कर सकती हूँ।",
            agent_turns=["नमस्ते!", "मैं आपकी मदद कर सकती हूँ।"],
            has_email=False,
            language="hi",
            agent_gender="male",
            pinned_greeting="नमस्ते!",
        )
    )
    caller_side, _ = check_hi_gender_agreement(
        CheckerContext(
            agent_text="आप अभी प्रतीक्षा कर रही हैं।",
            agent_turns=["नमस्ते!", "आप अभी प्रतीक्षा कर रही हैं।"],
            has_email=False,
            language="hi",
            caller_gender="male",
            pinned_greeting="नमस्ते!",
        )
    )

    assert feminine == caller_side == FAIL


# --- hybrid honorific-agreement prechecks ---------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "आप अपना नंबर बताओ।",
        "आप यह काम करो।",
        "आप इसे अभी लो।",
        "आप confirmation दो।",
        "आप तैयार है।",
        "आप यह कर सकता है।",
        "आप अब बता।",
    ],
)
def test_hi_honorific_agreement_fails_bounded_aap_contradictions(text):
    outcome, evidence = check_hi_honorific_agreement(
        CheckerContext(
            agent_text=text,
            agent_turns=[text],
            has_email=False,
            language="hi",
        )
    )

    assert outcome == FAIL
    assert "source=" in (evidence or "")


@pytest.mark.parametrize(
    "text",
    [
        "आपका खाता बंद है।",
        "आपकी booking तैयार है।",
        "आपके खाते में समस्या है।",
        "आपका फोन दो साल पुराना है।",
        "आप बता सकते हैं।",
        "आप तैयार हैं।",
        "आप बताइए।",
        "आपको confirmation देनी होगी।",
        "आप इसे लें।",
        "आप यह करें।",
        "आप जानते हैं कि वह यह कर सकता है।",
        "आप देख सकते हैं कि booking तैयार है।",
    ],
)
def test_hi_honorific_agreement_defers_possessives_and_formal_agreement(text):
    outcome, evidence = check_hi_honorific_agreement(
        CheckerContext(
            agent_text=text,
            agent_turns=[text],
            has_email=False,
            language="hi",
        )
    )

    assert outcome == NONE
    assert evidence is None


def test_hi_honorific_agreement_bounds_adjacent_customer_fragments():
    same_turn, _ = check_hi_honorific_agreement(
        CheckerContext(
            agent_text="आप तैयार हैं। फिर बताओ।",
            agent_turns=["आप तैयार हैं। फिर बताओ।"],
            has_email=False,
            language="hi",
        )
    )
    next_turn, _ = check_hi_honorific_agreement(
        CheckerContext(
            agent_text="क्या आप तैयार हैं?\nफिर बताओ।",
            agent_turns=["क्या आप तैयार हैं?", "फिर बताओ।"],
            has_email=False,
            language="hi",
        )
    )
    too_far, _ = check_hi_honorific_agreement(
        CheckerContext(
            agent_text="क्या आप तैयार हैं?\nमैं जाँच करता हूँ।\nफिर बताओ।",
            agent_turns=[
                "क्या आप तैयार हैं?",
                "मैं जाँच करता हूँ।",
                "फिर बताओ।",
            ],
            has_email=False,
            language="hi",
        )
    )

    assert same_turn == next_turn == FAIL
    assert too_far == NONE


def test_hi_honorific_agreement_defers_only_bare_bata_at_interrupted_tail():
    interrupted_tail, evidence = check_hi_honorific_agreement(
        CheckerContext(
            agent_text="आप बता",
            agent_turns=["आप बता"],
            agent_turn_interruptions=[True],
            has_email=False,
            language="hi",
        )
    )
    complete_turn, _ = check_hi_honorific_agreement(
        CheckerContext(
            agent_text="आप बता",
            agent_turns=["आप बता"],
            agent_turn_interruptions=[False],
            has_email=False,
            language="hi",
        )
    )
    punctuated_complete_turn, _ = check_hi_honorific_agreement(
        CheckerContext(
            agent_text="आप बता।",
            agent_turns=["आप बता।"],
            agent_turn_interruptions=[True],
            has_email=False,
            language="hi",
        )
    )
    complete_before_cut, _ = check_hi_honorific_agreement(
        CheckerContext(
            agent_text="आप बता। मैं अभी जाँच क",
            agent_turns=["आप बता। मैं अभी जाँच क"],
            agent_turn_interruptions=[True],
            has_email=False,
            language="hi",
        )
    )

    assert interrupted_tail == NONE
    assert evidence is None
    assert complete_turn == punctuated_complete_turn == complete_before_cut == FAIL


@pytest.mark.parametrize(
    "text",
    [
        "제가 확인하시겠습니다.",
        "제가 처리하실게요.",
        "저희가 안내하시겠습니다.",
        "고객님께서 신청한 예약을 확인하겠습니다.",
    ],
)
def test_ko_honorific_agreement_fails_reviewed_same_clause_contradictions(text):
    outcome, evidence = check_ko_honorific_agreement(
        CheckerContext(
            agent_text=text,
            agent_turns=[text],
            has_email=False,
            language="ko",
        )
    )

    assert outcome == FAIL
    assert "source=" in (evidence or "")


@pytest.mark.parametrize(
    "turns",
    [
        ["제가 고객님께서 신청하신 예약을 확인하겠습니다."],
        ["제가 고객님께서 처리하실게요라고 하신 내용을 확인하겠습니다."],
        ["고객님이 필요한 서류를 보내 주세요."],
        ["신청한 예약을 확인하겠습니다."],
        ["제가 확인하겠습니다. 기다려 주세요."],
        ["제가 처리하겠습니다."],
        ["제가 안내합니다.", "확인하시겠습니다."],
    ],
)
def test_ko_honorific_agreement_defers_correct_ambiguous_and_cross_turn(turns):
    outcome, evidence = check_ko_honorific_agreement(
        CheckerContext(
            agent_text="\n".join(turns),
            agent_turns=turns,
            has_email=False,
            language="ko",
        )
    )

    assert outcome == NONE
    assert evidence is None


@pytest.mark.parametrize(
    ("checker", "language", "marker"),
    [
        (check_hi_honorific_agreement, "hi", "आप तैयार है"),
        (check_ko_honorific_agreement, "ko", "제가 확인하시겠습니다"),
    ],
)
def test_honorific_agreement_ignores_customer_speech_and_task_literals(
    checker, language, marker
):
    user_only, _ = checker(
        CheckerContext(
            agent_text="ठीक है" if language == "hi" else "알겠습니다.",
            agent_turns=["ठीक है" if language == "hi" else "알겠습니다."],
            user_turns=[marker],
            has_email=False,
            language=language,
        )
    )
    supplied, _ = checker(
        CheckerContext(
            agent_text=marker,
            agent_turns=[marker],
            allowed_literals=[marker],
            has_email=False,
            language=language,
        )
    )

    assert user_only == supplied == NONE


# --- hybrid regional-consistency prechecks ---------------------------------


@pytest.mark.parametrize(
    ("checker", "language", "turns"),
    [
        (
            check_pt_regional_consistency,
            "pt",
            ["Abra o ficheiro.", "Confira no telemóvel."],
        ),
        (
            check_pt_regional_consistency,
            "pt",
            ["Estou a verificar agora.", "Estamos a processar o reembolso."],
        ),
        (
            check_pt_regional_consistency,
            "pt",
            ["Confirme a morada.", "Fale com a nossa equipa de suporte."],
        ),
        (
            check_zh_regional_consistency,
            "zh",
            ["请打开软体。", "请确认您的行动电话号码。"],
        ),
        (
            check_zh_regional_consistency,
            "zh",
            ["我会发一封简讯。", "请确认您的手机门号。"],
        ),
        (
            check_zh_regional_consistency,
            "zh",
            ["请核对您的邮递区号。", "我会查看宅配状态。"],
        ),
    ],
)
def test_regional_prechecks_fail_repeated_strong_non_target_usage(
    checker, language, turns
):
    outcome, evidence = checker(
        CheckerContext(
            agent_text="\n".join(turns),
            agent_turns=turns,
            has_email=False,
            language=language,
        )
    )

    assert outcome == FAIL
    assert "regional-v1" in (evidence or "")


@pytest.mark.parametrize(
    ("checker", "language", "turns"),
    [
        (
            check_pt_regional_consistency,
            "pt",
            ["Abra o ficheiro.", "Envie o ficheiro."],
        ),
        (
            check_zh_regional_consistency,
            "zh",
            ["请打开软体。", "请更新软体。"],
        ),
    ],
)
def test_regional_prechecks_count_same_recurring_choice_across_agent_turns(
    checker, language, turns
):
    outcome, _ = checker(
        CheckerContext(
            agent_text="\n".join(turns),
            agent_turns=turns,
            has_email=False,
            language=language,
        )
    )

    assert outcome == FAIL


@pytest.mark.parametrize(
    ("checker", "language", "turns"),
    [
        (check_pt_regional_consistency, "pt", ["Abra o ficheiro.", "Certo."]),
        (
            check_pt_regional_consistency,
            "pt",
            ["Informe-me o código.", "Tu pode aguardar."],
        ),
        (check_zh_regional_consistency, "zh", ["请打开软体。", "好的。"]),
        (
            check_zh_regional_consistency,
            "zh",
            ["請核對身分證和訂單。", "請更新軟件。"],
        ),
    ],
)
def test_regional_prechecks_defer_single_ambiguous_and_script_only_usage(
    checker, language, turns
):
    outcome, evidence = checker(
        CheckerContext(
            agent_text="\n".join(turns),
            agent_turns=turns,
            has_email=False,
            language=language,
        )
    )

    assert outcome == NONE
    assert evidence is None


@pytest.mark.parametrize(
    ("checker", "language", "marker"),
    [
        (check_pt_regional_consistency, "pt", "telemóvel"),
        (check_zh_regional_consistency, "zh", "行動電話"),
    ],
)
def test_regional_prechecks_ignore_customer_mirroring_task_literals_and_quotes(
    checker, language, marker
):
    mirrored, _ = checker(
        CheckerContext(
            agent_text=f"Ha dicho {marker}.\nRepito: {marker}.",
            agent_turns=[f"Ha dicho {marker}.", f"Repito: {marker}."],
            user_turns=[f"Mi término es {marker}."],
            has_email=False,
            language=language,
        )
    )
    supplied, _ = checker(
        CheckerContext(
            agent_text=f"Use {marker}.\nConfirme {marker}.",
            agent_turns=[f"Use {marker}.", f"Confirme {marker}."],
            allowed_literals=[marker],
            has_email=False,
            language=language,
        )
    )
    quoted, _ = checker(
        CheckerContext(
            agent_text=f'El sistema muestra "{marker}".\nLa etiqueta dice “{marker}”.',
            agent_turns=[
                f'El sistema muestra "{marker}".',
                f"La etiqueta dice “{marker}”.",
            ],
            has_email=False,
            language=language,
        )
    )

    assert mirrored == supplied == quoted == NONE


@pytest.mark.parametrize(
    ("checker", "language", "marker"),
    [
        (check_pt_regional_consistency, "pt", "ficheiro"),
        (check_zh_regional_consistency, "zh", "软体"),
    ],
)
def test_regional_prechecks_never_score_customer_only_markers(
    checker, language, marker
):
    outcome, _ = checker(
        CheckerContext(
            agent_text="Entendido.\nDe acuerdo.",
            agent_turns=["Entendido.", "De acuerdo."],
            user_turns=[marker, marker],
            has_email=False,
            language=language,
        )
    )

    assert outcome == NONE


# --- hybrid counting-unit prechecks ---------------------------------------


@pytest.mark.parametrize(
    ("checker", "language", "text"),
    [
        (check_ko_counting_units, "ko", "수하물 일 개를 확인했습니다."),
        (check_ko_counting_units, "ko", "처리에는 삼 시간이 걸립니다."),
        (check_ko_counting_units, "ko", "승객 두 개가 등록되어 있습니다."),
        (check_ko_counting_units, "ko", "두 개의 사람이 탑승합니다."),
        (check_zh_counting_units, "zh", "代码包含一位字母。"),
        (check_zh_counting_units, "zh", "我找到了二个订单。"),
        (check_zh_counting_units, "zh", "这里有二张机票。"),
    ],
)
def test_counting_unit_prechecks_fail_only_reviewed_contradictions(
    checker, language, text
):
    outcome, evidence = checker(
        CheckerContext(
            agent_text=text,
            agent_turns=[text],
            has_email=False,
            language=language,
        )
    )

    assert outcome == FAIL
    assert "source=" in (evidence or "")


@pytest.mark.parametrize(
    ("checker", "language", "text"),
    [
        (
            check_ko_counting_units,
            "ko",
            "수하물 한 개, 승객 두 명, 세 시간, 한 시, 삼십 분입니다.",
        ),
        (check_ko_counting_units, "ko", "표 한 장과 예약 한 건입니다."),
        (check_zh_counting_units, "zh", "两张票和两个订单。"),
        (check_zh_counting_units, "zh", "第二个订单是二零二五年的。"),
        (check_zh_counting_units, "zh", "二百元和两百元都可以，另有几个问题。"),
        (check_zh_counting_units, "zh", "系统显示2个订单。"),
        (check_zh_counting_units, "zh", "这张预订需要修改。"),
        (check_zh_counting_units, "zh", "一位订单管理员会帮助您。"),
        (check_ko_counting_units, "ko", "이 시간에는 처리가 어렵습니다."),
        (check_ko_counting_units, "ko", "승객이 두 개의 주문을 했습니다."),
    ],
)
def test_counting_unit_prechecks_defer_accepted_or_ambiguous_forms(
    checker, language, text
):
    outcome, evidence = checker(
        CheckerContext(
            agent_text=text,
            agent_turns=[text],
            has_email=False,
            language=language,
        )
    )

    assert outcome == NONE
    assert evidence is None


@pytest.mark.parametrize(
    ("checker", "language", "bad_text"),
    [
        (check_ko_counting_units, "ko", "수하물 일 개입니다."),
        (check_zh_counting_units, "zh", "二个订单。"),
    ],
)
def test_counting_unit_prechecks_use_agent_authored_unquoted_text_only(
    checker, language, bad_text
):
    customer_only, _ = checker(
        CheckerContext(
            agent_text="확인하겠습니다。",
            agent_turns=["확인하겠습니다。"],
            user_turns=[bad_text],
            has_email=False,
            language=language,
        )
    )
    quoted, _ = checker(
        CheckerContext(
            agent_text=f'고객이 "{bad_text}"라고 말했습니다。',
            agent_turns=[f'고객이 "{bad_text}"라고 말했습니다。'],
            has_email=False,
            language=language,
        )
    )
    supplied, _ = checker(
        CheckerContext(
            agent_text=bad_text,
            agent_turns=[bad_text],
            allowed_literals=[bad_text],
            has_email=False,
            language=language,
        )
    )

    assert customer_only == quoted == supplied == NONE


# --- hybrid Korean name/address precheck -----------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "민준 김 고객님, 잠시만 기다려 주세요.",
        "고객님 성함이 민준 김 맞으신가요?",
        "김 씨, 예약 번호를 확인해 주세요.",
    ],
)
def test_ko_name_address_precheck_fails_only_known_direct_contradictions(text):
    outcome, evidence = check_ko_name_address_conventions(
        CheckerContext(
            agent_text=text,
            agent_turns=[text],
            has_email=False,
            language="ko",
            caller_name=_ko_name(),
        )
    )

    assert outcome == FAIL
    assert "name-address-v1" in (evidence or "")


@pytest.mark.parametrize(
    "text",
    [
        "김민준 고객님, 확인했습니다.",
        "김민준 님, 확인했습니다.",
        "고객님, 확인했습니다.",
        "동승객 민준 김 고객님의 예약도 확인했습니다.",
        '기록에는 "민준 김 고객님"이라고 적혀 있습니다.',
        "Minjun Kim, please wait.",
    ],
)
def test_ko_name_address_precheck_defers_valid_ambiguous_and_reported_names(text):
    outcome, evidence = check_ko_name_address_conventions(
        CheckerContext(
            agent_text=text,
            agent_turns=[text],
            has_email=False,
            language="ko",
            caller_name=_ko_name(),
        )
    )

    assert outcome == NONE
    assert evidence is None


def test_ko_name_address_precheck_requires_typed_locale_identity():
    outcome, _ = check_ko_name_address_conventions(
        CheckerContext(
            agent_text="민준 김 고객님, 잠시만 기다려 주세요.",
            agent_turns=["민준 김 고객님, 잠시만 기다려 주세요."],
            has_email=False,
            language="ko",
        )
    )

    assert outcome == NONE


def test_ko_name_address_precheck_uses_customer_only_as_exemption():
    text = "민준 김 고객님, 잠시만 기다려 주세요."
    outcome, _ = check_ko_name_address_conventions(
        CheckerContext(
            agent_text=text,
            agent_turns=[text],
            user_turns=["제 이름은 민준 김입니다."],
            has_email=False,
            language="ko",
            caller_name=_ko_name(),
        )
    )

    assert outcome == NONE


def test_ko_name_address_precheck_preserves_name_literals_but_strips_other_literals():
    name = _ko_name()
    reversed_address = "민준 김 고객님,"
    name_outcome, _ = check_ko_name_address_conventions(
        CheckerContext(
            agent_text=reversed_address,
            agent_turns=[reversed_address],
            allowed_literals=[name.given_spoken, name.family_spoken],
            has_email=False,
            language="ko",
            caller_name=name,
        )
    )
    task_literal_outcome, _ = check_ko_name_address_conventions(
        CheckerContext(
            agent_text=reversed_address,
            agent_turns=[reversed_address],
            allowed_literals=[name.reversed_spoken],
            has_email=False,
            language="ko",
            caller_name=name,
        )
    )

    assert name_outcome == FAIL
    assert task_literal_outcome == NONE


# --- pinned_agent_greeting (structural extraction) --------------------------


def test_pinned_agent_greeting_tick_path_is_structural():
    # Tick 0: the injected greeting — text-only, silence audio,
    # contains_speech False (see DiscreteTimeAudioNativeAgent
    # .create_initial_message / FullDuplexOrchestrator._initialize_full_duplex).
    sim = SimulationRun(
        id="s1",
        task_id="t1",
        start_time="x",
        end_time="y",
        duration=1.0,
        termination_reason="agent_stop",
        ticks=[
            Tick(
                tick_id=0,
                timestamp="0",
                agent_chunk=AssistantMessage(
                    role="assistant",
                    content=OLD_HI_PINNED_GREETING,
                    is_audio=False,
                    contains_speech=False,
                    chunk_id=0,
                    is_final_chunk=True,
                ),
            ),
            Tick(
                tick_id=1,
                timestamp="1",
                agent_chunk=AssistantMessage(
                    role="assistant",
                    content="मैं देखती हूँ।",
                    utterance_ids=["u1"],
                ),
            ),
        ],
    )
    assert pinned_agent_greeting(sim) == OLD_HI_PINNED_GREETING


def test_pinned_agent_greeting_none_when_first_chunk_is_speech():
    sim = _voice_sim("मैं देखती हूँ।")
    assert pinned_agent_greeting(sim) is None


def test_pinned_agent_greeting_messages_path_is_first_assistant_message():
    sim = _sim("नमस्ते! कैसे मदद करूँ?")
    assert pinned_agent_greeting(sim) == "नमस्ते! कैसे मदद करूँ?"
    empty = _sim("x")
    empty.messages = []
    assert pinned_agent_greeting(empty) is None


def test_evaluate_nativeness_does_not_fail_gender_on_the_pinned_greeting():
    # Messages path: the first assistant message IS the injected opener, so a
    # masculine pinned greeting with a known-female agent must not produce a
    # deterministic gender FAIL (the factor stays with the LLM judge —
    # DEFERRED here because the judge is off).
    sim = _sim(OLD_HI_PINNED_GREETING)
    info = evaluate_nativeness(
        sim,
        _task("help"),
        "hi",
        "deva",
        settings=NO_JUDGE,
        agent_gender="female",
    )
    assert info is not None
    check = next(c for c in info.factor_checks if c.id == "gender_agreement")
    assert check.outcome == JudgeOutcome.DEFERRED


# --- evaluate_nativeness (end to end) --------------------------------------


def _sim(agent_text: str) -> SimulationRun:
    return SimulationRun(
        id="s1",
        task_id="t1",
        start_time="x",
        end_time="y",
        duration=1.0,
        termination_reason="agent_stop",
        messages=[AssistantMessage(role="assistant", content=agent_text)],
    )


def _voice_sim(agent_text: str) -> SimulationRun:
    sim = _sim("")
    sim.messages = None
    sim.ticks = [
        Tick(
            tick_id=0,
            timestamp="0",
            agent_chunk=AssistantMessage.voice(
                content=agent_text, utterance_ids=["a1"]
            ),
        )
    ]
    return sim


def _task(instr: str) -> Task:
    return Task(
        id="t1",
        user_scenario=UserScenario(
            instructions=StructuredUserInstructions(
                domain="airline",
                reason_for_call="book",
                task_instructions=instr,
            )
        ),
    )


@pytest.mark.parametrize(
    ("language", "text", "factor_id"),
    [
        ("hi", "तुम अपना booking ID बताओ।", "register_formality"),
        ("es", "Usted puede darme tu número.", "register_formality"),
        ("zh", "您好，请问您能提供订单号吗？", "register_formality"),
        ("ko", "이름 말해. 잠깐 기다려.", "honorific_levels"),
        ("ko", "수하물 일 개입니다.", "counting_units"),
    ],
)
def test_retired_pack_checkers_are_not_scheduled(language, text, factor_id):
    info = evaluate_nativeness(
        _sim(text),
        _task("help"),
        language,
        None,
        settings=NO_JUDGE,
    )
    assert info is not None
    assert factor_id not in {check.id for check in info.factor_checks}
    assert info.judge_model is None


def test_english_voice_run_keeps_deterministic_interaction_factors():
    info = evaluate_nativeness(
        _voice_sim("anything"), _task("x"), "en", None, settings=NO_JUDGE
    )
    assert info is not None
    assert {check.id for check in info.factor_checks} == {
        "backchannel_frequency",
        "email_symbol_verbalization",
    }
    assert (
        evaluate_nativeness(_sim("anything"), _task("x"), None, None, settings=NO_JUDGE)
        is None
    )


def test_english_text_run_has_no_applicable_factors():
    # Both universal deterministic factors are voice-only (backchannels and
    # spoken symbol readouts have no typed-chat analog), so an English text
    # run records no factor checks and no score.
    info = evaluate_nativeness(
        _sim("anything"), _task("x"), "en", None, settings=NO_JUDGE
    )
    assert info is not None
    assert info.factor_checks == []
    assert info.score is None


def test_non_english_perfect_run_scores_one():
    sim = _voice_sim("su código es JMO1MG. correo: jose arroba correo punto es")
    task = _task("email jose@correo.es, code JMO1MG")
    info = evaluate_nativeness(sim, task, "es", "latn", settings=NO_JUDGE)
    assert info is not None
    # Ambiguous ``su`` does not create a register opportunity.
    assert info.score == 1.0
    fired = [c for c in info.factor_checks if c.outcome in (PASS, FAIL)]
    assert {c.id for c in fired} == {"email_symbol_verbalization"}


def test_non_english_failure_lowers_score():
    sim = _voice_sim("correo: jose at correo dot es")
    task = _task("email jose@correo.es")
    info = evaluate_nativeness(sim, task, "es", "latn", settings=NO_JUDGE)
    assert info is not None
    assert info.score == 0.0
    assert info.num_pass == 0 and info.num_fail == 1


def test_simulation_interruption_reaches_hi_bare_bata_precheck():
    sim = _voice_sim("आप बता")
    assert sim.ticks is not None
    assert sim.ticks[0].agent_chunk is not None
    sim.ticks[0].agent_chunk.raw_data = {"was_truncated": True}

    context = checker_context_for_simulation(sim, _task("help"), "hi")
    outcome, evidence = check_hi_honorific_agreement(context)

    assert context.agent_turns == ["आप बता"]
    assert context.agent_turn_interruptions == [True]
    assert outcome == NONE
    assert evidence is None


def test_zh_text_run_scores_script_consistency_but_voice_does_not():
    text_info = evaluate_nativeness(
        _sim("我來幫您取消訂單。"),
        _task("help"),
        "zh",
        "hans",
        settings=NO_JUDGE,
    )
    assert text_info is not None
    text_check = next(
        check for check in text_info.factor_checks if check.id == "script_consistency"
    )
    assert text_check.outcome == FAIL

    voice_info = evaluate_nativeness(
        _voice_sim("我會處理這個訂單。"),
        _task("help"),
        "zh",
        "hans",
        settings=NO_JUDGE,
    )
    assert voice_info is not None
    assert "script_consistency" not in {check.id for check in voice_info.factor_checks}


# --- live factor configuration -----------------------------------------------


def test_configured_korean_factors_are_all_live():
    sim = _voice_sim("hong 골뱅이 mail 점 com")
    info = evaluate_nativeness(
        sim,
        _task("email hong@mail.com"),
        "ko",
        "kore",
        settings=NO_JUDGE,
    )
    assert info is not None

    assert all(not check.shadow for check in info.factor_checks)
    deferred = [
        check for check in info.factor_checks if check.outcome == JudgeOutcome.DEFERRED
    ]
    assert info.num_deferred == len(deferred)
    assert info.num_pass == 1
    assert info.num_fail == 0
    assert info.score == 1.0
    assert info.score_coverage == pytest.approx(1 / len(info.factor_checks))


def test_judge_factor_is_deferred_and_ignored(monkeypatch):
    import tau2.judges.nativeness.harness as harness
    from tau2.judges.nativeness.factors import (
        NativenessFactorConfig,
        NativenessRubricText,
    )

    judge = NativenessFactorConfig(
        id="register_tv",
        category="register",
        type="judge",
        severity=3,
        params=NativenessRubricText(
            question="Did the agent use the right register?",
            opportunity="agent addresses the customer",
            shared_allowed="uses the right register",
            shared_violation="uses the wrong register",
            language_criterion="tv",
            language_question="Did the agent use the right local form?",
            language_allowed="x",
            language_violation="y",
        ),
    )
    configured = harness.judge_factors_for("es")
    monkeypatch.setattr(
        harness,
        "judge_factors_for",
        lambda language: [*configured, judge],
    )

    sim = _voice_sim("su código es JMO1MG. jose arroba correo punto es")
    task = _task("email jose@correo.es, code JMO1MG")
    info = evaluate_nativeness(sim, task, "es", "latn", settings=NO_JUDGE)
    deferred = [c for c in info.factor_checks if c.id == "register_tv"]
    assert deferred and deferred[0].outcome == JudgeOutcome.DEFERRED
    # Deferred factor must not move the score (still 1.0 from the email PASS).
    assert info.score == 1.0


def test_voice_corpus_merges_chunks_smoothly():
    # Per-tick gold-transcript fragments must merge into flowing speech (no
    # mid-word newline breaks); separate utterances become separate lines.
    from tau2.data_model.message import Tick
    from tau2.judges.nativeness.harness import build_agent_corpus

    def tick(i, content, uid):
        return Tick(
            tick_id=i,
            timestamp="t",
            agent_chunk=AssistantMessage(
                role="assistant", content=content, utterance_ids=[uid]
            ),
        )

    sim = SimulationRun(
        id="s1",
        task_id="t1",
        start_time="x",
        end_time="y",
        duration=1.0,
        termination_reason="agent_stop",
        ticks=[tick(1, "नमस्ते ", "u1"), tick(2, "जी", "u1"), tick(3, "कैसे हैं", "u2")],
    )
    corpus = build_agent_corpus(sim)
    assert "नमस्ते जी" in corpus  # same-utterance fragments joined, no break
    assert "कैसे हैं" in corpus  # second utterance present
    assert "नमस्ते\nजी" not in corpus  # no mid-utterance newline
    assert corpus.startswith("agent: ")  # explicit per-utterance turn marker
    assert corpus.count("agent: ") == 2  # one per merged utterance


def test_voice_corpus_extracts_delivered_text_from_markup_gold():
    # Streaming voice runs carry audio_script_gold as a marked-up chunk template
    # (<message uuid=".." active=".."><chunk id=N>..</chunk>..). The judge corpus
    # must contain only the plain text of the delivered (active) chunks — never
    # the XML markup, never chunks the caller interrupted.
    from tau2.data_model.message import Tick
    from tau2.judges.nativeness.harness import build_agent_corpus

    gold = (
        '<message uuid="abc" active="0,1">'
        "<chunk id=0>Hola, su reembo</chunk>"
        "<chunk id=1>lso está en camino</chunk>"
        "<chunk id=2> y llegará el viernes.</chunk>"
        "</message>"
    )
    sim = SimulationRun(
        id="s1",
        task_id="t1",
        start_time="x",
        end_time="y",
        duration=1.0,
        termination_reason="agent_stop",
        ticks=[
            Tick(
                tick_id=1,
                timestamp="t",
                agent_chunk=AssistantMessage(
                    role="assistant",
                    content="Hola, su reembo",
                    audio_script_gold=gold,
                    utterance_ids=["u1"],
                ),
            )
        ],
    )
    corpus = build_agent_corpus(sim)
    # Delivered chunks joined without separator (chunks split text mid-word).
    assert "agent: Hola, su reembolso está en camino" in corpus
    assert "<chunk" not in corpus and "<message" not in corpus  # no markup leaks
    assert "llegará el viernes" not in corpus  # interrupted chunk never delivered


def test_interrupted_turns_are_marked_for_the_llm_judge_only():
    # A barged-in turn legitimately ends mid-word ("reembo…"); the LLM judge
    # must see it marked as cut off (so a truncated tail never reads as a
    # grammar/honorific violation), while checkers and corpus exports keep the
    # raw text.
    from tau2.data_model.message import Tick
    from tau2.judges.base import INTERRUPTED_TURN_MARKER
    from tau2.judges.nativeness.harness import (
        build_agent_corpus,
        build_agent_turns,
        checker_context_from_turns,
    )

    gold = (
        '<message uuid="abc" active="0">'
        "<chunk id=0>Su transcripció</chunk>"
        "<chunk id=1>n combinada está lista.</chunk>"
        "</message>"
    )
    sim = SimulationRun(
        id="s1",
        task_id="t1",
        start_time="x",
        end_time="y",
        duration=1.0,
        termination_reason="agent_stop",
        ticks=[
            Tick(
                tick_id=1,
                timestamp="t",
                agent_chunk=AssistantMessage(
                    role="assistant",
                    content="Su transcripció",
                    audio_script_gold=gold,
                    utterance_ids=["u1"],
                ),
            )
        ],
    )
    turns = build_agent_turns(sim)
    assert len(turns) == 1
    assert turns[0].interrupted is True
    assert turns[0].text == "Su transcripció"  # raw text: no marker
    assert turns[0].judged_text == f"Su transcripció {INTERRUPTED_TURN_MARKER}"
    assert INTERRUPTED_TURN_MARKER not in build_agent_corpus(sim)
    checker_context = checker_context_from_turns(turns, "es")
    assert checker_context.agent_turns == ["Su transcripció"]
    assert checker_context.agent_text == "agent: Su transcripció"
    assert checker_context.agent_turn_interruptions == [True]


def test_pre_markup_runs_detect_interruption_from_truncated_chunks():
    # Runs that predate markup golds (plain content, no audio_script_gold)
    # carry the barge-in truth in the runner's per-chunk raw_data.was_truncated
    # — the same signal the voice metrics use. The turn must be marked.
    from tau2.data_model.message import Tick
    from tau2.judges.nativeness.harness import build_agent_turns

    def tick(i, content, uid, truncated=False):
        return Tick(
            tick_id=i,
            timestamp="t",
            agent_chunk=AssistantMessage(
                role="assistant",
                content=content,
                utterance_ids=[uid],
                raw_data={"was_truncated": True} if truncated else None,
            ),
        )

    sim = SimulationRun(
        id="s1",
        task_id="t1",
        start_time="x",
        end_time="y",
        duration=1.0,
        termination_reason="agent_stop",
        ticks=[
            tick(1, "कुल free ", "u1"),
            tick(2, "checked su", "u1", truncated=True),
            tick(3, "ठीक है।", "u2"),
        ],
    )
    turns = build_agent_turns(sim)
    assert [t.interrupted for t in turns] == [True, False]
    assert turns[0].text == "कुल free checked su"


def test_tick_overlap_barge_in_marks_turn_without_any_chunk_signal():
    # Realtime cancellation: the provider cancels generation and every chunk
    # it emitted stays "active" (plain content, no undelivered gold chunks,
    # no raw_data.was_truncated) — the tick timeline is the ONLY record of
    # the cut. The tick-overlap detector must mark the turn for the judge.
    from tau2.data_model.message import Tick, UserMessage
    from tau2.judges.base import INTERRUPTED_TURN_MARKER
    from tau2.judges.nativeness.harness import (
        build_agent_corpus,
        build_agent_turns,
    )

    def agent_tick(i, content, uid):
        # 1s ticks -> a 2-tick barge-in grace window.
        return Tick(
            tick_id=i,
            timestamp=f"t{i}",
            tick_duration_seconds=1.0,
            agent_chunk=AssistantMessage(
                role="assistant", content=content, utterance_ids=[uid]
            ),
        )

    ticks = [
        agent_tick(0, "Su transcripción ", "u1"),
        agent_tick(1, "combinada es", "u1"),
        agent_tick(2, "tá lis", "u1"),
        agent_tick(3, "claro, ¿algo más?", "u2"),
    ]
    # Caller speech overlaps the tail of u1's agent speech.
    ticks[2].user_chunk = UserMessage(
        role="user", content="espera", contains_speech=True
    )
    sim = SimulationRun(
        id="s1",
        task_id="t1",
        start_time="x",
        end_time="y",
        duration=4.0,
        termination_reason="agent_stop",
        ticks=ticks,
    )
    turns = build_agent_turns(sim)
    # u1 is cut (tick overlap), u2 airs after the caller finished: mapping is
    # by tick-range overlap onto the barged span, never by ordinal.
    assert [t.interrupted for t in turns] == [True, False]
    assert turns[0].judged_text.endswith(INTERRUPTED_TURN_MARKER)
    assert turns[1].judged_text == "claro, ¿algo más?"
    # Raw text and corpus exports never carry the marker.
    assert turns[0].text == "Su transcripción combinada está lis"
    assert INTERRUPTED_TURN_MARKER not in build_agent_corpus(sim)


def test_caller_speech_after_agent_finished_never_marks_the_turn():
    # No-barge-in regression for the tick-overlap union: normal turn-taking
    # (caller starts on a tick with no agent speech, past the grace tail)
    # must keep interrupted=False with no chunk signal present.
    from tau2.data_model.message import Tick, UserMessage
    from tau2.judges.nativeness.harness import build_agent_turns

    ticks = [
        Tick(
            tick_id=i,
            timestamp=f"t{i}",
            tick_duration_seconds=1.0,
            agent_chunk=AssistantMessage(
                role="assistant", content=text, utterance_ids=["u1"]
            ),
        )
        for i, text in enumerate(["Hola, ", "¿en qué puedo ayudarle?"])
    ]
    for i in (5, 6):  # past the 2-tick grace tail of the agent's speech
        ticks.append(
            Tick(
                tick_id=i,
                timestamp=f"t{i}",
                tick_duration_seconds=1.0,
                user_chunk=UserMessage(
                    role="user", content="quiero cambiar mi vuelo", contains_speech=True
                ),
            )
        )
    sim = SimulationRun(
        id="s1",
        task_id="t1",
        start_time="x",
        end_time="y",
        duration=7.0,
        termination_reason="agent_stop",
        ticks=ticks,
    )
    turns = build_agent_turns(sim)
    assert [t.interrupted for t in turns] == [False]


def test_no_opportunity_yields_none_score():
    # An unregistered language has no selected factors, so nothing fires.
    sim = _sim("bonjour, comment puis-je vous aider")
    task = _task("greet the user")
    info = evaluate_nativeness(sim, task, "fr", "latn", settings=NO_JUDGE)
    assert info is not None
    assert info.score is None
    assert not [c for c in info.factor_checks if c.outcome in (PASS, FAIL)]
