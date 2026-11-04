"""Locale-facing clock notation, separate from canonical H:MM AM/PM answers."""

import re
from typing import Literal

TIME_DISPLAY_VERSION = "intake-local-time-display-v2"
TimeDisplayFormat = Literal[
    "en", "hi", "hi-latn", "es", "pt", "zh", "zh-latn", "ko", "ko-latn"
]
LANGUAGE_TIME_FORMATS: dict[str, TimeDisplayFormat] = {
    "hi": "hi",
    "es": "es",
    "pt": "pt",
    "zh": "zh",
    "ko": "ko",
}
_CANONICAL_TIME = re.compile(r"(1[0-2]|[1-9]):([0-5][0-9]) (AM|PM)")
TIME_CALLER_GUIDANCE = (
    "Some time look-ups give a way to say the time followed by an annotation "
    "beginning 'the real time this refers to, from your records:'. You know the "
    "complete time in that annotation, including its day period. Initially say "
    "only the expression before the annotation, without volunteering AM/PM or "
    "morning/afternoon/evening/night. If asked which day period, the exact time, "
    "or its written form, answer accurately from the complete time in the "
    "annotation. For a read-back, use that complete time to confirm or correct "
    "the caller. Never read the annotation or explain the lookup aloud."
)
_HOURS = {
    "en": (
        "twelve",
        "one",
        "two",
        "three",
        "four",
        "five",
        "six",
        "seven",
        "eight",
        "nine",
        "ten",
        "eleven",
    ),
    "hi": (
        "बारह",
        "एक",
        "दो",
        "तीन",
        "चार",
        "पाँच",
        "छह",
        "सात",
        "आठ",
        "नौ",
        "दस",
        "ग्यारह",
    ),
    "hi-latn": (
        "baarah",
        "ek",
        "do",
        "teen",
        "chaar",
        "paanch",
        "chhah",
        "saat",
        "aath",
        "nau",
        "das",
        "gyaarah",
    ),
    "es": (
        "doce",
        "una",
        "dos",
        "tres",
        "cuatro",
        "cinco",
        "seis",
        "siete",
        "ocho",
        "nueve",
        "diez",
        "once",
    ),
    "pt": (
        "doze",
        "uma",
        "duas",
        "três",
        "quatro",
        "cinco",
        "seis",
        "sete",
        "oito",
        "nove",
        "dez",
        "onze",
    ),
}


def locale_time_format(
    language: str, representation: str = "native"
) -> TimeDisplayFormat | None:
    """Select local notation, keeping romanized review text in Latin script."""
    if representation == "romanized" and language in {"hi", "zh", "ko"}:
        return {"hi": "hi-latn", "zh": "zh-latn", "ko": "ko-latn"}[language]
    return LANGUAGE_TIME_FORMATS.get(language)


def display_time(
    value: str, style: TimeDisplayFormat, *, include_period: bool = True
) -> str:
    """Render the same clock instant; omitted periods retain a twelve-hour clock.

    This is the complete written anchor, separate from idiomatic spoken variants.
    The simulator reads it in the conversation language. No timezone conversion
    or change to the benchmark's frozen difficulty assignment occurs.
    """
    match = _CANONICAL_TIME.fullmatch(value)
    if not match:
        raise ValueError(f"Expected canonical H:MM AM/PM time, got {value!r}")
    hour, minute = int(match[1]), int(match[2])
    hour24 = hour % 12 + (12 if match[3] == "PM" else 0)
    if style == "en":
        return value if include_period else f"{hour}:{minute:02}"
    if style in {"hi", "hi-latn"}:
        period = (
            0
            if hour24 < 4 or hour24 >= 20
            else 1
            if hour24 < 12
            else 2
            if hour24 < 16
            else 3
        )
        labels = (
            ("रात", "सुबह", "दोपहर", "शाम")
            if style == "hi"
            else ("raat", "subah", "dopahar", "shaam")
        )
        suffix = "बजे" if style == "hi" else "baje"
        clock = f"{hour}:{minute:02} {suffix}"
        return f"{labels[period]} {clock}" if include_period else clock
    if style in {"es", "pt"}:
        if style == "es":
            period = (
                "de la madrugada"
                if hour24 < 6
                else "de la mañana"
                if hour24 < 12
                else "de la tarde"
                if hour24 < 20
                else "de la noche"
            )
            if hour24 in {0, 12}:
                period = "de la noche" if hour24 == 0 else "del mediodía"
            clock = f"{hour}:{minute:02}"
        else:
            period = (
                "da madrugada"
                if hour24 < 6
                else "da manhã"
                if hour24 < 12
                else "da tarde"
                if hour24 < 18
                else "da noite"
            )
            clock = f"{hour}h{minute:02}"
        return f"{clock} {period}" if include_period else clock
    if style in {"zh", "zh-latn"}:
        period = (
            0
            if hour24 < 6
            else 1
            if hour24 < 12
            else 2
            if hour24 == 12
            else 3
            if hour24 < 18
            else 4
        )
        labels = (
            ("凌晨", "上午", "中午", "下午", "晚上")
            if style == "zh"
            else ("lingchen", "shangwu", "zhongwu", "xiawu", "wanshang")
        )
        if hour24 == 0:
            labels = ("午夜", *labels[1:]) if style == "zh" else ("wuye", *labels[1:])
        if style == "zh":
            clock = f"{hour}点" + (f"{minute}分" if minute else "整")
            return labels[period] + clock if include_period else clock
        clock = f"{hour} dian" + (f" {minute} fen" if minute else " zheng")
        return f"{labels[period]} {clock}" if include_period else clock
    if style in {"ko", "ko-latn"}:
        if style == "ko":
            period = (
                "밤"
                if hour24 == 0
                else "낮"
                if hour24 == 12
                else "오전"
                if hour24 < 12
                else "오후"
            )
            clock = f"{hour}시" + (f" {minute}분" if minute else "")
        else:
            period = (
                "bam"
                if hour24 == 0
                else "nat"
                if hour24 == 12
                else "ojeon"
                if hour24 < 12
                else "ohu"
            )
            clock = f"{hour}si" + (f" {minute}bun" if minute else "")
        return f"{period} {clock}" if include_period else clock
    raise ValueError(f"Unknown time display format: {style}")


def display_time_payload(value: str, style: TimeDisplayFormat) -> str:
    """Return a caller expression plus the known complete time when idiomatic."""
    from tau2.domains.intake.utils import render_time_dual_form

    if not _CANONICAL_TIME.fullmatch(value):
        return value
    full = display_time(value, style)
    spoken = time_expression(value, style)
    return render_time_dual_form(spoken, full) if spoken else full


def time_expression(value: str, style: TimeDisplayFormat) -> str | None:
    """Render conventional spoken clock expressions without supplying the day period.

    Korean uses minutes-before rather than a literal quarter calque; Brazilian
    Portuguese uses quinze. Hindi half past one/two have dedicated forms.
    """
    match = _CANONICAL_TIME.fullmatch(value)
    if not match:
        raise ValueError(f"Expected canonical H:MM AM/PM time, got {value!r}")
    hour, minute = int(match[1]), int(match[2])
    if style == "es" and minute == 0:
        article = "la" if hour == 1 else "las"
        return f"{article} {_HOURS[style][hour % 12]}"
    if minute not in {15, 30, 45}:
        return None
    if style in _HOURS:
        current = _HOURS[style][hour % 12]
        following = _HOURS[style][(hour + 1) % 12]
    if style == "en":
        return (
            f"quarter past {current}"
            if minute == 15
            else f"half past {current}"
            if minute == 30
            else f"quarter to {following}"
        )
    if style in {"hi", "hi-latn"}:
        native = style == "hi"
        if minute == 15:
            clock = f"{'सवा' if native else 'sava'} {current}"
        elif minute == 45:
            clock = f"{'पौने' if native else 'paune'} {following}"
        elif hour == 1:
            clock = "डेढ़" if native else "derh"
        elif hour == 2:
            clock = "ढाई" if native else "dhai"
        else:
            clock = f"{'साढ़े' if native else 'saadhe'} {current}"
        return f"{clock} {'बजे' if native else 'baje'}"
    if style == "es":
        target = current if minute != 45 else following
        article = "la" if target == "una" else "las"
        suffix = (
            "y cuarto"
            if minute == 15
            else "y media"
            if minute == 30
            else "menos cuarto"
        )
        return f"{article} {target} {suffix}"
    if style == "pt":
        if minute == 45:
            article = "a" if following == "uma" else "as"
            return f"quinze para {article} {following}"
        return f"{current} e {'quinze' if minute == 15 else 'meia'}"
    if style in {"zh", "zh-latn"}:
        fraction = (
            {15: "一刻", 30: "半", 45: "三刻"}
            if style == "zh"
            else {15: "yi ke", 30: "ban", 45: "san ke"}
        )
        return (
            f"{hour}点{fraction[minute]}"
            if style == "zh"
            else f"{hour} dian {fraction[minute]}"
        )
    if style in {"ko", "ko-latn"}:
        if minute == 15:
            return None
        if minute == 30:
            return f"{hour}시 반" if style == "ko" else f"{hour}si ban"
        following_hour = hour % 12 + 1
        return (
            f"{following_hour}시 15분 전"
            if style == "ko"
            else f"{following_hour}si 15bun jeon"
        )
    raise ValueError(f"Unknown time display format: {style}")


def omission_instruction(canonical: str, style: TimeDisplayFormat) -> str:
    """Omit the local day-period marker without revealing it through a 24h clock."""
    partial = time_expression(canonical, style) or display_time(
        canonical, style, include_period=False
    )
    return (
        "Scripted complication (lazy omission): the first time you give the requested "
        f'time, give only "{partial}" in the conversation language, without its '
        "morning/afternoon/evening/night marker. Supply that missing day period only "
        "when the caller asks for it or asks you to confirm it. You are omitting only "
        "the day period, not changing the time. Use the time from get_entity."
    )
