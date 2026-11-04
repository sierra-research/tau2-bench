"""Locale-facing date surfaces, separate from unchanged ISO tool submissions."""

import re
from datetime import date
from typing import Literal

DATE_DISPLAY_VERSION = "intake-local-date-display-v1"
DateDisplayFormat = Literal[
    "dmy-hyphen", "dmy-slash", "ymd-hanzi", "ymd-hangul", "ymd-hyphen"
]
LANGUAGE_DATE_FORMATS: dict[str, DateDisplayFormat] = {
    "hi": "dmy-hyphen",
    "es": "dmy-slash",
    "pt": "dmy-slash",
    "zh": "ymd-hanzi",
    "ko": "ymd-hangul",
}


def locale_date_format(
    language: str, representation: str = "native"
) -> DateDisplayFormat | None:
    """Use local date order and keep romanized metadata free of native-script labels."""
    if representation == "romanized" and language in {"zh", "ko"}:
        return "ymd-hyphen"
    return LANGUAGE_DATE_FORMATS.get(language)


def display_date(
    value: str, style: DateDisplayFormat, *, include_year: bool = True
) -> str:
    """Render the same ISO calendar date using a locale's ordinary component order."""
    parsed = date.fromisoformat(value)
    if style == "ymd-hyphen":
        return value if include_year else value[5:]
    if style in {"dmy-hyphen", "dmy-slash"}:
        separator = "-" if style == "dmy-hyphen" else "/"
        components = [f"{parsed.day:02}", f"{parsed.month:02}"]
        if include_year:
            components.append(str(parsed.year))
        return separator.join(components)
    if style == "ymd-hanzi":
        return (
            f"{parsed.year}年" if include_year else ""
        ) + f"{parsed.month}月{parsed.day}日"
    return (
        f"{parsed.year}년 " if include_year else ""
    ) + f"{parsed.month}월 {parsed.day}일"


def display_date_payload(value: str, style: DateDisplayFormat) -> str:
    """Format plain date lookups and the exact-date anchor of relative lookups."""
    from tau2.domains.intake.utils import _ENTITY_DUAL_FORM_PATTERN, render_dual_form

    if re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
        return display_date(value, style)
    match = _ENTITY_DUAL_FORM_PATTERN.fullmatch(value)
    if match and re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", match["value"]):
        return render_dual_form(match["spoken"], display_date(match["value"], style))
    return value


def matches_date_payload(value: str, canonical: str, style: DateDisplayFormat) -> bool:
    """Compare caller-facing date data to one exact canonical ISO answer."""
    from tau2.domains.intake.utils import _ENTITY_DUAL_FORM_PATTERN

    match = _ENTITY_DUAL_FORM_PATTERN.fullmatch(value)
    return (match["value"] if match else value) == display_date(canonical, style)


def omission_instruction(canonical: str, style: DateDisplayFormat) -> str:
    """Keep the existing omit-year complication in the caller's local date order."""
    partial = display_date(canonical, style, include_year=False)
    return (
        "Scripted complication (lazy omission): the first time you give the requested "
        f'date, leave off the year and give only "{partial}" in the conversation language. '
        "Supply the year only when the caller asks for it or asks you to confirm it. "
        "You are omitting only the year, not changing the date. Use the date from get_entity."
    )
