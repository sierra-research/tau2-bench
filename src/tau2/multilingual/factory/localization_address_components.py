"""Locale road-before-premise ordering for structured street components."""

import re
from collections import Counter


def repair_street_line(value: str, language: str, representation: str) -> str:
    """Reorder an unambiguous number-before-road defect, preserving all digits.

    This operates only on known street/address1 roles. It is not an address parser:
    units, postal codes, P.O. boxes and unsupported road shapes remain untouched.
    Latin forms retain native reference ordering, not international mailing order.
    """
    if language not in {"es", "pt", "zh", "ko"}:
        return value
    match = re.fullmatch(r"(\d+)\s*(?:号\s*|\s+)(.+)", value)
    if match is None:
        return value
    number, road = match.groups()
    if language == "es":
        if not re.match(
            r"(?:Calle|Carrer|Carretera|Avenida|Avinguda|Gran V[ií]a|Rúa|Paseo|Passeig|Camino|Ronda|Rambla|Plaza|Plaça)\b",
            road,
        ):
            return value
        result = f"{road} {number}"
    elif language == "pt":
        if not re.match(
            r"(?:Rua|Avenida|Travessa|Alameda|Estrada|Viela|Passagem|Praça|Rodovia|Ladeira|Servidão)\b",
            road,
        ):
            return value
        result = f"{road}, {number}"
    elif language == "zh":
        if representation == "native":
            if not re.fullmatch(r"[^\d,，]+[路街巷道]", road):
                return value
            result = f"{road}{number}号"
        else:
            road = re.sub(r"^Hao\s+", "", road, flags=re.IGNORECASE)
            if not re.search(r"(?:lu|jie|xiang|dao)$", road, re.IGNORECASE):
                return value
            result = f"{road} {number} Hao"
    else:
        suffix = r"(?:로|길)$" if representation == "native" else r"(?:ro|gil)$"
        if not re.search(suffix, road, re.IGNORECASE):
            return value
        result = f"{road} {number}"
    if Counter(re.findall(r"\d+", value)) != Counter(re.findall(r"\d+", result)):
        raise ValueError("Address repair changed numeric identifiers")
    return result
