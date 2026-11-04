"""Reject source-style intake address component ordering at authoring time."""

import re
from collections import Counter

from tau2.multilingual.localization import LocalizedForms, SourceEntry


def reorder_intake_address(value: str, language: str, task_id: str) -> str:
    """Apply proven component permutations, never infer an address from punctuation.

    The selected task ID bounds the reviewed repair. Unknown layouts stay unchanged;
    no province, postcode, unit or premise identifier is invented or discarded.
    """
    parts = value.split(", ")
    prefix = []
    if language == "pt" and task_id in {
        f"intake_addresses_hard_{i:02d}" for i in (2, 3, 4, 5, 6, 10)
    }:
        if (
            len(parts) >= 5
            and re.fullmatch(r"nº \d+", parts[1])
            and re.match(r"(?:Rua|Viela|Travessa|Passagem)\b", parts[2])
            and re.match(r"(?:Lote|Casa|Unidade|Galpão|Apto)\b", parts[0])
        ):
            prefix = [2, 1, 0]
    elif language == "es":
        rules = {
            "intake_addresses_hard_01": ([1, 2, 0], 5),
            "intake_addresses_hard_03": ([3, 4, 0, 1, 2], 8),
            "intake_addresses_hard_04": ([2, 3, 0, 1], 6),
            "intake_addresses_hard_05": ([2, 3, 0, 1], 5),
            "intake_addresses_hard_10": ([2, 3, 0, 1], 5),
        }
        rule = rules.get(task_id)
        if rule is not None:
            order, count = rule
            if (
                len(parts) == count
                and re.match(r"(?:Calle|Calleja|Travesía)\b", parts[order[0]])
                and re.fullmatch(r"\d+", parts[order[1]])
            ):
                prefix = order
        if prefix and task_id == "intake_addresses_hard_03":
            # Exact audited locality pair: official Asturias/Correos directories
            # identify 33940 with El Entrego. Only move these existing tokens.
            if parts[5:] != [
                "El Entrego",
                "San Martín del Rey Aurelio",
                "33940 Asturias",
            ]:
                raise ValueError(
                    "Spanish Hard 03 locality changed; review the bound repair"
                )
            parts[5], parts[7] = "33940 El Entrego", "Asturias"
    if not prefix:
        return value
    ordered = [parts[i] for i in prefix] + parts[len(prefix) :]
    result = ", ".join(ordered)
    if Counter(re.findall(r"\w+", value)) != Counter(re.findall(r"\w+", result)):
        raise ValueError("Intake address repair changed component tokens")
    return result


def validate_intake_address_forms(
    entry: SourceEntry, forms: LocalizedForms, language: str | None
) -> None:
    """Reject recurrence of the exact selected address-order defects at authoring."""
    if language not in {"es", "pt"}:
        return
    if not (
        entry.context.startswith("intake field:home_address;")
        or "dependency=decoys;" in entry.context
    ):
        return
    tasks = set(re.findall(r"intake_addresses_(?:easy|hard)_[0-9]{2}", entry.context))
    if not tasks:
        return
    for representation in ("native", "romanized"):
        value = getattr(forms, representation)
        values = value if isinstance(value, list) else [value]
        for text in values:
            if not isinstance(text, str):
                continue
            if any(
                reorder_intake_address(text, language, task_id) != text
                for task_id in tasks
            ):
                raise ValueError(
                    "Intake address uses source-style unit/street or postcode ordering; use native component order while preserving every identifier"
                )
