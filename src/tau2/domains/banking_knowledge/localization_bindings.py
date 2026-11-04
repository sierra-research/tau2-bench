"""Strict references for incidental source entities not stored as DB fields."""

from tau2.multilingual.localization import LocalizationMap


def validate_prose_bindings(proposal: LocalizationMap) -> None:
    """Require source coverage, exact reuse, and actual generated-form provenance."""
    entries = {entry.id: entry for entry in proposal.entries}
    seen = set()
    for binding in proposal.prose_bindings:
        if not isinstance(binding.forms.native, str) or not isinstance(
            binding.forms.romanized, str
        ):
            raise ValueError("Prose binding forms must be strings")
        key = (binding.entry_id, binding.source.casefold())
        if key in seen:
            raise ValueError("Duplicate prose binding")
        seen.add(key)
        entry = entries.get(binding.entry_id)
        if (
            entry is None
            or entry.category != "prose"
            or not isinstance(entry.source, str)
            or binding.source not in entry.source
        ):
            raise ValueError("Prose binding must match its exact source entry")
        if binding.source_part is not None and binding.kind != "reuse":
            raise ValueError("Address source_part requires a reuse binding")
        if binding.kind == "protected":
            if (
                binding.forms.native != binding.source
                or binding.forms.romanized != binding.source
            ):
                raise ValueError("Protected prose surface must remain unchanged")
        elif binding.kind == "reuse":
            referenced = entries.get(binding.source_entry_id)
            if referenced is None:
                raise ValueError(
                    "Reused prose binding must retain exact referenced forms"
                )
            expected = (
                address_component(referenced, binding.source_part)
                if binding.source_part
                else referenced.forms
            )
            if binding.forms != expected:
                raise ValueError(
                    "Reused prose binding must retain exact referenced forms"
                )
        elif binding.kind == "reviewed":
            if binding.review is None:
                raise ValueError(
                    "Reviewed binding requires exact human review evidence"
                )
            binding.review.require(binding.forms)
        elif binding.provenance is None:
            raise ValueError("Generated prose binding requires generation provenance")


def bind_explicit_person_aliases(proposal: LocalizationMap) -> LocalizationMap:
    """Resolve an explicit daughter's given-name reference to one same-task identity.

    Reuse the complete localized name instead of inferring a culturally dependent
    given-name split. An ambiguous or absent source identity stays unresolved.
    """
    import re

    from tau2.domains.banking_knowledge.localization_prose import _tasks
    from tau2.multilingual.localization import ProseEntityBinding

    result = proposal.model_copy(deep=True)
    seen = {(b.entry_id, b.source.casefold()) for b in result.prose_bindings}
    for entry in result.entries:
        if entry.category != "prose" or not isinstance(entry.source, str):
            continue
        for match in re.finditer(r"daughter ([A-Z][a-z]+)", entry.source):
            source = match[1]
            candidates = [
                e
                for e in result.entries
                if e.category == "identity_name"
                and isinstance(e.source, str)
                and e.source.startswith(source + " ")
                and _tasks(entry) & _tasks(e)
            ]
            if len(candidates) == 1 and (entry.id, source.casefold()) not in seen:
                ref = candidates[0]
                result.prose_bindings.append(
                    ProseEntityBinding(
                        entry_id=entry.id,
                        source=source,
                        forms=ref.forms,
                        kind="reuse",
                        source_entry_id=ref.id,
                        reason="banking-explicit-person-alias-v1: explicit daughter reference matches one same-task source identity; reuse complete localized name without guessing name parts.",
                    )
                )
                seen.add((entry.id, source.casefold()))
    return result


def address_component(entry, part):
    """Read semantic locality components; formatted punctuation is not a schema."""
    from tau2.multilingual.address_format import format_locality
    from tau2.multilingual.localization import LocalizedForms

    if part not in {"address_city", "address_region", "address_city_region"}:
        raise ValueError("Unknown address component")
    if entry.category != "address":
        raise ValueError("Address component requires an address source entry")
    address = entry.forms.address
    if address is None:
        raise ValueError(
            "Explicit address components required on the referenced address entry"
        )
    values = {}
    for representation in ("native", "romanized"):
        parts = getattr(address, representation)
        city, region = parts.city, parts.region
        values[representation] = (
            city
            if part == "address_city"
            else region
            if part == "address_region"
            else format_locality(parts, address.language, representation)
        )
        if not values[representation]:
            raise ValueError(
                "Requested locality is absent from address component evidence"
            )
    return LocalizedForms(
        **values,
        english="Exact locality component derived from the referenced localized address; no new entity authored.",
    )
