"""Versioned authoring exceptions for selected intake speech stimuli."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from tau2.multilingual.factory.localization_reuse import ReviewedReuseSource
    from tau2.multilingual.localization import (
        LocalizationEntry,
        LocalizationSource,
        LocalizedForms,
        SourceEntry,
    )


INTAKE_AUTHORING_VERSION = "intake-authoring-v7"
INTAKE_AUTHORING_PROMPT = """Intake-specific exceptions to the general regional prose rules above:
Return exactly request_contract.required_count entries, in the listed required_entry_ids
order. Each requested id appears exactly once. completed_mappings is read-only
reference context, never additional output. Do not omit preserved requested values.
The selected intake inventory includes explicit speech stimuli. In entries whose
context identifies caller_greeting or agent_opener_template, write the spoken
utterance in the target language, retaining exactly one {org} in the opener.
These two fields are utterances, not English simulator instructions; their
text_language is the target language. Keep clarification_guidance, spelling_styles,
omission_instruction, reason and reference explanations in English, but localize
any quoted spoken examples and use natural script-specific clarification conventions.
The selected pronunciations explicitly request TTS-ready distorted native tokens;
author those fields even though unrelated task prose must not gain TTS coaching.
For localized pronunciation records, reference is the exact corresponding written
token, not a syllable respelling: never add stress capitals, hyphenated phonetics
or pronunciation notation. Native Hindi, Korean and Mandarin token, reference and distorted must use
Devanagari, Hangul and Hanzi respectively, with no Latin letters. The separate romanized representation is metadata;
its reference likewise equals its plain romanized token. Preserve real distortions.
Never add unsolicited spelling to the agent prompt. Empty style/omission strings
mean that a source phenomenon is inapplicable. Preserve the frozen family/tier,
numeric medication strengths and unit strings, required date/time representations,
protocol keys and checksum validity. Phone/email/VIN/member-ID protocol values remain ASCII. Local license plates may use normal local script in native form (Hangul in Korea, Hanzi in China), with the same identity romanized separately.
Pronunciation boundary must be exactly the protocol value word or grapheme. Preserve distinctive capture hazards and
wrong-value distinctions. Dependency entries carry their exact selected task and original written value.
Use the completed mapping of that written value as authoritative: pronunciation
tokens must be literal substrings of its corresponding native/romanized form;
decoys must be distinct plausible capture errors of that same localized answer.
Do not replace a measured value merely to match a generated dependency.
All content remains draft pending native and audio review.
Preserve reviewed_source_gender on each named person, separately for the callee
and measured second guest. Never change gender to match a donor identity. Each
name decoy preserves its own source naming gender convention, which can differ
from the primary person's gender; ambiguous source names remain explicitly ambiguous.
When an owner name changes, update its owner-linked email components and dependent
pronunciations, decoys and clarification examples consistently.
Use native address order: Chinese and Korean administrative areas go from large
to small; Spanish and Brazilian Portuguese street names precede premise numbers.
In Spanish put the postal code before its locality. Keep apartment, portal and
building numbers distinct from the street number, including in hard addresses
and intentionally wrong decoys; retain every identifier and difficulty feature.
"""


def prompt_version(domain: str, base: str) -> str:
    """Record the domain-specific authoring contract in checkpoints and provenance."""
    return f"{base}+{INTAKE_AUTHORING_VERSION}" if domain == "intake" else base


def reuse_intake_callees(
    source: "LocalizationSource", reference: "ReviewedReuseSource", language: str
) -> list["LocalizationEntry"]:
    """Reuse approved incidental identities without assigning measurement hazards."""
    from tau2.multilingual.factory.localization_reuse import banking_reuse_candidates
    from tau2.multilingual.localization import LocalizationEntry

    if source.domain != "intake":
        raise ValueError("Intake callee reuse requires the intake inventory")
    donors = banking_reuse_candidates(reference, language)
    unique = {}
    for donor in donors:
        unique.setdefault(donor.name.native, donor)
    entries = sorted(
        (entry for entry in source.entries if entry.category == "identity_name"),
        key=lambda entry: entry.id,
    )
    import re

    result = []
    used = set()
    for entry in entries:
        required = re.search(
            r"reviewed_source_gender=(male|female)(?:;|$)", entry.context
        )
        if required is None:
            continue
        donor = next(
            (
                item
                for item in unique.values()
                if item.name.native not in used
                and item.name_gender is not None
                and item.name_gender.target_gender in {required.group(1), "unisex"}
            ),
            None,
        )
        if donor is None:
            # Unknown donor gender is not permission to change a known source
            # identity. Ordinary versioned authoring fills this un-reused slot.
            continue
        donor.name_gender.require(donor.name)
        used.add(donor.name.native)
        from tau2.multilingual.localization import NameGenderEvidence

        evidence = NameGenderEvidence(
            **{
                **donor.name_gender.model_dump(),
                "source_gender": required.group(1),
                "source_basis": "reviewed_metadata",
                "source_evidence": f"Frozen intake source bank: {entry.source!r}; {required.group(0)}.",
            }
        )
        result.append(
            LocalizationEntry(
                **entry.model_dump(),
                forms=donor.name,
                name_gender=evidence,
                operation="adapt",
                notes="Reused approved retail callee identity; incidental context only. New intake use remains pending native review.",
                reuse=donor.lineage.model_copy(
                    update={"source_entry_ids": donor.name_ids}
                ),
            )
        )
    return result


def validate_pronunciation_script(
    values, language: str | None, representation: str
) -> None:
    """Reject romanized native speech fields and stress respellings, not ASCII protocols."""
    import unicodedata

    if language in {None, "en"}:
        return
    for value in values:
        if (
            language in {"hi", "ko", "zh"}
            and representation == "native"
            and any(
                "LATIN" in unicodedata.name(character, "")
                for field in ("token", "reference", "distorted")
                for character in value[field]
            )
        ):
            raise ValueError(
                f"Intake native {dict(hi='Hindi', ko='Korean', zh='Mandarin')[language]} pronunciation fields cannot contain Latin letters"
            )
        if value["reference"] != value["token"]:
            raise ValueError(
                "Intake pronunciation reference must equal its exact token; no stress respelling"
            )


def _selected_decoy_family(
    entry: "SourceEntry",
) -> Literal["dates", "times", "medications", "phones"] | None:
    """Resolve constrained decoy ownership from the exact selected-stimulus role."""
    import re

    role = entry.context.split(";", 1)[0]
    match = re.fullmatch(
        r"intake stimulus:intake_(dates|times|medications|phones)_(?:easy|hard)_[0-9]+:decoys",
        role,
    )
    if match is None or not any(
        point.document_id == "intake_localization"
        and len(point.path) == 3
        and point.path[0] == "stimuli"
        and isinstance(point.path[1], int)
        and point.path[2] == "decoys"
        for point in entry.occurrences
    ):
        return None
    family = match.group(1)
    if family in {"medications", "phones"}:
        return family
    return "dates" if family == "dates" else "times"


def _validate_temporal_decoys(entry: "SourceEntry", forms: "LocalizedForms") -> None:
    """Keep stored wrong-answer protocol separate from later caller localization."""
    import re
    from datetime import date

    from tau2.domains.intake.time_display import display_time

    family = _selected_decoy_family(entry)
    if family not in {"dates", "times"}:
        return
    required = "YYYY-MM-DD" if family == "dates" else "H:MM AM/PM"
    for representation in ("native", "romanized"):
        values = getattr(forms, representation)
        if not isinstance(values, list):
            raise ValueError(
                f"Intake {entry.id} {representation} decoys must be an array of canonical {required} strings"
            )
        for value in values:
            try:
                if not isinstance(value, str):
                    raise ValueError("Temporal decoy is not a string")
                if family == "dates":
                    if re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value) is None:
                        raise ValueError("Noncanonical date syntax")
                    date.fromisoformat(value)
                else:
                    # Reuse exactly the canonical parser used by caller rendering.
                    display_time(value, "en")
            except ValueError as error:
                raise ValueError(
                    f"Intake {entry.id} {representation} {family} decoys require canonical {required} protocol strings; received {value!r}. Keep stored decoys in the canonical format in BOTH forms. Local date/time notation is rendered later for the caller, not authored into this array."
                ) from error


def _validate_medication_decoys(entry: "SourceEntry", forms: "LocalizedForms") -> None:
    """Preserve the strength and unit of each exact selected source decoy."""
    from tau2.domains.intake.localization import medication_strengths

    if _selected_decoy_family(entry) != "medications":
        return
    for representation in ("native", "romanized"):
        values = getattr(forms, representation)
        if (
            not isinstance(entry.source, list)
            or not isinstance(values, list)
            or len(values) != len(entry.source)
        ):
            raise ValueError(
                f"Intake {entry.id} {representation} medication decoys must preserve the source array shape"
            )
        for index, (source, target) in enumerate(
            zip(entry.source, values, strict=True)
        ):
            if not isinstance(source, str) or not isinstance(target, str):
                raise ValueError(
                    f"Intake {entry.id} {representation} medication decoy {index} must be a string"
                )
            if medication_strengths(source) != medication_strengths(target):
                raise ValueError(
                    f"Intake {entry.id} {representation} medication decoy {index} changes source strength or units: {source!r} became {target!r}. Preserve EACH source decoy's numeric strength and unit at its original array position; decimal comma and point are equivalent. Localize the medicine spelling without changing its dose."
                )


def _validate_phone_decoys(
    entry: "SourceEntry",
    forms: "LocalizedForms",
    completed: Sequence["LocalizationEntry"],
) -> None:
    """Preserve exact source main-number/extension length errors relative to its primary."""
    from tau2.multilingual.factory.intake_phone_invariants import (
        phone_decoy_length_evidence,
    )

    if _selected_decoy_family(entry) != "phones":
        return
    paths = [
        point.path[:-1]
        for point in entry.occurrences
        if point.document_id == "intake_localization"
        and len(point.path) == 3
        and point.path[0] == "stimuli"
        and point.path[2] == "decoys"
    ]
    primary = next(
        (
            item
            for item in completed
            if any(
                point.document_id == "intake_localization"
                and point.path[:-1] in paths
                and point.path[-1] == "written"
                for point in item.occurrences
            )
        ),
        None,
    )
    if primary is None:
        # Dependencies authored before their primary cannot be evaluated yet;
        # ordinary generation supplies the completed exact written mapping.
        return
    for representation in ("native", "romanized"):
        values = getattr(forms, representation)
        if (
            not isinstance(entry.source, list)
            or not isinstance(values, list)
            or len(entry.source) != len(values)
        ):
            raise ValueError("Phone decoys must retain their exact source array shape")
        for index, (source, current) in enumerate(
            zip(entry.source, values, strict=True)
        ):
            evidence = phone_decoy_length_evidence(
                primary.source, source, getattr(primary.forms, representation), current
            )
            if not evidence.preserves_source_error_shape:
                raise ValueError(
                    f"Intake {entry.id} {representation} phone decoy {index} changes the source-relative dialing/extension digit-count error. Source primary {primary.source!r}, source decoy {source!r}; completed primary {evidence.current_primary!r}, proposed decoy {current!r}. Preserve each source decoy's main-number length difference from the primary and its separate extension error shape. Equal-length source pairs require equal-length localized pairs; genuine source missing/extra digits remain intentional errors. Do not invent length errors from an audit claim."
                )


def validate_intake_forms(
    entry: "SourceEntry",
    forms: "LocalizedForms",
    completed: Sequence["LocalizationEntry"] = (),
    language: str | None = None,
) -> None:
    """Validate protocol answers and dependent pronunciation contracts before caching."""
    import re

    from tau2.multilingual.factory.intake_address_repair import (
        validate_intake_address_forms,
    )

    validate_intake_address_forms(entry, forms, language)
    _validate_temporal_decoys(entry, forms)
    _validate_medication_decoys(entry, forms)
    _validate_phone_decoys(entry, forms, completed)
    if any(
        f"dependency={kind};" in entry.context for kind in ("pronunciations", "decoys")
    ):
        from pydantic import TypeAdapter

        from tau2.domains.intake.localized import SelectedPronunciation

        occurrence = entry.occurrences[0]
        written = next(
            (
                item
                for item in completed
                if any(
                    point.document_id == occurrence.document_id
                    and point.path == [*occurrence.path[:-1], "written"]
                    for point in item.occurrences
                )
            ),
            None,
        )
        for representation in ("native", "romanized"):
            if "dependency=decoys;" in entry.context:
                if written is not None:
                    from tau2.domains.intake.folds import FoldKind, fold_value

                    kinds = {
                        "dates": FoldKind.DATE,
                        "times": FoldKind.TIME,
                        "phones": FoldKind.PHONE,
                        "emails": FoldKind.EMAIL,
                        "codes": FoldKind.CODE,
                        "person_names": FoldKind.NATIVE_NAME_DIGITS,
                        "coined": FoldKind.NATIVE_NAME,
                        "properties": FoldKind.NATIVE_NAME,
                        "medications": FoldKind.NATIVE_NAME_DIGITS,
                        "addresses": (
                            FoldKind.NATIVE_ADDRESS
                            if language not in {None, "en"}
                            else FoldKind.NAME_DIGITS
                        ),
                    }
                    family = (
                        entry.context.split("intake_", 1)[-1]
                        .split("_easy_", 1)[0]
                        .split("_hard_", 1)[0]
                    )
                    kind = kinds.get(family)
                    target = getattr(written.forms, representation)
                    if kind is not None and any(
                        fold_value(kind, value) == fold_value(kind, target)
                        for value in getattr(forms, representation)
                    ):
                        raise ValueError(
                            f"Intake {entry.id} {representation} decoy collapses to the completed correct answer {target!r}: received {getattr(forms, representation)!r}. Every decoy must have a different normalized value; retain the authoritative answer and change the decoys."
                        )
                continue
            validate_pronunciation_script(
                getattr(forms, representation), language, representation
            )
            pronunciations = TypeAdapter(list[SelectedPronunciation]).validate_python(
                getattr(forms, representation)
            )
            if written is not None:
                target = getattr(written.forms, representation)
                for pronunciation in pronunciations:
                    if pronunciation.token not in target:
                        raise ValueError(
                            f"Intake pronunciation token {pronunciation.token!r} must be an exact substring of the completed {representation} written value {target!r}"
                        )
    if not entry.context.startswith("intake field:") or not isinstance(
        entry.source, str
    ):
        return
    field = entry.context.split(";", 1)[0].removeprefix("intake field:")
    if field in {
        "contact_phone",
        "email",
        "contact_email",
        "vin",
        "member_id",
    }:
        if not all(
            isinstance(value, str) and value.isascii()
            for value in (forms.native, forms.romanized)
        ):
            raise ValueError(
                f"Intake protocol identifier {entry.id} requires ASCII forms"
            )
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}|\d{1,2}:\d{2} [AP]M", entry.source):
        if forms.native != entry.source or forms.romanized != entry.source:
            raise ValueError(
                f"Intake protocol answer {entry.id} requires native and romanized forms exactly equal to source {entry.source!r}; localize spoken delivery, not the required submission format"
            )


INTAKE_REPAIR_VERSION = "intake-candidate-repair-v5"
INTAKE_REPAIR_RULES = """Mechanical intake compilation repair; this is not native review or approval.
For checksummed VIN/member-ID written answers and decoy arrays, retain the exact source
protocol strings in BOTH native and romanized forms. Local license plates instead follow target-country script/format in native form and the same identity romanized; preserve their capture difficulty. Do not translate member-ID
prefixes or invent VINs/check digits. This preservation does not apply to local license plates.
For selected time-family written answers and decoy arrays, retain exact source
H:MM AM/PM protocol strings in BOTH forms. Update selected time style/omission
instructions to refer to those exact retained values and representations; no stale
24-hour or translated numeric display may survive in dependent instructions.
Speech can sound natural in the target language while tool submissions retain the
required stored representation. Preserve source date annotation delimiters and ISO
values; only the spoken phrase may change.
Protocol phone answers must use ASCII digits and punctuation, including ASCII
ext. before extensions. Retain the generated localized number's digits and country
convention, converting native digits to ASCII; do not replace it with a US number.
Refresh linked phone decoys and instructions to those exact representations.
Every decoy must differ after the field's actual normalization in BOTH forms.
For names, accents alone or homophones with identical romanization do not create
a distinct romanized decoy. For phones, formatting alone is not a different number.
Generate a different plausible capture error while preserving array length.
For Chinese name decoys, every array element in BOTH arrays must differ from the
completed written answer. Do not output the correct name as one of the decoys.
Choose a near-sounding syllable substitution with DIFFERENT PINYIN, not merely a
homophonous character replacement. Verify every romanized decoy against the exact
completed romanized answer before returning. Distinct Chinese glyphs with identical
pinyin are still invalid for this two-representation artifact. Use two different
wrong phonetic variants when the source decoy array has length two; neither may be
the gold answer. This constraint overrides copying the source error's exact type.
For selected pronunciation arrays, use literal substrings of the authoritative
completed written value in each representation, including its exact casing. Every
distortion must differ from its token and represent an audible error; romanization
must retain that distinction too. Never fill an inapplicable distortion with a no-op.
Each pronunciation reference must equal its exact token in that representation;
no stress capitals, syllable respellings or hyphenated phonetic guides. Native Hindi, Korean and Mandarin
token/reference/distorted fields contain no Latin letters. Keep actual distortions
in native orthography and romanized review metadata in its separate representation.
Preserve list lengths and field keys. Existing source/task/difficulty constraints
remain in force. Keep unselected correct mappings unchanged. These repairs remain
pending native review, particularly their phonetics and assigned difficulty.
"""
