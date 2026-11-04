"""Versioned intake scoring projection over frozen task artifacts."""

from tau2.data_model.tasks import Task
from tau2.domains.intake.folds import FoldKind

_DIGIT_NAME_FIELDS = frozenset({"second_guest_full_name"})
_ADDRESS_FIELD = "home_address"
_MEDICATION_FIELD = "current_medication"
_NATIVE_MEDICATION_FOLDS = {
    "hi": FoldKind.HINDI_MEDICATION,
    "ko": FoldKind.KOREAN_MEDICATION,
    "zh": FoldKind.CHINESE_MEDICATION,
    "pt": FoldKind.PORTUGUESE_MEDICATION,
    "es": FoldKind.SPANISH_MEDICATION,
}


def project_scoring_folds(task: Task) -> Task:
    """Apply field-specific scoring without changing published source task bytes.

    A generic NAME fold is shared by person names, medicine, coined names,
    properties, and addresses in the frozen task set. The measured field name
    distinguishes the seven families that accept alternate decimal digit
    scripts from coined/property names, whose digits remain identity-bearing.
    """
    projected = task.model_copy(deep=True)
    language = task.id.rsplit("_", 1)[-1]
    korean = task.id.endswith("_ko")
    chinese = task.id.endswith("_zh")
    portuguese = task.id.endswith("_pt")
    for action in projected.initial_state.initialization_actions:
        if action.func_name != "seed_record":
            continue
        for field in action.arguments["record"]["fields"]:
            kind = FoldKind(field["fold"])
            if field["name"] == _ADDRESS_FIELD:
                if kind in {FoldKind.NAME, FoldKind.NAME_DIGITS}:
                    field["fold"] = FoldKind.NAME_DIGITS.value
                elif kind in {FoldKind.NATIVE_NAME, FoldKind.NATIVE_NAME_DIGITS}:
                    field["fold"] = (
                        FoldKind.KOREAN_ADDRESS.value
                        if korean
                        else FoldKind.PORTUGUESE_ADDRESS.value
                        if portuguese
                        else FoldKind.NATIVE_ADDRESS.value
                    )
            elif field["name"] == _MEDICATION_FIELD:
                if kind in {FoldKind.NAME, FoldKind.NAME_DIGITS}:
                    field["fold"] = FoldKind.MEDICATION.value
                elif kind in {FoldKind.NATIVE_NAME, FoldKind.NATIVE_NAME_DIGITS}:
                    field["fold"] = _NATIVE_MEDICATION_FOLDS.get(
                        language, FoldKind.NATIVE_MEDICATION
                    ).value
            elif field["name"] in _DIGIT_NAME_FIELDS:
                if kind is FoldKind.NAME:
                    field["fold"] = FoldKind.NAME_DIGITS.value
                elif kind is FoldKind.NATIVE_NAME:
                    field["fold"] = (
                        FoldKind.KOREAN_NAME_DIGITS.value
                        if korean
                        else FoldKind.NATIVE_NAME_DIGITS.value
                    )
            elif korean and field["name"] == "destination_property":
                if kind is FoldKind.NATIVE_NAME:
                    field["fold"] = FoldKind.KOREAN_NAME.value
            if korean and field["name"] == "contact_phone" and kind is FoldKind.PHONE:
                field["fold"] = FoldKind.KOREAN_PHONE.value
            elif (
                chinese and field["name"] == "contact_phone" and kind is FoldKind.PHONE
            ):
                field["fold"] = FoldKind.CHINESE_PHONE.value
    return projected
