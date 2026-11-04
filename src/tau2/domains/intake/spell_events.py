# Copyright Sierra
"""Deterministic spell-out detection for the intake agent-directed arm.

When the simulated caller actually renders a character-by-character
spell-out, the run stamps a ``spell_out`` event into the sim's effect
timeline (``tau2.data_model.audio_effects``), recording which callback field
was spelled and how many spell tokens were spoken. Detection runs on the
EXACT pre-synthesis utterance text — the user side is simulated, so the text
is ground truth — and can be compared with the caller's ``note_spell_request`` tool records.

Mechanics (fixed, versioned — bump :data:`SPELL_EVENT_DETECTION_VERSION` on
any change):

- A SPELL RUN is a maximal sequence of >= 2 spell tokens separated by commas
  (the fixed convention the voice guidelines mandate for character-by-
  character spell-outs: ``"J, O, H, N"``, ``"five, five, five"``). A spell
  token is a single alphanumeric character or one of the digit words
  zero/oh/one..nine.
- A run is attributed to the entity field whose VALUE SKELETON contains the
  run's character skeleton. Skeletons are alphanumeric-only, casefolded,
  with digit words folded to digits; both the spoken form and the dual-form
  written value of the field's payload are matched, so orthography variants
  and relative-date payloads both attribute. Ties go to the longest matching
  skeleton, then first field in payload order. Unattributed runs (the agent
  spelling its own words back, say) stamp nothing.
- Runs for the same field within one utterance merge into ONE detection with
  ``letters_spoken`` summed.
- RESTARTS (v1.1.0, the ``spell_correction`` complication's construct): a run
  that attributes to no field directly is counted as a FALTERED ATTEMPT of a
  field when the same utterance also carries a direct (containment) run for
  that field and the faltered run's skeleton is exactly ONE EDIT (one
  substituted or one dropped character) away from a prefix of that field's
  value skeleton. The direct run stays authoritative for attribution; the
  faltered attempt's tokens ADD to ``letters_spoken`` (restarts are
  re-delivery effort) and increment the detection's ``restarts`` count.

Stated limits: a spell-out rendered without the comma convention is missed
(under-count, never an over-count), and the ``doubled_letters`` complication
("double five") collapses two tokens into words this tokenizer does not
expand, under-counting ``letters_spoken`` on those calls. A faltered attempt
whose complete restart lands in a LATER utterance is missed (the pairing is
per utterance), and a correct-prefix restart ("J, O, H" then "J, O, H, N")
counts its letters via ordinary containment without incrementing
``restarts``.
"""

import re
from typing import Annotated, Dict, List

from pydantic import Field

from tau2.data_model.audio_effects import SpellOutDetection
from tau2.data_model.tasks import Task
from tau2.domains.intake.utils import spoken_form
from tau2.utils.pydantic_utils import BaseModelNoExtra

#: Version of the detection rules. Bump on any change to tokenization,
#: attribution, or merging.
#: 1.1.0: restart handling — an unattributed run one edit away from a prefix
#: of a directly-attributed field's skeleton counts as a faltered attempt
#: (letters added, ``restarts`` incremented). Catalog v2.4.0
#: (spell_correction) companion.
#: 1.2.0: time dual-form payloads match spoken expressions and full written
#: anchors, excluding the private annotation from counted spell-out units.
SPELL_EVENT_DETECTION_VERSION = "1.2.0"

# Digit words the spell-out convention voices for digits ("oh" for zero is a
# stock convention and a scripted spelling_style). Case-insensitive.
_DIGIT_WORDS: Dict[str, str] = {
    "zero": "0",
    "oh": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}

_TOKEN = r"(?:[A-Za-z0-9]|" + "|".join(_DIGIT_WORDS) + r")"

# A maximal comma-separated run of >= 2 spell tokens. Token boundaries are
# guarded so a run never starts or ends inside an ordinary word.
_SPELL_RUN_RE = re.compile(
    rf"(?<![A-Za-z0-9])({_TOKEN}(?:\s*,\s*{_TOKEN})+)(?![A-Za-z0-9])",
    re.IGNORECASE,
)

_TOKEN_RE = re.compile(rf"^{_TOKEN}$", re.IGNORECASE)

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def _token_char(token: str) -> str:
    """One skeleton character for one spell token."""
    lowered = token.lower()
    if lowered in _DIGIT_WORDS:
        return _DIGIT_WORDS[lowered]
    if _TOKEN_RE.match(token):
        return lowered
    raise ValueError(f"Not a spell token: {token!r}")


def value_skeleton(text: str) -> str:
    """The match skeleton of a value or utterance: digit words folded to
    digits, then alphanumerics only, casefolded."""
    folded = _WORD_RE.sub(
        lambda match: _DIGIT_WORDS.get(match.group(0).lower(), match.group(0)),
        text,
    )
    return _NON_ALNUM_RE.sub("", folded.lower())


class SpellEventDetector(BaseModelNoExtra):
    """Attributes spell runs in an utterance to tracked entity fields.

    ``field_skeletons`` maps each field name to the skeletons of the match
    texts of its payload (spoken form first, dual-form written value second
    when present). Built once per simulation from the task's ``set_entities``
    payloads (:func:`build_spell_event_detector`)."""

    field_skeletons: Annotated[
        Dict[str, List[str]],
        Field(
            description="field name -> non-empty match skeletons of its "
            "gold payload (spoken form, plus the dual-form written value)."
        ),
    ]

    def detect(self, text: str) -> List[SpellOutDetection]:
        """Every field-attributed spell-out in one utterance text.

        Deterministic and pure, in two passes. Pass 1 attributes each run by
        skeleton CONTAINMENT (the run appears inside a field's value
        skeleton). Pass 2 revisits the unattributed runs: one that is exactly
        one edit away from a prefix of a pass-1 field's skeleton is that
        field's FALTERED ATTEMPT (the ``spell_correction`` restart construct)
        — its tokens add to ``letters_spoken`` and increment ``restarts``.
        Runs for the same field merge into one detection, in utterance
        order; fields are returned in payload order."""
        if not text:
            return []
        runs: List[tuple[int, str, List[str], str]] = []  # (pos, run, tokens, skel)
        for match in _SPELL_RUN_RE.finditer(text):
            run = match.group(1)
            tokens = [tok.strip() for tok in run.split(",")]
            runs.append(
                (
                    match.start(),
                    run,
                    tokens,
                    "".join(_token_char(tok) for tok in tokens),
                )
            )
        attributed: Dict[int, str] = {}  # run index -> field
        restart_of: Dict[int, str] = {}  # run index -> field (faltered attempt)
        for index, (_pos, _run, _tokens, run_skeleton) in enumerate(runs):
            field = self._attribute(run_skeleton)
            if field is not None:
                attributed[index] = field
        direct_fields = set(attributed.values())
        for index, (_pos, _run, _tokens, run_skeleton) in enumerate(runs):
            if index in attributed:
                continue
            field = self._attribute_restart(run_skeleton, direct_fields)
            if field is not None:
                restart_of[index] = field
        merged: Dict[str, List[str]] = {}
        letters: Dict[str, int] = {}
        restarts: Dict[str, int] = {}
        for index, (_pos, run, tokens, _skel) in enumerate(runs):
            field = attributed.get(index) or restart_of.get(index)
            if field is None:
                continue
            merged.setdefault(field, []).append(run)
            letters[field] = letters.get(field, 0) + len(tokens)
            if index in restart_of:
                restarts[field] = restarts.get(field, 0) + 1
        return [
            SpellOutDetection(
                field=field,
                letters_spoken=letters[field],
                spelled_text=" / ".join(field_runs),
                restarts=restarts.get(field, 0),
            )
            for field, field_runs in merged.items()
        ]

    def _attribute(self, run_skeleton: str) -> str | None:
        """The field whose value skeleton contains the run, or None.

        Longest matching skeleton wins; ties go to payload order."""
        best_field: str | None = None
        best_len = -1
        for field, skeletons in self.field_skeletons.items():
            for skeleton in skeletons:
                if run_skeleton in skeleton and len(skeleton) > best_len:
                    best_field = field
                    best_len = len(skeleton)
        return best_field

    def _attribute_restart(self, run_skeleton: str, direct_fields: set) -> str | None:
        """The directly-attributed field this run is a faltered attempt of.

        A faltered attempt is one edit — one substituted character or one
        dropped character — away from a PREFIX of the field's value skeleton,
        exactly the ``spell_correction`` generator's construct. Only fields
        with a direct run in the SAME utterance qualify (the restart pairs
        with its complete re-delivery). Longest matching skeleton wins."""
        if len(run_skeleton) < 2:
            return None
        best_field: str | None = None
        best_len = -1
        for field, skeletons in self.field_skeletons.items():
            if field not in direct_fields:
                continue
            for skeleton in skeletons:
                if len(skeleton) > best_len and _is_faltered_prefix(
                    run_skeleton, skeleton
                ):
                    best_field = field
                    best_len = len(skeleton)
        return best_field


def _is_faltered_prefix(run_skeleton: str, value_skeleton: str) -> bool:
    """Whether ``run_skeleton`` is one edit away from a prefix of
    ``value_skeleton``.

    Exactly the ``spell_correction`` generator's two operators:

    - SUBSTITUTE: same length as the prefix, differing in exactly one
      position (Hamming distance 1 to ``value_skeleton[:len(run)]``).
    - DROP: the prefix one character longer, with exactly one character
      removed (``run == prefix[:i] + prefix[i+1:]`` for some ``i``).

    An exact prefix returns False — that run attributes by ordinary
    containment in pass 1 and never reaches this check.
    """
    m = len(run_skeleton)
    if m <= len(value_skeleton):
        prefix = value_skeleton[:m]
        if sum(a != b for a, b in zip(run_skeleton, prefix)) == 1:
            return True
    if m + 1 <= len(value_skeleton):
        prefix = value_skeleton[: m + 1]
        for i in range(m + 1):
            if run_skeleton == prefix[:i] + prefix[i + 1 :]:
                return True
    return False


def entity_match_texts(payload: str) -> List[str]:
    """The match texts of one ``set_entities`` payload: its spoken form,
    plus the dual-form written value when the payload carries one."""
    spoken = spoken_form(payload)
    texts = [spoken]
    if spoken != payload:
        # Dual form: "<spoken> (the real date this refers to, ...: <value>)".
        # The written value is the payload minus the fixed template around it;
        # re-deriving through the template regex keeps the contract in one
        # place (tau2.domains.intake.utils).
        from tau2.domains.intake.utils import (
            _ENTITY_DUAL_FORM_PATTERN,
            _ENTITY_TIME_DUAL_FORM_PATTERN,
        )

        match = _ENTITY_DUAL_FORM_PATTERN.match(
            payload
        ) or _ENTITY_TIME_DUAL_FORM_PATTERN.match(payload)
        if match is None:  # pragma: no cover - spoken_form and the
            raise ValueError(  # pattern can only disagree on a code bug
                f"Payload {payload!r} has a spoken form but no dual-form match"
            )
        texts.append(match.group("value"))
    return texts


def build_spell_event_detector(task: Task) -> SpellEventDetector:
    """The detector for one intake task, from its ``set_entities`` payloads.

    Fails loud when the task carries no ``set_entities`` initialization
    action or a payload yields an empty skeleton — a detector that silently
    tracks nothing would zero the ground truth without a trace."""
    actions = (task.initial_state and task.initial_state.initialization_actions) or []
    entities: Dict[str, str] | None = None
    for action in actions:
        if action.func_name == "set_entities":
            entities = dict(action.arguments["entities"])
            break
    if not entities:
        raise ValueError(
            f"Task {task.id} has no set_entities initialization action; "
            "cannot build a spell-event detector"
        )
    field_skeletons: Dict[str, List[str]] = {}
    for field, payload in entities.items():
        skeletons = [
            skeleton
            for skeleton in (value_skeleton(t) for t in entity_match_texts(payload))
            if skeleton
        ]
        if not skeletons:
            raise ValueError(
                f"Task {task.id}: entity {field!r} payload {payload!r} "
                "yields no match skeleton"
            )
        field_skeletons[field] = skeletons
    return SpellEventDetector(field_skeletons=field_skeletons)


NATIVE_SPELL_EVENT_DETECTION_VERSION = "intake-native-graphemes-v1"


class NativeSpellEventDetector(BaseModelNoExtra):
    """Count comma-delimited Unicode graphemes or reviewed digit names.

    Character descriptions and unseparated read-outs are intentionally not
    inferred. Counts are graphemes, not comparable English letter counts.
    """

    field_skeletons: Annotated[
        dict[str, list[str]],
        Field(
            description="NFC written entity skeletons; combining marks retain identity."
        ),
    ]
    digit_words: Annotated[
        dict[str, str],
        Field(description="Reviewed spoken names mapped to ASCII digits."),
    ]

    def detect(self, text: str) -> list[SpellOutDetection]:
        """Attribute explicit grapheme runs; never infer a spelling from prose."""
        import regex

        candidates_by_run = []
        for match in regex.finditer(
            r"(?<!\S)([^\s,;:.!?]+(?:,\s*[^\s,;:.!?]+)+)", text
        ):
            tokens = [token.strip() for token in match.group(1).split(",")]
            units = []
            for token in tokens:
                normalized = _native_skeleton(token)
                if normalized in self.digit_words:
                    units.append(self.digit_words[normalized])
                elif len(regex.findall(r"\X", normalized)) == 1:
                    units.append(normalized)
                else:
                    break
            else:
                skeleton = "".join(units)
                candidates = [
                    (len(value), field)
                    for field, values in self.field_skeletons.items()
                    for value in values
                    if skeleton and skeleton in value
                ]
                candidates_by_run.append(
                    (match.group(1), units, max(candidates)[1] if candidates else None)
                )
        direct_fields = {
            field for _, _, field in candidates_by_run if field is not None
        }
        totals, runs, restarts = {}, {}, {}
        for run, units, field in candidates_by_run:
            restarted = False
            if field is None:
                candidates = [
                    (len(value), candidate)
                    for candidate, values in self.field_skeletons.items()
                    if candidate in direct_fields
                    for value in values
                    if _is_faltered_prefix(units, regex.findall(r"\X", value))
                ]
                if candidates:
                    field = max(candidates)[1]
                    restarted = True
            if field is not None:
                totals[field] = totals.get(field, 0) + len(units)
                restarts[field] = restarts.get(field, 0) + int(restarted)
                runs.setdefault(field, []).append(run)
        return [
            SpellOutDetection(
                field=field,
                letters_spoken=total,
                spelled_text=" / ".join(runs[field]),
                restarts=restarts[field],
            )
            for field, total in totals.items()
        ]


def _native_skeleton(value: str) -> str:
    import unicodedata

    return "".join(
        character
        for character in unicodedata.normalize("NFC", value.casefold())
        if unicodedata.category(character)[0] in {"L", "M", "N"}
    )


def build_native_spell_event_detector(
    task: Task, digit_words: dict[str, str]
) -> NativeSpellEventDetector:
    """Build only from this task's private values and reviewed locale digit names."""
    entities = next(
        action.arguments["entities"]
        for action in task.initial_state.initialization_actions
        if action.func_name == "set_entities"
    )
    return NativeSpellEventDetector(
        field_skeletons={
            field: [_native_skeleton(value) for value in entity_match_texts(payload)]
            for field, payload in entities.items()
        },
        digit_words={
            _native_skeleton(word): digit for digit, word in digit_words.items()
        },
    )
