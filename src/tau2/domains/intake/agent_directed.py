# Copyright Sierra
"""The intake domain's default agent-directed elicitation strategy.

The agent chooses when and how to elicit, verify, or request spelling. It is
told to capture every value accurately and to verify uncertain values. Caller
spelling effort is measured through ``tau2.domains.intake.spell_events``.

This voice-only strategy shares the database, caller records, and frozen
200-task set with the scaffolded ablation. Its policy is
``agent_directed_policy.md``. The differences are:

- ``submit_fields`` omits the scaffolded ``confirmed_with_user`` attestation;
  write semantics and final-record scoring are shared.
- Task copies use the agent-directed caller frame: callers never volunteer
  spelling, spell from ``get_entity`` when asked, and record silent
  ``note_spell_request`` and ``note_readback`` events.
- Golden submission actions omit ``confirmed_with_user`` to match the tool
  schema. Frozen task files remain unchanged.

Each run records its strategy in ``Info.elicitation``. See
``docs/elicitation.md`` for the run instructions and supported scope.
"""

import json
from typing import Dict, Optional

from tau2.data_model.tasks import Task
from tau2.domains.intake.call_frame import USER_TASK_INSTRUCTIONS
from tau2.domains.intake.data_model import IntakeDB
from tau2.domains.intake.environment import (
    english_time_display_task,
    load_frozen_tasks,
    populate_callback_order,
)
from tau2.domains.intake.environment import (
    get_tasks_split as intake_get_tasks_split,
)
from tau2.domains.intake.tools import ScaffoldedIntakeTools
from tau2.domains.intake.user_data_model import IntakeUserDB
from tau2.domains.intake.user_tools import IntakeUserTools
from tau2.domains.intake.utils import (
    INTAKE_DATA_DIR,
    INTAKE_DB_PATH,
    INTAKE_USER_DB_PATH,
)
from tau2.environment.environment import Environment
from tau2.environment.toolkit import ToolType, is_tool
from tau2.utils import load_file

# 1.4.0: allow required field-format conversion without inventing components.
# 1.3.0: the work order is supplied in the prompt before the call.
# 1.2.0: added the verification sentence while leaving strategy agent-directed.
# 1.1.0: field descriptions in the callback order own the required format.
AGENT_DIRECTED_POLICY_VERSION = "1.4.0"

AGENT_DIRECTED_POLICY_PATH = INTAKE_DATA_DIR / "agent_directed_policy.md"


# The sentence the verification guard is anchored to. Placement is
# load-bearing (the verification sentence elaborates "how is up to you"), so
# loading fails loud if the policy file ever drops or rewords either sentence
# instead of silently shipping a drifted benchmark policy.
_POLICY_VERIFICATION_ANCHOR = "How you get each value right is up to you."

# The verification sentence introduced in policy 1.2.0, byte-for-byte as it
# appears immediately after the anchor in agent_directed_policy.md.
POLICY_VERIFICATION_SENTENCE = (
    "Getting each value exactly right is what matters: if you are not sure "
    "you heard a value exactly, verify it with the person before recording "
    "it — for example by asking them to spell it letter by letter."
)


def load_agent_directed_policy() -> str:
    """The versioned agent-directed intake policy, verbatim, drift-guarded.

    Fails loud unless the file carries the anchor sentence plus the
    verification sentence exactly once — the policy is versioned prose, and
    a silent reword must never ship under the same version constant.
    """
    policy = load_file(AGENT_DIRECTED_POLICY_PATH)
    canonical_block = f"{_POLICY_VERIFICATION_ANCHOR} {POLICY_VERIFICATION_SENTENCE}"
    if policy.count(canonical_block) != 1:
        raise ValueError(
            "agent_directed_policy.md no longer carries the anchor plus the "
            "verification sentence exactly once; the policy drifted — bump "
            "AGENT_DIRECTED_POLICY_VERSION and retune the guard instead of "
            "silently re-basing it"
        )
    return policy


# Fixed caller instructions for every agent-directed task: cooperative consent,
# private tool grounding, spelling only when requested, transcript-aware
# read-backs, and silent spelling/read-back measurement notes. The aggregate
# ELICITATION_PROMPT_VERSION in prompts.py versions this caller frame.
AGENT_DIRECTED_USER_TASK_INSTRUCTIONS = (
    'Open your very first turn with exactly "Hello?" and nothing else: you '
    "are answering the phone and do not yet know who is calling. You are "
    "cooperative: once the caller explains what they need, help them "
    "complete it, and when they name you as the person the record belongs "
    "to, acknowledge it naturally. If the caller asks whether they may ask "
    "you about the form or its details, say yes and let them ask. Answer "
    "only what you are asked, and never "
    "volunteer information the caller has not asked for. When the caller "
    "asks for any factual detail, call get_entity for that field and answer "
    "only AFTER the tool returns; never speak a value in a turn where you "
    "have not yet fetched it. Your tools are private: "
    "never mention a tool, a lookup, or internal field names to the "
    "caller. If a tool call errors and the error lists your record's "
    "field names, silently call it again with the listed field name; the "
    "caller only ever hears the value. Say each value the natural way you "
    "would say it aloud, and never volunteer a spelling or a written form "
    "the caller did not ask for; if the caller asks you to repeat a value, "
    "repeat it the way you would say it aloud — spell only when asked to "
    "spell. When the caller asks you to spell a "
    "value or confirm its exact written form (letter by letter, a digit "
    "versus a spelled-out word, dot versus underscore), first silently call "
    "note_spell_request with that field's name, then answer from the "
    "exact text your get_entity tool returned. Some look-ups return the "
    "way you say a value plus the real written value it refers to: lead "
    "with the way you say it, and give the real written value when the "
    "caller asks for the exact date or its written form. When the caller "
    "reads a value back, compare it with the exact value from your lookup, "
    "keeping in mind that you are receiving a transcript. Ignore differences "
    "in spacing or punctuation that would not be audible—for example, spaces "
    "within a name—unless the caller explicitly spells or describes those "
    "details aloud. Silently call note_readback with that field’s name, "
    "setting i_affirmed to true when the spoken value matches and false "
    "otherwise; then confirm it or correct the spoken error. "
    "End the call once the caller "
    "says they have everything they need."
)


class AgentDirectedIntakeTools(ScaffoldedIntakeTools):
    """The intake toolkit under the agent-directed arm.

    Same tool names as the scaffolded desk; only ``submit_fields`` changes —
    its ``confirmed_with_user`` self-report attestation is dropped from the
    schema entirely (the arm mandates no read-backs to attest to). The write
    semantics delegate to the shared write, so folding and the
    exactly-the-missing-fields gates stay identical."""

    @is_tool(ToolType.WRITE)
    def submit_fields(self, record_id: str, fields: Dict[str, str]) -> str:
        """
        Write all collected field values to the record at once. Submit
        exactly once, after you have collected every missing field: the
        submission must contain exactly the record's missing fields — no
        others — and a value for each.

        Submit dates as YYYY-MM-DD and times as H:MM AM/PM, with no leading
        hour zero. Convert only information the person provided into these
        formats; ask for any missing or ambiguous components. Date and time
        strings are stored as submitted, apart from surrounding whitespace.
        Preserve decimal points within numbers and the person's unit wording.

        Phones are stored as digit sequences, codes uppercase without
        separators, and names case/diacritic-folded. Capture the full value
        accurately before applying the field's required format.

        Args:
            record_id: The record the callback order names.
            fields: All collected values, keyed by the missing field names
                from the callback order.

        Returns:
            Confirmation message listing the fields written.

        Raises:
            ValueError: If the record does not exist, a submitted field is not
                one of the record's missing fields, a missing field has no
                value in the submission, a value is empty, or the record's
                missing fields were already submitted.
        """
        return self._submit_fields(record_id, fields)


class AgentDirectedIntakeUserTools(IntakeUserTools):
    """The callee's toolkit plus the two silent measurement notes.

    The notes are measurement, not behavior: stateless (validated against the
    callee's known fields but written nowhere — the tool-call trace is the
    record, the ``log_capture`` pattern), and silent/invisible to the agent
    exactly like ``get_entity`` (user tool calls never carry audio and are
    stripped from stored chunks by the orchestrator)."""

    def _known_field(self, field: str) -> None:
        if field not in self.db.entities:
            known = ", ".join(sorted(self.db.entities)) or "(none)"
            raise ValueError(
                f"You have no record of a field named {field!r}. "
                f"Fields in your records: {known}"
            )

    @is_tool(ToolType.GENERIC)
    def note_spell_request(self, field: str) -> str:
        """
        Silently note that the caller asked you to spell this field's value
        (or to confirm its exact written form). Call this once per such
        request, before you spell. This is a private note: it is not an
        answer, and the caller never hears about it.

        Args:
            field: The field whose value you were asked to spell, exactly as
                named in your records.

        Returns:
            Confirmation that the note was recorded.

        Raises:
            ValueError: If you have no record of a field by that name.
        """
        self._known_field(field)
        return f"Noted spell request for {field}."

    @is_tool(ToolType.GENERIC)
    def note_readback(self, field: str, i_affirmed: bool) -> str:
        """
        Silently note that the caller read this field's value back to you,
        and whether you affirmed it. Call this once per read-back, before you
        answer: i_affirmed is true when the spoken value matched your lookup
        and you confirmed it, false when it did not match and you had to
        correct them. You are receiving a transcript: ignore differences in
        spacing or punctuation that would not be audible unless the caller
        explicitly spells or describes those details aloud. This is a private
        note: the caller never hears about it.

        Args:
            field: The field whose value was read back, exactly as named in
                your records.
            i_affirmed: True when you affirmed the read-back; false when you
                had to correct it.

        Returns:
            Confirmation that the note was recorded.

        Raises:
            ValueError: If you have no record of a field by that name.
        """
        self._known_field(field)
        outcome = "affirmed" if i_affirmed else "corrected"
        return f"Noted read-back of {field} ({outcome})."


# ---------------------------------------------------------------------------
# Environment, task loading, and splits
# ---------------------------------------------------------------------------


def get_environment(
    db: Optional[IntakeDB] = None,
    policy: str | None = None,
    user_db: Optional[IntakeUserDB] = None,
    solo_mode: bool = False,
    channel: str = "voice",
    task: Optional[Task] = None,
) -> Environment:
    """Build the voice-only agent-directed intake environment.

    The runner passes the run's channel; text runs require the scaffolded strategy.
    The policy is loaded verbatim with drift checks. When ``task`` is provided,
    its callback order is appended before agent construction.
    """
    if solo_mode:
        raise ValueError("Solo mode not supported for intake domain")
    if channel != "voice":
        raise ValueError(
            f"The agent_directed intake strategy is voice-only (got channel {channel!r}); "
            "text runs require the scaffolded strategy"
        )
    if db is None:
        db = IntakeDB.load(INTAKE_DB_PATH)
    if user_db is None:
        user_db = IntakeUserDB.load(INTAKE_USER_DB_PATH)
    environment = Environment(
        domain_name="intake",
        policy=policy if policy is not None else load_agent_directed_policy(),
        tools=AgentDirectedIntakeTools(db),
        user_tools=AgentDirectedIntakeUserTools(user_db),
    )
    if task is not None:
        populate_callback_order(environment, task)
    return environment


def _agent_directed_task_variant(task: Task) -> Task:
    """One frozen intake task rewritten, on a deep copy, for this arm.

    Two rewrites, both in memory (the frozen tasks.json is never touched):

    - the user-sim ``task_instructions`` swap, guarded by a drift check
      against the canonical call frame — if the frozen set's instructions
      ever change, this variant must be re-decided, not silently re-based;
    - golden ``submit_fields`` actions drop ``confirmed_with_user`` so the
      evaluator's golden-action replay matches the variant tool schema.
    """
    variant = task.model_copy(deep=True)
    instructions = variant.user_scenario.instructions
    if isinstance(instructions, str):
        raise ValueError(
            f"Task {task.id} carries free-form user instructions; the "
            "agent-directed adaptation requires the structured call frame"
        )
    stored, separator, clarification = instructions.task_instructions.partition(
        "\n\n# Intake written clarification\n"
    )
    before, after = USER_TASK_INSTRUCTIONS.split('"Hello?"', 1)
    if not stored.startswith(before) or not stored.endswith(after):
        raise ValueError(
            f"Task {task.id}: stored task_instructions drifted from the canonical intake call frame"
        )
    greeting = stored[len(before) : len(stored) - len(after)]
    if not isinstance(json.loads(greeting), str):
        raise ValueError("Intake greeting must be a quoted string")
    instructions.task_instructions = AGENT_DIRECTED_USER_TASK_INSTRUCTIONS.replace(
        '"Hello?"', greeting, 1
    )
    if separator:
        instructions.task_instructions += separator + clarification
    criteria = variant.evaluation_criteria
    for action in (criteria.actions or []) if criteria else []:
        if action.name == "submit_fields" and action.arguments:
            action.arguments.pop("confirmed_with_user", None)
    return variant


def get_tasks(task_split_name: Optional[str] = "base") -> list[Task]:
    """The frozen intake tasks under the agent-directed rewrite.

    Same set, same ids, same splits and English time surfaces as scaffolded
    mode, plus the in-memory strategy rewrites."""
    from tau2.domains.intake.scoring import project_scoring_folds

    return [
        _agent_directed_task_variant(
            english_time_display_task(project_scoring_folds(task))
        )
        for task in load_frozen_tasks(task_split_name)
    ]


def get_tasks_split() -> dict[str, list[str]]:
    """The canonical intake splits (task ids are shared verbatim)."""
    return intake_get_tasks_split()
