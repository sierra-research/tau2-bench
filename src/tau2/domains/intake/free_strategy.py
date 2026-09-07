# Copyright Sierra
"""The ``intake_free`` domain: free-strategy entity capture.

The canonical intake desk mandates a rigid capture protocol (spell every
value letter by letter, log, read back, get an explicit yes). This variant
is the INFORMED-FREE benchmark mode: the agent is told the record must end
up correct and that an unsure value should be verified (spelling named once
as a worked example) — when and how it elicits, confirms, or spells is
entirely its own choice — and the CALLER'S effort is measured
deterministically from the user side (spell-event stamping in
``tau2.domains.intake.spell_events``; ledger in
``tau2.metrics.caller_effort``). Design doc:
docs/designs/intake-free-strategy.md.

Registered as its OWN domain so the canonical ``intake`` surface stays
byte-stable (the ``intake_staged`` pattern): same DB / user DB / frozen
200-task set; the policy is a NEW file (``free_strategy_policy.md``); the
toolkits subclass the canonical ones. Differences, all owner-decided:

- ``submit_fields`` DROPS the ``confirmed_with_user`` parameter entirely
  (schema = record_id + fields); write semantics are unchanged.
- The user simulator's task instructions are swapped (on deep copies — the
  tasks.json file is never touched) for the free-strategy caller frame: the
  callee never VOLUNTEERS a spelling or written form, spells accurately from
  their ``get_entity`` payloads when asked, and records measurement notes
  through two new silent user-side tools (``note_spell_request``,
  ``note_readback``) that never reach the agent-visible stream.
- Golden ``submit_fields`` actions drop ``confirmed_with_user`` so the
  DB-hash reward's golden-action replay matches the variant schema.

VOICE ONLY: ``get_environment`` refuses ``channel="text"`` — the text policy
and text channel belong to the canonical domain and are untouched.

Era note: free-strategy results are never comparable to protocol-era intake
runs; every run records ``domain_name="intake_free"`` in its Info.
"""

from typing import Dict, Optional

from tau2.data_model.tasks import Task
from tau2.domains.intake.call_frame import USER_TASK_INSTRUCTIONS
from tau2.domains.intake.data_model import IntakeDB
from tau2.domains.intake.environment import get_tasks as intake_get_tasks
from tau2.domains.intake.environment import (
    get_tasks_split as intake_get_tasks_split,
)
from tau2.domains.intake.tools import IntakeTools
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

# 1.2.0: verification sentence added (owner, two-mode decision 2026-09-02):
# the benchmark settles on TWO modes — full protocol (canonical intake, the
# scaffolded reference) and informed-free (this policy, the benchmark). The
# informed-free policy tells the agent that exactness is what matters and
# names verification, with spelling as one worked example — the STRATEGY
# (when to verify, on which fields) stays the agent's own, and that is what
# the arm measures.
# 1.1.0: format-conversion line removed entirely (owner, Q1): the field
# description delivered by get_callback_order IS the format spec.
FREE_STRATEGY_POLICY_VERSION = "1.2.0"

FREE_STRATEGY_POLICY_PATH = INTAKE_DATA_DIR / "free_strategy_policy.md"


# The sentence the verification guard is anchored to. Placement is
# load-bearing (the verification sentence elaborates "how is up to you"), so
# loading fails loud if the policy file ever drops or rewords either sentence
# instead of silently shipping a drifted benchmark policy.
_POLICY_VERIFICATION_ANCHOR = "How you get each value right is up to you."

# The canonical policy's verification sentence (FREE_STRATEGY_POLICY_VERSION
# 1.2.0), byte-for-byte as it appears in free_strategy_policy.md right after
# the anchor — the "informed" in informed-free. The one canonical mode-B
# policy has no switches (owner, 2026-09-02: the prompt-informativeness
# ablation machinery was measured once and removed; results in the design
# doc).
POLICY_VERIFICATION_SENTENCE = (
    "Getting each value exactly right is what matters: if you are not sure "
    "you heard a value exactly, verify it with the person before recording "
    "it — for example by asking them to spell it letter by letter."
)


def load_free_strategy_policy() -> str:
    """The canonical intake_free agent policy, verbatim, drift-guarded.

    Fails loud unless the file carries the anchor sentence plus the
    verification sentence exactly once — the policy is versioned prose, and
    a silent reword must never ship under the same version constant.
    """
    policy = load_file(FREE_STRATEGY_POLICY_PATH)
    canonical_block = f"{_POLICY_VERIFICATION_ANCHOR} {POLICY_VERIFICATION_SENTENCE}"
    if policy.count(canonical_block) != 1:
        raise ValueError(
            "free_strategy_policy.md no longer carries the anchor plus the "
            "verification sentence exactly once; the policy drifted — bump "
            "FREE_STRATEGY_POLICY_VERSION and retune the guard instead of "
            "silently re-basing it"
        )
    return policy


# Version of the variant caller frame (the swapped task_instructions plus the
# note-tool contract). Bump on any retune.
FREE_STRATEGY_FRAME_VERSION = "1.0.0"

# The ONE fixed user-sim task-instructions block every intake_free task
# carries (machine, not scripts). It keeps the canonical caller frame —
# "Hello?" opener, cooperative consent, answer-only-what-is-asked, get_entity
# grounding, tool privacy, dual-form lead, confirm/correct read-backs — and
# changes exactly three things: the callee never VOLUNTEERS a spelling or
# written form (overriding the outbound guidelines' offer-to-spell nudge on
# repeats), spells accurately from the get_entity payload when asked, and
# records the two measurement notes (note_spell_request / note_readback)
# silently when those events happen.
FREE_STRATEGY_USER_TASK_INSTRUCTIONS = (
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
    "reads a value back, silently call note_readback with that field's "
    "name — i_affirmed true when the read-back matched the exact text from "
    "your look-up, false when it did not — then confirm it out loud if it "
    "matched and correct them if it did not. End the call once the caller "
    "says they have everything they need."
)


class FreeStrategyIntakeTools(IntakeTools):
    """The intake toolkit under the free-strategy arm.

    Same tool names as the canonical desk; only ``submit_fields`` changes —
    its ``confirmed_with_user`` self-report attestation is dropped from the
    schema entirely (the arm mandates no read-backs to attest to). The write
    semantics delegate to the canonical write, so folding and the
    exactly-the-missing-fields gates stay identical."""

    @is_tool(ToolType.WRITE)
    def submit_fields(self, record_id: str, fields: Dict[str, str]) -> str:
        """
        Write all collected field values to the record at once. Submit
        exactly once, after you have collected every missing field: the
        submission must contain exactly the record's missing fields — no
        others — and a value for each.

        Values are stored in canonical form (phones as their digit sequence,
        codes uppercase without separators, names case/diacritic-folded, dates
        as YYYY-MM-DD), so formatting variants of a correctly captured value
        are equivalent — capture the value itself accurately.

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
        # The canonical write, minus the attestation: confirmed_with_user was
        # trace-only (never written to the DB), so passing a constant keeps
        # the DB writes byte-identical to the canonical desk's.
        return IntakeTools.submit_fields(
            self, record_id=record_id, fields=fields, confirmed_with_user=False
        )


class FreeStrategyIntakeUserTools(IntakeUserTools):
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
        answer: i_affirmed is true when what they read back matched the exact
        text from your records and you confirmed it, false when it did not
        match and you had to correct them. This is a private note: the caller
        never hears about it.

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
    user_db: Optional[IntakeUserDB] = None,
    solo_mode: bool = False,
    channel: str = "voice",
) -> Environment:
    """Build the free-strategy intake environment. VOICE ONLY.

    ``channel`` mirrors the canonical intake seam (the runner passes the
    run's actual modality); a text run fails loud — the free-strategy arm is
    a spoken-capture experiment, and the text policy/channel belong to the
    canonical domain, untouched.

    The policy is the ONE canonical mode-B policy, verbatim and
    drift-guarded — see :func:`load_free_strategy_policy`.
    """
    if solo_mode:
        raise ValueError("Solo mode not supported for intake_free domain")
    if channel != "voice":
        raise ValueError(
            f"intake_free is voice-only (got channel {channel!r}): the text "
            "channel belongs to the canonical intake domain"
        )
    if db is None:
        db = IntakeDB.load(INTAKE_DB_PATH)
    if user_db is None:
        user_db = IntakeUserDB.load(INTAKE_USER_DB_PATH)
    return Environment(
        domain_name="intake_free",
        policy=load_free_strategy_policy(),
        tools=FreeStrategyIntakeTools(db),
        user_tools=FreeStrategyIntakeUserTools(user_db),
    )


def _free_strategy_task_variant(task: Task) -> Task:
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
            "intake_free swap is defined only for the structured call frame"
        )
    if instructions.task_instructions != USER_TASK_INSTRUCTIONS:
        raise ValueError(
            f"Task {task.id}: stored task_instructions do not match the "
            "canonical intake call frame; the frozen set drifted — re-decide "
            "the free-strategy swap instead of silently re-basing it"
        )
    instructions.task_instructions = FREE_STRATEGY_USER_TASK_INSTRUCTIONS
    criteria = variant.evaluation_criteria
    for action in (criteria.actions or []) if criteria else []:
        if action.name == "submit_fields" and action.arguments:
            action.arguments.pop("confirmed_with_user", None)
    return variant


def get_tasks(task_split_name: Optional[str] = "base") -> list[Task]:
    """The frozen intake tasks under the free-strategy rewrite.

    Same set, same ids, same splits as the canonical domain — only the two
    in-memory rewrites of :func:`_free_strategy_task_variant` differ."""
    return [
        _free_strategy_task_variant(task) for task in intake_get_tasks(task_split_name)
    ]


def get_tasks_split() -> dict[str, list[str]]:
    """The canonical intake splits (task ids are shared verbatim)."""
    return intake_get_tasks_split()
