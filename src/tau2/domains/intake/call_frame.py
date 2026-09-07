# Copyright Sierra
"""The fixed call frame for the intake domain (design v4, outbound).

Versioned in-code constants (machine, not scripts -- the Phase C generator
renders every task through these, and the canonical tasks are test-asserted
against them):

- :data:`OPENER_TEMPLATE` -- the deterministic agent opener each task supplies
  (design doc section 3): injected as the seeded first assistant turn in text
  runs and unsynthesized (silence audio) via ``create_initial_message(...)``
  in voice runs, through ``Task.agent_opener``.
- :func:`render_user_scenario` -- the ONE user-sim scenario template every
  task shares (section 5.5): callee persona, the "Hello?" first-turn rule,
  cooperative disposition, answer-only-when-asked, and tool-only grounding:
  every factual value reaches the sim through ``get_entity``, and the
  scenario lists the exact field names the tool expects (names only, never
  values) so the sim never improvises an argument. No per-task prose.
"""

from tau2.data_model.tasks import StructuredUserInstructions, UserScenario

CALL_FRAME_VERSION = "4.5.0"

OPENER_TEMPLATE = "Hello, this is {org} calling."

USER_PERSONA_TEMPLATE = (
    "You are {callee_full_name}. You recently submitted a form with "
    "{org_name}, and you are going about your day when your phone rings."
)

USER_REASON_FOR_CALL = (
    "You are not making this call: you are answering your phone. You do not "
    "yet know who is calling or why."
)

USER_KNOWN_INFO_TEMPLATE = (
    "You are {callee_full_name}. You recently submitted a form with {org_name}."
)

USER_UNKNOWN_INFO = (
    "You do not recall any of the record's details from memory: any name, "
    "number, date, time, amount, email, or code you produce without looking "
    "it up WILL BE WRONG. Your get_entity tool is the only source of true "
    "values."
)

# Appended to unknown_info: the exact get_entity field names, so the sim
# never guesses an argument string (mirrors how telecom's per-question tool
# NAMES carry the grounding map). Names only, never values.
USER_FETCHED_FIELDS_TEMPLATE = (
    "Your records hold these look-up fields; call get_entity with the field "
    "name exactly as written here:\n{field_lines}"
)

USER_TASK_INSTRUCTIONS = (
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
    "have not yet fetched it. Your get_entity tool is private: "
    "never mention the tool, your lookup, or internal field names to the "
    "caller. If a get_entity call errors and the error lists your record's "
    "field names, silently call it again with the listed field name; the "
    "caller only ever hears the value. When the caller asks you to spell a "
    "value or confirm its exact written form (letter by letter, a digit "
    "versus a spelled-out word, dot versus underscore), answer from the "
    "exact text your get_entity tool returned. Some look-ups return the "
    "way you say a value plus the real written value it refers to: lead "
    "with the way you say it, and give the real written value when the "
    "caller asks for the exact date or its written form. When the caller reads a "
    "value back, confirm it if it matches that exact text and correct them "
    "if it does not. End the call once the caller says they have everything "
    "they need."
)


def render_opener(org_name: str) -> str:
    """The deterministic opener for a task (fills ``Task.agent_opener``)."""
    if not org_name.strip():
        raise ValueError("org_name must not be blank")
    return OPENER_TEMPLATE.format(org=org_name)


def render_user_scenario(
    callee_full_name: str,
    org_name: str,
    fetched_fields: list[str],
) -> UserScenario:
    """The one fixed user-sim scenario for a task (design doc section 5.5).

    ``fetched_fields`` names the task's fields -- the exact ``get_entity``
    argument strings, never their values, which reach the sim only through
    the tool.
    """
    if not callee_full_name.strip():
        raise ValueError("callee_full_name must not be blank")
    if not org_name.strip():
        raise ValueError("org_name must not be blank")
    if not fetched_fields:
        raise ValueError("fetched_fields must not be empty")
    for field in fetched_fields:
        if not field.strip():
            raise ValueError("fetched field names must not be blank")
    known_info = USER_KNOWN_INFO_TEMPLATE.format(
        callee_full_name=callee_full_name, org_name=org_name
    )
    field_lines = "\n".join(f"- {field}" for field in sorted(fetched_fields))
    unknown_info = (
        f"{USER_UNKNOWN_INFO}\n\n"
        f"{USER_FETCHED_FIELDS_TEMPLATE.format(field_lines=field_lines)}"
    )
    return UserScenario(
        persona=USER_PERSONA_TEMPLATE.format(
            callee_full_name=callee_full_name, org_name=org_name
        ),
        instructions=StructuredUserInstructions(
            domain="intake",
            reason_for_call=USER_REASON_FOR_CALL,
            known_info=known_info,
            unknown_info=unknown_info,
            task_instructions=USER_TASK_INSTRUCTIONS,
        ),
    )
