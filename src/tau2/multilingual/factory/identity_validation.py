"""Check that a staged task's scenario agrees with its initialized caller records."""

import re

from tau2.data_model.tasks import Task

_WRONG_ZIP = re.compile(
    r"If asked for your zip code, say that it is (?P<wrong>\S+) first "
    r"\(common mistake\), then correct yourself and say (?P<correct>\S+) if an error is found\."
)


def _validate_telecom_callers(task: Task, agent_data: dict) -> None:
    from tau2.domains.telecom.data_model import Customer

    customers = [
        Customer.model_validate(row) for row in agent_data.get("customers", [])
    ]
    if not customers:
        raise ValueError(
            "materialize canonical telecom caller records before localization"
        )
    for action in task.initial_state.initialization_actions or []:
        if action.func_name != "set_user_info":
            continue
        name = action.arguments["name"]
        if sum(row.full_name == name for row in customers) != 1:
            raise ValueError("ambiguous telecom caller identity")
        if name not in task.user_scenario.instructions.known_info:
            raise ValueError("telecom scenario name disagrees with caller DB")


def _validate_wrong_zip(task: Task) -> None:
    """A wrong-then-correct ZIP scenario needs two distinct goldens in that order."""
    instructions = task.user_scenario.instructions
    condition = _WRONG_ZIP.fullmatch(
        getattr(instructions, "task_instructions", "") or ""
    )
    if not condition:
        return
    lookups = [
        action
        for action in (task.evaluation_criteria.actions or [])
        if action.name == "find_user_id_by_name_zip"
    ]
    if len(lookups) != 2 or condition["wrong"] == condition["correct"]:
        raise ValueError("wrong/correct ZIP scenario requires two distinct lookups")
    if [action.arguments.get("zip") for action in lookups] != [
        condition["wrong"],
        condition["correct"],
    ]:
        raise ValueError("wrong/correct ZIP goldens disagree with scenario")


def validate_identity_consistency(task: Task, domain: str) -> None:
    """Reject staged tasks whose scenario identity disagrees with initialized data."""
    state = task.initial_state
    agent_data = (
        (state.initialization_data.agent_data or {})
        if state and state.initialization_data
        else {}
    )
    if domain == "telecom":
        _validate_telecom_callers(task, agent_data)
    elif domain == "retail" and agent_data.get("users"):
        _validate_wrong_zip(task)
