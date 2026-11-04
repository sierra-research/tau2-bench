"""Explicit name-order facts derived from structured caller records."""

import re
from typing import TYPE_CHECKING

from tau2.data_model.language import NameOrder

if TYPE_CHECKING:
    from tau2.data_model.tasks import Task
    from tau2.domains.retail.data_model import User
    from tau2.domains.retail.tools import RetailTools


def resolve_materialized_retail_caller(task: "Task") -> "User":
    """Read the single explicitly materialized retail caller, rejecting ambiguity."""
    from tau2.data_model.tasks import StructuredUserInstructions
    from tau2.domains.retail.data_model import User

    instructions = task.user_scenario.instructions
    if (
        not isinstance(instructions, StructuredUserInstructions)
        or instructions.domain != "retail"
    ):
        raise ValueError("caller resolution requires a structured retail task")
    state = task.initial_state
    patch = (
        state.initialization_data.agent_data
        if state and state.initialization_data
        else None
    )
    users = (patch or {}).get("users", {})
    if not isinstance(users, dict) or len(users) != 1:
        raise ValueError("retail task must materialize exactly one explicit caller")
    caller_id, record = next(iter(users.items()))
    caller = User.model_validate(record)
    if caller.user_id != caller_id:
        raise ValueError("materialized retail caller key differs from user_id")
    return caller


def clarify_retail_name(
    task: "Task", *, caller_id: str, caller: "User", name_order: NameOrder
) -> "Task":
    """Append retail name roles and spoken order once; reject stale annotations.

    The caller must come from the effective database under the explicit task caller
    ID. Display names play no part in determining the two name components.
    """
    from tau2.data_model.tasks import StructuredUserInstructions
    from tau2.domains.retail.data_model import User

    name_order = NameOrder(name_order)
    instructions = task.user_scenario.instructions
    if (
        isinstance(instructions, StructuredUserInstructions)
        and instructions.domain != "retail"
    ):
        raise ValueError("name facts require a retail task")
    caller = User.model_validate(caller.model_dump())
    if not caller_id or caller.user_id != caller_id:
        raise ValueError("retail caller ID differs from structured caller record")
    names = (caller.name.first_name, caller.name.last_name)
    if any(not name.strip() or "\n" in name or "\r" in name for name in names):
        raise ValueError("retail name components must be nonblank single-line facts")
    given, family = names
    separator = (
        ""
        if re.fullmatch(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+", given + family)
        else " "
    )
    ordered = (
        (family, given) if name_order == NameOrder.FAMILY_FIRST else (given, family)
    )
    full_name = separator.join(ordered)
    first_role, second_role = (
        ("last", "first") if name_order == NameOrder.FAMILY_FIRST else ("first", "last")
    )
    block = (
        f"Name details:\nYour first name is {given}, and your last name is {family}. "
        f"When asked for your full name, give your {first_role} name first and your "
        f"{second_role} name second: {full_name}."
    )
    known = (
        instructions if isinstance(instructions, str) else instructions.known_info or ""
    )
    if "Name details:" in known:
        if known.count("Name details:") != 1 or not (
            known == block or known.endswith("\n\n" + block)
        ):
            raise ValueError("retail name facts are stale, malformed or duplicated")
        return task
    result = task.model_copy(deep=True)
    clarified = known + "\n\n" + block if known else block
    if isinstance(instructions, str):
        result.user_scenario.instructions = clarified
    else:
        result.user_scenario.instructions.known_info = clarified
    return result


def declares_retail_name(known: str, pattern: str) -> bool:
    """Match a complete opening name declaration, excluding ID-only statements."""
    return (
        re.match(
            r"\A(?:You're|You are|Your name is|You name is) "
            + pattern
            + r"(?=[,.\n]|$| (?:and|in|with|living)\b)",
            known,
        )
        is not None
    )


def resolve_retail_caller(
    task: "Task",
    tools: "RetailTools",
    source_task: "Task",
    source_tools: "RetailTools",
    aliases: dict[str, str],
) -> "User":
    """Resolve a unique caller from explicit references and real identity lookups."""
    candidates = set()
    actions = task.evaluation_criteria.actions if task.evaluation_criteria else None
    for action in actions or []:
        user_id, order_id = (
            action.arguments.get("user_id"),
            action.arguments.get("order_id"),
        )
        if isinstance(user_id, str) and user_id in tools.db.users:
            candidates.add(user_id)
        if isinstance(order_id, str) and order_id in tools.db.orders:
            candidates.add(tools.db.orders[order_id].user_id)
        if action.name in {"find_user_id_by_name_zip", "find_user_id_by_email"}:
            try:
                candidates.add(getattr(tools, action.name)(**action.arguments))
            except ValueError:
                # A deliberate wrong-first lookup supplies no successful identity.
                pass
    if not candidates:
        state = task.initial_state
        patch = (
            state.initialization_data.agent_data
            if state and state.initialization_data
            else None
        )
        users = (patch or {}).get("users", {})
        if isinstance(users, dict) and len(users) == 1:
            candidates.add(resolve_materialized_retail_caller(task).user_id)
    if not candidates:
        instructions = source_task.user_scenario.instructions
        known = (
            instructions
            if isinstance(instructions, str)
            else instructions.known_info or ""
        )
        # Some transfer-only tasks have no golden ID or order reference. Match a
        # complete canonical name assertion; never split a displayed name into roles.
        matches = {
            user_id
            for user_id, user in source_tools.db.users.items()
            if declares_retail_name(
                known, re.escape(f"{user.name.first_name} {user.name.last_name}")
            )
        }
        if len(matches) == 1:
            canonical = next(iter(matches))
            candidates.add(aliases.get(canonical, canonical))
    if len(candidates) != 1 or not candidates <= tools.db.users.keys():
        raise ValueError(f"Retail task {task.id} has no unique structured caller")
    return tools.db.users[next(iter(candidates))]
