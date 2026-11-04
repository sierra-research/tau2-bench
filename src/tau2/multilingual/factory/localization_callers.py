"""Bind reviewed telecom caller rotation to approved regional identity values."""

import re

from tau2.data_model.language import TaskMetadata
from tau2.data_model.tasks import Task
from tau2.domains.telecom.data_model import TelecomDB
from tau2.multilingual.caller_identity import (
    TaskCallerIdentity,
    TelecomCallerPublication,
    TelecomCallerRoles,
)
from tau2.multilingual.factory.localization_emails import (
    EmailAssignment,
    EmailCandidate,
    allocate_emails,
    email_domain,
)
from tau2.multilingual.factory.localization_identity import ascii_name_parts
from tau2.multilingual.factory.localization_user_ids import (
    customer_name_components,
    remap_user_id_references,
    user_id_alias_mapping,
)
from tau2.multilingual.localization import LocalizationMap, UserIdAlias
from tau2.utils.pydantic_utils import update_pydantic_model_with_dict
from tau2.utils.text_match import fold_for_match


def _caller_prose(text: str, old: str, new: str, prefix: str, suffix: str) -> str:
    """Replace one explicit caller declaration, preserving every other byte."""
    pattern = re.compile(re.escape(prefix) + r"(.+?)" + re.escape(suffix))
    matches = list(pattern.finditer(text))
    if len(matches) != 1 or matches[0].group(1) != old:
        raise ValueError("Telecom caller declaration differs from its account name")
    start, end = matches[0].span(1)
    return text[:start] + new + text[end:]


def localize_telecom_callers(
    proposal: LocalizationMap,
    database: TelecomDB,
    tasks: list[Task],
    metadata: dict[str, TaskMetadata],
    roles: TelecomCallerRoles,
) -> TelecomCallerPublication:
    """Preserve caller gender assignments through task-specific account identities.

    Only names come from the declared name source. The target keeps its birth date,
    phone, account history, service lines, bills and every task condition. Names
    are already-approved map values; IDs and emails use existing deterministic
    normalization rules. No language content or gender is inferred here.
    """
    if proposal.source.domain != "telecom" or proposal.user_id_revision is None:
        raise ValueError("Caller binding requires a telecom map with localized IDs")
    if set(metadata) != {task.id for task in tasks}:
        raise ValueError("Caller metadata must cover exactly the published tasks")
    source_document = next(d for d in proposal.source.documents if d.id == "database")
    source_db = TelecomDB.model_validate(source_document.content)
    source_customers = {c.customer_id: c for c in source_db.customers}
    for identity in (roles.target, roles.male, roles.female):
        if (
            identity.customer_id not in source_customers
            or source_customers[identity.customer_id].full_name != identity.full_name
        ):
            raise ValueError("Declared caller role differs from canonical source")
    if roles.male.customer_id == roles.female.customer_id:
        raise ValueError("Caller genders require distinct declared name sources")
    aliases = user_id_alias_mapping(proposal.user_id_revision.aliases)
    native = customer_name_components(proposal, "native")
    romanized = customer_name_components(proposal, "romanized")
    source_target = source_customers[roles.target.customer_id]
    target_id = aliases[roles.target.customer_id]
    suffix = re.search(r"\d+$", roles.target.customer_id)
    if suffix is None:
        raise ValueError("Caller customer ID requires a numeric source suffix")
    source_tasks = {
        row["id"]: row
        for row in next(d for d in proposal.source.documents if d.id == "tasks").content
    }
    published, identities = [], []
    for task in tasks:
        assignment = metadata[task.id]
        if assignment.caller_gender is None:
            raise ValueError("Telecom caller rotation requires explicit task gender")
        canonical = source_tasks[assignment.canonical_task_id]
        canonical_calls = canonical["initial_state"]["initialization_actions"]
        canonical_callers = [
            call for call in canonical_calls if call["func_name"] == "set_user_info"
        ]
        if (
            len(canonical_callers) != 1
            or canonical_callers[0]["arguments"]["name"] != roles.target.full_name
            or canonical_callers[0]["arguments"]["phone_number"]
            != source_target.phone_number
        ):
            raise ValueError("Task caller differs from the declared target account")
        name_source = getattr(roles, assignment.caller_gender)
        new_name = " ".join(native[name_source.customer_id])
        romanized_name = " ".join(romanized[name_source.customer_id])
        parts = ascii_name_parts(romanized[name_source.customer_id])
        new_id = "_".join([*parts, suffix.group()])
        state = task.initial_state
        if state is None:
            raise ValueError("Telecom task lacks initialization")
        data = state.initialization_data
        effective = update_pydantic_model_with_dict(
            database, data.agent_data if data and data.agent_data else {}
        )
        target = next(c for c in effective.customers if c.customer_id == target_id)
        others = [c for c in effective.customers if c.customer_id != target_id]
        if any(c.customer_id == new_id for c in others):
            raise ValueError("Rotated caller ID collides with another customer")
        if any(
            fold_for_match(c.full_name) == fold_for_match(new_name)
            and c.date_of_birth == target.date_of_birth
            for c in others
        ):
            raise ValueError("Rotated caller name and birth date are ambiguous")
        reserved = [
            EmailAssignment(source=f"reserved{index}@example.com", value=customer.email)
            for index, customer in enumerate(others)
        ]
        # Rotated emails follow the reviewed regional database's mailbox domain.
        email = allocate_emails(
            [EmailCandidate(source=source_target.email, name_parts=parts)],
            email_domain([c.email for c in effective.customers]),
            reserved,
        )[0].value
        identity = TaskCallerIdentity(
            task_id=task.id,
            canonical_task_id=assignment.canonical_task_id,
            gender=assignment.caller_gender,
            canonical_target_id=roles.target.customer_id,
            canonical_name_source_id=name_source.customer_id,
            native_name=new_name,
            romanized_name=romanized_name,
            customer_id=new_id,
            email=email,
        )
        identities.append(identity)
        known_info = _caller_prose(
            task.user_scenario.instructions.known_info,
            target.full_name,
            new_name,
            "You are ",
            " with phone number ",
        )
        ticket = _caller_prose(
            task.ticket,
            target.full_name,
            new_name,
            "Customer name: ",
            ", phone number: ",
        )
        original_callers = [
            call
            for call in state.initialization_actions or []
            if call.func_name == "set_user_info"
        ]
        if (
            len(original_callers) != 1
            or original_callers[0].arguments.get("name") != target.full_name
            or original_callers[0].arguments.get("phone_number") != target.phone_number
        ):
            raise ValueError("Localized task caller differs from its effective account")
        if (target.full_name, target.customer_id, target.email) == (
            new_name,
            new_id,
            email,
        ):
            published.append(task.model_copy(deep=True))
            continue
        replacement = [UserIdAlias(canonical=target_id, localized=new_id)]
        renamed_db = remap_user_id_references(
            effective.model_dump(mode="json"), replacement
        )
        renamed_customer = next(
            row for row in renamed_db["customers"] if row["customer_id"] == new_id
        )
        renamed_customer.update(full_name=new_name, email=email)
        TelecomDB.model_validate(renamed_db)
        rewritten = remap_user_id_references(task.model_dump(mode="json"), replacement)
        instructions = rewritten["user_scenario"]["instructions"]
        instructions["known_info"] = known_info
        rewritten["ticket"] = ticket
        calls = rewritten["initial_state"]["initialization_actions"]
        user_calls = [call for call in calls if call["func_name"] == "set_user_info"]
        user_calls[0]["arguments"]["name"] = new_name
        initialization = rewritten["initial_state"]["initialization_data"] or {}
        agent_data = initialization.get("agent_data") or {}
        original_db = effective.model_dump(mode="json")
        agent_data.update(
            {
                key: value
                for key, value in renamed_db.items()
                if value != original_db[key]
            }
        )
        initialization["agent_data"] = agent_data
        rewritten["initial_state"]["initialization_data"] = initialization
        published.append(Task.model_validate(rewritten))
    return TelecomCallerPublication(tasks=published, identities=identities)
