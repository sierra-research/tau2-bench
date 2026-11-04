"""Compile checked regional proposals without executing generators or publishing data."""

import re
from copy import deepcopy
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Literal, get_args, get_origin

from pydantic import BaseModel, JsonValue, TypeAdapter

from tau2.data_model.language import NameOrder
from tau2.data_model.tasks import Task
from tau2.domains.airline.data_model import FlightDB
from tau2.domains.banking_knowledge.data_model import TransactionalDB
from tau2.domains.retail.data_model import RetailDB
from tau2.domains.retail.tools import RetailTools
from tau2.domains.telecom.data_model import TelecomDB
from tau2.domains.telecom.user_data_model import TelecomUserDB
from tau2.multilingual.localization import (
    LocalizationEntry,
    LocalizationMap,
    LocalizationSource,
    SourceDocument,
    SourceEntry,
    json_content_digest,
)
from tau2.multilingual.names import (
    clarify_retail_name,
    declares_retail_name,
    resolve_retail_caller,
)
from tau2.utils.pydantic_utils import update_pydantic_model_with_dict

_DB_MODELS = {
    "airline": FlightDB,
    "retail": RetailDB,
    "telecom": TelecomDB,
    "banking_knowledge": TransactionalDB,
}
_MONEY_FIELDS = frozenset(
    {
        "amount",
        "balance",
        "price",
        "prices",
        "total_due",
        "price_per_month",
        "data_refueling_price_per_gb",
        "exchange_price_difference",
        "amount_due",
        "goodwill_credit_used_this_year",
        "credit_amount",
    }
)


def _at(value: JsonValue, path: list[str | int]) -> JsonValue:
    for part in path:
        if isinstance(value, dict) and isinstance(part, str) and part in value:
            value = value[part]
        elif isinstance(value, list) and type(part) is int and 0 <= part < len(value):
            value = value[part]
        else:
            raise ValueError(f"Invalid localization source path: {path!r}")
    return value


def _same_shape(source: JsonValue, target: JsonValue) -> None:
    if type(source) is not type(target):
        raise ValueError("Localized value must preserve source JSON scalar type/shape")
    if isinstance(source, dict):
        if source.keys() != target.keys():
            raise ValueError("Localized object must preserve source JSON keys/shape")
        for key, value in source.items():
            _same_shape(value, target[key])
    elif isinstance(source, list):
        if len(source) != len(target):
            raise ValueError("Localized array must preserve source JSON length/shape")
        for value, other in zip(source, target, strict=True):
            _same_shape(value, other)


def _leaf_paths(value: JsonValue, path: tuple = ()):
    if isinstance(value, dict) and value:
        for key, item in value.items():
            yield from _leaf_paths(item, path + (key,))
    elif isinstance(value, list) and value:
        for index, item in enumerate(value):
            yield from _leaf_paths(item, path + (index,))
    else:
        yield path


def _check_source(proposal: LocalizationMap) -> None:
    aliases = []
    if proposal.user_id_revision is not None:
        from tau2.multilingual.factory.localization_user_ids import (
            derive_user_id_aliases,
            localize_user_id_entry,
            remap_user_id_references,
        )

        aliases = proposal.user_id_revision.aliases
        if aliases != derive_user_id_aliases(proposal):
            raise ValueError("User-ID aliases must match the localized customer names")
    if (
        proposal.caller_name_revision is not None
        or proposal.user_id_revision is not None
    ):
        from tau2.multilingual.factory.localization_caller_names import (
            localized_caller_entries,
        )

        if localized_caller_entries(proposal) != proposal.entries:
            raise ValueError(
                "Localized caller names must match their database identity"
            )
    documents = {document.id: document for document in proposal.source.documents}
    if len(documents) != len(proposal.source.documents):
        raise ValueError("Duplicate localization source document IDs")
    for document in documents.values():
        if json_content_digest(document.content) != document.sha256:
            raise ValueError(f"Stale source document digest: {document.id}")
    sources = {entry.id: entry for entry in proposal.source.entries}
    entries = {entry.id: entry for entry in proposal.entries}
    if len(sources) != len(proposal.source.entries) or len(entries) != len(
        proposal.entries
    ):
        raise ValueError("Duplicate localization entry IDs")
    if sources.keys() != entries.keys():
        raise ValueError("Localization entries must cover every source entry exactly")
    targets: set[tuple[str | int, ...]] = set()
    ancestors: set[tuple[str | int, ...]] = set()
    for identifier, source in sources.items():
        entry = entries[identifier]
        recorded_source = SourceEntry.model_validate(
            entry.model_dump(include=set(SourceEntry.model_fields))
        )
        if recorded_source != source:
            raise ValueError(f"Stale localization source entry: {identifier}")
        if aliases:
            revised = localize_user_id_entry(entry, aliases)
            if revised.forms != entry.forms or revised.operation != entry.operation:
                raise ValueError(
                    f"Localization entry has an unlocalized user-ID reference: {identifier}"
                )
        for occurrence in source.occurrences:
            document = documents.get(occurrence.document_id)
            if document is None:
                raise ValueError("Localization occurrence references missing document")
            key = (document.id, *occurrence.path)
            if (
                key in targets
                or key in ancestors
                or any(key[:index] in targets for index in range(1, len(key)))
            ):
                raise ValueError("Duplicate or overlapping localization occurrence")
            targets.add(key)
            ancestors.update(key[:index] for index in range(1, len(key)))
            observed = _at(document.content, occurrence.path)
            _same_shape(source.source, observed)
            if observed != source.source:
                raise ValueError(f"Stale localization occurrence: {identifier}")
        for representation in ("native", "romanized"):
            value = getattr(entry.forms, representation)
            _same_shape(source.source, value)
            if entry.operation == "preserve" and value != source.source:
                raise ValueError("preserve operation must keep both forms unchanged")
        if source.category == "protocol" and aliases:
            for occurrence in source.occurrences:
                expected = remap_user_id_references(
                    source.source, aliases, path=tuple(occurrence.path)
                )
                if (
                    entry.forms.native != expected
                    or entry.forms.romanized != expected
                    or entry.operation
                    != ("adapt" if expected != source.source else "preserve")
                ):
                    raise ValueError(
                        f"Protocol edits must match the declared user-ID aliases: {identifier}"
                    )
        elif (
            source.category in {"protocol", "date_time"}
            and entry.operation != "preserve"
        ):
            raise ValueError(f"{source.category} entries require preserve operation")
    for document in documents.values():
        for path in _leaf_paths(document.content):
            key = (document.id, *path)
            if not any(key[:index] in targets for index in range(1, len(key) + 1)):
                raise ValueError(f"Source inventory has uncovered content: {key}")


def _compile_documents(
    proposal: LocalizationMap, representation: str
) -> list[SourceDocument]:
    documents = [
        document.model_copy(deep=True) for document in proposal.source.documents
    ]
    by_id = {document.id: document for document in documents}
    for entry in proposal.entries:
        replacement = getattr(entry.forms, representation)
        for occurrence in entry.occurrences:
            document = by_id[occurrence.document_id]
            if not occurrence.path:
                document.content = deepcopy(replacement)
            else:
                parent = _at(document.content, occurrence.path[:-1])
                parent[occurrence.path[-1]] = deepcopy(replacement)
    for document in documents:
        if proposal.user_id_revision is not None:
            from tau2.multilingual.factory.localization_user_ids import (
                remap_user_id_references,
            )

            document.content = remap_user_id_references(
                document.content, proposal.user_id_revision.aliases
            )
    if proposal.source.domain == "intake":
        from tau2.domains.intake.localization import compile_intake_documents

        compile_intake_documents(
            documents,
            proposal.settings.language,
            proposal.source.documents,
            representation,
            caller_surfaces=proposal.intake_caller_surfaces,
        )
    _validate_documents(proposal, documents, representation)
    if proposal.source.domain == "retail":
        _clarify_regional_retail_names(proposal, documents)
    for document in documents:
        document.sha256 = json_content_digest(document.content)
    return documents


def _retail_tools_for_task(database: RetailDB, task: Task) -> RetailTools:
    """Reuse runtime initialization on an isolated database, never goal actions."""
    if task.initial_state is None:
        return RetailTools(database)
    from tau2.domains.retail.environment import get_environment
    from tau2.multilingual.factory.task_state import initialize_task_state

    environment = get_environment(db=database.model_copy(deep=True))
    return initialize_task_state(environment, task).tools


def _clarify_regional_retail_names(
    proposal: LocalizationMap, documents: list[SourceDocument]
) -> None:
    """Check caller identity and saved-address goals, then append name-role facts."""
    database = RetailDB.model_validate(
        next(document.content for document in documents if document.kind == "database")
    )
    source_database = RetailDB.model_validate(
        next(
            document.content
            for document in proposal.source.documents
            if document.kind == "database"
        )
    )
    source_tasks = {
        task.id: task
        for document in proposal.source.documents
        if document.kind == "tasks"
        for task in TypeAdapter(list[Task]).validate_python(document.content)
    }
    aliases = (
        {
            alias.canonical: alias.localized
            for alias in proposal.user_id_revision.aliases
        }
        if proposal.user_id_revision
        else {}
    )
    for document in documents:
        if document.kind != "tasks":
            continue
        for raw in document.content:
            task = Task.model_validate(raw)
            tools = _retail_tools_for_task(database, task)
            if task.id not in source_tasks:
                raise ValueError("Localization changed protocol task ID")
            source_task = source_tasks[task.id]
            source_tools = _retail_tools_for_task(source_database, source_task)
            caller = resolve_retail_caller(
                task, tools, source_task, source_tools, aliases
            )
            instructions = task.user_scenario.instructions
            source_instructions = source_task.user_scenario.instructions
            known = (
                instructions
                if isinstance(instructions, str)
                else instructions.known_info or ""
            )
            original_known = (
                source_instructions
                if isinstance(source_instructions, str)
                else source_instructions.known_info or ""
            )
            canonical_ids = {value: key for key, value in aliases.items()}
            original = source_tools.db.users.get(
                canonical_ids.get(caller.user_id, caller.user_id)
            )
            if original and declares_retail_name(
                original_known,
                re.escape(f"{original.name.first_name} {original.name.last_name}"),
            ):
                parts = (caller.name.first_name, caller.name.last_name)
                if proposal.settings.name_order == NameOrder.FAMILY_FIRST:
                    parts = tuple(reversed(parts))
                if not declares_retail_name(
                    known, r"\s*".join(re.escape(part) for part in parts)
                ):
                    raise ValueError(
                        f"Retail task {task.id} display name disagrees with its caller"
                    )
            if original and re.search(
                r"(?<!\w)" + re.escape(original.address.zip) + r"(?!\w)", original_known
            ):
                if not re.search(
                    r"(?<!\w)" + re.escape(caller.address.zip) + r"(?!\w)", known
                ):
                    raise ValueError(
                        f"Retail task {task.id} postal code disagrees with its caller"
                    )
                try:
                    source_match = source_tools.find_user_id_by_name_zip(
                        first_name=original.name.first_name,
                        last_name=original.name.last_name,
                        zip=original.address.zip,
                    )
                except ValueError:
                    source_match = None
                if source_match == original.user_id:
                    try:
                        match = tools.find_user_id_by_name_zip(
                            first_name=caller.name.first_name,
                            last_name=caller.name.last_name,
                            zip=caller.address.zip,
                        )
                    except ValueError as exc:
                        raise ValueError(
                            f"Retail task {task.id} caller lookup failed: {exc}"
                        ) from exc
                    if match != caller.user_id:
                        raise ValueError(
                            f"Retail task {task.id} resolves another caller"
                        )
            if original and source_task.evaluation_criteria:
                source_actions = source_task.evaluation_criteria.actions or []
                localized_actions = task.evaluation_criteria.actions or []
                address = original.address.model_dump()
                for source_action, action in zip(
                    source_actions, localized_actions, strict=True
                ):
                    if source_action.name == "modify_pending_order_address" and all(
                        source_action.arguments.get(key) == value
                        for key, value in address.items()
                    ):
                        if any(
                            action.arguments.get(key) != value
                            for key, value in caller.address.model_dump().items()
                        ):
                            raise ValueError(
                                f"Retail task {task.id} saved-address goal disagrees with its caller"
                            )
            clarified = clarify_retail_name(
                task,
                caller_id=caller.user_id,
                caller=caller,
                name_order=proposal.settings.name_order,
            )
            if isinstance(clarified.user_scenario.instructions, str):
                raw["user_scenario"]["instructions"] = (
                    clarified.user_scenario.instructions
                )
            else:
                raw["user_scenario"]["instructions"]["known_info"] = (
                    clarified.user_scenario.instructions.known_info
                )


def _has_literal(annotation: object) -> bool:
    return get_origin(annotation) is Literal or any(
        _has_literal(argument) for argument in get_args(annotation)
    )


def _typed_invariants(
    before: object, after: object, scale: Decimal, path: tuple = ()
) -> None:
    """Keep protocol fields and numeric predicates; scale only monetary quantities."""
    if path and isinstance(path[-1], str):
        key = path[-1]
        if (
            key
            in {
                "id",
                "domain",
                "flight_number",
                "func_name",
                "env_type",
                "requestor",
                "compare_args",
                "reward_basis",
                "dob",
                "date_of_birth",
                "date",
            }
            or key.endswith(("_id", "_ids", "_date", "_at", "_time_est"))
        ) and before != after:
            raise ValueError(
                f"Localization changed protocol identifier or task date: {path}"
            )
        if (
            key == "name"
            and len(path) >= 3
            and path[-3] in {"actions", "initialization_actions"}
            and before != after
        ):
            raise ValueError(f"Localization changed task action: {path}")
        if (
            key
            in {
                "orders",
                "reservations",
                "exchange_items",
                "exchange_new_items",
                "return_items",
            }
            and isinstance(before, list)
            and before != after
        ):
            raise ValueError(f"Localization changed protocol reference graph: {path}")
    if isinstance(before, BaseModel):
        for name, field in type(before).model_fields.items():
            old, new = getattr(before, name), getattr(after, name)
            if (
                not isinstance(old, (BaseModel, dict, list))
                and _has_literal(field.annotation)
                and old != new
            ):
                raise ValueError(
                    f"Localization changed protocol field: {path + (name,)}"
                )
            _typed_invariants(old, new, scale, path + (name,))
    elif isinstance(before, dict):
        if before.keys() != after.keys():
            raise ValueError("Localization changed protocol dictionary keys")
        for key in before:
            _typed_invariants(before[key], after[key], scale, path + (key,))
    elif isinstance(before, list):
        if len(before) != len(after):
            raise ValueError("Localization changed record or task count")
        for index, (old, new) in enumerate(zip(before, after, strict=True)):
            _typed_invariants(old, new, scale, path + (index,))
    elif type(before) in {int, float}:
        monetary = any(part in _MONEY_FIELDS for part in path if isinstance(part, str))
        expected = Decimal(str(before)) * (scale if monetary else Decimal(1))
        if Decimal(str(after)) != expected:
            kind = "currency scale" if monetary else "numeric task predicate"
            raise ValueError(f"Localization changed {kind}: {path}")
    elif isinstance(before, (bool, Enum, date, datetime)) and before != after:
        raise ValueError(f"Localization changed protocol state or date: {path}")


def _check_value_relations(before: JsonValue, after: JsonValue) -> None:
    """Airport and phone equality classes must survive edits across linked records."""
    forward: dict[tuple[str, str], str] = {}
    reverse: dict[tuple[str, str], str] = {}

    def visit(old, new, field=""):
        if isinstance(old, dict):
            for key, value in old.items():
                visit(value, new[key], key)
        elif isinstance(old, list):
            for value, other in zip(old, new, strict=True):
                visit(value, other, field)
        elif (
            isinstance(old, str)
            and old
            and field in {"origin", "destination", "phone_number"}
        ):
            family = "airport" if field in {"origin", "destination"} else "phone"
            if family == "airport" and re.fullmatch(r"[A-Z]{3}", new) is None:
                raise ValueError(
                    "Localized airport values must remain three-letter IATA codes"
                )
            if (
                forward.setdefault((family, old), new) != new
                or reverse.setdefault((family, new), old) != old
            ):
                raise ValueError(
                    f"Localization changed linked {family} graph or merged distinct values"
                )

    visit(before, after)


def _lookup_results(
    domain: str, database: BaseModel, tasks: list[Task]
) -> list[object]:
    """Use domain lookup implementations to preserve successful and failed auth attempts."""
    if domain == "retail":
        from tau2.domains.retail.tools import RetailTools

        toolkit = RetailTools
        names = {"find_user_id_by_name_zip", "find_user_id_by_email"}
    elif domain == "banking_knowledge":
        from tau2.domains.banking_knowledge.tools import KnowledgeTools

        toolkit = KnowledgeTools
        names = {"get_user_information_by_name", "get_user_information_by_email"}
    elif domain == "telecom":
        from tau2.domains.telecom.tools import TelecomTools

        toolkit = TelecomTools
        names = {"get_customer_by_phone", "get_customer_by_name"}
    else:
        return []
    outcomes = []
    for task in tasks:
        db = database
        if task.initial_state and task.initial_state.initialization_data:
            data = task.initial_state.initialization_data.agent_data
            if data:
                db = update_pydantic_model_with_dict(db, data)
        tools = toolkit(db)
        for action in (
            task.evaluation_criteria.actions or [] if task.evaluation_criteria else []
        ):
            if action.name not in names:
                continue
            try:
                result = getattr(tools, action.name)(**action.arguments)
            except ValueError:
                result = None
            if domain == "banking_knowledge" and isinstance(result, str):
                result = tuple(re.findall(r"Record ID: ([^\n]+)", result))
            if isinstance(result, list):
                result = [row.customer_id for row in result]
            elif isinstance(result, BaseModel):
                result = result.customer_id
            outcomes.append(result)
    return outcomes


def _validate_documents(
    proposal: LocalizationMap,
    compiled: list[SourceDocument],
    representation: str = "native",
) -> None:
    if proposal.source.domain == "intake":
        from tau2.domains.intake.data_model import IntakeDB
        from tau2.domains.intake.localization import validate_intake_tasks
        from tau2.domains.intake.user_data_model import IntakeUserDB

        documents = {document.id: document for document in compiled}
        IntakeDB.model_validate(documents["database"].content)
        IntakeUserDB.model_validate(documents["user_database"].content)
        validate_intake_tasks(
            TypeAdapter(list[Task]).validate_python(documents["tasks"].content),
            proposal.settings.language,
            representation,
        )
        return
    originals = {document.id: document for document in proposal.source.documents}
    if proposal.source.domain == "banking_knowledge":
        from tau2.domains.banking_knowledge.localization import (
            validate_banking_documents,
        )

        validate_banking_documents(proposal.source, compiled)
    if proposal.user_id_revision is not None:
        from tau2.multilingual.factory.localization_user_ids import (
            remap_user_id_references,
        )

        originals = {
            identifier: document.model_copy(
                update={
                    "content": remap_user_id_references(
                        document.content, proposal.user_id_revision.aliases
                    )
                }
            )
            for identifier, document in originals.items()
        }
    scale = Decimal(proposal.settings.currency_scale)
    if proposal.source.domain == "banking_knowledge":
        if proposal.settings.currency_code != "USD" or scale != 1:
            raise ValueError(
                "Banking requires unchanged USD amounts and currency scale 1"
            )
    source_db = localized_db = None
    source_tasks: list[Task] = []
    localized_tasks: list[Task] = []
    for document in compiled:
        original = originals[document.id]
        if document.kind == "database":
            if source_db is not None:
                raise ValueError("Localization must have one domain database")
            model = _DB_MODELS[proposal.source.domain]
            source_db = model.model_validate(original.content)
            localized_db = model.model_validate(document.content)
            _typed_invariants(source_db, localized_db, scale)
        elif document.kind == "user_database":
            if proposal.source.domain != "telecom":
                raise ValueError(
                    "User database localization is supported only for telecom"
                )
            _typed_invariants(
                TelecomUserDB.model_validate(original.content),
                TelecomUserDB.model_validate(document.content),
                scale,
            )
        elif document.kind == "tasks":
            old = TypeAdapter(list[Task]).validate_python(original.content)
            new = TypeAdapter(list[Task]).validate_python(document.content)
            _typed_invariants(old, new, scale)
            source_tasks.extend(old)
            localized_tasks.extend(new)
        elif document.kind == "policy" and not isinstance(document.content, str):
            raise ValueError("Policy document must contain text")
    if source_db is None:
        raise ValueError("Localization requires a domain database")
    if len({task.id for task in source_tasks}) != len(source_tasks):
        raise ValueError("Duplicate source task IDs")
    _check_value_relations(
        {identifier: document.content for identifier, document in originals.items()},
        {document.id: document.content for document in compiled},
    )
    if _lookup_results(
        proposal.source.domain, source_db, source_tasks
    ) != _lookup_results(proposal.source.domain, localized_db, localized_tasks):
        raise ValueError("Localization changed task authentication lookup results")


def validate_localization_source(proposal: LocalizationMap) -> LocalizationMap:
    """Check complete source/shape fidelity before repairing failed domain semantics."""
    proposal = LocalizationMap.model_validate(proposal.model_dump())
    _check_source(proposal)
    return proposal


def rebind_localization_inventory(
    previous: LocalizationMap, source: LocalizationSource
) -> LocalizationMap:
    """Explicitly refine entry ownership over identical canonical documents.

    Exact occurrences carry the previous proposals into split entries. A merge is
    possible only when every inherited proposal agrees. This does not translate,
    approve content, or permit actual source-document changes.
    """
    previous = validate_localization_source(previous)
    source = LocalizationSource.model_validate(source.model_dump())
    if (
        source.domain != previous.source.domain
        or source.documents != previous.source.documents
    ):
        raise ValueError(
            "Inventory rebinding requires identical canonical source documents"
        )
    by_occurrence = {
        (occurrence.document_id, tuple(occurrence.path)): entry
        for entry in previous.entries
        for occurrence in entry.occurrences
    }
    remaining = set(by_occurrence)
    rebound = []
    for entry in source.entries:
        proposals = []
        for occurrence in entry.occurrences:
            key = (occurrence.document_id, tuple(occurrence.path))
            if key not in remaining:
                raise ValueError(
                    "Inventory rebinding changed or duplicated a source occurrence"
                )
            remaining.remove(key)
            original = by_occurrence[key]
            if original.category != entry.category or original.source != entry.source:
                raise ValueError(
                    "Inventory rebinding changed a source value or category"
                )
            proposals.append(original.model_dump(exclude=set(SourceEntry.model_fields)))
        if not proposals or any(value != proposals[0] for value in proposals[1:]):
            raise ValueError("Inventory rebinding has ambiguous merged proposals")
        rebound.append(LocalizationEntry(**entry.model_dump(), **proposals[0]))
    if remaining:
        raise ValueError("Inventory rebinding omitted source occurrences")
    result = previous.model_copy(update={"source": source, "entries": rebound})
    return validate_localization_source(result)


def validate_localization(proposal: LocalizationMap) -> LocalizationMap:
    """Check source coverage and both forms without changing or approving the draft."""
    proposal = validate_localization_source(proposal)
    if proposal.source.domain == "intake" and proposal.settings.language != "en":
        from tau2.multilingual.factory.intake_address_repair import (
            validate_intake_address_forms,
        )
        from tau2.multilingual.factory.intake_authoring import (
            validate_pronunciation_script,
        )

        for entry in proposal.entries:
            validate_intake_address_forms(
                entry, entry.forms, proposal.settings.language
            )
            if "dependency=pronunciations;" in entry.context:
                for representation in ("native", "romanized"):
                    validate_pronunciation_script(
                        getattr(entry.forms, representation),
                        proposal.settings.language,
                        representation,
                    )
    for representation in ("native", "romanized"):
        _compile_documents(proposal, representation)
    if proposal.source.domain == "banking_knowledge":
        from tau2.domains.banking_knowledge.localization_prose import (
            validate_banking_prose,
        )

        validate_banking_prose(proposal)
        from tau2.domains.banking_knowledge.localization_households import (
            validate_banking_households,
        )

        validate_banking_households(proposal)
    return proposal


def compile_localization(
    proposal: LocalizationMap, representation: Literal["native", "romanized"] = "native"
) -> list[SourceDocument]:
    """Return source-checked domain documents; publication and review remain separate."""
    if representation not in {"native", "romanized"}:
        raise ValueError("Unknown localization representation")
    proposal = validate_localization(proposal)
    return _compile_documents(proposal, representation)
