"""Banking-owned inventory and invariants; the English KB is immutable evidence."""

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from pydantic import JsonValue

from tau2.data_model.tasks import Task
from tau2.multilingual.localization import (
    LocalizationSource,
    SourceDocument,
    SourceEntry,
    SourceOccurrence,
    json_content_digest,
)

BANKING_LOCALIZATION_VERSION = "banking-entities-v1"
_NAME_FIELDS = {
    "name",
    "customer_name",
    "full_name",
    "cardholder_name",
}
_ADDRESS_FIELDS = {"address", "shipping_address", "new_address"}
_CONTACT_FIELDS = {"email", "new_email", "phone", "phone_number"}
_PROSE_FIELDS = {"instructions", "communicate_info", "nl_assertions", "summary"}


def english_knowledge_manifest(directory: Path) -> dict[str, str]:
    """Pin every English document and retrieval policy by exact file bytes."""
    paths = sorted((directory / "documents").glob("*.json"))
    if not paths:
        raise ValueError("English knowledge documents are missing")
    return {
        p.relative_to(directory).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
    }


def knowledge_digest(directory: Path) -> str:
    """Digest the ordered filename-to-byte-digest manifest, not localized text."""
    return json_content_digest(english_knowledge_manifest(directory))


def inventory_banking(data_root: Path) -> LocalizationSource:
    """Inventory the full DB, task-only entities, policies and immutable KB pins."""
    directory = data_root / "tau2/domains/banking_knowledge"
    documents = []

    def document(identifier, kind, path, content):
        documents.append(
            SourceDocument(
                id=identifier,
                kind=kind,
                source_path=path.relative_to(data_root).as_posix(),
                sha256=json_content_digest(content),
                content=content,
            )
        )

    database = json.loads((directory / "db.json").read_text())
    document("database", "database", directory / "db.json", database)
    tasks = [
        json.loads(p.read_text())
        for p in sorted((directory / "tasks").glob("task_*.json"))
    ]
    for task in tasks:
        Task.model_validate(task)
    if not tasks or len({t["id"] for t in tasks}) != len(tasks):
        raise ValueError("Banking task cohort is empty or contains duplicate IDs")
    document("tasks", "tasks", directory / "tasks", tasks)
    document(
        "dependency:english_knowledge",
        "dependency",
        directory / "documents",
        english_knowledge_manifest(directory),
    )
    for path in sorted((directory / "prompts").glob("*")):
        if path.is_file():
            document(
                f"dependency:policy:{path.name}", "dependency", path, path.read_text()
            )

    # Identity is linked by explicit user ID; unique full-name lookup is a fallback
    # for the source tools which themselves resolve customers by exact name.
    users = database["users"]["data"]
    name_owners: dict[str, set[str]] = {}
    for key, user in users.items():
        name_owners.setdefault(user["name"].casefold(), set()).add(key)
    entries: dict[str, SourceEntry] = {}

    def add(doc, path, value, category, owner, role, context):
        identifier = f"{category}:" + json_content_digest(
            [BANKING_LOCALIZATION_VERSION, owner, role, value]
        )
        occurrence = SourceOccurrence(document_id=doc, path=path)
        if identifier in entries:
            entries[identifier].occurrences.append(occurrence)
        else:
            entries[identifier] = SourceEntry(
                id=identifier,
                category=category,
                source=value,
                occurrences=[occurrence],
                context=context,
            )

    def walk(
        doc: str,
        value: JsonValue,
        path: list,
        owner: str = "",
        people: dict | None = None,
        names: dict | None = None,
    ):
        people = users if people is None else people
        names = name_owners if names is None else names
        field = str(path[-1]) if path else ""
        if isinstance(value, dict) and value:
            if doc == "tasks" and len(path) == 1:
                owner = "task:" + value["id"]
                initial = value.get("initial_state") or {}
                initialization = initial.get("initialization_data") or {}
                agent_data = initialization.get("agent_data") or {}
                injected = (agent_data.get("users") or {}).get("data", {})
                people = {**users, **injected}
                names = {}
                for identifier, record in people.items():
                    if isinstance(record.get("name"), str):
                        names.setdefault(record["name"].casefold(), set()).add(
                            identifier
                        )
            owner = value.get("user_id", owner)
            if not value.get("user_id"):
                name = next(
                    (
                        value[k]
                        for k in ("customer_name", "full_name", "name")
                        if isinstance(value.get(k), str)
                    ),
                    None,
                )
                matches = names.get(name.casefold(), set()) if name else set()
                if len(matches) == 1:
                    owner = next(iter(matches))
                elif name and ("customer_name" in value or "full_name" in value):
                    owner = f"{owner}:person:{name}"
            for key, child in value.items():
                walk(doc, child, [*path, key], owner, people, names)
            return
        if isinstance(value, list) and value:
            for index, child in enumerate(value):
                walk(doc, child, [*path, index], owner, people, names)
            return
        category, role, semantic_owner = "protocol", field, ""
        if isinstance(value, str) and value:
            if field in _NAME_FIELDS and not (
                field == "name"
                and any(part in path for part in ("actions", "initialization_actions"))
                and len(path) > 1
                and path[-2] != "arguments"
            ):
                category, role = "identity_name", "full_name"
                matches = names.get(value.casefold(), set())
                semantic_owner = next(iter(matches)) if len(matches) == 1 else owner
            elif field in _ADDRESS_FIELDS:
                category, role, semantic_owner = "address", "address", owner
            elif field in _CONTACT_FIELDS:
                category, role = (
                    "contact",
                    "email" if "email" in field else "phone_number",
                )
                # Same contact is a source-defined lookup key across snapshots.
                semantic_owner = "contact"
            elif field in {"merchant_name", "employer", "employer_name"}:
                category, role, semantic_owner = "other", field, "organization"
            elif field == "description" and "data" in path:
                category, role, semantic_owner = (
                    "prose",
                    "transaction_description",
                    owner,
                )
            elif field in _PROSE_FIELDS or (field == "arguments"):
                category, role, semantic_owner = "prose", field, owner
        identity = people.get(owner, {})
        identity_context = json.dumps(
            {"full_name": identity.get("name", ""), "email": identity.get("email", "")},
            ensure_ascii=False,
        )
        context_owner = owner if category == "contact" else semantic_owner
        context = (
            f"source_identity={identity_context}; "
            f"role={role}; entity={context_owner}; banking={BANKING_LOCALIZATION_VERSION}. "
            "Localize identity values consistently across DB, historical snapshots, initialization, tool arguments and English task prose. "
            "Keep English instruction sentences and financial/geographic rules unchanged. Preserve all USD amounts, dates, product/bank names, codes, IDs, categories and deliberately wrong values. "
            "Transaction descriptions retain merchant category, ACH/ATM identifiers and clues. JSON-encoded arguments remain valid JSON with only mapped entity values changed. "
            f"source_occurrence={doc}:{json.dumps(path)}"
        )
        add(doc, path, value, category, semantic_owner, role, context)

    for doc in documents:
        if doc.kind == "dependency":
            add(
                doc.id,
                [],
                doc.content,
                "protocol",
                doc.id,
                "immutable_dependency",
                "Preserve English KB and retrieval-policy dependencies byte-for-byte.",
            )
        else:
            walk(doc.id, doc.content, [])
    return LocalizationSource(
        domain="banking_knowledge",
        documents=documents,
        entries=list(entries.values()),
    )


def validate_banking_documents(
    source: LocalizationSource, compiled: list[SourceDocument]
) -> None:
    """Guard financial strings, JSON-encoded tool calls, and entity equality classes."""
    original = {d.id: d for d in source.documents}
    forward, reverse = {}, {}

    def check(old, new, field="", path=()):
        if type(old) is not type(new):
            raise ValueError(f"Banking JSON shape changed: {path}")
        if isinstance(old, dict):
            if old.keys() != new.keys():
                raise ValueError(f"Banking protocol keys changed: {path}")
            for key in old:
                check(old[key], new[key], key, (*path, key))
        elif isinstance(old, list):
            if len(old) != len(new):
                raise ValueError("Banking list length changed")
            for index, (left, right) in enumerate(zip(old, new, strict=True)):
                check(left, right, field, (*path, index))
        elif field == "arguments" and isinstance(old, str):
            try:
                before, after = json.loads(old), json.loads(new)
            except json.JSONDecodeError as error:
                raise ValueError(
                    "Banking encoded tool arguments must retain valid JSON"
                ) from error
            check(before, after, "", path)
        elif field in _CONTACT_FIELDS and isinstance(old, str) and old:
            family = "email" if "email" in field else "phone"
            if (
                forward.setdefault((family, old), new) != new
                or reverse.setdefault((family, new), old) != old
            ):
                raise ValueError(
                    "Banking contact mapping changed lookup equality or merged distinct entities"
                )
        elif field in (_PROSE_FIELDS | {"description"}) and isinstance(old, str):
            financial = r"\$\s*\d[\d,]*(?:\.\d+)?|\b\d+(?:\.\d+)?\s*%"
            if Counter(re.findall(financial, old)) != Counter(
                re.findall(financial, new)
            ):
                raise ValueError(f"Banking prose changed financial literals: {path}")
        elif old != new and field not in (
            _NAME_FIELDS
            | _ADDRESS_FIELDS
            | _CONTACT_FIELDS
            | _PROSE_FIELDS
            | {"merchant_name", "employer", "employer_name", "description"}
        ):
            raise ValueError(
                f"Banking changed a protected rule, amount or identifier: {path}"
            )

    for document in compiled:
        before = original[document.id]
        if document.kind == "dependency" and document.content != before.content:
            raise ValueError(
                "English knowledge and retrieval policies must remain unchanged"
            )
        if document.kind in {"database", "tasks"}:
            check(before.content, document.content)
