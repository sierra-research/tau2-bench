"""Inventory canonical regional values without changing domain data or behavior."""

import ast
import json
import os
import re
import tomllib
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
from tau2.task_subsets.schema import TaskSubset

_REGIONAL_LITERAL = re.compile(
    r"\b(?:USD|USA|EST|dollars?|United States|San Francisco|New York|ZIP|postal)\b|\$|555-",
    re.IGNORECASE,
)

_MONEY_FIELDS = {
    "amount",
    "balance",
    "price",
    "price_per_month",
    "data_refueling_price_per_gb",
    "total_due",
    "amount_due",
    "goodwill_credit_used_this_year",
    "credit_amount",
    "exchange_price_difference",
}
_ADDRESS_FIELDS = {
    "address1",
    "address2",
    "street",
    "city",
    "state",
    "country",
    "zip",
    "zip_code",
}
_PROSE_FIELDS = {
    "reason_for_call",
    "known_info",
    "unknown_info",
    "task_instructions",
    "ticket",
    "purpose",
    "relevant_policies",
    "notes",
    "communicate_info",
    "nl_assertions",
    "message",
    "summary",
    "expression",
    "persona",
}
_PROTOCOL_SUBTREES = {
    "available_seats",
    "reward_basis",
    "compare_args",
    "line_ids",
    "bill_ids",
    "orders",
    "reservations",
    "tracking_id",
    "item_ids",
    "new_item_ids",
    "fulfillments",
}


def _document(
    document_id: str,
    kind: str,
    path: Path,
    data_root: Path,
    content: JsonValue,
    *,
    source_path: str | None = None,
) -> SourceDocument:
    return SourceDocument(
        id=document_id,
        kind=kind,
        source_path=source_path or os.path.relpath(path, data_root),
        sha256=json_content_digest(content),
        content=content,
    )


def _read(path: Path) -> JsonValue:
    if path.suffix == ".toml":
        return tomllib.loads(path.read_text())
    return json.loads(path.read_text())


def _source_documents(domain: str, data_root: Path) -> list[SourceDocument]:
    directory = data_root / "tau2" / "domains" / domain
    database_path = directory / ("db.toml" if domain == "telecom" else "db.json")
    documents = [
        _document(
            "database", "database", database_path, data_root, _read(database_path)
        )
    ]
    if domain == "telecom":
        path = directory / "user_db.toml"
        documents.append(
            _document("user_database", "user_database", path, data_root, _read(path))
        )
    subset = TaskSubset.model_validate_json(
        (data_root / "tau2" / "task_subsets" / f"{domain}_50.json").read_text()
    )
    if subset.domain != domain or len(subset.task_ids) != 50:
        raise ValueError(f"Expected canonical {domain} frozen50 subset")
    task_path = directory / "tasks.json"
    raw_tasks = _read(task_path)
    by_id = {task["id"]: task for task in raw_tasks}
    if len(by_id) != len(raw_tasks):
        raise ValueError(f"Duplicate canonical {domain} task IDs")
    missing = set(subset.task_ids) - set(by_id)
    if missing:
        raise ValueError(
            f"Canonical {domain} task subset has missing IDs: {sorted(missing)}"
        )
    tasks = [by_id[task_id] for task_id in subset.task_ids]
    for task in tasks:
        Task.model_validate(task)
    documents.append(_document("tasks", "tasks", task_path, data_root, tasks))
    policies = (
        ("main_policy.md", "tech_support_manual.md", "tech_support_workflow.md")
        if domain == "telecom"
        else ("policy.md",)
    )
    for name in policies:
        path = directory / name
        documents.append(
            _document(
                f"policy:{path.stem}", "policy", path, data_root, path.read_text()
            )
        )
    # These are evidence snapshots, not replacement Python source. A compiler must
    # explicitly wire translated tool surfaces; applying a DB map cannot do that.
    source_dir = Path(__file__).resolve().parents[2] / "domains" / domain
    for name in ("tools.py", "user_tools.py", "data_model.py", "user_data_model.py"):
        path = source_dir / name
        if not path.exists():
            continue
        source_text = path.read_text()
        tree = ast.parse(source_text)
        airport_literals = {
            id(keyword.value)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "AirportCode"
            for keyword in node.keywords
            if keyword.arg in {"iata", "city"}
        }
        literals = [
            {"line": node.lineno, "text": node.value}
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value
            and (_REGIONAL_LITERAL.search(node.value) or id(node) in airport_literals)
        ]
        # Include executable monetary arithmetic as evidence too: constants such
        # as 30/person insurance and 50/bag do not appear in string literals.
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AugAssign, ast.Return)):
                continue
            excerpt = ast.get_source_segment(source_text, node)
            regional_money_call = any(
                isinstance(part, ast.Call)
                and isinstance(part.func, ast.Attribute)
                and part.func.attr in {"_fee", "scale_fee", "format_money"}
                for part in ast.walk(node)
            )
            if excerpt and (
                regional_money_call
                or (
                    any(word in excerpt for word in ("price", "amount", "fee", "cost"))
                    and any(isinstance(part, ast.BinOp) for part in ast.walk(node))
                )
            ):
                literals.append({"line": node.lineno, "text": excerpt})
        documents.append(
            _document(
                f"tool_text:{name}",
                "tool_text",
                path,
                data_root,
                literals,
                source_path=f"../src/tau2/domains/{domain}/{name}",
            )
        )
    return documents


def _airport_contexts(documents: list[SourceDocument]) -> dict[str, str]:
    """Carry actual frozen-task geography constraints into the joint airport draft."""
    database = next(
        document.content for document in documents if document.id == "database"
    )
    airports = {
        flight[key]
        for flight in database["flights"].values()
        for key in ("origin", "destination")
    }
    directory = {}
    for document in documents:
        if document.id != "tool_text:tools.py":
            continue
        lines = {}
        for literal in document.content:
            lines.setdefault(literal["line"], []).append(literal["text"])
        for values in lines.values():
            codes = [value for value in values if value in airports]
            cities = [value for value in values if value not in airports]
            if len(codes) == len(cities) == 1:
                directory[codes[0]] = cities[0]
    code_pattern = re.compile(r"\b(?:" + "|".join(sorted(airports)) + r")\b")
    relation_pattern = re.compile(
        r"convenient|either|do not accept|nearby|west coast|consider both",
        re.IGNORECASE,
    )
    snippets = []
    tasks = next(document.content for document in documents if document.id == "tasks")
    for task in tasks:
        instructions = task["user_scenario"]["instructions"]
        if not isinstance(instructions, dict):
            continue
        for field in ("reason_for_call", "task_instructions"):
            for paragraph in (instructions.get(field) or "").split("\n\n"):
                if code_pattern.search(paragraph) and relation_pattern.search(
                    paragraph
                ):
                    snippets.append(
                        {
                            "task_id": task["id"],
                            "field": field,
                            "source": paragraph.strip(),
                        }
                    )
    evidence = json.dumps(snippets, ensure_ascii=False)
    return {
        code: (
            f"Source airport {code}; directory city={directory.get(code, 'not specified')}. "
            "Choose all source airports jointly as distinct regional graph nodes. Preserve nearby-airport choices, "
            "explicit rejected/accepted airports and regional destination sets. Different listed city labels may "
            "still be grouped by a task. Adapt the associated city/catchment wording coherently; do not collapse codes. "
            f"Frozen task relationship evidence (exact source excerpts): {evidence}"
        )
        for code in airports
    }


class _Inventory:
    """Private builder; only validated SourceEntry objects leave this module."""

    def __init__(self, domain: str, database: dict):
        self.domain = domain
        self.database = database
        self.entries: dict[str, SourceEntry] = {}
        self.airport_contexts: dict[str, str] = {}
        self.people: dict[tuple, list[str]] = {}
        self.retail_lookup: dict[tuple, list[str]] = {}
        self.telecom_names: dict[str, list[str]] = {}
        self.accounts = dict(database.get("users", {}))
        self.accounts.update(
            {c["customer_id"]: c for c in database.get("customers", [])}
        )
        for user_id, user in database.get("users", {}).items():
            name = user["name"]
            person = (name["first_name"], name["last_name"], user.get("dob"))
            self.people.setdefault(person, []).append(user_id)
            key = (name["first_name"], name["last_name"], user["address"]["zip"])
            self.retail_lookup.setdefault(key, []).append(user_id)
        for customer in database.get("customers", []):
            self.telecom_names.setdefault(customer["full_name"], []).append(
                customer["customer_id"]
            )

    def add(
        self,
        document_id: str,
        path: list,
        value: JsonValue,
        category: str,
        semantic_key: str,
        context: str,
    ):
        digest = json_content_digest([category, semantic_key, value])
        entry_id = f"{category}:{digest}"
        occurrence = SourceOccurrence(document_id=document_id, path=path)
        if entry_id in self.entries:
            self.entries[entry_id].occurrences.append(occurrence)
        else:
            self.entries[entry_id] = SourceEntry(
                id=entry_id,
                category=category,
                source=value,
                occurrences=[occurrence],
                context=context,
            )

    def person(self, value: dict, owner: str) -> str:
        if {"first_name", "last_name", "dob"} <= value.keys():
            person = (value["first_name"], value["last_name"], value["dob"])
            matches = self.people.get(person, [])
            if owner in matches:
                return owner
            if len(matches) == 1:
                return matches[0]
            # DOB distinguishes unrelated people sharing a display name. Ambiguous
            # duplicate users remain attached to their reservation/account owner.
            scope = owner if matches else ""
            return f"passenger:{scope}:{json_content_digest(list(person))}"
        return owner

    def task_owner(self, task: dict) -> str:
        ids = set()
        lookups = set()

        def collect(value):
            if isinstance(value, dict):
                for field in ("user_id", "customer_id"):
                    if isinstance(value.get(field), str):
                        ids.add(value[field])
                if {"first_name", "last_name", "zip"} <= value.keys():
                    key = (value["first_name"], value["last_name"], value["zip"])
                    lookups.update(self.retail_lookup.get(key, []))
                for child in value.values():
                    collect(child)
            elif isinstance(value, list):
                for child in value:
                    collect(child)

        collect(task)
        if len(ids) == 1:
            return next(iter(ids))
        if len(lookups) == 1:
            return next(iter(lookups))
        return f"task:{task['id']}"

    def walk(
        self,
        document_id: str,
        value: JsonValue,
        path: list,
        owner: str = "",
        product: str = "",
        person: str = "",
        container: dict | None = None,
    ):
        field = path[-1] if path else ""
        parent = path[-2] if len(path) > 1 else ""
        if isinstance(value, dict) and value:
            owner = value.get("user_id", value.get("customer_id", owner))
            product = value.get("product_id", product)
            person = self.person(value, owner)
            if document_id == "tasks" and len(path) == 1:
                owner = self.task_owner(value)
                person = owner
            if field in _PROTOCOL_SUBTREES and field not in ("orders", "reservations"):
                self.add(
                    document_id,
                    path,
                    value,
                    "protocol",
                    str(field),
                    "Preserve protocol structure and numeric task conditions exactly.",
                )
                return
            # name and address leaves are intentionally separate: flat lookup tool
            # arguments must share exact occurrences with nested DB fields. Their
            # account key keeps each coherent entity distinct from namesakes.
            for key, child in value.items():
                child_owner = key if field == "users" else owner
                self.walk(
                    document_id,
                    child,
                    [*path, key],
                    child_owner,
                    product,
                    person,
                    value,
                )
            return
        if isinstance(value, list) and value:
            if field in _PROTOCOL_SUBTREES:
                self.add(
                    document_id,
                    path,
                    value,
                    "protocol",
                    str(field),
                    "Preserve ID references, order and scoring protocol exactly.",
                )
                return
            for index, child in enumerate(value):
                self.walk(
                    document_id,
                    child,
                    [*path, index],
                    owner,
                    product,
                    person,
                    container,
                )
            return
        category, key, context = (
            "protocol",
            str(field),
            "Preserve opaque IDs, enums, dates, counts and domain control values exactly.",
        )
        account = self.accounts.get(owner, {})
        original_identity = dict(account.get("name", {}))
        for name_field in (
            "full_name",
            "dob",
            "date_of_birth",
            "email",
            "phone_number",
        ):
            if name_field in account:
                original_identity[name_field] = account[name_field]
        original_identity.update(
            {
                k: v
                for k, v in (container or {}).items()
                if k in {"first_name", "last_name", "dob", "full_name"}
            }
        )
        original_address = dict(account.get("address", {}))
        original_address.update(
            {k: v for k, v in (container or {}).items() if k in _ADDRESS_FIELDS}
        )
        identity_context = json.dumps(
            original_identity, ensure_ascii=False, sort_keys=True
        )
        address_context = json.dumps(
            original_address, ensure_ascii=False, sort_keys=True
        )
        if value is not None and isinstance(value, str):
            if field in ("first_name", "last_name"):
                category, key = "identity_name", f"{person or owner}:{field}"
                context = f"role={field}; entity={person or owner}; source_identity={identity_context}. Keep this person's components coherent across DB, saved passengers, reservations and task lookups."
            elif field == "full_name" or (
                field == "name"
                and self.domain == "telecom"
                and value in self.telecom_names
            ):
                matches = self.telecom_names.get(value, [])
                identity = matches[0] if len(matches) == 1 else owner
                category, key = "identity_name", f"{identity}:full_name"
                context = f"role=full_name; entity={identity}; source_identity={identity_context}. Preserve normal phone/name authentication behavior."
            elif field in _ADDRESS_FIELDS:
                category, key = "address", f"{owner}:{field}"
                if self.domain == "retail" and field == "state":
                    # One customer can ship to several cities in the same source
                    # state. Their target cities may belong to different regions.
                    key += ":" + json_content_digest(
                        [original_address.get("city"), original_address.get("country")]
                    )
                context = f"role={field}; entity={owner}; source_address={address_context}; source_identity={identity_context}. Generate this complete address coherently; retain intentionally different old/new or wrong/correct values."
            elif field in ("email", "phone_number"):
                category, key = "contact", str(field)
                context = f"role={field}; entity={owner}; source_identity={identity_context}; source_address={address_context}. Identical values link customer, owned lines, task initialization and lookups; preserve distinct values and lookup success/failure."
            elif field in ("origin", "destination"):
                category, key = "airport", "airport"
                context = self.airport_contexts[value]
            elif field == "name" and product:
                category, key = "product_name", product
                context = f"Product {product}; catalog and copied order item display name. Preserve the product category and task intent."
            elif parent == "options":
                category, key = "product_option", str(field)
                context = f"Product option {field}; keys are protocol and must stay unchanged. Preserve sizes, units, quantities and variant distinctions."
            elif field == "name" and "plans" in path:
                category, key, context = (
                    "plan",
                    "plan_name",
                    "Plan display name; do not change plan IDs, capacities or relative price relationships.",
                )
            elif field == "model" and "devices" in path:
                category, key, context = (
                    "device",
                    "device_model",
                    "Device model display name; retain technical capabilities and device IDs.",
                )
            elif field == "description" and "line_items" in path:
                category, key, context = (
                    "billing",
                    str(field),
                    "Bill description; synchronize embedded plan names and phone references with the structured mappings.",
                )
            elif field in _PROSE_FIELDS or parent in (
                "communicate_info",
                "nl_assertions",
            ):
                category, key = "prose", f"{owner}:{field}"
                context = f"Scenario/evaluation prose for {owner}; author AFTER structured mappings. Preserve goals, authentication, wrong-first/correct-later values, numeric conditions and policy semantics."
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and (field in _MONEY_FIELDS or parent == "prices")
        ):
            category, key, context = (
                "money",
                "money",
                "Apply the fixed benchmark currency scale exactly; keep int/float type, balances, totals, affordability and relative prices consistent.",
            )
        self.add(document_id, path, value, category, key, context)


def inventory_localization(domain: str, data_root: Path) -> LocalizationSource:
    """Snapshot complete canonical DBs, frozen50 tasks, policies and tool evidence.

    Every value has exactly one source-checked owner. Unclassified values are
    explicit preserve-only protocol entries rather than guessed translations.
    This inventories source; it does not claim generated content is reviewed.
    """
    if domain == "intake":
        from tau2.domains.intake.localization import inventory_intake

        return inventory_intake(Path(data_root))
    if domain == "banking_knowledge":
        from tau2.domains.banking_knowledge.localization import inventory_banking

        return inventory_banking(Path(data_root))
    if domain not in ("airline", "retail", "telecom"):
        raise ValueError(f"Unsupported regional localization domain: {domain}")
    documents = _source_documents(domain, Path(data_root))
    inventory = _Inventory(domain, documents[0].content)
    if domain == "airline":
        inventory.airport_contexts = _airport_contexts(documents)
    for document in documents:
        if document.kind == "policy":
            inventory.add(
                document.id,
                [],
                document.content,
                "policy",
                document.id,
                "Domain policy; regional facts and currency/time-zone references must match structured mappings without changing authentication or policy obligations.",
            )
        elif document.kind == "tool_text":
            for index, literal in enumerate(document.content):
                inventory.add(
                    document.id,
                    [index, "line"],
                    literal["line"],
                    "protocol",
                    "source_line",
                    "Source line provenance; preserve.",
                )
                inventory.add(
                    document.id,
                    [index, "text"],
                    literal["text"],
                    "tool_text",
                    f"{document.id}:{literal['line']}",
                    "Code literal evidence: requires explicit runtime wiring; map compilation alone does not change Python tools, schemas or fee calculations. Protocol literals must remain unchanged.",
                )
        else:
            inventory.walk(document.id, document.content, [])
    return LocalizationSource(
        domain=domain,
        documents=documents,
        entries=list(inventory.entries.values()),
    )
