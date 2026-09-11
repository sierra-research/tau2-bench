# Copyright Sierra
"""Pure, importable core of the translator round-trip tooling.

`the retired translator CSV round-trip` is the CLI face of localization; this module is
its engine, extracted so the Language Factory (automated translation with
verification) can drive the same flatten/inject logic programmatically. Every
function here is pure with respect to the process: no ``sys.exit``, no
printing — file paths in, data structures out, problems returned as
human-readable strings (the same convention as
``tau2.multilingual.invariants``).

The CSV format (one row per translatable field per task; ``(metadata)`` rows
are description prose never spoken aloud) must not change shape: external
translators hold filled copies of it.
"""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from typing import Iterator

from tau2.config import DEFAULT_MULTILINGUAL_DOMAIN
from tau2.multilingual.domain_profiles import (
    DOMAIN_PROFILES,
    CallerIdentityKind,
)
from tau2.multilingual.invariants import (
    INSTRUCTION_FIELDS,
    AddressProseFormat,
    IdentityRename,
    caller_address_variant,
    caller_email_variant_pairs,
    caller_set_user_info,
    caller_zip_variant_pairs,
    check_task_set_localization,
    effective_customer_records,
    expected_eval_criteria,
    find_caller_user_id,  # noqa: F401  (re-exported: callers import it from here)
    localized_address_variant,
    phone_digits,
    resolve_caller_user_id,
)
from tau2.multilingual.names import NameOrder, compose_full_name
from tau2.utils import load_file

METADATA_FIELDS = {"purpose": "purpose (metadata)", "notes": "notes (metadata)"}
LABEL_TO_FIELD = {v: k for k, v in METADATA_FIELDS.items()}

# Header of the translator CSV written by `extract` and read by `inject`.
TRANSLATION_CSV_HEADER = [
    "row_id",
    "task_id",
    "field",
    "english_original",
    "translation",
]


def data_dir() -> Path:
    """The CURRENT ``tau2.utils.DATA_DIR`` (resolved dynamically, never
    snapshotted at import time, so isolated test environments work).

    THE single data-dir accessor for the multilingual pillar — every path
    helper below and in the factory resolves through it.
    """
    import tau2.utils

    return tau2.utils.DATA_DIR


def multilingual_data_dir() -> Path:
    """The CURRENT ``data/tau2/multilingual`` directory."""
    return data_dir() / "tau2" / "multilingual"


def load_domain_tasks(domain: str) -> list[dict]:
    """The domain's English SOURCE tasks (raw dicts) — the set translations,
    identity variants, and the per-language arm files derive from.

    The file is the domain profile's ``source_tasks_filename`` (airline ships
    a curated ``tasks.json``; telecom's ``tasks_multilingual.json`` is the
    seed split ``tau2 factory seed-tasks`` materializes). Unprofiled domains
    (test fixtures) fall back to ``tasks.json``.
    """
    profile = DOMAIN_PROFILES.get(domain)
    filename = profile.source_tasks_filename if profile else "tasks.json"
    path = data_dir() / "tau2" / "domains" / domain / filename
    if not path.exists() and profile and filename != "tasks.json":
        raise FileNotFoundError(
            f"No multilingual source task set at {path}; run "
            f"`tau2 factory seed-tasks --domain {domain}` to materialize it."
        )
    with open(path) as fp:
        loaded = json.load(fp)
    if isinstance(loaded, dict):
        # Some domain task files wrap the list as {"tasks": [...]}.
        return loaded["tasks"]
    return loaded


def load_domain_db(domain: str) -> dict:
    """The domain's AGENT-side database, as a raw dict.

    Single owner of domain-DB loading for the multilingual pillar: probes
    ``db.json`` then ``db.toml`` (telecom's DB is TOML), exactly like the
    identity-rename resolution in ``english_prompts`` — which delegates here.
    """
    domain_dir = data_dir() / "tau2" / "domains" / domain
    for name in ("db.json", "db.toml"):
        path = domain_dir / name
        if path.exists():
            return load_file(path)
    raise FileNotFoundError(f"No domain DB (db.json / db.toml) under {domain_dir}")


def iter_rows(task: dict) -> Iterator[tuple[str, str]]:
    """(field_label, text) pairs for every translatable, non-empty field."""
    instructions = task["user_scenario"]["instructions"]
    for field in INSTRUCTION_FIELDS:
        if instructions.get(field):
            yield field, instructions[field]
    description = task.get("description") or {}
    for field, label in METADATA_FIELDS.items():
        if description.get(field):
            yield label, description[field]


def resolve_script_code(lang: str, override: str | None = None) -> str:
    """ISO 15924 script code for a language: the override, or the pack's.

    Raises:
        ValueError: If no override is given and no pack (or no pack persona
            script) is registered for the language.
    """
    if override:
        return override
    from tau2.multilingual.registry import get_language_pack

    pack = get_language_pack(lang)
    if pack is None:
        raise ValueError(
            f"No language pack registered for '{lang}' yet — pass "
            "--script-code (ISO 15924, e.g. hans, deva) explicitly."
        )
    scripts = sorted({p.script for p in pack.personas.values() if p.script})
    if not scripts:
        raise ValueError(
            f"Pack '{lang}' declares no script codes — pass --script-code."
        )
    return scripts[0]


def write_translation_csv(tasks: list[dict], out_path: Path) -> int:
    """English tasks -> translator CSV with an empty `translation` column.

    Returns the number of data rows written.
    """
    with open(out_path, "w", newline="", encoding="utf-8-sig") as fp:
        writer = csv.writer(fp)
        writer.writerow(TRANSLATION_CSV_HEADER)
        row_id = 0
        for task in tasks:
            for label, text in iter_rows(task):
                row_id += 1
                writer.writerow([row_id, task["id"], label, text, ""])
    return row_id


def read_filled_csv(path: Path) -> dict[tuple[str, str], str]:
    """Filled translator CSV -> {(task_id, field_label): translation}."""
    translations: dict[tuple[str, str], str] = {}
    with open(path, encoding="utf-8-sig") as fp:
        for row in csv.DictReader(fp):
            translations[(row["task_id"], row["field"])] = (
                row.get("translation") or ""
            ).strip()
    return translations


def inject_translations(
    source_tasks: list[dict],
    translations: dict[tuple[str, str], str],
    lang: str,
    script_code: str,
) -> tuple[list[dict], list[str]]:
    """Build the localized task set and check it (pure core of `inject`).

    Applies each translation onto a deep copy of its source task (ids become
    ``<source_id>_<lang>``) and runs the shared localization invariants.

    Returns:
        (localized_tasks, problems). ``problems`` is empty iff the set is
        complete and invariant-clean; callers decide whether to write.
    """
    missing: list[str] = []
    localized: list[dict] = []
    for task in source_tasks:
        out = copy.deepcopy(task)
        out["id"] = f"{task['id']}_{lang}"
        for label, _text in iter_rows(task):
            translated = translations.get((task["id"], label), "")
            if not translated:
                missing.append(f"task {task['id']}: field '{label}' not translated")
                continue
            if label in LABEL_TO_FIELD:
                out["description"][LABEL_TO_FIELD[label]] = translated
            else:
                out["user_scenario"]["instructions"][label] = translated
        localized.append(out)

    problems = missing + check_task_set_localization(
        localized, source_tasks, suffix=lang, script_code=script_code
    )
    return localized, problems


def write_task_set(localized: list[dict], out_path: Path) -> None:
    """Write a localized task set JSON exactly as the CLI tool does."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fp:
        json.dump(localized, fp, ensure_ascii=False, indent=2)
        fp.write("\n")


# ---------------------------------------------------------------------------
# derive-names engine (identity-variant task sets)
# ---------------------------------------------------------------------------


def derive_identity_tasks(
    localized_tasks: list[dict],
    source_tasks: list[dict],
    db: dict,
    identity_map: dict[str, dict],
    lang: str,
    script_code: str | None,
    *,
    domain: str = DEFAULT_MULTILINGUAL_DOMAIN,
    instruction_template: str | None = None,
    name_order: NameOrder = NameOrder.GIVEN_FIRST,
    address_format: AddressProseFormat | None = None,
    variant: str = "identity",
) -> tuple[list[dict], list[str]]:
    """Identity-variant ("_identity") task set from a localized set + identity map.

    Dispatches on the domain profile's ``caller_identity`` kind:

    - ``USER_ID_HANDLE`` (airline, and unprofiled test domains): the caller is
      swapped to a locale identity — name, user_id, email, and (when present)
      address — declared via ``initial_state.initialization_data``: the full
      localized user record under the NEW id, plus every matching passenger
      entry on the caller's reservations re-pointed to the new id.
      ``identity_map`` is keyed by the ORIGINAL caller user id; each entry:
      ``{user_id, first_name, last_name, email, gender, address?}``.
    - ``STRUCTURED_NAME_PHONE`` (telecom): the caller keeps their customer id
      AND phone number; name and email become a locale identity in
      ``set_user_info``, prose/ticket, and the customer record of a wholesale
      ``agent_data.customers`` list patch. The phone number stays canonical:
      it is a benchmark-wide environment key (every task's caller line is the
      same number), not per-caller identity content, and the line tools
      resolve it only in its canonical form. ``identity_map`` is keyed by the
      English caller full name; each entry carries ``first_name``,
      ``last_name``, ``full_name``, ``email``, ``phone_number`` (unused), and
      ``gender``.

    In both shapes the prose carries the new identity and
    ``evaluation_criteria`` is rebuilt with :func:`expected_eval_criteria`,
    which applies exactly that rename. All identity values are Latin
    (native == romanized). ``instruction_template`` is an optional string
    appended to ``task_instructions`` with {first}/{last}/{user_id}
    placeholders (e.g. a spelling aid); omit it for languages that don't need
    one. ``script_code=None`` skips the prose-is-localized checks (deferred-
    translation domains keep English prose — see
    ``DomainLocalizationProfile.tasks_translated``).

    ``name_order`` is the locale's DISPLAY name order (from its language pack).
    The English source is always given-first; the new caller's display name —
    in the prose, the agent-side record, and the rebuilt evaluation criteria —
    follows the locale, so a family-first language's caller says the same
    string the agent looks up.

    ``variant`` is the task-set variant token ('identity', or
    'identity_native' for the native-script DB ablation): it names the emitted
    ids ``<source_id>_<lang>_<variant>`` and the invariant-check suffix. The
    identity values themselves come from ``identity_map`` — a native-script
    map yields a native-script set through the same deriver ("all identity
    values are Latin" holds only for the default variant's folded maps).

    Returns:
        (derived_tasks, problems) — ``problems`` collects missing/unknown
        callers plus every localization-invariant violation found by
        ``check_task_set_localization`` (empty == clean; callers decide
        whether to write).
    """
    profile = DOMAIN_PROFILES.get(domain)
    kind = profile.caller_identity if profile else CallerIdentityKind.USER_ID_HANDLE
    if profile and not profile.tasks_translated:
        script_code = None
    if kind is CallerIdentityKind.STRUCTURED_NAME_PHONE:
        if variant != "identity":
            raise ValueError(
                f"Domain '{domain}' (STRUCTURED_NAME_PHONE) only derives the "
                f"default 'identity' variant; got {variant!r}."
            )
        return _derive_identity_tasks_structured(
            localized_tasks,
            source_tasks,
            db,
            identity_map,
            lang,
            script_code,
            name_order,
            domain=domain,
        )
    problems: list[str] = []
    out_tasks: list[dict] = []
    source_by_id = {task["id"]: task for task in source_tasks}
    for task in localized_tasks:
        source_id = task["id"].removesuffix(f"_{lang}")
        if source_id not in source_by_id:
            problems.append(
                f"task {task['id']}: no source task '{source_id}' "
                "in the domain task set"
            )
            continue
        caller = resolve_caller_user_id(task, db.get("users") or {})
        if caller is None:
            problems.append(f"task {task['id']}: no caller user id in known_info")
            continue
        if caller not in identity_map:
            problems.append(f"task {task['id']}: caller '{caller}' not in identity map")
            continue
        if caller not in db["users"]:
            problems.append(f"task {task['id']}: caller '{caller}' not in domain DB")
            continue

        entry = identity_map[caller]
        new_id = entry["user_id"]
        new_first, new_last = entry["first_name"], entry["last_name"]
        new_email = entry.get("email")
        user = db["users"][caller]
        old_first = user["name"]["first_name"]
        old_last = user["name"]["last_name"]
        old_email = user.get("email")

        out = copy.deepcopy(task)
        out["id"] = f"{source_id}_{lang}_{variant}"

        # Full localized user record under the NEW id (deep-merge adds the key;
        # the canonical caller record is left untouched and inert).
        new_user = copy.deepcopy(user)
        new_user["user_id"] = new_id
        new_user["name"] = {"first_name": new_first, "last_name": new_last}
        if new_email:
            new_user["email"] = new_email
        if entry.get("address"):
            new_user["address"] = entry["address"]
        agent_data: dict = {"users": {new_id: new_user}}

        reservations_patch = {}
        for res_id in user.get("reservations", []):
            reservation = db["reservations"][res_id]
            reservations_patch[res_id] = {
                "user_id": new_id,
                "passengers": [
                    (
                        {**p, "first_name": new_first, "last_name": new_last}
                        if p["first_name"] == old_first and p["last_name"] == old_last
                        else copy.deepcopy(p)
                    )
                    for p in reservation["passengers"]
                ],
            }
        if reservations_patch:
            agent_data["reservations"] = reservations_patch

        # Retail's orders are the analogue of airline's reservations: the
        # caller's own orders must follow them to the new id, or the DB
        # contradicts itself. This is not cosmetic — the retail tools resolve
        # a refund's payment method through `order.user_id`
        # (`_get_payment_method(order.user_id, ...)`), and `get_user_details`
        # on the new caller lists orders that would otherwise still be owned
        # by the canonical record. The shipping address moves too, but ONLY
        # where it was a copy of the caller's old default: an order shipped
        # somewhere else (200 of 1000 in the retail DB) is a deliberate
        # different destination, not identity material.
        orders_patch = {}
        old_address = user.get("address")
        new_address = entry.get("address")
        old_variant = caller_address_variant(source_by_id[source_id], old_address)
        for order_id in user.get("orders", []):
            order = db["orders"][order_id]
            order_patch: dict = {"user_id": new_id}
            if new_address and order.get("address") == old_address:
                order_patch["address"] = copy.deepcopy(new_address)
            orders_patch[order_id] = order_patch
        if orders_patch:
            agent_data["orders"] = orders_patch
        out["initial_state"] = {
            "initialization_data": {"agent_data": agent_data, "user_data": None},
            "initialization_actions": None,
            "message_history": None,
        }

        # Prose: swap the caller's old identity tokens for the new ones. Names,
        # ids, and emails survive translation verbatim (Latin), so a plain
        # replace on the localized prose is exact. The swap itself lives on
        # IdentityRename, shared with english-prompt mode, which re-derives it
        # at run time and must land on the same text (see apply_to_prose).
        instructions = out["user_scenario"]["instructions"]
        rename = IdentityRename(
            old_user_id=caller,
            new_user_id=new_id,
            old_name=(old_first, old_last),
            new_name=(new_first, new_last),
            # The source name is English (given-first); the locale name takes
            # the pack's display order.
            old_full_name=compose_full_name(old_first, old_last),
            new_full_name=compose_full_name(new_first, new_last, name_order),
            old_email=old_email,
            new_email=new_email,
            old_address=old_address,
            new_address=new_address,
            old_address_variant=old_variant,
            new_address_variant=localized_address_variant(
                old_address, new_address, old_variant
            ),
            email_variant_pairs=caller_email_variant_pairs(
                source_by_id[source_id],
                old_email,
                new_email,
                {
                    user.get("email")
                    for user in db["users"].values()
                    if user.get("email")
                },
            ),
            zip_variant_pairs=caller_zip_variant_pairs(
                source_by_id[source_id],
                (old_address or {}).get("zip"),
                (new_address or {}).get("zip"),
            ),
        )
        if old_variant and rename.new_address_variant is None:
            # The near-miss the task turns on could not be carried into the
            # locale (no single digit run to shift) — shipping without it
            # leaves an English address in localized prose and eval criteria,
            # so refuse loudly rather than degrade silently.
            problems.append(
                f"task {out['id']}: the near-miss address this task turns on "
                f"({old_variant.get('address1')!r}/{old_variant.get('address2')!r}) "
                "has no localized variant — the locale street line must carry "
                "exactly one number to shift"
            )
        # unknown_info too: today no retail caller is anchored there, but the
        # english-prompt producer renames it, and the two must stay in
        # lockstep (see IdentityRename.apply_to_prose).
        for field in (
            "task_instructions",
            "reason_for_call",
            "known_info",
            "unknown_info",
        ):
            if instructions.get(field):
                instructions[field] = rename.apply_to_prose(
                    instructions[field], kind, address_format
                )
        # A PROSE_NAME_ZIP caller may be anchored by name+zip or email with no
        # handle in the prose at all (91 of retail's 114 tasks), so the new id
        # legitimately does not appear. The rename is still verified — by the
        # rebuilt evaluation_criteria and the invariant sweep below.
        if kind is not CallerIdentityKind.PROSE_NAME_ZIP and new_id not in (
            instructions.get("known_info") or ""
        ):
            problems.append(
                f"task {out['id']}: new user id '{new_id}' missing from "
                f"known_info after rename (old id '{caller}' not verbatim)"
            )
        if instruction_template:
            instructions["task_instructions"] = (
                instructions.get("task_instructions") or ""
            ) + instruction_template.format(
                first=new_first, last=new_last, user_id=new_id
            )

        out["evaluation_criteria"] = expected_eval_criteria(
            source_by_id[source_id], out, db, name_order
        )
        out_tasks.append(out)

    problems += check_task_set_localization(
        out_tasks,
        source_tasks,
        suffix=f"{lang}_{variant}",
        script_code=script_code,
        domain_db=db,
        name_order=name_order,
        domain=domain,
    )
    return out_tasks, problems


def customer_by_phone(db: dict, phone_number: str) -> dict | None:
    """The DB customer whose ``phone_number`` matches (digit-sequence match)."""
    wanted = phone_digits(phone_number)
    if not wanted:
        return None
    for customer in db.get("customers") or []:
        if phone_digits(customer.get("phone_number") or "") == wanted:
            return customer
    return None


def _derive_identity_tasks_structured(
    localized_tasks: list[dict],
    source_tasks: list[dict],
    db: dict,
    identity_map: dict[str, dict],
    lang: str,
    script_code: str | None,
    name_order: NameOrder = NameOrder.GIVEN_FIRST,
    *,
    domain: str = DEFAULT_MULTILINGUAL_DOMAIN,
) -> tuple[list[dict], list[str]]:
    """The STRUCTURED_NAME_PHONE (telecom) branch of :func:`derive_identity_tasks`.

    The caller's name and email become a locale identity. The customer id,
    line id, PHONE NUMBER, and every other concrete value stay canonical.
    The phone number is deliberately NOT localized: it is a benchmark-wide
    constant environment key (every task's caller line is the same canonical
    number), not per-caller identity content, and the agent-facing line tools
    only resolve it in canonical form — a localized number would make
    line-scoped tasks (the data_usage_exceeded family) unsolvable. Identity
    verification, where the tasks exercise it, flows through name + DOB.

    The OLD identity is the source task's EFFECTIVE caller record
    (the diversified seed set patches a pool caller into every task, so the
    DB record is only a validity anchor for the phone number), and the locale
    rename COMPOSES on top: the new customers patch is the effective list with
    just the caller's full_name/email swapped (DOB, phone, and everything else
    carried through), merged into any initialization_data the source already
    ships. The rename lands in three places: the ``set_user_info``
    initialization action, the prose (instructions + ticket), and the caller's
    record in the ``agent_data.customers`` patch so authentication, profile
    readback, and line tools all agree without duplicating the domain's
    wholesale lines list into every task.
    """
    problems: list[str] = []
    out_tasks: list[dict] = []
    source_by_id = {task["id"]: task for task in source_tasks}
    for task in localized_tasks:
        source_id = task["id"].removesuffix(f"_{lang}")
        if source_id not in source_by_id:
            problems.append(
                f"task {task['id']}: no source task '{source_id}' "
                "in the domain task set"
            )
            continue
        caller = caller_set_user_info(task)
        if caller is None:
            problems.append(
                f"task {task['id']}: no set_user_info initialization action "
                "with name + phone_number"
            )
            continue
        if customer_by_phone(db, caller["phone_number"]) is None:
            problems.append(
                f"task {task['id']}: caller phone '{caller['phone_number']}' "
                "matches no customer in the domain DB"
            )
            continue
        # The record the task actually establishes at run time — the source's
        # effective customers list, matched by the caller's phone.
        wanted_phone = phone_digits(caller["phone_number"])
        effective = effective_customer_records(source_by_id[source_id], db)
        customer = next(
            (
                record
                for record in effective
                if phone_digits(record.get("phone_number") or "") == wanted_phone
            ),
            None,
        )
        if customer is None:
            problems.append(
                f"task {task['id']}: caller phone '{caller['phone_number']}' "
                "matches no record in the source task's effective customers"
            )
            continue
        english_name = caller["name"]
        if english_name not in identity_map:
            problems.append(
                f"task {task['id']}: caller '{english_name}' not in identity map"
            )
            continue

        entry = identity_map[english_name]
        new_full = entry["full_name"]
        new_email = entry.get("email")
        old_full = customer["full_name"]
        old_email = customer.get("email")

        out = copy.deepcopy(task)
        out["id"] = f"{source_id}_{lang}_identity"

        # The caller announces the locale name to the simulator; the phone
        # number stays the canonical one already in set_user_info.
        for action in (out["initial_state"] or {}).get("initialization_actions") or []:
            if action.get("func_name") == "set_user_info":
                action["arguments"]["name"] = new_full

        # Customers-list patch composed over the EFFECTIVE list (list patches
        # replace, so every record ships; only the caller's is renamed — the
        # seed diversification's DOB/email-bearing record carries through).
        # Agent-side name+DOB lookup and profile readback then agree with
        # what the caller says.
        patched_customers = [
            (
                {
                    **copy.deepcopy(record),
                    "full_name": new_full,
                    **({"email": new_email} if new_email else {}),
                }
                if record["customer_id"] == customer["customer_id"]
                else copy.deepcopy(record)
            )
            for record in effective
        ]
        # MERGE into any initialization_data the source already carries
        # (diversified seed tasks all do) — replace only the customers list,
        # preserving other agent_data keys and user_data.
        init_data = copy.deepcopy(out["initial_state"].get("initialization_data") or {})
        agent_data = init_data.get("agent_data") or {}
        agent_data["customers"] = patched_customers
        init_data["agent_data"] = agent_data
        init_data.setdefault("user_data", None)
        out["initial_state"]["initialization_data"] = init_data

        # Prose: the caller's old name/email tokens for the new ones (the
        # phone number stays canonical, so no phone pair). Identity literals
        # survive translation verbatim, so a plain replace is exact.
        # Near-miss decoy emails ride along exactly as the run-time producer
        # applies them (IdentityRename.string_pairs) — the two must land on
        # the same text or the consistency check fails sims.
        pairs = [(old_full, new_full)]
        if old_email and new_email:
            pairs.append((old_email, new_email))
        pairs.extend(
            caller_email_variant_pairs(
                source_by_id[source_id],
                old_email,
                new_email,
                {record.get("email") for record in effective if record.get("email")},
            )
        )
        instructions = out["user_scenario"]["instructions"]
        for field in INSTRUCTION_FIELDS:
            if instructions.get(field):
                for old, new in pairs:
                    instructions[field] = instructions[field].replace(old, new)
        if out.get("ticket"):
            for old, new in pairs:
                out["ticket"] = out["ticket"].replace(old, new)
        if new_full not in (instructions.get("known_info") or ""):
            problems.append(
                f"task {out['id']}: new caller name '{new_full}' missing from "
                f"known_info after rename (old name '{old_full}' not verbatim)"
            )

        out["evaluation_criteria"] = expected_eval_criteria(
            source_by_id[source_id], out, db, name_order
        )
        out_tasks.append(out)

    problems += check_task_set_localization(
        out_tasks,
        source_tasks,
        suffix=f"{lang}_identity",
        script_code=script_code,
        domain_db=db,
        name_order=name_order,
        domain=domain,
    )
    return out_tasks, problems
