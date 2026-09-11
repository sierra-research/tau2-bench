# Copyright Sierra
"""Deterministic caller diversification for generated-pool seed sets.

Generated pools (telecom) hold every scenario constant except the workflow
under test — including the caller: all 2285 pool tasks are John Smith. That
is the right call for pool generation and the wrong one for the multilingual
benchmark, where caller identity is experimental material (name/date entity
round-trips, gender-matched voices, name-vs-phone authentication). This
module rewrites the seed split so each task carries one of the reviewed
identities in ``data/tau2/domains/<domain>/caller_pool.yaml``:

- Every diversified caller is a RENAME of the pool's single canonical caller:
  customer_id, phone number, lines, and bills stay canonical, and the task
  ships the renamed customer record via a wholesale ``initialization_data``
  customers patch (addict-style list patches replace wholesale). ``db.toml``
  is never edited.
- Assignment is positional and deterministic: task ``i`` gets pool caller
  ``i % len(pool)`` and auth mode ``name_dob`` when ``i`` is odd — over the
  114-task telecom split that is an exact 57/57 auth split, and the pool's
  reviewed gender ordering makes the task-level gender split 57/57 too.
- ``phone`` mode keeps the canonical phone in ``known_info`` (pure name
  substitution). ``name_dob`` mode replaces the phone clause with the
  caller's date of birth in natural English — the agent must elicit it and
  convert it to ISO for ``get_customer_by_name``.
- ``name_dob`` mode also appends a fixed LINE-IDENTIFICATION clause to
  ``task_instructions``: the caller knows which line they are calling about
  and will name it if asked, but declines to authenticate by phone number.
  Without it the arm is unsolvable rather than hard — the canonical caller
  owns three lines, and with the phone withheld entirely nothing in the
  conversation, the tools, or the policy identifies which line the call is
  about (``customer.phone_number`` matches a line for C1001 alone, and for
  no other customer in the DB). Measured over 24 telecom runs, name-auth
  calls acted on a line the ground truth never names 39.8% of the time
  against the phone arm's 9.1%. The clause restores line identification
  without weakening authentication: the number is withheld from the lookup
  path, so ``get_customer_by_name`` stays the only way in.
- The ticket keeps the canonical phone number in BOTH modes: it is agent-side
  context consumed only by the solo agent, never by the conversational arms,
  so it cannot leak the phone to a name-auth user sim.

The locale identity swap (``tau2 factory localize-entities``) composes on
top: it renames each task's EFFECTIVE caller record (the one patched here)
to a locale identity, inheriting the pool caller's gender and date of birth.
"""

from __future__ import annotations

import copy
import datetime
from enum import Enum
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from tau2.multilingual.factory.author_form import FactoryDraftError
from tau2.multilingual.factory.name_genders import source_caller_gender
from tau2.multilingual.invariants import (
    caller_set_user_info,
    expected_eval_criteria,
    phone_digits,
)
from tau2.multilingual.localize_lib import data_dir

CALLER_POOL_FILENAME = "caller_pool.yaml"

# Plausible-adult DOB bounds for pool validation (inclusive years).
DOB_MIN_YEAR = 1940
DOB_MAX_YEAR = 2005

# Fixed known_info templates (machine-not-scripts: the prose the seed set
# ships is this code, never improvised content).
PHONE_AUTH_KNOWN_INFO_TEMPLATE = "You are {full_name} with phone number {phone_number}."
NAME_AUTH_KNOWN_INFO_TEMPLATE = (
    "You are {full_name} and your date of birth is {dob_natural}."
)
# Line identification for the name-auth arm. Lives in task_instructions, not
# known_info: known_info is the identity material the caller leads with, and
# this number's use is conditional — it answers "which line?", never "who are
# you?". Keeping it out of known_info also leaves the auth-mode derivation in
# `task_auth_mode` (phone digits in known_info) exact.
#
# The refusal offers the NAME only, never "name and date of birth": a real
# caller offers a name and lets the agent drive verification, and knowing that
# name lookup requires a date of birth is the AGENT's job (main_policy.md:
# "For name lookup, date of birth is required for verification purposes") —
# pre-announcing it hands over the procedure under test. Nothing is lost: the
# caller's known_info already carries the natural-English date of birth, so
# they answer when asked, and `get_customer_by_name(full_name, dob)` takes dob
# as a required argument, so the agent cannot skip the step.
NAME_AUTH_LINE_CLAUSE_TEMPLATE = (
    "The phone line you are calling about is {phone_number}. If the agent "
    "asks for your phone number in order to look you up or verify your "
    "identity, say you would rather be looked up by your name, even if the "
    "agent insists. Give the number only when the agent asks which line or "
    "which phone number your issue concerns."
)


class AuthMode(str, Enum):
    """How a seed task's user sim can identify itself to the agent."""

    PHONE = "phone"
    """known_info carries the canonical phone number (primary lookup path)."""

    NAME_DOB = "name_dob"
    """known_info carries name + date of birth; the phone is unknown, forcing
    the ``get_customer_by_name`` verification path."""


# Month names hardcoded rather than strftime('%B'): rendering must not depend
# on the process locale — a non-English LC_TIME would ship non-English seed
# prose and break byte-idempotent regeneration.
_MONTH_NAMES = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)


def natural_date(iso_date: str) -> str:
    """``1987-03-21`` -> ``March 21, 1987`` (what the user sim knows/says)."""
    date = datetime.date.fromisoformat(iso_date)
    return f"{_MONTH_NAMES[date.month - 1]} {date.day}, {date.year}"


class CallerRecord(BaseModel):
    """One reviewed caller identity from ``caller_pool.yaml``."""

    model_config = ConfigDict(extra="forbid")

    full_name: Annotated[
        str, Field(description="Given + family name, e.g. 'Olivia Bennett'")
    ]
    gender: Annotated[
        Literal["male", "female"],
        Field(
            description="Voice-routing gender; must agree with the given "
            "name's GIVEN_NAME_GENDERS entry"
        ),
    ]
    email: Annotated[str, Field(description="Contact email, unique in the pool")]
    date_of_birth: Annotated[
        str,
        Field(
            description="ISO YYYY-MM-DD; a real calendar date in the "
            "plausible adult range"
        ),
    ]

    @model_validator(mode="after")
    def _validate(self) -> "CallerRecord":
        if len(self.full_name.split()) < 2:
            raise ValueError(f"'{self.full_name}' is not a full name")
        try:
            dob = datetime.date.fromisoformat(self.date_of_birth)
        except ValueError as exc:
            raise ValueError(
                f"{self.full_name}: date_of_birth {self.date_of_birth!r} is "
                "not a valid ISO calendar date"
            ) from exc
        if not DOB_MIN_YEAR <= dob.year <= DOB_MAX_YEAR:
            raise ValueError(
                f"{self.full_name}: date_of_birth {self.date_of_birth} is "
                f"outside the plausible adult range "
                f"{DOB_MIN_YEAR}-{DOB_MAX_YEAR}"
            )
        catalog_gender = source_caller_gender(self.full_name.split()[0])
        if catalog_gender != self.gender:
            raise ValueError(
                f"{self.full_name}: pool gender '{self.gender}' disagrees "
                f"with the GIVEN_NAME_GENDERS catalog ('{catalog_gender}')"
            )
        return self


class CallerPool(BaseModel):
    """The reviewed caller pool; list order is load-bearing (round-robin)."""

    model_config = ConfigDict(extra="forbid")

    callers: Annotated[list[CallerRecord], Field(min_length=1)]

    @model_validator(mode="after")
    def _unique(self) -> "CallerPool":
        for field in ("full_name", "email"):
            values = [getattr(c, field) for c in self.callers]
            dupes = sorted({v for v in values if values.count(v) > 1})
            if dupes:
                raise ValueError(f"duplicate caller {field}(s): {dupes}")
        return self


def caller_pool_path(domain: str):
    return data_dir() / "tau2" / "domains" / domain / CALLER_POOL_FILENAME


def load_caller_pool(domain: str) -> CallerPool:
    """The domain's reviewed caller pool, or FactoryDraftError if absent."""
    path = caller_pool_path(domain)
    if not path.exists():
        raise FactoryDraftError(
            f"Domain '{domain}' has no {CALLER_POOL_FILENAME} — the seed "
            "diversification needs a reviewed caller pool."
        )
    pool = CallerPool.model_validate(yaml.safe_load(path.read_text()))
    return pool


def auth_mode_for_index(index: int) -> AuthMode:
    """Deterministic 50/50 auth split: odd task positions are name-auth."""
    return AuthMode.NAME_DOB if index % 2 else AuthMode.PHONE


def task_auth_mode(task: dict) -> AuthMode:
    """A diversified task's auth mode, derived from the task itself.

    Name-auth tasks are exactly those whose ``known_info`` does not carry the
    caller's phone number (matching ignores formatting). SINGLE
    implementation — the manifest, tests, and any analysis split all derive
    the mode through here; there is deliberately no stored field.
    """
    info = caller_set_user_info(task)
    if info is None:
        raise ValueError(
            f"Task '{task.get('id')}' has no set_user_info caller — "
            "auth mode is undefined."
        )
    known_info = ((task.get("user_scenario") or {}).get("instructions") or {}).get(
        "known_info"
    ) or ""
    if phone_digits(info["phone_number"]) in phone_digits(known_info):
        return AuthMode.PHONE
    return AuthMode.NAME_DOB


def _canonical_caller(tasks: list[dict], db: dict) -> dict:
    """The pool's single canonical caller record, resolved and verified.

    Every seed task must anchor on the SAME db customer (the generated pool's
    invariant this stage relies on); anything else fails loud.
    """
    customers = {
        phone_digits(c.get("phone_number") or ""): c for c in db.get("customers") or []
    }
    resolved: dict | None = None
    for task in tasks:
        info = caller_set_user_info(task)
        if info is None:
            raise FactoryDraftError(
                f"Task '{task.get('id')}' has no set_user_info caller — "
                "cannot diversify."
            )
        customer = customers.get(phone_digits(info["phone_number"]))
        if customer is None or customer.get("full_name") != info["name"]:
            raise FactoryDraftError(
                f"Task '{task.get('id')}' caller {info['name']!r} / "
                f"{info['phone_number']!r} does not match a db customer."
            )
        if resolved is None:
            resolved = customer
        elif customer is not resolved:
            raise FactoryDraftError(
                "Seed tasks anchor on more than one db customer "
                f"({resolved['customer_id']} vs {customer['customer_id']}) — "
                "the diversification stage assumes a single canonical caller."
            )
    assert resolved is not None
    return resolved


def check_seed_customer_patch_integrity(
    seed_tasks: list[dict], db: dict, pool: CallerPool
) -> list[str]:
    """Integrity problems for a DIVERSIFIED seed set (empty == pass).

    The seed-level counterpart of ``invariants.check_customer_patch_integrity``
    (whose baseline is the source task's own patch): here the baseline is the
    domain DB itself, and the caller record must differ in exactly the fields
    the assigned pool identity owns. Enforced per task, in seed order:

    - the patch carries exactly the DB's customer ids, in DB order;
    - bystanders are byte-identical to their DB records;
    - the caller record equals its DB record with ``full_name``/``email``/
      ``date_of_birth`` replaced by position ``i % len(pool)``'s pool entry;
    - ``set_user_info`` speaks the pool name with the canonical phone;
    - the task's derived auth mode matches the deterministic 50/50 split, and
      known_info carries the mode's identity material (phone, or the natural-
      English date of birth);
    - name-auth tasks carry the line-identification clause in
      ``task_instructions`` and no task carries ``unknown_info``.
    """
    problems: list[str] = []
    db_customers = db.get("customers") or []
    db_ids = [c["customer_id"] for c in db_customers]
    for index, task in enumerate(seed_tasks):
        task_id = task.get("id", "<no id>")
        caller = pool.callers[index % len(pool.callers)]
        expected_mode = auth_mode_for_index(index)

        info = caller_set_user_info(task)
        if info is None:
            problems.append(f"{task_id}: no set_user_info caller")
            continue
        wanted = phone_digits(info["phone_number"])
        caller_db = next(
            (
                c
                for c in db_customers
                if phone_digits(c.get("phone_number") or "") == wanted
            ),
            None,
        )
        if caller_db is None:
            problems.append(
                f"{task_id}: set_user_info phone {info['phone_number']!r} "
                "does not resolve to a db customer"
            )
            continue

        patched = (
            ((task.get("initial_state") or {}).get("initialization_data") or {}).get(
                "agent_data", {}
            )
            or {}
        ).get("customers") or []
        if [(c or {}).get("customer_id") for c in patched] != db_ids:
            problems.append(
                f"{task_id}: customers patch must carry exactly the DB's "
                f"customer ids in DB order"
            )
            continue
        expected_caller = dict(
            caller_db,
            full_name=caller.full_name,
            email=caller.email,
            date_of_birth=caller.date_of_birth,
        )
        for db_record, patch in zip(db_customers, patched):
            if db_record["customer_id"] == caller_db["customer_id"]:
                if patch != expected_caller:
                    problems.append(
                        f"{task_id}: caller record does not equal the DB "
                        f"record with pool identity '{caller.full_name}' "
                        "applied (full_name/email/date_of_birth)"
                    )
            elif patch != db_record:
                problems.append(
                    f"{task_id}: bystander '{db_record['customer_id']}' "
                    "differs from its DB record"
                )

        if info["name"] != caller.full_name:
            problems.append(
                f"{task_id}: set_user_info name {info['name']!r} != assigned "
                f"pool caller {caller.full_name!r}"
            )
        instructions = (task.get("user_scenario") or {}).get("instructions") or {}
        known_info = instructions.get("known_info") or ""
        if instructions.get("unknown_info"):
            problems.append(
                f"{task_id}: carries unknown_info — the diversified seed set "
                "does not use that field"
            )
        expected_clause = NAME_AUTH_LINE_CLAUSE_TEMPLATE.format(
            phone_number=caller_db["phone_number"]
        )
        has_clause = expected_clause in (instructions.get("task_instructions") or "")
        if task_auth_mode(task) is not expected_mode:
            problems.append(
                f"{task_id}: auth mode {task_auth_mode(task).value} != "
                f"expected {expected_mode.value} for position {index}"
            )
        elif expected_mode is AuthMode.NAME_DOB:
            if natural_date(caller.date_of_birth) not in known_info:
                problems.append(
                    f"{task_id}: name-auth known_info lacks the caller's "
                    f"date of birth {natural_date(caller.date_of_birth)!r}"
                )
            if not has_clause:
                problems.append(
                    f"{task_id}: name-auth task_instructions lacks the "
                    "line-identification clause"
                )
        elif has_clause:
            problems.append(
                f"{task_id}: phone-auth task carries the name-auth "
                "line-identification clause"
            )
        if caller.full_name not in known_info:
            problems.append(
                f"{task_id}: known_info does not name the assigned caller "
                f"{caller.full_name!r}"
            )
    return problems


def diversify_seed_tasks(tasks: list[dict], pool: CallerPool, db: dict) -> list[dict]:
    """The seed tasks with per-task callers and the 50/50 auth split applied.

    Pure and deterministic in (tasks, pool, db); the input tasks must be the
    canonical pool shape (single caller, no ``initialization_data``,
    known_info starting with the canonical identity sentence) — any deviation
    raises :class:`FactoryDraftError` rather than guessing.
    """
    canonical = _canonical_caller(tasks, db)
    canonical_prefix = PHONE_AUTH_KNOWN_INFO_TEMPLATE.format(
        full_name=canonical["full_name"],
        phone_number=canonical["phone_number"],
    )

    out_tasks: list[dict] = []
    for index, task in enumerate(tasks):
        caller = pool.callers[index % len(pool.callers)]
        mode = auth_mode_for_index(index)
        out = copy.deepcopy(task)
        task_id = task.get("id", "<no id>")

        instructions = out["user_scenario"]["instructions"]
        if not isinstance(instructions, dict):
            raise FactoryDraftError(
                f"Task '{task_id}' has non-structured user instructions."
            )
        known_info = instructions.get("known_info") or ""
        if not known_info.startswith(canonical_prefix):
            raise FactoryDraftError(
                f"Task '{task_id}' known_info does not start with the "
                f"canonical identity sentence {canonical_prefix!r} — refusing "
                "to rewrite prose I don't recognize."
            )
        if instructions.get("unknown_info"):
            raise FactoryDraftError(
                f"Task '{task_id}' already carries unknown_info — the "
                "diversified seed set does not use that field."
            )
        # Everything after the identity sentence (location sentences etc.)
        # is preserved byte-identically.
        rest = known_info[len(canonical_prefix) :]
        if mode is AuthMode.PHONE:
            identity_sentence = PHONE_AUTH_KNOWN_INFO_TEMPLATE.format(
                full_name=caller.full_name,
                phone_number=canonical["phone_number"],
            )
        else:
            identity_sentence = NAME_AUTH_KNOWN_INFO_TEMPLATE.format(
                full_name=caller.full_name,
                dob_natural=natural_date(caller.date_of_birth),
            )
            existing = instructions.get("task_instructions")
            if not existing:
                raise FactoryDraftError(
                    f"Task '{task_id}' has no task_instructions — the "
                    "name-auth line clause has nowhere to attach."
                )
            instructions["task_instructions"] = (
                existing.rstrip("\n")
                + "\n"
                + NAME_AUTH_LINE_CLAUSE_TEMPLATE.format(
                    phone_number=canonical["phone_number"]
                )
                + "\n"
            )
        instructions["known_info"] = identity_sentence + rest

        if out.get("ticket"):
            out["ticket"] = out["ticket"].replace(
                canonical["full_name"], caller.full_name
            )

        for action in (out.get("initial_state") or {}).get(
            "initialization_actions"
        ) or []:
            if (action or {}).get("func_name") == "set_user_info":
                action["arguments"]["name"] = caller.full_name

        if (out.get("initial_state") or {}).get("initialization_data"):
            raise FactoryDraftError(
                f"Task '{task_id}' already carries initialization_data — "
                "the diversification patch owns that field."
            )
        patched = copy.deepcopy(db["customers"])
        record = next(
            r for r in patched if r["customer_id"] == canonical["customer_id"]
        )
        record["full_name"] = caller.full_name
        record["email"] = caller.email
        record["date_of_birth"] = caller.date_of_birth
        out.setdefault("initial_state", {})["initialization_data"] = {
            "agent_data": {"customers": patched}
        }

        if task.get("evaluation_criteria"):
            out["evaluation_criteria"] = expected_eval_criteria(task, out, db)

        out_tasks.append(out)
    return out_tasks
