# Copyright Sierra
"""Per-language ARM task files — the one emitter both source regimes share.

A multilingual run of domain ``<d>`` in language ``<l>`` scores on task set
``<d>_<l>``, backed by ``data/tau2/multilingual/<l>/<d>_tasks_<l>.json``. That
file is the domain's English SOURCE task set with every id suffixed
``_<lang>`` and the prose left byte-identical: english-prompt mode is how
every multilingual run works (``tau2.multilingual.english_prompts``), so the
prose the user simulator reads is English and the LANGUAGE lives entirely in
the language pack. Task translation is a retired stage, quarantined under
the retired task-translation loop.

Two verbs produce arm files, differing ONLY in where the source set comes
from — so the emit + validate + sidecar step lives here, once:

- ``tau2 factory seed-tasks`` (``factory.seed_tasks``) — domains with a
  GENERATED task pool (telecom): materializes the reviewed split as the
  profile's ``source_tasks_filename``, then emits its arms;
- ``tau2 factory arm-tasks`` (this module) — domains whose source set is
  hand-CURATED and checked in (airline): nothing to materialize, so it emits
  arms straight off ``load_domain_tasks``.

The two are exact complements — ``DomainLocalizationProfile.curated_source``
decides which owns a domain's arm files, and each verb refuses the other's
domains, so no file has two owners.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Annotated, Optional

from pydantic import BaseModel, ConfigDict, Field

from tau2.multilingual.domain_profiles import (
    DomainLocalizationProfile,
    get_domain_profile,
)
from tau2.multilingual.factory.author_form import FactoryDraftError
from tau2.multilingual.factory.entity_localization import (
    build_english_gender_sidecar,
    gender_sidecar_path,
)
from tau2.multilingual.invariants import check_task_set_localization
from tau2.multilingual.localize_lib import (
    data_dir,
    load_domain_db,
    load_domain_tasks,
    multilingual_data_dir,
)


class ArmTasksOutcome(BaseModel):
    """What ``tau2 factory arm-tasks`` produced (or found already in place)."""

    model_config = ConfigDict(extra="forbid")

    domain: Annotated[str, Field(description="The domain whose arms were emitted")]
    task_count: Annotated[int, Field(description="Tasks in the source set")]
    source_path: Annotated[
        Path, Field(description="The curated English source task file the arms copy")
    ]
    emitted: Annotated[
        dict[str, Path],
        Field(description="Per-language arm files ({lang: path})"),
    ]
    written: Annotated[
        list[Path],
        Field(
            description="Files actually (re)written this run — empty when "
            "everything was already up to date"
        ),
    ]


def arm_task_set_path(lang: str, domain: str) -> Path:
    """``data/tau2/multilingual/<lang>/<domain>_tasks_<lang>.json``."""
    return multilingual_data_dir() / lang / f"{domain}_tasks_{lang}.json"


def build_arm_tasks(source_tasks: list[dict], lang: str) -> list[dict]:
    """The language arm of a source set: same prose, ids suffixed ``_<lang>``."""
    arm_tasks = []
    for task in source_tasks:
        out = copy.deepcopy(task)
        out["id"] = f"{task['id']}_{lang}"
        arm_tasks.append(out)
    return arm_tasks


def write_if_changed(path: Path, text: str, written: list[Path]) -> None:
    """Write ``text`` to ``path`` only when it differs; record the write.

    Idempotence is a factory contract (same inputs + same code version -> same
    outputs, re-running writes nothing), and both arm producers need it.
    """
    if path.exists() and path.read_text() == text:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    written.append(path)


def emit_language_arms(
    domain: str,
    source_tasks: list[dict],
    langs: list[str],
    db: dict,
    profile: DomainLocalizationProfile,
    written: list[Path],
) -> dict[str, Path]:
    """Emit ``<domain>_tasks_<lang>.json`` for each language (+ the en sidecar).

    Each arm is :func:`build_arm_tasks` of the source set, gated on the shared
    localization invariants before it is written. ``script_code`` is None
    throughout: the prose IS the English source, so the prose-is-localized
    checks do not apply — the entity, evaluation-criteria and customer-patch
    invariants all still run (``domain_db`` is passed).

    The ``en`` arm additionally gets the caller-gender voice-routing sidecar
    (``<domain>_caller_gender_en.json``): without one an English arm falls back
    to the domain's pre-sampled ``tasks_voice.json`` personas, whose gender is
    uncorrelated with the caller's name (measured 25/50 contradictory on
    airline). The localized arms' sidecars are owned by ``localize-entities``;
    all of them read the caller through :func:`english_caller_gender`.

    Raises:
        FactoryDraftError: An emitted set fails the invariants.
    """
    emitted: dict[str, Path] = {}
    for lang in langs:
        arm_tasks = build_arm_tasks(source_tasks, lang)
        problems = check_task_set_localization(
            arm_tasks,
            source_tasks,
            suffix=lang,
            script_code=None,
            domain_db=db,
            domain=domain,
        )
        if problems:
            raise FactoryDraftError(
                f"Emitted '{domain}_tasks_{lang}' set fails invariants: "
                + "; ".join(problems[:5])
            )
        arm_path = arm_task_set_path(lang, domain)
        write_if_changed(
            arm_path,
            json.dumps(arm_tasks, ensure_ascii=False, indent=2) + "\n",
            written,
        )
        emitted[lang] = arm_path
        if lang == "en":
            # One derivation owns the sidecar's shape (including the
            # Recompute it from the arm
            # file just written — the same reading the shipped-artifact test
            # asserts against, so the two can never drift.
            gender = build_english_gender_sidecar(domain)
            write_if_changed(
                gender_sidecar_path("en", domain),
                json.dumps(gender, ensure_ascii=False, indent=2) + "\n",
                written,
            )
    return emitted


def check_arm_emission_allowed(
    profile: DomainLocalizationProfile, langs: list[str]
) -> None:
    """Raise unless every requested arm is legitimately English-prose.

    A domain whose prose is TRANSLATED has its localized arm files owned by
    the retired task-translation loop; emitting English source prose over them
    would silently revert the translation. The ``en`` arm is exempt — it IS
    the source prose under either regime.
    """
    blocked = [lang for lang in langs if lang != "en"]
    if profile.tasks_translated and blocked:
        raise FactoryDraftError(
            f"Domain '{profile.domain}' declares TRANSLATED task prose "
            f"(tasks_translated=True): the {', '.join(blocked)} arm file(s) "
            "are owned by the retired task-translation loop — emitting English "
            "source prose over them would silently revert the translation."
        )


def run_arm_tasks(
    domain: str,
    langs: Optional[list[str]] = None,
) -> ArmTasksOutcome:
    """``tau2 factory arm-tasks`` — arms for a CURATED-source domain.

    The complement of ``tau2 factory seed-tasks``: nothing is materialized
    (the source set is the reviewed, checked-in ``tasks.json`` the registry
    itself loads), so this verb only emits the per-language arm files and the
    English caller-gender sidecar. Deterministic and idempotent.

    Raises:
        FactoryDraftError: Unknown domain, a domain whose source set is
            materialized by ``seed-tasks``, a domain whose prose is
            translated, or an emitted set failing the invariants.
    """
    try:
        profile = get_domain_profile(domain)
    except KeyError as exc:
        raise FactoryDraftError(str(exc)) from exc
    if not profile.curated_source:
        raise FactoryDraftError(
            f"Domain '{domain}' derives its multilingual source set from a "
            f"generated pool ({profile.source_tasks_filename}) — its arm files "
            "are emitted by `tau2 factory seed-tasks --domain "
            f"{domain}`, which materializes that set first."
        )
    langs = list(langs or ["en"])
    check_arm_emission_allowed(profile, langs)

    source_tasks = load_domain_tasks(domain)
    db = load_domain_db(domain)
    written: list[Path] = []
    emitted = emit_language_arms(domain, source_tasks, langs, db, profile, written)
    return ArmTasksOutcome(
        domain=domain,
        task_count=len(source_tasks),
        source_path=(
            data_dir() / "tau2" / "domains" / domain / profile.source_tasks_filename
        ),
        emitted=emitted,
        written=written,
    )
