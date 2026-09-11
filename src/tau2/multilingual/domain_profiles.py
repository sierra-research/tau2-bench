# Copyright Sierra
"""Closed catalog of the benchmark domains the multilingual pipeline runs.

One ``DomainLocalizationProfile`` per domain is the single owner of the
per-domain knowledge the multilingual layer needs — everything that used to
be implicit "airline" assumptions. The registry is a closed set like the
judge-factor and localization catalogs: an unknown domain raises at every
lookup seam, and a coverage-guard test keeps each profile complete. Opening
a new domain to the pipeline is exactly one reviewed entry here (plus its
term catalog in ``tau2.multilingual.localization_catalog``).

What a profile owns:

- the English SOURCE task file the per-language arm files and identity
  variants derive from (airline ships a curated hand-authored set; telecom's
  is the seed subset ``tau2 factory seed-tasks`` materializes out of its 2285
  generated tasks) — and with it, via :attr:`curated_source`, WHICH verb owns
  the domain's arm files (``tau2 factory arm-tasks`` vs ``seed-tasks``);
- the domain's closed term catalog (via ``localization_catalog``), which the
  pack ``domain_glossaries.<domain>`` entries validate against;
- HOW the caller's identity appears in tasks (``caller_identity``), which
  drives caller extraction for the identity-swap and caller-gender stages:
  airline names callers by a ``first_last_1234`` user-id handle in
  ``known_info``; telecom carries a structured name + phone number in the
  ``set_user_info`` initialization action; retail anchors the caller in
  ``known_info`` prose alone, by name + zip code, handle, or email
  depending on the task;
- whether the identity-swap stage supports the domain, and the entity kinds
  translation prompts illustrate verbatim-preservation with.

NOT owned here: the spelled-entity normalizer's entity grammars
(``tau2.multilingual.spelled_entity_normalizer``) stay airline-shaped module
constants — the normalizer is a pack-level opt-in with no domain seam at TTS
time; opting a telecom pack in requires adding telecom grammars there first.
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

# The registry's own task file. A profile whose multilingual source IS this
# file ships a hand-CURATED task set (nothing to materialize); any other
# filename names a set derived from a generated pool by `tau2 factory
# seed-tasks`.
REGISTRY_TASKS_FILENAME = "tasks.json"


class CallerIdentityKind(str, Enum):
    """How a domain's tasks carry the calling customer's identity."""

    USER_ID_HANDLE = "user_id_handle"
    """The caller is named by a ``first_last_1234`` user-id handle in the
    scenario's ``known_info`` prose (airline)."""

    STRUCTURED_NAME_PHONE = "structured_name_phone"
    """The caller's name and phone number are structured arguments of the
    ``set_user_info`` initialization action (telecom); prose repeats them."""

    PROSE_NAME_ZIP = "prose_name_zip"
    """The caller is anchored ONLY in free ``known_info`` prose (retail), by
    whichever of three lookup keys that task's author wrote: a display name
    plus zip code ("You are Yusuf Rossi in zip code 19122" — 73 of 114 tasks),
    a ``first_last_1234`` user-id handle (23), or an email address (18). The
    retail policy authenticates by email OR name+zip, so all three are
    load-bearing and none is guaranteed present; resolution walks the keys in
    precedence order against the domain DB rather than reading one field."""


class DomainLocalizationProfile(BaseModel):
    """Everything the multilingual pipeline knows about one benchmark domain."""

    model_config = ConfigDict(extra="forbid")

    domain: str = Field(description="The registered tau2 domain name.")
    source_tasks_filename: str = Field(
        description="The English source task file (relative to "
        "data/tau2/domains/<domain>/) that the per-language arm files and "
        "identity variants derive from."
    )
    caller_identity: CallerIdentityKind = Field(
        description="How tasks carry the calling customer's identity; drives "
        "caller extraction in the identity-swap and caller-gender stages."
    )
    identity_swap_supported: bool = Field(
        description="Whether `tau2 factory localize-entities` can derive "
        "locale-identity task variants for this domain."
    )
    tasks_translated: bool = Field(
        description="Whether the domain's localized task sets carry TRANSLATED "
        "prose. False is the standard regime: the per-language sets keep the "
        "English source prose for english-prompt-mode runs (the language "
        "lives in the pack), which disables the script-presence localization "
        "invariants and hands the arm files to the English-prose emitters "
        "(`tau2 factory arm-tasks` / `seed-tasks`). True hands them to the "
        "quarantined the retired task-translation loop instead."
    )
    vocabulary_context: str = Field(
        description="One English sentence grounding the localization-drafting "
        "prompt in the domain's call subject matter."
    )
    translation_example_entities: str = Field(
        description="The domain's verbatim-preservation examples for the "
        "translator/verifier prompts (entity shapes with sample values)."
    )
    default_smoke_task_stem: str = Field(
        description="The task-id stem (without the language suffix) the "
        "domain's smoke stage runs when the author/CLI does not pick one. "
        "Every language shares the domain's task ids, so one stem per domain "
        "is universal."
    )

    @property
    def curated_source(self) -> bool:
        """Whether the multilingual source set is hand-curated and checked in.

        True when the source IS the registry's own ``tasks.json`` (airline):
        there is nothing to materialize, so ``tau2 factory arm-tasks`` emits
        the per-language arm files straight off it. False when the source is
        a separate file derived from a generated pool (telecom's
        ``tasks_multilingual.json``), which ``tau2 factory seed-tasks``
        materializes and emits arms for in one step. Exactly one of the two
        verbs owns any domain's arm files.
        """
        return self.source_tasks_filename == REGISTRY_TASKS_FILENAME


DOMAIN_PROFILES: dict[str, DomainLocalizationProfile] = {
    p.domain: p
    for p in [
        DomainLocalizationProfile(
            domain="airline",
            source_tasks_filename=REGISTRY_TASKS_FILENAME,
            caller_identity=CallerIdentityKind.USER_ID_HANDLE,
            identity_swap_supported=True,
            tasks_translated=False,
            vocabulary_context=(
                "Calls are to an airline's customer service line: flight "
                "reservations, cancellations, baggage, cabin classes, "
                "refunds, and loyalty membership."
            ),
            translation_example_entities=(
                "user ids like 'mia_li_3668', reservation codes like "
                "'ZFA04Y', flight numbers like 'HAT123', dates, and dollar "
                "amounts"
            ),
            default_smoke_task_stem="3",
        ),
        DomainLocalizationProfile(
            domain="telecom",
            source_tasks_filename="tasks_multilingual.json",
            caller_identity=CallerIdentityKind.STRUCTURED_NAME_PHONE,
            identity_swap_supported=True,
            tasks_translated=False,
            vocabulary_context=(
                "Calls are to a mobile carrier's support line: overdue "
                "bills, line suspensions, data refueling, roaming, and "
                "device troubleshooting (mobile data, SIM, APN, Wi-Fi, "
                "MMS) the caller performs on their own phone."
            ),
            translation_example_entities=(
                "phone numbers like '555-123-2002', customer ids like "
                "'C1001', line ids like 'L1002', and data amounts like "
                "'2.0 GB'"
            ),
            default_smoke_task_stem=(
                "[mobile_data_issue]airplane_mode_on|data_mode_off[PERSONA:None]"
            ),
        ),
        DomainLocalizationProfile(
            domain="retail",
            # Curated like airline: the registry's own 114-task tasks.json IS
            # the multilingual source, so `arm-tasks` owns the arm files and
            # there is no generated pool to seed from. The run frame is a
            # fixed 50-task sample pinned in the pool spec, not a narrower
            # source file — every task is localized so any of the 114 can be
            # run later without re-emitting arms.
            source_tasks_filename=REGISTRY_TASKS_FILENAME,
            caller_identity=CallerIdentityKind.PROSE_NAME_ZIP,
            identity_swap_supported=True,
            tasks_translated=False,
            vocabulary_context=(
                "Calls are to an online retail store's customer service line: "
                "cancelling or modifying pending orders, returning or "
                "exchanging delivered items, product options and variants, "
                "refunds to the original payment method or a gift card, and "
                "the shipping address on file."
            ),
            translation_example_entities=(
                "order ids like '#W2378156', product ids like '1656367028', "
                "item ids like '4983901480', payment method ids like "
                "'credit_card_9513926', zip codes like '19122', and dollar "
                "amounts"
            ),
            # Inside the canonical retail_50 frame (a stem outside it fails
            # preset generation), single-action, and authenticated by NAME +
            # ZIP — retail's dominant caller shape and the one whose
            # localization carries the most risk, so the smoke exercises the
            # identity swap's hardest path rather than the easy email lookup.
            default_smoke_task_stem="83",
        ),
    ]
}


def get_domain_profile(domain: str) -> DomainLocalizationProfile:
    """Look up a domain's profile, or raise if not in the closed set."""
    try:
        return DOMAIN_PROFILES[domain]
    except KeyError:
        raise KeyError(
            f"Unknown multilingual domain '{domain}'. It must be one of the "
            f"profiled domains in tau2.multilingual.domain_profiles: "
            f"{sorted(DOMAIN_PROFILES)}"
        )
