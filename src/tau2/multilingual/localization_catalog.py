# Copyright Sierra
"""Closed catalogs for per-pack SPOKEN-LOCALIZATION data.

Four fixed menus back ``LocalizationPackConfig`` (``tau2.multilingual.schema``):

- ``SYMBOL_CATALOG`` — the technical symbols a caller voices while reading
  emails, IDs, codes, and phone numbers aloud. A pack supplies the native
  verbalization(s) per symbol; it may NOT invent a new symbol key.
- ``DOMAIN_TERM_CATALOGS`` — the benchmark's service-domain vocabulary, one
  closed catalog per domain (airline: user ID, refund, reservation, …;
  telecom: plan, roaming, SIM card, …; retail: order, exchange, gift card,
  …). A pack supplies the natural native
  term per catalog id under the domain's key; it may NOT invent a new term
  id or a new domain key.
- ``GUIDELINE_EXAMPLE_CATALOG`` — the behavioral example utterances the ENGLISH
  voice guidelines demonstrate inline (disfluency palette, conversational
  confirmations, silence check-ins, don't-know responses, …). A pack supplies
  the native utterances per catalog kind; it may NOT invent a new kind. Each
  kind also carries the FIXED English default items (today's guidelines text),
  so English personas render byte-identical prompts.
- ``HONORIFIC_CONCEPT_CATALOG`` — the everyday CONCEPTS whose word has an
  honorific form that is other-directed ONLY (name, age, birthday, home).
  A pack supplies the honorific/plain word pair per concept; it may
  NOT invent a new concept id. The concepts are shared across languages even
  though the words are not — the same situation the domain-term glossary is in.
  Only the languages with observed evidence of the trap carry pairs.

All are closed sets validated at pack load (unknown ids raise), mirroring the
nativeness/delivery judge-factor catalogs: authoring a language can never
silently introduce an un-reviewed symbol, term, example axis, or honorific
pair, and a coverage-guard test can require every pack to cover the full menus.

Observed localization failures include de 'underscore' voiced in English instead
of 'Unterstrich'; pl hallucinated 'archeśnika' for a symbol readout; fr 'tiret
du bas' was mistranscribed as a hyphen; ko produced mixed-language dates ('May
십칠') and numbers ('이십two'); vi ran 'May21' together; pl callers said 'user
ID'/'refund'/'future trip' in English mid-sentence. The example
catalog is grounded in the es arm-diff: models copy the guidelines' English
disfluency/confirmation palettes verbatim into target-language calls, and the
nativeness judge penalizes exactly those English fillers.

The honorific-concept catalog prevents a caller from applying listener-only
honorifics to itself. In Korean, for example, 성함 is the honorific word for the
other person's name, while 이름 is the plain self-reference form. These pairs
must be supplied explicitly because drafting-time translation guidance never
reaches the simulator at runtime.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class SymbolDef(BaseModel):
    """Universal definition of one voiced technical symbol (no language text)."""

    symbol: str = Field(
        description="The literal symbol character; the pack references this key."
    )
    english_name: str = Field(
        description="The English name of the symbol (what a non-native readout "
        "leaks); shown to the drafting model and in rendered prompts."
    )
    description: str = Field(
        description="Where the symbol shows up in calls (documentary, not scored)."
    )


def _s(symbol: str, english_name: str, description: str) -> SymbolDef:
    return SymbolDef(symbol=symbol, english_name=english_name, description=description)


# The closed set of symbols a pack provides native readouts for, keyed by the
# literal character. Every ``SymbolReadout.symbol`` MUST reference one of these.
SYMBOL_CATALOG: dict[str, SymbolDef] = {
    s.symbol: s
    for s in [
        _s("@", "at sign", "email addresses"),
        _s(".", "dot", "email addresses and URLs (e.g. '.com')"),
        _s("_", "underscore", "email local parts and user IDs (e.g. 'mia_li')"),
        _s("-", "hyphen", "emails, booking codes, and phone-number grouping"),
        _s("/", "slash", "dates and codes read aloud (e.g. '05/17')"),
        _s("+", "plus sign", "international phone prefixes (e.g. '+49')"),
        _s(
            "0",
            "zero",
            "the digit zero in IDs and phone numbers (English 'zero'/'oh' "
            "leakage is the common non-native readout)",
        ),
    ]
}


class DomainTermDef(BaseModel):
    """Universal definition of one domain-vocabulary term (no language text)."""

    term_id: str = Field(
        description="Stable catalog id; the pack's glossary references this."
    )
    english: str = Field(
        description="The English surface term as it appears in tasks/policies "
        "(what an unlocalized caller says mid-sentence)."
    )
    description: str = Field(
        description="What the term means in its domain (documentary)."
    )


def _t(term_id: str, english: str, description: str) -> DomainTermDef:
    return DomainTermDef(term_id=term_id, english=english, description=description)


# The closed set of AIRLINE domain terms a pack provides native glosses for,
# keyed by term id. Every ``DomainTermGloss.term_id`` under the pack's
# ``domain_glossaries.airline`` key MUST reference one of these.
_AIRLINE_TERM_CATALOG: dict[str, DomainTermDef] = {
    t.term_id: t
    for t in [
        _t("user_id", "user ID", "the account identifier the agent asks for"),
        _t("reservation", "reservation", "a booking of one or more flights"),
        _t(
            "reservation_id",
            "reservation ID",
            "the booking reference code (e.g. 'ZFA04Y')",
        ),
        _t("flight", "flight", "a single flight segment"),
        _t("flight_number", "flight number", "the flight designator (e.g. 'HAT123')"),
        _t("refund", "refund", "money returned after a cancellation/change"),
        _t(
            "travel_insurance",
            "travel insurance",
            "the optional per-passenger insurance add-on",
        ),
        _t("checked_bag", "checked bag", "luggage checked into the hold"),
        _t("carry_on", "carry-on bag", "cabin luggage"),
        _t("future_trip", "future trip", "an upcoming (not yet flown) trip"),
        _t("one_way", "one-way trip", "a trip without a return segment"),
        _t("round_trip", "round trip", "a trip with outbound and return segments"),
        _t(
            "basic_economy",
            "basic economy",
            "the most restricted cabin/fare class in the domain",
        ),
        _t("economy", "economy", "the standard cabin class"),
        _t("business", "business class", "the premium cabin class"),
        _t("credit_card", "credit card", "a stored card payment method"),
        _t("gift_card", "gift card", "a stored gift-card payment method"),
        _t(
            "travel_certificate",
            "travel certificate",
            "a compensation credit usable for future travel",
        ),
        _t(
            "membership_tier",
            "membership tier",
            "the loyalty status level (regular / silver / gold)",
        ),
        _t("cancellation", "cancellation", "cancelling a reservation"),
        _t("payment_method", "payment method", "how a booking is paid for"),
    ]
}


# The closed set of TELECOM domain terms, keyed by term id. Vocabulary is
# drawn from the telecom policy (data/tau2/domains/telecom/main_policy.md:
# customer/line/plan/bill basics, overdue-bill payment, line suspension, data
# refueling, roaming) and the tech-support manual (mobile-data, MMS, and
# service troubleshooting: airplane mode, SIM, APN, data saver, Wi-Fi, VPN,
# speed test, signal) — the terms a caller actually says on a support call.
_TELECOM_TERM_CATALOG: dict[str, DomainTermDef] = {
    t.term_id: t
    for t in [
        _t("customer_id", "customer ID", "the account identifier (e.g. 'C1001')"),
        _t(
            "phone_number",
            "phone number",
            "the caller's number, the primary account lookup key",
        ),
        _t("line", "line", "one phone line on the account (e.g. 'L1002')"),
        _t("plan", "plan", "the subscription plan a line is on"),
        _t(
            "data_allowance",
            "data allowance",
            "the plan's included mobile-data volume (in GB)",
        ),
        _t("data_usage", "data usage", "how much of the data allowance is used"),
        _t(
            "data_refuel",
            "data refuel",
            "a paid one-off top-up of extra data before the cycle ends",
        ),
        _t("mobile_data", "mobile data", "cellular internet on the phone"),
        _t(
            "roaming",
            "data roaming",
            "using mobile data on a foreign network while abroad",
        ),
        _t(
            "airplane_mode",
            "airplane mode",
            "the device toggle that cuts all radios",
        ),
        _t("sim_card", "SIM card", "the physical subscriber identity card"),
        _t("esim", "eSIM", "the embedded (non-physical) SIM"),
        _t(
            "apn_settings",
            "APN settings",
            "the access-point-name configuration mobile data needs",
        ),
        _t(
            "data_saver",
            "data saver mode",
            "the device mode that restricts background data",
        ),
        _t("wifi", "Wi-Fi", "wireless LAN internet on the device"),
        _t(
            "wifi_calling",
            "Wi-Fi calling",
            "placing calls over Wi-Fi instead of the cellular network",
        ),
        _t("vpn", "VPN", "a virtual private network connection on the device"),
        _t(
            "speed_test",
            "speed test",
            "the on-device check of current internet speed",
        ),
        _t(
            "network_signal",
            "network signal",
            "cellular signal strength (the bars on the phone)",
        ),
        _t(
            "status_bar",
            "status bar",
            "the phone's top bar showing signal/data icons",
        ),
        _t("mms", "MMS", "picture/multimedia messaging"),
        _t("bill", "bill", "a monthly statement on the account"),
        _t(
            "overdue_bill",
            "overdue bill",
            "an unpaid bill past its due date (can suspend the line)",
        ),
        _t(
            "payment_request",
            "payment request",
            "the payment link the agent sends the caller to settle a bill",
        ),
        _t(
            "line_suspension",
            "suspended line",
            "a line disabled for non-payment or by request",
        ),
        _t("payment_method", "payment method", "how the account pays its bills"),
    ]
}


# The closed set of RETAIL domain terms, keyed by term id. Vocabulary is
# drawn from the retail policy (data/tau2/domains/retail/policy.md: the
# user/product/order model, the two authentication keys, the cancel / modify
# / return / exchange action rules, and the three payment-method types) — the
# terms a caller actually says on an online-store support call. Note the
# policy's own load-bearing distinction between a PRODUCT and one of its
# variant ITEMS: a language that glosses both with the same word makes the
# agent's "which item" question unanswerable.
_RETAIL_TERM_CATALOG: dict[str, DomainTermDef] = {
    t.term_id: t
    for t in [
        _t("user_id", "user ID", "the account identifier the agent looks up"),
        _t(
            "zip_code",
            "zip code",
            "the postal code that, with the name, authenticates the caller",
        ),
        _t("order", "order", "one purchase on the account"),
        _t("order_id", "order ID", "the order reference (e.g. '#W2378156')"),
        _t(
            "order_status",
            "order status",
            "where an order stands: pending, processed, delivered, cancelled",
        ),
        _t(
            "pending_order",
            "pending order",
            "an order not yet shipped (still cancellable/modifiable)",
        ),
        _t(
            "delivered_order",
            "delivered order",
            "an order already received (returnable/exchangeable)",
        ),
        _t("cancellation", "cancellation", "cancelling a pending order"),
        _t("return", "return", "sending a delivered item back for a refund"),
        _t(
            "exchange",
            "exchange",
            "swapping a delivered item for another variant of the same product",
        ),
        _t("refund", "refund", "money returned after a cancellation or return"),
        _t(
            "product",
            "product",
            "a TYPE of merchandise (e.g. 't-shirt'), not one buyable unit",
        ),
        _t(
            "item",
            "item",
            "one buyable VARIANT of a product (a specific colour/size) — what "
            "an order actually contains",
        ),
        _t(
            "product_option",
            "option",
            "an attribute distinguishing variants (colour, size, material)",
        ),
        _t("item_id", "item ID", "the variant identifier (e.g. '4983901480')"),
        _t("product_id", "product ID", "the product-type identifier"),
        _t("availability", "availability", "whether an item is in stock"),
        _t(
            "price_difference",
            "price difference",
            "the amount charged or refunded when an exchange swaps prices",
        ),
        _t("tracking_number", "tracking number", "the shipment tracking id"),
        _t(
            "shipping_address",
            "shipping address",
            "the delivery address on an order",
        ),
        _t(
            "default_address",
            "default address",
            "the address on the user's profile, used for new orders",
        ),
        _t("payment_method", "payment method", "how an order is paid or refunded"),
        _t("credit_card", "credit card", "a stored card payment method"),
        _t("gift_card", "gift card", "a stored gift-card payment method/balance"),
        _t("paypal", "PayPal account", "a stored PayPal payment method"),
        _t(
            "human_agent",
            "human agent",
            "the human the call is transferred to when out of scope",
        ),
    ]
}


# One closed term catalog per benchmark domain a pack may carry a glossary
# for. Adding a domain here (with its term catalog) is the reviewed,
# one-time engineering step that opens `domain_glossaries.<domain>` to packs.
DOMAIN_TERM_CATALOGS: dict[str, dict[str, DomainTermDef]] = {
    "airline": _AIRLINE_TERM_CATALOG,
    "telecom": _TELECOM_TERM_CATALOG,
    "retail": _RETAIL_TERM_CATALOG,
}


class HonorificConceptDef(BaseModel):
    """Universal definition of one self-honorification-prone concept.

    No language text: the concept is what the word MEANS ('name', 'age'), and
    each pack supplies its own honorific/plain pair for it.
    """

    concept_id: str = Field(
        description="Stable catalog id; the pack's honorific_self_reference "
        "entries reference this."
    )
    english: str = Field(
        description="The English gloss of the concept, shown in the rendered "
        "prompt line as the left-hand side of the pair."
    )
    description: str = Field(
        description="Where the concept comes up on a call and why its "
        "honorific form is other-directed (documentary, not scored)."
    )


def _h(concept_id: str, english: str, description: str) -> HonorificConceptDef:
    return HonorificConceptDef(
        concept_id=concept_id, english=english, description=description
    )


# The closed set of concepts a pack may supply an honorific/plain word pair
# for. Every ``HonorificSelfReference.concept_id`` MUST reference one of these.
# Deliberately small and evidence-led: these are the concepts a caller talks
# about ITSELF on an identity-verification call, which is exactly where the
# observed self-honorification happens.
HONORIFIC_CONCEPT_CATALOG: dict[str, HonorificConceptDef] = {
    h.concept_id: h
    for h in [
        _h(
            "name",
            "name",
            "the caller's own name, given during identity verification — the "
            "measured ko failure (성함 said of oneself in 71/300 calls)",
        ),
        _h(
            "age",
            "age",
            "the caller's own age, volunteered in eligibility/plan talk",
        ),
        _h(
            "birthday",
            "date of birth",
            "the caller's own date of birth, the second identity-verification "
            "factor in the telecom domain",
        ),
        _h(
            "home",
            "home",
            "the caller's own home/house, mentioned when describing where a "
            "service problem happens",
        ),
    ]
}


class ExampleJoinStyle(str, Enum):
    """How a kind's utterances render into the guidelines' English prose.

    The instructional sentence around each example stays fixed English (the
    ENGLISH-INSTRUCTION / TARGET-LANGUAGE-EXAMPLES design rule); only the
    quoted utterances vary by language.
    """

    QUOTED_LIST = "quoted_list"  # "a", "b", "c"
    QUOTED_OR = "quoted_or"  # "a" or "b"
    QUOTED_SINGLE = "quoted_single"  # "a"
    CHAIN = "chain"  # "a" → (connector) → "b" → (connector) → "c"
    APPENDED_EXAMPLE = "appended_example"  # '' or ' For example: "a"'


class GuidelineExampleKindDef(BaseModel):
    """Universal definition of one guidelines example kind.

    ``english_items`` is the FIXED English default (exactly today's guidelines
    text) — the rendering used for English personas and whenever a pack lacks
    the kind. It is fixed prompt scaffold, not drafted content.
    """

    kind_id: str = Field(
        description="Stable catalog id; the pack's guideline_examples reference it."
    )
    description: str = Field(
        description="What a native version must demonstrate (shown to the "
        "drafting model; documentary for reviewers)."
    )
    join_style: ExampleJoinStyle = Field(
        description="How the utterances render into the fixed English prose."
    )
    min_items: int = Field(description="Minimum utterances a pack must supply.")
    max_items: int = Field(description="Maximum utterances a pack may supply.")
    english_items: list[str] = Field(
        description="The fixed English default items (today's guidelines text). "
        "May be empty for kinds the English guidelines state without an example."
    )
    chain_connectors: list[str] = Field(
        default_factory=list,
        description="CHAIN kinds only: the fixed English connector "
        "parentheticals between consecutive utterances.",
    )
    native_note: Optional[str] = Field(
        default=None,
        description="Fixed English note appended ONLY to a native rendering "
        "(e.g. the persona-precedence pointer for the confirmations palette).",
    )


def _e(
    kind_id: str,
    description: str,
    join_style: ExampleJoinStyle,
    min_items: int,
    max_items: int,
    english_items: list[str],
    chain_connectors: Optional[list[str]] = None,
    native_note: Optional[str] = None,
) -> GuidelineExampleKindDef:
    return GuidelineExampleKindDef(
        kind_id=kind_id,
        description=description,
        join_style=join_style,
        min_items=min_items,
        max_items=max_items,
        english_items=english_items,
        chain_connectors=chain_connectors or [],
        native_note=native_note,
    )


# The closed set of guidelines example kinds a pack provides native utterances
# for. Every ``GuidelineExample.kind`` MUST reference one of these. The
# english_items reproduce the English voice guidelines' inline examples
# EXACTLY — English personas must render today's text byte-identically.
GUIDELINE_EXAMPLE_CATALOG: dict[str, GuidelineExampleKindDef] = {
    e.kind_id: e
    for e in [
        _e(
            "disfluency_fillers",
            "The language's natural hesitation fillers and discourse-particle "
            "palette (the words a native actually stalls with on the phone). "
            "THE highest-risk palette: models copy it verbatim, so English "
            "fillers here leak straight into target-language calls.",
            ExampleJoinStyle.QUOTED_LIST,
            4,
            8,
            ["um", "uh", "you know", "like", "I mean"],
        ),
        _e(
            "restart_example",
            "One utterance showing a mid-sentence restart/self-repair "
            "([pause] marker welcome).",
            ExampleJoinStyle.QUOTED_SINGLE,
            1,
            1,
            ["Can you [pause] sorry, I meant to ask, can you help me with..."],
        ),
        _e(
            "filler_pause_example",
            "One utterance stringing natural fillers and hedges through a "
            "simple request.",
            ExampleJoinStyle.QUOTED_SINGLE,
            1,
            1,
            ["So, um, I was wondering if you could, you know, help me out"],
        ),
        _e(
            "pause_marker_examples",
            "Two utterances demonstrating em-dash (—) and [pause] pause "
            "markers mid-thought.",
            ExampleJoinStyle.QUOTED_OR,
            2,
            2,
            [
                "I was trying to—wait, let me think [pause]",
                "The issue started [pause] maybe three days ago?",
            ],
        ),
        _e(
            "self_interruption_example",
            "One utterance where the caller interrupts their own thought to "
            "switch tack.",
            ExampleJoinStyle.QUOTED_SINGLE,
            1,
            1,
            [
                "I've been trying to... oh wait, should I give you my account number first?"
            ],
        ),
        _e(
            "clarification_request",
            "One natural way to ask the agent to repeat something not caught.",
            ExampleJoinStyle.QUOTED_SINGLE,
            1,
            1,
            ["Sorry, could you repeat that? I didn't quite catch it"],
        ),
        _e(
            "emotion_examples",
            "Two utterance OPENERS showing emotion naturally: first a negative "
            "(frustrated) one, then a positive (pleased) one. These demonstrate "
            "HOW emotion sounds in the language when a task assigns it — they "
            "must not prescribe a mood.",
            ExampleJoinStyle.QUOTED_OR,
            2,
            2,
            ["I'm really frustrated because...", "Oh great, that would be wonderful!"],
        ),
        _e(
            "conversational_confirmations",
            "The language-level palette of brief conversational confirmations "
            "/ acknowledgments a native listener drops ('got it' / 'okay' "
            "equivalents). Must AGREE with the pack personas' backchannel "
            "phrases — one list family, not a divergent second inventory.",
            ExampleJoinStyle.QUOTED_LIST,
            4,
            8,
            ["Uh huh", "Yeah", "Okay", "Got it"],
            native_note="(if your persona guidelines give their own phrases, "
            "those take precedence)",
        ),
        _e(
            "silence_checkins",
            "Three check-in utterances for an unresponsive agent: still "
            "there?, found anything?, any updates on my query?",
            ExampleJoinStyle.QUOTED_LIST,
            3,
            3,
            [
                "Hello? Are you still there?",
                "Did you find anything?",
                "Any updates on my query about ...?",
            ],
        ),
        _e(
            "frustrated_endings",
            "Two natural ways a caller gives up and ends a dead call in frustration.",
            ExampleJoinStyle.QUOTED_OR,
            2,
            2,
            [
                "This is ridiculous, I'll try calling back later",
                "I don't have time for this, goodbye",
            ],
        ),
        _e(
            "dont_know_responses",
            "Three natural ways to say you don't know / don't remember / "
            "would have to look something up.",
            ExampleJoinStyle.QUOTED_LIST,
            3,
            3,
            [
                "Um, I'm not sure actually",
                "I don't remember off the top of my head",
                "Hmm, I'd have to look that up",
            ],
        ),
        _e(
            "make_agent_work_chain",
            "A three-step progressive-disclosure chain: a vague complaint, "
            "then slightly more specific, then the concrete thing — each step "
            "only as specific as the agent's question forces.",
            ExampleJoinStyle.CHAIN,
            3,
            3,
            ["It's not working", "The app", "Your mobile app"],
            chain_connectors=[
                "(agent asks what's not working)",
                "(agent asks which app)",
            ],
        ),
        _e(
            "one_at_a_time_example",
            "One utterance giving a single requested item and letting the "
            "agent ask for the next (e.g. giving the email, then asking "
            "whether the phone number is needed too). The English guidelines "
            "state this rule without an example; the localized files carry "
            "one.",
            ExampleJoinStyle.APPENDED_EXAMPLE,
            1,
            1,
            [],
        ),
        _e(
            "forget_details_example",
            "One utterance of momentarily forgetting a detail and stalling "
            "while checking.",
            ExampleJoinStyle.QUOTED_SINGLE,
            1,
            1,
            ["My order number is... um, let me check... hold on..."],
        ),
        _e(
            "vague_openers",
            "Two deliberately vague opening statements of a problem.",
            ExampleJoinStyle.QUOTED_OR,
            2,
            2,
            ["I have a problem", "Something's wrong with my account"],
        ),
        _e(
            "unknown_info_phrases",
            "Two neutral ways to state that a piece of information is unknown "
            "/ not at hand.",
            ExampleJoinStyle.QUOTED_OR,
            2,
            2,
            ["I'm not sure about that", "I don't have that information"],
        ),
        # --- kinds used only by the TOOLS variant of the voice guidelines ---
        _e(
            "tool_acknowledgment",
            "One utterance verbally acknowledging the agent's request to "
            "perform an action before doing it (an 'okay, let me try that, "
            "hold on' move). Tools-variant guidelines only.",
            ExampleJoinStyle.QUOTED_SINGLE,
            1,
            1,
            ["Oh, okay, let me try that... hold on a sec"],
        ),
        _e(
            "tool_result_report",
            "One utterance OPENER naturally reporting what an action/tool "
            "just showed ('alright, I just did that and it says...'). "
            "Tools-variant guidelines only.",
            ExampleJoinStyle.QUOTED_SINGLE,
            1,
            1,
            ["Alright, I just did that and, um, it says..."],
        ),
        _e(
            "tool_failure_report",
            "One utterance conversationally reporting that an attempted "
            "action failed / errored. Tools-variant guidelines only.",
            ExampleJoinStyle.QUOTED_SINGLE,
            1,
            1,
            ["Hmm, that didn't work... I'm getting an error"],
        ),
        _e(
            "multiple_actions_pushback",
            "One utterance pushing back on being asked to do several actions "
            "at once and requesting a step-by-step walkthrough. Tools-variant "
            "guidelines only.",
            ExampleJoinStyle.QUOTED_SINGLE,
            1,
            1,
            ["Whoa, that's a lot at once... could you walk me through one at a time?"],
        ),
        _e(
            "gratitude_examples",
            "Two natural ways to thank the agent for helping. These "
            "demonstrate HOW thanks sounds in the language when warranted — "
            "they must not prescribe warmth. Tools-variant guidelines only.",
            ExampleJoinStyle.QUOTED_OR,
            2,
            2,
            ["Oh thank you so much!", "Great, I really appreciate your help"],
        ),
    ]
}


def _lookup(catalog: dict, key: str, kind: str):
    """Closed-catalog lookup: the entry for ``key``, or a loud KeyError
    naming the closed set in this module."""
    try:
        return catalog[key]
    except KeyError:
        raise KeyError(
            f"Unknown {kind} '{key}'. It must be one of the catalog entries "
            f"in tau2.multilingual.localization_catalog: {sorted(catalog)}"
        )


def get_guideline_example_kind(kind_id: str) -> GuidelineExampleKindDef:
    """Look up a catalog example kind, or raise if not in the closed set."""
    return _lookup(GUIDELINE_EXAMPLE_CATALOG, kind_id, "guideline example kind")


def render_guideline_example(
    kind_def: GuidelineExampleKindDef, items: list[str], *, native: bool
) -> str:
    """Render a kind's utterances into the guidelines' fixed English prose.

    The joiners/connectors are fixed English scaffold; only the quoted
    utterances vary. ``native=True`` appends the kind's ``native_note`` (e.g.
    the persona-precedence pointer); the English default rendering never
    carries it, so English personas keep today's text byte-identical.
    """
    style = kind_def.join_style
    if style == ExampleJoinStyle.APPENDED_EXAMPLE:
        if not items:
            return ""
        rendered = f' For example: "{items[0]}"'
    elif style == ExampleJoinStyle.QUOTED_SINGLE:
        rendered = f'"{items[0]}"'
    elif style == ExampleJoinStyle.QUOTED_LIST:
        rendered = ", ".join(f'"{item}"' for item in items)
    elif style == ExampleJoinStyle.QUOTED_OR:
        rendered = " or ".join(f'"{item}"' for item in items)
    elif style == ExampleJoinStyle.CHAIN:
        parts = [f'"{items[0]}"']
        for connector, item in zip(kind_def.chain_connectors, items[1:]):
            parts.append(f"→ {connector} →")
            parts.append(f'"{item}"')
        rendered = " ".join(parts)
    else:  # pragma: no cover - closed enum
        raise ValueError(f"Unhandled join style: {style}")
    if native and kind_def.native_note:
        rendered = f"{rendered} {kind_def.native_note}"
    return rendered


def get_symbol_def(symbol: str) -> SymbolDef:
    """Look up a catalog symbol, or raise if it is not in the closed set."""
    return _lookup(SYMBOL_CATALOG, symbol, "localization symbol")


def get_domain_term_catalog(domain: str) -> dict[str, DomainTermDef]:
    """The closed term catalog for a domain, or raise on an unknown domain."""
    return _lookup(DOMAIN_TERM_CATALOGS, domain, "glossary domain")


def get_honorific_concept(concept_id: str) -> HonorificConceptDef:
    """Look up a catalog honorific concept, or raise if not in the closed set."""
    return _lookup(HONORIFIC_CONCEPT_CATALOG, concept_id, "honorific concept")


def get_domain_term(domain: str, term_id: str) -> DomainTermDef:
    """Look up a domain's catalog term, or raise if not in the closed set."""
    return _lookup(
        get_domain_term_catalog(domain),
        term_id,
        f"localization domain term (domain '{domain}')",
    )
