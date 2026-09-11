# Copyright Sierra
"""Closed catalog of language-independent nativeness factor metadata.

Language packs select these ids and supply language-specific rubric text. The
pack loader rejects unknown ids, so a pack cannot invent an unreviewed axis.

Canonical questions follow the same style as pack rubrics: short, direct, and
plain-language. Each asks one observable question about agent speech. Concrete
target-language examples and exceptions belong in the language pack, not here.
"""

from functools import partial

from pydantic import BaseModel, ConfigDict, Field

from tau2.multilingual.factor_catalog import (
    FactorDefinition,
    FactorModality,
    catalog_factor,
    factor_definition,
)

DEFAULT_JUDGE_SEVERITY = 3


class NativenessFactorPromptBase(BaseModel):
    """Language-independent judge text shared by every use of one factor id."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    question: str = Field(description="Canonical binary question shown to the judge.")
    opportunity: str = Field(description="When the agent can exhibit the factor.")
    allowed: str = Field(description="Language-independent passing behavior.")
    violation: str = Field(description="Language-independent failing behavior.")


# Every factor owns one canonical question here. The catalog definition supplies
# the shared opportunity, allowed behavior, and violation; packs supply only the
# language's concrete rules and examples.
FACTOR_QUESTIONS: dict[str, str] = {
    "register_formality": (
        "Did the agent use an appropriate and consistent way of addressing the "
        "customer?"
    ),
    "register_diglossia": (
        "Did the agent use the spoken variety appropriate for a service call?"
    ),
    "regional_consistency": (
        "Did the agent stay within the expected regional variety of the language?"
    ),
    "honorific_agreement": (
        "Did the agent attach honorific grammar to the correct person and action?"
    ),
    "honorific_levels": (
        "Did the agent maintain the appropriate honorific speech level?"
    ),
    "keigo_levels": (
        "Did the agent use respectful and humble forms in the correct direction?"
    ),
    "honorific_address": (
        "Did the agent attach the expected title or honorific when using a name?"
    ),
    "pronoun_address": (
        "Did the agent use the socially appropriate form of address consistently?"
    ),
    "name_address_conventions": (
        "Did the agent use natural name order and title attachment?"
    ),
    "request_politeness": (
        "Did the agent frame requests with a natural level of politeness?"
    ),
    "cushioning": "Did the agent soften sensitive requests or refusals naturally?",
    "diminutives": (
        "Did the agent use natural softening forms where this language calls for them?"
    ),
    "routines": (
        "Did the agent use natural service-call opening, closing, and thanks routines?"
    ),
    "natural_word_choice": (
        "Did the agent use natural, idiomatic, and grammatically well-formed "
        "language in this utterance?"
    ),
    "code_switching": (
        "Did the agent use the language's natural balance of local and borrowed terms?"
    ),
    "hinglish_codeswitch": "Did the agent use natural Hindi or Hinglish word choice?",
    "konglish": "Did the agent use established Korean loanwords naturally?",
    "untranslated_terms": (
        "Did the agent avoid unnecessary untranslated English when an ordinary "
        "local equivalent exists?"
    ),
    "translationese": "Did the agent avoid word-for-word English-shaped phrasing?",
    "modal_particles": (
        "Did the agent use modal particles naturally when the language calls for them?"
    ),
    "sentence_final": (
        "Did the agent use sentence endings with the right meaning and politeness?"
    ),
    "discourse_markers": (
        "Did the agent connect spoken discourse with natural conversational markers?"
    ),
    "gender_agreement": (
        "Did gendered language agree internally and match the known gender of the "
        "agent or customer?"
    ),
    "gender_case": "Did gender and case marking agree in the same construction?",
    "aspect": "Did the agent use the correct aspect for the intended meaning?",
    "classifiers": "Did the agent use the correct classifier for the noun?",
    "counting_units": (
        "Did the agent use the natural number form and counting word for each item?"
    ),
    "counters": "Did the agent use the correct number system and counter?",
    "verb_morphology": (
        "Did the agent use the correct verb person, mood, tense, and core function words?"
    ),
    "word_order": "Did the agent use natural word order for this language?",
    "vowel_harmony": "Did suffixes follow the language's vowel-harmony rules?",
}


_f = partial(factor_definition, default_severity=DEFAULT_JUDGE_SEVERITY)


# The closed set. Keyed by id. Every pack judge factor MUST reference one of these.
JUDGE_FACTOR_CATALOG: dict[str, FactorDefinition] = {
    f.id: f
    for f in [
        # --- register / formality ------------------------------------------------
        _f(
            "register_formality",
            "register",
            "agent addresses the customer / makes requests",
            "Use a service-appropriate form of address and keep the relationship "
            "consistent across the call.",
            "Uses an inappropriate form of address or abruptly mixes incompatible "
            "levels of formality",
        ),
        # VOICE-ONLY: the factor's question IS the spoken/written axis —
        # written service chat legitimately uses the written/formal variety,
        # so the rubric's polarity inverts in text.
        _f(
            "register_diglossia",
            "register",
            "agent addresses the customer / makes requests",
            "Picks the right variety on a diglossic continuum (e.g. educated "
            "spoken vs. literary/dialectal) for a phone service call.",
            "Uses the wrong variety of the language for a phone call "
            "(e.g. bookish/literary forms where people speak the everyday variety)",
            modality=FactorModality.VOICE,
        ),
        _f(
            "regional_consistency",
            "register",
            "agent uses region-marked lexicon/conjugation",
            "Keeps one regional variety's lexicon and conjugation consistent "
            "rather than mixing incompatible regional forms.",
            "Mixes words or grammar from different regions of the "
            "language in the same call",
        ),
        # --- honorifics ----------------------------------------------------------
        _f(
            "honorific_agreement",
            "honorifics",
            "agent uses honorific address or refers to customer/agent actions",
            "Matches honorific address with the grammar it requires and aims subject "
            "honorification at the customer rather than the agent.",
            "Uses familiar agreement with honorific address or aims subject "
            "honorification at the wrong participant",
        ),
        _f(
            "honorific_levels",
            "honorifics",
            "agent addresses the customer / makes requests",
            "Maintains the appropriate honorific speech level for the customer "
            "throughout, without dropping levels.",
            "Drops or switches the polite speech level partway through the call",
        ),
        _f(
            "keigo_levels",
            "honorifics",
            "agent makes a request or refers to its own/customer actions",
            "Correct keigo-style separation of respectful vs. humble forms and "
            "consistent polite endings.",
            "Mixes up respectful vs. humble forms (keigo), or lets polite endings slip",
        ),
        # --- address terms -------------------------------------------------------
        _f(
            "honorific_address",
            "address",
            "agent addresses the customer by name",
            "Pairs names/titles with the expected honorific address terms rather "
            "than a bare or mismatched form.",
            "Addresses the customer by name without the expected "
            "title/honorific, or with the wrong one",
        ),
        _f(
            "pronoun_address",
            "address",
            "agent addresses the customer / makes requests",
            "Selects the socially appropriate address pronoun/kin-term system for "
            "a service interaction and keeps it consistent.",
            "Uses the wrong word for 'you' (or the wrong kin-term) for "
            "a service call, or switches mid-call",
        ),
        _f(
            "name_address_conventions",
            "address",
            "agent names or directly addresses the customer",
            "Uses the language's expected name order and attaches service titles "
            "or honorifics naturally.",
            "Reorders the name or uses a bare or mismatched name-title form",
        ),
        # --- politeness ----------------------------------------------------------
        _f(
            "request_politeness",
            "politeness",
            "agent asks the customer to do or provide something",
            "Frames requests indirectly and politely rather than as blunt or "
            "calqued imperatives.",
            "Phrases requests as blunt commands instead of the polite, "
            "indirect asks natives use",
        ),
        _f(
            "cushioning",
            "politeness",
            "agent asks the customer for something",
            "Softens requests/refusals with the expected cushioning phrases "
            "(set-phrase apologies, hedges) before the ask.",
            "Asks or refuses abruptly, skipping the customary softening "
            "phrases that come first",
        ),
        _f(
            "diminutives",
            "politeness",
            "agent asks the customer to wait / softens",
            "Uses diminutive/softening morphology where natives do to sound warm "
            "rather than curt.",
            "Sounds curt in moments where a native would soften the "
            "phrase (e.g. 'un momentito')",
        ),
        _f(
            "routines",
            "politeness",
            "agent opens, closes, or responds to thanks",
            "Uses the conventional politeness routines for openings, closings, "
            "and responses to thanks.",
            "Opens, closes, or responds to 'thank you' with unnatural "
            "phrasing instead of the standard formulas",
        ),
        # --- code-switching ------------------------------------------------------
        _f(
            "natural_word_choice",
            "natural_word_choice",
            "every non-empty agent utterance",
            "Uses locally natural vocabulary, idiomatic sentence structure, and "
            "well-formed verb and function-word grammar.",
            "Contains a locally identifiable defect in word choice, translated "
            "sentence structure, or verb and function-word grammar",
        ),
        _f(
            "code_switching",
            "code_switching",
            "agent uses domain/tech nouns",
            "Uses the natural balance of local terms and established borrowings for "
            "the target language and region.",
            "Uses unnecessary raw English or forces a literal local coinage where "
            "native speakers use an ordinary local term or established borrowing",
        ),
        _f(
            "hinglish_codeswitch",
            "code_switching",
            "agent uses domain/tech nouns",
            "Natural Hinglish: keeps common domain/tech nouns in English rather "
            "than Sanskritized shuddh-Hindi calques.",
            "Uses stiff pure-Hindi coinages where natural Hinglish "
            "keeps the English word",
        ),
        _f(
            "konglish",
            "code_switching",
            "agent uses domain/tech nouns",
            "Natural Konglish: uses the established borrowed forms for domain/tech "
            "nouns rather than forced pure-Korean coinages.",
            "Forces unnatural pure-Korean coinages where the "
            "established Konglish loanword is what people say",
        ),
        _f(
            "untranslated_terms",
            "code_switching",
            "agent refers to domain labels/entities or explains next steps",
            "Uses ordinary local terms while allowing established loanwords, names, "
            "product labels, and identifiers.",
            "Leaves English words or whole English phrases untranslated "
            "where the language has its own everyday way to say them",
        ),
        # --- idiom ---------------------------------------------------------------
        _f(
            "translationese",
            "idiom",
            "agent expresses intent or explains",
            "Idiomatic native phrasing for intents/connectives rather than "
            "word-for-word English calques.",
            "Phrasing sounds like a word-for-word translation from "
            "English rather than how a native would say it",
        ),
        # --- particles -----------------------------------------------------------
        _f(
            "modal_particles",
            "particles",
            "agent makes statements/requests",
            "Uses modal/discourse particles to color statements and requests as "
            "natives do (not omitting or overusing them).",
            "Speech sounds flat or off because the little 'flavor' "
            "particles are missing — or are overused",
        ),
        _f(
            "sentence_final",
            "particles",
            "agent makes statements/requests",
            "Uses sentence-final particles with the right pragmatic force and "
            "politeness, not omitted or mismatched.",
            "Sentence-ending particles are missing or carry the wrong "
            "tone/politeness for the moment",
        ),
        # --- discourse -----------------------------------------------------------
        # VOICE-ONLY: the pass/fail contrast penalizes written-style
        # connectives in favor of natural SPOKEN markers; in writing the
        # written connectives are the native behavior.
        _f(
            "discourse_markers",
            "discourse",
            "agent speaks in connected discourse",
            "Connects discourse with native filler/transition markers instead of "
            "stiff written connectives.",
            "Links sentences with stiff written-style connectives "
            "instead of the natural spoken ones",
            modality=FactorModality.VOICE,
        ),
        # --- gender --------------------------------------------------------------
        _f(
            "gender_agreement",
            "gender",
            "agent uses gendered words, titles, adjectives, participles, or numerals",
            "Keep gender agreement internally consistent and match any agent or "
            "customer gender explicitly provided in the judge context.",
            "Uses conflicting grammatical gender or refers to the known agent or "
            "customer with the wrong gender",
        ),
        _f(
            "gender_case",
            "gender",
            "agent uses past-tense verbs / case-governed nouns",
            "Gender AND case marking co-vary correctly on past-tense verbs, "
            "numerals, and case-governed nouns.",
            "Gets gender or case endings wrong on past-tense verbs, "
            "numerals, or nouns that require them",
        ),
        # --- aspect --------------------------------------------------------------
        _f(
            "aspect",
            "aspect",
            "agent describes actions/state changes",
            "Picks the correct aspect (perfective/imperfective) for completed vs. "
            "ongoing actions.",
            "Uses the wrong verb form for completed vs. ongoing actions "
            "(aspect errors)",
        ),
        # --- classifiers / counters ----------------------------------------------
        _f(
            "classifiers",
            "classifiers",
            "agent counts or refers to classified nouns",
            "Uses the correct classifier for counted/referenced nouns.",
            "Uses the wrong (or a lazy generic) classifier when "
            "counting or referring to things",
        ),
        _f(
            "counting_units",
            "counting",
            "agent counts people or items",
            "Uses the natural number form and counting word for each person or item.",
            "Uses a number form or counting word that does not fit what is counted",
        ),
        _f(
            "counters",
            "counters",
            "agent counts items (bags, tickets, seats, …)",
            "Uses the correct number system and counter for the counted item.",
            "Uses the wrong number system or counter for the counted item",
        ),
        # --- morphosyntax --------------------------------------------------------
        _f(
            "verb_morphology",
            "morphosyntax",
            "agent conjugates verbs / selects copulas or core function words",
            "Uses the correct verb person, mood, tense, copula, and core function words.",
            "Conjugates a verb in the wrong person, mood, or tense, picks the "
            "wrong copula, or misuses a core function word",
        ),
        _f(
            "word_order",
            "morphosyntax",
            "agent uses subordinate clauses",
            "Correct native word order, especially verb placement in subordinate "
            "clauses.",
            "Puts words in an un-native order — especially the verb in "
            "the wrong place in subordinate clauses",
        ),
        _f(
            "vowel_harmony",
            "morphosyntax",
            "agent suffixes nouns (incl. names/loanwords)",
            "Suffixes obey vowel harmony, including on names and loanwords.",
            "Attaches suffixes that break vowel harmony, especially on "
            "names and loanwords",
        ),
    ]
}


def get_catalog_factor(factor_id: str) -> FactorDefinition:
    """Look up a catalog factor by id, or raise if it is not in the closed set."""
    return catalog_factor(
        JUDGE_FACTOR_CATALOG,
        factor_id,
        axis="nativeness judge",
        module="tau2.multilingual.nativeness_catalog",
    )


def get_factor_prompt_base(factor_id: str) -> NativenessFactorPromptBase:
    """Return the canonical language-independent judge base for one factor."""
    factor = get_catalog_factor(factor_id)
    return NativenessFactorPromptBase(
        question=FACTOR_QUESTIONS[factor_id],
        opportunity=factor.opportunity,
        allowed=factor.description,
        violation=factor.annotator_label,
    )
