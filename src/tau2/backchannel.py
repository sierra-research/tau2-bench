"""Backchannel density knob — single source of truth for how often a voice
user-simulator emits brief continuers ("uh-huh", "जी", "はい") while listening.

A backchannel check fires roughly every ~2s while the agent speaks and an LLM
decides YES/NO each time, so density is governed by a small per-check fire
probability. Density is set by ONE parameter — :class:`BackchannelLevel`
(``low``/``medium``/``high``) — mapped to a structured :class:`BackchannelProfile`
and rendered into a single shared template. The correctness gates are IDENTICAL
across levels; only the density axes (trigger threshold, refractory window,
target frequency, eagerness) vary. Language packs supply only the surface
(continuer phrases + display name); they do not control density.

The three levels are RELATIVE density tiers, not absolute bc/conversation
guarantees: the achievable rate is domain-capped because task-oriented calls
have few genuinely-eligible moments with the correctness gates enforced.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# Upper bound on a persona's ``backchannel_phrases`` inventory.
#
# The decision to fire is LLM-gated, but the phrase is a UNIFORM RANDOM DRAW
# with zero context (``user_simulator_streaming._generate_backchannel_message``).
# A list that mixes true continuers with acknowledgments therefore emits
# acknowledgments at contextually random moments — a Mandarin reviewer caught
# the sim saying 明白了 ("I understand how to fix it") mid-explanation, which
# the agent read as "issue resolved" and ended the call. Keeping the inventory
# to one or two PURE continuers ("mm-hmm"-class: "I'm listening, keep going"
# and nothing else) makes the random draw safe by construction.
MAX_BACKCHANNEL_PHRASES = 2

# Upper bound on the LENGTH of one continuer, in words (converted per language
# by ``tau2.multilingual.brevity``). Two rather than one, because several
# languages' natural continuer is a doubled hum written with a space — Turkish
# "hı hı", Japanese "うんうん". Anything longer has stopped being a hum and is a
# turn-level response, which is the same failure the count bound guards
# against arriving by a different route.
MAX_BACKCHANNEL_PHRASE_WORDS = 2


class BackchannelLevel(str, Enum):
    """How often the user backchannels. One knob, three settings."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class BackchannelProfile:
    """The structured density axes a level expands into.

    The correctness gates (no mid-question, not just-started, respect the
    refractory window) are fixed in the template; this profile only tunes how
    *dense* backchanneling is once those gates pass.
    """

    min_sentences_word: str  # how many complete sentences before eligible
    refractory_desc: str  # how recently the user must have been silent
    target_per_n: str  # ~1 continuer per N substantive sentences
    eagerness: str  # how readily to fire ONCE the gates pass (the density dial)


BACKCHANNEL_PROFILES: dict[BackchannelLevel, BackchannelProfile] = {
    BackchannelLevel.LOW: BackchannelProfile(
        min_sentences_word="two",
        refractory_desc="the last 2-3 exchanges",
        target_per_n="3-4",
        eagerness="a brief continuer at a clear pause in a long explanation is natural — once or twice over the course of the call.",
    ),
    BackchannelLevel.MEDIUM: BackchannelProfile(
        min_sentences_word="one",
        refractory_desc="the last 1-2 exchanges",
        target_per_n="2-3",
        eagerness="give a brief continuer at the natural pauses — a few times across the agent's longer turns.",
    ),
    BackchannelLevel.HIGH: BackchannelProfile(
        min_sentences_word="one",
        refractory_desc="the immediately preceding beat",
        target_per_n="1-2",
        eagerness="readily give a brief continuer at most natural pauses — an attentive listener in this language does this often, so on a clearly eligible pause, lean YES.",
    ),
}

# Single shared template. The four "Say YES only if ALL hold" gates are constant
# across every level and language — that is what keeps correctness uniform. Only
# the {slots} drawn from the profile + the language surface change.
#
# The floor-yielding gate covers questions AND directives: "please read the full
# phone number digit by digit" is an imperative, not a question, so the
# question-only wording let the sim backchannel over a direct request for
# information (caught by a Mandarin reviewer on live runs). The gate is stated
# language-generically — no per-level or per-language fork.
BACKCHANNEL_PROMPT_TEMPLATE = """You simulate a natural {language_name}-speaking listener who gives brief continuers like {phrase_examples} to show they are following along.

<conversation_history>
{{conversation_history}}
</conversation_history>

The agent is still speaking [CURRENTLY SPEAKING, INCOMPLETE]. Ignore the trailing incomplete word/phrase — focus only on the COMPLETE sentences delivered so far in the agent's current turn.

Continuers (like {phrase_examples}) are brief sounds that mean "I'm listening, keep going." They signal attention without taking the floor — they are NOT answers to the agent's content.

Say YES only if ALL of the following hold:
- The agent has completed at least {min_sentences_word} full, substantive sentence(s) in their current turn (greetings and short fillers like "let me check" do NOT count as substantive)
- The user has NOT spoken or backchanneled within {refractory_desc}
- The agent's current turn does NOT contain or end with a question, a request, or an instruction addressed to the user — anything that asks the user to answer, confirm, supply information, or do something ("read me the full number", "check your settings and tell me what you see", "let me know when you're there") deserves a real response, not a continuer
- The agent has NOT just started their turn

Say NO if ANY of those fail — the gates are absolute and override the frequency guidance below.

Frequency guidance (applies ONLY once every gate above is satisfied):
- During the agent's extended turns, {eagerness}
- Aim for about 1 continuer per {target_per_n} substantive sentences
- Never stack two continuers back-to-back

Respond with ONLY "YES" or "NO".
"""


def render_backchannel_prompt(
    level: BackchannelLevel,
    *,
    language_name: str,
    phrases: list[str] | None,
) -> str:
    """Render the decision prompt for ``level`` in ``language_name``.

    ``phrases`` are the persona's continuer phrases (a few are shown as
    examples). The returned prompt still contains a ``{conversation_history}``
    placeholder for the caller to ``.format()`` at decision time — matching the
    contract every downstream consumer relies on.
    """
    if not isinstance(level, BackchannelLevel):
        level = BackchannelLevel(level)
    profile = BACKCHANNEL_PROFILES[level]
    examples = phrases[:4] if phrases else ["uh-huh", "mm-hmm"]
    phrase_examples = ", ".join(f'"{p}"' for p in examples)
    return BACKCHANNEL_PROMPT_TEMPLATE.format(
        language_name=language_name,
        phrase_examples=phrase_examples,
        min_sentences_word=profile.min_sentences_word,
        refractory_desc=profile.refractory_desc,
        target_per_n=profile.target_per_n,
        eagerness=profile.eagerness,
    )
