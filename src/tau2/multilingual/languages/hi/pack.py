# Copyright Sierra
"""The Hindi LanguagePack: personas, localized guidelines, conversation norms.

This module builds the single ``LanguagePack`` for Hindi and self-registers it
at import time (see ``__init__``). It is the reference implementation other
language owners copy: it touches NO shared code, only authors content.
"""

from tau2.multilingual.languages.hi.personas import HINDI_PERSONAS
from tau2.multilingual.registry import register_language_pack
from tau2.multilingual.schema import LanguagePack
from tau2.utils import DATA_DIR

# Localized voice user-simulator guidelines (re-authored, not translated).
GUIDELINES_VOICE_HI_PATH = (
    DATA_DIR / "tau2" / "multilingual" / "hi" / "simulation_guidelines_voice_hi.md"
)

# Re-authored from BACKCHANNEL_DECISION_PROMPT_CONTINUER in
# tau2.user.user_simulator_streaming, adapted to Hindi conversational norms.
# The YES/NO output contract and the {conversation_history} placeholder are
# preserved exactly; only the density guidance and few-shot examples change.
# Hindi-speaking listeners backchannel markedly more often than English ones
# (roughly every 5-10 seconds vs every 20-30 seconds in English), so the
# threshold for saying YES is lower here.
BACKCHANNEL_DECISION_PROMPT_HI = """You simulate a natural Hindi-speaking listener who frequently says brief continuers like "जी", "हाँ", or "अच्छा" to show they are following along.

<conversation_history>
{conversation_history}
</conversation_history>

The agent is still speaking [CURRENTLY SPEAKING, INCOMPLETE]. Ignore the trailing incomplete word/phrase — focus only on the COMPLETE sentences delivered so far in the agent's current turn.

Continuers ("जी", "जी हाँ", "हाँ", "अच्छा", "हम्म") are brief sounds that mean "I'm listening, keep going." They:
- Happen naturally and OFTEN during extended speech — Hindi listeners are very active backchannelers
- Show engagement without interrupting
- Are NOT responses to specific content — just signals of attention

Say YES if:
- The agent has completed at least 1 substantive sentence in their current turn
  (Very short fillers like "एक मिनट" or "देखता हूँ" on their own don't count)
- The user hasn't backchanneled in the immediately preceding beat (avoid two continuers back-to-back with nothing in between)
- It would feel natural to briefly signal "I'm still here / go on"

Say NO if:
- The agent has not yet completed even one substantive sentence
- The user backchanneled on the immediately preceding beat (don't stack two in a row)
- The agent's current turn contains or ends with a direct question (you should answer, not just backchannel)
- The agent is clearly mid-word with nothing complete yet

Frequency guidance:
- Continuers are FREQUENT in Hindi conversation — roughly 1 continuer per 1-2 substantive sentences of extended agent speech
- A warm, attentive Hindi listener says "जी" or "हाँ" often; long silences feel cold or like a dropped call
- Still avoid stacking two continuers with nothing in between
- When genuinely unsure, leaning YES is fine — active listening is the norm here

Examples:

AGENT: "नमस्ते! मैं आपकी कैसे म [CURRENTLY SPEAKING, INCOMPLETE]"
→ NO (just started, nothing complete yet)

AGENT: "जी बिल्कुल, मैं इसमें आपकी मदद कर सकता हूँ। पहले मुझे आपका account verify करना होगा। क्या आप अपना booking reference बता सकत [CURRENTLY SPEAKING, INCOMPLETE]"
→ NO (agent is asking a question — answer it, don't backchannel)

AGENT: "ठीक है, कोई बात नहीं। मैं आपका record अभी देख लेता हूँ। एक पल रुकिएगा, मैं system में check कर रह [CURRENTLY SPEAKING, INCOMPLETE]"
→ YES (a couple of substantive sentences, agent explaining the process — natural moment for "जी")

AGENT: "मुझे आपकी booking मिल गई है। इसमें एक flight Mumbai से Delhi की है। ये कल सुबह की है। अब cancellation के लिए हमारे पास कुछ opti [CURRENTLY SPEAKING, INCOMPLETE]"
→ YES (extended explanation with specifics — "अच्छा" or "हाँ" fits naturally)

[If the user said "जी" on the immediately preceding beat]
AGENT: "...तो ये रहे आपके options। अब मुझे बताइए कि आप कौन सा [CURRENTLY SPEAKING, INCOMPLETE]"
→ NO (user just backchanneled; don't stack another right away)

Respond with ONLY "YES" or "NO".
"""

# English defaults (cited so reviewers can see the multiplier):
#   DEFAULT_BACKCHANNEL_POISSON_RATE = 1.0 / 10.0 = 0.1 /s  (tau2.config)
#   OUT_OF_TURN_SPEECH_EVENTS_PER_MINUTE = 0.7              (tau2.voice_config)
# Hindi listeners backchannel ~2-3x as often, so we raise the Poisson rate to
# ~0.25 /s (2.5x the English 0.1 /s). Out-of-turn (side-talk to family/driver)
# is modestly higher than English given multi-generational households and
# calling-from-traffic contexts: 1.0 /min vs the English 0.7 /min.
HINDI_BACKCHANNEL_POISSON_RATE = 0.25
HINDI_OUT_OF_TURN_EVENTS_PER_MINUTE = 1.0

AGENT_LANGUAGE_CLAUSE_HI = (
    "Respond in Hindi using Devanagari script for Hindi and Latin script for "
    "English insertions; mirror the user's code-switching register; keep all "
    "tool calls, arguments, and structured outputs in English exactly as "
    "specified."
)


HINDI_PACK = LanguagePack(
    language="hi",
    display_name="Hindi",
    guidelines_voice_path=GUIDELINES_VOICE_HI_PATH,
    personas=HINDI_PERSONAS,
    acoustic_presets={},  # Indian acoustic presets land in PR7.
    backchannel_decision_prompt=BACKCHANNEL_DECISION_PROMPT_HI,
    agent_language_clause=AGENT_LANGUAGE_CLAUSE_HI,
    default_backchannel_poisson_rate=HINDI_BACKCHANNEL_POISSON_RATE,
    default_out_of_turn_events_per_minute=HINDI_OUT_OF_TURN_EVENTS_PER_MINUTE,
)


def register() -> None:
    """Register the Hindi pack. Idempotent-safe is the loader's job."""
    register_language_pack(HINDI_PACK)
