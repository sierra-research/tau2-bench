# Copyright Sierra
"""Audition the Hindi personas (Rishika, Fatima) — native-review artifact.

For each persona this builds the REAL user-simulator system prompt (localized
Hindi voice guidelines + persona guidelines via the same code paths the
benchmark uses) and asks the model to produce sample utterances across a fixed
set of scenarios: a calm opening, spelling a booking reference, being confused,
being frustrated, and a long closing.

This is a MANUAL tool. It is NOT wired into CI and requires API keys
(it calls the LLM). The output is meant to be read by a native Hindi speaker to
judge whether the Hinglish is ecological.

Usage:
    # Needs whatever API key your configured model uses (e.g. ANTHROPIC_API_KEY,
    # OPENAI_API_KEY), same as the rest of tau2.
    PYTHONPATH=$PWD/src python scripts/audition_hindi_personas.py
    PYTHONPATH=$PWD/src python scripts/audition_hindi_personas.py \
        --persona fatima_hindi_v1 --model gpt-4o --n 5

The default model mirrors the benchmark default; override with --model.
"""

from __future__ import annotations

import argparse

from tau2.config import DEFAULT_LLM_USER
from tau2.data_model.message import SystemMessage, UserMessage
from tau2.multilingual.registry import get_language_pack
from tau2.user.user_simulator import SYSTEM_PROMPT, get_global_user_sim_guidelines_voice
from tau2.utils.llm_utils import generate

# (scenario label, agent's turn that the persona must respond to). Each is a
# self-contained mini-context so the persona's voice can be judged in isolation.
SCENARIOS: list[tuple[str, str]] = [
    (
        "calm open",
        "नमस्ते, thank you for calling. मैं आपकी कैसे मदद कर सकता हूँ?",
    ),
    (
        "stating the request",
        "ज़रूर, मैं देख लेता हूँ। आपकी क्या समस्या है exactly?",
    ),
    (
        "spelling a booking ref",
        "Sorry, ठीक से सुनाई नहीं दिया — क्या आप अपना booking reference एक-एक "
        "letter करके spell कर सकते हैं?",
    ),
    (
        "confused / asks to repeat",
        "आपकी ticket पर एक fare difference of two thousand three hundred rupees "
        "है, plus a change fee, और वो non-refundable component अलग से है।",
    ),
    (
        "frustrated / long hold",
        "I'm sorry for the wait. मुझे एक बार फिर से system check करना पड़ेगा, "
        "इसमें थोड़ा और time लगेगा।",
    ),
    (
        "long closing",
        "आपका काम हो गया है। Is there anything else I can help you with today?",
    ),
]

# A neutral scenario block so the user-sim has a concrete task to stay in.
INSTRUCTIONS = (
    "You are calling an airline customer-support line about a booking on your "
    "account. Your booking reference is RX4T9K. You want to understand the "
    "change fees on your ticket. You are not certain you want to proceed."
)


def build_system_prompt(persona_id: str) -> str:
    pack = get_language_pack("hi")
    if pack is None:
        raise SystemExit("Hindi pack not registered — check the loader.")
    persona = pack.get_persona(persona_id)
    if persona is None:
        raise SystemExit(
            f"Unknown persona '{persona_id}'. Options: {sorted(pack.personas)}"
        )

    guidelines = get_global_user_sim_guidelines_voice(language="hi")
    persona_guidelines = persona.to_guidelines_text() or ""
    if persona_guidelines:
        persona_guidelines = f"\n\n{persona_guidelines}\n"
    guidelines_with_persona = guidelines.replace(
        "<PERSONA_GUIDELINES>", persona_guidelines
    )
    return SYSTEM_PROMPT.format(
        global_user_sim_guidelines_with_persona=guidelines_with_persona,
        instructions=INSTRUCTIONS,
    )


def audition(persona_id: str, model: str, n: int) -> None:
    system_prompt = build_system_prompt(persona_id)
    print("=" * 78)
    print(f"PERSONA: {persona_id}   MODEL: {model}")
    print("=" * 78)

    count = 0
    for label, agent_turn in SCENARIOS:
        for i in range(n):
            messages = [
                SystemMessage(role="system", content=system_prompt),
                # The agent is the "user" turn from the simulator's perspective.
                UserMessage(role="user", content=agent_turn),
            ]
            msg = generate(
                model=model,
                messages=messages,
                call_name="audition_hindi_persona",
                temperature=0.9,
            )
            count += 1
            print(f"\n[{count}] scenario: {label}  (sample {i + 1}/{n})")
            print(f"  AGENT: {agent_turn}")
            print(f"  {persona_id.upper()}: {msg.content}")
    print(f"\nGenerated {count} sample utterances for {persona_id}.\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--persona",
        default="all",
        help="Persona id, or 'all' for both (default: all)",
    )
    parser.add_argument("--model", default=DEFAULT_LLM_USER, help="LLM to use")
    parser.add_argument(
        "--n",
        type=int,
        default=3,
        help="Samples per scenario (default: 3 → ~18-20 utterances per persona)",
    )
    args = parser.parse_args()

    if args.persona == "all":
        persona_ids = ["rishika_hindi_v1", "fatima_hindi_v1"]
    else:
        persona_ids = [args.persona]

    for persona_id in persona_ids:
        audition(persona_id, args.model, args.n)


if __name__ == "__main__":
    main()
