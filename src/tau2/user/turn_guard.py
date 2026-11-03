"""Guard against simulated-user replies that contain more than one turn.

Some user-simulator models (e.g. gpt-oss-120b under the v2.0 English caller
persona, see issue #710) write several turns in a single message: sentences
glued together without whitespace, lines spoken for the agent, or text that
continues after a control token such as ``###STOP###``. Because the stop check
is a plain substring test, such a message ends the conversation before the
agent can act, and the failure is charged to the agent.

This module provides a cheap, deterministic detector and an optional repair.
"""

import re
from dataclasses import dataclass, field
from typing import Literal, Optional

from tau2.user.user_simulator_base import OUT_OF_SCOPE, STOP, TRANSFER

TurnGuardMode = Literal["off", "detect", "truncate"]

CONTROL_TOKENS = (STOP, TRANSFER, OUT_OF_SCOPE)

# Sentence end immediately followed by the start of a new sentence, e.g.
# "handy.Okay" or "2023.Can". Requires a lowercase letter or digit before the
# punctuation so abbreviations like "U.S.A" and decimals like "4.1" don't fire.
_GLUED_SENTENCE = re.compile(r"(?<=[a-z0-9\)\"'])[.?!](?=[A-Z])")

# A line or inline span where the simulator speaks as the other party.
_ROLE_PREFIX = re.compile(
    r"(?:^|\n|(?<=[.?!]))\s*(?:\*\*)?(?:agent|assistant|representative|rep|support)"
    r"(?:\*\*)?\s*:",
    re.IGNORECASE,
)


@dataclass
class TurnGuardResult:
    issues: list[str] = field(default_factory=list)
    repaired: Optional[str] = None

    @property
    def is_multi_turn(self) -> bool:
        return bool(self.issues)


def _first_control_token(text: str) -> Optional[tuple[int, str]]:
    hits = [(text.find(t), t) for t in CONTROL_TOKENS if t in text]
    return min(hits) if hits else None


def detect_multi_turn(text: Optional[str]) -> list[str]:
    """Return a list of issue labels found in a user-simulator message."""
    if not text:
        return []
    issues = []
    hit = _first_control_token(text)
    if hit is not None:
        idx, token = hit
        if text[idx + len(token) :].strip():
            issues.append("text_after_control_token")
    if _ROLE_PREFIX.search(text):
        issues.append("agent_impersonation")
    if _GLUED_SENTENCE.search(text):
        issues.append("glued_sentences")
    return issues


def truncate_to_first_turn(text: str) -> str:
    """Keep only the first user turn.

    Cuts at the earliest of: an agent/assistant role prefix, or a glued sentence
    boundary. A control token that only appears after the cut is dropped, so a
    leaked ``###STOP###`` no longer ends the call. If the message starts with
    a control token after the cut point it is kept as-is.
    """
    cut = len(text)
    m = _ROLE_PREFIX.search(text)
    if m and m.start() > 0:
        cut = min(cut, m.start())
    g = _GLUED_SENTENCE.search(text)
    if g:
        cut = min(cut, g.end())
    hit = _first_control_token(text)
    if hit is not None:
        idx, token = hit
        # Control token inside the first turn: keep through the token.
        if idx < cut:
            cut = min(cut, idx + len(token))
    head = text[:cut].strip()
    return head or text.strip()


def check_user_message(text: Optional[str], mode: TurnGuardMode) -> TurnGuardResult:
    if mode == "off" or not text:
        return TurnGuardResult()
    result = TurnGuardResult(issues=detect_multi_turn(text))
    if result.is_multi_turn and mode == "truncate":
        result.repaired = truncate_to_first_turn(text)
    return result
