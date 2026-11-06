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

from tau2.data_model.message import Message, UserMessage
from tau2.user.user_simulator_base import OUT_OF_SCOPE, STOP, TRANSFER

TurnGuardMode = Literal["off", "detect", "truncate"]

CONTROL_TOKENS = (STOP, TRANSFER, OUT_OF_SCOPE)

# Sentence end immediately followed by the start of a new sentence, e.g.
# "handy.Okay", "2023.Can" or "exchange?Actually". The stop must come after two
# lowercase letters or digits and before a capitalised word or "I", so
# abbreviations ("e.g.The", "Dr.Lee", "Ph.D"), version strings ("2.0.Beta"),
# file names ("photo.JPG") and decimals ("4.1") don't match.
_GLUED_SENTENCE = re.compile(r"(?<=[a-z0-9][a-z0-9\)\"'])[.?!](?=[A-Z][a-z]|I\b)")

# A stop after an English, Spanish or Portuguese title or "vs" ("Mrs.Davis",
# "mr.Jones", "Sra.Ruiz", "vs.Apple") or after a dotted number ("3.12.Final",
# "10.0.19045.Update") is not a sentence end.
_NOT_A_SENTENCE_END = re.compile(
    r"(?:\b(?:mrs?|ms|dra?|profa?|sra?|srta|vs)|\d\.\d+)$", re.IGNORECASE
)

# Emails, links and domains ("Mei.Davis@example.com", "Amazon.Com") are
# skipped: a stop before a capital there is not a sentence end.
_LINK_OR_EMAIL = re.compile(
    r"@|/|www\.|\.(?:com|net|org|io|ai|co|uk|us)\b", re.IGNORECASE
)

# Chinese, Japanese and Thai don't put a space after a full stop, so glued
# sentences are normal there. Messages containing these scripts are not
# checked for glue.
_UNSPACED_SCRIPT = re.compile(
    "[\u0e00-\u0e7f"  # Thai
    "\u3000-\u30ff"  # CJK punctuation, Hiragana, Katakana
    "\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"  # CJK ideographs
    "\uff00-\uffef]"  # Fullwidth and halfwidth forms
)

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


def _glued_sentence_end(text: str) -> Optional[int]:
    """Index just after the first sentence end glued to the next sentence."""
    if _UNSPACED_SCRIPT.search(text):
        return None
    for token in re.finditer(r"\S+", text):
        word = token.group()
        if _LINK_OR_EMAIL.search(word):
            continue
        for glue in _GLUED_SENTENCE.finditer(word):
            if not _NOT_A_SENTENCE_END.search(word[: glue.start()]):
                return token.start() + glue.end()
    return None


def detect_multi_turn(text: Optional[str]) -> list[str]:
    """Return a list of issue labels found in a user-simulator message."""
    if not text:
        return []
    issues = []
    hit = _first_control_token(text)
    if hit is not None:
        idx, token = hit
        tail = text[idx + len(token) :]
        for t in CONTROL_TOKENS:
            tail = tail.replace(t, " ")
        # A letter or digit must follow, so "###STOP###." is still one turn.
        if any(ch.isalnum() for ch in tail):
            issues.append("text_after_control_token")
    if _ROLE_PREFIX.search(text):
        issues.append("agent_impersonation")
    if _glued_sentence_end(text) is not None:
        issues.append("glued_sentences")
    return issues


def truncate_to_first_turn(text: str) -> str:
    """Keep only the first user turn.

    Cuts at the earliest of: an agent/assistant role prefix, or a glued sentence
    boundary. A control token that only appears after the cut is dropped, so a
    leaked ``###STOP###`` no longer ends the call. A control token before the
    cut is kept, and anything after it is dropped.
    """
    cut = len(text)
    m = _ROLE_PREFIX.search(text)
    if m and m.start() > 0:
        cut = min(cut, m.start())
    glue = _glued_sentence_end(text)
    if glue is not None:
        cut = min(cut, glue)
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


def merge_counts(messages: list[Message], mode: TurnGuardMode) -> dict:
    """Count the user replies the guard flagged in one simulation.

    Truncated runs are not comparable with default runs, so the orchestrator
    stores this in ``SimulationRun.info["turn_guard"]`` when truncating, and
    the metrics report the merge rate next to the reward.
    """
    replies = [m for m in messages if isinstance(m, UserMessage) and m.content]
    merged = sum(1 for m in replies if "turn_guard" in (m.raw_data or {}))
    return {"mode": mode, "user_messages": len(replies), "merged_messages": merged}
