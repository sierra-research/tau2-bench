# Copyright Sierra
"""Parser for hyper-tau training-record transcripts (``case_*.md``).

Phone-call records use exactly three markers::

    **Turn N · Agent:** <spoken text>
    **Turn N · Customer:** <spoken text>
    **After turn N · Support console:** <non-spoken system event>

Turn text may hard-wrap onto continuation lines; those are joined with a
space. Any other bold marker is a format drift and raises, so renders never
silently drop content.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal, Union

from pydantic import BaseModel

TURN_RE = re.compile(r"^\*\*Turn (\d+) · (Agent|Customer):\*\*\s*(.*)$")
CONSOLE_RE = re.compile(r"^\*\*After turn (\d+) · Support console:\*\*\s*(.*)$")
CASE_RE = re.compile(r"^# Case (.+)$")
CHANNEL_RE = re.compile(r"^Channel:\s*(.+?)\s*$")
QA_STATUS_RE = re.compile(r"^QA status:\s*(.+?)\s*$")


class SpokenTurn(BaseModel):
    kind: Literal["turn"] = "turn"
    turn: int
    role: Literal["agent", "customer"]
    text: str


class ConsoleEvent(BaseModel):
    kind: Literal["console"] = "console"
    after_turn: int
    text: str


CallEvent = Union[SpokenTurn, ConsoleEvent]


class CallTranscript(BaseModel):
    case_id: str
    channel: str
    qa_status: str
    source_path: Path
    events: list[CallEvent]

    @property
    def spoken_turns(self) -> list[SpokenTurn]:
        return [event for event in self.events if isinstance(event, SpokenTurn)]

    @property
    def console_events(self) -> list[ConsoleEvent]:
        return [event for event in self.events if isinstance(event, ConsoleEvent)]

    @property
    def is_phone_call(self) -> bool:
        return self.channel == "phone call"


def parse_call_transcript(path: Path) -> CallTranscript:
    """Parse a training-record markdown file into a CallTranscript."""
    case_id = ""
    channel = ""
    qa_status = ""
    events: list[CallEvent] = []
    current: CallEvent | None = None

    for line_no, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.rstrip()
        if not line:
            continue

        if match := TURN_RE.match(line):
            current = SpokenTurn(
                turn=int(match.group(1)),
                role=match.group(2).lower(),
                text=match.group(3).strip(),
            )
            events.append(current)
        elif match := CONSOLE_RE.match(line):
            current = ConsoleEvent(
                after_turn=int(match.group(1)), text=match.group(2).strip()
            )
            events.append(current)
        elif line.startswith("**"):
            raise ValueError(
                f"{path}:{line_no}: unrecognized marker line: {line!r}"
            )
        elif match := CASE_RE.match(line):
            case_id = match.group(1).strip()
            current = None
        elif match := CHANNEL_RE.match(line):
            channel = match.group(1)
            current = None
        elif match := QA_STATUS_RE.match(line):
            qa_status = match.group(1)
            current = None
        elif current is not None:
            # Hard-wrapped continuation of the current turn/event.
            current.text = f"{current.text} {line.strip()}".strip()
        else:
            raise ValueError(
                f"{path}:{line_no}: content line outside any turn: {line!r}"
            )

    if not case_id:
        raise ValueError(f"{path}: missing '# Case ...' header")
    if not events:
        raise ValueError(f"{path}: no turns found")

    return CallTranscript(
        case_id=case_id,
        channel=channel,
        qa_status=qa_status,
        source_path=path,
        events=events,
    )


_PHONE_CHANNEL_RE = re.compile(r"^Channel:\s*phone call\s*$", re.MULTILINE)


def is_phone_call_file(path: Path) -> bool:
    """Check the Channel header line (not a substring anywhere in dialogue)."""
    return _PHONE_CHANNEL_RE.search(path.read_text()) is not None


def find_phone_call_transcripts(root: Path) -> list[Path]:
    """Find all phone-call training records under a directory."""
    paths = []
    for path in sorted(root.rglob("case_*.md")):
        if ".claude" in path.parts:
            continue
        if is_phone_call_file(path):
            paths.append(path)
    return paths
