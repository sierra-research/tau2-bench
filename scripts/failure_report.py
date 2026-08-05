#!/usr/bin/env python3
"""Print a tau2 run's failures as a report you can hand to a coding agent.

    uv run python scripts/failure_report.py <run> [<run> ...] [--top N] [--out FILE]

Answers the three questions a fix needs, in order:

1. WHICH tasks failed, and did they fail at the reward or at the wire?
2. WHAT did the agent do instead — expected action vs. the call it really made,
   plus the judge's own justification for each missed assertion.
3. Is the run even TRUSTWORTHY — rate-limited caller speech, unfinished tasks,
   and the wire-level anomalies that make a score say more about the harness
   than about the agent.

Three things it is deliberate about, each a trap that has produced a wrong
conclusion on this benchmark before:

- **Retries are collapsed to the highest trial per task.** The runner retries,
  and a retry OVERWRITES the earlier score, so counting every simulation row
  double-counts a task and mixes a superseded score into the mean.
- **Unfinished runs are labelled, not silently averaged.** A mean over 12 of 20
  tasks is not the run's score, and the difference is invisible once it has been
  pasted somewhere as a number.
- **Wire anomalies are reported separately from reward.** A 0.0 caused by a
  provider that never connected is not evidence about agent quality, and the two
  want different fixes.

Reads only the run directory (`data/simulations/<run>/`) — no API calls, no
imports from tau2, so it works on a copied artifact directory.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

TS_RE = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+)")
EVENT_RE = re.compile(r"AAI event: ([a-z_]+) - (\{.*\})\s*$")
TEXT_RE = re.compile(r"'text': (\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*')")

# A turn whose first agent word lands later than this is dead air the caller
# hears as an unanswered question; the pipeline's own hold phrase and dead-air
# cover are supposed to keep any turn well under it.
SLOW_FIRST_WORD_SEC = 10.0


def _parse_events(path: Path) -> list[tuple[datetime, str, str | None]]:
    """Timestamped wire events from one sim's task.log."""
    events = []
    with path.open(errors="replace") as fh:
        for line in fh:
            match = EVENT_RE.search(line)
            if not match:
                continue
            stamp = TS_RE.match(line)
            if not stamp:
                continue
            when = datetime.strptime(stamp.group(1), "%Y-%m-%d %H:%M:%S.%f")
            text_match = TEXT_RE.search(match.group(2))
            text = text_match.group(1)[1:-1] if text_match else None
            events.append((when, match.group(1), text))
    return events


def _wire_anomalies(log_path: Path) -> dict:
    """Protocol-level findings for one session.

    Each of these is a client-visible ordering or timing property, not a style
    preference: a transcript after the turn's terminal frame, a committed user
    turn that never got a reply, or a gap long enough that the caller gives up.
    """
    events = _parse_events(log_path)
    kinds = Counter(kind for _, kind, _ in events)
    out = {
        "events": kinds,
        "transcript_after_cancel": [],
        "slow_first_word": [],
        "abandoned_turns": [],
        "zero_duration_speech": 0,
    }

    for i, (when, kind, _) in enumerate(events):
        if kind == "cancelled":
            # `cancelled` is the terminal frame of an interrupted turn (a
            # cancelled reply emits no reply_done), so text after it belongs to
            # a reply the client has already flushed.
            for when2, kind2, text2 in events[i + 1 :]:
                if kind2 in ("cancelled", "reply_done", "user_transcript"):
                    break
                if kind2 == "agent_transcript":
                    out["transcript_after_cancel"].append(
                        {
                            "delta_ms": round((when2 - when).total_seconds() * 1000, 1),
                            "at": when2.strftime("%H:%M:%S.%f")[:-3],
                            "text": (text2 or "")[:160],
                        }
                    )
                    break

        if kind == "user_transcript":
            first_agent = done = next_user = None
            for when2, kind2, _ in events[i + 1 :]:
                if kind2 == "agent_transcript" and first_agent is None:
                    first_agent = when2
                if kind2 == "reply_done" and done is None:
                    done = when2
                if kind2 == "user_transcript":
                    next_user = when2
                    break
            if first_agent:
                gap = (first_agent - when).total_seconds()
                if gap > SLOW_FIRST_WORD_SEC:
                    out["slow_first_word"].append(
                        {
                            "gap_sec": round(gap, 1),
                            "at": when.strftime("%H:%M:%S.%f")[:-3],
                        }
                    )
            if done is None and (first_agent is None or next_user):
                # The user committed a turn and the reply never completed.
                until = next_user or events[-1][0]
                out["abandoned_turns"].append(
                    {
                        "waited_sec": round((until - when).total_seconds(), 1),
                        "at": when.strftime("%H:%M:%S.%f")[:-3],
                    }
                )

    for i in range(len(events) - 1):
        if events[i][1] == "speech_started" and events[i + 1][1] == "speech_stopped":
            if (events[i + 1][0] - events[i][0]).total_seconds() < 0.05:
                out["zero_duration_speech"] += 1

    return out


def _session_logs(run_dir: Path) -> dict[str, Path]:
    """sim id -> task.log. The artifact dir names each session `sim_<id>`."""
    logs = {}
    for log in run_dir.glob("artifacts/task_*/sim_*/task.log"):
        logs[log.parent.name.removeprefix("sim_")] = log
    return logs


def _rate_limit_hits(log: Path) -> int:
    """429s on the caller's own voice synthesis.

    Not cosmetic: a synthesis that burns its retries is a caller utterance that
    arrives late or not at all, and the delayed ones are disproportionately the
    spelled-out names and digits that authentication turns on.
    """
    try:
        return sum(
            1
            for line in log.open(errors="replace")
            if "synthesize_voice failed" in line
        )
    except OSError:
        return 0


def _latest_trials(index) -> dict[str, dict]:
    """Collapse retries: the highest trial per task is the authoritative score.

    `simulation_index` is a LIST of entries; accept a mapping too, so a
    hand-assembled or older results.json still reports.
    """
    entries = index.values() if isinstance(index, dict) else (index or [])
    best: dict[str, dict] = {}
    for entry in entries:
        if entry is None:
            continue
        task_id = str(entry.get("task_id"))
        current = best.get(task_id)
        if current is None or entry.get("trial", 0) >= current.get("trial", 0):
            best[task_id] = entry
    return best


def _actual_calls(sim: dict) -> list[dict]:
    """Every tool call the agent really made, in order, from the ticks.

    `sim["messages"]` is empty in audio-native runs — the per-tick
    `agent_tool_calls` are the only record.
    """
    calls = []
    for tick in sim.get("ticks") or []:
        for call in tick.get("agent_tool_calls") or []:
            calls.append(
                {"name": call.get("name"), "arguments": call.get("arguments") or {}}
            )
    return calls


def _heard_chars(sim: dict) -> int:
    """How much agent speech the simulated caller actually received."""
    total = 0
    for tick in sim.get("ticks") or []:
        chunk = tick.get("agent_chunk") or {}
        total += len(chunk.get("content") or "")
    return total


def _classify(sim: dict, actual: list[dict], failed_actions: list[dict]) -> str:
    """Name the failure mode, because they want different fixes.

    'RED' hides at least four problems: an agent that never got in the door, one
    that acted on mis-heard arguments, one that ran out of steps, and one that
    behaved correctly and merely failed to say something.
    """
    auth_tools = {"find_user_id_by_name_zip", "find_user_id_by_email"}
    attempted_auth = [c for c in actual if c["name"] in auth_tools]
    reward_info = sim.get("reward_info") or {}
    breakdown = reward_info.get("reward_breakdown") or {}

    if not actual:
        return "NO_TOOL_CALLS — agent never acted at all"
    if attempted_auth and len(actual) == len(attempted_auth):
        # Tried to authenticate, never got past it: the task body is unreachable,
        # so every expected action is missing for one upstream reason.
        return "NEVER_AUTHENTICATED — only auth calls made; task body unreachable"
    if failed_actions and breakdown.get("NL_ASSERTION", 0) == 1.0:
        return "WRONG_ACTIONS_GOOD_TALK — converses well, acts wrong (check args)"
    if failed_actions:
        return "WRONG_OR_MISSING_ACTIONS"
    if breakdown.get("DB") == 1.0:
        return "COMMUNICATION_ONLY — actions correct, assertions missed"
    return "OTHER"


def _load_sims(run_dir: Path) -> dict[str, dict]:
    sims = {}
    for path in run_dir.glob("simulations/*.json"):
        try:
            sim = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            print(f"  ! unreadable {path.name}: {e}", file=sys.stderr)
            continue
        sims[sim.get("id", path.stem)] = sim
    return sims


def report(run: str, root: Path, top: int, out) -> None:
    run_dir = root / "data" / "simulations" / run
    results_path = run_dir / "results.json"
    if not results_path.exists():
        print(f"\n## {run}\n\nNo results.json at {results_path} — skipped.\n", file=out)
        return

    results = json.loads(results_path.read_text())
    index = results.get("simulation_index") or {}
    latest = _latest_trials(index)
    sims = _load_sims(run_dir)
    logs = _session_logs(run_dir)

    expected = (results.get("info") or {}).get("num_tasks")
    rewards = [e["reward"] for e in latest.values() if e.get("reward") is not None]
    mean = sum(rewards) / len(rewards) if rewards else float("nan")
    retried = sum(1 for e in latest.values() if e.get("trial", 0) > 0)

    print(f"\n## {run}\n", file=out)
    status = f"{len(latest)} tasks scored"
    if expected and len(latest) < expected:
        status += f" of {expected} — **RUN INCOMPLETE, this mean is not the score**"
    print(f"- {status}", file=out)
    print(
        f"- mean reward **{mean:.3f}**  (retries collapsed; {retried} task(s) retried)",
        file=out,
    )
    print(
        f"- rewards: {Counter(round(r, 2) for r in rewards).most_common()}",
        file=out,
    )
    print(
        f"- termination: {Counter(e.get('termination_reason') for e in latest.values()).most_common()}",
        file=out,
    )

    # Trust checks, before any per-task detail: if these are non-zero the scores
    # below are partly measuring the harness.
    total_429 = sum(_rate_limit_hits(p) for p in logs.values())
    wire = {sid: _wire_anomalies(p) for sid, p in logs.items()}
    all_events = Counter()
    for w in wire.values():
        all_events.update(w["events"])
    n_after_cancel = sum(len(w["transcript_after_cancel"]) for w in wire.values())
    n_abandoned = sum(len(w["abandoned_turns"]) for w in wire.values())
    n_slow = sum(len(w["slow_first_word"]) for w in wire.values())
    n_zero = sum(w["zero_duration_speech"] for w in wire.values())

    print("\n### Run trust\n", file=out)
    print(
        f"- caller-voice 429s: **{total_429}**"
        + (
            "  ← delayed caller speech; treat comparisons as noisy" if total_429 else ""
        ),
        file=out,
    )
    print(
        f"- wire `error` events: {all_events.get('error', 0)}   "
        f"`idle_timeout`: {all_events.get('idle_timeout', 0)}",
        file=out,
    )
    print(
        f"- sessions: {len(logs)}   `config` frames: {all_events.get('config', 0)} "
        f"(more than one per session = reconnects)",
        file=out,
    )
    # A session that produced no scored row is a task that died before grading —
    # an infrastructure error, or a run still in flight. Either way the mean
    # above is over fewer conversations than were actually held.
    unscored = len(logs) - len(latest)
    if unscored > 0:
        print(
            f"- sessions with no scored result: **{unscored}** "
            f"(died before grading, or still running)",
            file=out,
        )
    print(
        f"- `cancelled` / `reply_done`: {all_events.get('cancelled', 0)}"
        f" / {all_events.get('reply_done', 0)}",
        file=out,
    )

    print("\n### Wire anomalies (SDK-side, independent of reward)\n", file=out)
    print(
        f"- `agent_transcript` after `cancelled`: **{n_after_cancel}** "
        f"(text emitted past the turn's terminal frame)",
        file=out,
    )
    print(f"- user turns with no `reply_done`: **{n_abandoned}**", file=out)
    print(
        f"- turns >{SLOW_FIRST_WORD_SEC:.0f}s to first agent word: **{n_slow}**",
        file=out,
    )
    print(f"- zero-duration speech windows (<50ms): **{n_zero}**", file=out)

    worst_slow = sorted(
        (s for w in wire.values() for s in w["slow_first_word"]),
        key=lambda s: -s["gap_sec"],
    )[:5]
    if worst_slow:
        print(
            f"- worst first-word gaps: "
            + ", ".join(f"{s['gap_sec']}s @ {s['at']}" for s in worst_slow),
            file=out,
        )
    example = next(
        (t for w in wire.values() for t in w["transcript_after_cancel"]), None
    )
    if example:
        print(
            f"- example post-cancel transcript (+{example['delta_ms']}ms @ "
            f"{example['at']}): {example['text']!r}",
            file=out,
        )

    # Failures, worst first.
    failures = sorted(
        (e for e in latest.values() if (e.get("reward") or 0) < 1.0),
        key=lambda e: (e.get("reward") or 0, -(e.get("duration") or 0)),
    )
    print(f"\n### Failures ({len(failures)} of {len(latest)}), worst first\n", file=out)
    if not failures:
        print("None.\n", file=out)
        return

    for entry in failures[:top]:
        sim = sims.get(entry["id"], {})
        reward_info = sim.get("reward_info") or {}
        breakdown = reward_info.get("reward_breakdown") or {}
        actual = _actual_calls(sim)
        failed_actions = [
            c
            for c in (reward_info.get("action_checks") or [])
            if not c.get("action_match")
        ]
        missed = [
            a for a in (reward_info.get("nl_assertions") or []) if not a.get("met")
        ]

        print(
            f"#### task `{entry['task_id']}` — reward {entry.get('reward')} "
            f"(trial {entry.get('trial')}, {entry.get('duration', 0):.0f}s, "
            f"{entry.get('termination_reason')})\n",
            file=out,
        )
        print(
            f"- classification: **{_classify(sim, actual, failed_actions)}**", file=out
        )
        print(
            f"- breakdown: {breakdown}   db_match: "
            f"{(reward_info.get('db_check') or {}).get('db_match')}",
            file=out,
        )
        print(
            f"- agent speech the caller received: {_heard_chars(sim)} chars "
            f"across {len(sim.get('ticks') or [])} ticks",
            file=out,
        )
        print(
            f"- sim: `data/simulations/{run}/simulations/{entry['id']}.json`", file=out
        )
        log = logs.get(entry["id"])
        if log:
            print(f"- wire log: `{log.relative_to(root)}`", file=out)

        if failed_actions:
            print("\n  **Expected actions not matched:**", file=out)
            for check in failed_actions:
                action = check.get("action") or {}
                print(
                    f"  - `{action.get('name')}` expected "
                    f"`{json.dumps(action.get('arguments') or {}, sort_keys=True)}`",
                    file=out,
                )
                # The same tool as actually called: for STT-bound domains the
                # argument diff IS the finding.
                same = [c for c in actual if c["name"] == action.get("name")]
                if same:
                    for call in same[:3]:
                        print(
                            f"    agent called `{json.dumps(call['arguments'], sort_keys=True)}`",
                            file=out,
                        )
                else:
                    print("    agent never called this tool", file=out)

        if actual:
            print(
                f"\n  **Calls made** ({len(actual)}): "
                + ", ".join(f"`{c['name']}`" for c in actual[:12])
                + (" …" if len(actual) > 12 else ""),
                file=out,
            )

        if missed:
            print("\n  **Missed assertions (judge's words):**", file=out)
            for a in missed[:4]:
                print(f"  - {a.get('nl_assertion')}", file=out)
                just = (a.get("justification") or "").strip().replace("\n", " ")
                if just:
                    print(f"    → {just[:300]}", file=out)

        w = wire.get(entry["id"])
        if w:
            bits = []
            if w["abandoned_turns"]:
                bits.append(
                    f"{len(w['abandoned_turns'])} turn(s) with no reply_done "
                    f"(waited {max(t['waited_sec'] for t in w['abandoned_turns'])}s)"
                )
            if w["slow_first_word"]:
                bits.append(
                    f"{len(w['slow_first_word'])} slow first word "
                    f"(max {max(s['gap_sec'] for s in w['slow_first_word'])}s)"
                )
            if w["transcript_after_cancel"]:
                bits.append(
                    f"{len(w['transcript_after_cancel'])} post-cancel transcript(s)"
                )
            hits = _rate_limit_hits(log) if log else 0
            if hits:
                bits.append(f"{hits} caller-voice 429s")
            if bits:
                print(f"\n  **Wire:** {'; '.join(bits)}", file=out)
        print("", file=out)

    if len(failures) > top:
        print(
            f"_({len(failures) - top} further failure(s) not shown; --top to raise.)_\n",
            file=out,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="run name(s) under data/simulations/")
    parser.add_argument("--top", type=int, default=8, help="failures to detail per run")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="repo root")
    parser.add_argument(
        "--out", type=Path, help="write markdown here instead of stdout"
    )
    args = parser.parse_args()

    out = args.out.open("w") if args.out else sys.stdout
    try:
        print("# tau2 failure report", file=out)
        print(
            "\nRetries collapsed to the highest trial per task. Wire anomalies are "
            "SDK-side and independent of reward — a 0.0 from a provider that never "
            "connected is not evidence about agent quality.",
            file=out,
        )
        for run in args.runs:
            report(run, args.root, args.top, out)
    finally:
        if args.out:
            out.close()
            print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
