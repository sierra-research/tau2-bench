#!/usr/bin/env python3
"""Diff what the caller SAID against what the agent HEARD, per utterance.

    uv run python scripts/stt_errors.py <run> [<run> ...] [--all] [--out FILE]

tau2 synthesizes the caller's speech from text, so ground truth exists: every
`user_chunk` carries an `audio_script_gold` naming the exact utterance that was
spoken. The agent's side of it is the `user_transcript` wire events in the
session log. Pairing them turns "the agent did the wrong thing" into "the agent
was told the wrong thing", which are different bugs with different owners.

Defaults to FAILING tasks only (`--all` for every task), since a mis-hearing
that changed nothing is not what you are looking for.

## Why the comparison needs normalization

A spoken ZIP is gold `"one, nine, one, two, two"` and heard `"19122"` — the
same utterance, correctly transcribed. A spelled name is gold `"Y, U, S, U, F"`
and heard `"Y-U-S-U-F"`. Diffing raw strings reports both as errors, which
buries the real ones: on the run this was built against, unnormalized diffing
flagged nearly every authentication utterance. So digit words are folded to
digits, adjacent digits are joined, punctuation goes, and case goes — and only
then is what remains a disagreement.

## Why alignment is not positional

STT decides utterance boundaries, the simulator decides utterances, and they
disagree: a pause or a [sneeze] splits one utterance into two transcripts, and
back-to-back short ones merge into one. Pairing by index silently shifts every
later row after the first split, so every subsequent utterance reads as an
error. This aligns greedily over 1:1, 1:2 (split) and 2:1 (merge), picking
whichever scores best — and reports the cardinality, because a split IS a
finding: the agent answered half a sentence.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path

GOLD_CHUNK_RE = re.compile(r"<chunk id=\d+>(.*?)</chunk>", re.S)
TRANSCRIPT_RE = re.compile(
    r"AAI event: user_transcript - \{'type': 'user_transcript', 'text': "
    r"(\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*')"
)
# Speech effects the harness injects into the audio; they are not words the
# caller said, so they must not count as text STT failed to hear.
EFFECT_RE = re.compile(r"\[[a-z_]+\]")

NUMBER_WORDS = {
    "zero": "0",
    "oh": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
}

# Utterances carrying these are the ones whose mis-hearing reaches a tool
# argument, which is where an STT error becomes a score error.
IDENTITY_HINT_RE = re.compile(r"[@#]|\d|\b(?:zip|email|order|dot com|at)\b", re.I)
NEGATION_RE = re.compile(
    r"\b(?:not|no|don'?t|doesn'?t|didn'?t|can'?t|won'?t|never)\b", re.I
)
# Any letter outside Latin-1: the transcript came back in a different script.
NON_LATIN_RE = re.compile(r"[^\x00-\xff]")

# Above this, the pair differs only in inflection or a filler word — real, but
# not what anyone is hunting. Kept out of OTHER so that bucket stays meaningful.
MINOR_SIMILARITY = 0.8

# A pair at or above this normalized similarity is treated as heard correctly.
SAME_ENOUGH = 0.95


def _normalize(text: str) -> list[str]:
    """Tokens for comparison: spoken digits folded, punctuation and case gone."""
    text = EFFECT_RE.sub(" ", text)
    text = text.replace("’", "'").lower()
    # Hyphens are DELETED, not spaced: the gold script and the transcript
    # disagree freely on them ("tshirt" / "t-shirt", "v-neck" / "v neck"), and
    # spacing them splits one word into two tokens so a perfectly transcribed
    # sentence scores as a mismatch. Deleting also folds a spelled-out
    # "Y-U-S-U-F" straight into "yusuf".
    text = re.sub(r"[-–—]+", "", text)
    text = re.sub(r"[^a-z0-9']+", " ", text)
    tokens = [NUMBER_WORDS.get(t, t) for t in text.split() if t]
    # Join digit runs, so a spoken "1 9 1 2 2" and a transcribed "19122" are one
    # token. Spelled letters fold the same way ("y u s u f" / "yusuf") — but only
    # into a RUN of single letters, tracked explicitly. Folding a lone letter into
    # whatever preceded it merges "many" + "t" from "many t-shirts" into "manyt",
    # inventing a mismatch in a sentence that was transcribed perfectly.
    out: list[str] = []
    letter_run = False
    for token in tokens:
        single_letter = len(token) == 1 and token.isalpha()
        if token.isdigit() and out and out[-1].isdigit():
            out[-1] += token
        elif single_letter and letter_run and out:
            out[-1] += token
        else:
            out.append(token)
            letter_run = single_letter
    return out


def _similarity(gold: str, heard: str) -> float:
    a, b = _normalize(gold), _normalize(heard)
    if not a and not b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _gold_utterances(sim: dict) -> list[str]:
    """The exact text synthesized for the caller, in order, one per utterance."""
    seen: dict[str, str] = {}
    order: list[str] = []
    for tick in sim.get("ticks") or []:
        chunk = tick.get("user_chunk") or {}
        gold = chunk.get("audio_script_gold")
        ids = chunk.get("utterance_ids") or []
        if not gold or not ids:
            continue
        uid = ids[0]
        if uid in seen:
            continue
        pieces = GOLD_CHUNK_RE.findall(gold)
        text = "".join(pieces).strip()
        if not text:
            continue
        seen[uid] = text
        order.append(uid)
    return [seen[u] for u in order]


def _heard_transcripts(log: Path) -> list[str]:
    out = []
    with log.open(errors="replace") as fh:
        for line in fh:
            m = TRANSCRIPT_RE.search(line)
            if m:
                out.append(m.group(1)[1:-1])
    return out


def _align(gold: list[str], heard: list[str]) -> list[tuple[list[str], list[str]]]:
    """Greedy alignment over 1:1, 1:2 (split) and 2:1 (merge) pairings."""
    pairs: list[tuple[list[str], list[str]]] = []
    i = j = 0
    while i < len(gold) and j < len(heard):
        one_one = _similarity(gold[i], heard[j])
        split = (
            _similarity(gold[i], heard[j] + " " + heard[j + 1])
            if j + 1 < len(heard)
            else -1.0
        )
        merge = (
            _similarity(gold[i] + " " + gold[i + 1], heard[j])
            if i + 1 < len(gold)
            else -1.0
        )
        best = max(one_one, split, merge)
        # Require a real margin before claiming a split/merge: near-ties are
        # 1:1, or a long correct transcript absorbs its innocent neighbour.
        if best == split and split > one_one + 0.05:
            pairs.append(([gold[i]], [heard[j], heard[j + 1]]))
            i, j = i + 1, j + 2
        elif best == merge and merge > one_one + 0.05:
            pairs.append(([gold[i], gold[i + 1]], [heard[j]]))
            i, j = i + 2, j + 1
        else:
            pairs.append(([gold[i]], [heard[j]]))
            i, j = i + 1, j + 1
    for k in range(i, len(gold)):
        pairs.append(([gold[k]], []))  # said, never transcribed
    for k in range(j, len(heard)):
        pairs.append(([], [heard[k]]))  # heard, never said
    return pairs


def _classify(gold: str, heard: str, cardinality: tuple[int, int]) -> str:
    """Name the error class, because they need different remedies.

    Retrying only rescues a homophone (one candidate may be right); a digit
    substitution is an independent coin flip per attempt, and a dropped negation
    or a collapsed structure is not recoverable agent-side at all.
    """
    if not heard:
        return "NOT_HEARD — utterance never transcribed"
    if not gold:
        return "PHANTOM — transcript with no matching utterance"
    # Checked before everything else because it subsumes the rest: an English
    # utterance returned in Devanagari or Hebrew script is a language-detection
    # failure, and every downstream measure (truncated, substituted) is just a
    # side effect of comparing across alphabets. Filed as TRUNCATED it looks
    # like an audio problem.
    if NON_LATIN_RE.search(heard):
        return (
            "NON_LATIN_SCRIPT — English returned in another script (language detection)"
        )
    if cardinality[0] == 1 and cardinality[1] > 1:
        return "SPLIT — one utterance became several turns"
    if cardinality[0] > 1 and cardinality[1] == 1:
        return "MERGED — several utterances became one turn"

    g_tokens, h_tokens = _normalize(gold), _normalize(heard)
    g_set, h_set = set(g_tokens), set(h_tokens)
    lost, gained = g_set - h_set, h_set - g_set

    if bool(NEGATION_RE.search(gold)) != bool(NEGATION_RE.search(heard)):
        return "NEGATION — polarity changed; meaning inverted"
    if re.search(r"@|\bat\b.*\bdot\b", gold, re.I) and "@" not in heard:
        return "EMAIL — address structure lost"
    if any(t.isdigit() for t in lost | gained):
        return "DIGITS — numeric substitution (zip / order id / phone)"
    if lost and all(len(t) > 2 for t in lost) and len(lost) == len(gained):
        return "SUBSTITUTION — words swapped (homophone or proper noun)"
    if len(h_tokens) < len(g_tokens) * 0.6:
        return "TRUNCATED — most of the utterance missing"
    if difflib.SequenceMatcher(None, g_tokens, h_tokens).ratio() >= MINOR_SIMILARITY:
        return "MINOR — inflection or filler only"
    return "OTHER"


def _word_diff(gold: str, heard: str) -> str:
    """Compact token diff, on the NORMALIZED forms actually compared."""
    a, b = _normalize(gold), _normalize(heard)
    bits = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b).get_opcodes():
        if op == "equal":
            continue
        said = " ".join(a[i1:i2]) or "∅"
        got = " ".join(b[j1:j2]) or "∅"
        bits.append(f"{said!r}→{got!r}")
    return "; ".join(bits) if bits else "(normalized forms match)"


def _latest_trials(index) -> dict[str, dict]:
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


def report(run: str, root: Path, include_all: bool, out) -> None:
    run_dir = root / "data" / "simulations" / run
    results_path = run_dir / "results.json"
    if not results_path.exists():
        print(f"\n## {run}\n\nNo results.json — skipped.\n", file=out)
        return

    latest = _latest_trials(
        json.loads(results_path.read_text()).get("simulation_index")
    )
    logs = {
        p.parent.name.removeprefix("sim_"): p
        for p in run_dir.glob("artifacts/task_*/sim_*/task.log")
    }

    print(f"\n## {run}\n", file=out)
    totals: dict[str, int] = {}
    utterances = errors = 0
    identity_errors: list[str] = []
    shown = 0

    for entry in sorted(latest.values(), key=lambda e: (e.get("reward") or 0)):
        reward = entry.get("reward") or 0
        if reward >= 1.0 and not include_all:
            continue
        sim_path = run_dir / "simulations" / f"{entry['id']}.json"
        log = logs.get(entry["id"])
        if not sim_path.exists() or not log:
            continue
        sim = json.loads(sim_path.read_text())
        pairs = _align(_gold_utterances(sim), _heard_transcripts(log))

        rows = []
        for gold_group, heard_group in pairs:
            gold = " ".join(gold_group)
            heard = " ".join(heard_group)
            cardinality = (len(gold_group), len(heard_group))
            utterances += 1
            score = _similarity(gold, heard) if gold and heard else 0.0
            if cardinality == (1, 1) and score >= SAME_ENOUGH:
                continue
            kind = _classify(gold, heard, cardinality)
            errors += 1
            totals[kind] = totals.get(kind, 0) + 1
            identity = bool(IDENTITY_HINT_RE.search(gold))
            if identity:
                identity_errors.append(f"{run} task {entry['task_id']}: {kind}")
            rows.append((kind, gold, heard, score, identity))

        if not rows:
            continue
        shown += 1
        print(f"### task `{entry['task_id']}` — reward {reward}\n", file=out)
        for kind, gold, heard, score, identity in rows:
            flag = "  ⚠️ **feeds a tool argument**" if identity else ""
            print(f"- **{kind}**{flag}", file=out)
            print(f"  - said:  `{gold}`", file=out)
            print(f"  - heard: `{heard or '(nothing)'}`", file=out)
            print(
                f"  - diff:  {_word_diff(gold, heard)}  (similarity {score:.2f})",
                file=out,
            )
        print("", file=out)

    scope = "all tasks" if include_all else "failing tasks"
    print(f"### Summary ({scope})\n", file=out)
    print(
        f"- utterances compared: **{utterances}**, mis-heard: **{errors}**"
        + (f"  ({100 * errors / utterances:.0f}%)" if utterances else ""),
        file=out,
    )
    for kind, n in sorted(totals.items(), key=lambda kv: -kv[1]):
        print(f"  - {n}x {kind}", file=out)
    print(
        f"- mis-hearings on utterances that feed a tool argument: "
        f"**{len(identity_errors)}** — these are the ones that move the score",
        file=out,
    )
    if not shown:
        print("\n_No mis-hearings found in scope._", file=out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+")
    parser.add_argument(
        "--all", action="store_true", help="include passing tasks, not just failures"
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    out = args.out.open("w") if args.out else sys.stdout
    try:
        print("# STT error report", file=out)
        print(
            "\nGold text is what the harness SYNTHESIZED (`audio_script_gold`); "
            "heard text is the agent's `user_transcript`. Comparison is on "
            "normalized tokens, so a ZIP spoken as words and transcribed as "
            "digits is not an error.",
            file=out,
        )
        for run in args.runs:
            report(run, args.root, args.all, out)
    finally:
        if args.out:
            out.close()
            print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
