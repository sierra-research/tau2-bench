"""Recompute tau-Elicitation behavioral claims from the compact reviewer archive.

This analysis deliberately reads only checked-in release artifacts.  It does not need
the detached tick/audio corpus or API access.  Its output is deterministic and is
included in ``reproduction/analysis_inputs/behavioral_recomputed.json``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

SYSTEMS = {
    "gpt_xhigh": "openai_xhigh",
    "gemini_high": "gemini_high",
    "grok": "xai_10",
}
CONDITIONS = ("regular", "chanheavy", "speechheavy")
FAMILIAR_BANKS = {"dates", "times"}
UNFAMILIAR_BANKS = {"coined", "medications"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _agent_path(root: Path, token: str, condition: str) -> Path:
    date = "2026-09-02"
    if token == "openai_minimal" and condition != "regular":
        date = "2026-09-05"
    return root / "transcripts" / f"main_runs__modeb_{token}_{condition}_{date}.jsonl"


def _scaffolded_path(root: Path, token: str, condition: str) -> Path:
    if condition == "regular":
        name = f"main_runs__intake_m_{token}_regular.jsonl"
    elif condition == "chanheavy":
        name = f"ablations__scaffolded__modea_{token}_chanheavy_2026-09-02.jsonl"
    else:
        name = f"main_runs__intake_m_{token}_chanlight_speechheavy.jsonl"
    return root / "transcripts" / name


def _bank_and_tier(task_id: str) -> tuple[str, str]:
    match = re.fullmatch(r"intake_(.+)_(easy|hard)_\d+", task_id)
    if match is None:
        raise ValueError(f"Not a canonical single-field intake task: {task_id}")
    return match.group(1), match.group(2)


def _fold(value: object) -> str:
    return re.sub(r"[^a-z0-9@.]+", " ", str(value).casefold()).strip()


def _gold_fields(root: Path, task_ids: set[str]) -> dict[str, dict[str, str]]:
    gold: dict[str, dict[str, str]] = {}
    for path in sorted((root / "prompts" / "task_snapshots").glob("*.json")):
        for task in json.loads(path.read_text()).get("tasks") or []:
            task_id = str(task["id"])
            if task_id not in task_ids:
                continue
            fields: dict[str, str] = {}
            criteria = task.get("evaluation_criteria") or {}
            for action in criteria.get("actions") or []:
                if action.get("name") == "submit_fields":
                    fields.update((action.get("arguments") or {}).get("fields") or {})
            if not fields:
                continue
            if task_id in gold and gold[task_id] != fields:
                raise ValueError(f"Historical task snapshots disagree for {task_id}")
            gold[task_id] = fields
    if len(gold) != 200:
        raise ValueError(f"Expected 200 canonical gold tasks, found {len(gold)}")
    return gold


def _effort(row: dict[str, Any]) -> tuple[int, int, int]:
    spell = 0
    readback = 0
    for turn in row.get("turns") or []:
        if turn.get("source") != "user_tool_event":
            continue
        spell += turn.get("tool_name") == "note_spell_request"
        readback += turn.get("tool_name") == "note_readback"
    return spell + readback, spell, readback


def _field_rows(
    calls: Iterable[dict[str, Any]], gold: dict[str, dict[str, str]]
) -> list[dict[str, bool]]:
    output: list[dict[str, bool]] = []
    for call in calls:
        first: dict[str, str] = {}
        submitted: dict[str, str] = {}
        spelled: set[str] = set()
        read_back: set[str] = set()
        for turn in call.get("turns") or []:
            args = turn.get("tool_arguments") or {}
            if turn.get("source") == "agent_tool_event":
                if turn.get("tool_name") == "log_capture":
                    field = args.get("field_name")
                    if field and field not in first:
                        first[field] = str(args.get("value") or "")
                elif turn.get("tool_name") == "submit_fields":
                    submitted.update(args.get("fields") or {})
            elif turn.get("source") == "user_tool_event":
                field = args.get("field")
                if field and turn.get("tool_name") == "note_spell_request":
                    spelled.add(field)
                elif field and turn.get("tool_name") == "note_readback":
                    read_back.add(field)
        evaluation_fields = call.get("evaluation_fields")
        if evaluation_fields is None:
            evaluation_fields = gold[call["task_id"]]
        for field, expected in evaluation_fields.items():
            if _fold(expected) != _fold(gold[call["task_id"]][field]):
                raise ValueError(
                    f"Evaluator and task snapshot disagree for {call['task_id']}:{field}"
                )
            attempt = first.get(field)
            final = submitted.get(field)
            output.append(
                {
                    "attempt_seen": attempt is not None,
                    "attempt_correct": attempt is not None
                    and _fold(attempt) == _fold(expected),
                    "final_correct": final is not None
                    and _fold(final) == _fold(expected),
                    "spelled": field in spelled,
                    "read_back": field in read_back,
                }
            )
    return output


def _fraction(numerator: int, denominator: int) -> dict[str, float | int]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": numerator / denominator,
    }


def _behavior(calls: list[dict[str, Any]], gold: dict[str, dict[str, str]]) -> dict:
    fields = _field_rows(calls, gold)
    wrong = [
        row for row in fields if row["attempt_seen"] and not row["attempt_correct"]
    ]
    right = [row for row in fields if row["attempt_correct"]]
    verified_wrong = [row for row in wrong if row["spelled"] or row["read_back"]]
    verified_right = [row for row in right if row["spelled"] or row["read_back"]]
    return {
        "gold_fields": len(fields),
        "attempt_wrong": len(wrong),
        "attempt_right": len(right),
        "attempt_unseen": sum(not row["attempt_seen"] for row in fields),
        "verified_given_wrong": _fraction(len(verified_wrong), len(wrong)),
        "verified_given_right": _fraction(len(verified_right), len(right)),
        "verified_wrong_repaired": _fraction(
            sum(row["final_correct"] for row in verified_wrong), len(verified_wrong)
        ),
    }


def _pooled_repair(
    regular_calls: dict[str, list[dict[str, Any]]], gold: dict[str, dict[str, str]]
) -> dict[str, dict[str, float | int]]:
    fields = [
        row
        for calls in regular_calls.values()
        for row in _field_rows(calls, gold)
        if row["attempt_seen"] and not row["attempt_correct"]
    ]
    output = {}
    for label, spelled, read_back in (
        ("neither", False, False),
        ("readback_only", False, True),
        ("spelling_only", True, False),
        ("both", True, True),
    ):
        selected = [
            row
            for row in fields
            if row["spelled"] is spelled and row["read_back"] is read_back
        ]
        output[label] = _fraction(
            sum(row["final_correct"] for row in selected), len(selected)
        )
    return output


def _group_mean(calls: list[dict[str, Any]], selector) -> float:
    selected = [row for row in calls if selector(row)]
    return mean(_effort(row)[0] for row in selected)


def _adaptivity(
    by_system_condition: dict[tuple[str, str], list[dict[str, Any]]],
) -> dict[str, Any]:
    output = {}
    for system in SYSTEMS:
        regular = by_system_condition[system, "regular"]
        noisy = by_system_condition[system, "chanheavy"]
        voices: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in regular:
            voices[row["speech_environment"]["persona_name"]].append(row)
        ranked = sorted(
            voices,
            key=lambda voice: mean(row["reward"] for row in voices[voice]),
            reverse=True,
        )
        output[system] = {
            "noise": {
                "baseline": mean(_effort(row)[0] for row in regular),
                "challenge": mean(_effort(row)[0] for row in noisy),
            },
            "difficulty": {
                "baseline": _group_mean(
                    regular, lambda row: _bank_and_tier(row["task_id"])[1] == "easy"
                ),
                "challenge": _group_mean(
                    regular, lambda row: _bank_and_tier(row["task_id"])[1] == "hard"
                ),
            },
            "entity_familiarity": {
                "baseline": _group_mean(
                    regular,
                    lambda row: _bank_and_tier(row["task_id"])[0] in FAMILIAR_BANKS,
                ),
                "challenge": _group_mean(
                    regular,
                    lambda row: _bank_and_tier(row["task_id"])[0] in UNFAMILIAR_BANKS,
                ),
            },
            "caller_voice": {
                "best_voice": ranked[0],
                "worst_voice": ranked[-1],
                "baseline": mean(_effort(row)[0] for row in voices[ranked[0]]),
                "challenge": mean(_effort(row)[0] for row in voices[ranked[-1]]),
            },
        }
    return output


def _entity_results(
    by_system_condition: dict[tuple[str, str], list[dict[str, Any]]],
) -> dict[str, Any]:
    outcomes: dict[tuple[str, str], dict[str, float]] = {}
    regular: dict[str, list[float]] = defaultdict(list)
    for system in SYSTEMS:
        for condition in CONDITIONS:
            for row in by_system_condition[system, condition]:
                outcomes.setdefault((system, row["task_id"]), {})[condition] = float(
                    row["reward"]
                )
                if condition == "regular":
                    regular[_bank_and_tier(row["task_id"])[0]].append(row["reward"])
    robust: dict[str, list[float]] = defaultdict(list)
    for (_, task_id), values in outcomes.items():
        if set(values) != set(CONDITIONS):
            raise ValueError(f"Incomplete realization triple for {task_id}")
        robust[_bank_and_tier(task_id)[0]].append(min(values.values()))
    return {
        bank: {
            "regular_observations": len(regular[bank]),
            "pass_at_1": mean(regular[bank]),
            "robust_triples": len(robust[bank]),
            "pass_robust_3": mean(robust[bank]),
        }
        for bank in sorted(regular)
    }


def _strategy_comparison(
    agent_regular: dict[str, list[dict[str, Any]]],
    scaffolded_regular: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    output = {}
    for system in SYSTEMS:
        output[system] = {}
        for name, calls in (
            ("agent_directed", agent_regular[system]),
            ("scaffolded", scaffolded_regular[system]),
        ):
            output[system][name] = {
                "pass_at_1": mean(row["reward"] for row in calls),
                "easy_pass_at_1": mean(
                    row["reward"]
                    for row in calls
                    if _bank_and_tier(row["task_id"])[1] == "easy"
                ),
                "hard_pass_at_1": mean(
                    row["reward"]
                    for row in calls
                    if _bank_and_tier(row["task_id"])[1] == "hard"
                ),
                "simulated_duration_seconds": mean(
                    row["tick_count"] * 0.2 for row in calls
                ),
            }
    return output


def _voice_results(regular_calls: dict[str, list[dict[str, Any]]]) -> dict:
    output = {}
    for system, calls in regular_calls.items():
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in calls:
            grouped[row["speech_environment"]["persona_name"]].append(row)
        output[system] = {
            voice: _fraction(sum(row["reward"] == 1.0 for row in rows), len(rows))
            for voice, rows in sorted(grouped.items())
        }
    return output


def _realism_descriptive(
    by_system_condition: dict[tuple[str, str], list[dict[str, Any]]],
) -> dict:
    calls = [
        row
        for key, rows in by_system_condition.items()
        for row in rows
        if key[0] in SYSTEMS
    ]
    clean = [row for row in calls if row.get("complication") is None]
    assigned = [row for row in calls if row.get("complication") is not None]
    return {
        "clean": _fraction(sum(row["reward"] == 1.0 for row in clean), len(clean)),
        "assigned": _fraction(
            sum(row["reward"] == 1.0 for row in assigned), len(assigned)
        ),
        "note": "Descriptive only; causal Hajek estimates use eligibility weights.",
    }


def analyze(root: Path) -> dict[str, Any]:
    by_system_condition = {
        (system, condition): _read_jsonl(_agent_path(root, token, condition))
        for system, token in SYSTEMS.items()
        for condition in CONDITIONS
    }
    gold = _gold_fields(
        root,
        {
            row["task_id"]
            for (system, _), rows in by_system_condition.items()
            if system in SYSTEMS
            for row in rows
        },
    )
    scaffolded_regular = {
        system: _read_jsonl(_scaffolded_path(root, token, "regular"))
        for system, token in SYSTEMS.items()
    }
    regular = {system: by_system_condition[system, "regular"] for system in SYSTEMS}
    return {
        "schema_version": "tau-elicit-behavioral-recomputation-v1",
        "source": "checked-in compact transcripts and historical task snapshots",
        "behavior": {
            system: _behavior(calls, gold) for system, calls in regular.items()
        },
        "pooled_initially_wrong_recovery": _pooled_repair(regular, gold),
        "adaptivity_effort": _adaptivity(by_system_condition),
        "strategy_comparison": _strategy_comparison(regular, scaffolded_regular),
        "entity_results": _entity_results(by_system_condition),
        "agent_directed_voice_results": _voice_results(regular),
        "realism_assignment_descriptive": _realism_descriptive(by_system_condition),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--release-root",
        type=Path,
        default=Path("papers/tau-intake/v1/reproduction"),
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Compare against analysis_inputs/behavioral_recomputed.json.",
    )
    args = parser.parse_args()
    document = analyze(args.release_root.resolve())
    rendered = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if args.check:
        expected_path = (
            args.release_root.resolve()
            / "analysis_inputs"
            / "behavioral_recomputed.json"
        )
        if json.loads(expected_path.read_text()) != document:
            print(f"FAIL: recomputation differs from {expected_path}")
            sys.exit(1)
        print(f"PASS: behavioral claims reproduce from {args.release_root}")
    elif args.out is None:
        print(rendered, end="")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered)


if __name__ == "__main__":
    main()
