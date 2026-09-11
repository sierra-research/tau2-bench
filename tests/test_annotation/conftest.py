# Copyright Sierra
"""Fixtures for the annotation-factory suite.

Everything human-facing is built FROM the row models (headers, cells, filled
CSVs), so these tests cannot drift from the contract they guard. The
run/sim/check builders live in ``tests/fixtures_runs.py`` (shared with the
judges suite); this conftest pins the annotation-suite shape: a real two-sim
dir-format run (written via ``Results.save``) whose Hindi sims carry COMPLETE
stored nativeness verdicts — one severity-3 FAIL (``fail_factor``) among
severity-2 PASSes — so every judge path runs with ``reuse_existing`` and zero
LLM calls.
"""

import csv
import json
import re
from pathlib import Path
from typing import Optional

import fixtures_runs
import pytest
from fixtures_runs import make_hi_results

# Re-exported fixture: an isolated DATA_DIR with the toy language pack (used
# by the translation-review round-trip tests).
from test_multilingual.factory_testing.conftest_helpers import (  # noqa: F401
    isolated_pack_env,
)
from test_multilingual.factory_testing.results_factory import make_run_dir
from test_multilingual.factory_testing.toy_language import TOY_DOMAIN, TOY_LANGUAGE

from tau2.annotation.artifacts import CSV_ENCODING
from tau2.data_model.message import (
    AssistantMessage,
    Tick,
    ToolCall,
    ToolMessage,
    TurnTakingAction,
    UserMessage,
)
from tau2.data_model.simulation import (
    NativenessFactorCheck,
    SimulationRun,
)
from tau2.multilingual.factory.translation_rows import (
    RowState,
    RowStatus,
    TranslationAttempt,
    TranslationState,
    rows_path,
)

FAIL_FACTOR = "natural_word_choice"


def hi_factor_checks(
    fail_factor: Optional[str] = FAIL_FACTOR,
) -> list[NativenessFactorCheck]:
    """A COMPLETE set of hi judge-factor checks (so reuse-existing holds):
    every judge factor PASSes except ``fail_factor``."""
    return fixtures_runs.hi_factor_checks(fail_factor=fail_factor, pass_severity=2)


def hi_sim(
    sim_id: str,
    task_id: str,
    *,
    reward: float = 1.0,
    trial: int = 0,
    fail_factor: Optional[str] = FAIL_FACTOR,
) -> SimulationRun:
    return fixtures_runs.hi_sim(
        sim_id,
        task_id,
        checks=hi_factor_checks(fail_factor),
        judge_model="fake-judge",
        reward=reward,
        trial=trial,
    )


def make_hi_results_dir(tmp_path: Path, name: str = "hi_run") -> Path:
    """A two-sim dir-format Hindi run with complete stored judge verdicts."""
    return make_hi_results(
        tmp_path,
        [
            hi_sim("s1", "t1", reward=1.0),
            hi_sim("s2", "t2", reward=0.0, fail_factor=None),
        ],
        name=name,
    )


@pytest.fixture
def hi_results_dir(tmp_path) -> Path:
    return make_hi_results_dir(tmp_path)


# ---------------------------------------------------------------------------
# Packet fixtures: the same two-sim dir-format run, but full-duplex (ticks)
# with a fake both.wav for one sim — everything a packet build exercises.
# ---------------------------------------------------------------------------

FAKE_WAV = b"RIFF\x24\x00\x00\x00WAVEfmt fake-los-audio"

PACKET_CONFIG_RE = re.compile(r"window\.PACKET_CONFIG = (.+?);</script>")


def packet_config(html_path: Path) -> dict:
    """The parsed ``window.PACKET_CONFIG`` block of a built packet page."""
    match = PACKET_CONFIG_RE.search(html_path.read_text())
    assert match, f"no PACKET_CONFIG block in {html_path}"
    return json.loads(match.group(1).replace("<\\/", "</"))


def packet_ticks() -> list[Tick]:
    """A tiny full-duplex tick stream covering the renderer's grouping cases:
    consecutive same-pattern speech ticks (merged), a tool tick (isolated),
    and a trailing user reply."""
    ts = "2026-01-01T00:00:00"
    return [
        Tick(
            tick_id=0,
            timestamp=ts,
            agent_chunk=AssistantMessage(
                role="assistant",
                content="नमस्ते, ",
                turn_taking_action=TurnTakingAction(action="generate_message"),
            ),
        ),
        Tick(
            tick_id=1,
            timestamp=ts,
            agent_chunk=AssistantMessage(
                role="assistant",
                content="मैं कैसे <मदद> करूँ?",
                turn_taking_action=TurnTakingAction(action="keep_talking"),
            ),
        ),
        Tick(
            tick_id=2,
            timestamp=ts,
            agent_tool_calls=[
                ToolCall(
                    id="tc1", name="find_reservation", arguments={"code": "JMO1MG"}
                )
            ],
            agent_tool_results=[
                ToolMessage(id="tc1", role="tool", content='{"status": "found"}')
            ],
        ),
        Tick(
            tick_id=3,
            timestamp=ts,
            user_chunk=UserMessage(role="user", content="मेरा कोड JMO1MG है"),
        ),
    ]


def make_packet_results_dir(
    tmp_path: Path,
    name: str = "hi_voice_run",
    sim_ids: tuple[str, str] = ("s1", "s2"),
) -> Path:
    """A two-sim dir-format run with ticks; the first sim carries a fake
    both.wav (under tasks/task_<id>/sim_<id>/audio/, the inline layout)."""
    sims = [
        hi_sim(sim_ids[0], "t1", reward=1.0),
        hi_sim(sim_ids[1], "t2", reward=0.0, fail_factor=None),
    ]
    for sim in sims:
        sim.ticks = packet_ticks()
    run_dir = make_hi_results(tmp_path, sims, name=name)
    audio_dir = run_dir / "tasks" / "task_t1" / f"sim_{sim_ids[0]}" / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    (audio_dir / "both.wav").write_bytes(FAKE_WAV)
    return run_dir


@pytest.fixture
def packet_results_dir(tmp_path) -> Path:
    return make_packet_results_dir(tmp_path)


def packet_messages() -> list:
    """A tiny half-duplex transcript covering the message renderer's cases:
    a plain agent turn, a user turn, a tool-only agent turn (skipped on the
    message-only transcript), and a formatted closing turn."""
    return [
        AssistantMessage(
            role="assistant", content="Hello! How can I help you today?", turn_idx=0
        ),
        UserMessage(
            role="user", content="I want to close my <savings> account.", turn_idx=1
        ),
        AssistantMessage(
            role="assistant",
            content=None,
            turn_idx=2,
            tool_calls=[
                ToolCall(id="tc1", name="query_database", arguments={"q": "accounts"})
            ],
        ),
        ToolMessage(id="tc1", role="tool", content='{"status": "found"}', turn_idx=3),
        AssistantMessage(
            role="assistant",
            content="Here is what I need:\n- your date of birth\n- your postcode",
            turn_idx=4,
        ),
    ]


def make_text_packet_results_dir(
    tmp_path: Path,
    name: str = "en_text_run",
    sim_ids: tuple[str, str] = ("s1", "s2"),
) -> Path:
    """The text sibling of ``make_packet_results_dir``: a two-sim dir-format
    run whose sims are tick-less and half-duplex — their transcript is their
    messages — and which ships no audio at all."""
    sims = [
        hi_sim(sim_ids[0], "t1", reward=1.0),
        hi_sim(sim_ids[1], "t2", reward=0.0, fail_factor=None),
    ]
    for sim in sims:
        sim.mode = "half_duplex"
        sim.messages = packet_messages()
    return make_hi_results(tmp_path, sims, name=name)


# ---------------------------------------------------------------------------
# Translation-pipeline fixtures (toy pack; used with isolated_pack_env)
# ---------------------------------------------------------------------------


def pass_verdict() -> dict:
    return {
        "equivalent": True,
        "equivalence_issues": [],
        "natural": True,
        "register_ok": True,
        "naturalness_issues": [],
        "back_translation": "bt",
    }


def fail_verdict(issues: list[str]) -> dict:
    return {
        "equivalent": False,
        "equivalence_issues": issues,
        "natural": True,
        "register_ok": True,
        "naturalness_issues": [],
        "back_translation": "bt",
    }


def seed_pipeline_state(n_verified: int = 5, n_flagged: int = 5) -> TranslationState:
    """Persist a synthetic rows file: n_verified verified + n_flagged flagged."""
    rows: list[RowState] = []
    for i in range(1, n_verified + n_flagged + 1):
        verified = i <= n_verified
        rows.append(
            RowState(
                row_id=i,
                task_id=str(i),
                field="task_instructions",
                english=f"English source {i}.",
                status=RowStatus.VERIFIED if verified else RowStatus.FLAGGED,
                attempts=[
                    TranslationAttempt(
                        translation=f"Vertaling nummer {i} zo.",
                        verdict=pass_verdict()
                        if verified
                        else fail_verdict(["meaning drift"]),
                    )
                ],
                translator_model="model-a",
                verifier_model="model-b",
            )
        )
    state = TranslationState(
        language=TOY_LANGUAGE, domain=TOY_DOMAIN, script_code="latn", rows=rows
    )
    state.save(rows_path(TOY_LANGUAGE, TOY_DOMAIN))
    return state


def judged_run_dir(tmp_path):
    """A toy-language run carrying [llm_judge] communicate checks."""
    return make_run_dir(
        tmp_path,
        "toylang",
        rewards={"1": [1.0], "3": [0.0]},
        communicate_checks_by_task={
            "1": [
                {
                    "info": "$250 refund to the original payment method",
                    "met": True,
                    "justification": "[llm_judge] refund amount conveyed",
                }
            ],
            "3": [
                {
                    "info": "980 loyalty points",
                    "met": False,
                    "justification": "[llm_judge] balance never stated",
                }
            ],
        },
        agent_messages_by_task={
            "1": ["Je refund van $250 naar je originele betaalmethode is verwerkt."]
        },
    )


# ---------------------------------------------------------------------------
# Filled-CSV helpers (built from the models, never hand-maintained headers)
# ---------------------------------------------------------------------------


def read_family_csv(path: Path) -> tuple[list[str], list[dict]]:
    with open(path, newline="", encoding=CSV_ENCODING) as fp:
        reader = csv.DictReader(fp)
        return list(reader.fieldnames or []), list(reader)


def write_family_csv(path: Path, headers: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding=CSV_ENCODING) as fp:
        writer = csv.DictWriter(fp, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def fill_csv(path: Path, fill) -> None:
    """Rewrite a family CSV applying ``fill(index, row_dict)`` to each row."""
    headers, rows = read_family_csv(path)
    for i, row in enumerate(rows):
        fill(i, row)
    write_family_csv(path, headers, rows)
