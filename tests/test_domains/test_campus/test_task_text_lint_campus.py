"""Task-text forbidden-token lint (task-data wording surface).

tasks.json is model-visible prompt data just like tool docstrings:
``description.purpose`` / ``description.relevant_policies`` and the
``user_scenario.instructions`` blocks are handed to the agent and the user
simulator verbatim. The statement-rewrite batch (W17) removed the
evaluation vocabulary listed below from those leaves; this lint pins that
surface so the words cannot creep back.

The detector is shared with the tool-description lint: ``find_forbidden``
lives in test_tool_desc_lint_campus.py and takes a pattern set, so there is
exactly ONE mechanism. The two pattern sets (tool-description vs task-text)
are distinct surfaces and cross-reference each other by comment — no
duplicated constants.

Direction note: test_task_lint_campus.REFUSAL_MARKERS is a "must appear"
list (zero-shape tasks must carry refusal semantics); the set below is a
"must never appear" list. Opposite directions — never merge them.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from tau2.domains.campus.utils import CAMPUS_TASK_SET_PATH

# load the sibling lint module by path: tests/ has no top-level __init__.py,
# so cross-module reuse goes through importlib (same approach as the W16
# guard batch), not a package import.
_sibling_spec = importlib.util.spec_from_file_location(
    "test_tool_desc_lint_campus",
    Path(__file__).with_name("test_tool_desc_lint_campus.py"),
)
_sibling = importlib.util.module_from_spec(_sibling_spec)
sys.modules[_sibling_spec.name] = _sibling
_sibling_spec.loader.exec_module(_sibling)

find_forbidden = _sibling.find_forbidden

# (pattern, why it is forbidden) — the task-text surface set. Distinct from
# test_tool_desc_lint_campus.FORBIDDEN_PATTERNS (agent tool descriptions);
# both are fed to the same find_forbidden detector.
TASK_TEXT_FORBIDDEN_PATTERNS = (
    (r"判分", "evaluation phrasing (grading)"),
    (r"终态校验将不通过", "evaluation phrasing (final-state assertion)"),
    (r"终态不达标", "evaluation phrasing (final-state assertion)"),
    (r"闸门", "evaluation phrasing (gate)"),
    (r"命中", "evaluation phrasing (hit)"),
    (r"预检", "evaluation phrasing (pre-check)"),
    (r"拦截", "evaluation phrasing (interception)"),
    (r"断言", "evaluation phrasing (assertion)"),
)

# tasks rewritten by the W17 batch — pinned so the lint surface provably
# covers exactly the leaves the rewrite touched.
REWRITTEN_TASK_IDS = (
    "E02", "E04", "E08", "E14", "M05", "M08", "M20", "H02", "H04", "H10", "H11",
)

# pre-rewrite leaf texts kept as counterexamples (must stay flagged). The
# H04 purpose carried 判分 twice; the E08 fragment carried 预检.
_PRE_REWRITE_H04_PURPOSE = (
    "窗口＋名额波动＋终态判分：中途 drop 不算错，判分只看 EN 终态=已选"
    "（单终态）；两次 enroll 用新行、drop 的行留'已退课'。"
)
_PRE_REWRITE_E08_POLICY = (
    "政策第16条（5 个工作日窗口）、第34条（申诉先行学院）；"
    "create_ticket 预检：成绩类申诉须在查分窗口内提交，逾期即不受理（咨询类不受拦）。"
)


def iter_string_leaves(obj, path=()):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from iter_string_leaves(v, path + (str(k),))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from iter_string_leaves(v, path + (str(i),))
    elif isinstance(obj, str):
        yield path, obj


@pytest.fixture(scope="module")
def task_text_leaves() -> list[tuple[tuple[str, ...], str]]:
    """Every string leaf of tasks.json with its field path."""
    data = json.loads(CAMPUS_TASK_SET_PATH.read_text(encoding="utf-8"))
    return list(iter_string_leaves(data))


@pytest.fixture(scope="module")
def task_ids() -> list[str]:
    data = json.loads(CAMPUS_TASK_SET_PATH.read_text(encoding="utf-8"))
    return [task["id"] for task in data]


def test_task_text_leaves_free_of_forbidden_tokens(task_text_leaves):
    """Every string leaf of tasks.json carries zero task-text forbidden hits."""
    offenders = {
        ".".join(path): find_forbidden(text, TASK_TEXT_FORBIDDEN_PATTERNS)
        for path, text in task_text_leaves
        if find_forbidden(text, TASK_TEXT_FORBIDDEN_PATTERNS)
    }
    assert offenders == {}


def test_task_text_lint_surface_pinned(task_text_leaves, task_ids):
    """Non-vacuous coverage pin: the lint scans all 50 tasks / 1667 string
    leaves, including every leaf the W17 rewrite touched. Bump deliberately
    if the document itself grows."""
    assert len(task_ids) == 50
    assert len(task_text_leaves) == 1667
    assert set(REWRITTEN_TASK_IDS) <= set(task_ids)


def test_detector_flags_pre_rewrite_counterexamples():
    """Counterexample red-proof: the pre-rewrite wording must be flagged by
    the very detector the lint uses."""
    h04_hits = find_forbidden(_PRE_REWRITE_H04_PURPOSE, TASK_TEXT_FORBIDDEN_PATTERNS)
    assert len(h04_hits) == 2, h04_hits
    assert all(pattern == r"判分" for pattern, _ in h04_hits)

    e08_hits = find_forbidden(_PRE_REWRITE_E08_POLICY, TASK_TEXT_FORBIDDEN_PATTERNS)
    assert [pattern for pattern, _ in e08_hits] == [r"预检"]


def test_post_rewrite_leaves_no_longer_flagged(task_ids):
    """Green-on-new: the rewritten leaves are clean under the same detector."""
    data = json.loads(CAMPUS_TASK_SET_PATH.read_text(encoding="utf-8"))
    by_id = {task["id"]: task for task in data}
    for tid in ("H04", "E02", "H10"):
        task = by_id[tid]
        texts = [
            task["description"]["purpose"],
            task["description"].get("relevant_policies") or "",
        ]
        for text in texts:
            assert find_forbidden(text, TASK_TEXT_FORBIDDEN_PATTERNS) == []
