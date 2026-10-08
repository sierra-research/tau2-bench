"""Task self-consistency lint.

Three machine checks that task TEXT stays consistent with each task's
evaluation SHAPE and TOOL FACE — the M06 class of defect, where a purpose
described a submission flow the data could not support:

1. **tool mentions**: every campus tool name appearing in ``purpose`` or
   ``task_instructions`` must belong to the task's tool face (the agent
   tool face ∪ that task's ``user_tools``);
2. **zero-shape refusal semantics**: a task whose golden criteria have
   ``actions=[]`` AND ``env_assertions=[]`` (no graded action, no env
   assertion — everything it grades lives in what the agent must NOT do)
   must state refusal / zero-write semantics explicitly in ``purpose``
   (marker set: ``拒绝|零写|无需|不受理|不产生任何数据写入``);
3. **golden user-side actions**: every golden action with
   ``requestor="user"`` must name a tool inside that task's
   ``user_tools``.

The pre-cleanup M06 purpose is embedded as a counterexample and asserted
to FAIL check 2 — proving the lint would have caught it on the old data.
"""

import copy
import json

import pytest

from tau2.domains.campus.environment import get_environment, get_tasks
from tau2.domains.campus.utils import CAMPUS_TASK_SET_PATH

# refusal / zero-write semantics the purpose of a zero-shape task must carry.
# "不产生任何数据写入" is the plain-language phrasing the corpus uses since the
# statement rewrite (it replaced the "零写" shorthand); the old markers stay so
# historical wording keeps counting as refusal semantics too.
REFUSAL_MARKERS = ("拒绝", "零写", "无需", "不受理", "不产生任何数据写入")

# pre-cleanup M06 purpose: describes a submit→upload flow, ends in grading
# shorthand, carries NONE of the REFUSAL_MARKERS → must fail check 2.
_PRE_CLEAN_M06_PURPOSE = (
    "考 因病缓考无效材料半闭环：社区诊所证明直接驳回（第12条），"
    "Agent 不得替学生改医院等级；金标=提交+学生如实上传（其他机构）→驳回。"
)


@pytest.fixture(scope="module")
def tasks() -> list[dict]:
    return json.loads(CAMPUS_TASK_SET_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def tool_faces() -> tuple[frozenset[str], frozenset[str]]:
    """(agent face, user face) as registered in the runtime environment."""
    env = get_environment()
    return (
        frozenset(t.name for t in env.get_tools()),
        frozenset(t.name for t in env.get_user_tools()),
    )


def _has_refusal_marker(purpose: str) -> bool:
    return any(marker in purpose for marker in REFUSAL_MARKERS)


def test_lint_tool_mentions_belong_to_task_tool_face(tasks, tool_faces):
    """Check ①: a tool name in purpose/task_instructions must be either an
    agent-face tool or one of this task's user tools."""
    agent_face, user_face = tool_faces
    offenders = []
    for task in tasks:
        allowed = agent_face | frozenset(task.get("user_tools") or [])
        texts = {
            "purpose": task["description"].get("purpose") or "",
            "task_instructions": (
                (task.get("user_scenario") or {}).get("instructions") or {}
            ).get("task_instructions")
            or "",
        }
        for field, text in texts.items():
            for name in sorted(agent_face | user_face):
                if name in text and name not in allowed:
                    offenders.append(f"{task['id']}.{field}: {name}")
    assert not offenders, f"tool mentions outside the task's tool face: {offenders}"


def test_lint_zero_shape_tasks_state_refusal_semantics(tasks):
    """Check ②: actions=[] AND env_assertions=[] tasks must spell out
    refusal / zero-write semantics in purpose (the M06 class of defect)."""
    zero_shape = [
        task
        for task in tasks
        if (task["evaluation_criteria"].get("actions") or []) == []
        and task["evaluation_criteria"].get("env_assertions") == []
    ]
    offenders = [
        task["id"]
        for task in zero_shape
        if not _has_refusal_marker(task["description"].get("purpose") or "")
    ]
    assert not offenders, (
        "zero-action/zero-assertion tasks without refusal or zero-write "
        f"semantics in purpose: {offenders}"
    )


def test_lint_golden_user_actions_within_user_tools(tasks):
    """Check ③: golden user-side actions must name a tool in user_tools."""
    offenders = []
    for task in tasks:
        user_tools = frozenset(task.get("user_tools") or [])
        for action in task["evaluation_criteria"].get("actions") or []:
            if action.get("requestor") == "user" and action["name"] not in user_tools:
                offenders.append(f"{task['id']}: {action['name']}")
    assert not offenders, f"golden user actions outside user_tools: {offenders}"


def test_lint_counterexample_pre_cleanup_m06_purpose_fails_refusal_check(tasks):
    """Counterexample: feeding the pre-cleanup M06 purpose into check ② must
    turn it RED, while live data stays green — red-on-old / green-on-new on
    the very task the lint targets."""
    assert not _has_refusal_marker(_PRE_CLEAN_M06_PURPOSE), (
        "pre-cleanup M06 purpose unexpectedly carries a refusal marker — "
        "counterexample no longer demonstrates the defect"
    )
    doctored = copy.deepcopy(tasks)
    next(t for t in doctored if t["id"] == "M06")["description"][
        "purpose"
    ] = _PRE_CLEAN_M06_PURPOSE
    with pytest.raises(AssertionError, match="M06"):
        test_lint_zero_shape_tasks_state_refusal_semantics(doctored)

    m06 = next(task for task in tasks if task["id"] == "M06")
    assert _has_refusal_marker(m06["description"]["purpose"]), (
        "live M06 purpose lost its refusal/zero-write semantics"
    )
