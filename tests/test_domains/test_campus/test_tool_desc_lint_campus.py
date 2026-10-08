"""Tool-description forbidden-token lint (agent-visible prompt surface).

Tool docstrings are part of the model-visible prompt: whatever the
docstring carries lands verbatim in ``openai_schema["function"]
["description"]`` (short description + blank line + long description,
see ``environment/tool.py::_get_description``). This lint pins the
wording rule from the tool-description cleanup: internal version tags,
hyphenated error codes, pitfall codes, internal decision / review-record
pointers and evaluation-design phrasings must never appear there.

Policy clause numbers (第 N 条) and document/field codes (CE/DF/APP/
SIG/GR-xxx/abandon_count) are domain vocabulary and stay allowed.

The counterexample test feeds pre-cleanup wording through the SAME
detector, proving the lint has teeth (it would go red on the old text).
"""

import re

import pytest

from tau2.domains.campus.environment import get_environment

# (pattern, why it is forbidden) — order mirrors the wording-rule list.
# This is the TOOL-DESCRIPTION surface only. The task-text surface has its own
# set (TASK_TEXT_FORBIDDEN_PATTERNS in test_task_text_lint_campus.py); the two
# sets are distinct on purpose and cross-reference each other by comment —
# find_forbidden below is the single shared detector for both surfaces.
FORBIDDEN_PATTERNS = (
    (r"v\d+\.\d+-[A-Z]", "internal version tag (v1.1-Fx)"),
    (r"E-[A-Z]{2,}", "hyphenated error code (E-LEVEL)"),
    (r"P\d{2}", "pitfall code (P02/P13)"),
    (r"T\d+\s*裁定", "internal decision pointer (T7 裁定)"),
    (r"R\d+-review", "internal review-record pointer"),
    (r"零写入", "evaluation-design phrasing"),
    (r"引导能力测试抓手", "evaluation-design phrasing"),
    (r"半闭环", "evaluation-design phrasing"),
    (r"外审|评审|审查报告|handoff|中枢", "external-review / internal-workflow word"),
)


def find_forbidden(
    text: str, patterns: tuple[tuple[str, str], ...] = FORBIDDEN_PATTERNS
) -> list[tuple[str, str]]:
    """Return [(pattern, matched_text)] for every forbidden hit in ``text``.

    ``patterns`` defaults to the tool-description set; the task-text lint
    passes its own set (single mechanism, documented sets — see the comment
    on FORBIDDEN_PATTERNS above).
    """
    return [
        (pattern, m.group(0))
        for pattern, _label in patterns
        for m in re.finditer(pattern, text)
    ]


@pytest.fixture(scope="module")
def tool_descriptions() -> dict[str, str]:
    """Runtime descriptions of both tool packages: env.tools + env.user_tools
    (15 agent tools + 4 user tools), taken from the live openai_schema."""
    env = get_environment()
    tool_objs = list(env.get_tools()) + list(env.get_user_tools())
    return {
        t.name: t.openai_schema["function"]["description"]
        for t in tool_objs
    }


def test_tool_inventory_covered(tool_descriptions):
    """Non-vacuous coverage pin: all 19 tools (15 agent + 4 user) are linted,
    including every tool the cleanup touched. Bump deliberately if the
    inventory itself ever changes."""
    assert len(tool_descriptions) == 19
    assert {
        "submit_deferral",
        "request_certificate",
        "submit_scholarship_app",
        "create_ticket",
        "withdraw_application",
        "reject_suggestion",
    } <= set(tool_descriptions)


def test_descriptions_free_of_forbidden_tokens(tool_descriptions):
    """Every runtime tool description carries zero forbidden hits."""
    offenders = {
        name: find_forbidden(desc)
        for name, desc in tool_descriptions.items()
        if find_forbidden(desc)
    }
    assert offenders == {}


def test_detector_flags_pre_cleanup_counterexamples():
    """Counterexample red-proof: synthetic descriptions reproducing the
    pre-cleanup wording must be flagged by the very detector the lint uses."""
    old_ticket = (
        "工单（第34/35条）。仅拦'申诉+成绩+逾期'（v1.1-F2）；"
        "复核申诉必须挂学院原工单（越级 E-LEVEL）。"
    )
    old_reject = (
        "拒签/放弃（候补确认单拒签＝放弃候补，abandon_count+1）。"
        "引导能力测试抓手：Agent 建议不当，学生可当面拒签。"
    )
    old_scholarship = "受理即时资格校验，任一不过逐条拒绝零写入；通过置'待签署'+SIG。"

    ticket_hits = {p for p, _ in find_forbidden(old_ticket)}
    assert r"v\d+\.\d+-[A-Z]" in ticket_hits
    assert r"E-[A-Z]{2,}" in ticket_hits

    reject_hits = {m for _, m in find_forbidden(old_reject)}
    assert "引导能力测试抓手" in reject_hits

    scholarship_hits = {m for _, m in find_forbidden(old_scholarship)}
    assert "零写入" in scholarship_hits


def test_allowed_domain_vocabulary_not_flagged():
    """Policy clause numbers and document/field codes stay legal: the lint
    must not over-fire on domain vocabulary."""
    clean = (
        "撤回申请（第13/30条）。CE 限出具前撤回，DF/APP 限签署确认前撤回；"
        "撤回不占额度/不产生记录。候补确认单拒签＝放弃候补（abandon_count+1）；"
        "申诉目标成绩行 GR-xxx 按第16条判窗。"
    )
    assert find_forbidden(clean) == []
