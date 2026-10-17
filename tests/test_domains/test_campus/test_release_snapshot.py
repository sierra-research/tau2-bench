"""Release-snapshot regression guards for the campus domain (W3, P1-13).

Freezes two kinds of "must not drift back" facts as pytest assertions:

1. **Runtime metadata constants** exposed by ``tau2.domains.campus``
   (``__version__`` / ``POLICY_VERSION`` / ``SCORING_PROTOCOL``).
2. **The data-cleaning results** on the two public data files:
   - ``policy.md``: zero internal meta-annotations — "坑点" pitfall tags,
     ``v1.x/F|R|M`` internal version labels, and the ``P1-10`` changelog
     reference (W1 scrub); plus a single-line version footer
     (``本规程版本：vX.Y.Z（YYYY-MM-DD）`` as the exact last line, exactly
     once, version equal to ``POLICY_VERSION``);
   - ``tasks.json``: zero ``description.notes`` keys (W1), zero ``issues``
     keys, and zero exam-point number hints (``P0[0-9]``) / "坑点" in the
     agent-visible ``purpose`` / ``relevant_policies`` fields (W2 scrub);
     plus an **all-string-leaves** census — no authoring marker
     (金标 / D-S# / v1.x-bracket / 钉死 / 抽签 / 考点 / tools-spec /
     E-DATA / leading "考 " / M#-LETTER / P#-#) anywhere in any string
     leaf of any task (scope: the whole file, not just the
     agent-visible fields). The census also covers the statement-rewrite
     vocabulary (工具规格 / 题面 / 半闭环 / 读列习惯 / 常识臆断 / 零写 /
     单通道 / 防两种 / 锚点 / DB 双哈希), with embedded pre-rewrite samples
     asserted red so the extended set provably rejects the old wording.

Version linkage: ``POLICY_VERSION == "1.4.2"`` is the **current contract
value**. If a future release bumps it, this assertion and the release
manifest must be updated **in the same change** — they move together.
The same lockstep holds for the benchmark axis: ``manifest.json``
(benchmark_version / policy_version / task_count / per-file SHA-256 of
the five data payloads) must agree with the runtime constants, with the
domain README (both language halves), and with the data files on disk —
see the manifest section at the bottom of this module.

A failure here means a cleaned field regressed (or an authorized version
bump landed without updating this snapshot): fix the drift or update the
snapshot deliberately — never weaken the assertions to get green.
"""

import hashlib
import json
import re
from pathlib import Path

import pytest

import tau2.domains.campus as campus
from tau2.domains.campus.utils import (
    CAMPUS_DATA_DIR,
    CAMPUS_POLICY_PATH,
    CAMPUS_TASK_SET_PATH,
)

# --- fixtures (read once per module) -------------------------------------


@pytest.fixture(scope="module")
def policy_text() -> str:
    return CAMPUS_POLICY_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def tasks_text() -> str:
    return CAMPUS_TASK_SET_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def tasks(tasks_text) -> list[dict]:
    return json.loads(tasks_text)


# --- 1. runtime metadata constants ---------------------------------------


def test_version_is_semver():
    """__version__ tracks the campus benchmark axis (campus-vX.Y.Z tags)."""
    assert re.fullmatch(r"\d+\.\d+\.\d+", campus.__version__)


def test_policy_version_contract():
    """Current contract value — bump together with the release manifest."""
    assert campus.POLICY_VERSION == "1.4.2"


def test_scoring_protocol_contract():
    assert campus.SCORING_PROTOCOL == "tau2-v1.0.1-compatible"


# --- 2. policy.md — W1 meta-annotation scrub must hold --------------------


def test_policy_has_no_pitfall_annotations(policy_text):
    assert "坑点" not in policy_text


def test_policy_has_no_internal_version_tags(policy_text):
    """v1.x/F|R|M labels were internal review tags, not policy content."""
    assert re.search(r"v1\.[0-9]/[FRM]", policy_text) is None


def test_policy_has_no_p1_10_reference(policy_text):
    assert "P1-10" not in policy_text


# --- 3. tasks.json — W1/W2 scrubs must hold -------------------------------


def test_task_count(tasks):
    assert len(tasks) == 50


def test_descriptions_have_no_notes_key(tasks):
    """W1 stripped description.notes from 50/50 tasks (gold-hinting QA)."""
    offenders = [t["id"] for t in tasks if "notes" in t["description"]]
    assert not offenders, f"description.notes reintroduced: {offenders}"


def test_tasks_text_has_no_issues_key(tasks_text):
    assert '"issues"' not in tasks_text


def test_agent_visible_fields_free_of_exam_point_hints(tasks):
    """W2 scrub: no P0x-style exam-point numbers or 坑点 in fields the
    agent can see through TaskDescription rendering."""
    offenders = []
    for task in tasks:
        description = task["description"]
        for field in ("purpose", "relevant_policies"):
            value = description.get(field) or ""
            if re.search(r"P0[0-9]", value) or "坑点" in value:
                offenders.append(f"{task['id']}.{field}")
    assert not offenders, f"exam-point hints reintroduced: {offenders}"


# --- 4. all-string-leaves marker census + policy version footer ------------

# Authoring/internal markers scrubbed from the public data. Scope: EVERY
# string leaf of tasks.json (not only the agent-visible fields). "考 " is
# leaf-anchored (leading only) so words like "缺考" never match.
#
# The second block below is the statement-rewrite vocabulary: design/grading
# shorthand that must never come back into any task leaf. Generic-word policy
# (corpus-wide false-positive evaluation): "口径" is deliberately NOT in the
# set — policy.md itself uses it legitimately (two clauses), and tasks are
# allowed to paraphrase or quote that language; "零写"/"单通道"/"锚点"/
# "防两种" are in because they are compressed authoring shorthand with no
# legitimate use in this corpus (the plain-language phrasing is standardized)
# and match nothing outside tasks.json.
INTERNAL_MARKER_PATTERNS = {
    "金标": r"金标",
    "D-S": r"D-S\d",
    "v1.x-bracket": r"v1\.\d[/（( ]",
    "钉死": r"钉死",
    "抽签": r"抽签",
    "考点": r"考点",
    "tools-spec": r"tools-spec",
    "E-DATA": r"E-DATA",
    "leading-考": r"^考 ",
    "M#-LETTER": r"M\d+-[A-Z]",
    "P#-#": r"P\d+-\d",
    # unpublished-spec citation
    "工具规格": r"工具规格",
    # authoring-face references
    "题面": r"题面",
    # design shorthand replaced by plain wording
    "半闭环": r"半闭环",
    "读列习惯": r"读列习惯",
    "常识臆断": r"常识臆断",
    "零写": r"零写",
    "单通道": r"单通道",
    "防两种": r"防两种",
    "锚点": r"锚点",
    # grading shorthand ("DB 双哈希必挂" family)
    "DB 双哈希": r"DB\s*双哈希",
}

# Statement-rewrite samples: each is the pre-rewrite text of the named leaf
# and must be rejected by (at least) the patterns listed for it. The live
# counterpart of every sample must stay clean — that pair is asserted below.
PRE_REWRITE_SAMPLES = [
    {
        "id": "E03",
        "path": ("description", "relevant_policies"),
        "text": "政策第7条（窗口内退课即时生效）；工具规格 drop_course。",
        "must_match": ["工具规格"],
    },
    {
        "id": "H04",
        "path": ("user_scenario", "persona"),
        "text": (
            "深度改主意型：补选→反悔退→再反悔要回；反悔节奏由题面固定，"
            "单终态=已选；对 Agent 提议'顺其自然别再选'顺从=否（坚持要回）。"
        ),
        "must_match": ["题面"],
    },
    {
        "id": "H09",
        "path": ("description", "purpose"),
        "text": (
            "递补确认时序＋第5条学分意识：递补确认单 6/12 18:00 到期（当日死线）、"
            "退课窗口 6/20 内随时可退——先后次序由 Agent 提醒固定；按'先退后签'"
            "学生意愿执行（题面固定单通道，防两种时序混用）。"
        ),
        "must_match": ["题面", "单通道", "防两种"],
    },
    {
        "id": "M06",
        "path": ("description", "purpose"),
        "text": (
            "因病缓考无效材料半闭环：社区诊所证明按第12条无效驳回，"
            "Agent 不得代改医院等级、不得代传材料；零写。"
        ),
        "must_match": ["半闭环", "零写"],
    },
    {
        "id": "E01",
        "path": ("description", "purpose"),
        "text": (
            "get_course_offerings 读列习惯：须读 adddrop_deadline 列判断还能否加选，"
            "而非'开学了就还能选'的常识臆断；顺带报出容量与余量。"
        ),
        "must_match": ["读列习惯", "常识臆断"],
    },
    {
        "id": "H10",
        "path": ("description", "purpose"),
        "text": (
            "立场（逾期不受理查分）与咨询/申诉边界（咨询可建、不得以申诉替代）："
            "唯一允许写入=一张咨询工单，若生成申诉工单或任何成绩变更，DB 双哈希必挂。"
        ),
        "must_match": ["DB 双哈希"],
    },
    {
        "id": "E12",
        "path": ("description", "purpose"),
        "text": (
            "get_service_requests(certificate) 到达态读取：委托代领件已'可领取'；"
            "月末结账提醒按条件触发（锚点 6/12 不撞结账）。"
        ),
        "must_match": ["锚点"],
    },
    {
        "id": "M05",
        "path": ("description", "purpose"),
        "text": (
            "因病考后 3 个工作日补办时限计算，与 dual-control 半闭环：Agent 提交申请后"
            "必须引导学生本人上传三甲材料并签署，缺一则终态不达标（DB 双哈希必挂）。"
        ),
        "must_match": ["半闭环", "DB 双哈希"],
    },
    {
        "id": "E13",
        "path": ("description", "purpose"),
        "text": (
            "search_policy 第20条（转专业）流程读取：须主动说'仅限一次'与"
            "辅修/双学位替代路径；零写。"
        ),
        "must_match": ["零写"],
    },
]

# patterns added by the statement rewrite (must each be exercised below)
EXTENDED_MARKER_NAMES = (
    "工具规格",
    "题面",
    "半闭环",
    "读列习惯",
    "常识臆断",
    "零写",
    "单通道",
    "防两种",
    "锚点",
    "DB 双哈希",
)

POLICY_FOOTER_RE = re.compile(r"本规程版本：v(\d+\.\d+\.\d+)（(\d{4}-\d{2}-\d{2})）")


def _walk_strings(node, path=""):
    """Yield (path, text) for every string leaf in a JSON-like structure."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _walk_strings(value, f"{path}.{key}")
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from _walk_strings(value, f"{path}[{i}]")
    elif isinstance(node, str):
        yield path, node


def test_tasks_all_string_leaves_free_of_internal_markers(tasks):
    """No authoring marker in ANY string leaf of tasks.json — including
    persona, actions[].info, instructions, etc."""
    offenders = []
    for task in tasks:
        for path, text in _walk_strings(task):
            for name, pattern in INTERNAL_MARKER_PATTERNS.items():
                if re.search(pattern, text):
                    offenders.append(f"{task['id']}{path}:{name}")
    assert not offenders, f"internal markers reintroduced: {offenders}"


def test_counterexample_pre_rewrite_samples_rejected_by_marker_set(tasks):
    """Red-on-old / green-on-new for the extended marker set: every embedded
    pre-rewrite sample must be rejected by the patterns named for it, every
    added pattern must be exercised by at least one sample, and the live leaf
    behind each sample must stay clean."""
    exercised = set()
    for sample in PRE_REWRITE_SAMPLES:
        label = f"{sample['id']}{'.'.join(sample['path'])}"
        caught = [
            name
            for name, pattern in INTERNAL_MARKER_PATTERNS.items()
            if re.search(pattern, sample["text"])
        ]
        for name in sample["must_match"]:
            assert name in caught, (
                f"pre-rewrite sample {label} slips past pattern {name!r} "
                f"(caught only: {caught})"
            )
        exercised.update(caught)
        # live counterpart passes
        task = next(t for t in tasks if t["id"] == sample["id"])
        node = task
        for part in sample["path"]:
            node = node[part]
        live_caught = [
            name
            for name, pattern in INTERNAL_MARKER_PATTERNS.items()
            if re.search(pattern, node)
        ]
        assert not live_caught, (
            f"live leaf {label} matches marker patterns {live_caught}"
        )
    missing = [name for name in EXTENDED_MARKER_NAMES if name not in exercised]
    assert not missing, f"added patterns never exercised by a sample: {missing}"


def test_policy_has_version_footer(policy_text):
    """policy.md ends with exactly one single-line version footer whose
    version equals POLICY_VERSION (footer and constant move together)."""
    lines = [line for line in policy_text.splitlines() if line.strip()]
    assert lines, "policy.md is empty"
    last = lines[-1]
    match = POLICY_FOOTER_RE.fullmatch(last)
    assert match, f"policy.md last line must be the version footer, got: {last!r}"
    footers = [line for line in lines if line.startswith("本规程版本：")]
    assert len(footers) == 1, f"expected exactly one version footer: {footers}"
    assert match.group(1) == campus.POLICY_VERSION, (
        f"policy footer v{match.group(1)} != POLICY_VERSION "
        f"{campus.POLICY_VERSION} — bump both in the same change"
    )


# --- manifest.json — machine-readable code↔data pairing -------------------
#
# ``manifest.json`` binds the benchmark version, the policy version, the task
# count and the per-file SHA-256 of the five data payloads to this code tree.
# Hashes are computed over each file's LF-normalized bytes, so the recorded
# digests hold regardless of checkout line endings (CRLF worktree vs LF CI).

MANIFEST_DATA_PAYLOADS = frozenset(
    {"db.json", "policy.md", "split_tasks.json", "tasks.json", "user_db.json"}
)


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads((CAMPUS_DATA_DIR / "manifest.json").read_text(encoding="utf-8"))


def test_manifest_version_constants_twin(manifest):
    """__version__ / POLICY_VERSION / SCORING_PROTOCOL ⇔ manifest: moving
    either side without the other fails here (same-change discipline).
    ``code_tag`` is prefilled at commit time and twin-asserted against
    ``__version__`` (``campus-v`` + version, loose campus-vX.Y.Z shape); the
    concrete commit sha and dataset revision live in the Release asset manifest
    of the matching tag (``release_binding``). The old truthy-only
    ``code_commit`` / ``dataset_revision`` placeholders are removed — a
    placeholder can no longer satisfy the guard (P1-7, R25-b)."""
    assert manifest["benchmark_version"] == campus.__version__
    assert manifest["policy_version"] == campus.POLICY_VERSION
    assert manifest["evaluator_protocol"] == campus.SCORING_PROTOCOL
    assert re.fullmatch(r"\d+\.\d+\.\d+", manifest["benchmark_version"])
    import re as _re
    assert _re.fullmatch(r"campus-v\d+\.\d+\.\d+", manifest["code_tag"]), (
        "code_tag must look like campus-vX.Y.Z"
    )
    assert manifest["code_tag"] == "campus-v" + campus.__version__, (
        f"code_tag {manifest['code_tag']!r} must twin campus-v"
        f"{campus.__version__!r}"
    )
    assert "code_commit" not in manifest and "dataset_revision" not in manifest, (
        "release placeholders code_commit/dataset_revision must be removed"
    )
    assert "release_binding" in manifest, "release_binding note must be present"


def test_no_release_placeholder_text_in_campus_tree():
    """P1-7 regression: the 'filled at release' placeholder must be absent from
    the campus source package and the published data directory, so the twin
    guard can never again be satisfied by a truthy placeholder. Scan scope is
    the two canonical homes of code/data version facts (src package + data)."""
    needle = "filled at release"
    roots = [Path(campus.__file__).parent, CAMPUS_DATA_DIR]
    hits = []
    for root in roots:
        for pattern in ("*.py", "*.json", "*.md"):
            for path in sorted(root.rglob(pattern)):
                try:
                    text = path.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    continue
                if needle in text:
                    hits.append(str(path))
    assert not hits, f"'filled at release' placeholder reintroduced: {hits}"


def test_manifest_data_file_hashes(manifest):
    """Per-file digests ⇔ bytes on disk: editing any data payload without
    regenerating manifest.json fails here, and vice versa."""
    assert set(manifest["files"]) == MANIFEST_DATA_PAYLOADS
    for name, recorded in sorted(manifest["files"].items()):
        path = CAMPUS_DATA_DIR / name
        assert path.exists(), f"data payload missing: {name}"
        digest = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        assert digest == recorded, (
            f"{name}: manifest sha256 {recorded[:12]} != actual {digest[:12]} "
            "— regenerate manifest.json together with the data change"
        )


def test_manifest_task_count_twin(manifest, tasks):
    assert manifest["task_count"] == len(tasks) == 50


def test_domain_readme_states_current_policy_version(manifest):
    """manifest ⇔ domain README: both language halves of the README state
    the current policy version, so the constant, the machine-readable
    manifest and the documentation cannot silently drift apart."""
    assert manifest["policy_version"] == campus.POLICY_VERSION
    readme = Path(campus.__file__).with_name("README.md")
    text = readme.read_text(encoding="utf-8")
    stated = f"v{campus.POLICY_VERSION}"
    assert text.count(stated) >= 2, (
        f"domain README must state {stated} in both language halves "
        f"(found {text.count(stated)})"
    )
