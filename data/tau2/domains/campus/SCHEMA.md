# SCHEMA — field reference for the campus data files

> Bilingual (EN below, 中文见下半区) companion to the domain README (`src/tau2/domains/campus/README.md`). It documents every field of the published data payloads so the files can be consumed without reading the domain source. Field *shapes* follow the upstream τ²-bench task schema; the notes state this domain's fixed values. Counts are those of this release; `manifest.json` carries the per-file SHA-256 pairing.

## 1. `tasks.json` — 50 tasks (JSON array of objects)

Top-level keys, present on every one of the 50 tasks: `id`, `description`, `user_scenario`, `ticket`, `initial_state`, `evaluation_criteria`, `required_documents`, `user_tools`.

| Field | Type | Shape / fixed values in this domain |
|---|---|---|
| `id` | string | `E`/`M`/`H` prefix + two-digit number (E01–E15, M01–M20, H01–H15). The prefix **is** the difficulty label — see §4. |
| `description` | object | exactly 2 keys: `purpose` (string — the task's goal statement, rendered into the agent prompt) and `relevant_policies` (string — the policy clauses that govern the task). |
| `user_scenario` | object | exactly 2 keys: `persona` (string — the scripted student character) and `instructions` (object with exactly 5 keys: `domain` = `"campus"` on all 50 tasks; `reason_for_call`, `known_info`, `unknown_info`, `task_instructions` — all strings). |
| `ticket` | null | **`null` on all 50 tasks.** The upstream schema field is kept for shape compatibility; this domain's initial states open no ticket. |
| `initial_state` | object | 3 keys — see §2. |
| `evaluation_criteria` | object | 5 keys — see §3. |
| `required_documents` | null | **`null` on all 50 tasks.** Kept for upstream shape compatibility; required documents are expressed through the policy text and `evaluation_criteria`, not through this field. |
| `user_tools` | array of string | The student-side tools available in this task, drawn from the 4 user tools (`check_student_app`, `upload_material`, `confirm_action`, `reject_suggestion`): 39 tasks expose `check_student_app` only, 6 expose 2, 2 expose 3, 3 expose all 4. |

## 2. `initial_state` — 3 keys

| Key | Shape |
|---|---|
| `initialization_data` | `{agent_data, user_data}`. `agent_data` always carries `env.current_time` (the task's server-time anchor; overrides only this key) and may pre-load business-table rows for this task (`students` / `enrollments` / `grades` / `course_offerings` / `exam_arrangements` / `deferral_requests` / `certificates`). `user_data` is `null` on 46 tasks; on 4 tasks (M04, M12, H07, H08) it carries user-table row overrides (`pending_signatures` / `app_todos` / `uploads`). Its non-DB key `student_id` binds the student-side identity at initialization and never reaches the database. |
| `initialization_actions` | array — exactly one entry on every task: `{"env_type": "user", "func_name": "bind_student", "arguments": {"student_id": …}}`, binding the student-side identity. |
| `message_history` | `null` on all 50 tasks (no pre-seeded chat turns). |

**Time anchors**: 42 tasks anchor `current_time` at `2026-06-12 10:00`; 7 tasks use boundary anchors around it (2026-06-11 10:00 … 2026-06-30 17:30). **Task H06 anchors at `2026-03-20 10:00`** — a deliberate design re-anchor introduced together with the special-channel 10-working-day submission window (v1.1.1): H06's scenario must begin inside the special-channel expiry window, ~3 months before the shared anchor, so the window is still open when the task starts.

## 3. `evaluation_criteria` — 5 keys

| Key | Shape / fixed values in this release |
|---|---|
| `actions` | array of reference (gold) action objects — 21 tasks have ≥1, **29 tasks have `[]`** (the zero-write tasks). Object keys (all 6 on every action): `action_id` (e.g. `"E02_0"`), `requestor` (`"assistant"` ×33 / `"user"` ×18), `name` (tool name), `arguments` (object), `info` (free-text annotation, 48 strings + 3 `null` — never read by the evaluator), `compare_args` (`null` throughout this release). |
| `env_assertions` | array of diagnostic assertions, `null`, or `[]`. **54 assertions across 25 tasks** (`null` on 24 tasks, `[]` on 1). Object keys: `env_type` (`"assistant"` ×34 / `"user"` ×20), `func_name` (one of 7 `assert_*` helpers), `arguments` (object), `assert_value` (boolean). Diagnostic only — reported in `RewardInfo.env_assertions`, never gates the reward under the v1.0.1 contract. |
| `communicate_info` | array of plain strings — must-convey entities; **98 strings** across the 50 tasks, graded by deterministic substring match against the agent's replies. |
| `nl_assertions` | `null` on all 50 tasks (upstream field, unused here). |
| `reward_basis` | `["DB", "COMMUNICATE"]` on all 50 tasks (upstream default; `RewardType.ACTION` not used). |

## 4. `split_tasks.json` — difficulty mapping

`{"base": [50 ids], "easy": [15 ids], "medium": [20 ids], "hard": [15 ids]}`. The split lists **are** the difficulty mapping: `easy` is exactly the set of `E*` ids, `medium` the `M*` ids, `hard` the `H*` ids (verified programmatically for this release). There is no per-task `difficulty` field — the id prefix is the single difficulty signal, and `split_tasks.json` is the machine-readable index of the same mapping.

## 5. `db.json` / `user_db.json` — table inventory

Both files are JSON objects mapping primary key → row object.

**`db.json`** — `env` (server-time anchor `current_time` 2026-06-12 10:00, `term` 2026SP, maintenance windows) plus 11 business tables (agent-side academic-affairs system):

| Table | Rows | Table | Rows |
|---|---:|---|---:|
| `students` | 40 | `exam_arrangements` | 58 |
| `courses` | 30 | `deferral_requests` | 6 |
| `course_offerings` | 56 | `scholarships` | 8 |
| `enrollments` | 434 | `scholarship_apps` | 8 |
| `grades` | 276 | `certificates` | 6 |
| `tickets` | 5 | | |

**`user_db.json`** — 3 student-side (app) tables: `app_todos` (16 rows), `pending_signatures` (16), `uploads` (4). Together with `db.json`'s 11 business tables this is the 14-table dual-control surface.

## 6. `manifest.json` — code↔data pairing

Machine-readable binding published with the data: `benchmark_version`, `policy_version`, `evaluator_protocol`, `code_tag` (prefilled as `campus-vX.Y.Z`, twin-validated against `__version__`), `release_binding`, `task_count`, and `files` — the per-file SHA-256 of the five data payloads (`tasks.json`, `split_tasks.json`, `policy.md`, `db.json`, `user_db.json`), hashed over each file's LF-normalized bytes so the value is independent of checkout line endings. The concrete commit SHA and dataset revision are recorded in the Release asset manifest of the matching tag (see the `release_binding` key). Twin assertions in the test suite keep `manifest.json` ⇔ runtime version constants ⇔ file bytes in lockstep — including the `code_tag`⇔`__version__` pair: tampering with any of them fails the suite.

---

# SCHEMA — campus 数据文件字段字典

> 双语（上半区英文，下半区中文）；域 README（`src/tau2/domains/campus/README.md`）的配套文件。本字典覆盖已发布数据载荷的全部字段，使读者无需翻阅域源码即可消费这些文件。字段**形状**沿用上游 τ²-bench 任务 schema；注记给出本域的固定取值。计数为本版本口径；`manifest.json` 携带逐文件 SHA-256 配对。

## 1. `tasks.json` — 50 道任务（JSON 对象数组）

50 道任务每题都有的顶层键：`id`、`description`、`user_scenario`、`ticket`、`initial_state`、`evaluation_criteria`、`required_documents`、`user_tools`。

| 字段 | 类型 | 本域的形态 / 固定取值 |
|---|---|---|
| `id` | string | `E`/`M`/`H` 前缀＋两位编号（E01–E15、M01–M20、H01–H15）。前缀**就是**难度标签——见 §4。 |
| `description` | object | 恰 2 键：`purpose`（string，任务目标陈述，渲染进 agent 提示）、`relevant_policies`（string，管辖该任务的政策条款）。 |
| `user_scenario` | object | 恰 2 键：`persona`（string，逐题钉死的学生角色）与 `instructions`（object，恰 5 键：`domain`＝`"campus"`（50 题全部）；`reason_for_call`、`known_info`、`unknown_info`、`task_instructions` 均为 string）。 |
| `ticket` | null | **50 题恒为 `null`。** 保留该键以维持上游 schema 形状；本域初始状态不开任何工单。 |
| `initial_state` | object | 3 键——见 §2。 |
| `evaluation_criteria` | object | 5 键——见 §3。 |
| `required_documents` | null | **50 题恒为 `null`。** 保留以维持上游 schema 形状；所需材料经政策正文与 `evaluation_criteria` 表达，不走该字段。 |
| `user_tools` | string 数组 | 本题可用的学生侧工具，取自 4 个用户工具（`check_student_app`、`upload_material`、`confirm_action`、`reject_suggestion`）：39 题仅暴露 `check_student_app`，6 题 2 个，2 题 3 个，3 题全 4 个。 |

## 2. `initial_state` — 3 键

| 键 | 形态 |
|---|---|
| `initialization_data` | `{agent_data, user_data}`。`agent_data` 恒含 `env.current_time`（任务服务器时间锚；覆写只动这一个键），并可为本题预载业务表行（`students` / `enrollments` / `grades` / `course_offerings` / `exam_arrangements` / `deferral_requests` / `certificates`）。`user_data` 在 46 题为 `null`；在 4 题（M04、M12、H07、H08）携带用户表行覆写（`pending_signatures` / `app_todos` / `uploads`）。其中的非库键 `student_id` 在初始化时绑定学生侧身份，永不入库。 |
| `initialization_actions` | 数组——每题恰 1 条：`{"env_type": "user", "func_name": "bind_student", "arguments": {"student_id": …}}`，绑定学生侧身份。 |
| `message_history` | 50 题恒为 `null`（无预置对话轮次）。 |

**时间锚**：42 题的 `current_time` 锚在 `2026-06-12 10:00`；7 题使用其周边界锚（2026-06-11 10:00 … 2026-06-30 17:30）。**任务 H06 锚在 `2026-03-20 10:00`**——这是随特别通道十工作日提交窗（v1.1.1）一并引入的设计性重锚：H06 的情景必须在特别通道过期窗内开始（早于公共锚约 3 个月），使该窗在任务开始时仍然开放。

## 3. `evaluation_criteria` — 5 键

| 键 | 本版本的形态 / 固定取值 |
|---|---|
| `actions` | 参考（gold）动作对象数组——21 题 ≥1 条，**29 题为 `[]`**（零写题）。对象键（每条动作恒 6 键）：`action_id`（如 `"E02_0"`）、`requestor`（`"assistant"` ×33 / `"user"` ×18）、`name`（工具名）、`arguments`（object）、`info`（自由文本注记，48 条 string＋3 条 `null`——evaluator 不读）、`compare_args`（本版本恒 `null`）。 |
| `env_assertions` | 诊断断言数组、`null` 或 `[]`。**25 题共 54 条**（24 题 `null`、1 题 `[]`）。对象键：`env_type`（`"assistant"` ×34 / `"user"` ×20）、`func_name`（7 个 `assert_*` 助手之一）、`arguments`（object）、`assert_value`（boolean）。仅作诊断——经 `RewardInfo.env_assertions` 报告，在 v1.0.1 契约下不参与 reward 门槛。 |
| `communicate_info` | 纯 string 数组——必现实体；50 题共 **98 条**，按对 agent 全程回复的确定性子串匹配判分。 |
| `nl_assertions` | 50 题恒为 `null`（上游字段，本域不使用）。 |
| `reward_basis` | 50 题恒为 `["DB", "COMMUNICATE"]`（上游默认；不使用 `RewardType.ACTION`）。 |

## 4. `split_tasks.json` — 难度映射

`{"base": [50 个 id], "easy": [15], "medium": [20], "hard": [15]}`。分片表**就是**难度映射：`easy` 恰为全部 `E*` id、`medium` 恰为 `M*`、`hard` 恰为 `H*`（本版本已程序化验证）。任务没有单独的 `difficulty` 字段——id 前缀是唯一的难度信号，`split_tasks.json` 是同一映射的机器可读索引。

## 5. `db.json` / `user_db.json` — 表清单

两个文件都是「主键 → 行对象」的 JSON 对象。

**`db.json`**——`env`（服务器时间锚 `current_time` 2026-06-12 10:00、`term` 2026SP、维护窗口）＋11 张业务表（agent 侧教务系统）：

| 表 | 行数 | 表 | 行数 |
|---|---:|---|---:|
| `students` | 40 | `exam_arrangements` | 58 |
| `courses` | 30 | `deferral_requests` | 6 |
| `course_offerings` | 56 | `scholarships` | 8 |
| `enrollments` | 434 | `scholarship_apps` | 8 |
| `grades` | 276 | `certificates` | 6 |
| `tickets` | 5 | | |

**`user_db.json`**——3 张学生侧（App）表：`app_todos`（16 行）、`pending_signatures`（16）、`uploads`（4）。与 `db.json` 的 11 张业务表合计即 14 表双控面。

## 6. `manifest.json` — 代码↔数据配对

随数据发布的机器可读绑定：`benchmark_version`、`policy_version`、`evaluator_protocol`、`code_tag`（提交时预填为 `campus-vX.Y.Z`，可与 `__version__` 孪生校验）、`release_binding`、`task_count`，以及 `files`——五个数据载荷（`tasks.json`、`split_tasks.json`、`policy.md`、`db.json`、`user_db.json`）的逐文件 SHA-256，按各文件 LF 归一化后的字节计算，使取值与检出换行符无关。具体 commit SHA 与数据修订记录于对应版本的 Release 资产清单（见 `release_binding` 键）。测试套件中的孪生断言让 `manifest.json` ⇔ 运行时版本常量 ⇔ 文件字节保持同步（含 `code_tag`⇔`__version__` 配对）：任何一处被篡改都会让测试变红。
