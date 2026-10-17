"""Toolkit for the campus domain (Agent side).

Implements the campus tools spec v1.1:
READ 7 + WRITE 8 (incl. withdraw_application), all WRITE pass the maintenance-window gate,
all tool results carry `server_time` as first field (政策第2条), and all time reads come
from `db.env.current_time` — never the system clock.

State-machine settlement (时序契约) lives in `settle_*` helpers called
by `CampusEnvironment.sync_tools()` after every message / initialization action, so
live runs and evaluator replays converge identically.
"""

import re
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

from tau2.domains.campus.data_model import (
    AwardAppRow,
    AwardAppStatus,
    AwardChannel,
    CampusDB,
    CertDelivery,
    CertRow,
    CertStatus,
    DeferralFilingType,
    DeferralReasonType,
    DeferralRow,
    DeferralStatus,
    DropChannel,
    EnrollmentRow,
    EnrollmentSource,
    EnrollmentStatus,
    ExamStatus,
    HospitalLevel,
    OfferingRow,
    OfferingStatus,
    ProxyInfo,
    ReviewStage,
    SigDocType,
    SignatureRow,
    SigStatus,
    StudentRow,
    StudentStatus,
    TicketCategory,
    TicketLevel,
    TicketModule,
    TicketRow,
    TicketStatus,
    TodoRow,
    TodoStatus,
    TodoType,
    UploadDocType,
)
from tau2.domains.campus.user_data_model import UserDB
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool
from tau2.utils.pydantic_utils import BaseModelNoExtra

TIME_FMT = "%Y-%m-%d %H:%M"


def _s(v: Any) -> str:
    """Enum-safe stringification: str-Enum 直接 f-string 会打印 'Class.MEMBER'，
    统一转中文 value（重放两侧同源，判分/展示一致）。"""
    return v.value if isinstance(v, Enum) else str(v)

# --------------------------------------------------------------- 错误文案目录（措辞冻结：测试与金标依赖原文）

E_WINDOW_CLOSED = "补退选已于 {deadline} 截止（政策第7条）。本学期不再受理{action}；如为毕业班学生可走特别通道（第9条）。"
E_PREREQ_MISSING = "《{course}》要求先修《{prereq}》合格或正在修读（政策第6条）。当前不满足。"
E_CAPACITY_FULL = "该课已满，可加入候补（第8条）；候补递补后须在24小时内确认。"
E_TIME_CONFLICT = "与《{other}》上课时间冲突（星期{w} {t1}-{t2}），不可同时选修。"
E_DEFERRAL_EXPIRED = "该场考试已过补办时限（考后3个工作日，政策第12条），无法申请缓考；未考将按'缺'记载（第18条）。"
E_DEFERRAL_USED = "《{course}》本学期已申请过缓考（第13条：仅一次），不再受理。"
E_APPROVAL_PENDING = "毕业班特别通道申请仍在学院/教务处双重审核中（第9条），请等待结果。"
E_AID_POOL = "国家助学金/励志奖学金要求已认定入库（第27条）。可先申请临时困难补助（若符合第28条突发情形）。"
E_RECORD_UNELIGIBLE = "评定学年存在{cause}（第24条），不符合申请条件；重修通过不消除当学年记录（第19条）。"
E_STACK_CONFLICT = "与已申请/已获{award}构成兼得限制（第25条），请撤回其一。"
E_MAINTENANCE = "当前处于系统维护时段（周日23:00–周一06:00，第3条），写操作暂停，查询不受影响。"
E_CERT_BATCH = "证明系统月末结账中（最后工作日17:00–22:00，第31条），申请已顺延。"
E_REVIEW_EXPIRED = "成绩公布已超5个工作日，查分不受理（第16条）；申诉通道不适用于替代查分。"
E_LEVEL = "复核须先经学院处理（第34条），请提供原学院工单号后提交。"
# 候补冻结错误码：join_waitlist 引用了该错误码，但目录原本未收录文案；
# 按目录体例补写（中文、含指引）。
E_WAITLIST_FROZEN = "本学期候补放弃已累计3次（政策第8条），候补功能已关闭，不再受理加入候补。"

# 周几中文映射（weekday(): 0=周一）
WEEKDAY_CN = ["一", "二", "三", "四", "五", "六", "日"]

VALID_MEDICAL_LEVELS = {HospitalLevel.TIER3.value, HospitalLevel.CAMPUS_CLINIC.value}


def parse_time(s: str) -> datetime:
    return datetime.strptime(s, TIME_FMT)


def fmt_time(dt: datetime) -> str:
    return dt.strftime(TIME_FMT)


def day_end(dt: datetime) -> datetime:
    """当日 23:59（政策第2条：时限截止时刻为当日23:59）。"""
    return dt.replace(hour=23, minute=59, second=0, microsecond=0)


def is_workday(dt: datetime) -> bool:
    """第2条：工作日＝周一至周五（法定节假日不入计算，已知简化）。"""
    return dt.weekday() < 5


def add_workdays(dt: datetime, n: int) -> datetime:
    """自 dt 起顺延 n 个工作日（逐日跳过周六日），保留时分。"""
    cur = dt
    remaining = n
    while remaining > 0:
        cur += timedelta(days=1)
        if is_workday(cur):
            remaining -= 1
    return cur


def nth_workday_after_date(d: datetime, n: int) -> datetime:
    """d 所在日期之后第 n 个工作日（返回该日 23:59 用）。先跳到下一日再计数。"""
    cur = d.replace(hour=0, minute=0, second=0, microsecond=0)
    remaining = n
    while remaining > 0:
        cur += timedelta(days=1)
        if is_workday(cur):
            remaining -= 1
    return cur


def last_workday_of_month(year: int, month: int) -> datetime:
    if month == 12:
        nxt = datetime(year + 1, 1, 1)
    else:
        nxt = datetime(year, month + 1, 1)
    d = nxt - timedelta(days=1)
    while not is_workday(d):
        d -= timedelta(days=1)
    return d


def in_maintenance(dt: datetime) -> bool:
    """第3条：周日23:00–周一06:00。"""
    wd = dt.weekday()  # 0=Mon..6=Sun
    if wd == 6 and dt.hour >= 23:
        return True
    if wd == 0 and dt.hour < 6:
        return True
    return False


def in_cert_batch(dt: datetime) -> bool:
    """第31条：每月最后工作日17:00–22:00。"""
    lw = last_workday_of_month(dt.year, dt.month)
    if dt.date() != lw.date():
        return False
    return 17 <= dt.hour < 22 or (dt.hour == 22 and dt.minute == 0)


def slots_conflict(slots_a: List[Any], slots_b: List[Any]) -> Optional[Any]:
    """两条上课时间线存在交集则返回冲突的 (weekday, start, end) 对，否则 None。"""
    for a in slots_a:
        for b in slots_b:
            if a.weekday == b.weekday and a.start < b.end and b.start < a.end:
                return (a, b)
    return None


def next_table_id(table: Dict[str, Any], key_field: str, prefix: str, width: int) -> str:
    """确定性主键生成：现有同前缀行最大序号+1（重放同初态→同 id）。"""
    max_n = 0
    pat = re.compile(rf"^{re.escape(prefix)}-?(\d+)$")
    for pk in table.keys():
        m = pat.match(pk)
        if m:
            max_n = max(max_n, int(m.group(1)))
    return f"{prefix}-{max_n + 1:0{width}d}"


# --------------------------------------------------------------- 返回视图模型（签名落定；
# 列表型返回统一包 server_time 首字段对象）


class StudentDetails(BaseModelNoExtra):
    server_time: str
    student_id: str
    name: str
    college: str
    major: str
    degree_class: str
    enrollment_year: int
    student_status: str
    is_graduating_cohort: bool
    expected_graduation_term: Optional[str] = None
    gpa: float
    aid_pool_status: str
    aid_pool_valid_through: Optional[str] = None
    warning_records: List[Dict[str, str]]
    disciplinary_records: List[Dict[str, str]]
    major_change_used: bool
    temp_aid_used_this_year: bool
    waitlist_abandon_count: int
    credit_limit_override: bool


class OfferingInfo(BaseModelNoExtra):
    offering_id: str
    course_id: str
    course_name: str
    term: str
    instructor: str
    classroom: str
    credits: float
    course_type: str
    prerequisite_course_ids: List[str]
    session_slots: List[Dict[str, Any]]
    capacity: int
    enrolled_count: int
    seats_left: int
    waitlist_count: int
    status: str
    start_date: str
    adddrop_deadline: str
    term_end_date: str


class OfferingsResult(BaseModelNoExtra):
    server_time: str
    offerings: List[OfferingInfo]


class EnrollmentInfo(BaseModelNoExtra):
    enrollment_id: str
    offering_id: str
    course_name: str
    term: str
    status: str
    source: str
    enrolled_at: str
    dropped_at: Optional[str] = None
    drop_channel: Optional[str] = None
    waitlist_position: Optional[int] = None
    promote_sig_id: Optional[str] = None


class EnrollmentsResult(BaseModelNoExtra):
    server_time: str
    enrollments: List[EnrollmentInfo]


class GradeInfo(BaseModelNoExtra):
    grade_id: str
    offering_id: str
    course_name: str
    term: str
    academic_year: str
    attempt_no: int
    score: Optional[float] = None
    grade_level: str
    gpa_points: Optional[float] = None
    is_final: bool
    recorded_at: Optional[str] = None
    is_retake: bool = False
    replaces_grade_id: Optional[str] = None
    note: str = ""


class GradesResult(BaseModelNoExtra):
    server_time: str
    grades: List[GradeInfo]


class ExamInfo(BaseModelNoExtra):
    exam_id: str
    offering_id: str
    course_name: str
    term: str
    exam_type: str
    scheduled_at: str
    duration_min: int
    location: str
    seat_no: Optional[str] = None
    published: bool
    status: str


class ExamsResult(BaseModelNoExtra):
    server_time: str
    exams: List[ExamInfo]


class RequestInfo(BaseModelNoExtra):
    request_id: str
    kind: str
    status: str
    stage: str
    deadline_at: Optional[str] = None
    reject_reason: Optional[str] = None
    missing: List[str] = []
    sig_id: Optional[str] = None
    detail: str = ""


class RequestsResult(BaseModelNoExtra):
    server_time: str
    requests: List[RequestInfo]


class PolicyClause(BaseModelNoExtra):
    chapter: str
    clause_no: int
    title: str
    text: str


class PolicyResult(BaseModelNoExtra):
    server_time: str
    clauses: List[PolicyClause]


class EnrollResult(BaseModelNoExtra):
    server_time: str
    enrollment_id: str
    offering_id: str
    course_name: str
    status: str
    enrolled_at: str
    message: str


class DropResult(BaseModelNoExtra):
    server_time: str
    enrollment_id: str
    status: str
    drop_channel: Optional[str] = None
    special_sig_id: Optional[str] = None
    message: str


class WaitlistResult(BaseModelNoExtra):
    server_time: str
    enrollment_id: str
    offering_id: str
    status: str
    waitlist_position: int
    message: str


class DeferralResult(BaseModelNoExtra):
    server_time: str
    request_id: str
    status: str
    sig_id: str
    todo_ids: List[str]
    deadline_at: str
    message: str


class AwardAppResult(BaseModelNoExtra):
    server_time: str
    app_id: str
    award_name: str
    amount: float
    status: str
    sig_id: str
    message: str


class CertResult(BaseModelNoExtra):
    server_time: str
    cert_id: str
    status: str
    ready_at: Optional[str] = None
    sig_id: Optional[str] = None
    message: str


class TicketResult(BaseModelNoExtra):
    server_time: str
    ticket_id: str
    status: str
    level: str
    promised_reply_at: str
    message: str


class WithdrawResult(BaseModelNoExtra):
    server_time: str
    record_id: str
    kind: str
    new_status: str
    message: str


# --------------------------------------------------------------- Toolkit


class CampusTools(ToolKitBase):
    """Agent-side tools for the campus domain. Reads/writes CampusDB (`db`) and may
    GENERATE rows in UserDB (`user_db`) for signature/todo hand-offs — the dual-control
    boundary forbids flipping pending_signatures.status /
    uploads.status here."""

    db: CampusDB
    user_db: UserDB

    def __init__(self, db: CampusDB, user_db: Optional[UserDB] = None) -> None:
        super().__init__(db)
        self.user_db = user_db if user_db is not None else UserDB()

    def use_tool(self, tool_name: str, **kwargs):
        """先结算后变更（pre-settle）+ 执行后结算。

        - pre-settle：任何调用（含只读）执行前先把状态机收敛到当前 current_time——
          "变更永远发生在已结算状态之上"。选课计数由
          `_recount_enrolled` 权威重算（settle 内执行），pre-settle 保证该重算在
          每次分支判断/写入前已反映，杜绝基于陈旧计数的判断（TOCTOU 类缺口）。
        - 执行后 settle：评测金标重放走 make_tool_call→use_tool 且不
          额外调 sync_tools（evaluator_env 循环），故结算必须挂在工具执行路径上。
        - 只读调用同样触发两侧（settle 是状态/时间的纯函数，幂等，多调无害）。
        - 现役冻结时间下 pre-settle 恒为 no-op（set_state 末尾与上一次调用的后置
          settle 已收敛状态，两次调用间 current_time 不变）——由 50 题金标重放
          逐题哈希与基线逐位一致证明。"""
        self.settle()
        resp = super().use_tool(tool_name, **kwargs)
        self.settle()
        return resp

    # ------------------------------------------------- 内部工具函数（非 @is_tool）

    @property
    def now(self) -> datetime:
        return parse_time(self.db.env.current_time)

    def _server_time(self) -> str:
        return self.db.env.current_time

    def _check_maintenance(self) -> None:
        """维护窗口闸门：所有 WRITE 先查。"""
        if in_maintenance(self.now):
            raise ValueError(E_MAINTENANCE)

    def _student(self, student_id: str) -> StudentRow:
        if student_id not in self.db.students:
            raise ValueError(f"未找到学号 {student_id} 的学生（请核对学号，政策第35条要求实名注明学号）。")
        return self.db.students[student_id]

    def _offering(self, offering_id: str) -> OfferingRow:
        if offering_id not in self.db.course_offerings:
            raise ValueError(f"未找到开课记录 {offering_id}。")
        return self.db.course_offerings[offering_id]

    def _course_name(self, offering: OfferingRow) -> str:
        course = self.db.courses.get(offering.course_id)
        return course.name if course else offering.course_id

    def _enrolled_students(self, student_id: str, term: str) -> List[EnrollmentRow]:
        return [
            e for e in self.db.enrollments.values()
            if e.student_id == student_id and e.term == term and e.status == EnrollmentStatus.ENROLLED.value
        ]

    def _grade_passed(self, student_id: str, course_id: str) -> bool:
        """先修合格：该课程存在 grade_level ∈ {A..D} 的成绩行（第6/17条，D及以上获学分）。"""
        for g in self.db.grades.values():
            if g.student_id != student_id:
                continue
            off = self.db.course_offerings.get(g.offering_id)
            if off and off.course_id == course_id and g.grade_level in ("A", "B+", "B", "C+", "C", "D"):
                return True
        return False

    def _prereq_in_progress(self, student_id: str, course_id: str) -> bool:
        """先修在修（第6条视同满足）：任一无终行成绩的 已选 选课指向该课程。"""
        for e in self.db.enrollments.values():
            if e.student_id != student_id or e.status != EnrollmentStatus.ENROLLED.value:
                continue
            off = self.db.course_offerings.get(e.offering_id)
            if off and off.course_id == course_id:
                return True
        return False

    def _sig_id_gen(self) -> str:
        return next_table_id(self.user_db.pending_signatures, "sig_id", "SIG", 3)

    def _todo_id_gen(self) -> str:
        return next_table_id(self.user_db.app_todos, "todo_id", "TODO", 3)

    def _make_sig(self, student_id: str, doc_type: SigDocType, ref_type: str, ref_id: str,
                  summary: str, deadline_at: Optional[str] = None) -> str:
        """生成待签署单（Agent 只能生成行，不能翻状态——双控硬边界）。"""
        sid = self._sig_id_gen()
        row = SignatureRow(
            sig_id=sid, student_id=student_id, doc_type=doc_type,
            ref_type=ref_type, ref_id=ref_id, summary=summary,
            status=SigStatus.PENDING, created_at=self._server_time(),
            deadline_at=deadline_at, acted_at=None,
        )
        self.user_db.pending_signatures[sid] = row
        return sid

    def _make_todo(self, student_id: str, todo_type: TodoType, ref_type: str, ref_id: str,
                   title: str, deadline_at: Optional[str] = None) -> str:
        tid = self._todo_id_gen()
        row = TodoRow(
            todo_id=tid, student_id=student_id, type=todo_type, ref_type=ref_type,
            ref_id=ref_id, title=title, status=TodoStatus.PENDING,
            created_at=self._server_time(), deadline_at=deadline_at, completed_at=None,
        )
        self.user_db.app_todos[tid] = row
        return tid

    def _expire_open_signatures(self, ref_type: str, ref_id: str) -> List[str]:
        """撤回时失效关联待签署单/待办（SIG/TODO 枚举无"已撤回"，用"已过期"作废）。"""
        done: List[str] = []
        for sig in self.user_db.pending_signatures.values():
            if sig.ref_type == ref_type and sig.ref_id == ref_id and sig.status == SigStatus.PENDING:
                sig.status = SigStatus.EXPIRED
                sig.acted_at = self._server_time()
                done.append(sig.sig_id)
        for todo in self.user_db.app_todos.values():
            if todo.ref_type == ref_type and todo.ref_id == ref_id and todo.status == TodoStatus.PENDING:
                todo.status = TodoStatus.EXPIRED
                done.append(todo.todo_id)
        return done

    # ------------------------------------------------- 时间钩子（非工具，不进轨迹）

    def advance_time(self, hours: int = 0, days: int = 0) -> str:
        """环境快进钩子（经 initial_state.initialization_actions
        的 EnvFunctionCall 调用，本身不是工具动作、不进用户轨迹）。推进后由
        sync_tools 结算规则化终态。"""
        self.db.env.current_time = fmt_time(self.now + timedelta(hours=hours, days=days))
        return self.db.env.current_time

    # ------------------------------------------------- 候补递补（§5 链4 + 第8条）

    def _promote_next_waitlist(self, offering_id: str) -> Optional[str]:
        """有空位时按队列顺位发起递补（生成 SIG 24h 确认单）。返回 SIG id 或 None。"""
        off = self._offering(offering_id)
        if off.status == OfferingStatus.SUSPENDED.value or off.enrolled_count >= off.capacity:
            return None
        waitlisted = [
            e for e in self.db.enrollments.values()
            if e.offering_id == offering_id and e.status == EnrollmentStatus.WAITLISTED.value
        ]
        # 队列前序尚有未过期确认单 → 本位不抢跑（第8条按队列顺序自动递补）
        for sig in self.user_db.pending_signatures.values():
            if sig.doc_type == SigDocType.WAITLIST_PROMOTE and sig.status == SigStatus.PENDING:
                ref = self.db.enrollments.get(sig.ref_id)
                if ref is not None and ref.offering_id == offering_id and ref.status == EnrollmentStatus.WAITLISTED.value:
                    return None
        candidates = [e for e in waitlisted if not e.promote_sig_id]
        if not candidates:
            return None
        cand = min(candidates, key=lambda e: (e.waitlist_position if e.waitlist_position is not None else 10 ** 6))
        deadline = fmt_time(self.now + timedelta(hours=24))
        sig_id = self._make_sig(
            cand.student_id, SigDocType.WAITLIST_PROMOTE, "enrollments", cand.enrollment_id,
            summary=f"《{self._course_name(off)}》候补递补确认（24小时内签署，逾期视为放弃，第8条）",
            deadline_at=deadline,
        )
        cand.promote_sig_id = sig_id
        self._make_todo(
            cand.student_id, TodoType.WAITLIST_PROMOTE_CONFIRM, "enrollments", cand.enrollment_id,
            title=f"《{self._course_name(off)}》候补递补确认", deadline_at=deadline,
        )
        return sig_id

    # ------------------------------------------------- 状态机结算（sync_tools 调）

    def settle(self) -> None:
        """确定性结算（全部规则化、零裁判）。只依赖 db 状态与 current_time，
        因此 live 轨迹与评估重放（金标/预测）收敛一致。"""
        now = self.now
        self._settle_deferrals(now)
        self._settle_award_apps(now)
        self._settle_certificates(now)
        self._settle_waitlist_sigs(now)
        self._settle_special_channel()
        self._expire_signatures(now)

    def _settle_deferrals(self, now: datetime) -> None:
        for df in list(self.db.deferral_requests.values()):
            if df.status == DeferralStatus.SUBMITTED.value:
                # §6：材料齐（因病＝诊断证明+病假条双件，第12条）＋签署齐＋额度时限内＝通过；
                # 任一缺＝驳回并给出 reject_reason
                if df.reason_type == DeferralReasonType.ILLNESS.value:
                    uploads = [
                        self.user_db.uploads[u] for u in df.proof_upload_ids if u in self.user_db.uploads
                    ]
                    invalid = any(u.status == "无效材料" for u in uploads)
                    required = {UploadDocType.MEDICAL_DIAGNOSIS.value, UploadDocType.SICK_LEAVE.value}
                    valid_docs = {u.doc_type for u in uploads if u.status in ("已上传", "已核验")}
                    valid = required.issubset(valid_docs) and df.hospital_level in VALID_MEDICAL_LEVELS
                    if invalid:
                        df.status = DeferralStatus.REJECTED.value
                        df.reject_reason = "证明材料无效：其他医疗机构或私人诊所出具的证明视为无效材料（第12条），直接驳回。"
                        df.reviewed_at = self._server_time()
                        continue
                    if not valid:
                        df.status = DeferralStatus.PENDING_MATERIAL.value
                        continue
                df.status = DeferralStatus.APPROVED.value
                df.review_stage = ReviewStage.DONE
                df.reviewed_at = self._server_time()
                for up_id in df.proof_upload_ids:
                    up = self.user_db.uploads.get(up_id)
                    if up is not None and up.status == "已上传":
                        up.status = "已核验"
                        up.reviewer = "系统规则核验"
                exam = self.db.exam_arrangements.get(df.exam_id)
                if exam is not None:
                    exam.status = ExamStatus.DEFERRED
            elif df.status in (DeferralStatus.PENDING_SIGN.value, DeferralStatus.PENDING_MATERIAL.value):
                if df.deadline_at is not None and now > parse_time(df.deadline_at):
                    df.status = DeferralStatus.EXPIRED.value
                    df.reviewed_at = self._server_time()

    def _award_year_ok(self, student: StudentRow, award: Any) -> List[str]:
        """受理即时资格校验（第24/27/28条），返回逐条拒绝原因；空列表＝通过。
        注：gpa_rank/comprehensive_rank 依赖年级排名名单（第29条：口径以教务处当期公布计算表
        为准），数据源不在本域 schema 内，按契约不校验。"""
        reasons: List[str] = []
        elig = award.eligibility
        ay = self._current_academic_year()
        if elig.no_fail_this_year:
            for g in self.db.grades.values():
                if g.student_id != student.student_id or g.academic_year != ay:
                    continue
                if g.grade_level in ("F", "缺"):
                    reasons.append(E_RECORD_UNELIGIBLE.format(cause="不及格记录（含'缺'）"))
                    break
        if elig.no_warning:
            for w in student.warning_records:
                if self._term_to_academic_year(w.term) == ay:
                    reasons.append(E_RECORD_UNELIGIBLE.format(cause="学业警示记录"))
                    break
        if elig.no_discipline and student.disciplinary_records:
            reasons.append(E_RECORD_UNELIGIBLE.format(cause="处分记录"))
        if elig.aid_pool_required:
            if student.aid_pool_status != "已入库" or student.aid_pool_valid_through is None or \
                    not self._aid_valid(student.aid_pool_valid_through, ay):
                reasons.append(E_AID_POOL)
        return reasons

    @staticmethod
    def _term_to_academic_year(term: str) -> str:
        """学期→学年：20xxSP/20xxSU 属于 20xx-1 至 20xx 学年；20xxFA 属于 20xx-20xx+1。"""
        m = re.match(r"^(\d{4})(SP|SU|FA|WI)$", term)
        if not m:
            return term
        y = int(m.group(1))
        if m.group(2) in ("SP", "SU", "WI"):
            return f"{y - 1}-{y}"
        return f"{y}-{y + 1}"

    def _current_academic_year(self) -> str:
        return self._term_to_academic_year(self.db.env.term)

    @staticmethod
    def _aid_valid(valid_through: str, academic_year: str) -> bool:
        try:
            end = int(valid_through.split("-")[1])
            ay_end = int(academic_year.split("-")[1])
        except (ValueError, IndexError):
            return False
        return end >= ay_end

    def _settle_award_apps(self, now: datetime) -> None:
        for app in list(self.db.scholarship_apps.values()):
            if app.status == AwardAppStatus.COLLEGE_REVIEW.value:
                student = self.db.students.get(app.student_id)
                award = self.db.scholarships.get(app.award_id)
                if student is None or award is None:
                    continue
                reasons = self._award_year_ok(student, award)
                if app.channel == AwardChannel.FAST_TRACK.value:
                    uploads = [
                        self.user_db.uploads[u] for u in app.material_upload_ids if u in self.user_db.uploads
                    ]
                    has_material = any(
                        u.status in ("已上传", "已核验") and u.doc_type in ("事故证明", "医疗票据")
                        for u in uploads
                    )
                    invalid = any(u.status == "无效材料" for u in uploads)
                    if invalid:
                        app.status = AwardAppStatus.REJECTED.value
                        app.reject_reason = "材料无效（第12/28条：其他机构证明视为无效材料）。"
                        continue
                    if not has_material:
                        app.status = AwardAppStatus.PENDING_MATERIAL.value
                        continue
                if reasons:
                    app.status = AwardAppStatus.REJECTED.value
                    app.reject_reason = "；".join(sorted(set(reasons)))
                else:
                    app.status = AwardAppStatus.PUBLICIZING.value
                    app.publicized_until = fmt_time(day_end(add_workdays(now, 5)))
                    for up_id in app.material_upload_ids:
                        up = self.user_db.uploads.get(up_id)
                        if up is not None and up.status == "已上传":
                            up.status = "已核验"
                            up.reviewer = "学院受理核验"
            elif app.status == AwardAppStatus.PUBLICIZING.value:
                if app.publicized_until and now >= parse_time(app.publicized_until):
                    app.status = AwardAppStatus.APPROVED.value
            elif app.status == AwardAppStatus.PENDING_MATERIAL.value:
                uploads = [
                    self.user_db.uploads[u] for u in app.material_upload_ids if u in self.user_db.uploads
                ]
                valid = any(
                    u.status in ("已上传", "已核验") and u.doc_type in ("事故证明", "医疗票据")
                    for u in uploads
                )
                if valid:
                    app.status = AwardAppStatus.COLLEGE_REVIEW.value
                else:
                    for u in uploads:
                        if u.status == "无效材料":
                            app.status = AwardAppStatus.REJECTED.value
                            app.reject_reason = "材料无效（第12/28条）。"
                            break

    def _settle_certificates(self, now: datetime) -> None:
        for ce in list(self.db.certificates.values()):
            if ce.status == CertStatus.BATCH_DEFERRED.value:
                lw = last_workday_of_month(now.year, now.month)
                if now.date() > lw.date() or now.hour < 17:
                    ce.status = CertStatus.MAKING.value
            if ce.status == CertStatus.MAKING.value and ce.ready_at and now >= parse_time(ce.ready_at):
                ce.status = CertStatus.READY.value
                ce.verify_code = f"{ce.cert_id.replace('-', '')}-{parse_time(ce.ready_at).strftime('%Y%m%d')}"
                ce.verify_code_expiry = fmt_time(parse_time(ce.ready_at) + timedelta(days=90))

    def _settle_waitlist_sigs(self, now: datetime) -> None:
        """§5 链4：递补确认 24h 逾期＝放弃（EN 失效、abandon+1、顺位递补下一位）。"""
        for sig in list(self.user_db.pending_signatures.values()):
            if sig.doc_type != SigDocType.WAITLIST_PROMOTE or sig.status != SigStatus.PENDING:
                continue
            if sig.deadline_at and now > parse_time(sig.deadline_at):
                sig.status = SigStatus.EXPIRED
                sig.acted_at = self._server_time()
                en = self.db.enrollments.get(sig.ref_id)
                if en is not None and en.status == EnrollmentStatus.WAITLISTED.value:
                    en.status = EnrollmentStatus.VOIDED
                    en.dropped_at = self._server_time()
                    en.waitlist_position = None
                    self._abandon_waitlist(en)

    def _abandon_waitlist(self, en: EnrollmentRow) -> None:
        off = self.db.course_offerings.get(en.offering_id)
        if off is not None:
            off.waitlist_count = max(0, off.waitlist_count - 1)
        student = self.db.students.get(en.student_id)
        if student is not None:
            student.waitlist_abandon_count += 1
        for todo in self.user_db.app_todos.values():
            if todo.ref_type == "enrollments" and todo.ref_id == en.enrollment_id and todo.status == TodoStatus.PENDING:
                todo.status = TodoStatus.EXPIRED
        if off is not None:
            self._promote_next_waitlist(off.offering_id)

    def _recount_enrolled(self, offering_id: str) -> None:
        """权威重算 enrolled_count：以该开课 status=已选 行数为唯一
        口径（data_model.py:391 契约，特别通道审核中行不计入），替代结算分支手工 ±1——
        种子构造与结算分支的三方约定由此自洽（幂等）。"""
        off = self.db.course_offerings[offering_id]
        off.enrolled_count = sum(1 for e in self.db.enrollments.values()
                                 if e.offering_id == offering_id
                                 and e.status == EnrollmentStatus.ENROLLED.value)

    def _settle_special_channel(self) -> None:
        """§5 链3/第9条：毕业班特别通道——学生签署确认后环境按双重程序规则化判定。
        可替代性规则：该课程本学期另有开放且有余量的其它开课＝驳回（有替代安排），
        否则＝通过（已退课，drop_channel=特别通道）。"""
        for en in list(self.db.enrollments.values()):
            if en.status != EnrollmentStatus.SPECIAL_CHANNEL_REVIEW.value:
                continue
            sig = self._special_sig_for(en)
            if sig is None or sig.status == SigStatus.PENDING:
                continue
            off = self.db.course_offerings.get(en.offering_id)
            if sig.status == SigStatus.CONFIRMED:
                if sig.deadline_at and self.now > parse_time(sig.deadline_at):
                    # 终态迁移：确认已受理但签署时限已过。
                    # 原 continue 会让 EN 永久卡在"特别通道审核中"（卡死侧无出口）；
                    # 现迁移为终态：EN 回'已选'（维持原状，计数走 _recount_enrolled
                    # 权威重算，与 REJECTED/EXPIRED 回退分支同构）+ SIG 置'已过期'
                    # （acted_at 保留学生确认时刻，不覆盖）。该分支现役不可达——
                    # D1 确认守卫前置（过期单据不可再确认，pre-settle 亦先置已过期
                    # 双保险），drop_course 新建特别单不带 deadline——
                    # 但按终态完备性必须存在（构造性测试
                    # test_special_channel_confirmed_past_deadline_migrates 见证）。
                    en.status = EnrollmentStatus.ENROLLED
                    en.promote_sig_id = None
                    if off is not None:
                        self._recount_enrolled(off.offering_id)
                    sig.status = SigStatus.EXPIRED
                    continue
                alternative = False
                if off is not None:
                    for other in self.db.course_offerings.values():
                        if other.offering_id == off.offering_id or other.term != off.term:
                            continue
                        if other.course_id != off.course_id:
                            continue
                        if other.status in (OfferingStatus.OPEN.value,) and other.enrolled_count < other.capacity:
                            alternative = True
                en.promote_sig_id = None
                if alternative:
                    en.status = EnrollmentStatus.ENROLLED
                    self._expire_open_signatures("enrollments", en.enrollment_id)
                    if off is not None:
                        self._recount_enrolled(off.offering_id)
                    continue
                en.status = EnrollmentStatus.DROPPED
                en.dropped_at = self._server_time()
                en.drop_channel = DropChannel.SPECIAL
                if off is not None:
                    self._recount_enrolled(off.offering_id)
                    if off.status == OfferingStatus.FULL.value and off.enrolled_count < off.capacity:
                        off.status = OfferingStatus.OPEN
                self._promote_next_waitlist(en.offering_id)
                for todo in self.user_db.app_todos.values():
                    if todo.ref_type == "enrollments" and todo.ref_id == en.enrollment_id and todo.status == TodoStatus.PENDING:
                        todo.status = TodoStatus.DONE
                        todo.completed_at = self._server_time()
            elif sig.status in (SigStatus.REJECTED, SigStatus.EXPIRED):
                en.status = EnrollmentStatus.ENROLLED
                en.promote_sig_id = None
                if off is not None:
                    self._recount_enrolled(off.offering_id)

    def _special_sig_for(self, en: EnrollmentRow) -> Optional[SignatureRow]:
        for sig in self.user_db.pending_signatures.values():
            if sig.doc_type == SigDocType.SPECIAL_CHANNEL and sig.ref_type == "enrollments" and sig.ref_id == en.enrollment_id:
                return sig
        return None

    def _expire_signatures(self, now: datetime) -> None:
        """一般待签署单/待办超截止＝已过期（SIG 有 deadline 的行；特别通道单无 24h 时限，按第9条 10 个工作日窗口由局面造）。"""
        for sig in self.user_db.pending_signatures.values():
            if sig.status == SigStatus.PENDING and sig.deadline_at and now > parse_time(sig.deadline_at):
                if sig.doc_type == SigDocType.WAITLIST_PROMOTE:
                    continue  # 由 _settle_waitlist_sigs 统一处理（含放弃计数与递补）
                sig.status = SigStatus.EXPIRED
                sig.acted_at = self._server_time()
        for todo in self.user_db.app_todos.values():
            if todo.status == TodoStatus.PENDING and todo.deadline_at and now > parse_time(todo.deadline_at):
                todo.status = TodoStatus.EXPIRED

    # ================================================================ READ ×7

    @is_tool(ToolType.READ)
    def get_student_details(self, student_id: str) -> StudentDetails:
        """学生主档：学号、学院专业、学籍状态、毕业班标志、GPA、困难库状态(aid_pool)、
        警示/处分记录、转专业是否已用、临时补助是否已用、本学期候补放弃次数。

        Args:
            student_id: 学号，如 S20230102。
        """
        s = self._student(student_id)
        return StudentDetails(
            server_time=self._server_time(),
            student_id=s.student_id, name=s.name, college=s.college, major=s.major,
            degree_class=s.degree_class, enrollment_year=s.enrollment_year,
            student_status=s.student_status.value if hasattr(s.student_status, "value") else str(s.student_status),
            is_graduating_cohort=s.is_graduating_cohort,
            expected_graduation_term=s.expected_graduation_term, gpa=s.gpa,
            aid_pool_status=s.aid_pool_status.value if hasattr(s.aid_pool_status, "value") else str(s.aid_pool_status),
            aid_pool_valid_through=s.aid_pool_valid_through,
            warning_records=[w.model_dump() for w in s.warning_records],
            disciplinary_records=[d.model_dump() for d in s.disciplinary_records],
            major_change_used=s.major_change_used, temp_aid_used_this_year=s.temp_aid_used_this_year,
            waitlist_abandon_count=s.waitlist_abandon_count,
            credit_limit_override=s.credit_limit_override,
        )

    @is_tool(ToolType.READ)
    def get_course_offerings(self, term: str, keyword: str = "", college: str = "", offering_id: str = "") -> OfferingsResult:
        """开课列表/详情（合并版）：课程名、学分、先修课程名单、上课时间 slots、容量与余量、
        候补人数、start_date 与 adddrop_deadline、status(开放/满员/已停开)。
        offering_id 非空时返回单条详情。

        Args:
            term: 学期，如 2026SP。
            keyword: 课程名关键词（子串匹配），可为空。
            college: 开课学院（子串匹配），可为空。
            offering_id: 开课号 OF-xxx，非空时只返回该条详情。
        """
        result: List[OfferingInfo] = []
        for off in self.db.course_offerings.values():
            if offering_id:
                if off.offering_id != offering_id:
                    continue
            elif off.term != term:
                continue
            course = self.db.courses.get(off.course_id)
            if course is None:
                continue
            if keyword and keyword not in course.name:
                continue
            if college and college not in course.college:
                continue
            result.append(OfferingInfo(
                offering_id=off.offering_id, course_id=off.course_id, course_name=course.name,
                term=off.term, instructor=off.instructor, classroom=off.classroom,
                credits=course.credits, course_type=_s(course.course_type),
                prerequisite_course_ids=list(course.prerequisite_course_ids),
                session_slots=[s.model_dump() for s in off.session_slots],
                capacity=off.capacity, enrolled_count=off.enrolled_count,
                seats_left=max(0, off.capacity - off.enrolled_count), waitlist_count=off.waitlist_count,
                status=_s(off.status), start_date=off.start_date,
                adddrop_deadline=off.adddrop_deadline, term_end_date=off.term_end_date,
            ))
        if offering_id and not result:
            raise ValueError(f"未找到开课记录 {offering_id}。")
        return OfferingsResult(server_time=self._server_time(), offerings=result)

    @is_tool(ToolType.READ)
    def get_enrollments(self, student_id: str, term: str) -> EnrollmentsResult:
        """选课记录：状态(已选/候补中/已退课/失效/先修自动退选)、来源、选退时间、候补位次。

        Args:
            student_id: 学号。
            term: 学期，如 2026SP。
        """
        self._student(student_id)
        items: List[EnrollmentInfo] = []
        for e in self.db.enrollments.values():
            if e.student_id != student_id or e.term != term:
                continue
            off = self.db.course_offerings.get(e.offering_id)
            items.append(EnrollmentInfo(
                enrollment_id=e.enrollment_id, offering_id=e.offering_id,
                course_name=self._course_name(off) if off else e.offering_id,
                term=e.term, status=_s(e.status), source=_s(e.source),
                enrolled_at=e.enrolled_at, dropped_at=e.dropped_at,
                drop_channel=_s(e.drop_channel) if e.drop_channel else None,
                waitlist_position=e.waitlist_position, promote_sig_id=e.promote_sig_id,
            ))
        return EnrollmentsResult(server_time=self._server_time(), enrollments=items)

    @is_tool(ToolType.READ)
    def get_grades(self, student_id: str, academic_year: str = "") -> GradesResult:
        """成绩（含全部 attempt：重修替换前后行都返回并标注关系）：等级、学分、绩点、
        公布时间 recorded_at。'缓'与'缺'按第17/18条语义给出文字注解。

        Args:
            student_id: 学号。
            academic_year: 学年，如 2025-2026；为空返回全部。
        """
        self._student(student_id)
        replaced_by: Dict[str, str] = {}
        for g in self.db.grades.values():
            if g.replaces_grade_id:
                replaced_by[g.replaces_grade_id] = g.grade_id
        items: List[GradeInfo] = []
        for g in self.db.grades.values():
            if g.student_id != student_id:
                continue
            if academic_year and g.academic_year != academic_year:
                continue
            off = self.db.course_offerings.get(g.offering_id)
            course = self.db.courses.get(off.course_id) if off else None
            note = ""
            if g.grade_level == "缓":
                note = "（第13/17条）'缓'为缓考占位，不属于通过成绩；该课程不计入GPA，补缓考通过后按实际考核结果记载。"
            elif g.grade_level == "缺":
                note = "（第18条）'缺'＝放弃课程：计0分、计0绩点并计入不及格学分，可作为学业警示认定依据。"
            elif g.grade_id in replaced_by:
                note = f"（第19条）本行已被重修行 {replaced_by[g.grade_id]} 替换：成绩单显示取最高，但原始记录永久保留，奖助资格与学业警示按原始记录核查。"
            elif g.is_retake and g.replaces_grade_id:
                note = f"（第19条）重修考核行，替换原成绩 {g.replaces_grade_id}；按最高一次记载。"
            items.append(GradeInfo(
                grade_id=g.grade_id, offering_id=g.offering_id,
                course_name=course.name if course else g.offering_id,
                term=g.term, academic_year=g.academic_year, attempt_no=g.attempt_no,
                score=g.score, grade_level=g.grade_level, gpa_points=g.gpa_points,
                is_final=g.is_final, recorded_at=g.recorded_at, is_retake=g.is_retake,
                replaces_grade_id=g.replaces_grade_id, note=note,
            ))
        return GradesResult(server_time=self._server_time(), grades=items)

    @is_tool(ToolType.READ)
    def get_exam_arrangements(self, student_id: str, term: str, exam_type: str = "") -> ExamsResult:
        """考试安排：类型(期末/补考/缓考)、scheduled_at、地点座位、是否已确认。

        Args:
            student_id: 学号。
            term: 学期。
            exam_type: 可选，期末/补考/缓考。
        """
        self._student(student_id)
        my_offerings = {
            e.offering_id for e in self.db.enrollments.values()
            if e.student_id == student_id and e.term == term
            and e.status in (EnrollmentStatus.ENROLLED.value, EnrollmentStatus.SPECIAL_CHANNEL_REVIEW.value)
        }
        items: List[ExamInfo] = []
        for ex in self.db.exam_arrangements.values():
            if ex.offering_id not in my_offerings or ex.term != term:
                continue
            if exam_type and ex.exam_type.value != exam_type:
                continue
            off = self.db.course_offerings.get(ex.offering_id)
            published = ex.published_at is not None and self.now >= parse_time(ex.published_at)
            items.append(ExamInfo(
                exam_id=ex.exam_id, offering_id=ex.offering_id,
                course_name=self._course_name(off) if off else ex.offering_id,
                term=ex.term, exam_type=_s(ex.exam_type),
                scheduled_at=ex.scheduled_at if published else "未公布",
                duration_min=ex.duration_min,
                location=ex.location if published else "未公布",
                seat_no=ex.seat_no if published else None, published=published,
                status=_s(ex.status),
            ))
        return ExamsResult(server_time=self._server_time(), exams=items)

    @is_tool(ToolType.READ)
    def get_service_requests(self, student_id: str, kind: str, request_id: str = "") -> RequestsResult:
        """申请类记录统一进度查询。kind ∈ {deferral, scholarship, certificate, ticket}。
        返回状态机位置、待办缺口（缺材料/缺签署）、时限列(deadline_at)与驳回原因。

        Args:
            student_id: 学号。
            kind: deferral（缓考）/ scholarship（奖助）/ certificate（证明）/ ticket（工单）。
            request_id: 记录号（DF-/APP-/CE-/TK-），可选。
        """
        self._student(student_id)
        items: List[RequestInfo] = []
        if kind == "deferral":
            for df in self.db.deferral_requests.values():
                if df.student_id != student_id or (request_id and df.request_id != request_id):
                    continue
                missing = []
                if df.status == DeferralStatus.PENDING_SIGN.value:
                    missing.append("缺签署（缓考申请单）")
                    if df.reason_type == DeferralReasonType.ILLNESS.value:
                        missing.append("缺材料（诊断证明）")
                elif df.status == DeferralStatus.PENDING_MATERIAL.value:
                    missing.append("缺材料（诊断证明）")
                off = self.db.course_offerings.get(df.offering_id)
                items.append(RequestInfo(
                    request_id=df.request_id, kind=kind, status=df.status, stage=df.review_stage,
                    deadline_at=df.deadline_at, reject_reason=df.reject_reason, missing=missing,
                    detail=f"《{self._course_name(off) if off else df.offering_id}》{_s(df.reason_type)}缓考（{_s(df.filing_type)}）",
                ))
        elif kind == "scholarship":
            for app in self.db.scholarship_apps.values():
                if app.student_id != student_id or (request_id and app.app_id != request_id):
                    continue
                missing = []
                if app.status == AwardAppStatus.PENDING_SIGN.value:
                    missing.append("缺签署（申请表）")
                elif app.status == AwardAppStatus.PENDING_MATERIAL.value:
                    missing.append("缺材料")
                award = self.db.scholarships.get(app.award_id)
                items.append(RequestInfo(
                    request_id=app.app_id, kind=kind, status=app.status, stage="",
                    deadline_at=app.publicized_until, reject_reason=app.reject_reason, missing=missing,
                    sig_id=app.sig_id,
                    detail=f"{award.name if award else app.award_id}（{_s(app.channel)}）",
                ))
        elif kind == "certificate":
            for ce in self.db.certificates.values():
                if ce.student_id != student_id or (request_id and ce.cert_id != request_id):
                    continue
                missing = []
                if ce.status == CertStatus.PENDING_SIGN.value:
                    # 待签署有二因——未签署 vs 已签署缺受托人证件（第33条闭环新组合态）
                    auth_sig = self.user_db.pending_signatures.get(ce.proxy_info.auth_sig_id) if ce.proxy_info else None
                    if auth_sig is not None and auth_sig.status == SigStatus.CONFIRMED.value:
                        missing.append("缺受托人证件影像（第33条：签署已完成，上传有效证件即可进入制作）")
                    else:
                        missing.append("缺签署（代领授权书）")
                items.append(RequestInfo(
                    request_id=ce.cert_id, kind=kind, status=ce.status, stage="",
                    deadline_at=ce.pickup_deadline, reject_reason=None, missing=missing,
                    detail=f"{_s(ce.cert_type)}（{_s(ce.language)}，{_s(ce.delivery)}）",
                ))
        elif kind == "ticket":
            for tk in self.db.tickets.values():
                if tk.student_id != student_id or (request_id and tk.ticket_id != request_id):
                    continue
                items.append(RequestInfo(
                    request_id=tk.ticket_id, kind=kind, status=tk.status, stage=tk.level,
                    deadline_at=tk.promised_reply_at, reject_reason=tk.resolution, missing=[],
                    detail=f"{_s(tk.category)}/{_s(tk.module)}：{tk.title}",
                ))
        else:
            raise ValueError(f"未知 kind：{kind}（可选 deferral / scholarship / certificate / ticket）。")
        return RequestsResult(server_time=self._server_time(), requests=items)

    @is_tool(ToolType.READ)
    def search_policy(self, query: str, section: str = "") -> PolicyResult:
        """政策检索：对 policy.md 做确定性关键词/章节检索（无 LLM），返回条款原文引用
        （第 X 条（标题）+全文）。

        Args:
            query: 关键词（子串匹配条款标题与正文），可为空（配合 section 返回整章）。
            section: 章名过滤，可选：总则/选课/考试/成绩/学籍/奖助/证明/申诉。
        """
        from tau2.domains.campus.utils import CAMPUS_POLICY_PATH

        text = CAMPUS_POLICY_PATH.read_text(encoding="utf-8")
        clauses = self._parse_policy(text)
        out: List[PolicyClause] = []
        for c in clauses:
            if section and section not in c.chapter:
                continue
            if query and (query not in c.title and query not in c.text):
                continue
            out.append(c)
        if query:
            out = out[:12]
        return PolicyResult(server_time=self._server_time(), clauses=out)

    @staticmethod
    def _parse_policy(text: str) -> List[PolicyClause]:
        clause_pat = re.compile(r"\*\*第 (\d+) 条（([^）]*)）\*\* (.*?)(?=\n\n\*\*第 |\n\n---|\Z)", re.S)
        result: List[PolicyClause] = []
        for ch in re.finditer(r"## (第[一二三四五六七]章 [^\n]+)", text):
            chapter = ch.group(1).split(" ", 1)[1] if " " in ch.group(1) else ch.group(1)
            seg_start = ch.end()
            nxt = text.find("\n## ", seg_start)
            seg_end = nxt if nxt != -1 else len(text)
            for m in clause_pat.finditer(text[seg_start:seg_end]):
                result.append(PolicyClause(
                    chapter=chapter, clause_no=int(m.group(1)), title=m.group(2),
                    text=m.group(3).replace("\n", ""),
                ))
        return result

    # ================================================================ WRITE ×8（含 T7 withdraw_application）

    @is_tool(ToolType.WRITE)
    def enroll_course(self, student_id: str, offering_id: str) -> EnrollResult:
        """补选/加课。校验顺序：学籍→窗口→先修→名额→时间冲突→学分上限。成功即时生效，不产生签署单。

        Args:
            student_id: 学号。
            offering_id: 开课号 OF-xxx。
        """
        self._check_maintenance()
        s = self._student(student_id)
        # ① 学籍＝在读（第10条）
        if s.student_status != StudentStatus.ACTIVE.value:
            raise ValueError(f"当前学籍状态为'{s.student_status}'（政策第10条）：非在读状态不享受选课与考试服务。")
        off = self._offering(offering_id)
        course = self.db.courses.get(off.course_id)
        if course is None:
            raise ValueError(f"未找到课程目录 {off.course_id}。")
        if off.status == OfferingStatus.SUSPENDED.value:
            raise ValueError(f"《{course.name}》本学期已停开（第9条相关）：不可补选。")
        # ② 开课后14日内（第7条）
        if self.now > parse_time(off.adddrop_deadline):
            raise ValueError(E_WINDOW_CLOSED.format(deadline=off.adddrop_deadline, action="选课"))
        # ③ 先修合格或在修（第6条）
        for pre in course.prerequisite_course_ids:
            if not (self._grade_passed(student_id, pre) or self._prereq_in_progress(student_id, pre)):
                pre_course = self.db.courses.get(pre)
                raise ValueError(E_PREREQ_MISSING.format(course=course.name, prereq=pre_course.name if pre_course else pre))
        # ④ 名额（第8条：满则候补）
        if off.enrolled_count >= off.capacity:
            raise ValueError(E_CAPACITY_FULL)
        # ⑤ 上课时间冲突（第5/7条语境，slots 交集）
        for e in self._enrolled_students(student_id, off.term):
            other = self.db.course_offerings.get(e.offering_id)
            if other is None or other.offering_id == off.offering_id:
                continue
            hit = slots_conflict(off.session_slots, other.session_slots)
            if hit is not None:
                a, b = hit
                other_course = self.db.courses.get(other.course_id)
                raise ValueError(E_TIME_CONFLICT.format(
                    other=other_course.name if other_course else other.offering_id,
                    w=WEEKDAY_CN[a.weekday - 1], t1=a.start, t2=a.end))
        # ⑥ 总学分≤32（第5条）
        if not s.credit_limit_override:
            total = 0.0
            for e in self._enrolled_students(student_id, off.term):
                other = self.db.course_offerings.get(e.offering_id)
                oc = self.db.courses.get(other.course_id) if other else None
                if oc:
                    total += oc.credits
            if total + course.credits > 32:
                raise ValueError(f"本学期已选 {total:g} 学分，加《{course.name}》（{course.credits:g} 学分）将超过 32 学分上限（政策第5条）；请先取得学院教学副院长书面批准。")
        # 重复选课检查
        for e in self.db.enrollments.values():
            if e.student_id == student_id and e.offering_id == offering_id and e.status in (
                    EnrollmentStatus.ENROLLED.value, EnrollmentStatus.WAITLISTED.value,
                    EnrollmentStatus.SPECIAL_CHANNEL_REVIEW.value):
                raise ValueError(f"《{course.name}》已有选课/候补记录（状态：{_s(e.status)}），请勿重复选课。")
        en_id = next_table_id(self.db.enrollments, "enrollment_id", "EN", 4)
        self.db.enrollments[en_id] = EnrollmentRow(
            enrollment_id=en_id, student_id=student_id, offering_id=offering_id, term=off.term,
            status=EnrollmentStatus.ENROLLED, source=EnrollmentSource.ADD_DROP,
            enrolled_at=self._server_time(),
        )
        off.enrolled_count += 1
        if off.enrolled_count >= off.capacity:
            off.status = OfferingStatus.FULL
        return EnrollResult(
            server_time=self._server_time(), enrollment_id=en_id, offering_id=offering_id,
            course_name=course.name, status=EnrollmentStatus.ENROLLED.value,
            enrolled_at=self._server_time(),
            message=f"《{course.name}》补选成功（{course.credits:g} 学分），即时生效（第7条）。",
        )

    @is_tool(ToolType.WRITE)
    def drop_course(self, student_id: str, enrollment_id: str, reason: str = "") -> DropResult:
        """退课。窗口内即时生效；窗口外毕业班走特别通道（SIG 审批单+学生确认+双重程序）；窗口外非毕业班拒绝。

        Args:
            student_id: 学号。
            enrollment_id: 选课记录号 EN-xxxx。
            reason: 退课原因（供记录）。
        """
        self._check_maintenance()
        s = self._student(student_id)
        en = self.db.enrollments.get(enrollment_id)
        if en is None or en.student_id != student_id:
            raise ValueError(f"未找到选课记录 {enrollment_id}（或不属于该学号）。")
        if en.status not in (EnrollmentStatus.ENROLLED.value, EnrollmentStatus.SPECIAL_CHANNEL_REVIEW.value):
            raise ValueError(f"选课记录当前状态为'{_s(en.status)}'，不可退课。")
        if en.status == EnrollmentStatus.SPECIAL_CHANNEL_REVIEW.value:
            raise ValueError(E_APPROVAL_PENDING)
        off = self._offering(en.offering_id)
        course_name = self._course_name(off)
        if self.now <= parse_time(off.adddrop_deadline):
            en.status = EnrollmentStatus.DROPPED
            en.dropped_at = self._server_time()
            en.drop_channel = DropChannel.NORMAL
            off.enrolled_count = max(0, off.enrolled_count - 1)
            if off.status == OfferingStatus.FULL.value:
                off.status = OfferingStatus.OPEN
            sig = self._promote_next_waitlist(off.offering_id)
            extra = f"已按队列顺位向候补第1位发起递补确认（{sig}，24小时内签署）。" if sig else ""
            return DropResult(
                server_time=self._server_time(), enrollment_id=enrollment_id,
                status=en.status.value, drop_channel=en.drop_channel.value,
                message=f"《{course_name}》退课即时生效（第7条：窗口内无需审批）。{extra}",
            )
        # 窗口外：第9条 毕业班特别通道
        if s.is_graduating_cohort:
            # 第9条提交窗——补退选窗口关闭后 10 个工作日内，超窗拒收
            window_end = day_end(nth_workday_after_date(parse_time(off.adddrop_deadline), 10))
            if self.now > window_end:
                raise ValueError(f"特别退改申请须于补退选窗口关闭后 10 个工作日内提交"
                                 f"（第9条，截止 {fmt_time(window_end)} 前）；已超窗，无法办理。")
            sig_id = self._make_sig(
                student_id, SigDocType.SPECIAL_CHANNEL, "enrollments", enrollment_id,
                summary=f"《{course_name}》毕业班特别通道审批单（学院→教务处双重程序，第9条）",
            )
            en.status = EnrollmentStatus.SPECIAL_CHANNEL_REVIEW
            # 已知契约缺口：TodoType 无"特别通道确认"枚举值，本通道仅生成 SIG（不生成 TODO），
            # 学生经 check_student_app 待签署列表可见。
            return DropResult(
                server_time=self._server_time(), enrollment_id=enrollment_id,
                status=en.status.value, special_sig_id=sig_id,
                message=f"补退选窗口已过；已按第9条为毕业班学生发起特别通道审批（{sig_id}），"
                        f"需本人在小程序签署确认后进入学院/教务处双重审核。",
            )
        raise ValueError(E_WINDOW_CLOSED.format(deadline=off.adddrop_deadline, action="退课"))

    @is_tool(ToolType.WRITE)
    def join_waitlist(self, student_id: str, offering_id: str) -> WaitlistResult:
        """加入候补（第8条）。前置：名额已满、窗口关闭前48h内禁入、候补冻结检查。

        Args:
            student_id: 学号。
            offering_id: 开课号 OF-xxx。
        """
        self._check_maintenance()
        s = self._student(student_id)
        # ① 学籍＝在读（第10条：候补同享选课服务约束）
        if s.student_status != StudentStatus.ACTIVE.value:
            raise ValueError(f"当前学籍状态为'{_s(s.student_status)}'（政策第10条）：非在读状态不享受选课服务（含候补）。")
        off = self._offering(offering_id)
        course_name = self._course_name(off)
        # ② 已停开不可候补（第9条相关）
        if off.status == OfferingStatus.SUSPENDED.value:
            raise ValueError(f"《{course_name}》本学期已停开（第9条相关）：不可加入候补。")
        if s.waitlist_abandon_count >= 3:
            raise ValueError(E_WAITLIST_FROZEN)
        deadline_48h = parse_time(off.adddrop_deadline) - timedelta(hours=48)
        if self.now > deadline_48h:
            raise ValueError(f"《{course_name}》候补申请已于 {fmt_time(day_end(deadline_48h))} 截止"
                             f"（补退选窗口关闭前48小时，政策第8条），不再受理。")
        if off.enrolled_count < off.capacity:
            raise ValueError(f"《{course_name}》尚有余位，请直接选课（第8条：候补仅适用于名额已满课程）。")
        for e in self.db.enrollments.values():
            if e.student_id == student_id and e.offering_id == offering_id and e.status in (
                    EnrollmentStatus.ENROLLED.value, EnrollmentStatus.WAITLISTED.value):
                raise ValueError(f"《{course_name}》已有选课/候补记录（状态：{_s(e.status)}）。")
        # 位次单调（第8条队列顺位）：取该开课活跃候补行的最大位次+1——
        # 旧公式 waitlist_count+1 在前位放弃（计数-1）后重入会复用已占位次
        # （如 A=1/B=2，A 弃后 C 得 2 与 B 撞号）；max+1 保证位次只增不重复，
        # 且 C 恒排在存活的 B 之后（顺位正确）。空队列 default=0 → 首位 1。
        active_positions = [
            e.waitlist_position for e in self.db.enrollments.values()
            if e.offering_id == offering_id
            and e.status == EnrollmentStatus.WAITLISTED.value
            and e.waitlist_position is not None
        ]
        pos = max(active_positions, default=0) + 1
        en_id = next_table_id(self.db.enrollments, "enrollment_id", "EN", 4)
        self.db.enrollments[en_id] = EnrollmentRow(
            enrollment_id=en_id, student_id=student_id, offering_id=offering_id, term=off.term,
            status=EnrollmentStatus.WAITLISTED, source=EnrollmentSource.ADD_DROP,
            enrolled_at=self._server_time(), waitlist_position=pos,
        )
        off.waitlist_count += 1
        return WaitlistResult(
            server_time=self._server_time(), enrollment_id=en_id, offering_id=offering_id,
            status=EnrollmentStatus.WAITLISTED.value, waitlist_position=pos,
            message=f"已加入《{course_name}》候补队列，位次 {pos}（第8条；递补后须24小时内确认）。",
        )

    @is_tool(ToolType.WRITE)
    def submit_deferral(self, student_id: str, offering_id: str, exam_id: str,
                        reason_type: str, filing_type: str) -> DeferralResult:
        """缓考申请（第12/13条）。冲突限考前正常申报；因病限考后3个工作日内；同课程本学期仅一次（已撤回不占额度）。

        Args:
            student_id: 学号。
            offering_id: 开课号 OF-xxx。
            exam_id: 考试号 EX-xxxx（须属于该开课）。
            reason_type: 冲突 / 因病。
            filing_type: 考前正常 / 考后补办。
        """
        self._check_maintenance()
        self._student(student_id)
        exam = self.db.exam_arrangements.get(exam_id)
        if exam is None:
            raise ValueError(f"未找到考试安排 {exam_id}。")
        if exam.offering_id != offering_id:
            raise ValueError(f"考试 {exam_id} 不属于开课 {offering_id}，请核对考试与课程对应关系。")
        off = self._offering(offering_id)
        course_name = self._course_name(off)
        # 缓考绑定本人在该开课的在读选课行（政策第12/10条）
        if not any(
                e.student_id == student_id and e.offering_id == offering_id
                and e.status in (EnrollmentStatus.ENROLLED.value, EnrollmentStatus.SPECIAL_CHANNEL_REVIEW.value)
                for e in self.db.enrollments.values()):
            raise ValueError("未找到该课程的在读选课记录，无法申请缓考（政策第12/10条）：缓考仅适用于本人已选课程。")
        sched = parse_time(exam.scheduled_at)
        if reason_type == DeferralReasonType.CONFLICT.value:
            if filing_type != DeferralFilingType.BEFORE_EXAM.value:
                raise ValueError("冲突缓考须于考试日前一日23:59前通过系统提交（第12条一），考试当日或考后提出的冲突理由申请一律不受理。")
            deadline = day_end(sched - timedelta(days=1))
            if self.now > deadline:
                raise ValueError(E_DEFERRAL_EXPIRED)
        elif reason_type == DeferralReasonType.ILLNESS.value:
            # 因病仅考后补办（第12条二）
            if filing_type != DeferralFilingType.AFTER_EXAM.value:
                raise ValueError("因病缓考属考后补办（第12条二）：须于考试结束后3个工作日内补办；考前申报仅适用于冲突缓考。")
            deadline = day_end(nth_workday_after_date(sched, 3))
            if self.now > deadline:
                raise ValueError(E_DEFERRAL_EXPIRED)
        else:
            raise ValueError("缓考理由只能为：冲突 / 因病（第12条）。")
        # 同课程本学期缓考仅一次（"已撤回"行不占额度）
        for df in self.db.deferral_requests.values():
            if df.student_id == student_id and df.offering_id == offering_id and df.status != DeferralStatus.WITHDRAWN.value:
                raise ValueError(E_DEFERRAL_USED.format(course=course_name))
        df_id = next_table_id(self.db.deferral_requests, "request_id", "DF", 3)
        self.db.deferral_requests[df_id] = DeferralRow(
            request_id=df_id, student_id=student_id, offering_id=offering_id, exam_id=exam_id,
            reason_type=DeferralReasonType(reason_type), filing_type=DeferralFilingType(filing_type),
            proof_upload_ids=[], hospital_level=HospitalLevel.NONE,
            status=DeferralStatus.PENDING_SIGN, review_stage=ReviewStage.COLLEGE,
            submitted_at=self._server_time(), deadline_at=fmt_time(deadline),
        )
        sig_id = self._make_sig(
            student_id, SigDocType.DEFERRAL_APP, "deferral_requests", df_id,
            summary=f"《{course_name}》缓考申请单（{reason_type}，第12/13条）",
            deadline_at=fmt_time(deadline),
        )
        todo_ids = [self._make_todo(
            student_id, TodoType.DEFERRAL_SIGN, "deferral_requests", df_id,
            title=f"《{course_name}》缓考申请签署", deadline_at=fmt_time(deadline),
        )]
        if reason_type == DeferralReasonType.ILLNESS.value:
            todo_ids.append(self._make_todo(
                student_id, TodoType.DEFERRAL_UPLOAD, "deferral_requests", df_id,
                title=f"《{course_name}》缓考材料上传（三甲/校医院指定门诊诊断证明，第12条）",
                deadline_at=fmt_time(deadline),
            ))
        return DeferralResult(
            server_time=self._server_time(), request_id=df_id,
            status=DeferralStatus.PENDING_SIGN.value, sig_id=sig_id, todo_ids=todo_ids,
            deadline_at=fmt_time(deadline),
            message=f"《{course_name}》缓考申请已提交待签署（{df_id}）；须由本人在小程序完成签署"
                    + ("及诊断证明上传" if reason_type == DeferralReasonType.ILLNESS.value else "")
                    + f"方进入审核（截止 {fmt_time(deadline)}）。",
        )

    @is_tool(ToolType.WRITE)
    def submit_scholarship_app(self, student_id: str, award_id: str, channel: str = "常规") -> AwardAppResult:
        """奖助申请（第23–28条）。受理即时资格校验，任一不符即逐条拒绝（拒绝不落库）；通过置'待签署'+SIG。

        Args:
            student_id: 学号。
            award_id: 奖项目录号 AW-xxx。
            channel: 常规 / 突发快速通道（第28条，临时困难补助专用）。
        """
        self._check_maintenance()
        s = self._student(student_id)
        award = self.db.scholarships.get(award_id)
        if award is None:
            raise ValueError(f"未找到奖项目录 {award_id}。")
        if channel not in (AwardChannel.REGULAR.value, AwardChannel.FAST_TRACK.value):
            raise ValueError("申请通道只能为：常规 / 突发快速通道。")
        reasons: List[str] = []
        if award.eligibility.single_use_year:
            if channel != AwardChannel.FAST_TRACK.value:
                reasons.append(f"临时困难补助须以'突发快速通道'申请（第28条）。")
            if s.temp_aid_used_this_year:
                reasons.append("临时困难补助本学年已使用（第28条：该通道每学年限一次）。")
        elif channel == AwardChannel.FAST_TRACK.value:
            reasons.append("'突发快速通道'仅适用于临时困难补助（第28条）。")
        reasons.extend(self._award_year_ok(s, award))
        # 兼得冲突拦截（第25条）+ 重复申请检查
        for other in self.db.scholarship_apps.values():
            if other.student_id != student_id:
                continue
            if other.award_id == award_id:
                if other.status in ("待签署", "待材料", "资格校验中", "待学院审", "评审中", "公示中", "通过"):
                    reasons.append(f"已存在该项目的申请 {other.app_id}（状态：{_s(other.status)}），请勿重复申请。")
                continue
            if other.status in ("待签署", "待材料", "资格校验中", "待学院审", "评审中", "公示中", "通过"):
                oa = self.db.scholarships.get(other.award_id)
                if oa is None:
                    continue
                if award_id in oa.stack_conflicts or other.award_id in award.stack_conflicts:
                    reasons.append(E_STACK_CONFLICT.format(award=oa.name))
        if reasons:
            raise ValueError("申请不符合条件：" + " ".join(sorted(set(reasons))))
        ay = self._current_academic_year()
        app_id = next_table_id(self.db.scholarship_apps, "app_id", "APP", 3)
        sig_id = self._make_sig(
            student_id, SigDocType.AWARD_APP, "scholarship_apps", app_id,
            summary=f"{award.name}申请表（第26条：本人签署后进入学院受理）",
        )
        todo_id = self._make_todo(
            student_id, TodoType.AWARD_SIGN, "scholarship_apps", app_id,
            title=f"{award.name}申请签署",
        )
        self.db.scholarship_apps[app_id] = AwardAppRow(
            app_id=app_id, student_id=student_id, award_id=award_id, term=self.db.env.term,
            academic_year=ay, channel=AwardChannel(channel), status=AwardAppStatus.PENDING_SIGN,
            sig_id=sig_id, submitted_at=self._server_time(),
            fast_track_used_this_year=(True if channel == AwardChannel.FAST_TRACK.value else None),
        )
        if channel == AwardChannel.FAST_TRACK.value:
            s.temp_aid_used_this_year = True
        return AwardAppResult(
            server_time=self._server_time(), app_id=app_id, award_name=award.name,
            amount=award.amount, status=AwardAppStatus.PENDING_SIGN.value, sig_id=sig_id,
            message=f"{award.name}（{award.amount:g} 元）申请已提交（{app_id}），待本人小程序签署{todo_id}后进入学院受理与评审公示（第26条）。",
        )

    @is_tool(ToolType.WRITE)
    def request_certificate(self, student_id: str, cert_type: str, language: str = "中",
                            copy_count: int = 1, delivery: str = "电子",
                            proxy_name: str = "", proxy_id_masked: str = "") -> CertResult:
        """证明开具（第30–33条）。月末结账内顺延；代领需授权签署；英文成绩单未公布/缓成绩仅出已记载部分。

        Args:
            student_id: 学号。
            cert_type: 中文成绩单/英文成绩单/在读证明/预计毕业证明/学籍学历证明。
            language: 中 / 英。
            copy_count: 份数（≤5，第32条）。
            delivery: 电子 / 自助打印 / 人工领取 / 委托代领。
            proxy_name: 受托人姓名（委托代领必填）。
            proxy_id_masked: 受托人证件掩码（委托代领必填）。
        """
        self._check_maintenance()
        s = self._student(student_id)
        if copy_count < 1:
            raise ValueError("开具份数至少为 1 份（政策第32条）。")
        if copy_count > 5:
            raise ValueError("同一申请单笔开具份数不超过5份（政策第32条），超出请分单申请。")
        if delivery == CertDelivery.PROXY.value and (not proxy_name or not proxy_id_masked):
            raise ValueError("委托代领须提供受托人姓名与证件掩码（第33条），否则无法生成授权单。")
        ce_id = next_table_id(self.db.certificates, "cert_id", "CE", 3)
        status = CertStatus.MAKING
        ready_base = self.now
        sig_id: Optional[str] = None
        extra_msgs: List[str] = []
        if in_cert_batch(self.now):
            status = CertStatus.BATCH_DEFERRED
            next_day = add_workdays(self.now.replace(hour=0, minute=0), 1)
            ready_base = next_day
            extra_msgs.append(E_CERT_BATCH + f"下一工作日 {fmt_time(next_day)} 起恢复办理。")
        if delivery == CertDelivery.PROXY.value:
            status = CertStatus.PENDING_SIGN
            sig_id = self._make_sig(
                student_id, SigDocType.PROXY_AUTH, "certificates", ce_id,
                summary=f"《{cert_type}》代领授权书（受托人 {proxy_name}，第33条：签署后30日内有效）",
            )
            self._make_todo(student_id, TodoType.CERT_PROXY_AUTH, "certificates", ce_id,
                            title=f"《{cert_type}》代领授权签署")
        self.db.certificates[ce_id] = CertRow(
            cert_id=ce_id, student_id=student_id, cert_type=cert_type, language=language,
            copy_count=copy_count, delivery=delivery, status=status,
            applied_at=self._server_time(),
            ready_at=fmt_time(add_workdays(ready_base, 3)),
            proxy_info=None,
        )
        if delivery == CertDelivery.PROXY.value and sig_id is not None:
            self.db.certificates[ce_id].proxy_info = ProxyInfo(
                proxy_name=proxy_name, id_masked=proxy_id_masked,
                auth_sig_id=sig_id, valid_until=fmt_time(self.now + timedelta(days=30)),
            )
        if language == "英" and cert_type == "英文成绩单":
            unpublished = [
                g for g in self.db.grades.values()
                if g.student_id == student_id and g.grade_level in ("未公布", "缓")
            ]
            if unpublished:
                extra_msgs.append(f"英文成绩单仅含已正式记载成绩（第32条）：{len(unpublished)} 条'未公布/缓'成绩将不出现在英文成绩单中。")
        message = f"证明申请已受理（{ce_id}，{cert_type}）"
        if status == CertStatus.MAKING.value:
            message += f"；预计出具时间 {self.db.certificates[ce_id].ready_at}（第30条：约3个工作日）。"
        elif status == CertStatus.PENDING_SIGN.value:
            message += f"；须本人在小程序签署代领授权（{sig_id}）并上传受托人证件后方可进入制作（第33条）。"
        message += " ".join(extra_msgs)
        return CertResult(
            server_time=self._server_time(), cert_id=ce_id, status=status,
            ready_at=self.db.certificates[ce_id].ready_at, sig_id=sig_id, message=message,
        )

    @is_tool(ToolType.WRITE)
    def create_ticket(self, student_id: str, category: str, module: str, title: str,
                      content: str, parent_ticket_id: str = "",
                      target_grade_id: str = "") -> TicketResult:
        """工单（第34/35条）。仅拦'申诉+成绩+逾期'组合；复核申诉必须挂学院原工单（不得越级）。
        申诉+成绩可选传 target_grade_id 按该成绩判 5 工作日窗（第16条）；
        不传（默认空）时判窗行为与不带该参数完全一致（按"任一成绩在窗"口径）。

        Args:
            student_id: 学号。
            category: 咨询 / 申诉 / 建议。
            module: 选课/考试/成绩/学籍/奖助/证明/其他。
            title: 工单标题。
            content: 工单内容（实名+学号+事由，第35条）。
            parent_ticket_id: 复核申诉时必填的原学院工单号 TK-xxx。
            target_grade_id: 可选，申诉目标成绩行 GR-xxx；指定后仅按该成绩判窗，空＝原行为不变。
                仅作判窗输入、不写入工单行（避免 agent 显式传参使 DB 终态与金标缺省路径哈希分叉）。
        """
        self._check_maintenance()
        self._student(student_id)
        if category not in (TicketCategory.CONSULT.value, TicketCategory.APPEAL.value, TicketCategory.SUGGESTION.value):
            raise ValueError("工单类别只能为：咨询 / 申诉 / 建议（第35条）。")
        # 第16条查分预检：仅"申诉+成绩+逾期"组合拦截
        if category == TicketCategory.APPEAL.value and module == TicketModule.GRADE.value:
            if target_grade_id:
                # 绑定目标成绩时按该行判窗（缺省 target 时下方 any() 扫描为修前原行为）
                grade = self.db.grades.get(target_grade_id)
                if grade is None or grade.student_id != student_id:
                    raise ValueError(f"未找到成绩记录 {target_grade_id}（或不属于该学号）。")
                # 未公布成绩（recorded_at 为空）无查分窗口起点，不作逾期拦截（与原扫描"仅计已公布成绩"口径一致）
                if grade.recorded_at and self.now > day_end(add_workdays(parse_time(grade.recorded_at), 5)):
                    raise ValueError(E_REVIEW_EXPIRED)
            else:
                recorded = [
                    g for g in self.db.grades.values()
                    if g.student_id == student_id and g.recorded_at
                ]
                if recorded:
                    within = any(
                        self.now <= day_end(add_workdays(parse_time(g.recorded_at), 5)) for g in recorded
                    )
                    if not within:
                        raise ValueError(E_REVIEW_EXPIRED)
        level = TicketLevel.COLLEGE
        status = TicketStatus.PENDING if category != TicketCategory.APPEAL.value else TicketStatus.COLLEGE_WORKING
        if parent_ticket_id:
            parent = self.db.tickets.get(parent_ticket_id)
            if (parent is None or parent.student_id != student_id
                    or parent.level != TicketLevel.COLLEGE.value
                    or parent.status not in (TicketStatus.ANSWERED.value, TicketStatus.CLOSED.value)):
                raise ValueError(E_LEVEL)
            level = TicketLevel.ACADEMIC_AFFAIRS
            status = TicketStatus.RECHECKING
        reply_days = {TicketCategory.CONSULT.value: 2, TicketCategory.APPEAL.value: 5, TicketCategory.SUGGESTION.value: 10}
        tk_id = next_table_id(self.db.tickets, "ticket_id", "TK", 3)
        self.db.tickets[tk_id] = TicketRow(
            ticket_id=tk_id, student_id=student_id, category=category, module=module,
            title=title, content=content, status=status, level=level,
            created_at=self._server_time(),
            promised_reply_at=fmt_time(day_end(add_workdays(self.now, reply_days[category]))),
            parent_ticket_id=parent_ticket_id or None,
        )
        return TicketResult(
            server_time=self._server_time(), ticket_id=tk_id, status=status, level=level,
            promised_reply_at=self.db.tickets[tk_id].promised_reply_at,
            message=f"{category}工单已受理（{tk_id}，{level}）；承诺 {self.db.tickets[tk_id].promised_reply_at} 前答复"
                    f"（第35条：咨询2/申诉5/建议10个工作日）。",
        )

    @is_tool(ToolType.WRITE)
    def withdraw_application(self, student_id: str, kind: str, record_id: str) -> WithdrawResult:
        """撤回申请（第13/30条）。
        CE 限出具前撤回，DF/APP 限签署确认前撤回；撤回不占额度/不产生记录。
        SIG 拒签撤回仍走学生侧 reject_suggestion。

        Args:
            student_id: 学号。
            kind: deferral（缓考）/ scholarship（奖助）/ certificate（证明）。
            record_id: 记录号 DF-/APP-/CE-。
        """
        self._check_maintenance()
        self._student(student_id)
        if kind == "deferral":
            row = self.db.deferral_requests.get(record_id)
            if row is None or row.student_id != student_id:
                raise ValueError(f"未找到缓考申请 {record_id}（或不属于该学号）。")
            if row.status != DeferralStatus.PENDING_SIGN.value:
                raise ValueError(f"缓考申请当前状态'{_s(row.status)}'：签署确认前方可主动撤回（第13条），已撤回不计额度。")
            row.status = DeferralStatus.WITHDRAWN
            touched = self._expire_open_signatures("deferral_requests", record_id)
            return WithdrawResult(
                server_time=self._server_time(), record_id=record_id, kind=kind,
                new_status=row.status,
                message=f"缓考申请 {record_id} 已在签署确认前撤回（第13条）：不占本学期该课程缓考额度，可重新申请。关联签署单 {', '.join(touched) or '无'} 已作废。",
            )
        if kind == "scholarship":
            row = self.db.scholarship_apps.get(record_id)
            if row is None or row.student_id != student_id:
                raise ValueError(f"未找到奖助申请 {record_id}（或不属于该学号）。")
            if row.status != AwardAppStatus.PENDING_SIGN.value:
                raise ValueError(f"奖助申请当前状态'{_s(row.status)}'：仅签署确认前可撤回（第26条）。")
            row.status = AwardAppStatus.WITHDRAWN
            touched = self._expire_open_signatures("scholarship_apps", record_id)
            if row.channel == AwardChannel.FAST_TRACK.value:
                student = self._student(student_id)
                student.temp_aid_used_this_year = False
            return WithdrawResult(
                server_time=self._server_time(), record_id=record_id, kind=kind,
                new_status=row.status,
                message=f"奖助申请 {record_id} 已在签署确认前撤回（第26条：学生可撤回并改为申请符合条件的项目）；兼得限制随之解除。",
            )
        if kind == "certificate":
            row = self.db.certificates.get(record_id)
            if row is None or row.student_id != student_id:
                raise ValueError(f"未找到证明申请 {record_id}（或不属于该学号）。")
            if row.status not in (CertStatus.PENDING_SIGN.value, CertStatus.MAKING.value, CertStatus.BATCH_DEFERRED.value):
                raise ValueError(f"证明申请当前状态'{_s(row.status)}'：出具前方可撤回（第30条）；出具后不可撤回，验证期届满需重新开具。")
            row.status = CertStatus.WITHDRAWN
            touched = self._expire_open_signatures("certificates", record_id)
            return WithdrawResult(
                server_time=self._server_time(), record_id=record_id, kind=kind,
                new_status=row.status,
                message=f"证明申请 {record_id} 已在出具前撤回（第30条）：不再出具证明，申请记录留存并标注'已撤回'。",
            )
        raise ValueError("撤回仅支持 kind：deferral / scholarship / certificate（工单请按第34/35条流程处理；"
                         "签署拒签撤回请由学生本人在小程序使用拒签通道）。")

    # ------------------------------------------------- 断言/辅助函数（非工具，供 env_assertions 与测试用）

    def assert_deferral_status(self, request_id: str, expected_status: str) -> bool:
        df = self.db.deferral_requests.get(request_id)
        return df is not None and df.status == expected_status

    def assert_enrollment_status(self, enrollment_id: str, expected_status: str) -> bool:
        en = self.db.enrollments.get(enrollment_id)
        return en is not None and en.status == expected_status

    def assert_ticket_exists(self, student_id: str, title_keyword: str = "") -> bool:
        return any(
            t.student_id == student_id and (not title_keyword or title_keyword in t.title)
            for t in self.db.tickets.values()
        )

    def assert_ticket_count(self, student_id: str, expected: int) -> bool:
        return sum(1 for tk in self.db.tickets.values() if tk.student_id == student_id) == expected

    def assert_certificate_count(self, student_id: str, expected: int) -> bool:
        return sum(1 for ce in self.db.certificates.values() if ce.student_id == student_id) == expected
