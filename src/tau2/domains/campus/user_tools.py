"""User-side toolkit for the campus domain (学生小程序).

dual-control 硬边界：
- 学生侧独有动作＝上传材料、签署确认、拒签撤回——Agent 侧永不翻 pending_signatures.status /
  uploads.status；本侧是唯一能置"已确认/已拒绝/无效材料→已核验"的入口。
- Toolkit 绑定 student_id（初始化注入，经 initialization_actions 调用 `bind_student`，
  学生侧不可指定他人）。业务记录在主库（CampusDB），本侧只翻转 §5 时序契约中属于自己的格子。

工具返回值首字段 server_time；时间一律取 db.env.current_time。
"""

from datetime import timedelta
from typing import List, Optional

from pydantic import Field

from tau2.domains.campus.data_model import (
    AwardAppStatus,
    CampusDB,
    CertStatus,
    DeferralReasonType,
    DeferralStatus,
    EnrollmentSource,
    EnrollmentStatus,
    HospitalLevel,
    OfferingStatus,
    SigDocType,
    SigStatus,
    TodoStatus,
    TodoType,
    UploadDocType,
    UploadRow,
    UploadStatus,
)
from tau2.domains.campus.tools import (
    E_MAINTENANCE,
    TIME_FMT,
    _s,
    VALID_MEDICAL_LEVELS,
    fmt_time,
    in_cert_batch,
    in_maintenance,
    next_table_id,
    parse_time,
)
from tau2.domains.campus.user_data_model import UserDB
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool
from tau2.utils.pydantic_utils import BaseModelNoExtra

VALID_MEDICAL_DOC_TYPES = {UploadDocType.MEDICAL_DIAGNOSIS.value, UploadDocType.SICK_LEAVE.value}
VALID_FAST_DOC_TYPES = {UploadDocType.INCIDENT_PROOF.value, UploadDocType.MEDICAL_RECEIPT.value}


class TodoView(BaseModelNoExtra):
    todo_id: str
    type: str
    ref_id: str
    title: str
    status: str
    deadline_at: Optional[str] = None


class SigView(BaseModelNoExtra):
    sig_id: str
    doc_type: str
    ref_id: str
    summary: str
    status: str
    deadline_at: Optional[str] = None


class ProgressView(BaseModelNoExtra):
    kind: str
    request_id: str
    status: str
    deadline_at: Optional[str] = None
    reject_reason: Optional[str] = None


class AppSnapshot(BaseModelNoExtra):
    server_time: str
    student_id: str
    todos: List[TodoView] = Field(default_factory=list)
    pending_signatures: List[SigView] = Field(default_factory=list)
    progress: List[ProgressView] = Field(default_factory=list)
    maintenance_now: bool
    hint: str = ""


class UploadResult(BaseModelNoExtra):
    server_time: str
    upload_id: str
    status: str
    message: str


class ConfirmResult(BaseModelNoExtra):
    server_time: str
    sig_id: str
    business_status: str
    message: str


class RejectResult(BaseModelNoExtra):
    server_time: str
    sig_id: str
    business_status: str
    message: str


class CampusUserTools(ToolKitBase):
    """Tools for the student side (mini-program). `db`＝UserDB（哈希判分基准），
    `main_db`＝CampusDB（业务记录），`agent_tools`＝主库工具集（候补递补/放弃计数等
    共享规则复用，避免状态机逻辑两处漂移）。"""

    db: UserDB
    main_db: CampusDB

    def __init__(self, db: UserDB, main_db: CampusDB, agent_tools=None,
                 student_id: Optional[str] = None) -> None:
        super().__init__(db)
        self.main_db = main_db
        self.agent_tools = agent_tools
        self.bound_student_id: Optional[str] = student_id

    def use_tool(self, tool_name: str, **kwargs):
        """先结算后变更（pre-settle）+ 执行后结算：学生侧动作经共享的
        agent_tools.settle 收敛主库+用户库（两侧同一对 DB 实例，结算同源）。
        语义/幂等性/零现役影响与 CampusTools.use_tool 同（settle 为状态与时间的
        纯函数；金标重放逐题哈希对照证明 pre-settle 现役 no-op）。金标评测循环走
        make_tool_call→use_tool 且不额外调 sync_tools，结算必须挂在执行路径上。"""
        if self.agent_tools is not None:
            self.agent_tools.settle()
        resp = super().use_tool(tool_name, **kwargs)
        if self.agent_tools is not None:
            self.agent_tools.settle()
        return resp

    # ------------------------------------------------- 绑定与守卫（非工具）

    def bind_student(self, student_id: str) -> str:
        """初始化注入学生身份（非工具：仅经 initialization_actions / 构造参数调用）。"""
        if student_id not in self.main_db.students:
            raise ValueError(f"未找到学号 {student_id}。")
        self.bound_student_id = student_id
        return student_id

    @property
    def now_str(self) -> str:
        return self.main_db.env.current_time

    @property
    def now(self):
        return parse_time(self.main_db.env.current_time)

    def _student_or_raise(self) -> str:
        if self.bound_student_id is None:
            raise ValueError("学生端未绑定学号（任务需经 initialization_actions 调用 bind_student）。")
        return self.bound_student_id

    def _check_maintenance(self) -> None:
        if in_maintenance(self.now):
            raise ValueError(E_MAINTENANCE)

    def _complete_todo(self, todo_type: str, ref_id: str) -> None:
        for todo in self.db.app_todos.values():
            if (todo.student_id == self._student_or_raise() and todo.ref_id == ref_id
                    and todo.type == todo_type and todo.status == TodoStatus.PENDING):
                todo.status = TodoStatus.DONE
                todo.completed_at = self.now_str

    # ================================================================ 4 tools

    @is_tool(ToolType.READ)
    def check_student_app(self) -> AppSnapshot:
        """小程序快照：待办列表(含截止时刻)、待签署单、我的申请进度、维护/结账窗口提示。"""
        sid = self._student_or_raise()
        todos = [
            TodoView(todo_id=t.todo_id, type=_s(t.type), ref_id=t.ref_id, title=t.title,
                     status=_s(t.status), deadline_at=t.deadline_at)
            for t in self.db.app_todos.values() if t.student_id == sid
        ]
        sigs = [
            SigView(sig_id=s.sig_id, doc_type=_s(s.doc_type), ref_id=s.ref_id, summary=s.summary,
                    status=_s(s.status), deadline_at=s.deadline_at)
            for s in self.db.pending_signatures.values() if s.student_id == sid
        ]
        progress: List[ProgressView] = []
        for df in self.main_db.deferral_requests.values():
            if df.student_id == sid:
                progress.append(ProgressView(kind="缓考", request_id=df.request_id, status=_s(df.status),
                                             deadline_at=df.deadline_at, reject_reason=df.reject_reason))
        for app in self.main_db.scholarship_apps.values():
            if app.student_id == sid:
                progress.append(ProgressView(kind="奖助", request_id=app.app_id, status=_s(app.status),
                                             deadline_at=app.publicized_until, reject_reason=app.reject_reason))
        for ce in self.main_db.certificates.values():
            if ce.student_id == sid:
                progress.append(ProgressView(kind="证明", request_id=ce.cert_id, status=_s(ce.status),
                                             deadline_at=ce.ready_at, reject_reason=None))
        for tk in self.main_db.tickets.values():
            if tk.student_id == sid:
                progress.append(ProgressView(kind="工单", request_id=tk.ticket_id, status=_s(tk.status),
                                             deadline_at=tk.promised_reply_at, reject_reason=tk.resolution))
        maint = in_maintenance(self.now)
        hint = E_MAINTENANCE if maint else ""
        if not maint and in_cert_batch(self.now):
            hint = "证明系统当前处于月末结账时段（第31条），新申请将顺延。"
        return AppSnapshot(
            server_time=self.now_str, student_id=sid,
            todos=sorted(todos, key=lambda t: t.todo_id),
            pending_signatures=sorted(sigs, key=lambda s: s.sig_id),
            progress=progress, maintenance_now=maint, hint=hint,
        )

    @is_tool(ToolType.WRITE)
    def upload_material(self, ref_type: str, ref_id: str, doc_type: str,
                        file_name: str, hospital_level: str = "") -> UploadResult:
        """上传证明材料并挂接到业务记录（DF-/APP-/CE-/TK-）。医院等级由入参声明，
        环境按第12条三分类判定：声明'其他机构'=无效材料（结算驳回）；未申报（空串/'无'）
        =已退回补报；三甲/校医院指定门诊=有效。维护窗口内拒绝（第3条）。

        Args:
            ref_type: deferral_requests / scholarship_apps / certificates / tickets。
            ref_id: 业务记录号（DF-/APP-/CE-/TK-）。
            doc_type: 诊断证明/病假条/事故证明/医疗票据/困难认定表/身份证件影像/其他。
            file_name: 文件名。
            hospital_level: 三甲/校医院指定门诊/其他机构/无（医疗类材料须如实声明，第12条）。
        """
        self._check_maintenance()
        sid = self._student_or_raise()
        if ref_type not in ("deferral_requests", "scholarship_apps", "certificates", "tickets"):
            raise ValueError("材料只能挂接到：deferral_requests / scholarship_apps / certificates / tickets。")
        table = {
            "deferral_requests": self.main_db.deferral_requests,
            "scholarship_apps": self.main_db.scholarship_apps,
            "certificates": self.main_db.certificates,
            "tickets": self.main_db.tickets,
        }[ref_type]
        row = table.get(ref_id)
        if row is None:
            raise ValueError(f"未找到业务记录 {ref_id}。")
        if getattr(row, "student_id") != sid:
            raise ValueError(f"记录 {ref_id} 不属于本人：学生仅可办理本人事务（政策第4条）。")
        # 业务行 deadline 守卫（校验顺序前部、维护检查之后；第12/26条）。
        # 只对带硬截止的业务行生效：DF.deadline_at（缓考申报时限，第12条）与
        # APP.deadline_at（第26条，字段现无、分支预留）；certificate 无硬截止
        # （pickup_deadline 为领取期限，不拦上传）、工单 promised_reply_at 是
        # 答复 SLA——均不加守卫。语义边界同 confirm 守卫：`now > deadline` 才拒
        # （第2条 23:59 截止＝deadline 当刻含边界内）；与 B2 十工作日窗互不替代。
        if ref_type in ("deferral_requests", "scholarship_apps"):
            biz_deadline = getattr(row, "deadline_at", None)
            if biz_deadline and self.now > parse_time(biz_deadline):
                clause = 12 if ref_type == "deferral_requests" else 26
                raise ValueError(f"该申请的材料提交已于 {biz_deadline} 截止"
                                 f"（第{clause}条）；逾期不再受理。")
        if doc_type not in [d.value for d in UploadDocType]:
            raise ValueError("材料类型无效（诊断证明/病假条/事故证明/医疗票据/困难认定表/身份证件影像/其他）。")
        medical_doc = doc_type in (UploadDocType.MEDICAL_DIAGNOSIS.value, UploadDocType.SICK_LEAVE.value)
        status = UploadStatus.UPLOADED
        note = ""
        # 第12条三分类（医疗类材料）：
        #   声明"其他机构"或任何非 canonical 声明值 → 无效材料 → 结算驳回（第12条原文）；
        #   未申报（空串/"无"）→ 已退回，补报等级后重新上传，不作无效处理；
        #   canonical（三甲/校医院指定门诊）→ 有效。
        # 等级声明值的变体归一化仍为已披露的 future work。
        if medical_doc and hospital_level in ("", HospitalLevel.NONE.value):
            status = UploadStatus.RETURNED
            note = "未申报医院等级，暂不计入有效材料；请补充三甲/校医院指定门诊等级后重新上传（第12条）"
        elif medical_doc or hospital_level not in ("", HospitalLevel.NONE.value):
            if hospital_level not in VALID_MEDICAL_LEVELS:
                status = UploadStatus.INVALID
                note = "（第12条）其他医疗机构或私人诊所出具的证明视为无效材料。"
        up_id = next_table_id(self.db.uploads, "upload_id", "UP", 3)
        self.db.uploads[up_id] = UploadRow(
            upload_id=up_id, student_id=sid, ref_type=ref_type, ref_id=ref_id,
            doc_type=doc_type, file_name=file_name,
            size_kb=max(1, (len(file_name) + len(doc_type) * 4) % 5120) or 1,
            status=status, uploaded_at=self.now_str,
            reviewer=None, review_note=note or None,
        )
        message = f"材料已上传（{up_id}）并挂接 {ref_id}{note}"
        # 挂接到业务记录并推进该侧状态机格子
        if ref_type == "deferral_requests":
            row.proof_upload_ids.append(up_id)
            if hospital_level in VALID_MEDICAL_LEVELS:
                row.hospital_level = HospitalLevel(hospital_level)
            if status == UploadStatus.INVALID:
                if row.status in (DeferralStatus.PENDING_SIGN.value, DeferralStatus.PENDING_MATERIAL.value,
                                  DeferralStatus.SUBMITTED.value):
                    row.status = DeferralStatus.REJECTED.value
                    row.reject_reason = "证明材料无效：其他医疗机构或私人诊所出具的证明视为无效材料（第12条），直接驳回。"
                    row.reviewed_at = self.now_str
                    self._complete_todo(TodoType.DEFERRAL_UPLOAD.value, ref_id)
                    message += f"；缓考申请 {ref_id} 已被驳回（第12条）。"
            elif status == UploadStatus.RETURNED:
                # D-M6-1：已退回（未申报）不推进状态机、不办结上传待办——退回补报语义；
                # 结算侧 valid_docs 只计"已上传/已核验"，已退回不计入 → 必要集合不齐
                # → 既有"待材料"路径（settle 零改动自洽）。补传带等级材料后走既有迁移。
                pass
            elif row.status == DeferralStatus.PENDING_MATERIAL.value:
                row.status = DeferralStatus.SUBMITTED.value
                self._complete_todo(TodoType.DEFERRAL_UPLOAD.value, ref_id)
                message += f"；材料有效，缓考申请 {ref_id} 进入待审（已提交待审）。"
            elif row.status == DeferralStatus.PENDING_SIGN.value:
                # 签署前已上传有效材料——材料上传待办即完成（办结语义，第12条），
                # 业务状态仍留"待签署"，等学生本人完成签署确认。
                self._complete_todo(TodoType.DEFERRAL_UPLOAD.value, ref_id)
                message += f"；材料有效并挂接 {ref_id}，待本人完成签署确认。"
        elif ref_type == "scholarship_apps":
            row.material_upload_ids.append(up_id)
            if status == UploadStatus.UPLOADED and doc_type in VALID_FAST_DOC_TYPES:
                if row.status == AwardAppStatus.PENDING_MATERIAL.value:
                    row.status = AwardAppStatus.COLLEGE_REVIEW.value
                    message += f"；材料齐全，{ref_id} 重新进入学院受理。"
            elif status == UploadStatus.INVALID:
                row.status = AwardAppStatus.REJECTED.value
                row.reject_reason = "材料无效（第12/28条）。"
        elif ref_type == "certificates":
            if row.delivery == "委托代领" and row.proxy_info is not None and status == UploadStatus.UPLOADED:
                # 第33条闭环：有效证件影像落库绑定；签署+证件齐备（任一顺序）方进入制作
                row.proxy_info.proxy_doc_upload_id = up_id
                if row.status == CertStatus.PENDING_SIGN.value:
                    auth_sig = self.db.pending_signatures.get(row.proxy_info.auth_sig_id)
                    if auth_sig is not None and auth_sig.status == SigStatus.CONFIRMED.value:
                        row.status = CertStatus.MAKING
                        # 第33条：授权码自"签署"起 30 日——先签后传时锚=签署时刻
                        row.proxy_info.valid_until = fmt_time(parse_time(auth_sig.acted_at) + timedelta(days=30))
                        message += "；受托人证件影像已收到并绑定，代领授权齐备，证明进入制作（第33条）。"
                    else:
                        message += "；受托人证件影像已收到并绑定，待本人完成代领授权签署（第33条）。"
                else:
                    message += "；受托人证件影像已收到（证明已在制作中）。"
        return UploadResult(server_time=self.now_str, upload_id=up_id, status=status, message=message)

    @is_tool(ToolType.WRITE)
    def confirm_action(self, sig_id: str) -> ConfirmResult:
        """签署确认单。仅能确认本人待确认状态单据；成功翻转对应业务状态机一格
        （缓考申请→已提交待审；申请表→待学院审；候补递补→正式选课；特别通道→送审；
        代领授权→制作中）。

        Args:
            sig_id: 签署单号 SIG-xxx。
        """
        self._check_maintenance()
        sid = self._student_or_raise()
        sig = self.db.pending_signatures.get(sig_id)
        if sig is None or sig.student_id != sid:
            raise ValueError(f"未找到待签署单 {sig_id}（或不属于本人）。")
        # 单据确认 deadline 守卫（校验顺序前部、维护检查之后）。语义边界写死：
        # ① `now > deadline_at` 才拒——政策第2条 23:59 截止，deadline 当刻仍在界内；
        # ② 特别通道新单（drop_course 建）不带 deadline，天然不适用本守卫
        #    （种子历史单据如 SIG-016 带 deadline 则一致生效）；
        # ③ 与 B2"窗口关闭后10个工作日"提交窗是两个不同判据，互不替代。
        # 置于 status 检查之前：pre-settle 会先把过期单据置"已过期"，守卫保证给出
        # "逾期"专用文案（而非笼统状态文案），也是绕过结算路径时的显式双保险。
        if sig.deadline_at and self.now > parse_time(sig.deadline_at):
            raise ValueError(f"该单据确认已于 {sig.deadline_at} 截止（第8/12条）；"
                             f"逾期未确认已失效，请走重新申请流程。")
        if sig.status != SigStatus.PENDING:
            raise ValueError(f"签署单 {sig_id} 状态为'{_s(sig.status)}'，不可再确认。")
        sig.status = SigStatus.CONFIRMED
        sig.acted_at = self.now_str
        doc = sig.doc_type
        business_status = ""
        message = f"已签署确认（{sig_id}）。"
        if doc == SigDocType.DEFERRAL_APP:
            df = self.main_db.deferral_requests.get(sig.ref_id)
            if df is not None and df.status == DeferralStatus.PENDING_SIGN.value:
                # "进入待材料"按可用材料判定——仅存在已退回
                # （未申报等级）材料时同样进入待材料，确认回执不再误报"已提交待审"
                # （结算侧本会纠正为待材料，此处使响应与结算一致）。
                usable = df.reason_type != DeferralReasonType.ILLNESS.value or any(
                    (up := self.db.uploads.get(u)) is not None
                    and up.status in (UploadStatus.UPLOADED.value, UploadStatus.VERIFIED.value)
                    for u in df.proof_upload_ids
                )
                if not usable:
                    df.status = DeferralStatus.PENDING_MATERIAL
                    message += f"缓考申请 {df.request_id} 进入待材料（第12条：因病须上传三甲/校医院指定门诊诊断证明）。"
                else:
                    df.status = DeferralStatus.SUBMITTED
                    message += f"缓考申请 {df.request_id} 已提交待审，学院/教务处将按规则核验（第13条）。"
                business_status = df.status
            self._complete_todo(TodoType.DEFERRAL_SIGN.value, sig.ref_id)
        elif doc == SigDocType.AWARD_APP:
            app = self.main_db.scholarship_apps.get(sig.ref_id)
            if app is not None and app.status == AwardAppStatus.PENDING_SIGN.value:
                app.status = AwardAppStatus.COLLEGE_REVIEW
                message += f"奖助申请 {app.app_id} 进入学院受理（第26条）。"
                business_status = app.status
            self._complete_todo(TodoType.AWARD_SIGN.value, sig.ref_id)
        elif doc == SigDocType.WAITLIST_PROMOTE:
            en = self.main_db.enrollments.get(sig.ref_id)
            if en is not None and en.status == EnrollmentStatus.WAITLISTED.value:
                en.status = EnrollmentStatus.ENROLLED
                en.source = EnrollmentSource.WAITLIST_PROMOTE
                en.enrolled_at = self.now_str
                en.waitlist_position = None
                en.promote_sig_id = None
                off = self.main_db.course_offerings.get(en.offering_id)
                if off is not None:
                    off.enrolled_count += 1
                    off.waitlist_count = max(0, off.waitlist_count - 1)
                    if off.enrolled_count >= off.capacity:
                        off.status = OfferingStatus.FULL
                    if self.agent_tools is not None:
                        self.agent_tools._promote_next_waitlist(off.offering_id)
                message += f"候补递补确认完成：{en.enrollment_id} 转为正式选课（第8条）。"
                business_status = en.status
            self._complete_todo(TodoType.WAITLIST_PROMOTE_CONFIRM.value, sig.ref_id)
        elif doc == SigDocType.SPECIAL_CHANNEL:
            message += "特别通道申请已送审：学院审核→教务处备案，结果由系统按规则公示（第9条）。"
            business_status = "已送审"
        elif doc == SigDocType.PROXY_AUTH:
            ce = self.main_db.certificates.get(sig.ref_id)
            if ce is not None and ce.status == CertStatus.PENDING_SIGN.value:
                if ce.proxy_info is not None and ce.proxy_info.proxy_doc_upload_id:
                    ce.status = CertStatus.MAKING
                    ce.proxy_info.valid_until = fmt_time(self.now + timedelta(days=30))
                    message += f"代领授权生效：证明 {ce.cert_id} 进入制作（第33条：授权码30日内有效）。"
                else:
                    # 第33条闭环：签署完成但受托人证件未上传/无效——不进入制作；
                    # 后续有效证件经 upload_material 按已签署 auth_sig_id 推进闭环
                    message += f"代领授权书已签署；受托人证件影像尚未上传（或无效），证明 {ce.cert_id} 暂不进入制作（第33条）。"
                business_status = ce.status
            self._complete_todo(TodoType.CERT_PROXY_AUTH.value, sig.ref_id)
        elif doc in (SigDocType.ENROLL_CONFIRM, SigDocType.DROP_CONFIRM, SigDocType.APPEAL_BRIEF,
                     SigDocType.DEFERRED_EXAM_CONFIRM):
            business_status = "已确认"
            message += "确认已记录。"
        return ConfirmResult(server_time=self.now_str, sig_id=sig_id,
                             business_status=business_status, message=message)

    @is_tool(ToolType.WRITE)
    def reject_suggestion(self, sig_id: str, reason: str = "") -> RejectResult:
        """拒签/放弃：各类单据的拒签即撤销对应申请或放弃候补。

        候补确认单拒签＝放弃候补（abandon_count+1）；特别通道拒签＝撤回申请；
        其余单据拒签＝撤回该申请。Agent 建议不当的，学生可当面拒签。

        Args:
            sig_id: 签署单号 SIG-xxx。
            reason: 拒签理由（可选）。
        """
        self._check_maintenance()
        sid = self._student_or_raise()
        sig = self.db.pending_signatures.get(sig_id)
        if sig is None or sig.student_id != sid:
            raise ValueError(f"未找到待签署单 {sig_id}（或不属于本人）。")
        if sig.status != SigStatus.PENDING:
            # 拒签侧不设独立 deadline 判断——逾期单据
            # 已由 pre-settle 置"已过期"，本 status!=PENDING 检查即覆盖（拒签逾期单
            # 同样非法），以实现最简；与 confirm 侧守卫（需"逾期"专用文案）非对称。
            raise ValueError(f"签署单 {sig_id} 状态为'{_s(sig.status)}'，不可拒签。")
        sig.status = SigStatus.REJECTED
        sig.acted_at = self.now_str
        doc = sig.doc_type
        business_status = ""
        message = f"已拒签（{sig_id}）。"
        if doc == SigDocType.WAITLIST_PROMOTE:
            self._complete_todo(TodoType.WAITLIST_PROMOTE_CONFIRM.value, sig.ref_id)
            en = self.main_db.enrollments.get(sig.ref_id)
            if en is not None and en.status == EnrollmentStatus.WAITLISTED.value:
                en.status = EnrollmentStatus.VOIDED
                en.dropped_at = self.now_str
                en.promote_sig_id = None
                if self.agent_tools is not None:
                    self.agent_tools._abandon_waitlist(en)  # 内含顺延递补下一位
                student = self.main_db.students.get(sid)
                message += (f"候补放弃计入本学期放弃次数（现累计 {student.waitlist_abandon_count if student else '?'} 次，"
                            f"满3次候补功能关闭，第8条）。")
                business_status = en.status
        elif doc == SigDocType.DEFERRAL_APP:
            df = self.main_db.deferral_requests.get(sig.ref_id)
            if df is not None and df.status == DeferralStatus.PENDING_SIGN.value:
                df.status = DeferralStatus.WITHDRAWN
                message += f"缓考申请 {df.request_id} 于签署确认前撤回（第13条）：不占该课程本学期缓考额度。"
                business_status = df.status
            if self.agent_tools is not None:
                self.agent_tools._expire_open_signatures("deferral_requests", sig.ref_id)
        elif doc == SigDocType.AWARD_APP:
            app = self.main_db.scholarship_apps.get(sig.ref_id)
            if app is not None and app.status == AwardAppStatus.PENDING_SIGN.value:
                app.status = AwardAppStatus.WITHDRAWN
                if app.channel == "突发快速通道":
                    student = self.main_db.students.get(sid)
                    if student is not None:
                        student.temp_aid_used_this_year = False
                message += f"奖助申请 {app.app_id} 已于签署前撤回（第26条）。"
                business_status = app.status
        elif doc == SigDocType.SPECIAL_CHANNEL:
            en = self.main_db.enrollments.get(sig.ref_id)
            if en is not None and en.status == EnrollmentStatus.SPECIAL_CHANNEL_REVIEW.value:
                en.status = EnrollmentStatus.ENROLLED
                message += f"特别通道申请撤回：选课记录 {en.enrollment_id} 维持原状（第9条）。"
                business_status = en.status
        elif doc == SigDocType.PROXY_AUTH:
            ce = self.main_db.certificates.get(sig.ref_id)
            if ce is not None and ce.status == CertStatus.PENDING_SIGN.value:
                ce.status = CertStatus.WITHDRAWN
                message += f"证明申请 {ce.cert_id} 于出具前撤回（第30条）。"
                business_status = ce.status
        else:
            business_status = "已拒绝"
        return RejectResult(server_time=self.now_str, sig_id=sig_id,
                            business_status=business_status, message=message)

    # ------------------------------------------------- 断言辅助（非工具）

    def assert_signature_status(self, sig_id: str, expected_status: str) -> bool:
        sig = self.db.pending_signatures.get(sig_id)
        return sig is not None and sig.status == expected_status

    def assert_upload_status(self, upload_id: str, expected_status: str) -> bool:
        up = self.db.uploads.get(upload_id)
        return up is not None and up.status == expected_status
