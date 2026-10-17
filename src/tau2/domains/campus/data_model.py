"""campus 域数据模型与枚举。

接线约定：
- 行模型基类 = tau2.utils.pydantic_utils.BaseModelNoExtra（契约 NoExtra 等价物）；
- CampusDB/UserDB 基类 = tau2.environment.db.DB（load/dump/get_hash 由官方提供）；
- 字段集与枚举值与契约一字不改；发现契约缺口须显式决策，不得静默改。

全局约定：
- 时间一律 ISO 风格 `YYYY-MM-DD HH:MM` 字符串，服务器北京时间（政策第 2 条）；
- `null`＝未发生，禁空串/0 代替；
- 主键与表 dict 的键同值。
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import Field

from tau2.environment.db import DB
from tau2.utils.pydantic_utils import BaseModelNoExtra


# ---------------------------------------------------------------- 枚举


class StudentStatus(str, Enum):
    ACTIVE = "在读"
    SUSPENDED = "休学"
    RETAINED = "保留学籍"
    GRADUATED = "毕业"
    WITHDRAWN = "退学"


class AidPoolStatus(str, Enum):
    NOT_FILED = "未入库"
    FILED = "已入库"
    REMOVED = "已调出"


class CourseType(str, Enum):
    REQUIRED = "必修"
    ELECTIVE = "选修"
    GENERAL = "通识"
    PRACTICE = "实践"


class Assessment(str, Enum):
    EXAM = "考试"
    REVIEW = "考查"


class OfferingStatus(str, Enum):
    OPEN = "开放"
    FULL = "满员"
    SUSPENDED = "已停开"


class EnrollmentStatus(str, Enum):
    ENROLLED = "已选"
    WAITLISTED = "候补中"
    DROPPED = "已退课"
    VOIDED = "失效"           # 学籍变动失效（第10条）或候补放弃（第8条）
    PREREQ_AUTO_DROP = "先修自动退选"  # 第6条：先修期末未通过
    SPECIAL_CHANNEL_REVIEW = "特别通道审核中"  # 第9条


class EnrollmentSource(str, Enum):
    FIRST_ROUND = "首轮"
    ADD_DROP = "补退选"
    WAITLIST_PROMOTE = "候补递补"
    RETAKE = "重修"


class DropChannel(str, Enum):
    NORMAL = "正常"
    SPECIAL = "特别通道"      # v1.1 IC 定名，与蓝图 M10/H06 一致


class GradeLevel(str, Enum):
    A = "A"
    B_PLUS = "B+"
    B = "B"
    C_PLUS = "C+"
    C = "C"
    D = "D"
    F = "F"
    DEFERRED = "缓"           # 第13条占位：score 恒 null，不计 GPA
    ABSENT = "缺"             # 第18条：score=0，计 0 绩点与不及格学分
    UNPUBLISHED = "未公布"


GRADE_POINTS = {
    "A": 4.0, "B+": 3.5, "B": 3.0, "C+": 2.5, "C": 2.0, "D": 1.0, "F": 0.0, "缺": 0.0,
}  # 第 17 条换算表；"缓"/"未公布"无绩点。GPA 重算校验脚本与本表共用。


class ExamType(str, Enum):
    FINAL = "期末"
    MAKEUP = "补考"
    DEFERRED = "缓考"         # 第13条"第三周缓考时段"由 exam_type=缓考 的行承载


class ExamStatus(str, Enum):
    PUBLISHED = "已公布"
    CONFIRMED = "已确认"
    DONE = "已完成"
    DEFERRED = "已缓考"


class DeferralReasonType(str, Enum):
    CONFLICT = "冲突"
    ILLNESS = "因病"


class DeferralFilingType(str, Enum):
    BEFORE_EXAM = "考前正常"
    AFTER_EXAM = "考后补办"


class HospitalLevel(str, Enum):
    TIER3 = "三甲"
    CAMPUS_CLINIC = "校医院指定门诊"
    OTHER = "其他机构"        # 第12条：无效材料
    NONE = "无"


class DeferralStatus(str, Enum):
    PENDING_SIGN = "待签署"   # submit_deferral 成功即置
    PENDING_MATERIAL = "待材料"
    SUBMITTED = "已提交待审"
    APPROVED = "通过"
    REJECTED = "驳回"
    EXPIRED = "逾期"
    WITHDRAWN = "已撤回"      # 签署确认前主动撤回，不占额度


class ReviewStage(str, Enum):
    COLLEGE = "学院"
    ACADEMIC_AFFAIRS = "教务处"
    DONE = "完成"


class AwardCategory(str, Enum):
    NATIONAL = "国奖"
    INSPIRATION = "励志"
    PEOPLE = "人民奖学金"
    NEED_AID = "国助"
    EMERGENCY_GRANT = "临时补助"


class AwardScope(str, Enum):
    TERM = "学期"
    YEAR = "学年"


class AwardChannel(str, Enum):
    REGULAR = "常规"
    FAST_TRACK = "突发快速通道"   # 第28条，每学年限一次


class AwardAppStatus(str, Enum):
    PENDING_SIGN = "待签署"
    PENDING_MATERIAL = "待材料"
    ELIGIBILITY_CHECK = "资格校验中"
    COLLEGE_REVIEW = "待学院审"
    EVALUATING = "评审中"
    PUBLICIZING = "公示中"
    APPROVED = "通过"
    REJECTED = "驳回"
    WITHDRAWN = "已撤回"      # v1.1 IC 归一（第25/26条撤回其一）


class CertType(str, Enum):
    ZH_TRANSCRIPT = "中文成绩单"
    EN_TRANSCRIPT = "英文成绩单"
    ENROLLMENT = "在读证明"
    EXPECTED_GRAD = "预计毕业证明"
    STUDENT_STATUS_CERT = "学籍学历证明"


class CertLanguage(str, Enum):
    ZH = "中"
    EN = "英"


class CertDelivery(str, Enum):
    ELECTRONIC = "电子"
    SELF_SERVICE = "自助打印"
    COUNTER = "人工领取"
    PROXY = "委托代领"        # 第33条：需授权签署+证件影像


class CertStatus(str, Enum):
    PENDING_SIGN = "待签署"
    MAKING = "制作中"
    READY = "可领取"
    COLLECTED = "已领取"
    REJECTED = "驳回"
    BATCH_DEFERRED = "顺延结账"   # 第31条
    WITHDRAWN = "已撤回"          # 第30条出具前撤回，留行不出具


class TicketCategory(str, Enum):
    CONSULT = "咨询"          # 2 工作日答复；逾期查分可建咨询工单
    APPEAL = "申诉"           # 5 工作日；查分逾期时禁建
    SUGGESTION = "建议"       # 10 工作日


class TicketModule(str, Enum):
    COURSE = "选课"
    EXAM = "考试"
    GRADE = "成绩"
    STATUS = "学籍"
    AID = "奖助"
    CERT = "证明"
    OTHER = "其他"


class TicketStatus(str, Enum):
    PENDING = "待处理"
    COLLEGE_WORKING = "学院处理中"
    ANSWERED = "已答复"
    RECHECKING = "复核中"
    CLOSED = "已办结"


class TicketLevel(str, Enum):
    COLLEGE = "学院"
    ACADEMIC_AFFAIRS = "教务处"   # 须挂 parent_ticket_id（第34条逐级）


class TodoType(str, Enum):
    ENROLL_CONFIRM = "选课确认"
    WAITLIST_PROMOTE_CONFIRM = "候补递补确认"
    DEFERRAL_SIGN = "缓考申请签署"
    DEFERRAL_UPLOAD = "缓考材料上传"
    DEFERRED_EXAM_CONFIRM = "补缓考安排确认"
    AWARD_SIGN = "奖助申请签署"
    CERT_PROXY_AUTH = "证明代领授权"
    TICKET_PROGRESS = "工单进度查看"


class TodoStatus(str, Enum):
    PENDING = "待处理"
    DONE = "已完成"
    EXPIRED = "已过期"


class SigDocType(str, Enum):
    ENROLL_CONFIRM = "选课确认单"
    DROP_CONFIRM = "退课确认单"
    WAITLIST_PROMOTE = "候补递补确认单"
    DEFERRAL_APP = "缓考申请单"
    DEFERRED_EXAM_CONFIRM = "补缓考安排确认"
    AWARD_APP = "奖助学金申请表"
    APPEAL_BRIEF = "申诉书"
    PROXY_AUTH = "代领授权书"
    SPECIAL_CHANNEL = "特别通道审批单"   # 特别通道归一类型


class SigStatus(str, Enum):
    PENDING = "待确认"
    CONFIRMED = "已确认"
    REJECTED = "已拒绝"       # reject_suggestion：拒签＝撤回该申请（候补除外＝放弃）
    EXPIRED = "已过期"


class UploadDocType(str, Enum):
    MEDICAL_DIAGNOSIS = "诊断证明"
    SICK_LEAVE = "病假条"
    INCIDENT_PROOF = "事故证明"
    MEDICAL_RECEIPT = "医疗票据"
    AID_FORM = "困难认定表"
    ID_IMAGE = "身份证件影像"
    OTHER = "其他"


class UploadStatus(str, Enum):
    UPLOADED = "已上传"
    VERIFIED = "已核验"
    INVALID = "无效材料"      # 第12条：其他机构证明
    RETURNED = "已退回"


# ---------------------------------------------------------------- 嵌套小模型


class WarningRecord(BaseModelNoExtra):
    term: str = Field(description="警示所属学期")
    type: str = Field(description="警示类型（第21条：单学期8学分不及格/累计GPA<2.0）")


class DisciplinaryRecord(BaseModelNoExtra):
    date: str = Field(description="处分日期 YYYY-MM-DD")
    level: str = Field(description="处分等级（第14/24条：处分期内禁奖助）")


class SessionSlot(BaseModelNoExtra):
    weekday: int = Field(description="1=周一 … 7=周日")
    start: str = Field(description="开始时刻 HH:MM")
    end: str = Field(description="结束时刻 HH:MM")


class Eligibility(BaseModelNoExtra):
    """奖学金资格校验源（scholarships.eligibility，第24/25/27/28条）。"""

    no_fail_this_year: bool = Field(False, description="评定学年无不及格记录（含'缺'）")
    no_warning: bool = Field(False, description="无学业警示记录")
    no_discipline: bool = Field(False, description="无处分记录")
    gpa_rank_top_pct: Optional[float] = Field(None, description="GPA 年级排名前百分数（如 10.0）")
    comprehensive_rank_top_pct: Optional[float] = Field(None, description="综测排名前百分数")
    aid_pool_required: bool = Field(False, description="要求困难库已入库（第27条）")
    single_use_year: bool = Field(False, description="每学年限一次（第28条）")


class AppWindow(BaseModelNoExtra):
    term: str = Field(description="受理学期")
    start: str = Field(description="窗口开始 YYYY-MM-DD HH:MM")
    end: str = Field(description="窗口截止 YYYY-MM-DD HH:MM")


class Maintenance(BaseModelNoExtra):
    weekly: str = Field("周日23:00-周一06:00", description="第3条：全部写操作不受理")
    cert_batch: str = Field("每月最后工作日17:00-22:00", description="第31条：证明暂停")


class EnvState(BaseModelNoExtra):
    """环境时间锚（db.json 顶层保留键）。任务可覆写 current_time。"""

    current_time: str = Field(description="服务器北京时间 YYYY-MM-DD HH:MM")
    term: str = Field(description="当前学期，如 2026SP")
    maintenance: Maintenance = Field(default_factory=Maintenance)


class ProxyInfo(BaseModelNoExtra):
    proxy_name: str = Field(description="受托人姓名（脱敏种子）")
    id_masked: str = Field(description="受托人证件掩码")
    auth_sig_id: str = Field(description="指向 SIG- 代领授权书")
    valid_until: str = Field(description="授权码有效期＝签署日起 30 日（第33条）")
    proxy_doc_upload_id: Optional[str] = Field(
        None, description="受托人证件影像 UP-（第33条：签署+有效证件齐备方可进入制作）")


# ---------------------------------------------------------------- 行模型


class StudentRow(BaseModelNoExtra):
    student_id: str = Field(description="PK，S+学号，如 S20230102")
    name: str
    gender: str
    birth_date: str
    id_card_masked: str = Field(description="只存掩码（脱敏）")
    college: str
    major: str
    degree_class: str
    enrollment_year: int
    student_status: StudentStatus
    is_graduating_cohort: bool = False
    expected_graduation_term: Optional[str] = None
    phone: str
    email: str
    gpa: float = Field(description="缓存值：第17条公式重算须一致；非在读冻结")
    aid_pool_status: AidPoolStatus = AidPoolStatus.NOT_FILED
    aid_pool_valid_through: Optional[str] = Field(None, description="有效期至学年，如 2025-2026")
    warning_records: List[WarningRecord] = []
    disciplinary_records: List[DisciplinaryRecord] = []
    major_change_used: bool = Field(False, description="第20条")
    temp_aid_used_this_year: bool = Field(False, description="第28条")
    waitlist_abandon_count: int = Field(0, description="第8条：≥3 冻结候补")
    credit_limit_override: bool = Field(False, description="第5条：超32学分特批")


class CourseRow(BaseModelNoExtra):
    course_id: str = Field(description="PK，CRS-")
    name: str
    college: str
    course_type: CourseType
    credits: float
    hours: Optional[int] = None
    prerequisite_course_ids: List[str] = Field(default_factory=list, description="第6条")
    assessment: Assessment
    description: str = ""


class OfferingRow(BaseModelNoExtra):
    offering_id: str = Field(description="PK，OF-")
    course_id: str
    term: str
    instructor: str
    classroom: str
    session_slots: List[SessionSlot]
    capacity: int
    enrolled_count: int = Field(description="＝本学期 status=已选 的 EN 行数")
    waitlist_count: int = 0
    status: OfferingStatus
    start_date: str = Field(description="开课日（第7条窗口起算）")
    adddrop_deadline: str = Field(description="＝start_date+13 天 23:59（补退选截止判定列）")
    term_end_date: str


class EnrollmentRow(BaseModelNoExtra):
    enrollment_id: str = Field(description="PK，EN-")
    student_id: str
    offering_id: str
    term: str
    status: EnrollmentStatus
    source: EnrollmentSource
    enrolled_at: str
    dropped_at: Optional[str] = None
    drop_channel: Optional[DropChannel] = None
    waitlist_position: Optional[int] = Field(None, description="status=候补中 时有效")
    promote_sig_id: Optional[str] = Field(None, description="候补确认 SIG-（第8条 24h 签署链）")


class GradeRow(BaseModelNoExtra):
    grade_id: str = Field(description="PK，GR-")
    student_id: str
    offering_id: str
    term: str
    academic_year: str
    attempt_no: int = Field(1, description="考核 attempt 序号；重修行 attempt_no+1")
    score: Optional[float] = Field(None, description="'缓'恒 null；'缺'=0")
    grade_level: GradeLevel
    gpa_points: Optional[float] = Field(None, description="按第17条换算；'缓'为 null")
    is_final: bool = Field(False, description="终行标志：本学期在修课不得有 is_final=True（§5 规则2）")
    recorded_at: Optional[str] = Field(None, description="公布时刻（第16条查分窗口起算列）")
    is_retake: bool = False
    replaces_grade_id: Optional[str] = Field(None, description="第19条：替换的原成绩行；所有 attempt 行永久保留")


class ExamRow(BaseModelNoExtra):
    exam_id: str = Field(description="PK，EX-")
    offering_id: str
    term: str
    exam_type: ExamType
    scheduled_at: str
    duration_min: int
    location: str
    seat_no: Optional[str] = None
    published_at: Optional[str] = Field(None, description="考试周前2周（第11条）；未到＝未公布")
    status: ExamStatus


class DeferralRow(BaseModelNoExtra):
    request_id: str = Field(description="PK，DF-")
    student_id: str
    offering_id: str
    exam_id: str
    reason_type: DeferralReasonType
    filing_type: DeferralFilingType
    proof_upload_ids: List[str] = Field(default_factory=list, description="因病补办必填")
    hospital_level: HospitalLevel
    status: DeferralStatus
    review_stage: ReviewStage = ReviewStage.COLLEGE
    reject_reason: Optional[str] = None
    submitted_at: Optional[str] = None
    deadline_at: Optional[str] = Field(None, description="冲突＝考试日前一日23:59；因病＝考后第3工作日23:59；写入时预计算")
    reviewed_at: Optional[str] = None


class AwardRow(BaseModelNoExtra):
    award_id: str = Field(description="PK，AW-")
    name: str
    category: AwardCategory
    amount: float = Field(description="金额一律取自本目录行（§5 规则7），禁散落")
    level: Optional[str] = None
    scope: AwardScope
    eligibility: Eligibility = Field(default_factory=Eligibility)
    stack_conflicts: List[str] = Field(default_factory=list, description="第25条兼得表（国奖↔国助）")
    app_window: Optional[AppWindow] = None
    disbursed_to: str = Field("校园卡", description="第23条：发放至本人校园卡金融账户")


class AwardAppRow(BaseModelNoExtra):
    app_id: str = Field(description="PK，APP-")
    student_id: str
    award_id: str
    term: str
    academic_year: str
    channel: AwardChannel = AwardChannel.REGULAR
    status: AwardAppStatus
    sig_id: Optional[str] = Field(None, description="SIG- 申请表（小程序签署）")
    material_upload_ids: List[str] = []
    reject_reason: Optional[str] = None
    submitted_at: Optional[str] = None
    publicized_until: Optional[str] = Field(None, description="公示≥5工作日（第26条）；断言到'进入公示中'即止")
    fast_track_used_this_year: Optional[bool] = None


class CertRow(BaseModelNoExtra):
    cert_id: str = Field(description="PK，CE-")
    student_id: str
    cert_type: CertType
    language: CertLanguage
    copy_count: int = Field(1, description="单笔≤5（第32条）")
    delivery: CertDelivery
    status: CertStatus
    applied_at: str
    ready_at: Optional[str] = Field(None, description="出具时刻（第30条：约3工作日）")
    pickup_deadline: Optional[str] = None
    verify_code: Optional[str] = Field(None, description="电子证明在线验证码（第30条）")
    verify_code_expiry: Optional[str] = Field(None, description="出具后90日")
    proxy_info: Optional[ProxyInfo] = Field(None, description="delivery=委托代领 时有值（第33条）")


class TicketRow(BaseModelNoExtra):
    ticket_id: str = Field(description="PK，TK-")
    student_id: str
    category: TicketCategory
    module: TicketModule
    title: str
    content: str = Field(description="须实名并注明学号与事由（第35条）")
    status: TicketStatus
    level: TicketLevel
    created_at: str
    promised_reply_at: Optional[str] = Field(None, description="咨询2/申诉5/建议10 工作日")
    handler: Optional[str] = None
    resolution: Optional[str] = None
    parent_ticket_id: Optional[str] = Field(None, description="level=教务处 必填（第34条逐级）")


class TodoRow(BaseModelNoExtra):
    todo_id: str = Field(description="PK，TODO-")
    student_id: str
    type: TodoType
    ref_type: str = Field(description="指向主库表名（deferral_requests 等）")
    ref_id: str
    title: str
    status: TodoStatus
    created_at: str
    deadline_at: Optional[str] = Field(None, description="候补确认＝+24h；材料上传＝DF.deadline_at 同步")
    completed_at: Optional[str] = None


class SignatureRow(BaseModelNoExtra):
    sig_id: str = Field(description="PK，SIG-。Agent 侧 WRITE 只能生成本表行，不能置'已确认'（dual-control 硬边界）")
    student_id: str
    doc_type: SigDocType
    ref_type: str
    ref_id: str
    summary: str
    status: SigStatus
    created_at: str
    deadline_at: Optional[str] = None
    acted_at: Optional[str] = None


class UploadRow(BaseModelNoExtra):
    upload_id: str = Field(description="PK，UP-")
    student_id: str
    ref_type: str = Field(description="挂载 DF-/APP-/CE-/TK-")
    ref_id: str
    doc_type: UploadDocType
    file_name: str
    size_kb: int
    status: UploadStatus
    uploaded_at: str
    reviewer: Optional[str] = None
    review_note: Optional[str] = None


# ---------------------------------------------------------------- 库聚合


class CampusDB(DB):
    """主库 11 表 + env 保留键。JSON 布局：各表 主键→行对象。"""

    env: EnvState
    students: Dict[str, StudentRow] = {}
    courses: Dict[str, CourseRow] = {}
    course_offerings: Dict[str, OfferingRow] = {}
    enrollments: Dict[str, EnrollmentRow] = {}
    grades: Dict[str, GradeRow] = {}
    exam_arrangements: Dict[str, ExamRow] = {}
    deferral_requests: Dict[str, DeferralRow] = {}
    scholarships: Dict[str, AwardRow] = {}
    scholarship_apps: Dict[str, AwardAppRow] = {}
    certificates: Dict[str, CertRow] = {}
    tickets: Dict[str, TicketRow] = {}

    def get_statistics(self) -> Dict[str, Any]:
        """各表行数统计。"""
        return {
            "num_students": len(self.students),
            "num_courses": len(self.courses),
            "num_course_offerings": len(self.course_offerings),
            "num_enrollments": len(self.enrollments),
            "num_grades": len(self.grades),
            "num_exam_arrangements": len(self.exam_arrangements),
            "num_deferral_requests": len(self.deferral_requests),
            "num_scholarships": len(self.scholarships),
            "num_scholarship_apps": len(self.scholarship_apps),
            "num_certificates": len(self.certificates),
            "num_tickets": len(self.tickets),
        }


def get_db():
    from tau2.domains.campus.utils import CAMPUS_DB_PATH

    return CampusDB.load(str(CAMPUS_DB_PATH))


if __name__ == "__main__":
    db = get_db()
    print(db.get_statistics())

