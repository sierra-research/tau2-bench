"""Campus domain tool tests.

Two layers:
1. Error catalog table-driven: every error-catalog code triggered on seed data,
   message compared against the catalog template text (措辞即判分变量，逐字).
2. State-machine transition tables: the 5 dual-control chains
   plus withdraw/expiry branches, asserted end-state-only (env_assertions 口径).

Data targets are hardcoded against the contract seed;
regenerating seeds must keep these anchors or update this file in the same commit.
"""

import itertools
from datetime import timedelta

import pytest

from tau2.domains.campus.environment import get_environment
from tau2.domains.campus.tools import fmt_time, parse_time


@pytest.fixture
def env():
    return get_environment()


def set_now(env, ts):
    env.tools.db.env.current_time = ts


# ------------------------------------------------------------------ 错误目录逐条触发

class TestErrorCatalog:
    def test_e_maintenance_blocks_agent_and_user_writes(self, env):
        set_now(env, "2026-06-15 02:00")  # 周一 02:00，维护窗内
        with pytest.raises(ValueError, match="当前处于系统维护时段（周日23:00–周一06:00，第3条）"):
            env.tools.create_ticket("S20230103", "咨询", "其他", "x", "y")
        env.user_tools.bind_student("S20230103")
        with pytest.raises(ValueError, match="写操作暂停，查询不受影响"):
            env.user_tools.upload_material("deferral_requests", "DF-006", "诊断证明", "a.pdf", "三甲")
        # 读不受影响
        assert env.tools.get_student_details("S20230103").student_status == "在读"

    def test_e_window_closed_enroll_and_drop(self, env):
        with pytest.raises(ValueError, match="补退选已于 2026-03-15 23:59 截止（政策第7条）"):
            env.tools.enroll_course("S20250401", "OF-2026SP-301-1")
        # 窗口外非毕业班退课
        en = next(e for e in env.tools.db.enrollments.values()
                  if e.student_id == "S20230103" and e.term == "2026SP" and e.status == "已选")
        with pytest.raises(ValueError, match="本学期不再受理退课；如为毕业班学生可走特别通道（第9条）"):
            env.tools.drop_course("S20230103", en.enrollment_id)

    def test_e_prereq_missing(self, env):
        set_now(env, "2026-03-04 10:00")  # 窗口内
        with pytest.raises(ValueError, match="要求先修《高等数学A\\(上\\)》合格或正在修读（政策第6条）"):
            env.tools.enroll_course("S20250401", "OF-2026SP-102-1")

    def test_e_capacity_full(self, env):
        set_now(env, "2026-03-04 10:00")
        with pytest.raises(ValueError, match="该课已满，可加入候补（第8条）"):
            env.tools.enroll_course("S20250401", "OF-2026SP-402-1")

    def test_e_time_conflict(self, env):
        set_now(env, "2026-03-04 10:00")
        # S20230103 已选 OF-2026SP-403-1（周二10:00），302 同时段
        with pytest.raises(ValueError, match="上课时间冲突（星期二 10:00-11:40）"):
            env.tools.enroll_course("S20230103", "OF-2026SP-302-1")

    def test_e_deferral_expired(self, env):
        # 2025FA 804 的考试 2026-01-15，考后3工作日早已过
        with pytest.raises(ValueError, match="该场考试已过补办时限（考后3个工作日，政策第12条）"):
            env.tools.submit_deferral("S20230303", "OF-2025FA-804-1", "EX-0014", "因病", "考后补办")

    def test_e_deferral_used(self, env):
        # S20250403 对 OF-2026SP-702-1 已有 DF-006（待签署，未撤回）→ 不占则拦
        with pytest.raises(ValueError, match="本学期已申请过缓考（第13条：仅一次）"):
            env.tools.submit_deferral("S20250403", "OF-2026SP-702-1", "EX-0039", "冲突", "考前正常")

    def test_e_approval_pending(self, env):
        # EN-0014 (S20220101) 已是特别通道审核中
        with pytest.raises(ValueError, match="毕业班特别通道申请仍在学院/教务处双重审核中（第9条）"):
            env.tools.drop_course("S20220101", "EN-0014")

    def test_e_aid_pool(self, env):
        with pytest.raises(ValueError, match="国家助学金/励志奖学金要求已认定入库（第27条）"):
            env.tools.submit_scholarship_app("S20230103", "AW-002")

    def test_e_record_uneligible(self, env):
        # S20230301 评定学年(2025-2026)有 F/缺 → 人民奖学金 no_fail_this_year
        with pytest.raises(ValueError, match="评定学年存在不及格记录（含'缺'）（第24条）.*重修通过不消除当学年记录（第19条）"):
            env.tools.submit_scholarship_app("S20230301", "AW-003")

    def test_e_stack_conflict(self, env):
        # S20230401 已获 AW-007 国助（APP-006 通过），AW-001/AW-007 互斥
        with pytest.raises(ValueError, match="与已申请/已获国家助学金构成兼得限制（第25条），请撤回其一"):
            env.tools.submit_scholarship_app("S20230401", "AW-001")

    def test_e_review_expired_only_appeal_grade(self, env):
        with pytest.raises(ValueError, match="成绩公布已超5个工作日，查分不受理（第16条）；申诉通道不适用于替代查分"):
            env.tools.create_ticket("S20230103", "申诉", "成绩", "复核", "学号S20230103")
        # v1.1-F2：咨询类成绩工单不触发
        t = env.tools.create_ticket("S20230103", "咨询", "成绩", "了解", "学号S20230103")
        assert t.ticket_id == "TK-006"

    def test_e_level_skip_appeal(self, env):
        # parent 挂他人/状态不符 → 越级拦截
        with pytest.raises(ValueError, match="复核须先经学院处理（第34条）"):
            env.tools.create_ticket("S20250401", "申诉", "选课", "复核选课", "学号S20250401",
                                    parent_ticket_id="TK-001")

    def test_e_waitlist_frozen(self, env):
        set_now(env, "2026-03-04 10:00")
        env.tools.db.students["S20250401"].waitlist_abandon_count = 3
        with pytest.raises(ValueError, match="本学期候补放弃已累计3次（政策第8条）"):
            env.tools.join_waitlist("S20250401", "OF-2026SP-402-1")

    def test_waitlist_cutoff_48h(self, env):
        # 锚 6-12 已过 adddrop_deadline-48h（3-13 23:59）
        with pytest.raises(ValueError, match="候补申请已于 2026-03-13 23:59 截止"):
            env.tools.join_waitlist("S20250401", "OF-2026SP-402-1")

    def test_e_cert_batch_defers_application(self, env):
        set_now(env, "2026-06-30 18:00")  # 6月最后工作日（周二）结账窗内
        r = env.tools.request_certificate("S20230103", "在读证明")
        assert r.status == "顺延结账"
        assert "证明系统月末结账中（最后工作日17:00–22:00，第31条），申请已顺延" in r.message

    def test_no_data_boundary_interception(self, env):
        """v1.1-F5：读工具对代查不拦截（E-DATA-BOUNDARY 不注册运行时）。"""
        d = env.tools.get_student_details("S20230103")
        assert d.server_time == "2026-06-12 10:00"


# ------------------------------------------------------------------ 状态机转移表

class TestDeferralChain:
    def test_conflict_full_chain_approve(self, env):
        t, u = env.tools, env.user_tools
        r = t.submit_deferral("S20230103", "OF-2026SP-203-1", "EX-0049", "冲突", "考前正常")
        assert r.status == "待签署"
        u.bind_student("S20230103")
        c = u.confirm_action(r.sig_id)
        assert c.business_status == "已提交待审"
        env.sync_tools()
        df = t.db.deferral_requests[r.request_id]
        assert (df.status, df.review_stage) == ("通过", "完成")
        assert t.db.exam_arrangements["EX-0049"].status == "已缓考"
        todo = t.user_db.app_todos[[k for k, v in t.user_db.app_todos.items() if v.ref_id == r.request_id][0]]
        assert todo.status == "已完成"

    def test_illness_invalid_material_reject(self, env):
        t, u = env.tools, env.user_tools
        r = t.submit_deferral("S20230103", "OF-2026SP-203-1", "EX-0049", "因病", "考后补办")
        u.bind_student("S20230103")
        u.confirm_action(r.sig_id)  # 无材料 → 待材料
        df = t.db.deferral_requests[r.request_id]
        assert df.status == "待材料"
        up = u.upload_material("deferral_requests", r.request_id, "诊断证明", "私人诊所.pdf", "其他机构")
        assert up.status == "无效材料"
        assert df.status == "驳回"
        assert "无效材料" in df.reject_reason

    def test_illness_valid_material_approve(self, env):
        """因病缓考须诊断证明+病假条双材料齐（第12条）方可通过；
        顺序对齐 M05 合规样板：submit→confirm→上传两份→settle→通过。"""
        t, u = env.tools, env.user_tools
        r = t.submit_deferral("S20230103", "OF-2026SP-203-1", "EX-0049", "因病", "考后补办")
        u.bind_student("S20230103")
        u.confirm_action(r.sig_id)
        up = u.upload_material("deferral_requests", r.request_id, "诊断证明", "市一院.pdf", "三甲")
        up2 = u.upload_material("deferral_requests", r.request_id, "病假条", "病假条.pdf", "三甲")
        assert t.db.deferral_requests[r.request_id].status == "已提交待审"
        env.sync_tools()
        assert t.db.deferral_requests[r.request_id].status == "通过"
        assert t.user_db.uploads[up.upload_id].status == "已核验"
        assert t.user_db.uploads[up2.upload_id].status == "已核验"

    def test_illness_single_material_stays_pending(self, env):
        """仅有诊断证明一份（缺病假条）不满足第12条
        '诊断证明及病假建议'双材料必齐 → 结算留'待材料'，不得放行。"""
        t, u = env.tools, env.user_tools
        r = t.submit_deferral("S20230103", "OF-2026SP-203-1", "EX-0049", "因病", "考后补办")
        u.bind_student("S20230103")
        u.confirm_action(r.sig_id)
        u.upload_material("deferral_requests", r.request_id, "诊断证明", "市一院.pdf", "三甲")
        env.sync_tools()
        assert t.db.deferral_requests[r.request_id].status == "待材料"

    def test_undeclared_level_returned_not_rejected(self, env):
        """第12条只对声明'其他机构'的材料定无效→驳回；
        第二份医疗材料漏报医院等级＝已退回（补报），结算走既有'待材料'路径而非驳回
        （第一份已有效的诊断证明不被作废）；补传带等级的病假条后结算通过、双材料核验。"""
        t, u = env.tools, env.user_tools
        r = t.submit_deferral("S20230103", "OF-2026SP-203-1", "EX-0049", "因病", "考后补办")
        u.bind_student("S20230103")
        u.confirm_action(r.sig_id)  # 无材料 → 待材料
        up1 = u.upload_material("deferral_requests", r.request_id, "诊断证明", "市一院.pdf", "三甲")
        assert t.user_db.uploads[up1.upload_id].status == "已上传"
        up2 = u.upload_material("deferral_requests", r.request_id, "病假条", "病假条.pdf")  # 未申报
        assert t.user_db.uploads[up2.upload_id].status == "已退回"
        assert "未申报医院等级" in (t.user_db.uploads[up2.upload_id].review_note or "")
        env.sync_tools()
        df = t.db.deferral_requests[r.request_id]
        assert df.status == "待材料"  # 非驳回：未申报 ≠ 无效材料
        assert df.reject_reason is None
        assert t.user_db.uploads[up1.upload_id].status == "已上传"  # 第一份未被作废
        up3 = u.upload_material("deferral_requests", r.request_id, "病假条", "病假条-补报.pdf", "三甲")
        env.sync_tools()
        assert df.status == "通过"
        assert t.user_db.uploads[up1.upload_id].status == "已核验"
        assert t.user_db.uploads[up3.upload_id].status == "已核验"
        assert t.user_db.uploads[up2.upload_id].status == "已退回"  # 退回行不参与核验

    def test_none_level_treated_as_undeclared(self, env):
        """显式声明'无'与空串同路径＝未申报 → 已退回补报（非无效材料、非驳回）。"""
        t, u = env.tools, env.user_tools
        r = t.submit_deferral("S20230103", "OF-2026SP-203-1", "EX-0049", "因病", "考后补办")
        u.bind_student("S20230103")
        u.confirm_action(r.sig_id)
        u.upload_material("deferral_requests", r.request_id, "诊断证明", "市一院.pdf", "三甲")
        up = u.upload_material("deferral_requests", r.request_id, "病假条", "病假条.pdf", "无")
        row = t.user_db.uploads[up.upload_id]
        assert row.status == "已退回"
        assert "未申报医院等级" in (row.review_note or "")
        env.sync_tools()
        df = t.db.deferral_requests[r.request_id]
        assert df.status == "待材料"  # 非驳回
        assert df.reject_reason is None

    def test_other_institution_still_rejects(self, env):
        """边界（第12条原文不变）：声明'其他机构'＝无效材料→整单驳回；
        即便第一份三甲诊断证明已有效，第二份'其他机构'仍触发驳回。"""
        t, u = env.tools, env.user_tools
        r = t.submit_deferral("S20230103", "OF-2026SP-203-1", "EX-0049", "因病", "考后补办")
        u.bind_student("S20230103")
        u.confirm_action(r.sig_id)
        u.upload_material("deferral_requests", r.request_id, "诊断证明", "市一院.pdf", "三甲")
        up = u.upload_material("deferral_requests", r.request_id, "病假条", "小诊所说.pdf", "其他机构")
        assert t.user_db.uploads[up.upload_id].status == "无效材料"
        df = t.db.deferral_requests[r.request_id]
        assert df.status == "驳回"
        assert "无效材料" in df.reject_reason

    def test_confirm_with_only_returned_uploads_reports_pending(self, env):
        """仅存在已退回（未申报等级）材料时，签署确认回执
        如实报"待材料"而非"已提交待审"——响应与结算终态一致。"""
        t, u = env.tools, env.user_tools
        r = t.submit_deferral("S20230103", "OF-2026SP-203-1", "EX-0049", "因病", "考后补办")
        u.bind_student("S20230103")
        u.upload_material("deferral_requests", r.request_id, "病假条", "病假条.pdf")  # 未申报等级
        assert t.db.deferral_requests[r.request_id].status == "待签署"  # D-M6-1：退回不推进
        conf = u.confirm_action(r.sig_id)
        assert conf.business_status == "待材料"
        assert "待材料" in conf.message and "已提交待审" not in conf.message
        env.sync_tools()
        assert t.db.deferral_requests[r.request_id].status == "待材料"

    def test_upload_before_sign_completes_upload_todo(self, env):
        """签署前上传有效材料，材料上传待办即完成，业务行仍待签署。
        场景取本人已选的 203/EX-0049（绑定守卫后合法），断言不变。
        签署前传齐诊断证明+病假条两份（单份结算留待材料），断言不变。"""
        t, u = env.tools, env.user_tools
        r = t.submit_deferral("S20230103", "OF-2026SP-203-1", "EX-0049", "因病", "考后补办")
        u.bind_student("S20230103")
        up = u.upload_material("deferral_requests", r.request_id, "诊断证明", "诊断证明.pdf", "三甲")
        u.upload_material("deferral_requests", r.request_id, "病假条", "病假条.pdf", "三甲")
        df = t.db.deferral_requests[r.request_id]
        assert df.status == "待签署"
        todo = next(v for v in t.user_db.app_todos.values()
                    if v.ref_id == r.request_id and v.type == "缓考材料上传")
        assert todo.status == "已完成"
        # 后续签署一路走到通过（材料已齐，不再进入待材料）
        u.confirm_action(r.sig_id)
        env.sync_tools()
        assert t.db.deferral_requests[r.request_id].status == "通过"

    def test_pending_sign_expiry_by_advance_time(self, env):
        t, u = env.tools, env.user_tools
        r = t.submit_deferral("S20230103", "OF-2026SP-203-1", "EX-0049", "冲突", "考前正常")
        u.bind_student("S20230103")
        t.advance_time(days=15)  # 过 deadline（EX-0049 6/26 前一日 23:59）
        env.sync_tools()
        df = t.db.deferral_requests[r.request_id]
        assert df.status == "逾期"
        sig = t.user_db.pending_signatures[r.sig_id]
        assert sig.status == "已过期"

    def test_withdraw_before_sign_restores_quota(self, env):
        t, u = env.tools, env.user_tools
        r = t.submit_deferral("S20230103", "OF-2026SP-203-1", "EX-0049", "冲突", "考前正常")
        u.bind_student("S20230103")
        w = t.withdraw_application("S20230103", "deferral", r.request_id)
        assert w.new_status == "已撤回"
        assert t.user_db.pending_signatures[r.sig_id].status == "已过期"
        # P13 额度恢复：重新申请成功
        r2 = t.submit_deferral("S20230103", "OF-2026SP-203-1", "EX-0049", "冲突", "考前正常")
        assert r2.status == "待签署"
        # 已提交待审阶段不可撤回
        u.confirm_action(r2.sig_id)
        with pytest.raises(ValueError, match="签署确认前方可主动撤回"):
            t.withdraw_application("S20230103", "deferral", r2.request_id)


class TestAwardAndCertificate:
    def test_fast_track_full_flow(self, env):
        t, u = env.tools, env.user_tools
        r = t.submit_scholarship_app("S20220101", "AW-008", "突发快速通道")
        assert t.db.students["S20220101"].temp_aid_used_this_year is True
        u.bind_student("S20220101")
        u.upload_material("scholarship_apps", r.app_id, "事故证明", "山火.pdf")
        u.confirm_action(r.sig_id)
        assert t.db.scholarship_apps[r.app_id].status == "待学院审"
        env.sync_tools()
        app = t.db.scholarship_apps[r.app_id]
        assert app.status == "公示中" and app.publicized_until is not None
        t.advance_time(hours=360)  # +15天，过公示
        env.sync_tools()
        assert t.db.scholarship_apps[r.app_id].status == "通过"

    def test_award_sign_pending_seed_confirm(self, env):
        t, u = env.tools, env.user_tools
        u.bind_student("S20240202")
        c = u.confirm_action("SIG-004")  # 种子待签署：APP-004 人民奖学金一等
        assert c.business_status == "待学院审"
        env.sync_tools()
        assert t.db.scholarship_apps["APP-004"].status in ("公示中", "驳回")

    def test_english_transcript_notes_unpublished(self, env):
        r = env.tools.request_certificate("S20240101", "英文成绩单", language="英")
        assert "英文成绩单仅含已正式记载成绩（第32条）" in r.message

    def test_cert_proxy_authorize_chain(self, env):
        t, u = env.tools, env.user_tools
        r = t.request_certificate("S20230103", "在读证明", delivery="委托代领",
                                  proxy_name="李受托", proxy_id_masked="****5678")
        assert r.status == "待签署"
        u.bind_student("S20230103")
        up = u.upload_material("certificates", r.cert_id, "身份证件影像", "id.jpg")
        u.confirm_action(r.sig_id)
        ce = t.db.certificates[r.cert_id]
        assert ce.proxy_info.proxy_doc_upload_id == up.upload_id  # 证件绑定落库
        assert ce.status == "制作中"
        assert ce.proxy_info.valid_until == "2026-07-12 10:00"
        t.advance_time(hours=96)  # 6-16 10:00，未过 ready_at(6-17 10:00)
        env.sync_tools()
        assert ce.status == "制作中"
        t.advance_time(hours=48)  # 6-18 → 过 ready_at
        env.sync_tools()
        assert ce.status == "可领取" and ce.verify_code
        # 出具后不可撤回
        with pytest.raises(ValueError, match="出具前方可撤回（第30条）"):
            t.withdraw_application("S20230103", "certificate", r.cert_id)

    def test_cert_withdraw_before_issue(self, env):
        r = env.tools.request_certificate("S20230103", "中文成绩单")
        w = env.tools.withdraw_application("S20230103", "certificate", r.cert_id)
        assert w.new_status == "已撤回"
        assert env.tools.db.certificates[r.cert_id].ready_at is not None  # 留行标注


class TestWaitlistAndSpecialChannel:
    def test_drop_promotes_and_confirm(self, env):
        t, u = env.tools, env.user_tools
        set_now(env, "2026-03-04 10:00")
        r = t.drop_course("S20230101", "EN-0027")  # 402 空出一位
        assert r.status == "已退课"
        # 队列头 EN-0070 种子确认单 SIG-015 未过期 → 不为后序抢跑发新单
        assert t.db.course_offerings["OF-2026SP-402-1"].enrolled_count == 3
        sig159 = [s for s in t.user_db.pending_signatures.values()
                  if s.doc_type == "候补递补确认单" and s.ref_id == "EN-0159"]
        assert sig159 == []
        u.bind_student("S20230104")
        c = u.confirm_action("SIG-015")
        assert c.business_status == "已选"
        assert t.db.enrollments["EN-0070"].source == "候补递补"
        off = t.db.course_offerings["OF-2026SP-402-1"]
        assert off.enrolled_count == 4 and off.status == "满员"
        # 再退一位 → 队列下一位 EN-0159 收到新确认单
        t.drop_course("S20230102", "EN-0041")
        sig2 = [s for s in t.user_db.pending_signatures.values()
                if s.doc_type == "候补递补确认单" and s.ref_id == "EN-0159" and s.status == "待确认"]
        assert len(sig2) == 1
        assert t.db.enrollments["EN-0159"].promote_sig_id == sig2[0].sig_id

    def test_waitlist_reject_counts_abandon(self, env):
        t, u = env.tools, env.user_tools
        u.bind_student("S20230104")
        r = u.reject_suggestion("SIG-015", reason="已选别的课")
        assert r.business_status == "失效"
        assert t.db.students["S20230104"].waitlist_abandon_count == 1
        assert t.db.course_offerings["OF-2026SP-402-1"].waitlist_count == 1

    def test_waitlist_expiry_by_advance(self, env):
        t, u = env.tools, env.user_tools
        set_now(env, "2026-03-04 10:00")
        t.drop_course("S20230101", "EN-0027")  # 先空出一位（前序 SIG-015 未过期，不为 0159 抢跑）
        set_now(env, "2026-06-12 19:00")       # SIG-015 deadline 18:00 已过
        env.sync_tools()
        assert t.db.enrollments["EN-0070"].status == "失效"
        assert t.db.students["S20230104"].waitlist_abandon_count == 1
        # 空位顺延递补：EN-0159 生成新确认单
        sig = [s for s in t.user_db.pending_signatures.values()
               if s.doc_type == "候补递补确认单" and s.ref_id == "EN-0159" and s.status == "待确认"]
        assert len(sig) == 1

    def test_special_channel_confirm_approve(self, env):
        t, u = env.tools, env.user_tools
        # 第9条 10 工作日提交窗（03-15 关，窗至 03-27 23:59）——
        # 原 06-12 锚点已超窗会按新规拒收，链路测试改在窗内时点执行
        set_now(env, "2026-03-20 10:00")
        # S20220101 唯一开课 805 → 无替代 → 双重程序通过
        en = next(e for e in t.db.enrollments.values()
                  if e.student_id == "S20220101" and e.offering_id == "OF-2026SP-805-1" and e.status == "已选")
        r = t.drop_course("S20220101", en.enrollment_id)
        assert r.status == "特别通道审核中"
        u.bind_student("S20220101")
        u.confirm_action(r.special_sig_id)
        env.sync_tools()
        row = t.db.enrollments[en.enrollment_id]
        assert row.status == "已退课" and row.drop_channel == "特别通道"

    def test_special_channel_reject_when_alternative(self, env):
        t, u = env.tools, env.user_tools
        # 提交窗内时点（03-20，第9条 10 工作日窗至 03-27）
        set_now(env, "2026-03-20 10:00")
        # 造替代开课：CRS-805 另一门 2026SP 开放有余位
        from tau2.domains.campus.data_model import OfferingRow
        src = t.db.course_offerings["OF-2026SP-805-1"]
        alt = src.model_copy(update={"offering_id": "OF-2026SP-805-2", "enrolled_count": 10,
                                      "capacity": 40, "status": "开放"})
        t.db.course_offerings["OF-2026SP-805-2"] = alt
        en = next(e for e in t.db.enrollments.values()
                  if e.student_id == "S20220101" and e.offering_id == "OF-2026SP-805-1" and e.status == "已选")
        r = t.drop_course("S20220101", en.enrollment_id)
        u.bind_student("S20220101")
        u.confirm_action(r.special_sig_id)
        env.sync_tools()
        assert t.db.enrollments[en.enrollment_id].status == "已选"  # 有替代 → 维持原状

    def test_special_channel_student_rejects(self, env):
        t, u = env.tools, env.user_tools
        # 提交窗内时点（03-20，第9条 10 工作日窗至 03-27）
        set_now(env, "2026-03-20 10:00")
        en = next(e for e in t.db.enrollments.values()
                  if e.student_id == "S20220101" and e.offering_id == "OF-2026SP-805-1" and e.status == "已选")
        r = t.drop_course("S20220101", en.enrollment_id)
        u.bind_student("S20220101")
        u.reject_suggestion(r.special_sig_id, reason="再想想")
        assert t.db.enrollments[en.enrollment_id].status == "已选"

    def test_special_channel_within_10_workday_window(self, env):
        """正向（H06 场景）：补退选窗口 03-15 关闭，第 10 个工作日 03-27 23:59
        前（本例 03-20）发起特别通道成功，全链至已退课。"""
        set_now(env, "2026-03-20 10:00")
        r = env.tools.drop_course("S20220101", "EN-0011")
        assert r.status == "特别通道审核中"
        assert r.special_sig_id  # 审批单已建（种子 max+1=SIG-017，不带 deadline）
        env.user_tools.bind_student("S20220101")
        env.user_tools.confirm_action(r.special_sig_id)
        env.sync_tools()
        row = env.tools.db.enrollments["EN-0011"]
        assert row.status == "已退课" and row.drop_channel == "特别通道"
        assert env.tools.db.course_offerings["OF-2026SP-301-1"].enrolled_count == 7  # 8→7 权威重算

    def test_special_channel_reject_after_10_workday_window(self, env):
        """负向：窗口关闭超 10 个工作日（03-27 23:59 之后）→ 第9条拒收，
        零写（EN 保持已选、不建 SIG）。"""
        set_now(env, "2026-03-30 10:00")
        with pytest.raises(ValueError,
                            match="10 个工作日内提交（第9条，截止 2026-03-27 23:59 前）；已超窗"):
            env.tools.drop_course("S20220101", "EN-0011")
        row = env.tools.db.enrollments["EN-0011"]
        assert row.status == "已选"
        assert not any(s.doc_type == "特别通道审批单" and s.ref_id == "EN-0011"
                       for s in env.tools.user_db.pending_signatures.values())


class TestDualControlBoundary:
    def test_agent_cannot_flip_signature(self, env):
        # Agent 侧不存在任何能翻转 SIG 状态的工具（工具面清点）
        names = {t.name for t in env.get_tools()}
        assert all(not n.startswith(("confirm", "reject", "sign")) for n in names)
        r = env.tools.submit_deferral("S20230103", "OF-2026SP-203-1", "EX-0049", "冲突", "考前正常")
        assert env.tools.user_db.pending_signatures[r.sig_id].status == "待确认"

    def test_user_cannot_operate_others(self, env):
        u = env.user_tools
        u.bind_student("S20230103")
        with pytest.raises(ValueError, match="不属于本人"):
            u.upload_material("deferral_requests", "DF-006", "诊断证明", "x.pdf", "三甲")
        with pytest.raises(ValueError, match="或不属于本人"):
            u.confirm_action("SIG-013")  # DF-006 的签署单属 S20250403


class TestReadTools:
    def test_server_time_first_on_every_tool(self, env):
        results = {
            "student": env.tools.get_student_details("S20230103"),
            "offerings": env.tools.get_course_offerings("2026SP", keyword="算法"),
            "enrollments": env.tools.get_enrollments("S20230103", "2026SP"),
            "grades": env.tools.get_grades("S20230103"),
            "exams": env.tools.get_exam_arrangements("S20230103", "2026SP"),
            "requests": env.tools.get_service_requests("S20250403", "deferral"),
            "policy": env.tools.search_policy("维护"),
        }
        for name, res in results.items():
            first = list(res.model_dump().keys())[0]
            assert first == "server_time", f"{name} first field is {first}"

    def test_get_grades_semantic_annotations(self, env):
        g = env.tools.get_grades("S20240101")
        huang = [x for x in g.grades if x.grade_level == "缓"]
        assert huang and "不属于通过成绩" in huang[0].note
        g2 = env.tools.get_grades("S20230303")
        que = [x for x in g2.grades if x.grade_level == "缺"]
        assert que and "放弃课程" in que[0].note

    def test_get_grades_retake_relation_marked(self, env):
        # 种子含重修行（replaces_grade_id）
        from tau2.domains.campus.data_model import GradeRow
        retakes = [g for g in env.tools.db.grades.values() if g.is_retake and g.replaces_grade_id]
        assert retakes
        g = env.tools.get_grades(retakes[0].student_id)
        assert any(retakes[0].replaces_grade_id in x.note for x in g.grades)

    def test_exam_publish_gate(self, env):
        ex = env.tools.get_exam_arrangements("S20230103", "2026SP")
        assert all(x.scheduled_at != "" for x in ex.exams)
        # 未公布（published_at 未到）→ 掩码显示
        set_now(env, "2026-06-01 10:00")
        ex2 = env.tools.get_exam_arrangements("S20230103", "2026SP")
        assert all(x.scheduled_at == "未公布" for x in ex2.exams)

    def test_search_policy_chapters_and_full_text(self, env):
        allc = env.tools.search_policy("", section="")
        assert len(allc.clauses) == 36  # 六章36条一字不动的回归锚
        maint = env.tools.search_policy("例行维护", section="总则")
        assert [c.clause_no for c in maint.clauses] == [3]
        assert "周日 23:00 至周一 06:00" in maint.clauses[0].text
        sel = env.tools.search_policy("", section="选课")
        assert [c.clause_no for c in sel.clauses] == [5, 6, 7, 8, 9, 10]

    def test_service_requests_missing_flags(self, env):
        rs = env.tools.get_service_requests("S20250403", "deferral")
        df6 = [x for x in rs.requests if x.request_id == "DF-006"][0]
        assert "缺签署" in " ".join(df6.missing)


# ------------------------------------------------------------------ 一致性守卫（A2–A7）

class TestM1ConsistencyGuards:
    def test_appeal_target_grade_in_window_succeeds(self, env):
        # GR-0033 公布 2026-01-23 09:00（周五）→ 5 工作日窗至 2026-01-30 23:59
        set_now(env, "2026-01-26 10:00")
        r = env.tools.create_ticket("S20230103", "申诉", "成绩", "复核", "学号S20230103",
                                    target_grade_id="GR-0033")
        assert r.ticket_id in env.tools.db.tickets  # target 仅判窗不落库

    def test_appeal_target_grade_expired_rejected(self, env):
        # 种子时点 2026-06-12：GR-0033（2026-01-23 公布）早已越 5 工作日窗
        with pytest.raises(ValueError, match="成绩公布已超5个工作日"):
            env.tools.create_ticket("S20230103", "申诉", "成绩", "复核", "学号S20230103",
                                    target_grade_id="GR-0033")

    def test_appeal_target_overrides_any_grade_scan(self, env):
        """P1-3 修复核心语义：指定目标按该成绩判窗——其他成绩在窗也救不了越窗的目标。"""
        set_now(env, "2026-01-26 10:00")  # GR-0033 在窗；GR-0028（2025-01-24 公布）越窗
        with pytest.raises(ValueError, match="成绩公布已超5个工作日"):
            env.tools.create_ticket("S20230103", "申诉", "成绩", "复核", "学号S20230103",
                                    target_grade_id="GR-0028")

    def test_appeal_target_grade_missing_or_not_own(self, env):
        with pytest.raises(ValueError, match="未找到成绩记录 GR-9999（或不属于该学号）"):
            env.tools.create_ticket("S20230103", "申诉", "成绩", "复核", "学号S20230103",
                                    target_grade_id="GR-9999")
        other = next(g.grade_id for g in env.tools.db.grades.values() if g.student_id != "S20230103")
        with pytest.raises(ValueError, match="未找到成绩记录 .*（或不属于该学号）"):
            env.tools.create_ticket("S20230103", "申诉", "成绩", "复核", "学号S20230103",
                                    target_grade_id=other)

    def test_appeal_without_target_unchanged(self, env):
        """不传 target_grade_id：判窗行为与修前完全一致（种子时点拒、窗内过）。"""
        with pytest.raises(ValueError, match="成绩公布已超5个工作日"):
            env.tools.create_ticket("S20230103", "申诉", "成绩", "复核", "学号S20230103")
        set_now(env, "2026-01-26 10:00")
        r = env.tools.create_ticket("S20230103", "申诉", "成绩", "复核", "学号S20230103")
        assert r.ticket_id in env.tools.db.tickets

    def test_appeal_target_args_do_not_change_ticket_row(self, env):
        """target_grade_id 仅判窗、不落库——传/不传的工单行完全一致，
        agent 显式传参不会使 DB 终态与金标缺省路径哈希分叉。"""
        set_now(env, "2026-01-26 10:00")
        r1 = env.tools.create_ticket("S20230103", "申诉", "成绩", "复核", "学号S20230103",
                                     target_grade_id="GR-0033")
        d1 = env.tools.db.tickets[r1.ticket_id].model_dump()
        d1.pop("ticket_id")
        r2 = env.tools.create_ticket("S20230103", "申诉", "成绩", "复核", "学号S20230103")
        d2 = env.tools.db.tickets[r2.ticket_id].model_dump()
        d2.pop("ticket_id")
        assert d1 == d2

    def test_proxy_pickup_confirm_without_document_holds(self, env):
        """第33条闭环：先签署、后传有效证件——签署时无证件不进制作，上传侧闭环推进。"""
        t, u = env.tools, env.user_tools
        r = t.request_certificate("S20230103", "在读证明", delivery="委托代领",
                                  proxy_name="李受托", proxy_id_masked="****1234")
        u.bind_student("S20230103")
        res = u.confirm_action(r.sig_id)
        ce = t.db.certificates[r.cert_id]
        assert ce.status == "待签署"
        assert "尚未上传" in res.message
        up = u.upload_material("certificates", r.cert_id, "身份证件影像", "id.jpg")
        assert ce.status == "制作中"
        assert ce.proxy_info.proxy_doc_upload_id == up.upload_id

    def test_proxy_pickup_returned_document_not_bound(self, env):
        """无效/退回材料不绑定证件位，签署确认保持拦截（第33条）。"""
        t, u = env.tools, env.user_tools
        r = t.request_certificate("S20230103", "在读证明", delivery="委托代领",
                                  proxy_name="李受托", proxy_id_masked="****1234")
        u.bind_student("S20230103")
        u.upload_material("certificates", r.cert_id, "诊断证明", "a.pdf")  # 医疗材料无等级 → 已退回
        ce = t.db.certificates[r.cert_id]
        assert ce.proxy_info.proxy_doc_upload_id is None
        res = u.confirm_action(r.sig_id)
        assert ce.status == "待签署"
        assert "尚未上传" in res.message

    def test_proxy_late_upload_validity_anchors_signing(self, env):
        """先签后传时授权码有效期锚=签署时刻（第33条），非上传时刻。"""
        t, u = env.tools, env.user_tools
        r = t.request_certificate("S20230103", "在读证明", delivery="委托代领",
                                  proxy_name="李受托", proxy_id_masked="****1234")
        u.bind_student("S20230103")
        u.confirm_action(r.sig_id)  # T0 = 2026-06-12 10:00（种子时点）
        t.advance_time(hours=5)
        env.sync_tools()
        u.upload_material("certificates", r.cert_id, "身份证件影像", "id.jpg")
        ce = t.db.certificates[r.cert_id]
        assert ce.status == "制作中"
        assert ce.proxy_info.valid_until == "2026-07-12 10:00"  # 签署时刻+30d（若锚上传时刻则=15:00）

    def test_service_requests_cert_missing_distinguishes_doc_vs_sign(self, env):
        """待签署二因分流——未签署=缺签署；已签署=缺受托人证件（第33条）。"""
        t, u = env.tools, env.user_tools
        r = t.request_certificate("S20230103", "在读证明", delivery="委托代领",
                                  proxy_name="李受托", proxy_id_masked="****1234")
        u.bind_student("S20230103")
        rs = t.get_service_requests("S20230103", "certificate")
        m0 = " ".join(next(x for x in rs.requests if x.request_id == r.cert_id).missing)
        assert "缺签署" in m0
        u.confirm_action(r.sig_id)  # 已签署、无证件 → 不应再引导重新签署
        rs = t.get_service_requests("S20230103", "certificate")
        m1 = " ".join(next(x for x in rs.requests if x.request_id == r.cert_id).missing)
        assert "缺受托人证件影像" in m1 and "缺签署" not in m1
        u.upload_material("certificates", r.cert_id, "身份证件影像", "id.jpg")
        rs = t.get_service_requests("S20230103", "certificate")
        assert not next(x for x in rs.requests if x.request_id == r.cert_id).missing

    def test_campus_package_metadata(self):
        """Runtime-readable benchmark metadata — engine (pyproject, tau2==1.0.1)
        and campus benchmark versions are intentionally separate axes."""
        import re

        import tau2.domains.campus as campus

        assert re.fullmatch(r"\d+\.\d+\.\d+", campus.__version__)
        assert campus.POLICY_VERSION == "1.4.2"
        assert campus.SCORING_PROTOCOL == "tau2-v1.0.1-compatible"

    def test_deferral_requires_own_enrollment(self, env):
        # S20250401 在读但未选 OF-2026SP-203-1 → 缓考绑定守卫拒绝（第12/10条）
        with pytest.raises(ValueError, match="未找到该课程的在读选课记录，无法申请缓考（政策第12/10条）"):
            env.tools.submit_deferral("S20250401", "OF-2026SP-203-1", "EX-0049", "冲突", "考前正常")

    def test_illness_requires_post_exam_filing(self, env):
        # 因病 + 考前正常 → 拒（第12条二）；因病主路径由 test_illness_* 与 A1 金标重放钉住
        with pytest.raises(ValueError, match="因病缓考属考后补办（第12条二）"):
            env.tools.submit_deferral("S20230103", "OF-2026SP-203-1", "EX-0049", "因病", "考前正常")

    def test_waitlist_requires_active_status(self, env):
        # S20240105 休学 → 学籍守卫先于窗口/容量检查触发（第10条）
        with pytest.raises(ValueError, match="当前学籍状态为'休学'（政策第10条）：非在读状态不享受选课服务（含候补）"):
            env.tools.join_waitlist("S20240105", "OF-2026SP-402-1")

    def test_waitlist_rejects_suspended_offering(self, env):
        # OF-2026SP-401-1 已停开 → 停开守卫先于 48h 截止检查触发（第9条相关）
        with pytest.raises(ValueError, match="本学期已停开（第9条相关）：不可加入候补"):
            env.tools.join_waitlist("S20250401", "OF-2026SP-401-1")

    def test_cert_copy_count_lower_bound(self, env):
        with pytest.raises(ValueError, match="开具份数至少为 1 份（政策第32条）"):
            env.tools.request_certificate("S20230103", "在读证明", copy_count=0)


# ------------------------------------------------------------------ v1.1 加固

class TestPreSettle:
    """pre-settle（先结算后变更）：时钟推进后首次经 use_tool 分发的调用
    （连只读）即触发结算。现役冻结时间下 pre-settle 为 no-op——由金标重放逐题哈希
    与基线逐位一致另行证明。"""

    def test_agent_use_tool_pre_settle(self, env):
        set_now(env, "2026-06-12 19:00")  # SIG-015（18:00）已过，未手动 sync
        env.make_tool_call(tool_name="get_student_details", requestor="assistant",
                           student_id="S20230104")
        assert env.tools.user_db.pending_signatures["SIG-015"].status == "已过期"
        assert env.tools.db.enrollments["EN-0070"].status == "失效"
        assert env.tools.db.students["S20230104"].waitlist_abandon_count == 1

    def test_user_use_tool_pre_settle(self, env):
        env.user_tools.bind_student("S20230104")
        set_now(env, "2026-06-12 19:00")
        env.make_tool_call(tool_name="check_student_app", requestor="user")
        assert env.tools.user_db.pending_signatures["SIG-015"].status == "已过期"
        assert env.tools.db.enrollments["EN-0070"].status == "失效"


# 回归矩阵路由表：6 路写动作 × {deadline−1min 成功 / deadline 当刻成功 /
# deadline+1min 拒绝}。边界语义写死：`now > deadline_at` 才拒（政策第2条 23:59
# 截止＝deadline 当刻含边界内）。种子截止均取自 db/user_db 实测值；全部时点落在
# 非维护窗（周日23:00–周一06:00）与非结账窗内。
MATRIX_ROUTES = {
    # agent submit_deferral：EX-0049（2026-06-26 09:00）冲突缓考 → 考前一日23:59
    "agent_submit_deferral": {
        "deadline": "2026-06-25 23:59",
        "bind": None,
        "call": {"tool_name": "submit_deferral", "requestor": "assistant",
                 "student_id": "S20230103", "offering_id": "OF-2026SP-203-1",
                 "exam_id": "EX-0049", "reason_type": "冲突", "filing_type": "考前正常"},
        "reject": "该场考试已过补办时限",
        "ok": lambda r: r.status == "待签署" and r.deadline_at == "2026-06-25 23:59",
    },
    # user confirm：种子 DF-006/SIG-013（待签署，截止 6-23 23:59）
    "user_confirm_deferral": {
        "deadline": "2026-06-23 23:59",
        "bind": "S20250403",
        "call": {"tool_name": "confirm_action", "requestor": "user", "sig_id": "SIG-013"},
        "reject": "该单据确认已于 2026-06-23 23:59 截止",
        "ok": lambda r: r.business_status == "已提交待审",
    },
    # user upload：DF-006 业务行 deadline_at（第12条）；拒绝须零写（不落 UP 行）
    "user_upload": {
        "deadline": "2026-06-23 23:59",
        "bind": "S20250403",
        "call": {"tool_name": "upload_material", "requestor": "user",
                 "ref_type": "deferral_requests", "ref_id": "DF-006",
                 "doc_type": "诊断证明", "file_name": "诊断证明.pdf",
                 "hospital_level": "三甲"},
        "reject": "该申请的材料提交已于 2026-06-23 23:59 截止（第12条）",
        "ok": lambda r: r.upload_id.startswith("UP-"),
        "zero_write": "uploads",
    },
    # waitlist confirm：种子 SIG-015（EN-0070 24h 递补确认，截止 6-12 18:00）
    "user_confirm_waitlist": {
        "deadline": "2026-06-12 18:00",
        "bind": "S20230104",
        "call": {"tool_name": "confirm_action", "requestor": "user", "sig_id": "SIG-015"},
        "reject": "该单据确认已于 2026-06-12 18:00 截止",
        "ok": lambda r: r.business_status == "已选",
    },
    # special-channel confirm：种子 SIG-016（EN-0014；drop_course 新单不带
    # deadline 不适用守卫，种子历史单据带 6-20 23:59 → 守卫一致生效）
    "user_confirm_special_channel": {
        "deadline": "2026-06-20 23:59",
        "bind": "S20220101",
        "call": {"tool_name": "confirm_action", "requestor": "user", "sig_id": "SIG-016"},
        "reject": "该单据确认已于 2026-06-20 23:59 截止",
        "ok": lambda r: r.business_status == "已送审",
    },
    # award confirm：种子 SIG-004（APP-004，截止 6-20 23:59）
    "user_confirm_award": {
        "deadline": "2026-06-20 23:59",
        "bind": "S20240202",
        "call": {"tool_name": "confirm_action", "requestor": "user", "sig_id": "SIG-004"},
        "reject": "该单据确认已于 2026-06-20 23:59 截止",
        "ok": lambda r: r.business_status == "待学院审",
    },
}


class TestDeadlineGuardMatrix:
    """回归矩阵：6 路 × 3 时点（deadline−1min / 当刻 / +1min）＝18 例。
    动作一律走 env.make_tool_call（真实分发路径，含 pre-settle），时钟 set_now
    精确控制；成功侧断言业务返回值，拒绝侧断言守卫文案（+upload 路零写）。"""

    @pytest.mark.parametrize("offset", [-1, 0, 1], ids=["t-1min", "t-0", "t+1min"])
    @pytest.mark.parametrize("route", sorted(MATRIX_ROUTES))
    def test_matrix(self, env, route, offset):
        spec = MATRIX_ROUTES[route]
        if spec["bind"]:
            env.user_tools.bind_student(spec["bind"])
        set_now(env, fmt_time(parse_time(spec["deadline"]) + timedelta(minutes=offset)))
        uploads_before = len(env.user_tools.db.uploads)
        if offset <= 0:
            result = env.make_tool_call(**spec["call"])
            assert spec["ok"](result), f"{route}@{offset}: unexpected result {result}"
        else:
            with pytest.raises(ValueError, match=spec["reject"]):
                env.make_tool_call(**spec["call"])
            if spec.get("zero_write") == "uploads":
                assert len(env.user_tools.db.uploads) == uploads_before  # 守卫前置零写

    def test_reject_overdue_covered_by_status_check(self, env):
        """reject_suggestion 不设独立 deadline 判断——
        pre-settle 先置"已过期"，status!=PENDING 检查即覆盖逾期拒签
        （专用文案"不可拒签"），以实现最简。"""
        env.user_tools.bind_student("S20230104")
        set_now(env, "2026-06-12 19:00")  # SIG-015（18:00）已过
        with pytest.raises(ValueError, match="已过期.*不可拒签"):
            env.make_tool_call(tool_name="reject_suggestion", requestor="user",
                               sig_id="SIG-015")


class TestM4Hardening:
    """M4 D2/D3：特别通道卡死侧终态迁移 + 候补位次单调（分支隔离加固）。"""

    def test_special_channel_confirmed_past_deadline_migrates(self, env):
        """构造性测试：种子带 deadline 的特别单
        （SIG-016=2026-06-20 23:59）confirm 后推时间过 deadline → settle 终态
        迁移：EN-0014 回'已选'（计数 _recount_enrolled 权威重算）+ SIG 置
        '已过期'（acted_at 保留确认时刻）。confirm 走直接方法调用以保留
        "确认后、结算前"的格子——现役分发路径下 confirm 后即时结算，
        该格不可达（D1 守卫前置），按终态完备性必须存在。"""
        t, u = env.tools, env.user_tools
        set_now(env, "2026-06-19 10:00")
        u.bind_student("S20220101")
        c = u.confirm_action("SIG-016")  # 直接调用：不经 use_tool，暂不结算
        assert c.business_status == "已送审"
        assert t.db.enrollments["EN-0014"].status == "特别通道审核中"
        set_now(env, "2026-06-21 00:00")  # 推时间过 deadline（周日 00:00 非维护窗）
        env.sync_tools()                   # → 终态迁移
        en = t.db.enrollments["EN-0014"]
        sig = t.user_db.pending_signatures["SIG-016"]
        assert en.status == "已选" and en.promote_sig_id is None
        assert sig.status == "已过期"
        assert sig.acted_at == "2026-06-19 10:00"  # acted_at 保留学生确认时刻
        off = t.db.course_offerings["OF-2026SP-402-1"]
        rows = sum(1 for e in t.db.enrollments.values()
                   if e.offering_id == "OF-2026SP-402-1" and e.status == "已选")
        assert off.enrolled_count == rows == 5  # EN-0014 计入后权威重算

    def test_waitlist_position_monotonic_after_abandon_rejoin(self, env):
        """位次单调：A（EN-0070 位次1）放弃后 C 重入——位次=活跃候补最大位次+1：
        C 得 3（与存活 B=EN-0159 的 2 不重复、排在 B 之后）；旧公式
        waitlist_count+1 会复用位次 2 造成重复，本测试即其回归屏障。"""
        t, u = env.tools, env.user_tools
        set_now(env, "2026-03-04 10:00")  # 候补 48h 截止（03-13 23:59）前
        u.bind_student("S20230104")
        u.reject_suggestion("SIG-015", reason="已选别的课")  # A=EN-0070（位次1）放弃
        r = t.join_waitlist("S20230103", "OF-2026SP-402-1")  # C 重入
        assert r.waitlist_position == 3
        active = {e.enrollment_id: e.waitlist_position
                  for e in t.db.enrollments.values()
                  if e.offering_id == "OF-2026SP-402-1" and e.status == "候补中"}
        assert active == {"EN-0159": 2, r.enrollment_id: 3}  # B 保持 2、C=3：无重复
        assert active[r.enrollment_id] > active["EN-0159"]   # C 在 B 之后（顺位正确）
        assert t.db.course_offerings["OF-2026SP-402-1"].waitlist_count == 2


def _m4_step_join(env):
    env.make_tool_call(tool_name="join_waitlist", requestor="assistant",
                       student_id="S20230103", offering_id="OF-2026SP-402-1")


def _m4_step_abandon(env):
    env.user_tools.bind_student("S20230104")
    env.make_tool_call(tool_name="reject_suggestion", requestor="user", sig_id="SIG-015")


def _m4_step_confirm(env):
    env.user_tools.bind_student("S20230104")
    env.make_tool_call(tool_name="confirm_action", requestor="user", sig_id="SIG-015")


def _m4_step_drop(env):
    env.make_tool_call(tool_name="drop_course", requestor="assistant",
                       student_id="S20230101", enrollment_id="EN-0027", reason="D4")


def _m4_step_expire(env):
    set_now(env, "2026-06-12 19:00")  # SIG-015（18:00）过期链
    env.sync_tools()


def _m4_drop(student_id, enrollment_id):
    def _call(env):
        env.make_tool_call(tool_name="drop_course", requestor="assistant",
                           student_id=student_id, enrollment_id=enrollment_id,
                           reason="D4")
    return _call


def _m4_step_expire_lvl2(env):
    # C 的递补单截止 = expire_lvl1 时刻（06-12 19:00）+24h = 06-13 19:00
    set_now(env, "2026-06-13 20:00")
    env.sync_tools()


# D4① 确定性穷举的定向步进空间（5 步 → 全部一/二步序列 5+25=30 条路径）
M4_STEPS = [
    ("join_C", _m4_step_join),
    ("abandon_A", _m4_step_abandon),
    ("confirm_A", _m4_step_confirm),
    ("drop_EN0027", _m4_step_drop),
    ("expire_promote", _m4_step_expire),
]

# D4② 定向下探序列：候补计数 2→0、开课计数 4→0，逐骤断言无负数（时钟单调）
M4_COUNTDOWN = [
    ("abandon_A", _m4_step_abandon),                     # 候补 2→1（A 弃）
    ("rejoin_C", _m4_step_join),                         # 位次 3，候补 1→2
    ("drop_EN-0027", _m4_drop("S20230101", "EN-0027")),   # 计数 4→3，EN-0159 收递补单
    ("drop_EN-0041", _m4_drop("S20230102", "EN-0041")),   # 3→2（队列守卫不抢跑）
    ("drop_EN-0242", _m4_drop("S20240202", "EN-0242")),   # 2→1
    ("drop_EN-0305", _m4_drop("S20230302", "EN-0305")),   # 1→0
    ("expire_lvl1", _m4_step_expire),                     # EN-0159 弃 2→1；C 收递补单
    ("expire_lvl2", _m4_step_expire_lvl2),                # C 弃 1→0
]


class TestD4Invariants:
    """不变量测试：不引新依赖（无 hypothesis），确定性穷举定向构造小空间。
    判据沿用 A1：③活跃候补位次唯一；②各 offering 计数==行数且无负数。"""

    @staticmethod
    def _count_problems(env):
        db = env.tools.db
        problems = []
        for off in db.course_offerings.values():
            rows = [e for e in db.enrollments.values() if e.offering_id == off.offering_id]
            enrolled = sum(1 for e in rows if e.status == "已选")
            waitlisted = sum(1 for e in rows if e.status == "候补中")
            if off.enrolled_count < 0 or off.waitlist_count < 0:
                problems.append(f"negative {off.offering_id}: enrolled={off.enrolled_count} "
                                f"waitlist={off.waitlist_count}")
            if off.enrolled_count != enrolled:
                problems.append(f"enrolled_count {off.offering_id}: {off.enrolled_count} != rows {enrolled}")
            if off.waitlist_count != waitlisted:
                problems.append(f"waitlist_count {off.offering_id}: {off.waitlist_count} != rows {waitlisted}")
        return problems

    @staticmethod
    def _position_problems(env):
        db = env.tools.db
        problems = []
        for off in db.course_offerings.values():
            positions = [e.waitlist_position for e in db.enrollments.values()
                         if e.offering_id == off.offering_id and e.status == "候补中"]
            if len(positions) != len(set(positions)):
                problems.append(f"duplicate positions {off.offering_id}: {positions}")
        return problems

    def test_positions_unique_over_exhaustive_constructions(self):
        """D4①：5 个定向步进的全部一/二步序列（5+25=30 条确定性穷举路径），
        每步后断言活跃候补位次唯一（A1③ 判据）。守卫拒绝（ValueError）＝零写
        （D1 守卫前置），状态不变，不变量仍须成立。"""
        for length in (1, 2):
            for seq in itertools.product(M4_STEPS, repeat=length):
                env = get_environment()
                set_now(env, "2026-03-04 10:00")  # 窗口内锚点（48h 截止 03-13 23:59）
                names = []
                for name, step in seq:
                    names.append(name)
                    try:
                        step(env)
                    except ValueError:
                        pass  # 守卫拒绝＝零写，状态不变
                    problems = self._position_problems(env)
                    assert not problems, f"steps={'→'.join(names)}: {problems}"

    def test_counts_never_negative_over_constructions(self):
        """D4②：定向序列把计数下探到底（放弃→重入→连退 4 人至 0→两级过期链
        候补清零），每步后断言各 offering 计数==行数且无负数（A1② 口径）。"""
        env = get_environment()
        set_now(env, "2026-03-04 10:00")
        names = []
        for name, step in M4_COUNTDOWN:
            step(env)
            names.append(name)
            problems = self._count_problems(env)
            assert not problems, f"steps={'→'.join(names)}: {problems}"
        # 终态：开课计数与候补计数都已下探到 0，从未为负
        off = env.tools.db.course_offerings["OF-2026SP-402-1"]
        assert off.enrolled_count == 0 and off.waitlist_count == 0
