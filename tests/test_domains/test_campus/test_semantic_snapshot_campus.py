"""Semantic snapshot tests: inline guidance pinned to implementation.

Task purposes and relevant_policies carry behavioral guidance that nothing
else enforces against the code — prose and implementation can drift apart.
Three claims from the tool docstrings are pinned here:

1. **ticket precheck boundary** — only the "申诉＋成绩＋逾期" combination is
   intercepted by ``create_ticket``; 咨询 rows and non-grade appeals are
   never intercepted by the grade-overdue check;
2. **reject-suggestion coverage** — rejecting a pending document withdraws
   the corresponding application, for every document face: DF (deferral),
   APP (scholarship), waitlist confirmation (abandon + queue promotion),
   CE (certificate via proxy-auth signature);
3. **deferral chain order** — submit → upload (user side) → confirm →
   settle reaches a correct final state.

Group 3 is already pinned end-to-end by
``test_tools_campus.py::TestDeferralChain::test_upload_before_sign_completes_upload_todo``
(the exact submit → upload → confirm → settle order, final state 通过) and
``test_tools_campus.py::TestDeferralChain::test_illness_valid_material_approve``
(the confirm-first order); it is referenced here rather than duplicated —
mapping recorded in the W16 batch record.
"""

import pytest

from tau2.domains.campus.environment import get_environment
from tau2.domains.campus.tools import E_REVIEW_EXPIRED


@pytest.fixture
def env():
    return get_environment()


def set_now(env, ts):
    env.tools.db.env.current_time = ts


# ------------------------------------------------- ① create_ticket 预检组合边界


class TestTicketPrecheckBoundary:
    def test_only_appeal_grade_overdue_is_intercepted(self, env):
        """钉死"仅拦'申诉+成绩+逾期'"：三个边界格逐格过。
        S20230103 的成绩全部超窗（种子时点 2026-06-12），是"逾期"侧固定夹具。"""
        t = env.tools
        # ① 申诉＋成绩＋逾期 → 抛 E_REVIEW_EXPIRED（逐字目录文案）
        with pytest.raises(ValueError, match=E_REVIEW_EXPIRED):
            t.create_ticket("S20230103", "申诉", "成绩", "复核", "学号S20230103")
        # ② 咨询＋逾期 → 建单成功（逾期状态不拦咨询类）
        r2 = t.create_ticket("S20230103", "咨询", "成绩", "了解", "学号S20230103")
        assert r2.ticket_id in t.db.tickets
        # ③ 申诉＋非成绩（选课）＋逾期 → 建单成功（预检只在"成绩"模块触发）
        r3 = t.create_ticket("S20230103", "申诉", "选课", "复核选课", "学号S20230103")
        assert r3.ticket_id in t.db.tickets


# ------------------------------------------------- ② reject_suggestion 撤回覆盖


class TestRejectSuggestionWithdrawsCorrespondingApplication:
    def test_df_reject_withdraws_and_frees_deferral_quota(self, env):
        """DF 单拒签＝撤回缓考申请，且不占本学期该课程额度。"""
        t, u = env.tools, env.user_tools
        # 种子 DF-006（待签署）挡第二次申请 → 拒签前先钉"占额度"
        with pytest.raises(ValueError, match="本学期已申请过缓考"):
            t.submit_deferral("S20250403", "OF-2026SP-702-1", "EX-0039", "冲突", "考前正常")
        u.bind_student("S20250403")
        r = u.reject_suggestion("SIG-013", reason="改变计划")
        assert r.business_status == "已撤回"
        assert "不占该课程本学期缓考额度" in r.message
        assert t.db.deferral_requests["DF-006"].status == "已撤回"
        # 撤回后额度释放：同一开课本学期可再次申请（第13条仅一次，已撤回行不计）
        r2 = t.submit_deferral("S20250403", "OF-2026SP-702-1", "EX-0039", "冲突", "考前正常")
        assert r2.status == "待签署"

    def test_app_reject_withdraws_scholarship_application(self, env):
        """APP 单拒签＝撤回奖助申请（签署确认前，第26条）。"""
        t, u = env.tools, env.user_tools
        u.bind_student("S20240202")
        r = u.reject_suggestion("SIG-004", reason="不申请了")
        assert r.business_status == "已撤回"
        assert "已于签署前撤回（第26条）" in r.message
        assert t.db.scholarship_apps["APP-004"].status == "已撤回"

    def test_waitlist_reject_abandons_and_promotes_next(self, env):
        """候补确认单拒签＝放弃候补（abandon_count+1）且队列顺延递补下一位。"""
        t, u = env.tools, env.user_tools
        set_now(env, "2026-03-04 10:00")  # 补退选窗口内，先空出一个名额
        t.drop_course("S20230101", "EN-0027")
        # 前序 SIG-015 未处理：队列不抢跑，EN-0159 此刻不得拿到递补单
        assert not [s for s in t.user_db.pending_signatures.values()
                    if s.doc_type == "候补递补确认单" and s.ref_id == "EN-0159"]
        u.bind_student("S20230104")
        r = u.reject_suggestion("SIG-015", reason="已选别的课")
        assert r.business_status == "失效"
        assert t.db.students["S20230104"].waitlist_abandon_count == 1
        assert t.db.enrollments["EN-0070"].status == "失效"
        assert t.db.course_offerings["OF-2026SP-402-1"].waitlist_count == 1
        # 顺延：位次 2 的 EN-0159 接过递补确认单（24h 内确认）
        promoted = [s for s in t.user_db.pending_signatures.values()
                    if s.doc_type == "候补递补确认单" and s.ref_id == "EN-0159"
                    and s.status == "待确认"]
        assert len(promoted) == 1
        assert t.db.enrollments["EN-0159"].promote_sig_id == promoted[0].sig_id

    def test_ce_reject_withdraws_certificate(self, env):
        """CE 单拒签（代领授权书通道）＝证明申请撤回（出具前，第30条）。"""
        t, u = env.tools, env.user_tools
        r = t.request_certificate(
            "S20230103", "在读证明", delivery="委托代领",
            proxy_name="张同", proxy_id_masked="****2468")
        assert r.status == "待签署"
        u.bind_student("S20230103")
        rr = u.reject_suggestion(r.sig_id, reason="改为自取")
        assert rr.business_status == "已撤回"
        assert f"证明申请 {r.cert_id} 于出具前撤回（第30条）" in rr.message
        assert t.db.certificates[r.cert_id].status == "已撤回"
