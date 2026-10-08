"""Campus replay-determinism tests.

Covers the dual-hash determinism rule:
- strict_replay round-trip: a trajectory of agent/user writes (incl. rejected
  writes and time jumps) replays content-identical on a fresh environment;
- initial_state override path (agent_data/user_data incl. "student_id" binding
  and current_time) produces the same end state as mutating directly;
- gold-vs-predicted: replaying ONLY the write actions in reference order yields
  the same agent-DB and user-DB hashes as the full trajectory (extra reads free);
- settlement is call-count independent: interleaving reads cannot change results
  (time/state-driven only).
"""

import pytest

from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage
from tau2.data_model.tasks import EnvFunctionCall, InitializationData
from tau2.domains.campus.environment import get_environment


def call(env, requestor, name, arguments, i):
    tc = ToolCall(id=f"call_{i}", name=name, arguments=arguments, requestor=requestor)
    tm = env.get_response(tc)
    msg_cls = UserMessage if requestor == "user" else AssistantMessage
    role = "user" if requestor == "user" else "assistant"
    msg = msg_cls(role=role, content="", tool_calls=[tc])
    return [msg, tm]


def history(env, steps):
    msgs = []
    for i, (requestor, name, args) in enumerate(steps):
        msgs += call(env, requestor, name, args, i)
    return msgs


TRAJ = [
    # agent: deferral for conflicting exams, then user signs it (dual-control)
    ("assistant", "submit_deferral",
     {"student_id": "S20230103", "offering_id": "OF-2026SP-203-1", "exam_id": "EX-0049",
      "reason_type": "冲突", "filing_type": "考前正常"}),
    ("assistant", "get_service_requests", {"student_id": "S20230103", "kind": "deferral"}),
    ("user", "check_student_app", {}),
    ("user", "confirm_action", {"sig_id": "SIG-017"}),
    # rejected write (zero mutation) must also replay error-content identical
    ("assistant", "withdraw_application",
     {"student_id": "S20230103", "kind": "deferral", "record_id": "DF-007"}),
    ("assistant", "request_certificate", {"student_id": "S20230103", "cert_type": "在读证明"}),
    ("assistant", "create_ticket",
     {"student_id": "S20230103", "category": "咨询", "module": "证明",
      "title": "多久能拿到", "content": "学号S20230103 咨询证明时限"}),
]

WRITE_STEPS = [
    ("assistant", "submit_deferral",
     {"student_id": "S20230103", "offering_id": "OF-2026SP-203-1", "exam_id": "EX-0049",
      "reason_type": "冲突", "filing_type": "考前正常"}),
    ("user", "confirm_action", {"sig_id": "SIG-017"}),
    ("assistant", "request_certificate", {"student_id": "S20230103", "cert_type": "在读证明"}),
    ("assistant", "create_ticket",
     {"student_id": "S20230103", "category": "咨询", "module": "证明",
      "title": "多久能拿到", "content": "学号S20230103 咨询证明时限"}),
]

BIND = EnvFunctionCall(env_type="user", func_name="bind_student",
                       arguments={"student_id": "S20230103"})


class TestStrictReplay:
    def test_roundtrip_content_identical(self):
        env1 = get_environment()
        env1.set_state(None, [BIND], [])
        msgs = history(env1, TRAJ)
        # 重放全轨迹到全新环境：strict=True 下逐条 WRITE 输出必须一字不差
        env2 = get_environment()
        env2.set_state(None, [BIND], msgs, strict=True)
        assert env1.get_db_hash() == env2.get_db_hash()
        assert env1.get_user_db_hash() == env2.get_user_db_hash()

    def test_no_system_clock_read(self):
        """时间全取自 env.current_time：同轨迹不同 '墙钟' 环境下哈希不变（两次重放对比）。"""
        env1 = get_environment()
        env1.set_state(None, [BIND], [])
        msgs = history(env1, TRAJ)
        env2 = get_environment()
        env3 = get_environment()
        env2.set_state(None, [BIND], msgs, strict=True)
        env3.set_state(None, [BIND], list(msgs), strict=True)
        assert env2.get_db_hash() == env3.get_db_hash()

    def test_settlement_independent_of_extra_reads(self):
        """结算只看状态/时间：多插只读调用不改变终态（金标/预测写序列一致即可）。"""
        padded = list(TRAJ)
        padded.insert(2, ("assistant", "get_grades", {"student_id": "S20230103"}))
        padded.insert(4, ("user", "check_student_app", {}))
        runner = get_environment()
        runner.set_state(None, [BIND], [])
        m1 = history(runner, TRAJ)
        runner2 = get_environment()
        runner2.set_state(None, [BIND], [])
        m2 = history(runner2, padded)
        env1 = get_environment()
        env2 = get_environment()
        env1.set_state(None, [BIND], m1)
        env2.set_state(None, [BIND], m2)
        assert env1.get_db_hash() == env2.get_db_hash()
        assert env1.get_user_db_hash() == env2.get_user_db_hash()


class TestGoldVsPredicted:
    def test_gold_replay_of_write_subset_matches_full_trajectory(self):
        """金标环境只重放 evaluation actions（写序列），预测环境重放整条轨迹——
        无多余写时两库哈希全等（DB 分量口径）。"""
        env = get_environment()
        env.set_state(None, [BIND], [])
        full_msgs = history(env, TRAJ)
        predicted = get_environment()
        predicted.set_state(None, [BIND], full_msgs)

        gold = get_environment()
        gold.set_state(None, [BIND], [])
        for i, (requestor, name, args) in enumerate(WRITE_STEPS):
            gold.make_tool_call(name, requestor=requestor, **args)
            gold.sync_tools()
        assert predicted.get_db_hash() == gold.get_db_hash()
        assert predicted.get_user_db_hash() == gold.get_user_db_hash()

    def test_extra_user_write_breaks_db_equality(self):
        """复现防线：预测侧多一个用户写（金标未含）→ user 库哈希必须不等。"""
        env = get_environment()
        env.set_state(None, [BIND], [])
        steps = WRITE_STEPS + [("user", "check_student_app", {}),
                               ("assistant", "create_ticket",
                                {"student_id": "S20230103", "category": "建议", "module": "证明",
                                 "title": "建议", "content": "学号S20230103 建议"}),
                               ("user", "reject_suggestion", {"sig_id": "SIG-013"})]
        msgs = history(env, steps)
        predicted = get_environment()
        predicted.set_state(None, [BIND], msgs)

        gold = get_environment()
        gold.set_state(None, [BIND], [])
        for requestor, name, args in WRITE_STEPS:
            gold.make_tool_call(name, requestor=requestor, **args)
        gold.sync_tools()
        assert predicted.get_db_hash() != gold.get_db_hash()


class TestInitializationDataOverride:
    def test_override_replay_matches_direct_mutation(self):
        """initialization_data 覆写个别行+current_time 的重放路径与判分一致（复现笔记 §6.2）：
        覆写 env.current_time 后重放同一写序列，与"先改时间再跑"的环境哈希全等。"""
        init = InitializationData(
            agent_data={"env": {"current_time": "2026-06-30 18:00", "term": "2026SP"}},
            user_data={"student_id": "S20230103"},
        )
        env_a = get_environment()
        env_a.set_state(init, None, [])
        msgs_a = history(env_a, WRITE_STEPS)

        env_b = get_environment()
        env_b.set_state(init, None, msgs_a, strict=True)
        assert env_a.get_db_hash() == env_b.get_db_hash()
        assert env_a.get_user_db_hash() == env_b.get_user_db_hash()

    def test_user_row_override_applies_to_user_db_only(self):
        init = InitializationData(
            agent_data=None,
            user_data={"student_id": "S20250403",
                       "pending_signatures": {
                           "SIG-013": {
                               "sig_id": "SIG-013", "student_id": "S20250403",
                               "doc_type": "缓考申请单", "ref_type": "deferral_requests",
                               "ref_id": "DF-006", "summary": "改", "status": "已过期",
                               "created_at": "2026-06-11 09:00", "deadline_at": None,
                               "acted_at": None,
                           }}},
        )
        env = get_environment()
        env.set_state(init, None, [])
        assert env.user_tools.db.pending_signatures["SIG-013"].status == "已过期"
        # agent 主库不受影响
        assert env.tools.db.deferral_requests["DF-006"].status == "待签署"
