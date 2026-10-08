"""Environment for the campus domain (Tau2-ZH).

Wiring notes:
- Dual DB: CampusTools owns CampusDB (`db`) and references UserDB (`user_db`) to GENERATE
  signature/todo hand-off rows; CampusUserTools owns UserDB (`db`) and references CampusDB
  (`main_db`) to advance business state. One DB pair instance is shared across both
  toolkits; `get_db_hash`/`get_user_db_hash` therefore hash agent/user sides separately
  (dual-hash equality).
- `set_state` is overridden: the official base syncs the two toolkit db pointers onto one
  instance after `update_db` (single-DB assumption). Campus is a genuine two-DB domain
  (CampusDB ≠ UserDB, both extra="forbid"), so we apply agent_data/user_data to their own
  DBs, re-share the pair references, and delegate replay of initialization_actions and the
  message history to the base implementation with initialization_data=None.
- `sync_tools` = deterministic settlement hook: called after every tool call in live
  runs (orchestrator), after every replayed mutating call, and once at set_state end —
  gold and predicted environments converge identically because settlement is a pure
  function of DB state + env.current_time.
"""

from pathlib import Path
from typing import Optional

from tau2.data_model.tasks import InitializationData, Task
from tau2.domains.campus.data_model import CampusDB
from tau2.domains.campus.tools import CampusTools
from tau2.domains.campus.user_data_model import UserDB
from tau2.domains.campus.user_tools import CampusUserTools
from tau2.domains.campus.utils import (
    CAMPUS_DB_PATH,
    CAMPUS_POLICY_PATH,
    CAMPUS_TASK_SET_PATH,
    CAMPUS_USER_DB_PATH,
)
from tau2.environment.environment import Environment
from tau2.utils import load_file


class CampusEnvironment(Environment):
    tools: CampusTools
    user_tools: CampusUserTools

    def __init__(
        self,
        domain_name: str,
        policy: str,
        tools: CampusTools,
        user_tools: CampusUserTools,
    ):
        super().__init__(domain_name, policy, tools, user_tools)

    def sync_tools(self):
        """确定性结算钩子：见模块 docstring。"""
        self.tools.settle()

    def set_state(
        self,
        initialization_data: Optional[InitializationData],
        initialization_actions,
        message_history,
        strict: bool = True,
    ):
        agent_data = None
        user_data = None
        if initialization_data is not None:
            agent_data = initialization_data.agent_data
            user_data = initialization_data.user_data
        if agent_data is not None:
            self.tools.update_db(agent_data)
        if user_data is not None:
            # 约定通道：user_data 可携带非 DB 键 "student_id" 绑定学生端身份
            # （UserDB 为 extra=forbid，须在覆写前摘出；亦可用 initialization_actions
            # 调用 bind_student，两种等价）。
            if "student_id" in user_data:
                self.user_tools.bind_student(user_data["student_id"])
                user_data = {k: v for k, v in user_data.items() if k != "student_id"}
            self.user_tools.update_db(user_data)
        # 覆写会产生新 db 实例：重新共享同一对实例（双库域不做指针合并）
        self.user_tools.main_db = self.tools.db
        self.tools.user_db = self.user_tools.db
        self.user_tools.agent_tools = self.tools
        super().set_state(
            initialization_data=None,
            initialization_actions=initialization_actions,
            message_history=message_history,
            strict=strict,
        )

    # ------------------------------------------------- runner 钩子

    def advance_time(self, hours: int = 0, days: int = 0) -> str:
        """环境快进（非工具动作、不进用户轨迹）：推进 env.current_time 后由
        sync_tools 结算'待审→通过/驳回'等规则化终态。"""
        new_time = self.tools.advance_time(hours=hours, days=days)
        self.sync_tools()
        return new_time


def get_environment(
    db: Optional[CampusDB] = None,
    user_db: Optional[UserDB] = None,
    solo_mode: bool = False,
) -> CampusEnvironment:
    # 上游同款约定（airline/retail/banking_knowledge）：不支持 solo 即入口抛错，
    # 不走 set_solo_mode 静默声明路径（campus 50/50 题均无 solo 题面）。
    if solo_mode:
        raise ValueError("Solo mode not supported for campus")
    if db is None:
        db = CampusDB.load(str(CAMPUS_DB_PATH))
    if user_db is None:
        user_db = UserDB.load(str(CAMPUS_USER_DB_PATH))
    tools = CampusTools(db, user_db)
    user_tools = CampusUserTools(user_db, db, tools)
    policy = Path(CAMPUS_POLICY_PATH).read_text(encoding="utf-8")
    env = CampusEnvironment(
        domain_name="campus",
        policy=policy,
        tools=tools,
        user_tools=user_tools,
    )
    return env


def get_tasks(task_split_name: Optional[str] = "base") -> list[Task]:
    """Load campus tasks from the packaged tasks.json（缺失时返回空列表，
    不阻断域注册）。"""
    if not Path(CAMPUS_TASK_SET_PATH).exists():
        return []
    tasks = [Task.model_validate(task) for task in load_file(str(CAMPUS_TASK_SET_PATH))]
    if task_split_name is None:
        return tasks
    split_file = Path(CAMPUS_TASK_SET_PATH).parent / "split_tasks.json"
    if not split_file.exists():
        return tasks
    splits = load_file(str(split_file))
    if task_split_name not in splits:
        raise ValueError(
            f"Invalid task split name: {task_split_name}. Valid splits are: {splits.keys()}"
        )
    return [t for t in tasks if t.id in splits[task_split_name]]
