"""campus 域 user 侧数据模型（学生小程序数据）。

dual-control 的本质：Agent 侧 WRITE 只能**生成** pending_signatures 行，不能翻其状态；
材料上传（uploads）与签署确认只发生在 user 库。user 写动作全部须进
evaluation_criteria.actions（requestor=user），否则 DB 双哈希必挂（复现笔记 §5）。
"""

from typing import Any, Dict

from pydantic import Field

from tau2.domains.campus.data_model import (
    SignatureRow,
    TodoRow,
    UploadRow,
)
from tau2.environment.db import DB


class UserDB(DB):
    """user 侧 3 表（学生小程序数据）。"""

    app_todos: Dict[str, TodoRow] = {}
    pending_signatures: Dict[str, SignatureRow] = {}
    uploads: Dict[str, UploadRow] = {}

    def get_statistics(self) -> Dict[str, Any]:
        """Get the statistics of the database."""
        return {
            "num_app_todos": len(self.app_todos),
            "num_pending_signatures": len(self.pending_signatures),
            "num_uploads": len(self.uploads),
        }


def get_user_db():
    from tau2.domains.campus.utils import CAMPUS_USER_DB_PATH

    return UserDB.load(str(CAMPUS_USER_DB_PATH))


if __name__ == "__main__":
    print(get_user_db().get_statistics())
