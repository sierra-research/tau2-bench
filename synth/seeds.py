"""Load the 74 retail TRAIN-split tasks as transformation seeds.

The augmentation pipeline (build_augmented.py) takes each seed and injects one
failure-mode pattern. A `Seed` exposes the parsed grounding the pattern builders
need: the user, the target order, the write action(s), and the editable
instruction fields. Reads in the seed's gold action list are ignored — only the
WRITE actions define the DB target.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from lib import DB
from tau2.domains.retail.environment import get_tasks_split
from tau2.domains.retail.utils import RETAIL_TASK_SET_PATH
from tau2.utils import load_file

WRITE_TOOLS = {
    "exchange_delivered_order_items",
    "modify_pending_order_items",
    "return_delivered_order_items",
    "cancel_pending_order",
    "modify_pending_order_address",
    "modify_pending_order_payment",
    "modify_user_address",
}
VARIANT_TOOLS = {"exchange_delivered_order_items", "modify_pending_order_items"}


@dataclass
class Seed:
    task: dict
    write_actions: list[dict]
    user_id: Optional[str]
    target_order_id: Optional[str]

    @property
    def id(self) -> str:
        return self.task["id"]

    @property
    def user(self):
        return DB.users.get(self.user_id) if self.user_id else None

    @property
    def instructions(self) -> dict:
        instr = self.task["user_scenario"].get("instructions")
        return instr if isinstance(instr, dict) else {"task_instructions": str(instr)}

    @property
    def reason_for_call(self) -> str:
        return self.instructions.get("reason_for_call", "")

    @property
    def primary_write(self) -> Optional[dict]:
        return self.write_actions[-1] if self.write_actions else None

    @property
    def is_variant_seed(self) -> bool:
        return bool(self.primary_write) and self.primary_write["name"] in VARIANT_TOOLS

    def resolved(self) -> bool:
        """Has a single resolvable user and at least one write (DB target)."""
        return self.user is not None and bool(self.write_actions)


def _resolve_user(write_actions: list[dict]) -> Optional[str]:
    for a in write_actions:
        args = a.get("arguments", {})
        oid = args.get("order_id")
        if oid and oid in DB.orders:
            return DB.orders[oid].user_id
        if a["name"] == "modify_user_address" and args.get("user_id"):
            return args["user_id"]
    return None


def load_seeds() -> list[Seed]:
    train = set(get_tasks_split()["train"])
    seeds = []
    for t in load_file(RETAIL_TASK_SET_PATH):
        if t.get("id") not in train:
            continue
        actions = (t.get("evaluation_criteria") or {}).get("actions") or []
        writes = [a for a in actions if a["name"] in WRITE_TOOLS]
        uid = _resolve_user(writes)
        target = None
        if writes:
            target = writes[-1].get("arguments", {}).get("order_id")
        seeds.append(Seed(task=t, write_actions=writes, user_id=uid, target_order_id=target))
    return seeds


if __name__ == "__main__":
    seeds = load_seeds()
    resolved = [s for s in seeds if s.resolved()]
    variant = [s for s in resolved if s.is_variant_seed]
    print(f"train seeds: {len(seeds)} | resolved (user+write): {len(resolved)} "
          f"| variant seeds: {len(variant)}")
    unresolved = [s.id for s in seeds if not s.resolved()]
    print("unresolved seed ids (no single user / no write):", unresolved)
