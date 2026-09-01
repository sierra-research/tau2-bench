# -*- coding: utf-8 -*-
"""Печать диалога и разбора награды из results.json.

Использование:
    uv run python scripts/show_dialog.py                # последний прогон
    uv run python scripts/show_dialog.py <путь|подстрока> [task_id]
"""
import glob
import json
import sys

ROLE = {"user": "КЛИЕНТ", "assistant": "АГЕНТ", "tool": "СРЕДА", "system": "СИСТЕМА"}


def find_results(pattern):
    paths = sorted(glob.glob("data/simulations/*/results.json"))
    if pattern:
        hits = [p for p in paths if pattern in p]
        if not hits:
            sys.exit(f"Не найдено прогонов по подстроке {pattern!r}")
        return hits[-1]
    if not paths:
        sys.exit("В data/simulations/ нет результатов")
    return paths[-1]


def show(sim: dict) -> None:
    print("=" * 78)
    print(f"Задача: {sim.get('task_id')}   trial {sim.get('trial')}")
    ri = sim.get("reward_info") or {}
    print(f"Награда: {ri.get('reward')}   разбор: {ri.get('reward_breakdown')}")
    print(f"Ошибок: {sim.get('num_errors')}   длительность, с: {sim.get('duration')}")
    print("=" * 78)

    for m in sim.get("messages") or []:
        role = ROLE.get(m.get("role"), m.get("role", "?"))
        content = (m.get("content") or "").strip()
        if content:
            print(f"\n[{role}] {content}")
        for tc in m.get("tool_calls") or []:
            args = tc.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    pass
            print(f"\n[{role} → инструмент] {tc.get('name')}({json.dumps(args, ensure_ascii=False)})")
        if m.get("role") == "tool" and m.get("error"):
            print("   ↑ ошибка инструмента")

    print("\n" + "-" * 78)
    print("ПРОВЕРКИ")
    db = ri.get("db_check")
    if db:
        print(f"  DB совпал: {db.get('db_match')}")
    for c in ri.get("communicate_checks") or []:
        mark = "✅" if c.get("met") else "❌"
        print(f"  {mark} сказать «{c.get('info')}»")
    for a in ri.get("env_assertions") or []:
        mark = "✅" if a.get("met") else "❌"
        ea = a.get("env_assertion") or {}
        print(f"  {mark} {ea.get('func_name')}({json.dumps(ea.get('arguments'), ensure_ascii=False)})")


def main() -> None:
    pattern = sys.argv[1] if len(sys.argv) > 1 else None
    task_id = sys.argv[2] if len(sys.argv) > 2 else None
    path = find_results(pattern)
    print(f"Файл: {path}\n")
    data = json.load(open(path, encoding="utf-8"))
    sims = data.get("simulations") or []
    if task_id:
        sims = [s for s in sims if s.get("task_id") == task_id]
    if not sims:
        sys.exit("Симуляций не найдено")
    for sim in sims:
        show(sim)


if __name__ == "__main__":
    main()
