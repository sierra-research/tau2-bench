# Copyright Sierra
import json
import os
from typing import Optional

from tau2.data_model.tasks import Task
from tau2.domains.airline.data_model import FlightDB
from tau2.domains.airline.tools import AirlineTools
from tau2.domains.airline.utils import (
    AIRLINE_DB_PATH,
    AIRLINE_POLICY_PATH,
    AIRLINE_TASK_SET_PATH,
)
from tau2.environment.environment import Environment


def get_environment(
    db: Optional[FlightDB] = None,
    solo_mode: bool = False,
) -> Environment:
    if solo_mode:
        raise ValueError("Airline domain does not support solo mode")
    if db is None:
        db = FlightDB.load('/lustre/fsw/portfolios/nvr/users/hongjins/data/tool_use/original/tau2/domains/airline/db.json')
    tools = AirlineTools(db)
    with open('/lustre/fsw/portfolios/nvr/users/hongjins/data/tool_use/original/tau2/domains/airline/policy.md', "r") as fp:
        policy = fp.read()
    return Environment(
        domain_name="airline",
        policy=policy,
        tools=tools,
    )


def get_tasks(task_path,save_to) -> list[Task]:
    print(f"Load tasks from {task_path}")
    with open(task_path, "r") as fp:
        tasks = json.load(fp)
    tasks_dict = {}
    for t in tasks:
        tasks_dict[t['id']] = t
    save_dir = str(save_to)
    assert save_dir.endswith('.json'),f"{save_dir}"
    save_dir = save_dir[:-len('.json')]
    processed_task_ids = set()
    if os.path.isdir(save_dir):
        for subfile in os.listdir(save_dir):
            if subfile.endswith('.json'):
                try:
                    with open(os.path.join(save_dir,subfile)) as f:
                        o = json.load(f)
                    processed_task_ids.add(o['task_id'])
                except:
                    continue
    updated_tasks = []
    for k,v in tasks_dict.items():
        if not k in processed_task_ids:
            updated_tasks.append(v)
    print("Total tasks:",len(tasks))
    print("Tasks to run:",len(updated_tasks))
    tasks = updated_tasks
    # print(37,tasks[0])
    return_tasks = [Task.model_validate(task) for task in tasks]
    # print(39,return_tasks[0])
    # exit(0)
    return return_tasks
