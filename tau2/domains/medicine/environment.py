# Copyright Sierra
import json
import os
from typing import Optional

from tau2.data_model.tasks import Task
from tau2.domains.medicine.data_model import MedicineDB
from tau2.domains.medicine.tools import MedicineTools
from tau2.environment.environment import Environment


def get_environment(
    db: Optional[MedicineDB] = None,
    solo_mode: bool = False,
) -> Environment:
    if solo_mode:
        raise ValueError("Medicine domain does not support solo mode")
    if db is None:
        db = MedicineDB.load('/lustre/fsw/portfolios/nvr/users/hongjins/generate_data_tool_use/medicine/db.json')
    tools = MedicineTools(db)
    with open('/lustre/fsw/portfolios/nvr/users/hongjins/generate_data_tool_use/medicine/policy.md', "r") as fp:
        policy = fp.read()
    return Environment(
        domain_name="medicine",
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
    return_tasks = []
    skip = 0
    for task in tasks:
        try:
            processed_task = Task.model_validate(task)
            assert processed_task.id==task['id'],f"{processed_task.id}, {task['id']}"
            return_tasks.append(processed_task)
        except Exception as error:
            print(error)
            skip += 1
    print('skip:',skip)
    # return_tasks = [Task.model_validate(task) for task in tasks]
    # print(39,return_tasks[0])
    # exit(0)
    return return_tasks
