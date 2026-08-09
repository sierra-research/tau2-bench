import json
import random
from argparse import ArgumentParser
from collections import defaultdict
from hashlib import sha256

from tau2.data_model.tasks import Task
from tau2.domains.telecom.tasks.mms_issues import mms_issue_task_manager
from tau2.domains.telecom.tasks.mobile_data_issues import mobile_data_task_manager
from tau2.domains.telecom.tasks.service_issues import service_issues_task_manager
from tau2.domains.telecom.tasks.utils import (
    get_customer_from_task_id,
    get_persona_from_task_id,
)
from tau2.utils import DATA_DIR


def build_task_splits(
    sampled_tasks: list[Task], small_tasks: list[Task], full_tasks: list[Task]
) -> dict[str, list[str]]:
    """Build stable customer-stratified release splits without outcome knowledge."""
    train: list[str] = []
    test: list[str] = []
    by_customer_intent: dict[tuple[str, str], list[Task]] = defaultdict(list)
    for task in sampled_tasks:
        intent = task.id.split("]", 1)[0].removeprefix("[")
        by_customer_intent[(get_customer_from_task_id(task.id), intent)].append(task)
    for tasks in by_customer_intent.values():
        ordered = sorted(
            tasks,
            key=lambda task: sha256(
                f"tau3-telecom-multicustomer-v1\0{task.id}".encode()
            ).digest(),
        )
        test_count = max(1, len(ordered) // 3)
        test.extend(task.id for task in ordered[:test_count])
        train.extend(task.id for task in ordered[test_count:])
    return {
        "base": sorted(task.id for task in sampled_tasks),
        "train": sorted(train),
        "test": sorted(test),
        "small": sorted(task.id for task in small_tasks),
        "full": sorted(task.id for task in full_tasks),
    }


def create_tasks(
    save_tasks: bool = True, max_count_per_bin: int = 3, seed: int = 42
) -> list[Task]:
    rng = random.Random(seed)
    all_tasks: list[Task] = []
    mobile_data_tasks = mobile_data_task_manager.create_tasks(save_tasks=False)
    print(f"Number of mobile data issue tasks: {len(mobile_data_tasks)}")
    all_tasks.extend(mobile_data_tasks)

    service_tasks = service_issues_task_manager.create_tasks(save_tasks=False)
    print(f"Number of service issue tasks: {len(service_tasks)}")
    all_tasks.extend(service_tasks)

    mms_tasks = mms_issue_task_manager.create_tasks(save_tasks=False)
    print(f"Number of mms issue tasks: {len(mms_tasks)}")
    all_tasks.extend(mms_tasks)

    print(f"Number of tasks: {len(all_tasks)}")

    file = DATA_DIR / "tau2" / "domains" / "telecom" / f"tasks_full.json"
    if save_tasks:
        with open(file, "w") as f:
            json.dump([t.model_dump() for t in all_tasks], f, indent=2)

    # Build tasks with attributes
    tasks_with_attrs = []
    for intent_tasks, intent in [
        (mobile_data_tasks, "mobile_data"),
        (service_tasks, "service"),
        (mms_tasks, "mms"),
    ]:
        for task in intent_tasks:
            num_subtasks = len(task.id.split("|"))
            tasks_with_attrs.append(
                {
                    "task": task,
                    "intent": intent,
                    "num_subtasks": num_subtasks,
                    "persona": get_persona_from_task_id(task.id),
                    "customer": get_customer_from_task_id(task.id),
                }
            )

    file_small = DATA_DIR / "tau2" / "domains" / "telecom" / f"tasks_small.json"
    small_tasks = [t["task"] for t in tasks_with_attrs if t["num_subtasks"] == 1]
    print(f"Number of tasks in small set: {len(small_tasks)}")
    if save_tasks:
        with open(file_small, "w") as f:
            json.dump([t.model_dump() for t in small_tasks], f, indent=2)

    file_sampled = DATA_DIR / "tau2" / "domains" / "telecom" / f"tasks.json"
    tasks_by_bins = defaultdict(list)
    for task in tasks_with_attrs:
        if task["num_subtasks"] < 2:  # We only keep tasks with at least 2 subtasks
            continue
        tasks_by_bins[
            (
                task["customer"],
                task["intent"],
                task["num_subtasks"],
                task["persona"],
            )
        ].append(task["task"])

    # sample $n$ tasks per intent, difficulty level, and persona
    sampled_tasks = []
    for (customer, intent, num_subtasks, persona), bin_tasks in tasks_by_bins.items():
        num_sampled = min(max_count_per_bin, len(bin_tasks))
        sampled_tasks.extend(rng.sample(bin_tasks, num_sampled))
        print(
            f"Sampled {num_sampled} tasks for {customer}/{intent} with "
            f"{num_subtasks} subtasks and persona {persona}..."
        )

    print(f"Number of sampled tasks: {len(sampled_tasks)}")
    if save_tasks:
        with open(file_sampled, "w") as f:
            # The runtime always resolves every named split through tasks.json.
            # Keep all released definitions here and use split_tasks.json to
            # select the sampled base/train/test cohorts.
            json.dump([t.model_dump() for t in all_tasks], f, indent=2)
        split_file = file_sampled.parent / f"split_{file_sampled.stem}.json"
        with open(split_file, "w") as f:
            json.dump(
                build_task_splits(sampled_tasks, small_tasks, all_tasks), f, indent=2
            )

    return all_tasks


def main():
    parser = ArgumentParser()
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("-m", "--max-count-per-bin", type=int, default=3)
    args = parser.parse_args()
    create_tasks(max_count_per_bin=args.max_count_per_bin, seed=args.seed)


if __name__ == "__main__":
    main()
