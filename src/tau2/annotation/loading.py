# Copyright Sierra
"""THE results/task loader for annotation exports.

One streaming iterator (``iter_loaded_sims``) over any mix of results dirs /
``results.json`` files, replacing the three near-identical loaders the old
annotation code carried. Streams via ``Results.load_metadata`` +
``Results.iter_simulations`` so large audio-bearing runs never sit in memory
whole.
"""

import json
import operator
import re
from pathlib import Path
from typing import Annotated, Callable, Iterable, Iterator, Optional

from loguru import logger
from pydantic import BaseModel, Field

from tau2.data_model.simulation import Results, SimulationRun
from tau2.data_model.tasks import Task
from tau2.utils.utils import DATA_DIR


class LoadedSim(BaseModel):
    """One simulation plus everything an annotation export needs around it."""

    sim: Annotated[SimulationRun, Field(description="The simulation run.")]
    task: Annotated[
        Optional[Task],
        Field(
            description="The run's OWN task definition (localized for "
            "non-English runs); None when the results carry no tasks — fall "
            "back to the domain's base-language tasks.json via load_task."
        ),
    ] = None
    domain: Annotated[str, Field(description="Domain name of the run.")]
    results_dir: Annotated[
        Path, Field(description="Directory holding the run's results.json.")
    ]
    experiment_label: Annotated[
        str, Field(description="Human label for the run (its directory name).")
    ]


REWARD_FILTER_PATTERN = re.compile(r"^\s*([<>=!]+)\s*([0-9]*\.?[0-9]+)\s*$")

REWARD_OPS: dict[str, Callable[[float, float], bool]] = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}


def parse_reward_filter(
    filter_str: str,
) -> tuple[Callable[[float, float], bool], float]:
    """Parse a reward filter like ``"< 1"`` into (operator_fn, threshold)."""
    match = REWARD_FILTER_PATTERN.match(filter_str)
    if not match:
        raise ValueError(
            f"Invalid reward filter: '{filter_str}'. "
            "Expected format: '<op> <value>' (e.g., '< 1', '== 0', '>= 0.5')"
        )
    op_str, value_str = match.group(1), match.group(2)
    if op_str not in REWARD_OPS:
        raise ValueError(
            f"Unknown operator '{op_str}'. Supported: {', '.join(REWARD_OPS)}"
        )
    return REWARD_OPS[op_str], float(value_str)


def discover_results_files(path: Path) -> list[Path]:
    """All ``results.json`` files under ``path`` (or the file itself)."""
    path = Path(path)
    if path.is_file():
        return [path]
    if path.is_dir():
        results_files = sorted(path.rglob("results.json"))
        if not results_files:
            raise FileNotFoundError(f"No results.json files found in {path}")
        return results_files
    raise FileNotFoundError(f"Path does not exist: {path}")


def iter_loaded_sims(
    paths: Iterable[Path | str],
    *,
    domain_override: Optional[str] = None,
    reward_filter: Optional[str] = None,
    task_ids: Optional[list[str]] = None,
    trials: Optional[list[int]] = None,
) -> Iterator[LoadedSim]:
    """Stream filtered sims (with their tasks/domain) from results paths.

    Args:
        paths: Results dirs and/or results.json files; dirs are searched
            recursively for every run they contain.
        domain_override: Force the domain instead of reading it from run info.
        reward_filter: e.g. ``"< 1"`` — keep sims whose reward satisfies it
            (sims without a reward are dropped when a filter is set).
        task_ids: Keep only these task ids.
        trials: Keep only these trial indices.
    """
    reward_op = reward_threshold = None
    if reward_filter:
        reward_op, reward_threshold = parse_reward_filter(reward_filter)

    results_files = [f for p in paths for f in discover_results_files(Path(p))]
    for i, results_file in enumerate(results_files, 1):
        logger.info(f"loading results [{i}/{len(results_files)}]: {results_file}")
        meta = Results.load_metadata(results_file)
        sims = Results.iter_simulations(results_file)
        results_dir = results_file.parent
        domain = domain_override or meta.info.environment_info.domain_name
        experiment_label = results_dir.name
        tasks_by_id = {str(t.id): t for t in (meta.tasks or [])}

        for sim in sims:
            if task_ids and str(sim.task_id) not in task_ids:
                continue
            if trials and sim.trial not in trials:
                continue
            if reward_op is not None and reward_threshold is not None:
                reward = sim.reward_info.reward if sim.reward_info else None
                if reward is None or not reward_op(reward, reward_threshold):
                    continue
            yield LoadedSim(
                sim=sim,
                task=tasks_by_id.get(str(sim.task_id)),
                domain=domain,
                results_dir=results_dir,
                experiment_label=experiment_label,
            )


def load_task(domain: str, task_id: str) -> Optional[Task]:
    """One task from the domain's base-language tasks.json, as a typed Task."""
    tasks_file = DATA_DIR / "tau2" / "domains" / domain / "tasks.json"
    if not tasks_file.exists():
        return None
    with open(tasks_file) as fp:
        tasks = json.load(fp)
    for task in tasks:
        if str(task.get("id")) == str(task_id):
            return Task.model_validate(task)
    return None


def load_policy(domain: str) -> Optional[str]:
    """The domain's policy markdown, if present."""
    policy_file = DATA_DIR / "tau2" / "domains" / domain / "policy.md"
    if not policy_file.exists():
        return None
    return policy_file.read_text()
