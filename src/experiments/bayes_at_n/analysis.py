"""Binary Bayes@N posterior moments for one complete saved text experiment."""

import math

from pydantic import BaseModel

from tau2.data_model.simulation import Info, Results, SimulationRun, TerminationReason
from tau2.metrics.agent_metrics import is_successful
from tau2.orchestrator.modes import CommunicationMode


class TrialOutcome(BaseModel):
    """Identity and full-success outcome of one recorded trial."""

    simulation_id: str
    trial: int
    seed: int
    success: bool


class TaskOutcomes(BaseModel):
    """The complete trial cohort for one task."""

    task_id: str
    success_count: int
    outcomes: list[TrialOutcome]


class BayesAtNResult(BaseModel):
    """Posterior moments and the recorded experiment they summarize."""

    posterior_mean: float
    posterior_std: float
    num_tasks: int
    num_trials: int
    info: Info
    tasks: list[TaskOutcomes]


def _validate_trial(simulation: SimulationRun, num_trials: int) -> None:
    if type(simulation.trial) is not int or not 0 <= simulation.trial < num_trials:
        raise ValueError(
            f"Simulation {simulation.id!r} needs a trial index in 0..{num_trials - 1}."
        )
    if simulation.seed is None:
        raise ValueError(f"Simulation {simulation.id!r} lacks a recorded seed.")
    if (
        simulation.mode != CommunicationMode.HALF_DUPLEX.value
        or simulation.ticks is not None
        or simulation.speech_environment is not None
    ):
        raise ValueError("Bayes@N currently supports text half-duplex results only.")
    if simulation.complication is not None:
        raise ValueError(
            "Elicitation complications are not supported by this analysis."
        )
    if simulation.hallucination_retries_used != 0:
        raise ValueError("Hallucination-feedback retries are not independent trials.")
    if simulation.termination_reason == TerminationReason.INFRASTRUCTURE_ERROR:
        raise ValueError(
            "Infrastructure errors leave an incomplete cohort; rerun those trials."
        )
    if simulation.reward_info is None or not math.isfinite(
        simulation.reward_info.reward
    ):
        raise ValueError(f"Simulation {simulation.id!r} needs a finite graded reward.")


def compute_bayes_at_n(results: Results) -> BayesAtNResult:
    """Estimate the mean full-success probability over the declared task cohort.

    Each task must have exactly the recorded N trials, with indices 0 through
    N-1, distinct simulation IDs and distinct recorded seeds. The binary event
    uses tau2's ``is_successful`` reward rule. Each task has a Beta(1,1) prior;
    no prior observations are used. ``posterior_std`` is the posterior standard
    deviation of the equal-task mean, on the 0–1 scale.

    Only one complete text half-duplex Results experiment is supported. Raises
    ValueError for incomplete or ambiguous trial cohorts and unsupported modes.
    The input is not modified.
    """
    num_trials = results.info.num_trials
    if type(num_trials) is not int or num_trials < 1:
        raise ValueError("Info.num_trials must be a positive integer.")
    if results.info.audio_native_config is not None:
        raise ValueError("Bayes@N currently supports text half-duplex results only.")
    if results.info.elicitation is not None:
        raise ValueError("Elicitation experiments are not supported by this analysis.")

    task_ids = [task.id for task in results.tasks]
    if not task_ids or any(not task_id.strip() for task_id in task_ids):
        raise ValueError("Results.tasks must declare a nonempty task cohort.")
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("Results.tasks contains duplicate task IDs.")
    contexts = results.info.language_tasks
    if contexts and set(contexts) != set(task_ids):
        raise ValueError("Recorded language contexts differ from the task cohort.")

    grouped: dict[str, dict[int, SimulationRun]] = {task_id: {} for task_id in task_ids}
    simulation_ids: set[str] = set()
    for simulation in results.simulations:
        if simulation.task_id not in grouped:
            raise ValueError(f"Unknown simulation task ID {simulation.task_id!r}.")
        if not simulation.id.strip() or simulation.id in simulation_ids:
            raise ValueError("Simulation IDs must be nonempty and unique.")
        simulation_ids.add(simulation.id)
        _validate_trial(simulation, num_trials)
        trials = grouped[simulation.task_id]
        if simulation.trial in trials:
            raise ValueError(
                f"Duplicate task/trial pair: {simulation.task_id!r}/{simulation.trial}."
            )
        if contexts and simulation.language_context != contexts[simulation.task_id]:
            raise ValueError("Simulation language context differs from recorded Info.")
        trials[simulation.trial] = simulation

    task_outcomes = []
    means = []
    variances = []
    for task_id in sorted(grouped):
        trials = grouped[task_id]
        if len(trials) != num_trials:
            raise ValueError(
                f"Task {task_id!r}: expected {num_trials} trials "
                f"(0..{num_trials - 1}), found {len(trials)}."
            )
        ordered = [trials[trial] for trial in range(num_trials)]
        if len({simulation.seed for simulation in ordered}) != num_trials:
            raise ValueError(f"Task {task_id!r} reuses a seed across trials.")
        first = ordered[0]
        if any(
            simulation.policy != first.policy
            or simulation.language_context != first.language_context
            for simulation in ordered[1:]
        ):
            raise ValueError(f"Task {task_id!r} changes policy or language context.")
        outcomes = [
            TrialOutcome(
                simulation_id=simulation.id,
                trial=simulation.trial,
                seed=simulation.seed,
                success=is_successful(simulation.reward_info.reward),
            )
            for simulation in ordered
        ]
        successes = sum(outcome.success for outcome in outcomes)
        task_outcomes.append(
            TaskOutcomes(task_id=task_id, success_count=successes, outcomes=outcomes)
        )
        mean = (successes + 1) / (num_trials + 2)
        failure_probability = (num_trials - successes + 1) / (num_trials + 2)
        means.append(mean)
        variances.append(mean * failure_probability / (num_trials + 3))

    num_tasks = len(task_outcomes)
    return BayesAtNResult(
        posterior_mean=math.fsum(means) / num_tasks,
        posterior_std=math.sqrt(math.fsum(variances)) / num_tasks,
        num_tasks=num_tasks,
        num_trials=num_trials,
        info=results.info.model_copy(deep=True),
        tasks=task_outcomes,
    )
