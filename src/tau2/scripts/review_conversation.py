"""
Script to review conversation trajectories for errors.

This script loads a results.json file from a tau2 run() and reviews
the conversation for errors. It supports two review modes:
- "full": Review both agent and user simulator errors (default)
- "user": Review only user simulator errors

Note: This is different from evaluate_trajectories.py which computes rewards/metrics.
This script uses an LLM judge to identify conversation errors.

Usage:
    # Full review (agent + user errors)
    python -m tau2.scripts.review_conversation results.json

    # User simulator only review
    python -m tau2.scripts.review_conversation results.json --mode user

    # With custom output path
    python -m tau2.scripts.review_conversation results.json -o review.json
"""

import argparse
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from pathlib import Path
from typing import Optional, Union

from loguru import logger
from pydantic import BaseModel, Field
from rich.console import Console
from rich.markdown import Markdown
from rich.progress import Progress
from rich.table import Table

from tau2.data_model.simulation import (
    AuthenticationClassification,
    Results,
    Review,
    SimulationRun,
    UserOnlyReview,
)
from tau2.data_model.tasks import Task
from tau2.evaluator.auth_classifier import display_auth_classification
from tau2.evaluator.review_llm_judge_user_only import UserOnlyReviewer
from tau2.evaluator.reviewer import ReviewMode, review_simulation
from tau2.metrics.agent_metrics import compute_metrics
from tau2.utils.display import ConsoleDisplay, MarkdownDisplay

# =============================================================================
# Data Models for Review Output
# =============================================================================


class SimulationReviewFull(BaseModel):
    """
    Full review result for a single simulation (agent + user).
    """

    simulation_id: str = Field(description="The simulation ID.")
    task_id: str = Field(description="The task ID.")
    trial: int = Field(description="The trial number.")
    trajectory: str = Field(description="The conversation trajectory as a string.")
    review: Review = Field(description="The review result.")
    auth_classification: Optional[AuthenticationClassification] = Field(
        description="Authentication classification result.", default=None
    )


class SimulationReviewUser(BaseModel):
    """
    User simulator review result for a single simulation.
    """

    simulation_id: str = Field(description="The simulation ID.")
    task_id: str = Field(description="The task ID.")
    trial: int = Field(description="The trial number.")
    trajectory: str = Field(description="The user-visible trajectory as a string.")
    review: UserOnlyReview = Field(description="The review result.")


class FullReviewOutput(BaseModel):
    """
    Output of full conversation review (agent + user) for all simulations.
    """

    mode: str = Field(default="full", description="Review mode.")
    results_path: str = Field(description="Path to the original results.json file.")
    total_simulations: int = Field(description="Total number of simulations reviewed.")
    simulations_with_errors: int = Field(
        description="Number of simulations where errors were found."
    )
    simulations_with_agent_errors: int = Field(
        description="Number of simulations where the agent made errors."
    )
    simulations_with_user_errors: int = Field(
        description="Number of simulations where the user simulator made errors."
    )
    total_errors: int = Field(
        description="Total number of errors across all simulations."
    )
    total_agent_errors: int = Field(
        description="Total number of agent errors across all simulations."
    )
    total_user_errors: int = Field(
        description="Total number of user errors across all simulations."
    )
    # Authentication classification stats
    auth_succeeded: int = Field(
        description="Number of simulations where authentication succeeded.",
        default=0,
    )
    auth_failed: int = Field(
        description="Number of simulations where authentication failed.",
        default=0,
    )
    auth_not_needed: int = Field(
        description="Number of simulations where authentication was not needed.",
        default=0,
    )
    total_cost: float = Field(description="Total cost of all reviews.")
    reviews: list[SimulationReviewFull] = Field(
        description="List of review results for each simulation."
    )


class UserReviewOutput(BaseModel):
    """
    Output of user simulator review for all simulations.
    """

    mode: str = Field(default="user", description="Review mode.")
    results_path: str = Field(description="Path to the original results.json file.")
    total_simulations: int = Field(description="Total number of simulations reviewed.")
    simulations_with_errors: int = Field(
        description="Number of simulations where the user simulator made errors."
    )
    total_errors: int = Field(
        description="Total number of errors across all simulations."
    )
    total_cost: float = Field(description="Total cost of all reviews.")
    reviews: list[SimulationReviewUser] = Field(
        description="List of review results for each simulation."
    )


# =============================================================================
# Helper Functions
# =============================================================================


def is_full_duplex(simulation: SimulationRun) -> bool:
    """Check if the simulation used full-duplex mode (has ticks)."""
    return simulation.ticks is not None and len(simulation.ticks) > 0


def get_full_trajectory_string(simulation: SimulationRun) -> str:
    """Get the full conversation trajectory as a formatted string."""
    if is_full_duplex(simulation):
        return MarkdownDisplay.display_ticks_consolidated(
            simulation.ticks,
            effect_timeline=simulation.effect_timeline,
        )
    else:
        return MarkdownDisplay.display_messages(simulation.messages)


def get_user_visible_trajectory_string(simulation: SimulationRun) -> str:
    """Get the user-visible trajectory as a formatted string."""
    if is_full_duplex(simulation):
        return MarkdownDisplay.display_ticks_consolidated(
            simulation.ticks,
            user_visible_only=True,
            effect_timeline=simulation.effect_timeline,
        )
    else:
        messages = UserOnlyReviewer.make_user_visible_trajectory(simulation.messages)
        return MarkdownDisplay.display_messages(messages)


# =============================================================================
# Full Review (Agent + User)
# =============================================================================


def review_simulation_full(
    simulation: SimulationRun,
    task: Task,
    results: Results,
    interruption_enabled: bool = False,
) -> tuple[Review, AuthenticationClassification, str]:
    """
    Review a single simulation for both agent and user errors, plus auth classification.

    Returns:
        Tuple of (review result, auth classification, conversation trajectory string).
    """
    if not results.info or not results.info.user_info:
        raise ValueError("results.info.user_info is required for review")
    if not results.info.environment_info or not results.info.environment_info.policy:
        raise ValueError("results.info.environment_info.policy is required for review")

    user_info = results.info.user_info
    policy = results.info.environment_info.policy
    trajectory = get_full_trajectory_string(simulation)

    review, auth_classification = review_simulation(
        simulation=simulation,
        task=task,
        mode=ReviewMode.FULL,
        user_info=user_info,
        policy=policy,
        interruption_enabled=interruption_enabled,
    )

    # review_simulation returns Review for FULL mode, auth_classification is not None
    assert isinstance(review, Review)
    assert auth_classification is not None

    return review, auth_classification, trajectory


def run_full_review(
    results: Results,
    results_path: str,
    interruption_enabled: bool = False,
    show_details: bool = False,
    max_concurrency: int = 32,
    console: Optional[Console] = None,
) -> FullReviewOutput:
    """Run full review (agent + user) for all simulations."""
    if console is None:
        console = Console()

    tasks = {task.id: task for task in results.tasks}

    reviews: list[SimulationReviewFull] = []
    total_errors = 0
    total_agent_errors = 0
    total_user_errors = 0
    simulations_with_errors = 0
    simulations_with_agent_errors = 0
    simulations_with_user_errors = 0
    auth_succeeded = 0
    auth_failed = 0
    auth_not_needed = 0
    total_cost = 0.0

    console.print(
        f"\n🔍 Reviewing {len(results.simulations)} simulation(s) [mode: full, concurrency: {max_concurrency}]...",
        style="bold",
    )

    def process_simulation(
        simulation: SimulationRun,
    ) -> Optional[SimulationReviewFull]:
        """Process a single simulation."""
        task = tasks.get(simulation.task_id)
        if task is None:
            logger.warning(
                f"Task {simulation.task_id} not found, skipping simulation {simulation.id}"
            )
            return None

        try:
            review, auth_classification, trajectory = review_simulation_full(
                simulation=simulation,
                task=task,
                results=results,
                interruption_enabled=interruption_enabled,
            )

            # Determine trial number
            trial = getattr(simulation, "trial", 0)
            if trial == 0:
                same_task_sims = [
                    s for s in results.simulations if s.task_id == simulation.task_id
                ]
                trial = same_task_sims.index(simulation) + 1

            return SimulationReviewFull(
                simulation_id=simulation.id,
                task_id=simulation.task_id,
                trial=trial,
                trajectory=trajectory,
                review=review,
                auth_classification=auth_classification,
            )

        except Exception as e:
            logger.error(
                f"Error reviewing simulation {simulation.id}: {e}\n{traceback.format_exc()}"
            )
            return SimulationReviewFull(
                simulation_id=simulation.id,
                task_id=simulation.task_id,
                trial=0,
                trajectory="Error: Could not extract trajectory",
                review=Review(
                    summary="",
                    agent_error=False,
                    user_error=False,
                    has_errors=False,
                    errors=[],
                    cost=None,
                ),
                auth_classification=None,
            )

    with Progress(console=console) as progress:
        task_progress = progress.add_task(
            "Reviewing...", total=len(results.simulations)
        )

        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            # Submit all tasks - each with its own context copy so ContextVars propagate
            future_to_sim = {
                executor.submit(copy_context().run, process_simulation, sim): sim
                for sim in results.simulations
            }

            # Process completed tasks
            for future in as_completed(future_to_sim):
                progress.update(task_progress, advance=1)

                try:
                    sim_review = future.result()
                except Exception as e:
                    original_sim = future_to_sim[future]
                    logger.error(
                        f"Error in review thread for simulation {original_sim.id}: {e}\n{traceback.format_exc()}"
                    )
                    continue

                if sim_review is None:
                    continue

                reviews.append(sim_review)
                review = sim_review.review
                auth_class = sim_review.auth_classification

                # Set the review and auth_classification on the original simulation object
                original_sim = future_to_sim[future]
                original_sim.review = review
                original_sim.auth_classification = auth_class

                # Update stats
                if review.has_errors:
                    simulations_with_errors += 1
                if review.agent_error:
                    simulations_with_agent_errors += 1
                if review.user_error:
                    simulations_with_user_errors += 1

                for error in review.errors:
                    if error.source == "agent":
                        total_agent_errors += 1
                        total_errors += 1
                    elif error.source == "user":
                        total_user_errors += 1
                        total_errors += 1

                if review.cost:
                    total_cost += review.cost

                # Update auth classification stats
                if auth_class:
                    if auth_class.status == "succeeded":
                        auth_succeeded += 1
                    elif auth_class.status == "failed":
                        auth_failed += 1
                    else:
                        auth_not_needed += 1
                    if auth_class.cost:
                        total_cost += auth_class.cost

                if show_details:
                    console.print(
                        f"\n[bold]Simulation: {sim_review.simulation_id} (Task: {sim_review.task_id})[/bold]"
                    )
                    ConsoleDisplay.display_review(
                        review,
                        title=f"Simulation {sim_review.simulation_id}",
                        console=console,
                    )

    return FullReviewOutput(
        mode="full",
        results_path=str(results_path),
        total_simulations=len(reviews),
        simulations_with_errors=simulations_with_errors,
        simulations_with_agent_errors=simulations_with_agent_errors,
        simulations_with_user_errors=simulations_with_user_errors,
        total_errors=total_errors,
        total_agent_errors=total_agent_errors,
        total_user_errors=total_user_errors,
        auth_succeeded=auth_succeeded,
        auth_failed=auth_failed,
        auth_not_needed=auth_not_needed,
        total_cost=total_cost,
        reviews=reviews,
    )


# =============================================================================
# User Simulator Review
# =============================================================================


def review_simulation_user(
    simulation: SimulationRun,
    task: Task,
    results: Results,
    interruption_enabled: bool = False,
) -> tuple[UserOnlyReview, str]:
    """
    Review a single simulation for user simulator errors only.

    Returns:
        Tuple of (review result, user-visible trajectory string).
    """
    if not results.info or not results.info.user_info:
        raise ValueError("results.info.user_info is required for review")

    user_info = results.info.user_info
    trajectory = get_user_visible_trajectory_string(simulation)

    review, _ = review_simulation(
        simulation=simulation,
        task=task,
        mode=ReviewMode.USER,
        user_info=user_info,
        interruption_enabled=interruption_enabled,
    )

    # review_simulation returns UserOnlyReview for USER mode
    assert isinstance(review, UserOnlyReview)

    return review, trajectory


def run_user_review(
    results: Results,
    results_path: str,
    interruption_enabled: bool = False,
    show_details: bool = False,
    max_concurrency: int = 32,
    console: Optional[Console] = None,
) -> UserReviewOutput:
    """Run user simulator review for all simulations."""
    if console is None:
        console = Console()

    tasks = {task.id: task for task in results.tasks}

    reviews: list[SimulationReviewUser] = []
    total_errors = 0
    simulations_with_errors = 0
    total_cost = 0.0

    console.print(
        f"\n🔍 Reviewing {len(results.simulations)} simulation(s) [mode: user, concurrency: {max_concurrency}]...",
        style="bold",
    )

    def process_simulation(
        simulation: SimulationRun,
    ) -> Optional[SimulationReviewUser]:
        """Process a single simulation."""
        task = tasks.get(simulation.task_id)
        if task is None:
            logger.warning(
                f"Task {simulation.task_id} not found, skipping simulation {simulation.id}"
            )
            return None

        try:
            review, trajectory = review_simulation_user(
                simulation=simulation,
                task=task,
                results=results,
                interruption_enabled=interruption_enabled,
            )

            # Determine trial number
            trial = getattr(simulation, "trial", 0)
            if trial == 0:
                same_task_sims = [
                    s for s in results.simulations if s.task_id == simulation.task_id
                ]
                trial = same_task_sims.index(simulation) + 1

            return SimulationReviewUser(
                simulation_id=simulation.id,
                task_id=simulation.task_id,
                trial=trial,
                trajectory=trajectory,
                review=review,
            )

        except Exception as e:
            logger.error(
                f"Error reviewing simulation {simulation.id}: {e}\n{traceback.format_exc()}"
            )
            return SimulationReviewUser(
                simulation_id=simulation.id,
                task_id=simulation.task_id,
                trial=0,
                trajectory="Error: Could not extract trajectory",
                review=UserOnlyReview(
                    summary="",
                    user_error=False,
                    has_errors=False,
                    errors=[],
                    cost=None,
                ),
            )

    with Progress(console=console) as progress:
        task_progress = progress.add_task(
            "Reviewing...", total=len(results.simulations)
        )

        with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
            # Submit all tasks - each with its own context copy so ContextVars propagate
            future_to_sim = {
                executor.submit(copy_context().run, process_simulation, sim): sim
                for sim in results.simulations
            }

            # Process completed tasks
            for future in as_completed(future_to_sim):
                progress.update(task_progress, advance=1)

                try:
                    sim_review = future.result()
                except Exception as e:
                    original_sim = future_to_sim[future]
                    logger.error(
                        f"Error in review thread for simulation {original_sim.id}: {e}\n{traceback.format_exc()}"
                    )
                    continue

                if sim_review is None:
                    continue

                reviews.append(sim_review)
                review = sim_review.review

                # Set the user_only_review on the original simulation object
                original_sim = future_to_sim[future]
                original_sim.user_only_review = review

                # Update stats
                if review.has_errors:
                    simulations_with_errors += 1
                total_errors += len(review.errors)

                if review.cost:
                    total_cost += review.cost

                if show_details:
                    console.print(
                        f"\n[bold]Simulation: {sim_review.simulation_id} (Task: {sim_review.task_id})[/bold]"
                    )
                    ConsoleDisplay.display_user_only_review(
                        review,
                        title=f"Simulation {sim_review.simulation_id}",
                        console=console,
                    )

    return UserReviewOutput(
        mode="user",
        results_path=str(results_path),
        total_simulations=len(reviews),
        simulations_with_errors=simulations_with_errors,
        total_errors=total_errors,
        total_cost=total_cost,
        reviews=reviews,
    )


# =============================================================================
# Main Review Function
# =============================================================================


def review(
    results_path: str,
    mode: ReviewMode = ReviewMode.FULL,
    output_path: Optional[str] = None,
    interruption_enabled: bool = False,
    show_details: bool = False,
    max_concurrency: int = 32,
    limit: Optional[int] = None,
    task_ids: Optional[list[str]] = None,
    log_llm: bool = False,
) -> Optional[Union[FullReviewOutput, UserReviewOutput]]:
    """
    Review conversation trajectories for all simulations in a results file.

    Args:
        results_path: Path to the results.json file.
        mode: Review mode ("full" for agent+user, "user" for user only).
        output_path: Optional path to save the review output.
        interruption_enabled: Whether interruption was enabled for these simulations.
        show_details: Whether to show detailed results for each simulation.
        max_concurrency: Maximum number of concurrent reviews.
        limit: Optional limit on number of simulations to review.
        task_ids: Optional list of task IDs to filter simulations.
        log_llm: Whether to log LLM request/response.

    Returns:
        FullReviewOutput or UserReviewOutput depending on mode, or None if aborted.
    """
    console = Console()

    # Determine output path
    input_path = Path(results_path)
    if output_path:
        reviewed_file = Path(output_path)
    else:
        reviewed_file = input_path.parent / "results_reviewed.json"

    # Check if reviews already exist
    if reviewed_file.exists():
        console.print(
            f"\n⚠️  Reviewed results already exist: {reviewed_file}",
            style="bold yellow",
        )
        from rich.prompt import Confirm

        if not Confirm.ask("Do you want to overwrite?", default=False):
            console.print("Aborted.", style="dim")
            return None

    # Load results
    console.print(f"\n📂 Loading results from: {results_path}", style="bold blue")
    results = Results.load(Path(results_path))

    # Filter simulations if task_ids specified
    if task_ids:
        original_count = len(results.simulations)
        results.simulations = [
            sim for sim in results.simulations if sim.task_id in task_ids
        ]
        console.print(
            f"  Filtered to {len(results.simulations)}/{original_count} simulations for task_ids: {task_ids}"
        )

    # Limit simulations if specified
    if limit and limit < len(results.simulations):
        results.simulations = results.simulations[:limit]
        console.print(f"  Limited to first {limit} simulations")

    # Enable LLM logging if requested
    if log_llm:
        from tau2.utils.llm_utils import set_llm_log_dir

        log_path = input_path.parent / "llm_logs"
        log_path.mkdir(parents=True, exist_ok=True)
        set_llm_log_dir(log_path)
        console.print(f"  [yellow]LLM logs will be saved to: {log_path}[/]")

    # Run review based on mode
    if mode == ReviewMode.FULL:
        output = run_full_review(
            results=results,
            results_path=results_path,
            interruption_enabled=interruption_enabled,
            show_details=show_details,
            max_concurrency=max_concurrency,
            console=console,
        )
    else:
        output = run_user_review(
            results=results,
            results_path=results_path,
            interruption_enabled=interruption_enabled,
            show_details=show_details,
            max_concurrency=max_concurrency,
            console=console,
        )
        _ = "_user_review.json"

    # Save the modified Results object with reviews embedded in simulations
    results.save(reviewed_file)

    console.print(
        f"\n💾 Saved reviewed results to: {reviewed_file}", style="bold green"
    )
    console.print(f"  Review cost: ${output.total_cost:.4f}\n")

    # Display comprehensive agent metrics
    metrics = compute_metrics(results)
    ConsoleDisplay.display_agent_metrics(metrics)

    return output


# =============================================================================
# Display Functions for Existing Reviews
# =============================================================================


def display_review_interactive(
    review_path: str,
    console: Optional[Console] = None,
) -> None:
    """
    Interactively display reviews from a saved review JSON file.

    Args:
        review_path: Path to the review JSON file.
        console: Optional Console instance.
    """
    import json

    if console is None:
        console = Console()

    # Load review file
    console.print(f"\n📂 Loading review from: {review_path}", style="bold blue")
    with open(review_path) as f:
        data = json.load(f)

    mode = data.get("mode", "full")
    reviews = data.get("reviews", [])

    if not reviews:
        console.print("No reviews found in file.", style="yellow")
        return

    # Filter state: "all", "agent", "user", "errors"
    current_filter = "all"

    def get_filtered_reviews():
        """Get reviews based on current filter."""
        if current_filter == "all":
            return reviews
        elif current_filter == "errors":
            return [r for r in reviews if r.get("review", {}).get("has_errors", False)]
        elif current_filter == "agent":
            return [r for r in reviews if r.get("review", {}).get("agent_error", False)]
        elif current_filter == "user":
            return [r for r in reviews if r.get("review", {}).get("user_error", False)]
        return reviews

    def show_simulation_list():
        """Display the list of available simulations."""
        filtered = get_filtered_reviews()

        console.print("\n" + "=" * 80, style="dim")
        console.print(f"📊 Review Summary ({mode} mode)", style="bold")
        console.print(
            f"  Total simulations: {data.get('total_simulations', len(reviews))}"
        )
        console.print(
            f"  Simulations with errors: {data.get('simulations_with_errors', 0)}",
            style="yellow" if data.get("simulations_with_errors", 0) > 0 else "green",
        )

        # Show current filter
        filter_labels = {
            "all": "All",
            "errors": "With Errors",
            "agent": "[red]🤖 Agent Errors[/red]",
            "user": "[blue]👤 User Errors[/blue]",
        }
        console.print(
            f"  Current filter: [bold yellow]{filter_labels.get(current_filter, current_filter)}[/bold yellow] ({len(filtered)} shown)"
        )
        console.print("=" * 80, style="dim")

        if not filtered:
            console.print("\n[yellow]No simulations match the current filter.[/yellow]")
            console.print(
                "\n[dim]Commands: [a] all, [e] errors, [/dim][red][r] 🤖 agent[/red][dim], [/dim][blue][u] 👤 user[/blue][dim], [q] quit[/dim]"
            )
            return

        # Build simulation table
        table = Table(
            title="Simulations",
            show_header=True,
            header_style="bold cyan",
            show_lines=True,  # Add separator lines between rows
        )
        table.add_column("#", style="dim", width=4, justify="right")
        table.add_column("", width=2)  # Status column (emoji)
        table.add_column("Task", style="cyan", no_wrap=True)
        table.add_column("T", width=2, justify="center")  # Trial
        table.add_column("Errors", width=20, no_wrap=True)
        table.add_column("Summary")

        for i, rev_data in enumerate(filtered):
            task_id = rev_data.get("task_id", "unknown")
            trial = rev_data.get("trial", 0)
            rev = rev_data.get("review", {})
            has_errors = rev.get("has_errors", False)
            summary = rev.get("summary", "")

            status = "❌" if has_errors else "✅"

            if has_errors:
                num_errors = len(rev.get("errors", []))
                if mode == "full":
                    error_parts = []
                    if rev.get("agent_error", False):
                        error_parts.append("[red]🤖Agent[/red]")
                    if rev.get("user_error", False):
                        error_parts.append("[blue]👤User[/blue]")
                    error_info = f"{' '.join(error_parts)} ({num_errors})"
                else:
                    error_info = str(num_errors)
            else:
                error_info = "-"

            # Highlight "agent" and "user simulator" in summary with colors
            summary_display = (
                summary.replace("user simulator", "[blue]user simulator[/blue]")
                .replace("User simulator", "[blue]User simulator[/blue]")
                .replace("User Simulator", "[blue]User Simulator[/blue]")
                .replace("agent", "[red]agent[/red]")
                .replace("Agent", "[red]Agent[/red]")
            )

            table.add_row(
                str(i + 1),
                status,
                task_id,
                str(trial),
                error_info,
                summary_display,
            )

        console.print(table)
        console.print(
            "\n[dim]Commands: [number] view, [a] all, [e] errors, [/dim][red][r] 🤖 agent[/red][dim], [/dim][blue][u] 👤 user[/blue][dim], [q] quit[/dim]"
        )

    # Show initial list
    show_simulation_list()

    # Interactive selection loop
    while True:
        console.print("\n[bold cyan]Select:[/bold cyan] ", end="")
        try:
            user_input = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            console.print("\n")
            break

        if user_input == "q" or user_input == "quit":
            break

        # Filter commands
        if user_input == "a" or user_input == "all":
            current_filter = "all"
            show_simulation_list()
            continue

        if user_input == "e" or user_input == "errors":
            current_filter = "errors"
            show_simulation_list()
            continue

        if user_input == "r" or user_input == "agent" or user_input == "robot":
            current_filter = "agent"
            show_simulation_list()
            continue

        if user_input == "u" or user_input == "user":
            current_filter = "user"
            show_simulation_list()
            continue

        if user_input == "" or user_input == "l" or user_input == "list":
            # Refresh the list
            show_simulation_list()
            continue

        # View a specific simulation by number
        try:
            idx = int(user_input) - 1
            filtered = get_filtered_reviews()
            if 0 <= idx < len(filtered):
                _display_single_review(filtered[idx], mode, console)
                console.print("\n[dim]Press Enter to go back to list...[/dim] ", end="")
                try:
                    input()
                except (EOFError, KeyboardInterrupt):
                    break
                show_simulation_list()
            else:
                console.print(f"Invalid number. Enter 1-{len(filtered)}.", style="red")
        except ValueError:
            console.print("Invalid input. Enter a number or command.", style="red")


def _display_single_review(
    review_data: dict,
    mode: str,
    console: Console,
) -> None:
    """Display a single review with trajectory and evaluation result."""
    sim_id = review_data.get("simulation_id", "unknown")
    task_id = review_data.get("task_id", "unknown")
    trial = review_data.get("trial", 0)
    trajectory = review_data.get("trajectory", "")
    review = review_data.get("review", {})
    auth_data = review_data.get("auth_classification")

    console.print("\n" + "=" * 80, style="dim")
    console.print(
        f"[bold]Simulation: {sim_id}[/bold] (Task: {task_id}, Trial: {trial})"
    )
    console.print("=" * 80, style="dim")

    # Show trajectory - use Rich's Markdown renderer for tables
    console.print("\n[bold cyan]Trajectory:[/bold cyan]")
    console.print(Markdown(trajectory))

    # Show review result
    console.print("\n[bold cyan]Review:[/bold cyan]")
    if mode == "full":
        result = Review(**review)
        ConsoleDisplay.display_review(result, title="Review Result", console=console)

        # Show auth classification if available
        if auth_data:
            console.print("\n[bold cyan]Authentication Classification:[/bold cyan]")
            auth_result = AuthenticationClassification(**auth_data)
            display_auth_classification(auth_result, console=console)
    else:
        result = UserOnlyReview(**review)
        ConsoleDisplay.display_user_only_review(
            result, title="Review Result", console=console
        )


def display_review_batch(
    review_path: str,
    only_errors: bool = False,
    simulation_id: Optional[str] = None,
    console: Optional[Console] = None,
) -> None:
    """
    Display reviews from a saved review JSON file (non-interactive).

    Args:
        review_path: Path to the review JSON file.
        only_errors: Only show simulations with errors.
        simulation_id: Only show this specific simulation (partial match).
        console: Optional Console instance.
    """
    import json

    if console is None:
        console = Console()

    # Load review file
    with open(review_path) as f:
        data = json.load(f)

    mode = data.get("mode", "full")
    reviews = data.get("reviews", [])

    for review_data in reviews:
        sim_id = review_data.get("simulation_id", "")
        review = review_data.get("review", {})
        has_errors = review.get("has_errors", False)

        # Filter
        if only_errors and not has_errors:
            continue
        if simulation_id and simulation_id not in sim_id:
            continue

        _display_single_review(review_data, mode, console)


# =============================================================================
# CLI
# =============================================================================


def make_parser() -> argparse.ArgumentParser:
    """Create argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        description="Review conversation trajectories for agent and/or user simulator errors using an LLM judge.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Run subcommand (default behavior)
    run_parser = subparsers.add_parser(
        "run",
        help="Run LLM review on a results.json file or directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full review (agent + user errors) on a single file
  python -m tau2.scripts.review_conversation run results.json

  # Review all results.json files in a directory
  python -m tau2.scripts.review_conversation run data/simulations/

  # User simulator only review
  python -m tau2.scripts.review_conversation run results.json --mode user

  # With detailed output for each simulation
  python -m tau2.scripts.review_conversation run results.json --show-details

  # For full-duplex simulations with interruption
  python -m tau2.scripts.review_conversation run results.json --interruption-enabled
        """,
    )
    run_parser.add_argument(
        "results_path",
        type=str,
        help="Path to a results.json file or a directory containing results.json files.",
    )
    run_parser.add_argument(
        "-m",
        "--mode",
        type=str,
        choices=["full", "user"],
        default="full",
        help="Review mode: 'full' (agent+user, default) or 'user' (user simulator only).",
    )
    run_parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output path for the review JSON. Defaults to <input>_<mode>_review.json.",
    )
    run_parser.add_argument(
        "--interruption-enabled",
        action="store_true",
        help="Flag indicating that interruption was enabled for these simulations.",
    )
    run_parser.add_argument(
        "--show-details",
        action="store_true",
        help="Show detailed review results for each simulation.",
    )
    run_parser.add_argument(
        "-c",
        "--max-concurrency",
        type=int,
        default=10,
        help="Maximum number of concurrent reviews (default: 10).",
    )
    run_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    # Display subcommand
    display_parser = subparsers.add_parser(
        "display",
        help="Display reviews from a saved review JSON file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode - pick simulations to view
  python -m tau2.scripts.review_conversation display review.json

  # Show only simulations with errors
  python -m tau2.scripts.review_conversation display review.json --errors-only

  # Show a specific simulation
  python -m tau2.scripts.review_conversation display review.json --simulation abc123

  # Non-interactive: show all
  python -m tau2.scripts.review_conversation display review.json --all
        """,
    )
    display_parser.add_argument(
        "review_path",
        type=str,
        help="Path to the review JSON file (e.g., *_conversation_review.json).",
    )
    display_parser.add_argument(
        "--errors-only",
        action="store_true",
        help="Only show simulations with errors.",
    )
    display_parser.add_argument(
        "--simulation",
        "-s",
        type=str,
        default=None,
        help="Show only this simulation ID (partial match).",
    )
    display_parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="Show all reviews (non-interactive mode).",
    )

    return parser


def find_results_files(path: Path) -> list[Path]:
    """
    Find all results.json files in a path.

    If path is a file, returns [path].
    If path is a directory, recursively finds all results.json files.
    """
    if path.is_file():
        return [path]

    results_files = []
    # Look for results.json in immediate subdirectories (typical tau2 structure)
    for subdir in sorted(path.iterdir()):
        if subdir.is_dir():
            results_file = subdir / "results.json"
            if results_file.exists():
                results_files.append(results_file)

    # If no results found in subdirs, check if results.json is directly in the path
    if not results_files:
        direct_results = path / "results.json"
        if direct_results.exists():
            results_files.append(direct_results)

    return results_files


def main():
    """Main entry point."""
    parser = make_parser()
    args = parser.parse_args()

    # Default to 'run' if no command specified but a file is given
    if args.command is None:
        # Check if first positional arg looks like a results file
        if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
            # Legacy mode: treat as 'run' command
            sys.argv.insert(1, "run")
            args = parser.parse_args()
        else:
            parser.print_help()
            sys.exit(1)

    if args.command == "run":
        # Configure logging
        if args.verbose:
            logger.configure(handlers=[{"sink": sys.stderr, "level": "DEBUG"}])
        else:
            logger.configure(handlers=[{"sink": sys.stderr, "level": "WARNING"}])

        # Find all results files
        input_path = Path(args.results_path)
        results_files = find_results_files(input_path)

        if not results_files:
            console = Console()
            console.print(
                f"[red]No results.json files found in: {args.results_path}[/red]"
            )
            sys.exit(1)

        # Run review for each results file
        mode = ReviewMode.FULL if args.mode == "full" else ReviewMode.USER
        console = Console()

        if len(results_files) > 1:
            console.print(
                f"\n📁 Found {len(results_files)} results files to review:",
                style="bold blue",
            )
            for i, rf in enumerate(results_files, 1):
                console.print(f"  {i}. {rf.parent.name}/results.json")
            console.print()

        for i, results_file in enumerate(results_files):
            if len(results_files) > 1:
                console.print(
                    f"\n{'=' * 60}\n[bold cyan]Processing ({i + 1}/{len(results_files)}): {results_file.parent.name}[/bold cyan]\n{'=' * 60}"
                )

            review(
                results_path=str(results_file),
                mode=mode,
                output_path=args.output if len(results_files) == 1 else None,
                interruption_enabled=args.interruption_enabled,
                show_details=args.show_details,
                max_concurrency=args.max_concurrency,
            )

    elif args.command == "display":
        console = Console()
        if args.all or args.errors_only or args.simulation:
            # Non-interactive batch display
            display_review_batch(
                review_path=args.review_path,
                only_errors=args.errors_only,
                simulation_id=args.simulation,
                console=console,
            )
        else:
            # Interactive mode
            display_review_interactive(
                review_path=args.review_path,
                console=console,
            )


if __name__ == "__main__":
    main()
