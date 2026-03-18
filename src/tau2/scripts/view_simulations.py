#!/usr/bin/env python3
import json
import subprocess
import uuid
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table
from rich.text import Text

from tau2.data_model.simulation import (
    ErrorSource,
    ErrorType,
    Results,
    SimulationNote,
    SimulationRun,
    TerminationReason,
)
from tau2.data_model.tasks import Task, TaskIssue, TaskIssueStatus
from tau2.metrics.agent_metrics import compute_metrics, is_successful
from tau2.utils.display import ConsoleDisplay
from tau2.utils.utils import DATA_DIR


def get_tick_duration_ms(results: Results) -> Optional[int]:
    """Extract tick_duration_ms from Results.info.audio_native_config."""
    if results.info.audio_native_config is not None:
        return int(results.info.audio_native_config.tick_duration_ms)
    return None


def get_available_simulations(sim_dir: Optional[Path] = None):
    """Get list of available simulation result files.

    Prefers results_reviewed.json (with embedded reviews) over results.json.
    """
    if sim_dir is None:
        sim_dir = Path(DATA_DIR) / "simulations"

    if not sim_dir.exists():
        return []

    # Look for results files in subdirectories
    # Prefer results_reviewed.json (with reviews) over results.json
    sim_files = []
    for subdir in sim_dir.iterdir():
        if subdir.is_dir():
            reviewed_file = subdir / "results_reviewed.json"
            results_file = subdir / "results.json"
            if reviewed_file.exists():
                sim_files.append(reviewed_file)
            elif results_file.exists():
                sim_files.append(results_file)

    return sorted(sim_files)


def display_simulation_list(
    results: Results, only_show_failed: bool = False, only_show_all_failed: bool = False
):
    """Display a numbered list of simulations with basic info using a table."""
    # Calculate number of successful and total trials for each task
    num_success = defaultdict(int)
    for sim in results.simulations:
        reward = sim.reward_info.reward if sim.reward_info else None
        if reward is not None and is_successful(reward):
            num_success[sim.task_id] += 1

    # Filter simulations
    filtered_sims = []
    for i, sim in enumerate(results.simulations, 1):
        reward = sim.reward_info.reward if sim.reward_info else None
        if only_show_failed:
            if reward is None or is_successful(reward):
                continue
        if only_show_all_failed:
            if num_success[sim.task_id] > 0:
                continue
        filtered_sims.append((i, sim))

    if not filtered_sims:
        ConsoleDisplay.console.print("\n[yellow]No simulations match the filter.[/]")
        return

    # Create table with fixed column widths
    table = Table(
        title="Simulations",
        show_header=True,
        header_style="bold cyan",
        box=None,
        pad_edge=False,
        collapse_padding=True,
    )

    # Add columns with fixed widths for alignment
    table.add_column("#", width=4, justify="right", style="dim")
    table.add_column("Task", width=12, no_wrap=True)
    table.add_column("Trial", width=5, justify="center")
    table.add_column("Reward", width=6, justify="center")
    table.add_column("DB", width=2, justify="center")
    table.add_column("Read Acts", width=9, justify="center")
    table.add_column("Write Acts", width=10, justify="center")
    table.add_column("Auth", width=4, justify="center")
    table.add_column("Stop", width=4, justify="center")
    table.add_column("🔇", width=2, justify="center")  # Unresponsive period
    table.add_column("🤖 Err", width=6, justify="center")  # Agent errors
    table.add_column("👤 Err", width=6, justify="center")  # User errors
    table.add_column("🤖 Tags", no_wrap=False)  # Agent error tags
    table.add_column("👤 Tags", no_wrap=False)  # User error tags

    termination_color_map = {
        TerminationReason.USER_STOP: "green",
        TerminationReason.AGENT_STOP: "green",
        TerminationReason.MAX_STEPS: "yellow",
        TerminationReason.TOO_MANY_ERRORS: "red",
        TerminationReason.AGENT_ERROR: "red",
        TerminationReason.USER_ERROR: "red",
    }

    termination_icon_map = {
        TerminationReason.USER_STOP: "👤",
        TerminationReason.AGENT_STOP: "🤖",
        TerminationReason.MAX_STEPS: "⏱",
        TerminationReason.TOO_MANY_ERRORS: "💥",
        TerminationReason.AGENT_ERROR: "💥",
        TerminationReason.USER_ERROR: "💥",
    }

    for i, sim in filtered_sims:
        reward = sim.reward_info.reward if sim.reward_info else None

        # Pass/Fail
        if reward is not None:
            pass_str = "[green]✓[/]" if is_successful(reward) else "[red]✗[/]"
        else:
            pass_str = "[dim]-[/]"

        # DB Match
        if sim.reward_info and sim.reward_info.db_check:
            db_str = "[green]✓[/]" if sim.reward_info.db_check.db_match else "[red]✗[/]"
        else:
            db_str = "[dim]-[/]"

        # Read/Write actions
        read_str = "[dim]-[/]"
        write_str = "[dim]-[/]"
        if sim.reward_info and sim.reward_info.action_checks:
            partial = sim.reward_info.partial_action_reward
            if partial:
                # Show read/write breakdown if tool_type is populated
                if partial.get("read"):
                    r = partial["read"]
                    color = (
                        "green"
                        if r["correct"] == r["count"]
                        else ("yellow" if r["correct"] > 0 else "red")
                    )
                    read_str = f"[{color}]{r['correct']}/{r['count']}[/]"
                if partial.get("write"):
                    w = partial["write"]
                    color = (
                        "green"
                        if w["correct"] == w["count"]
                        else ("yellow" if w["correct"] > 0 else "red")
                    )
                    write_str = f"[{color}]{w['correct']}/{w['count']}[/]"
                # If no read/write breakdown, show total in read column as fallback
                if (
                    not partial.get("read")
                    and not partial.get("write")
                    and partial.get("total")
                ):
                    t = partial["total"]
                    color = (
                        "green"
                        if t["correct"] == t["count"]
                        else ("yellow" if t["correct"] > 0 else "red")
                    )
                    read_str = f"[{color}]{t['correct']}/{t['count']}[/]"
                    write_str = "[dim]n/a[/]"

        # Authentication
        auth_str = "[dim]-[/]"
        if sim.auth_classification:
            if sim.auth_classification.status == "succeeded":
                auth_str = "[green]✓[/]"
            elif sim.auth_classification.status == "failed":
                auth_str = "[red]✗[/]"
            else:  # not_needed
                auth_str = "[dim]n/a[/]"

        # Termination reason
        term_color = termination_color_map.get(sim.termination_reason, "white")
        term_icon = termination_icon_map.get(sim.termination_reason, "?")
        stop_str = f"[{term_color}]{term_icon}[/]"

        # Unresponsive period (from streaming/full-duplex mode)
        if sim.info and "had_unresponsive_period" in sim.info:
            unresponsive_str = (
                "[red]✗[/]" if sim.info["had_unresponsive_period"] else "[green]✓[/]"
            )
        else:
            unresponsive_str = "[dim]-[/]"

        # Tag abbreviations
        tag_abbrev = {
            "hallucination": "halluc",
            "incorrect_interpretation": "mis_interp",
            "guideline_violation": "guideline",
            "revealed_info_early": "early_reveal",
            "inconsistent_behavior": "inconsist",
            "tool_call_schema_error": "tc_schema",
            "tool_call_argument_error": "tc_arg",
            "irrelevant_tool_call": "tc_irrel",
            "premature_termination": "early_term",
            "missed_required_action": "missed_act",
            "wrong_sequence": "wrong_seq",
            "interruption_error": "interrupt",
            "other": "other",
        }

        def format_tags_with_severity(errors, is_user: bool = False) -> str:
            """Format error tags with color based on severity."""
            if not errors:
                return "[dim]-[/]"

            # Collect tags with their max severity
            tag_severity: dict[str, str] = {}  # tag -> worst severity
            for e in errors:
                severity = getattr(e, "severity", None)
                for tag in e.error_tags:
                    existing = tag_severity.get(tag)
                    if is_user:
                        # For user errors: critical_helped/critical_hindered > minor > None
                        if severity in ("critical_helped", "critical_hindered"):
                            tag_severity[tag] = "critical"
                        elif existing != "critical":
                            tag_severity[tag] = severity or "minor"
                    else:
                        # For agent errors: critical > minor > None
                        if severity == "critical":
                            tag_severity[tag] = "critical"
                        elif existing != "critical":
                            tag_severity[tag] = severity or "minor"

            # Format with colors
            parts = []
            for tag in sorted(tag_severity.keys()):
                abbrev = tag_abbrev.get(tag, tag[:8])
                sev = tag_severity[tag]
                if sev == "critical":
                    parts.append(f"[red]{abbrev}[/]")
                else:  # minor or None
                    parts.append(f"[yellow]{abbrev}[/]")

            return ", ".join(parts) if parts else "[dim]-[/]"

        # Agent and user errors
        agent_err_str = "[dim]-[/]"
        user_err_str = "[dim]-[/]"
        agent_tags_str = "[dim]-[/]"
        user_tags_str = "[dim]-[/]"

        if sim.review is not None:
            agent_errors = [e for e in sim.review.errors if e.source == "agent"]
            user_errors = [e for e in sim.review.errors if e.source == "user"]

            # Agent errors with severity coloring
            if agent_errors:
                has_critical = any(e.severity == "critical" for e in agent_errors)
                if has_critical:
                    agent_err_str = f"[red]{len(agent_errors)}![/]"
                else:
                    agent_err_str = f"[yellow]{len(agent_errors)}[/]"
                agent_tags_str = format_tags_with_severity(agent_errors, is_user=False)
            else:
                agent_err_str = "[green]0[/]"

            # User errors with severity coloring
            if user_errors:
                has_critical = any(
                    e.severity in ("critical_helped", "critical_hindered")
                    for e in user_errors
                )
                if has_critical:
                    user_err_str = f"[red]{len(user_errors)}![/]"
                else:
                    user_err_str = f"[yellow]{len(user_errors)}[/]"
                user_tags_str = format_tags_with_severity(user_errors, is_user=True)
            else:
                user_err_str = "[green]0[/]"

        elif sim.user_only_review is not None:
            user_errors = sim.user_only_review.errors
            if user_errors:
                has_critical = any(
                    e.severity in ("critical_helped", "critical_hindered")
                    for e in user_errors
                )
                if has_critical:
                    user_err_str = f"[red]{len(user_errors)}![/]"
                else:
                    user_err_str = f"[yellow]{len(user_errors)}[/]"
                user_tags_str = format_tags_with_severity(user_errors, is_user=True)
            else:
                user_err_str = "[green]0[/]"

        # Truncate task ID if needed
        task_id = sim.task_id
        if len(task_id) > 12:
            task_id = task_id[:9] + "..."

        table.add_row(
            str(i),
            task_id,
            str(sim.trial),
            pass_str,
            db_str,
            read_str,
            write_str,
            auth_str,
            stop_str,
            unresponsive_str,
            agent_err_str,
            user_err_str,
            agent_tags_str,
            user_tags_str,
        )

    ConsoleDisplay.console.print()
    ConsoleDisplay.console.print(table)

    if only_show_all_failed:
        num_all_failed = len([1 for v in num_success.values() if v == 0])
        ConsoleDisplay.console.print(
            f"\nTotal number of failed tasks: {num_all_failed}"
        )


def display_available_files(files):
    """Display a numbered list of available simulation files."""
    ConsoleDisplay.console.print("\n[bold blue]Available Simulation Files:[/]")
    for i, file in enumerate(files, 1):
        dir_name = file.parent.name
        ConsoleDisplay.console.print(f"[cyan]{i}.[/] {dir_name}")


def display_simulation_with_task(
    simulation: SimulationRun,
    task: Task,
    results_file: str,
    sim_index: int,
    domain: str,
    show_details: bool = True,
    consolidated_ticks: bool = True,
    tick_duration_ms: Optional[int] = None,
):
    """Display a simulation along with its associated task."""
    ConsoleDisplay.console.print("\n" + "=" * 80)  # Separator
    ConsoleDisplay.console.print("[bold blue]Task Details:[/]")
    ConsoleDisplay.display_task(task)

    ConsoleDisplay.console.print("\n" + "=" * 80)  # Separator
    ConsoleDisplay.console.print("[bold blue]Simulation Details:[/]")
    ConsoleDisplay.display_simulation(
        simulation,
        show_details=show_details,
        consolidated_ticks=consolidated_ticks,
        tick_duration_ms=tick_duration_ms,
    )

    # Show action menu (skip, add notes, or create task issue)
    handle_post_simulation_action(simulation, task, domain, results_file, sim_index)


def parse_key(key: str) -> tuple[str, int]:
    """Parse a key into a task ID and trial number."""
    task_id, trial = key.split("-")
    return task_id, int(trial)


def find_task_by_id(tasks, task_id):
    """Find a task in the task list by its ID."""
    for task in tasks:
        if task.id == task_id:
            return task
    return None


def find_simulation_by_task_id_and_trial(results, task_id, trial):
    """Get a simulation by its task ID and trial number."""
    return next(
        (
            sim
            for sim in results.simulations
            if sim.task_id == task_id and sim.trial == trial
        ),
        None,
    )


def get_notes_directories() -> list[Path]:
    """Get list of existing notes directories."""
    notes_base = Path(DATA_DIR) / "notes"
    if not notes_base.exists():
        return []
    return sorted([d for d in notes_base.iterdir() if d.is_dir()])


def prompt_for_notes_directory() -> Optional[Path]:
    """
    Prompt user to select an existing notes directory or create a new one.

    Returns:
        Path to the selected/created notes directory, or None if cancelled.
    """
    notes_base = Path(DATA_DIR) / "notes"
    existing_dirs = get_notes_directories()

    ConsoleDisplay.console.print("\n[bold blue]Select Notes Directory[/]")

    if existing_dirs:
        ConsoleDisplay.console.print("\nExisting directories:")
        for i, d in enumerate(existing_dirs, 1):
            # Count existing notes in directory (only *_note.json files, not *_simulation.json)
            note_count = len(list(d.glob("*_note.json")))
            ConsoleDisplay.console.print(f"  {i}. {d.name} ({note_count} notes)")
        ConsoleDisplay.console.print(
            f"  {len(existing_dirs) + 1}. [Create new directory]"
        )

        choice = Prompt.ask(
            "\nSelect directory number or enter new name",
            default=str(len(existing_dirs) + 1),
        )

        # Check if it's a number selecting existing dir
        try:
            choice_num = int(choice)
            if 1 <= choice_num <= len(existing_dirs):
                return existing_dirs[choice_num - 1]
            elif choice_num == len(existing_dirs) + 1:
                # Create new directory
                new_name = Prompt.ask("Enter new directory name")
                if not new_name.strip():
                    ConsoleDisplay.console.print(
                        "[yellow]Directory name required. Cancelled.[/]"
                    )
                    return None
                new_dir = notes_base / new_name.strip()
                new_dir.mkdir(parents=True, exist_ok=True)
                ConsoleDisplay.console.print(f"[green]Created directory: {new_dir}[/]")
                return new_dir
            else:
                ConsoleDisplay.console.print("[red]Invalid selection.[/]")
                return None
        except ValueError:
            # Treat as new directory name
            new_dir = notes_base / choice.strip()
            new_dir.mkdir(parents=True, exist_ok=True)
            ConsoleDisplay.console.print(f"[green]Created directory: {new_dir}[/]")
            return new_dir
    else:
        # No existing directories, prompt for new name
        ConsoleDisplay.console.print("[dim]No existing notes directories found.[/]")
        new_name = Prompt.ask("Enter new directory name")
        if not new_name.strip():
            ConsoleDisplay.console.print(
                "[yellow]Directory name required. Cancelled.[/]"
            )
            return None
        new_dir = notes_base / new_name.strip()
        new_dir.mkdir(parents=True, exist_ok=True)
        ConsoleDisplay.console.print(f"[green]Created directory: {new_dir}[/]")
        return new_dir


def prompt_for_simulation_note(
    simulation: SimulationRun, task: Task, results_file: str
) -> Optional[SimulationNote]:
    """
    Prompt user to create a simulation note with structured fields.

    Returns:
        SimulationNote object, or None if cancelled.
    """
    ConsoleDisplay.console.print("\n[bold blue]Create Simulation Note[/]")
    ConsoleDisplay.console.print(
        "[dim]Fill in the note details (press Enter to skip optional fields)[/]\n"
    )

    # Required: note
    note_text = Prompt.ask("[bold]Note[/] (required)")
    if not note_text.strip():
        ConsoleDisplay.console.print("[yellow]Note is required. Cancelled.[/]")
        return None

    # Error source (optional but recommended)
    ConsoleDisplay.console.print(
        "\n[dim]Error source options: agent, user, system (framework/orchestrator)[/]"
    )
    error_source_str = Prompt.ask(
        "[bold]Error source[/]",
        choices=["agent", "user", "system", ""],
        default="",
    )
    error_source: Optional[ErrorSource] = None
    if error_source_str:
        error_source = ErrorSource(error_source_str)

    # Error type (optional)
    error_type_choices = [e.value for e in ErrorType] + [""]
    ConsoleDisplay.console.print(
        "\n[dim]Error type options: transcription, vad, logical, hallucination, unresponsive, early_termination[/]"
    )
    error_type_str = Prompt.ask(
        "[bold]Error type[/]",
        choices=error_type_choices,
        default="",
    )
    error_type: Optional[ErrorType] = None
    if error_type_str:
        error_type = ErrorType(error_type_str)

    # Optional: author email
    author_email = Prompt.ask("[dim]Your email[/]", default="")
    author_email = author_email.strip() if author_email.strip() else None

    # Generate note ID
    note_id = f"note_{uuid.uuid4().hex[:8]}"

    return SimulationNote(
        id=note_id,
        note=note_text.strip(),
        author_email=author_email,
        created_at=datetime.now().isoformat(),
        simulation_id=simulation.id,
        task_id=simulation.task_id,
        trial=simulation.trial,
        source_results_file=results_file,
        error_source=error_source,
        error_type=error_type,
    )


def save_simulation_note(
    note: SimulationNote,
    simulation: SimulationRun,
    notes_dir: Path,
) -> bool:
    """
    Save a simulation note to a directory.

    .. deprecated::
        This function saves *_simulation.json files which are very large (100K-350K lines)
        and should NOT be committed to git. Use the results-based export mode in
        ``export_html.py --results`` instead, which reads results.json directly without
        creating intermediate simulation dump files.

    Saves both:
    1. The note metadata as a JSON file
    2. The full simulation run as a separate JSON file

    Args:
        note: The SimulationNote object to save.
        simulation: The simulation run to save alongside the note.
        notes_dir: The directory to save the note to.

    Returns:
        True if saved successfully, False otherwise.
    """
    import warnings

    warnings.warn(
        "save_simulation_note() creates large *_simulation.json files that should not "
        "be committed to git. Use 'export_html.py --results' instead, which reads "
        "results.json directly. See src/experiments/tau_voice/annotation/README.md.",
        DeprecationWarning,
        stacklevel=2,
    )
    ConsoleDisplay.console.print(
        "[yellow]⚠ Warning: This saves a large *_simulation.json file. "
        "These files are gitignored and should not be committed. "
        "Consider using 'export_html.py --results' instead.[/]"
    )
    try:
        # Create filename based on task and note ID
        base_filename = f"task_{note.task_id}_{note.id}"

        # Save simulation run
        sim_filename = f"{base_filename}_simulation.json"
        sim_path = notes_dir / sim_filename
        with open(sim_path, "w") as f:
            f.write(simulation.model_dump_json(indent=2))

        # Save note metadata (with reference to simulation file)
        note_with_sim_ref = note.model_copy()
        note_filename = f"{base_filename}_note.json"
        note_path = notes_dir / note_filename

        # Create a dict with note data and simulation file reference
        note_data = note_with_sim_ref.model_dump()
        note_data["simulation_file"] = sim_filename

        with open(note_path, "w") as f:
            json.dump(note_data, f, indent=2)

        ConsoleDisplay.console.print(f"[green]Saved note to: {note_path}[/]")
        ConsoleDisplay.console.print(f"[green]Saved simulation to: {sim_path}[/]")
        return True

    except Exception as e:
        ConsoleDisplay.console.print(f"[red]Error saving note: {e}[/]")
        return False


def prompt_for_task_issue(
    simulation: SimulationRun, task: Task, domain: str
) -> Optional[TaskIssue]:
    """Prompt user to create a task issue with all relevant details."""
    ConsoleDisplay.console.print("\n[bold blue]Create Task Issue[/]")
    ConsoleDisplay.console.print(
        "[dim]Fill in the issue details (press Enter to skip optional fields)[/]\n"
    )

    # Required: title
    title = Prompt.ask("[bold]Issue title[/] (required)")
    if not title.strip():
        ConsoleDisplay.console.print("[yellow]Issue title is required. Cancelled.[/]")
        return None

    # Optional: description
    description = Prompt.ask("[dim]Description[/]", default="")
    description = description.strip() if description.strip() else None

    # Status (default: open)
    status_choice = Prompt.ask(
        "[dim]Status[/]",
        choices=["open", "resolved", "wont_fix"],
        default="open",
    )
    status = TaskIssueStatus(status_choice)

    # Optional: resolution (more relevant if resolved/wont_fix)
    resolution = None
    if status in (TaskIssueStatus.RESOLVED, TaskIssueStatus.WONT_FIX):
        resolution = Prompt.ask("[dim]Resolution explanation[/]", default="")
        resolution = resolution.strip() if resolution.strip() else None

    # Optional: author email
    author_email = Prompt.ask("[dim]Your email[/]", default="")
    author_email = author_email.strip() if author_email.strip() else None

    # Optional: PR link
    pr_link = Prompt.ask("[dim]PR link (if any)[/]", default="")
    pr_link = pr_link.strip() if pr_link.strip() else None

    # Generate issue ID
    issue_id = f"issue_{uuid.uuid4().hex[:8]}"

    # Create the issue
    issue = TaskIssue(
        id=issue_id,
        title=title.strip(),
        description=description,
        status=status,
        resolution=resolution,
        created_at=datetime.now().strftime("%Y-%m-%d"),
        resolved_at=datetime.now().strftime("%Y-%m-%d")
        if status == TaskIssueStatus.RESOLVED
        else None,
        author_email=author_email,
        pr_link=pr_link,
        simulation_file=None,  # Will be set if user saves simulation
    )

    return issue


def save_task_issue(
    issue: TaskIssue,
    task: Task,
    domain: str,
    simulation: Optional[SimulationRun] = None,
) -> bool:
    """
    Save a task issue:
    1. Optionally save the simulation run to the task_issues directory
    2. Update the tasks.json file with the new issue
    """
    domain_data_dir = Path(DATA_DIR) / "tau2" / "domains" / domain
    tasks_file = domain_data_dir / "tasks.json"

    if not tasks_file.exists():
        ConsoleDisplay.console.print(f"[red]Tasks file not found: {tasks_file}[/]")
        return False

    # Save simulation if provided
    if simulation is not None:
        task_issues_dir = domain_data_dir / "task_issues"
        task_issues_dir.mkdir(exist_ok=True)

        # Create filename based on task and issue
        sim_filename = f"task_{task.id}_{issue.id}.json"
        sim_path = task_issues_dir / sim_filename

        # Save simulation run
        with open(sim_path, "w") as f:
            f.write(simulation.model_dump_json(indent=2))

        # Update issue with relative path to simulation file
        issue.simulation_file = f"task_issues/{sim_filename}"
        ConsoleDisplay.console.print(f"[green]Saved simulation to: {sim_path}[/]")

    # Load existing tasks
    try:
        with open(tasks_file, "r") as f:
            tasks_data = json.load(f)
    except Exception as e:
        ConsoleDisplay.console.print(f"[red]Error loading tasks: {e}[/]")
        return False

    # Find the task and add the issue
    task_found = False
    for task_dict in tasks_data:
        if task_dict.get("id") == task.id:
            task_found = True
            # Initialize issues list if not present
            if task_dict.get("issues") is None:
                task_dict["issues"] = []
            # Add the new issue
            task_dict["issues"].append(issue.model_dump(exclude_none=True))
            break

    if not task_found:
        ConsoleDisplay.console.print(f"[red]Task {task.id} not found in tasks.json[/]")
        return False

    # Save updated tasks
    try:
        with open(tasks_file, "w") as f:
            json.dump(tasks_data, f, indent=4)
        ConsoleDisplay.console.print(
            f"[green]Added issue to task {task.id} in tasks.json[/]"
        )
        return True
    except Exception as e:
        ConsoleDisplay.console.print(f"[red]Error saving tasks: {e}[/]")
        return False


def handle_post_simulation_action(
    simulation: SimulationRun,
    task: Task,
    domain: str,
    results_file: str,
    sim_index: int,
):
    """Handle the post-simulation action menu: skip, add notes, or create task issue."""
    ConsoleDisplay.console.print("\n" + "=" * 80)
    ConsoleDisplay.console.print("[bold blue]Actions:[/]")
    ConsoleDisplay.console.print("1. Skip (do nothing)")
    ConsoleDisplay.console.print("2. Add notes")
    ConsoleDisplay.console.print("3. Create task issue")

    choice = Prompt.ask(
        "\nWhat would you like to do?",
        choices=["1", "2", "3"],
        default="1",
    )

    if choice == "1":
        # Skip
        return

    elif choice == "2":
        # Add notes - prompt for directory first
        notes_dir = prompt_for_notes_directory()
        if notes_dir is None:
            return

        # Prompt for note details
        note = prompt_for_simulation_note(simulation, task, results_file)
        if note is None:
            return

        # Save the note and simulation
        success = save_simulation_note(note, simulation, notes_dir)

        if success:
            ConsoleDisplay.console.print("\n[bold green]Note saved successfully![/]")
            ConsoleDisplay.console.print(f"  Note ID: {note.id}")
            ConsoleDisplay.console.print(f"  Directory: {notes_dir.name}")

    elif choice == "3":
        # Create task issue
        issue = prompt_for_task_issue(simulation, task, domain)
        if issue is None:
            return

        # Ask if user wants to save simulation as example
        save_sim = Confirm.ask(
            "\nSave this simulation run as an example for the issue?",
            default=True,
        )

        sim_to_save = simulation if save_sim else None
        success = save_task_issue(issue, task, domain, sim_to_save)

        if success:
            ConsoleDisplay.console.print(
                "\n[bold green]Task issue created successfully![/]"
            )
            ConsoleDisplay.console.print(f"  Issue ID: {issue.id}")
            ConsoleDisplay.console.print(f"  Title: {issue.title}")
            if issue.simulation_file:
                ConsoleDisplay.console.print(f"  Simulation: {issue.simulation_file}")


def main(
    sim_file: Optional[str] = None,
    only_show_failed: bool = False,
    only_show_all_failed: bool = False,
    sim_dir: Optional[str] = None,
    expanded_ticks: bool = False,
):
    # Get available simulation files
    if sim_file is None:
        custom_sim_dir = Path(sim_dir) if sim_dir else None
        sim_files = get_available_simulations(custom_sim_dir)
    else:
        sim_files = [Path(sim_file)]

    if not sim_files:
        dir_path = sim_dir if sim_dir else f"{DATA_DIR}/simulations"
        ConsoleDisplay.console.print(f"[red]No simulation files found in {dir_path}[/]")
        return

    results = None
    current_file = None
    current_sim_path = None
    current_results_file = None  # Full path to results.json for notes
    while True:
        # Show main menu
        ConsoleDisplay.console.print("\n[bold yellow]Main Menu:[/]")
        ConsoleDisplay.console.print("1. Select simulation file")
        ConsoleDisplay.console.print(
            "   [dim]Choose a simulation results file to load and analyze[/]"
        )
        if results:
            ConsoleDisplay.console.print("2. View agent performance metrics")
            ConsoleDisplay.console.print("   [dim]Display agent performance metrics[/]")
            ConsoleDisplay.console.print("3. View simulation")
            ConsoleDisplay.console.print(
                "   [dim]Examine a specific simulation in detail with all its data[/]"
            )
            ConsoleDisplay.console.print("4. View task details")
            ConsoleDisplay.console.print(
                "   [dim]Look at the configuration and parameters of a specific task[/]"
            )
            ConsoleDisplay.console.print("5. View run configuration")
            ConsoleDisplay.console.print(
                "   [dim]Display the configuration used for this simulation run[/]"
            )

            has_voice = False
            voice_sim_path = None
            audio_file = None
            if current_sim_path:
                # Check for half-duplex voice (voice/sim_*/conversation.wav)
                voice_dirs = list((current_sim_path / "voice").glob("sim_*"))
                if voice_dirs:
                    voice_sim_path = voice_dirs[0]
                    conversation_audio = voice_sim_path / "conversation.wav"
                    if conversation_audio.exists():
                        has_voice = True
                        audio_file = conversation_audio

                # Check for full-duplex audio (audio/both.wav or tasks/*/sim_*/audio/both.wav)
                if not has_voice:
                    # First check top-level audio directory
                    top_audio = current_sim_path / "audio" / "both.wav"
                    if top_audio.exists():
                        has_voice = True
                        audio_file = top_audio
                        voice_sim_path = current_sim_path / "audio"
                    else:
                        # Check task-specific audio directories
                        task_audio_dirs = list(
                            current_sim_path.glob("tasks/*/sim_*/audio")
                        )
                        if task_audio_dirs:
                            voice_sim_path = task_audio_dirs[0]
                            both_audio = voice_sim_path / "both.wav"
                            if both_audio.exists():
                                has_voice = True
                                audio_file = both_audio

            if has_voice:
                ConsoleDisplay.console.print("6. Listen to voice conversation")
                ConsoleDisplay.console.print(
                    "   [dim]View simulation and play audio simultaneously[/]"
                )
                ConsoleDisplay.console.print("7. Exit")
                ConsoleDisplay.console.print("   [dim]Close the simulation viewer[/]")
                choices = ["1", "2", "3", "4", "5", "6", "7"]
            else:
                ConsoleDisplay.console.print("6. Exit")
                ConsoleDisplay.console.print("   [dim]Close the simulation viewer[/]")
                choices = ["1", "2", "3", "4", "5", "6"]
            default_choice = "3"
        else:
            ConsoleDisplay.console.print("2. Exit")
            ConsoleDisplay.console.print("   [dim]Close the simulation viewer[/]")
            choices = ["1", "2"]
            default_choice = "1"

        choice = Prompt.ask(
            "\nWhat would you like to do?", choices=choices, default=default_choice
        )

        if choice == "1":
            # Show available files and get selection
            display_available_files(sim_files)
            # default to view the last file
            file_num = IntPrompt.ask(
                f"\nSelect file number (1-{len(sim_files)})", default=len(sim_files)
            )

            if 1 <= file_num <= len(sim_files):
                try:
                    current_results_file = str(sim_files[file_num - 1])
                    current_sim_path = sim_files[file_num - 1].parent
                    current_file = current_sim_path.name
                    results = Results.load(sim_files[file_num - 1])
                    ConsoleDisplay.console.print(
                        f"\n[bold green]Loaded {len(results.simulations)} simulations from {current_file}[/]"
                    )

                    # Sort by task_id (as int if possible), then trial
                    def sort_key(sim):
                        task_id = sim.task_id
                        try:
                            return (0, int(task_id), sim.trial)
                        except ValueError:
                            return (1, task_id, sim.trial)

                    results.simulations = sorted(results.simulations, key=sort_key)
                except Exception as e:
                    ConsoleDisplay.console.print(
                        f"[red]Error loading results:[/] {str(e)}"
                    )
            else:
                ConsoleDisplay.console.print("[red]Invalid file number[/]")

        elif choice == "2" and not results:
            break

        elif results and choice == "2":
            # Display metrics
            ConsoleDisplay.console.clear()
            metrics = compute_metrics(results)
            ConsoleDisplay.display_agent_metrics(metrics)
            continue

        elif results and choice == "3":
            # Show list of simulations
            display_simulation_list(results, only_show_failed, only_show_all_failed)

            # Get simulation selection by index
            sim_count = len(results.simulations)
            sim_index = IntPrompt.ask(
                f"\nEnter simulation number (1-{sim_count})", default=1
            )

            if 1 <= sim_index <= sim_count:
                sim = results.simulations[sim_index - 1]
                task = find_task_by_id(results.tasks, sim.task_id)
                domain = results.info.environment_info.domain_name
                tick_duration = get_tick_duration_ms(results)
                if task:
                    display_simulation_with_task(
                        sim,
                        task,
                        current_results_file,
                        sim_index,
                        domain=domain,
                        show_details=True,
                        consolidated_ticks=not expanded_ticks,
                        tick_duration_ms=tick_duration,
                    )
                else:
                    ConsoleDisplay.console.print(
                        f"[red]Warning: Could not find task for simulation {sim.id}[/]"
                    )
                    ConsoleDisplay.display_simulation(
                        sim,
                        show_details=True,
                        consolidated_ticks=not expanded_ticks,
                        tick_duration_ms=tick_duration,
                    )
                continue
            else:
                ConsoleDisplay.console.print("[red]Invalid simulation number[/]")
                continue

        elif results and choice == "4":
            # Show list of tasks
            ConsoleDisplay.console.print("\n[bold blue]Available Tasks:[/]")
            for i, task in enumerate(results.tasks, 1):
                task_text = Text()
                task_text.append(f"{i}.", style="cyan")
                task_text.append(" Task ID: ")
                task_text.append(task.id)  # This will display square brackets correctly
                ConsoleDisplay.console.print(task_text)

            # Get task selection
            task_count = len(results.tasks)
            task_num = IntPrompt.ask(f"\nEnter task number (1-{task_count})", default=1)

            if 1 <= task_num <= task_count:
                ConsoleDisplay.console.clear()
                ConsoleDisplay.display_task(results.tasks[task_num - 1])
                continue
            else:
                ConsoleDisplay.console.print("[red]Invalid task number[/]")
                continue

        elif results and choice == "5":
            # Display run configuration
            ConsoleDisplay.console.clear()
            ConsoleDisplay.display_info(results.info)
            continue

        elif results and choice == "6" and has_voice:
            # Listen to voice conversation - select simulation first
            display_simulation_list(results, only_show_failed, only_show_all_failed)

            sim_count = len(results.simulations)
            sim_index = IntPrompt.ask(
                f"\nSelect simulation to listen to (1-{sim_count})", default=1
            )

            if 1 <= sim_index <= sim_count:
                sim = results.simulations[sim_index - 1]
                task = find_task_by_id(results.tasks, sim.task_id)

                # Find audio file for this specific simulation
                sim_audio_file = None
                # Check task-specific audio first
                task_audio_path = (
                    current_sim_path
                    / "tasks"
                    / f"task_{sim.task_id}"
                    / f"sim_{sim.id}"
                    / "audio"
                    / "both.wav"
                )
                if task_audio_path.exists():
                    sim_audio_file = task_audio_path
                elif audio_file:
                    # Fall back to top-level audio
                    sim_audio_file = audio_file

                if sim_audio_file and sim_audio_file.exists():
                    # Display simulation content
                    ConsoleDisplay.console.clear()
                    if task:
                        ConsoleDisplay.console.print("\n" + "=" * 80)
                        ConsoleDisplay.console.print("[bold blue]Task Details:[/]")
                        ConsoleDisplay.display_task(task)

                    ConsoleDisplay.console.print("\n" + "=" * 80)
                    ConsoleDisplay.console.print("[bold blue]Simulation Details:[/]")
                    ConsoleDisplay.display_simulation(
                        sim,
                        show_details=True,
                        consolidated_ticks=not expanded_ticks,
                        tick_duration_ms=get_tick_duration_ms(results),
                    )

                    # Start audio playback
                    ConsoleDisplay.console.print("\n" + "=" * 80)
                    ConsoleDisplay.console.print("\n[bold blue]🎧 Playing audio...[/]")
                    ConsoleDisplay.console.print(
                        "[dim]Press Ctrl+C to stop playback[/]"
                    )
                    ConsoleDisplay.console.print(
                        f"[dim]Audio file: {sim_audio_file}[/]"
                    )

                    process = subprocess.Popen(["afplay", str(sim_audio_file)])

                    try:
                        process.wait()
                        ConsoleDisplay.console.print(
                            "[green]✓ Finished playing audio[/]"
                        )
                    except KeyboardInterrupt:
                        process.terminate()
                        process.wait()
                        ConsoleDisplay.console.print(
                            "\n[yellow]Audio playback stopped[/]"
                        )

                    Prompt.ask("\n[dim]Press Enter to continue[/]")
                else:
                    ConsoleDisplay.console.print(
                        f"[red]Audio file not found for simulation {sim.id}[/]"
                    )
            else:
                ConsoleDisplay.console.print("[red]Invalid simulation number[/]")
            continue

        else:
            break

    ConsoleDisplay.console.print("\n[green]Thanks for using the simulation viewer![/]")


if __name__ == "__main__":
    main()
