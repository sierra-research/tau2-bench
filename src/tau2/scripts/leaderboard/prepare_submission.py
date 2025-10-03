import shutil
from datetime import date
from pathlib import Path

from rich.console import Console
from rich.prompt import Confirm, Prompt

from tau2.data_model.simulation import Results as TrajectoryResults
from tau2.metrics.agent_metrics import AgentMetrics, compute_metrics
from tau2.scripts.leaderboard.submission import (
    SUBMISSION_FILE_NAME,
    TRAJECTORY_FILES_DIR_NAME,
    ContactInfo,
    DomainResults,
    Methodology,
    Results,
    Submission,
    SubmissionData,
)
from tau2.scripts.leaderboard.verify_trajectories import (
    VerificationMode,
    verify_trajectories,
)
from tau2.utils.io_utils import expand_paths
from tau2.utils.utils import get_dict_hash


def check_and_load_submission_data(
    submission_dir: str,
) -> tuple[bool, str, SubmissionData]:
    """
    Checks submission directory and loads submission data.
    """
    if not Path(submission_dir).exists():
        return False, f"Submission directory {submission_dir} not found", None

    # Check that submission file exists
    submission_file = Path(submission_dir) / SUBMISSION_FILE_NAME
    if not submission_file.exists():
        return False, f"Submission file {submission_file} not found", None

    submission = None
    with open(submission_file, "r") as f:
        submission = Submission.model_validate_json(f.read())

    # Check that trajectory files directory exists
    trajectory_files_dir = Path(submission_dir) / TRAJECTORY_FILES_DIR_NAME
    if not trajectory_files_dir.exists():
        return False, f"Trajectory files directory {trajectory_files_dir} not found"

    # Get trajectory files
    trajectory_files = expand_paths([trajectory_files_dir], extension=".json")
    results = [TrajectoryResults.load(path) for path in trajectory_files]

    submission_data = SubmissionData(
        submission_dir=submission_dir,
        submission_file=str(submission_file),
        trajectory_files=trajectory_files,
        submission=submission,
        results=results,
    )
    return True, "", submission_data


def validate_submission_traj_set(
    all_results: list[TrajectoryResults],
) -> tuple[bool, str]:
    """
    Validate the submission trajectory set.
    Each domain should only appear once.
    All results should be using the same agent llm with same arguments.
    All results should be using the same user simulator with same arguments.
    Returns:
        tuple[bool, str]: True if the submission set is valid, False otherwise
    """
    domain_names = set()
    for results in all_results:
        domain = results.info.environment_info.domain_name
        if domain in domain_names:
            return False, f"Domain {domain} appears multiple times"
        domain_names.add(domain)
    agent_user_info = None
    for results in all_results:
        res_agent_user_info = {
            "llm_agent": results.info.agent_info.llm,
            "llm_args_agent": results.info.agent_info.llm_args,
            "llm_user": results.info.user_info.llm,
            "llm_args_user": results.info.user_info.llm_args,
        }
        if agent_user_info is None:
            agent_user_info = res_agent_user_info
        else:
            if get_dict_hash(res_agent_user_info) != get_dict_hash(agent_user_info):
                return (
                    False,
                    f"Agent / User Simulator should be the same for all results. Got {agent_user_info} and {res_agent_user_info}",
                )

    return True, ""


def validate_submission(
    submission_dir: str, mode: VerificationMode = VerificationMode.PUBLIC
):
    """
    Validate the submission.
    """
    console = Console()
    console.print("🔍 Validating submission...", style="bold blue")
    console.print(f"📂 Submission directory: {submission_dir}", style="bold")
    console.print("📋 Loading submission data...", style="bold")
    valid, error, submission_data = check_and_load_submission_data(submission_dir)
    if not valid:
        console.print(f"❌ Submission validation failed: {error}", style="red")
        return
    console.print("✅ Submission data loaded successfully!", style="green")
    console.print("📋 Validating submission trajectory set...", style="bold")
    valid, error = validate_submission_traj_set(submission_data.results)
    if not valid:
        console.print(
            f"❌ Submission trajectory set validation failed: {error}", style="red"
        )
        return

    verify_trajectories(submission_data.trajectory_files, mode=VerificationMode.PUBLIC)
    console.print("✅ Submission validation successful!", style="green")
    console.print("📋 Validating submission metrics...", style="bold")
    validate_submission_metrics(
        submission_data.submission, submission_data.results, console
    )


def get_metrics(
    submitted_results: list[TrajectoryResults],
) -> tuple[dict[str, AgentMetrics], dict[str, DomainResults], str, str]:
    """
    Computes the metrics for all submitted trajectories set.
    Returns:
        tuple[dict[str, AgentMetrics], dict[str, DomainResults], str, str]:
            - domain_metrics: Metrics for each domain
            - domain_results: Results for each domain
            - default_model: Default model used for the submission
            - default_user_simulator: Default user simulator used for the submission
    """
    domain_metrics: dict[str, AgentMetrics] = {}
    domain_results = {}
    default_model = None
    default_user_simulator = None

    for results in submitted_results:
        domain = results.info.environment_info.domain_name
        if default_model is None:
            default_model = results.info.agent_info.llm
        if default_user_simulator is None:
            default_user_simulator = results.info.user_info.llm
        if domain in domain_metrics:
            raise ValueError(f"Domain {domain} appears multiple times")

        # Compute metrics for this trajectory file
        metrics = compute_metrics(results)
        domain_metrics[domain] = metrics
        # Create DomainResults object
        domain_results[domain] = DomainResults(
            pass_1=metrics.pass_hat_ks.get(1),
            pass_2=metrics.pass_hat_ks.get(2),
            pass_3=metrics.pass_hat_ks.get(3),
            pass_4=metrics.pass_hat_ks.get(4),
            cost=metrics.avg_agent_cost,
        )

    return domain_metrics, domain_results, default_model, default_user_simulator


def validate_submission_metrics(
    submission: Submission, submitted_results: list[TrajectoryResults], console: Console
) -> None:
    """
    Validate the submission metrics.
    """
    num_warnings = 0
    warnings = []
    _, computed_domain_results, default_model, default_user_simulator = get_metrics(
        submitted_results
    )
    if submission.model_name != default_model:
        warnings.append(
            f"Model name {submission.model_name} does not match model used for the trajectories set {default_model}"
        )
        num_warnings += 1
    if submission.methodology.user_simulator != default_user_simulator:
        warnings.append(
            f"User simulator {submission.user_simulator} does not match user simulator used for the trajectories set {default_user_simulator}"
        )
    for domain, computed_results in computed_domain_results.items():
        submitted_results = submission.results.get_domain_results(domain)
        if submitted_results.pass_1 != computed_results.pass_1:
            warnings.append(
                f"Pass^1 for {domain} does not match computed results {computed_results.pass_1}"
            )
        if submitted_results.pass_2 != computed_results.pass_2:
            warnings.append(
                f"Pass^2 for {domain} does not match computed results {computed_results.pass_2}"
            )
        if submitted_results.pass_3 != computed_results.pass_3:
            warnings.append(
                f"Pass^3 for {domain} does not match computed results {computed_results.pass_3}"
            )
        if submitted_results.pass_4 != computed_results.pass_4:
            warnings.append(
                f"Pass^4 for {domain} does not match computed results {computed_results.pass_4}"
            )
        if submitted_results.cost != computed_results.cost:
            warnings.append(
                f"Cost for {domain} does not match computed results {computed_results.cost}"
            )
    if num_warnings > 0:
        console.print(f"❌ {num_warnings} warnings found", style="red")
        for warning in warnings:
            console.print(f"  • {warning}", style="red")
    else:
        console.print("✅ Submission metrics validation successful!", style="green")


def prepare_submission(
    input_paths: list[str], output_dir: str, run_verification: bool = True
):
    """
    Prepare the submission for the leaderboard.

    This function processes trajectory files to create a complete leaderboard submission.
    It performs trajectory verification (optional), copies files to an organized structure,
    computes metrics, and creates a submission file with interactive user input.

    Args:
        input_paths: List of paths to trajectory files, directories, or glob patterns
        output_dir: Directory to save the submission file and trajectories
        run_verification: Whether to run trajectory verification before processing

    Output Structure:
        Creates the following in output_dir:
        - submission.json: Complete leaderboard submission file with metadata and metrics
        - trajectories/: Directory containing copies of all processed trajectory files

    The submission.json contains:
        - Model and organization information
        - Contact details
        - Performance metrics aggregated by domain (retail, airline, telecom)
        - Pass^k success rates (k=1,2,3,4) as percentages
        - Optional methodology information

    Interactive Input:
        Prompts user for required fields (model name, organization, email) and
        optional fields (contact name, GitHub, evaluation details) that can be skipped.
    """
    console = Console()
    # Step 0: Collect trajectory files
    console.print("\n📂 Collecting trajectory files...", style="bold blue")
    files = expand_paths(input_paths, extension=".json")
    if not files:
        console.print("❌ No trajectory files found", style="red")
        return

    console.print(f"Found {len(files)} trajectory file(s):", style="green")
    for file_path in files:
        console.print(f"  • {file_path}")

    # Step 1: Verify trajectories if requested
    if run_verification:
        console.print("🔍 Running trajectory verification...", style="bold blue")
        try:
            verify_trajectories(paths=files, mode=VerificationMode.PUBLIC)
            console.print("✅ All trajectories passed verification!", style="green")
        except SystemExit:
            console.print(
                "❌ Trajectory verification failed. Aborting submission preparation.",
                style="red",
            )
            return
        except Exception as e:
            console.print(f"❌ Error during verification: {e}", style="red")
            return

    # Step 2: Validate submission set
    console.print("🔍 Validating submission set...", style="bold blue")
    # Load trajectory data from files
    trajectory_results = [TrajectoryResults.load(path) for path in files]
    valid, error = validate_submission_traj_set(trajectory_results)
    if not valid:
        console.print(f"❌ Submission set validation failed: {error}", style="red")
        return

    # Step 3: Create output directory and copy files
    console.print(f"\n📁 Creating output directory: {output_dir}", style="bold blue")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create trajectories subdirectory
    trajectories_dir = output_path / TRAJECTORY_FILES_DIR_NAME
    trajectories_dir.mkdir(exist_ok=True)

    console.print("📋 Copying trajectory files...", style="bold blue")
    copied_files = []
    for file_path in files:
        filename = Path(file_path).name
        dest_path = trajectories_dir / filename
        shutil.copy2(file_path, dest_path)
        copied_files.append(str(dest_path))
        console.print(f"  ✅ Copied: {filename}")

    # Step 4: Load trajectories and compute metrics by domain
    console.print("\n📊 Computing metrics...", style="bold blue")
    domain_metrics: dict[str, AgentMetrics] = {}
    domain_results = {}
    default_model = None
    default_user_simulator = None

    for file_path in copied_files:
        try:
            results = TrajectoryResults.load(Path(file_path))
            domain = results.info.environment_info.domain_name
            if default_model is None:
                default_model = results.info.agent_info.llm
            if default_user_simulator is None:
                default_user_simulator = results.info.user_info.llm
            if domain in domain_metrics:
                console.print(
                    f"  ❌ Domain {domain} appears multiple times", style="red"
                )
                return

            # Compute metrics for this trajectory file
            metrics = compute_metrics(results)
            domain_metrics[domain] = metrics
            # Create DomainResults object
            domain_results[domain] = DomainResults(
                pass_1=metrics.pass_hat_ks.get(1) * 100,
                pass_2=metrics.pass_hat_ks.get(2) * 100,
                pass_3=metrics.pass_hat_ks.get(3) * 100,
                pass_4=metrics.pass_hat_ks.get(4) * 100,
                cost=metrics.avg_agent_cost,
            )

            console.print(
                f"  ✅ Processed {domain} trajectories from {Path(file_path).name}"
            )

        except Exception as e:
            console.print(f"  ❌ Error processing {file_path}: {e}", style="red")
            return

    # Step 5: Create submission object and gather user input
    console.print("\n📝 Creating submission...", style="bold blue")

    # Gather required information
    model_name = Prompt.ask("Enter model name", default=default_model)
    user_simulator = Prompt.ask(
        "Enter user simulator model", default=default_user_simulator
    )
    organization = Prompt.ask("Enter organization name", default="My-Organization")
    email = Prompt.ask("Enter contact email")

    # Optional information
    console.print("\n📋 Optional information (press Enter to skip):", style="dim")
    contact_name = Prompt.ask("Contact name", default="") or None
    github_username = Prompt.ask("GitHub username", default="") or None

    is_new = Confirm.ask(
        "Should this model be highlighted as new on the leaderboard?", default=False
    )

    # Methodology information
    console.print("\n🔬 Methodology information (optional):", style="dim")
    evaluation_date_str = Prompt.ask("Evaluation date (YYYY-MM-DD)", default="")
    evaluation_date = None
    if evaluation_date_str:
        try:
            evaluation_date = date.fromisoformat(evaluation_date_str)
        except ValueError:
            console.print("Invalid date format, skipping...", style="yellow")

    tau2_version = Prompt.ask("Tau2-bench version", default="") or None
    notes = Prompt.ask("Additional notes", default="") or None

    # Create submission objects
    contact_info = ContactInfo(email=email, name=contact_name, github=github_username)

    methodology = None
    if evaluation_date or tau2_version or notes:
        methodology = Methodology(
            evaluation_date=evaluation_date,
            tau2_bench_version=tau2_version,
            user_simulator=user_simulator,
            notes=notes,
        )

    results_obj = Results(
        retail=domain_results.get("retail"),
        airline=domain_results.get("airline"),
        telecom=domain_results.get("telecom"),
    )

    submission = Submission(
        model_name=model_name,
        organization=organization,
        submission_date=date.today(),
        contact_info=contact_info,
        results=results_obj,
        is_new=is_new,
        methodology=methodology,
    )

    # Step 6: Save submission
    submission_file = output_path / SUBMISSION_FILE_NAME
    with open(submission_file, "w") as f:
        f.write(submission.model_dump_json(indent=2, exclude_none=True))

    console.print(f"\n🎉 Submission prepared successfully!", style="bold green")
    console.print(f"📁 Output directory: {output_path}")
    console.print(f"📊 Submission file: {submission_file}")
    console.print(f"📂 Trajectories: {trajectories_dir}")
    console.print(f"\n📈 Summary:", style="bold")
    for domain, results in domain_results.items():
        console.print(f"  {domain.capitalize()}: ", style="bold", end="")
        pass_scores = []
        for k in [1, 2, 3, 4]:
            score = getattr(results, f"pass_{k}")
            if score is not None:
                pass_scores.append(f"Pass^{k}: {score:.1f}%")
        console.print(" | ".join(pass_scores) if pass_scores else "No scores available")

    console.print(f"\n💡 Next steps:", style="bold blue")
    console.print(f"  1. Review the {SUBMISSION_FILE_NAME} file")
    console.print(
        "  2. Submit to the leaderboard according to the submission guidelines"
    )
    console.print("  3. Keep the trajectories directory for reference")
