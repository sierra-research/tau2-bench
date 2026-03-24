import math
import shutil
from datetime import date
from pathlib import Path
from typing import Literal, Optional

from rich.console import Console
from rich.prompt import Confirm, Prompt

from tau2.config import VOICE_USER_SIMULATOR_VERSION
from tau2.data_model.simulation import Results as TrajectoryResults
from tau2.metrics.agent_metrics import AgentMetrics, compute_metrics
from tau2.scripts.leaderboard.submission import (
    SUBMISSION_FILE_NAME,
    TRAJECTORY_FILES_DIR_NAME,
    ContactInfo,
    DomainResults,
    Methodology,
    Reference,
    Results,
    Submission,
    SubmissionData,
    Verification,
    VoiceConfig,
)
from tau2.scripts.leaderboard.verify_trajectories import (
    VerificationMode,
    verify_trajectories,
)
from tau2.utils.io_utils import expand_paths
from tau2.utils.utils import get_dict_hash, get_tau2_version


def _detect_voice_mode(results_list: list[TrajectoryResults]) -> bool:
    """Auto-detect whether the submission is voice-based.

    Returns True if any result has audio_native_config set in its info block.
    """
    return any(r.info.audio_native_config is not None for r in results_list)


def _extract_voice_config(results: TrajectoryResults) -> VoiceConfig:
    """Extract VoiceConfig from a voice trajectory's info block."""
    anc = results.info.audio_native_config
    if anc is None:
        raise ValueError("Cannot extract voice config: audio_native_config is None")

    # Extract user TTS provider info
    user_tts_provider = None
    user_voice = results.info.user_info.voice_settings
    if user_voice and user_voice.synthesis_config:
        sc = user_voice.synthesis_config
        provider = sc.provider
        model_id = None
        if sc.provider_config:
            model_id = getattr(sc.provider_config, "model_id", None)
        user_tts_provider = f"{provider}/{model_id}" if model_id else provider

    return VoiceConfig(
        provider=anc.provider,
        model=anc.model,
        tick_duration_seconds=getattr(anc, "tick_duration_seconds", None),
        max_steps_seconds=getattr(anc, "max_steps_seconds", None),
        user_tts_provider=user_tts_provider,
    )


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
        return (
            False,
            f"Trajectory files directory {trajectory_files_dir} not found",
            None,
        )

    # Get trajectory files.
    # For voice submissions the trajectories dir contains experiment
    # subdirectories (each with its own results.json + simulations/ etc.),
    # so we look for results.json one level deep to avoid picking up
    # individual sim files from the simulations/ subdirectory.
    is_voice = submission.modality == "voice"
    if is_voice:
        trajectory_files = sorted(
            str(f) for f in trajectory_files_dir.glob("*/results.json")
        )
        if not trajectory_files:
            trajectory_files = sorted(
                str(f) for f in trajectory_files_dir.glob("results.json")
            )
    else:
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

    verify_trajectories(submission_data.trajectory_files, mode=mode)
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
        # Create DomainResults object (values as percentages, matching submission format)
        _pct = lambda v: v * 100 if v is not None else None  # noqa: E731
        cost = metrics.avg_agent_cost
        if cost is not None and math.isnan(cost):
            cost = None
        domain_result = DomainResults(
            pass_1=_pct(metrics.pass_hat_ks.get(1)),
            pass_2=_pct(metrics.pass_hat_ks.get(2)),
            pass_3=_pct(metrics.pass_hat_ks.get(3)),
            pass_4=_pct(metrics.pass_hat_ks.get(4)),
            cost=cost,
        )
        # Include retrieval_config for banking_knowledge domain
        if domain == "banking_knowledge" and results.info.retrieval_config:
            domain_result.retrieval_config = results.info.retrieval_config
        domain_results[domain] = domain_result

    return domain_metrics, domain_results, default_model, default_user_simulator


def validate_submission_metrics(
    submission: Submission, submitted_results: list[TrajectoryResults], console: Console
) -> None:
    """
    Validate the submission metrics.
    """
    warnings = []
    _, computed_domain_results, default_model, default_user_simulator = get_metrics(
        submitted_results
    )
    if submission.model_name != default_model:
        warnings.append(
            f"Model name {submission.model_name} does not match model used for the trajectories set {default_model}"
        )
    if (
        submission.methodology
        and submission.methodology.user_simulator != default_user_simulator
    ):
        warnings.append(
            f"User simulator {submission.methodology.user_simulator} does not match user simulator used for the trajectories set {default_user_simulator}"
        )
    for domain, computed_results in computed_domain_results.items():
        domain_results = submission.results.get_domain_results(domain)
        if domain_results is None:
            warnings.append(
                f"Domain {domain} found in trajectories but missing from submission.json"
            )
            continue
        if domain_results.pass_1 != computed_results.pass_1:
            warnings.append(
                f"Pass^1 for {domain} does not match computed results {computed_results.pass_1}"
            )
        if domain_results.pass_2 != computed_results.pass_2:
            warnings.append(
                f"Pass^2 for {domain} does not match computed results {computed_results.pass_2}"
            )
        if domain_results.pass_3 != computed_results.pass_3:
            warnings.append(
                f"Pass^3 for {domain} does not match computed results {computed_results.pass_3}"
            )
        if domain_results.pass_4 != computed_results.pass_4:
            warnings.append(
                f"Pass^4 for {domain} does not match computed results {computed_results.pass_4}"
            )
        if domain_results.cost != computed_results.cost:
            warnings.append(
                f"Cost for {domain} does not match computed results {computed_results.cost}"
            )
    if warnings:
        console.print(f"❌ {len(warnings)} warning(s) found", style="red")
        for warning in warnings:
            console.print(f"  • {warning}", style="red")
    else:
        console.print("✅ Submission metrics validation successful!", style="green")


def _copy_voice_experiment_trimmed(
    exp_src: Path,
    exp_dst: Path,
    results: TrajectoryResults,
    console: Console,
) -> int:
    """Copy a voice experiment directory, keeping only what's needed.

    Saves the results in directory-based format (metadata in ``results.json``,
    individual simulations in ``simulations/``). If the source is in monolithic
    JSON format, it is automatically converted. For each task, only the
    canonical simulation's ``audio/`` subdirectory from ``artifacts/`` is
    copied.  Skips ``hallucination_discarded/``, ``llm_debug/``,
    ``sim_status.json``, ``task.log``, and non-canonical simulation
    directories.

    Args:
        exp_src: Source experiment directory.
        exp_dst: Destination directory for the trimmed copy.
        results: Already-loaded TrajectoryResults (used for task-to-sim
            mapping and for conversion when source is monolithic JSON).
        console: Rich console for output.

    Returns the total size in bytes of all copied files.
    """
    total_bytes = 0
    exp_dst.mkdir(parents=True, exist_ok=True)

    # Always save in dir format — converts from monolithic JSON if needed.
    src_fmt = TrajectoryResults._detect_format(exp_src / "results.json")
    results.save(exp_dst / "results.json", format="dir")

    if src_fmt != "dir":
        console.print("    Converted monolithic JSON → dir format", style="dim")

    # Tally written file sizes
    for f in (exp_dst / "results.json",):
        total_bytes += f.stat().st_size
    sims_dst = exp_dst / "simulations"
    if sims_dst.is_dir():
        n_sims = 0
        for f in sims_dst.rglob("*"):
            if f.is_file():
                total_bytes += f.stat().st_size
                n_sims += 1
        console.print(
            f"    Wrote simulations/ ({n_sims} file(s))",
            style="dim",
        )

    # Build task_id -> sim_id mapping from loaded results
    task_to_sim: dict[str, str] = {}
    for sim in results.simulations:
        task_to_sim[str(sim.task_id)] = sim.id

    artifacts_dir = exp_src / "artifacts"
    if not artifacts_dir.is_dir():
        return total_bytes

    copied_audio = 0
    skipped_sims = 0

    for task_dir in sorted(artifacts_dir.iterdir()):
        if not task_dir.is_dir() or not task_dir.name.startswith("task_"):
            continue

        task_id = task_dir.name.split("_", 1)[1]
        canonical_sim_id = task_to_sim.get(task_id)
        if canonical_sim_id is None:
            skipped_sims += sum(1 for d in task_dir.iterdir() if d.is_dir())
            continue

        canonical_sim_name = f"sim_{canonical_sim_id}"
        for sim_dir in sorted(task_dir.iterdir()):
            if not sim_dir.is_dir() or not sim_dir.name.startswith("sim_"):
                continue
            if sim_dir.name != canonical_sim_name:
                skipped_sims += 1
                continue

            audio_src = sim_dir / "audio"
            if not audio_src.is_dir():
                console.print(
                    f"    ⚠️  Missing audio/ in {task_dir.name}/{sim_dir.name}",
                    style="yellow",
                )
                continue

            audio_dst = exp_dst / "artifacts" / task_dir.name / sim_dir.name / "audio"
            shutil.copytree(audio_src, audio_dst)
            for f in audio_dst.rglob("*"):
                if f.is_file():
                    total_bytes += f.stat().st_size
            copied_audio += 1

    if skipped_sims:
        console.print(
            f"    Skipped {skipped_sims} non-canonical simulation dir(s)",
            style="dim",
        )
    console.print(
        f"    Kept audio for {copied_audio} task(s)",
        style="dim",
    )

    return total_bytes


def prepare_submission(
    input_paths: list[str],
    output_dir: str,
    run_verification: bool = True,
    voice: Optional[bool] = None,
):
    """Prepare the submission for the leaderboard.

    Processes trajectory files to create a complete leaderboard submission.
    Performs trajectory verification (optional), computes metrics, and creates
    a submission file with interactive user input.

    Supports both text (half-duplex) and voice (audio-native full-duplex)
    submissions.  Voice mode is auto-detected from the input data when
    ``voice`` is None.  For voice submissions, only results with "regular"
    speech complexity are accepted.

    Args:
        input_paths: List of paths to trajectory files, directories, or glob
            patterns.
        output_dir: Root directory for the prepared output.
        run_verification: Whether to run trajectory verification before
            processing.
        voice: If True, force voice submission mode. If False, force text
            mode.  If None (default), auto-detect from input data.

    Output Structure (text)::

        output_dir/
        └── {model}_{org}_{date}/
            ├── submission.json         # Goes into the repo
            └── trajectories/           # Uploaded to external storage
                ├── domain1_results.json
                └── domain2_results.json

    Output Structure (voice)::

        output_dir/
        └── {model}_{org}_{date}/
            ├── submission.json
            └── trajectories/
                └── <experiment_name>/       # One per domain
                    ├── results.json         # Metadata only
                    ├── simulations/         # Individual sim data files
                    │   ├── sim_0.json
                    │   └── ...
                    └── artifacts/           # Canonical audio only
                        └── task_<id>/
                            └── sim_<uuid>/
                                └── audio/

    Voice results are always stored in directory-based format. If the
    source uses monolithic JSON, it is automatically converted.

    The full directory (including trajectories/) is uploaded to external
    storage (S3, Google Drive, etc.).  Only ``submission.json`` is copied
    into ``web/leaderboard/public/submissions/`` in the repo.
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

    # Load all trajectory data upfront
    trajectory_results = [TrajectoryResults.load(path) for path in files]

    # Auto-detect or confirm voice mode
    is_voice = voice if voice is not None else _detect_voice_mode(trajectory_results)
    modality: Literal["text", "voice"] = "voice" if is_voice else "text"
    if is_voice:
        console.print(
            "🎙️  Voice submission detected (audio-native mode)", style="bold magenta"
        )
    else:
        console.print("📝 Text submission detected", style="bold blue")

    # For voice submissions, filter to "regular" complexity only
    if is_voice:
        regular_results = [
            r for r in trajectory_results if r.info.speech_complexity == "regular"
        ]
        non_regular = [
            r for r in trajectory_results if r.info.speech_complexity != "regular"
        ]
        if non_regular:
            skipped_complexities = {r.info.speech_complexity for r in non_regular}
            console.print(
                f"  ⚠️  Skipping {len(non_regular)} result file(s) with "
                f"non-regular complexity: {skipped_complexities}",
                style="yellow",
            )
        if not regular_results:
            console.print(
                "❌ No results with 'regular' speech complexity found. "
                "Voice submissions require 'regular' complexity results.",
                style="red",
            )
            return
        trajectory_results = regular_results
        # Update files list to match filtered results (for downstream steps)
        regular_files = []
        for f_path, r in zip(files, [TrajectoryResults.load(p) for p in files]):
            if r.info.speech_complexity == "regular":
                regular_files.append(f_path)
        files = regular_files

    # Step 1: Verify trajectories if requested (text only)
    if run_verification and not is_voice:
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
    valid, error = validate_submission_traj_set(trajectory_results)
    if not valid:
        console.print(f"❌ Submission set validation failed: {error}", style="red")
        return

    # Step 3: Compute metrics by domain
    console.print("\n📊 Computing metrics...", style="bold blue")
    domain_metrics: dict[str, AgentMetrics] = {}
    domain_results: dict[str, DomainResults] = {}
    default_model = None
    default_user_simulator = None
    voice_config: Optional[VoiceConfig] = None
    trajectory_files_map = {}  # domain -> filename for submission.json

    for results, file_path in zip(trajectory_results, files):
        try:
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

            # Extract voice config from the first voice result
            if is_voice and voice_config is None:
                voice_config = _extract_voice_config(results)

            # Track trajectory references by domain.
            # Populated with final filenames/paths during the copy step below.
            trajectory_files_map[domain] = file_path

            # Compute metrics for this trajectory file
            metrics = compute_metrics(results)
            domain_metrics[domain] = metrics

            # Create DomainResults object
            def _pct(val: float | None) -> float | None:
                return val * 100 if val is not None else None

            cost = metrics.avg_agent_cost
            if cost is not None and math.isnan(cost):
                cost = None

            domain_result = DomainResults(
                pass_1=_pct(metrics.pass_hat_ks.get(1)),
                pass_2=_pct(metrics.pass_hat_ks.get(2)),
                pass_3=_pct(metrics.pass_hat_ks.get(3)),
                pass_4=_pct(metrics.pass_hat_ks.get(4)),
                cost=cost,
            )
            # Include retrieval_config for banking_knowledge domain
            if domain == "banking_knowledge":
                if results.info.retrieval_config:
                    domain_result.retrieval_config = results.info.retrieval_config
                else:
                    console.print(
                        "  ⚠️  banking_knowledge trajectory is missing retrieval_config. "
                        "You will be prompted to enter it manually.",
                        style="yellow",
                    )
            domain_results[domain] = domain_result

            console.print(
                f"  ✅ Processed {domain} trajectories from {Path(file_path).name}"
            )

        except Exception as e:
            console.print(f"  ❌ Error processing {file_path}: {e}", style="red")
            return

    # Step 4: Create submission object and gather user input
    console.print("\n📝 Creating submission...", style="bold blue")

    # For voice, derive a better default model name from voice_config
    default_model_display = default_model
    if is_voice and voice_config:
        default_model_display = voice_config.model

    # Gather required information
    model_name = Prompt.ask("Enter model name", default=default_model_display)
    if is_voice:
        user_simulator = Prompt.ask(
            "Enter voice user simulator version (see git tags voice-user-sim-*)",
            default=VOICE_USER_SIMULATOR_VERSION,
        )
    else:
        user_simulator = Prompt.ask(
            "Enter user simulator model", default=default_user_simulator
        )
    model_organization = Prompt.ask(
        "Enter model organization (who developed the model)",
        default="My-Organization",
    )
    submitting_organization = Prompt.ask(
        "Enter submitting organization (who ran the evaluation)",
        default=model_organization,
    )
    email = Prompt.ask("Enter contact email")

    # Optional information
    console.print("\n📋 Optional information (press Enter to skip):", style="dim")
    contact_name = Prompt.ask("Contact name", default="") or None
    github_username = Prompt.ask("GitHub username", default="") or None

    is_new = Confirm.ask(
        "Should this model be highlighted as new on the leaderboard?", default=True
    )

    # Submission type
    submission_type = Prompt.ask(
        "Submission type",
        choices=["standard", "custom"],
        default="standard",
    )

    # Methodology information
    console.print("\n🔬 Methodology information:", style="dim")
    evaluation_date_str = Prompt.ask(
        "Evaluation date (YYYY-MM-DD)", default=str(date.today())
    )
    evaluation_date = None
    if evaluation_date_str:
        try:
            evaluation_date = date.fromisoformat(evaluation_date_str)
        except ValueError:
            console.print("Invalid date format, skipping...", style="yellow")

    tau2_version = Prompt.ask("Tau-bench version", default=get_tau2_version()) or None
    notes = Prompt.ask("Additional notes", default="") or None

    # Verification information
    console.print("\n🔍 Verification information:", style="dim")
    modified_prompts = Confirm.ask(
        "Did you modify any prompts (agent or user simulator)?", default=False
    )
    omitted_questions = Confirm.ask(
        "Did you omit any questions/tasks from the evaluation?", default=False
    )
    verification_details = None
    if modified_prompts or omitted_questions:
        verification_details = (
            Prompt.ask("Please describe the modifications/omissions", default="")
            or None
        )

    verification = Verification(
        modified_prompts=modified_prompts,
        omitted_questions=omitted_questions,
        details=verification_details,
    )

    # References (optional)
    console.print("\n📎 References (optional):", style="dim")
    references = []
    add_reference = Confirm.ask(
        "Add a reference link (paper, GitHub, etc.)?", default=False
    )
    while add_reference:
        ref_title = Prompt.ask("Reference title")
        ref_url = Prompt.ask("Reference URL")
        ref_type = Prompt.ask(
            "Reference type",
            choices=[
                "paper",
                "blog_post",
                "documentation",
                "model_card",
                "github",
                "huggingface",
                "other",
            ],
            default="other",
        )
        references.append(Reference(title=ref_title, url=ref_url, type=ref_type))
        add_reference = Confirm.ask("Add another reference?", default=False)

    # Create submission objects
    contact_info = ContactInfo(email=email, name=contact_name, github=github_username)

    methodology = Methodology(
        evaluation_date=evaluation_date,
        tau2_bench_version=tau2_version,
        user_simulator=user_simulator,
        notes=notes,
        verification=verification,
    )

    # Ensure banking_knowledge has retrieval_config (required by schema)
    banking_results = domain_results.get("banking_knowledge")
    if banking_results and not banking_results.retrieval_config:
        console.print(
            "\n🔍 Banking knowledge retrieval configuration:", style="bold blue"
        )
        retrieval_config_value = Prompt.ask(
            "Enter retrieval config used for banking_knowledge "
            "(e.g., 'terminal', 'text-emb-3-large', 'qwen3-emb', 'bm25')",
        )
        banking_results.retrieval_config = retrieval_config_value

    results_obj = Results(
        retail=domain_results.get("retail"),
        airline=domain_results.get("airline"),
        telecom=domain_results.get("telecom"),
        banking_knowledge=banking_results,
    )

    # Step 6: Create output directory and copy trajectory files
    def _slugify(s: str) -> str:
        return s.lower().replace(" ", "-").replace("/", "-").replace(".", "-")

    submission_dir_name = (
        f"{_slugify(model_name)}_{_slugify(model_organization)}"
        f"_{date.today().isoformat()}"
    )

    output_path = Path(output_dir)
    submission_dir = output_path / submission_dir_name
    submission_dir.mkdir(parents=True, exist_ok=True)

    trajectories_dir = submission_dir / TRAJECTORY_FILES_DIR_NAME
    trajectories_dir.mkdir(exist_ok=True)

    console.print(f"\n📁 Output: {submission_dir}", style="bold blue")

    # trajectory_files_map currently holds raw source paths; rebuild it with
    # the actual destination names/paths produced during copy.
    domain_source_paths = dict(trajectory_files_map)
    trajectory_files_map.clear()

    # Build domain -> loaded results lookup for the copy step
    domain_to_results: dict[str, TrajectoryResults] = {}
    for r in trajectory_results:
        domain_to_results[r.info.environment_info.domain_name] = r

    if is_voice:
        # Voice: copy the trimmed experiment directory per domain
        # (results data + canonical simulation audio only).
        for domain, src_path in domain_source_paths.items():
            exp_src = Path(src_path).parent
            exp_name = exp_src.name
            exp_dst = trajectories_dir / exp_name
            console.print(f"  📂 {TRAJECTORY_FILES_DIR_NAME}/{exp_name}/", style="bold")
            total_bytes = _copy_voice_experiment_trimmed(
                exp_src, exp_dst, domain_to_results[domain], console
            )
            console.print(f"    Total: {total_bytes / 1e6:.1f} MB")
            trajectory_files_map[domain] = exp_name
    else:
        # Text: copy results files, using {domain}_results.json to avoid
        # collisions when multiple domains share the same filename.
        for domain, src_path in domain_source_paths.items():
            src = Path(src_path)
            dest_name = (
                src.name if src.name != "results.json" else f"{domain}_results.json"
            )
            dest_path = trajectories_dir / dest_name
            shutil.copy2(src, dest_path)
            size_mb = dest_path.stat().st_size / 1e6
            console.print(
                f"  📂 {TRAJECTORY_FILES_DIR_NAME}/{dest_name} ({size_mb:.1f} MB)"
            )
            trajectory_files_map[domain] = dest_name

    # Step 7: Write submission.json (after copy so trajectory_files_map is final)
    submission = Submission(
        model_name=model_name,
        model_organization=model_organization,
        submitting_organization=submitting_organization,
        submission_date=date.today(),
        submission_type=submission_type,
        modality=modality,
        contact_info=contact_info,
        results=results_obj,
        is_new=is_new,
        trajectories_available=bool(trajectory_files_map),
        trajectory_files=trajectory_files_map if trajectory_files_map else None,
        references=references if references else None,
        methodology=methodology,
        voice_config=voice_config,
    )

    submission_file = submission_dir / SUBMISSION_FILE_NAME
    with open(submission_file, "w", encoding="utf-8") as f:
        f.write(
            submission.model_dump_json(indent=2, exclude_none=True, ensure_ascii=False)
        )
        f.write("\n")
    console.print(f"  📊 {SUBMISSION_FILE_NAME}")

    # Summary
    console.print(f"\n🎉 Submission prepared successfully!", style="bold green")
    console.print(f"🎯 Modality: {modality}", style="bold")
    console.print(f"\n📈 Results:", style="bold")
    for domain, dr in domain_results.items():
        console.print(f"  {domain.capitalize()}: ", style="bold", end="")
        pass_scores = []
        for k in [1, 2, 3, 4]:
            score = getattr(dr, f"pass_{k}")
            if score is not None:
                pass_scores.append(f"Pass^{k}: {score:.1f}%")
        console.print(" | ".join(pass_scores) if pass_scores else "No scores available")

    if is_voice and voice_config:
        console.print(f"\n🎙️  Voice config:", style="bold")
        console.print(f"  Provider: {voice_config.provider}")
        console.print(f"  Model: {voice_config.model}")
        if voice_config.tick_duration_seconds:
            console.print(f"  Tick duration: {voice_config.tick_duration_seconds}s")
        if voice_config.user_tts_provider:
            console.print(f"  User TTS: {voice_config.user_tts_provider}")

    manifest_array = "voice_submissions" if is_voice else "submissions"
    console.print(f"\n💡 Next steps:", style="bold blue")
    console.print(f"  1. Review {submission_dir_name}/{SUBMISSION_FILE_NAME}")
    console.print(
        f"  2. Upload the full [bold]{submission_dir_name}/[/bold] directory "
        f"(with trajectories) to external storage (S3, Google Drive, etc.) "
        f"and share the link with the maintainers"
    )
    console.print(
        f"  3. Copy [bold]only[/bold] {submission_dir_name}/{SUBMISSION_FILE_NAME} "
        f"to web/leaderboard/public/submissions/{submission_dir_name}/"
    )
    console.print(
        f"  4. Add [bold]{submission_dir_name}[/bold] to the "
        f"[bold]{manifest_array}[/bold] array in manifest.json"
    )
    console.print("  5. Submit a pull request")
