import argparse
import json

from loguru import logger

from experiments.hyperparam.responses_report import build_responses_report
from experiments.hyperparam.responses_sweep import (
    DEFAULT_BASELINE_REASONING,
    DEFAULT_BASELINE_SERVICE_TIER,
    DEFAULT_BASELINE_VERBOSITY,
    DEFAULT_BASELINE_WEB_SEARCH,
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORTS,
    DEFAULT_RESPONSES_MAX_DURATION_SECONDS,
    DEFAULT_RESPONSES_MAX_STEPS,
    DEFAULT_RESPONSES_TRANSPORTS,
    DEFAULT_SERVICE_TIERS,
    DEFAULT_VERBOSITIES,
    DEFAULT_WEB_SEARCH_MODES,
    SweepShape,
    canonicalize_response_repair_results,
    run_responses_sweep,
    validate_response_resume_state,
)
from experiments.hyperparam.responses_sweep import (
    RunMode as ResponsesRunMode,
)
from tau2.scripts.view_simulations import main as view_simulations_main
from tau2.utils.utils import DATA_DIR

DATA_EXP_DIR = DATA_DIR / "exp"

DEFAULT_LLM_SUPERVISOR = None
DEFAULT_LLM_USER = "gpt-4.1-2025-04-14"
DEFAULT_MAX_CONCURRENCY = 5
DEFAULT_NUM_TRIALS = 4
DEFAULT_SEED = 300
DEFAULT_MAX_STEPS = 200
DEFAULT_MAX_ERRORS = 10
DEFAULT_DOMAINS = ["retail", "airline", "telecom"]
DEFAULT_RESPONSES_DOMAINS = ["retail", "airline", "telecom"]
RESPONSES_DOMAINS = ["retail", "airline", "telecom", "banking_knowledge"]
DEFAULT_EVAL_MODES = [
    "default",
    "oracle-plan",
    "no-user",
    "no-user-op",
]
DEFAULT_RESPONSES_MODES = [
    "default",
]
DEFAULT_LLM_AGENT_ARGS = {"temperature": 0.0}
DEFAULT_LLM_USER_ARGS = {"temperature": 0.0}
DEFAULT_LLM_SUPERVISOR_ARGS = {"temperature": 0.0}


def get_cli_parser() -> argparse.ArgumentParser:
    """
    Get the CLI parser with subparsers for run-evals and analyze-results commands.
    """
    parser = argparse.ArgumentParser(
        description="Run evaluations and analyze results for experiments."
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Run evals subparser
    run_parser = subparsers.add_parser("run-evals", help="Run evaluation experiments")
    run_parser.add_argument(
        "--exp-dir",
        type=str,
        required=True,
        help=f"Path to the experiment directory relative to {DATA_EXP_DIR}. This will be created if it doesn't exist.",
    )
    run_parser.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="Number of tasks to run. Defaults to None.",
    )

    # Add hyperparameters arguments
    run_parser.add_argument(
        "--llms",
        type=str,
        nargs="+",
        required=True,
        help="List of LLMs to test (e.g. gpt-4.1-2025-04-14 claude-3-7-sonnet-20250219)",
    )
    run_parser.add_argument(
        "--domains",
        type=str,
        nargs="+",
        default=DEFAULT_DOMAINS,
        choices=DEFAULT_DOMAINS,
        help=f"List of domains to test. Default is {DEFAULT_DOMAINS}.",
    )
    run_parser.add_argument(
        "--modes",
        type=str,
        nargs="+",
        default=DEFAULT_EVAL_MODES,
        choices=DEFAULT_EVAL_MODES,
        help=f"List of modes to test. Default is {DEFAULT_EVAL_MODES}.",
    )

    # Add experiment parameters
    run_parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help="Random seed for experiments"
    )
    run_parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="Maximum number of steps per trial",
    )
    run_parser.add_argument(
        "--max-errors",
        type=int,
        default=DEFAULT_MAX_ERRORS,
        help="Maximum number of errors allowed",
    )
    run_parser.add_argument(
        "--max-concurrency",
        type=int,
        default=DEFAULT_MAX_CONCURRENCY,
        help=f"Maximum number of concurrent simulations. Default is {DEFAULT_MAX_CONCURRENCY}.",
    )
    run_parser.add_argument(
        "--num-trials",
        type=int,
        default=DEFAULT_NUM_TRIALS,
        help=f"Number of trials per configuration. Default is {DEFAULT_NUM_TRIALS}.",
    )

    # LLM model for user simulator
    run_parser.add_argument(
        "--llm-user",
        type=str,
        default=DEFAULT_LLM_USER,
        help=f"LLM model to use for user simulator. Default is {DEFAULT_LLM_USER}.",
    )

    # LLM arguments for agent and user simulator.
    run_parser.add_argument(
        "--agent-llm-args",
        type=str,
        default=json.dumps(DEFAULT_LLM_AGENT_ARGS),
        help=f"JSON string of arguments for agent LLM. Default is {DEFAULT_LLM_AGENT_ARGS}.",
    )
    run_parser.add_argument(
        "--user-llm-args",
        type=str,
        default=json.dumps(DEFAULT_LLM_USER_ARGS),
        help=f"JSON string of arguments for user LLM. Default is {DEFAULT_LLM_USER_ARGS}.",
    )

    # Analyze results subparser
    analyze_parser = subparsers.add_parser(
        "analyze-results", help="Analyze experiment results"
    )
    analyze_parser.add_argument(
        "--exp-dir",
        type=str,
        required=True,
        help="Path to the experiment directory containing results to analyze.",
    )

    responses_parser = subparsers.add_parser(
        "run-responses-sweep",
        help="Run a Responses API sweep across reasoning, verbosity, and hosted web search settings",
    )
    responses_parser.add_argument(
        "--exp-dir",
        type=str,
        default=None,
        help="Experiment name under data/exp/responses/. Defaults to a timestamped name.",
    )
    responses_parser.add_argument(
        "--shape",
        type=str,
        choices=[shape.value for shape in SweepShape],
        default=SweepShape.GRID.value,
        help="Sweep shape: full grid or one-factor-at-a-time baseline sweep.",
    )
    responses_parser.add_argument(
        "--llm",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Agent model to sweep. Default is {DEFAULT_MODEL}.",
    )
    responses_parser.add_argument(
        "--domains",
        type=str,
        nargs="+",
        default=DEFAULT_RESPONSES_DOMAINS,
        choices=RESPONSES_DOMAINS,
        help=f"List of domains to test. Default is {DEFAULT_RESPONSES_DOMAINS}.",
    )
    responses_parser.add_argument(
        "--modes",
        type=str,
        nargs="+",
        default=DEFAULT_RESPONSES_MODES,
        choices=[mode.value for mode in ResponsesRunMode],
        help="Modes to test. Default is ['default'].",
    )
    responses_parser.add_argument(
        "--reasoning-efforts",
        type=str,
        nargs="+",
        default=DEFAULT_REASONING_EFFORTS,
        help=f"Reasoning efforts to sweep. Default is {DEFAULT_REASONING_EFFORTS}.",
    )
    responses_parser.add_argument(
        "--verbosities",
        type=str,
        nargs="+",
        default=DEFAULT_VERBOSITIES,
        help=f"Verbosity levels to sweep. Default is {DEFAULT_VERBOSITIES}.",
    )
    responses_parser.add_argument(
        "--web-search-modes",
        type=str,
        nargs="+",
        default=DEFAULT_WEB_SEARCH_MODES,
        help=f"Hosted web search modes to sweep. Default is {DEFAULT_WEB_SEARCH_MODES}.",
    )
    responses_parser.add_argument(
        "--baseline-reasoning",
        type=str,
        default=DEFAULT_BASELINE_REASONING,
        help=f"OFAT baseline reasoning effort. Default is {DEFAULT_BASELINE_REASONING}.",
    )
    responses_parser.add_argument(
        "--baseline-verbosity",
        type=str,
        default=DEFAULT_BASELINE_VERBOSITY,
        help=f"OFAT baseline verbosity. Default is {DEFAULT_BASELINE_VERBOSITY}.",
    )
    responses_parser.add_argument(
        "--baseline-web-search",
        type=str,
        default=DEFAULT_BASELINE_WEB_SEARCH,
        help=f"OFAT baseline web search mode. Default is {DEFAULT_BASELINE_WEB_SEARCH}.",
    )
    responses_parser.add_argument(
        "--service-tiers",
        type=str,
        nargs="+",
        default=DEFAULT_SERVICE_TIERS,
        choices=["default", "batch", "flex", "priority"],
        help=f"Service tiers to sweep. Default is {DEFAULT_SERVICE_TIERS}.",
    )
    responses_parser.add_argument(
        "--baseline-service-tier",
        type=str,
        default=DEFAULT_BASELINE_SERVICE_TIER,
        choices=["default", "batch", "flex", "priority"],
        help=f"OFAT baseline service tier. Default is {DEFAULT_BASELINE_SERVICE_TIER}.",
    )
    responses_parser.add_argument(
        "--responses-transports",
        type=str,
        nargs="+",
        default=DEFAULT_RESPONSES_TRANSPORTS,
        choices=["http", "websocket"],
        help=(
            "Responses transport modes to sweep. Default is "
            f"{DEFAULT_RESPONSES_TRANSPORTS}."
        ),
    )
    responses_parser.add_argument(
        "--parallel-tool-calls",
        type=str,
        nargs="+",
        default=None,
        choices=["unset", "true", "false"],
        help=(
            "Optional explicit parallel_tool_calls values to sweep. "
            "Use unset to preserve the API default."
        ),
    )
    responses_parser.add_argument(
        "--known-variant-suite",
        action="store_true",
        default=False,
        help=(
            "Run the known follow-up variants: cached baseline, gpt-5.5 low, "
            "parallel_tool_calls true/false, and WebSocket transport."
        ),
    )
    responses_parser.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="Number of tasks to run per configuration. Use a small value for smoke tests.",
    )
    responses_parser.add_argument(
        "--task-ids",
        type=str,
        nargs="+",
        default=None,
        help="Specific task ids to run instead of sampling by num_tasks.",
    )
    responses_parser.add_argument(
        "--task-split-name",
        type=str,
        default="base",
        help="Task split name. Default is base.",
    )
    responses_parser.add_argument(
        "--domain-task-splits",
        type=str,
        default=None,
        help='Optional JSON object overriding task splits by domain, e.g. {"telecom":"full"}.',
    )
    responses_parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help="Random seed for experiments"
    )
    responses_parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_RESPONSES_MAX_STEPS,
        help=(
            "Maximum number of steps per trial. "
            f"Default is {DEFAULT_RESPONSES_MAX_STEPS} for Responses exploratory runs."
        ),
    )
    responses_parser.add_argument(
        "--max-duration-seconds",
        "--timeout",
        dest="max_duration_seconds",
        type=float,
        default=DEFAULT_RESPONSES_MAX_DURATION_SECONDS,
        help=(
            "Maximum wall-clock seconds per simulation. "
            f"Default is {DEFAULT_RESPONSES_MAX_DURATION_SECONDS:g}; pass 0 to disable."
        ),
    )
    responses_parser.add_argument(
        "--max-errors",
        type=int,
        default=DEFAULT_MAX_ERRORS,
        help="Maximum number of errors allowed.",
    )
    responses_parser.add_argument(
        "--max-concurrency",
        type=int,
        default=DEFAULT_MAX_CONCURRENCY,
        help=f"Maximum number of concurrent simulations. Default is {DEFAULT_MAX_CONCURRENCY}.",
    )
    responses_parser.add_argument(
        "--num-trials",
        type=int,
        default=DEFAULT_NUM_TRIALS,
        help=f"Number of trials per configuration. Default is {DEFAULT_NUM_TRIALS}.",
    )
    responses_parser.add_argument(
        "--llm-user",
        type=str,
        default=DEFAULT_LLM_USER,
        help=f"LLM model to use for the user simulator. Default is {DEFAULT_LLM_USER}.",
    )
    responses_parser.add_argument(
        "--agent-llm-args",
        type=str,
        default=json.dumps(DEFAULT_LLM_AGENT_ARGS),
        help=f"Base JSON args for the agent LLM. Default is {DEFAULT_LLM_AGENT_ARGS}.",
    )
    responses_parser.add_argument(
        "--user-llm-args",
        type=str,
        default=json.dumps(DEFAULT_LLM_USER_ARGS),
        help=f"JSON args for the user simulator LLM. Default is {DEFAULT_LLM_USER_ARGS}.",
    )
    responses_parser.add_argument(
        "--web-search-context-size",
        type=str,
        choices=["low", "medium", "high"],
        default="medium",
        help="Hosted web search context size. Default is medium.",
    )
    responses_parser.add_argument(
        "--web-search-allowed-domains",
        type=str,
        nargs="+",
        default=None,
        help="Optional allowlist for hosted web search.",
    )
    responses_parser.add_argument(
        "--auto-resume",
        action="store_true",
        default=False,
        help="Automatically resume from any existing checkpointed run outputs.",
    )
    responses_parser.add_argument(
        "--reuse-from-exp-dirs",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Responses experiment names or absolute directories to reuse matching "
            "completed simulations from before running new API calls."
        ),
    )
    responses_parser.add_argument(
        "--failed-only",
        type=str,
        nargs="+",
        default=None,
        choices=["infrastructure_error"],
        help=(
            "Only run configs that already have matching failed simulations in "
            "simulations.csv. Uses auto-resume so completed tasks in those configs "
            "are skipped."
        ),
    )
    responses_parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help=(
            "Plan which simulations would run after auto-resume/cache reuse without "
            "running simulations or writing sweep outputs."
        ),
    )

    responses_report_parser = subparsers.add_parser(
        "build-responses-report",
        help="Build the portable HTML report for a Responses API sweep directory",
    )
    responses_report_parser.add_argument(
        "--exp-dir",
        type=str,
        required=True,
        help="Experiment name under data/exp/responses/ or an absolute experiment directory.",
    )

    canonicalize_parser = subparsers.add_parser(
        "canonicalize-responses-repair",
        help=(
            "Install repair experiment results into a Responses experiment's "
            "canonical raw/checkpoint paths and rewrite aggregate CSV rows."
        ),
    )
    canonicalize_parser.add_argument(
        "--exp-dir",
        type=str,
        required=True,
        help="Target Responses experiment name or absolute directory.",
    )
    canonicalize_parser.add_argument(
        "--repair-exp-dirs",
        type=str,
        nargs="+",
        required=True,
        help="Repair Responses experiment names or absolute directories to merge.",
    )

    validate_resume_parser = subparsers.add_parser(
        "validate-responses-resume-state",
        help="Validate that aggregate rows point at canonical auto-resume checkpoints.",
    )
    validate_resume_parser.add_argument(
        "--exp-dir",
        type=str,
        required=True,
        help="Responses experiment name or absolute directory.",
    )

    # View simulations subparser
    view_parser = subparsers.add_parser(
        "view", help="View simulation results interactively"
    )
    view_parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help=f"Directory containing simulation files. Defaults to {DATA_DIR}/simulations if not specified.",
    )
    view_parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Specific simulation file to view (optional).",
    )
    view_parser.add_argument(
        "--only-failed",
        action="store_true",
        help="Only show failed simulations.",
    )
    view_parser.add_argument(
        "--only-all-failed",
        action="store_true",
        help="Only show tasks where all trials failed.",
    )

    return parser


def main():
    """
    Run the evaluations or analyze results based on the command.
    """
    parser = get_cli_parser()
    args = parser.parse_args()

    if args.command == "run-evals":
        from experiments.hyperparam.analyze_results import analyze_results
        from experiments.hyperparam.run_eval import RunMode, make_configs, run_evals

        # Convert relative path to absolute path using DATA_EXP_DIR
        exp_dir = DATA_EXP_DIR / args.exp_dir

        # Parse hyperparameters
        hyperparams = {
            "llm": args.llms,
            "domain": args.domains,
            "mode": [RunMode(mode) for mode in args.modes],
        }

        # Parse LLM arguments
        llm_agent_args = json.loads(args.agent_llm_args)
        llm_user_args = json.loads(args.user_llm_args)

        logger.info(
            f"Running experiment in {exp_dir} with num_tasks {args.num_tasks}..."
        )
        if exp_dir.exists():
            res = input(f"Experiment directory {exp_dir} already exists. Run it? (y/n)")
            if res.lower().strip() != "y":
                return
        exp_dir.mkdir(parents=True, exist_ok=True)
        configs = make_configs(
            hyperparams=hyperparams,
            llm_user=args.llm_user,
            llm_agent_args=llm_agent_args,
            llm_user_args=llm_user_args,
            seed=args.seed,
            max_steps=args.max_steps,
            max_duration_seconds=(
                None if args.max_duration_seconds == 0 else args.max_duration_seconds
            ),
            max_errors=args.max_errors,
            max_concurrency=args.max_concurrency,
            num_trials=args.num_trials,
            num_tasks=args.num_tasks,
            exp_dir=exp_dir,
        )
        run_evals(configs)
        logger.info(f"Experiment in {exp_dir} completed.")
        analyze_results(exp_dir)

    elif args.command == "analyze-results":
        from experiments.hyperparam.analyze_results import analyze_results

        # Convert relative path to absolute path using DATA_DIR
        exp_dir = DATA_EXP_DIR / args.exp_dir
        analyze_results(exp_dir)

    elif args.command == "run-responses-sweep":
        exp_dir = run_responses_sweep(
            exp_name=args.exp_dir,
            shape=SweepShape(args.shape),
            llm=args.llm,
            domains=args.domains,
            modes=[ResponsesRunMode(mode) for mode in args.modes],
            llm_user=args.llm_user,
            llm_user_args=json.loads(args.user_llm_args),
            agent_llm_args=json.loads(args.agent_llm_args),
            seed=args.seed,
            max_steps=args.max_steps,
            max_duration_seconds=(
                None if args.max_duration_seconds == 0 else args.max_duration_seconds
            ),
            max_errors=args.max_errors,
            max_concurrency=args.max_concurrency,
            num_trials=args.num_trials,
            num_tasks=args.num_tasks,
            task_ids=args.task_ids,
            task_split_name=args.task_split_name,
            domain_task_splits=(
                json.loads(args.domain_task_splits)
                if args.domain_task_splits is not None
                else None
            ),
            auto_resume=args.auto_resume,
            reasoning_efforts=args.reasoning_efforts,
            verbosities=args.verbosities,
            web_search_modes=args.web_search_modes,
            service_tiers=args.service_tiers,
            responses_transports=args.responses_transports,
            parallel_tool_calls=(
                None
                if args.parallel_tool_calls is None
                else [
                    None
                    if value == "unset"
                    else True
                    if value == "true"
                    else False
                    for value in args.parallel_tool_calls
                ]
            ),
            known_variant_suite=args.known_variant_suite,
            baseline_reasoning=args.baseline_reasoning,
            baseline_verbosity=args.baseline_verbosity,
            baseline_web_search=args.baseline_web_search,
            baseline_service_tier=args.baseline_service_tier,
            web_search_context_size=args.web_search_context_size,
            web_search_allowed_domains=args.web_search_allowed_domains,
            reuse_from_exp_dirs=args.reuse_from_exp_dirs,
            failed_only_termination_reasons=args.failed_only,
            dry_run=args.dry_run,
        )
        if args.dry_run:
            logger.info(f"Responses sweep dry run inspected {exp_dir}")
        else:
            logger.info(f"Responses sweep results saved to {exp_dir}")
            report_path = build_responses_report(exp_dir)
            logger.info(f"Responses sweep report saved to {report_path}")

    elif args.command == "build-responses-report":
        report_path = build_responses_report(args.exp_dir)
        logger.info(f"Responses sweep report saved to {report_path}")

    elif args.command == "canonicalize-responses-repair":
        result = canonicalize_response_repair_results(
            exp_dir=args.exp_dir,
            repair_exp_dirs=args.repair_exp_dirs,
        )
        report_path = build_responses_report(args.exp_dir)
        logger.info(json.dumps(result, indent=2))
        logger.info(f"Responses sweep report saved to {report_path}")

    elif args.command == "validate-responses-resume-state":
        problems = validate_response_resume_state(args.exp_dir)
        if problems:
            for problem in problems:
                logger.error(problem)
            raise SystemExit(1)
        logger.info("Responses resume state is valid.")

    elif args.command == "view":
        # Run the view simulations interactive tool
        view_simulations_main(
            sim_file=args.file,
            only_show_failed=args.only_failed,
            only_show_all_failed=args.only_all_failed,
            sim_dir=args.dir,
        )

    else:
        parser.print_help()
        return


if __name__ == "__main__":
    main()
