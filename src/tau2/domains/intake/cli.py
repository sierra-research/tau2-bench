"""Run options owned by the elicitation domain."""

import argparse

from tau2.config import (
    DEFAULT_ELICITATION_COMPLICATION_RATE,
    DEFAULT_ELICITATION_STRATEGY,
)
from tau2.data_model.elicitation import (
    ElicitationConfig,
    ElicitationStrategy,
)


def add_intake_run_args(parser: argparse.ArgumentParser) -> None:
    """Register the intake strategy ablation and caller stimulus overrides."""
    group = parser.add_argument_group("τ-entity (formerly elicitation; intake only)")
    group.add_argument(
        "--elicitation-strategy",
        choices=[strategy.value for strategy in ElicitationStrategy],
        default=None,
        help=f"Collection strategy. Default: {DEFAULT_ELICITATION_STRATEGY}; "
        "scaffolded selects the prescribed-protocol ablation.",
    )
    group.add_argument(
        "--complication-rate",
        type=float,
        default=None,
        help="Complication trigger scaling in [0, 1]; 0 disables complications. "
        f"Default: {DEFAULT_ELICITATION_COMPLICATION_RATE:g} for both strategies. "
        "Inapplicable kinds retain their no-complication probability.",
    )


def elicitation_config_from_args(args: argparse.Namespace) -> ElicitationConfig | None:
    """Validate explicit overrides; domain defaults are resolved by RunConfig."""
    values = {}
    if args.elicitation_strategy is not None:
        values["strategy"] = args.elicitation_strategy
    if args.complication_rate is not None:
        values["rate"] = args.complication_rate
    return ElicitationConfig.model_validate(values) if values else None
