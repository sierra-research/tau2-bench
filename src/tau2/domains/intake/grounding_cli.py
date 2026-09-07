# Copyright Sierra
"""CLI for the intake grounding program: ``tau2 intake-grounding``.

Package-owned registration (docs/CODE_DESIGN.md):
:func:`add_intake_grounding_args` is wired into the top-level parser in
:mod:`tau2.cli`.

``screen`` runs the negative reality screen over the checked-in entity banks
(email-domain DNS non-resolution, EDGAR/GLEIF registry checks for coined
names, insurance carriers, properties, and shops) and writes the
provenance-bearing report (see :mod:`tau2.domains.intake.grounding_screen`).

``vpic`` and ``rxnorm`` run the positive grounding screens: every
vehicles-bank value and decoy against NHTSA vPIC
(:mod:`tau2.domains.intake.grounding_vpic`) and every medications-bank value
and decoy against NLM RxNorm (:mod:`tau2.domains.intake.grounding_rxnorm`).

Every verb is report-only and never edits a bank: findings are printed for
the owner to act on, and the exit code is 0 even with findings — nonzero
means the screen itself failed to run. Network access happens only here, at
build time.
"""

import argparse
from pathlib import Path


def add_intake_grounding_args(parser: argparse.ArgumentParser) -> None:
    """Attach the intake-grounding subcommands and handlers to ``parser``."""
    subparsers = parser.add_subparsers(dest="intake_grounding_command", required=True)

    screen = subparsers.add_parser(
        "screen",
        help="Screen the fictional bank content against reality (email-domain "
        "DNS, SEC EDGAR + GLEIF registry names) and write the "
        "provenance-bearing report. Reports only; never edits a bank.",
    )
    screen.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="Report path (default: data/tau2/domains/intake/grounding_screen.json)",
    )
    screen.set_defaults(func=_run_screen)

    vpic = subparsers.add_parser(
        "vpic",
        help="Validate every vehicles-bank value and decoy (make + model) "
        "against NHTSA vPIC and write the provenance-bearing report. "
        "Reports only; never edits a bank.",
    )
    vpic.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="Report path (default: data/tau2/domains/intake/grounding_vpic.json)",
    )
    vpic.set_defaults(func=_run_vpic)

    rxnorm = subparsers.add_parser(
        "rxnorm",
        help="Validate every medications-bank value and decoy (ingredient + "
        "strength + form) against NLM RxNorm and write the provenance-bearing "
        "report. Reports only; never edits a bank.",
    )
    rxnorm.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="Report path (default: data/tau2/domains/intake/grounding_rxnorm.json)",
    )
    rxnorm.set_defaults(func=_run_rxnorm)


def _run_screen(args: argparse.Namespace) -> None:
    from rich.console import Console

    from tau2.domains.intake.grounding_screen import (
        Verdict,
        run_screen,
        write_screen_report,
    )

    report = run_screen()
    path = write_screen_report(report, out_path=args.out)

    console = Console()
    console.print(
        f"screened {len(report.dns_checks)} email domains and "
        f"{len(report.registry_checks)} registry names "
        f"(screen v{report.screen_version}, tau2 {report.tool_version})"
    )
    console.print(
        f"verdicts: {report.passes} pass, {report.findings} finding, "
        f"{report.unchecked} unchecked"
    )
    for check in report.dns_checks:
        if check.verdict is Verdict.FINDING:
            records = sorted(
                {lookup.rrtype for lookup in check.lookups if lookup.answers}
            )
            console.print(
                f"  FINDING emails domain {check.domain!r} resolves "
                f"({', '.join(records)} records exist)"
            )
    for check in report.registry_checks:
        if check.verdict is Verdict.FINDING:
            matched = check.edgar_matches + check.gleif_matches
            source = "EDGAR" if check.edgar_matches else "GLEIF"
            console.print(
                f"  FINDING {check.bank} name {check.name!r} matches real "
                f"{source} entity: {', '.join(matched)}"
            )
    if report.unchecked:
        console.print(
            f"  NOTE {report.unchecked} check(s) UNCHECKED (lookup failures) — "
            "re-run to cover them; unchecked is not a pass"
        )
    console.print(f"written: {path}")


def _print_positive_summary(console, report, what: str) -> None:
    console.print(
        f"checked {len(report.checks)} {what} "
        f"(screen v{report.screen_version}, tau2 {report.tool_version})"
    )
    console.print(
        f"verdicts: {report.passes} pass, {report.findings} finding, "
        f"{report.unchecked} unchecked"
    )
    for check in report.checks:
        if check.verdict.value == "finding":
            console.print(f"  FINDING {check.role} {check.value!r}: {check.note}")
    if report.unchecked:
        console.print(
            f"  NOTE {report.unchecked} check(s) UNCHECKED (lookup failures) — "
            "re-run to cover them; unchecked is not a pass"
        )


def _run_vpic(args: argparse.Namespace) -> None:
    from rich.console import Console

    from tau2.domains.intake.grounding_vpic import run_vpic_screen, write_vpic_report

    report = run_vpic_screen()
    path = write_vpic_report(report, out_path=args.out)
    console = Console()
    _print_positive_summary(console, report, "vehicles values/decoys against vPIC")
    console.print(f"written: {path}")


def _run_rxnorm(args: argparse.Namespace) -> None:
    from rich.console import Console

    from tau2.domains.intake.grounding_rxnorm import (
        run_rxnorm_screen,
        write_rxnorm_report,
    )

    report = run_rxnorm_screen()
    path = write_rxnorm_report(report, out_path=args.out)
    console = Console()
    _print_positive_summary(console, report, "medications values/decoys against RxNorm")
    console.print(f"written: {path}")
