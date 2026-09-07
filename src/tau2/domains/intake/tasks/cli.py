# Copyright Sierra
"""CLI for the intake name-bank pipeline: ``tau2 intake-names``.

Package-owned registration (docs/CODE_DESIGN.md): :func:`add_intake_names_args`
is wired into the top-level parser in :mod:`tau2.cli`.

``intake-names`` owns the census/SSA person-name bank and the pronunciation
columns: ``extract`` turns the raw public datasets (SSA, Census, CMUdict,
WikiPron, wordlist) into the checked-in provenance-bearing extracts,
``build`` deterministically rebuilds ``person_names.yaml`` plus the
medication bank's pronunciation columns from them, ``packet`` renders
the owner review packet over the result (see
:mod:`tau2.domains.intake.tasks.name_banks`), and ``token-packet`` renders
the oddball-token pronunciation table for review (see
:mod:`tau2.domains.intake.tasks.token_pronunciations`). Phase 3 (design doc
§8b) adds the foreign-token pipeline for the properties/vehicles banks:
``extract-foreign`` / ``build-foreign`` / ``foreign-packet`` (see
:mod:`tau2.domains.intake.tasks.foreign_banks`).

``intake-tasks`` owns the v4 task generator (design doc §6): ``enumerate``
reports the frame, ``freeze`` regenerates the canonical 300-task set +
manifest byte-identically, ``draw`` writes an additive draw (e.g. a
multi-entity band) somewhere else for inspection (see
:mod:`tau2.domains.intake.tasks.generator`), ``complications-packet``
renders the seeded scripted-complication draw as a pre-run review packet
(see :mod:`tau2.domains.intake.complications`), and
``audition-pronunciations`` synthesizes the pronunciation-audition wavs
(see :mod:`tau2.domains.intake.tasks.audition`).
"""

import argparse
from datetime import date
from pathlib import Path


def add_intake_names_args(parser: argparse.ArgumentParser) -> None:
    """Attach the intake-names subcommands and handlers to ``parser``."""
    subparsers = parser.add_subparsers(dest="intake_names_command", required=True)

    extract = subparsers.add_parser(
        "extract",
        help="Turn the raw SSA + Census zips, the pinned pronunciation "
        "lexicons (CMUdict, WikiPron), and the pinned wordlist into the "
        "checked-in extracts under name_sources/ (records raw shas + "
        "retrieval provenance)",
    )
    extract.add_argument(
        "--ssa-zip",
        type=Path,
        required=True,
        help="Path to the downloaded SSA national baby-names names.zip",
    )
    extract.add_argument(
        "--census-zip",
        type=Path,
        required=True,
        help="Path to the downloaded Census 2010 surnames names.zip",
    )
    extract.add_argument(
        "--cmudict",
        type=Path,
        required=True,
        help="Path to the downloaded cmudict.dict (pinned commit: "
        "see CMUDICT_SOURCE_URL)",
    )
    extract.add_argument(
        "--wikipron",
        type=Path,
        required=True,
        help="Path to the downloaded eng_latn_us_broad.tsv (pinned commit: "
        "see WIKIPRON_SOURCE_URL)",
    )
    extract.add_argument(
        "--wordlist",
        type=Path,
        required=True,
        help="Path to the downloaded web2 wordlist (pinned commit: "
        "see WORDLIST_SOURCE_URL)",
    )
    extract.add_argument(
        "--retrieved",
        type=str,
        default=date.today().isoformat(),
        help="Retrieval date of the raw files, YYYY-MM-DD (default: today)",
    )
    extract.add_argument(
        "--ssa-mirror-url",
        type=str,
        default=None,
        help="URL actually fetched for the SSA zip when the canonical host "
        "was unreachable (e.g. a Wayback snapshot)",
    )
    extract.add_argument(
        "--census-mirror-url",
        type=str,
        default=None,
        help="URL actually fetched for the Census zip when the canonical "
        "host was unreachable",
    )
    extract.set_defaults(func=_run_names_extract)

    build = subparsers.add_parser(
        "build",
        help="Deterministically rebuild person_names.yaml (and the "
        "medications bank's pronunciation columns) from the checked-in "
        "extracts (byte-identical per seed)",
    )
    build.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Draw seed (default: the checked-in banks' seed)",
    )
    build.set_defaults(func=_run_names_build)

    packet = subparsers.add_parser(
        "packet",
        help="Render the pronunciation review packet over the checked-in "
        "banks + manifest: one row per pronunciation-bearing token "
        "(value, phonemes, respelling, mispronounced, operator, source)",
    )
    packet.add_argument(
        "-o",
        "--out",
        type=Path,
        required=True,
        help="Markdown file to write the packet to",
    )
    packet.set_defaults(func=_run_names_packet)

    token_packet = subparsers.add_parser(
        "token-packet",
        help="Render the oddball-token pronunciation review packet: the full "
        "authored token -> respelling table (mixed-case, initialism, and "
        "ampersand tokens from the invented-content banks) plus the "
        "letter-digit tokens ignored by owner decision",
    )
    token_packet.add_argument(
        "-o",
        "--out",
        type=Path,
        required=True,
        help="Markdown file to write the packet to",
    )
    token_packet.set_defaults(func=_run_token_packet)

    extract_foreign = subparsers.add_parser(
        "extract-foreign",
        help="Scan the properties/vehicles banks for foreign-token "
        "candidates and index them against the 16 pinned per-language "
        "WikiPron TSVs; writes the checked-in extracts under name_sources/ "
        "with per-lexicon provenance (design doc sec. 8b)",
    )
    extract_foreign.add_argument(
        "--cmudict",
        type=Path,
        required=True,
        help="Path to the downloaded cmudict.dict (pinned commit: "
        "see CMUDICT_SOURCE_URL)",
    )
    extract_foreign.add_argument(
        "--wordlist",
        type=Path,
        required=True,
        help="Path to the downloaded web2 wordlist (pinned commit: "
        "see WORDLIST_SOURCE_URL)",
    )
    extract_foreign.add_argument(
        "--lexicons-dir",
        type=Path,
        required=True,
        help="Directory holding the 16 per-language WikiPron TSVs at the "
        "pinned commit, named as in LEXICON_SOURCES (loanwords.py)",
    )
    extract_foreign.add_argument(
        "--retrieved",
        type=str,
        default=date.today().isoformat(),
        help="Retrieval date of the raw files, YYYY-MM-DD (default: today)",
    )
    extract_foreign.set_defaults(func=_run_extract_foreign)

    build_foreign = subparsers.add_parser(
        "build-foreign",
        help="Deterministically rebuild the properties/vehicles language "
        "pins + foreign-token pronunciation columns in place from the "
        "checked-in foreign extracts (byte-identical per seed)",
    )
    build_foreign.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Draw seed (default: the checked-in banks' seed)",
    )
    build_foreign.set_defaults(func=_run_build_foreign)

    foreign_packet = subparsers.add_parser(
        "foreign-packet",
        help="Render the foreign-token pronunciation review packet over the "
        "checked-in properties/vehicles banks + manifest: language pins, "
        "one row per column-bearing token (source, phonemes, respelling, "
        "mispronounced, operator, citation), tier-3 reclassifications, and "
        "the operator-applicability proposal",
    )
    foreign_packet.add_argument(
        "-o",
        "--out",
        type=Path,
        required=True,
        help="Markdown file to write the packet to",
    )
    foreign_packet.set_defaults(func=_run_foreign_packet)


def add_intake_tasks_args(parser: argparse.ArgumentParser) -> None:
    """Attach the intake-tasks subcommands and handlers to ``parser``."""
    subparsers = parser.add_subparsers(dest="intake_tasks_command", required=True)

    enumerate_parser = subparsers.add_parser(
        "enumerate",
        help="Report the task frame the checked-in banks span "
        "(contract-checked; nothing is written)",
    )
    enumerate_parser.set_defaults(func=_run_tasks_enumerate)

    freeze = subparsers.add_parser(
        "freeze",
        help="Regenerate the canonical atomic task set (tasks.json + "
        "split_tasks.json + tasks.manifest.json) byte-identically",
    )
    freeze.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Freeze seed (default: the canonical seed)",
    )
    freeze.set_defaults(func=_run_tasks_freeze)

    draw = subparsers.add_parser(
        "draw",
        help="Write an additive draw (e.g. a multi-entity band) to an "
        "explicit directory — never touches the canonical task set",
    )
    draw.add_argument(
        "--n-entities",
        type=int,
        required=True,
        choices=[1, 2, 3, 5],
        help="Missing fields per call",
    )
    draw.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Draw seed (default: the canonical seed)",
    )
    draw.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Directory to write the drawn task set + manifest into",
    )
    draw.set_defaults(func=_run_tasks_draw)

    compose = subparsers.add_parser(
        "compose",
        help="Write the paired composition bands from the canonical freeze: "
        "flat bundles (compose_n2/compose_n3) plus their staged chain twins "
        "(chain_n2/chain_n3) for the intake_staged domain — additive bands, "
        "never touches the canonical task set "
        "(docs/designs/intake-entity-composition.md)",
    )
    compose.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Compose seed (default: the canonical freeze seed)",
    )
    compose.add_argument(
        "--parent-dir",
        type=Path,
        default=None,
        help="Directory holding the canonical tasks.json + manifest to pair "
        "against (default: the domain data dir)",
    )
    compose.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Directory to write the four band directories + compose "
        "manifest into (default: the domain's bands/ dir, where runs load "
        "band splits from)",
    )
    compose.set_defaults(func=_run_tasks_compose)

    complications = subparsers.add_parser(
        "complications-packet",
        help="Render the seeded scripted-complication draw over the frozen "
        "task set as a markdown review packet (pre-verify every additive "
        "line before a complication-armed run)",
    )
    complications.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Run seed to draw with (the packet pre-verifies exactly the "
        "lines a run at this --seed will use)",
    )
    complications.add_argument(
        "--profile",
        choices=("default", "hard"),
        default="default",
        help="Complication profile to draw with (default: default). "
        "'default' draws the literature-anchored per-kind rates over the "
        "full frozen set; 'hard' draws a complication on every call and "
        "renders hard-tier tasks only — exactly the tasks a hard run runs. "
        "Recorded in the packet's provenance header with the per-kind rates.",
    )
    complications.add_argument(
        "--rate",
        type=float,
        default=None,
        help="Explicit trigger-rate override in [0, 1]: uniformly scales the "
        "profile's per-kind rates, exactly like a run's "
        "--complication-rate (default: the profile's rates verbatim)",
    )
    complications.add_argument(
        "--channel",
        choices=("text", "voice"),
        default="voice",
        help="Run channel to draw for (default: voice — intake runs are voice "
        "runs). Voice-only kinds (mispronounced_term) never appear in a "
        "--channel text draw. Recorded in the packet's provenance header.",
    )
    complications.add_argument(
        "-o",
        "--out",
        type=Path,
        required=True,
        help="Markdown file to write the packet to",
    )
    complications.set_defaults(func=_run_complications_packet)

    audition = subparsers.add_parser(
        "audition-pronunciations",
        help="Synthesize pronunciation-audition wavs (design gate 2): "
        "gold (default TTS reading) / mispronounced pairs for every "
        "mispronounced-bearing bank entry, via the run's ElevenLabs path "
        "with a fixed stock voice; resumable (existing wavs kept), "
        "index.md + index.html grouped riskiest-first",
    )
    audition.add_argument(
        "-o",
        "--out",
        type=Path,
        required=True,
        help="Directory to write the wavs + index.md into",
    )
    audition.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of entries auditioned, in plan order (smoke "
        "the verb without paying for the full synthesis)",
    )
    audition.set_defaults(func=_run_audition)


def _run_names_extract(args: argparse.Namespace) -> None:
    from rich.console import Console

    from tau2.domains.intake.tasks.name_banks import extract_name_sources

    outcome = extract_name_sources(
        ssa_zip=args.ssa_zip,
        census_zip=args.census_zip,
        cmudict=args.cmudict,
        wikipron=args.wikipron,
        wordlist=args.wordlist,
        retrieved=args.retrieved,
        ssa_mirror_url=args.ssa_mirror_url,
        census_mirror_url=args.census_mirror_url,
    )
    console = Console()
    if outcome.written:
        console.print("written:")
        for path in outcome.written:
            console.print(f"  {path}")
    else:
        console.print(f"extracts in {outcome.sources_dir} already up to date")


def _run_names_build(args: argparse.Namespace) -> None:
    from rich.console import Console

    from tau2.domains.intake.tasks.name_banks import (
        DEFAULT_BUILD_SEED,
        build_name_banks,
    )

    seed = args.seed if args.seed is not None else DEFAULT_BUILD_SEED
    outcome = build_name_banks(seed=seed)
    console = Console()
    console.print(f"seed {outcome.seed}")
    console.print(f"sample easy: {', '.join(outcome.sample_easy)}")
    console.print(f"sample hard: {', '.join(outcome.sample_hard)}")
    stats = outcome.pool_stats
    console.print(
        f"pools: hard given {stats.hard_given_admissible}/"
        f"{stats.hard_given_band} admissible, tail surname "
        f"{stats.tail_surname_admissible}/{stats.tail_surname_band} admissible"
    )
    console.print(
        "pronunciation sources: "
        + ", ".join(
            f"{source} {count}"
            for source, count in sorted(outcome.pronunciation_sources.items())
        )
    )
    for bank, operators in sorted(outcome.operator_counts.items()):
        console.print(
            f"operators {bank}: "
            + ", ".join(f"{operator} {count}" for operator, count in operators.items())
        )
    if outcome.written:
        console.print("written:")
        for path in outcome.written:
            console.print(f"  {path}")
        console.print(
            "NOTE: the person-name banks feed the Phase C generator; re-run "
            "it to refresh tasks.json. Localized extensions can consume the "
            "manifest's given_name_genders map."
        )
    else:
        console.print(f"banks in {outcome.banks_dir} already up to date")


def _run_names_packet(args: argparse.Namespace) -> None:
    from rich.console import Console

    from tau2.domains.intake.tasks.name_banks import build_name_bank_packet

    packet = build_name_bank_packet()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(packet.markdown)
    console = Console()
    console.print(
        f"person tokens {packet.person_tokens}  "
        f"medication tokens {packet.medication_tokens}"
    )
    for source, count in sorted(packet.per_source.items()):
        console.print(f"  source {source}: {count}")
    for bank, operators in sorted(packet.per_operator.items()):
        for operator, count in sorted(operators.items()):
            console.print(f"  {bank} operator {operator}: {count}")
    console.print(f"written: {args.out}")


def _run_extract_foreign(args: argparse.Namespace) -> None:
    from rich.console import Console

    from tau2.domains.intake.tasks.foreign_banks import extract_foreign_sources

    outcome = extract_foreign_sources(
        cmudict=args.cmudict,
        wordlist=args.wordlist,
        lexicons_dir=args.lexicons_dir,
        retrieved=args.retrieved,
    )
    console = Console()
    console.print(
        "candidates: "
        + ", ".join(f"{bank} {count}" for bank, count in outcome.candidates.items())
        + f"  lexicon rows {outcome.lexicon_rows}"
    )
    if outcome.written:
        console.print("written:")
        for path in outcome.written:
            console.print(f"  {path}")
    else:
        console.print(f"extracts in {outcome.sources_dir} already up to date")


def _run_build_foreign(args: argparse.Namespace) -> None:
    from rich.console import Console

    from tau2.domains.intake.tasks.foreign_banks import build_foreign_banks
    from tau2.domains.intake.tasks.name_banks import DEFAULT_BUILD_SEED

    seed = args.seed if args.seed is not None else DEFAULT_BUILD_SEED
    outcome = build_foreign_banks(seed=seed)
    console = Console()
    console.print(f"seed {outcome.seed}")
    console.print(
        "pronunciation sources: "
        + ", ".join(
            f"{source} {count}"
            for source, count in sorted(outcome.source_counts.items())
            if count
        )
    )
    for bank, operators in sorted(outcome.operator_counts.items()):
        console.print(
            f"operators {bank}: "
            + ", ".join(f"{operator} {count}" for operator, count in operators.items())
        )
    if outcome.written:
        console.print("written:")
        for path in outcome.written:
            console.print(f"  {path}")
    else:
        console.print(f"banks in {outcome.banks_dir} already up to date")


def _run_foreign_packet(args: argparse.Namespace) -> None:
    from rich.console import Console

    from tau2.domains.intake.tasks.foreign_banks import build_foreign_packet

    packet = build_foreign_packet()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(packet.markdown)
    console = Console()
    console.print(f"column-bearing foreign tokens {packet.tokens}")
    for source, count in sorted(packet.per_source.items()):
        console.print(f"  source {source}: {count}")
    for language, count in sorted(packet.per_language.items()):
        console.print(f"  language {language}: {count}")
    for bank, operators in sorted(packet.per_operator.items()):
        for operator, count in sorted(operators.items()):
            console.print(f"  {bank} operator {operator}: {count}")
    console.print(f"written: {args.out}")


def _run_token_packet(args: argparse.Namespace) -> None:
    from rich.console import Console

    from tau2.domains.intake.tasks.token_pronunciations import build_token_packet

    packet = build_token_packet()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(packet.markdown)
    console = Console()
    console.print(f"tokens {packet.total}  ignored letter-digit {packet.ignored}")
    for token_class, count in packet.per_class.items():
        console.print(f"  {token_class}: {count}")
    console.print(f"written: {args.out}")


def _run_audition(args: argparse.Namespace) -> None:
    from rich.console import Console

    from tau2.domains.intake.tasks.audition import run_audition

    outcome = run_audition(args.out, limit=args.limit)
    console = Console()
    console.print(
        f"items {outcome.items}  synthesized {outcome.synthesized}  "
        f"skipped {outcome.skipped}"
    )
    console.print(f"index: {outcome.index_path}")


def _run_tasks_enumerate(args: argparse.Namespace) -> None:
    from rich.console import Console

    from tau2.domains.intake.tasks.generator import enumerate_frame

    frame = enumerate_frame()
    console = Console()
    console.print(
        f"frame: {frame.banks} banks x {frame.tiers} tiers x "
        f"{frame.values_per_cell} values = {frame.frame_size} drawable "
        "atomic tasks"
    )
    console.print(
        f"canonical draw: {frame.draw_per_cell} per cell = {frame.canonical_size} tasks"
    )


def _print_freeze_outcome(outcome) -> None:
    from rich.console import Console

    console = Console()
    console.print(
        f"seed {outcome.seed}  n_entities {outcome.n_entities}  "
        f"tasks {outcome.task_count}  frame {outcome.frame_size}"
    )
    if outcome.written:
        console.print("written:")
        for path in outcome.written:
            console.print(f"  {path}")
    else:
        console.print(f"task set in {outcome.out_dir} already up to date")


def _run_tasks_freeze(args: argparse.Namespace) -> None:
    from tau2.domains.intake.tasks.generator import (
        DEFAULT_FREEZE_SEED,
        freeze_tasks,
    )

    seed = args.seed if args.seed is not None else DEFAULT_FREEZE_SEED
    _print_freeze_outcome(freeze_tasks(seed=seed, n_entities=1))


def _run_complications_packet(args: argparse.Namespace) -> None:
    from rich.console import Console

    from tau2.data_model.simulation import ComplicationProfile
    from tau2.domains.intake.complications import build_complications_packet

    packet = build_complications_packet(
        seed=args.seed,
        profile=ComplicationProfile(args.profile),
        rate_override=args.rate,
        channel=args.channel,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(packet.markdown)
    console = Console()
    override = "none" if packet.rate_override is None else packet.rate_override
    console.print(
        f"seed {packet.seed}  profile {packet.profile.value}  "
        f"rate override {override}  channel {packet.channel}  "
        f"tasks {packet.total_tasks} (easy {packet.tier_mix['easy']} / "
        f"hard {packet.tier_mix['hard']})  "
        f"triggered {packet.triggered}/{packet.total_tasks}"
    )
    for kind, count in packet.per_kind.items():
        rate = packet.per_kind_rates.get(kind)
        suffix = "" if rate is None else f" (rate {rate:.4f})"
        console.print(f"  {kind}: {count}{suffix}")
    for label, rate in packet.per_kind_rates.items():
        if "(" in label:  # per-bank mispronounced_term rows
            console.print(f"  {label}: rate {rate:.4f}")
    console.print(f"written: {args.out}")


def _run_tasks_draw(args: argparse.Namespace) -> None:
    from tau2.domains.intake.tasks.generator import (
        DEFAULT_FREEZE_SEED,
        freeze_tasks,
    )

    seed = args.seed if args.seed is not None else DEFAULT_FREEZE_SEED
    _print_freeze_outcome(
        freeze_tasks(seed=seed, n_entities=args.n_entities, out_dir=args.out)
    )


def _run_tasks_compose(args: argparse.Namespace) -> None:
    from rich.console import Console

    from tau2.domains.intake.tasks.compose import freeze_compose
    from tau2.domains.intake.tasks.generator import DEFAULT_FREEZE_SEED

    seed = args.seed if args.seed is not None else DEFAULT_FREEZE_SEED
    outcome = freeze_compose(seed=seed, out_dir=args.out, parent_dir=args.parent_dir)
    console = Console()
    console.print(f"seed: {outcome.seed}")
    for band, count in outcome.band_task_counts.items():
        console.print(f"{band}: {count} tasks")
    console.print(f"reused slots: {outcome.reused_slots}")
    console.print(f"triggered-parent slots: {outcome.triggered_parent_slots}")
    console.print(f"files written/updated: {len(outcome.written)}")
    console.print(f"out: {outcome.out_dir}")
