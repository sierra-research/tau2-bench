# Copyright Sierra
"""The ``tau2 factory`` subcommand surface.

Each stage's implementation lives in its own module under
``tau2.multilingual.factory``.
"""

import argparse
import sys

from tau2.config import DEFAULT_MULTILINGUAL_DOMAIN


def _add_llm_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        default=None,
        help="LLM model override (default: tau2.config.DEFAULT_FACTORY_MODEL)",
    )
    parser.add_argument(
        "--reasoning",
        default=None,
        help="Reasoning effort override (default: the stage's "
        "DEFAULT_FACTORY_* constant in tau2.config)",
    )


def add_factory_args(parser: argparse.ArgumentParser) -> None:
    """Attach the factory sub-subcommands to the ``tau2 factory`` parser."""
    factory_sub = parser.add_subparsers(
        dest="factory_command", help="Factory commands", required=True
    )

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--lang", required=True, help="ISO 639-1 language code, e.g. 'sw'"
    )

    validate_parser = factory_sub.add_parser(
        "validate",
        parents=[common],
        help="Run the deterministic guardrails over a pack (draft or final)",
    )
    validate_parser.add_argument(
        "--draft",
        action="store_true",
        help="Validate the factory pack_draft.yaml under "
        "data/tau2/multilingual/_factory/<lang>/ instead of the final "
        "data/tau2/multilingual/<lang>/pack.yaml",
    )
    validate_parser.set_defaults(func=run_factory_validate)

    localize_entities_parser = factory_sub.add_parser(
        "localize-entities",
        parents=[common],
        help="Swap callers to locale identities (name/user_id/email/address) "
        "and emit the identity-variant '<domain>_tasks_<lang>_identity' task set",
    )
    localize_entities_parser.add_argument(
        "--domain",
        default=DEFAULT_MULTILINGUAL_DOMAIN,
        help="Domain whose tasks to identity-swap (default: %(default)s; must "
        "have a profile with identity_swap_supported)",
    )
    localize_entities_parser.add_argument(
        "--script-code",
        default=None,
        help="ISO 15924 script of the localized prose (e.g. 'deva'); inferred "
        "from the registered pack when omitted",
    )
    localize_entities_parser.add_argument(
        "--native-script",
        action="store_true",
        help="Build the NATIVE-SCRIPT DB ablation variant instead "
        "('<domain>_tasks_<lang>_identity_native'): the same deterministic "
        "identity draw with the name fields kept in the locale's native "
        "script (Devanagari/Hanzi) plus per-identity spell-out payloads. "
        "Requires the corpus's curated native name tables.",
    )
    localize_entities_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and validate without writing any artifacts",
    )
    localize_entities_parser.set_defaults(func=run_factory_localize_entities)

    draft_parser = factory_sub.add_parser(
        "draft",
        parents=[common],
        help="LLM draft of pack_draft.yaml + guidelines from an author form "
        "(see docs/multilingual/factory_form_example.md)",
    )
    draft_parser.add_argument(
        "--form",
        required=True,
        help="Path to the author form (YAML, or markdown with a ```yaml block)",
    )
    draft_parser.add_argument(
        "--max-repair-rounds",
        type=int,
        default=1,
        help="Max guardrail auto-repair rounds after the initial draft "
        "(default 1; raise to let the model iterate to a clean draft)",
    )
    draft_parser.add_argument(
        "--domain",
        default=None,
        help="Experiment domain for the deterministic experiment block "
        "(default: telecom, or the form's typed `domain` field)",
    )
    draft_parser.add_argument(
        "--smoke-task-stem",
        default=None,
        help="Smoke task id stem for the experiments entry (default: 3)",
    )
    _add_llm_overrides(draft_parser)
    draft_parser.set_defaults(func=run_factory_draft)

    draft_localization_parser = factory_sub.add_parser(
        "draft-localization",
        parents=[common],
        help="LLM-draft the pack's `localization:` block (native symbol "
        "readouts, date/numeral consistency examples, speech conventions + "
        "worked spoken-value examples, guideline example palettes, "
        "domain-term glossary) into the SHIPPED pack.yaml. A missing block "
        "is appended; a block that predates the speech-convention fields or "
        "the guideline example palettes is upgraded in place (reviewed "
        "fields preserved verbatim); a complete block is kept unless --force",
    )
    draft_localization_parser.add_argument(
        "--domain",
        default=DEFAULT_MULTILINGUAL_DOMAIN,
        help="Benchmark domain whose glossary to draft (a pack carries one "
        "glossary per domain under `domain_glossaries`; other domains' "
        "glossaries are always preserved). Default: %(default)s.",
    )
    draft_localization_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-draft the requested --domain's glossary even when it is "
        "already complete. Scoped like the verb: ONLY that glossary is "
        "re-rolled — other domains' glossaries and every non-glossary field "
        "(symbol readouts, date/number examples, speech conventions, "
        "guideline palettes) keep their exact reviewed values. To re-roll a "
        "whole language's block, delete its `localization:` key and re-run",
    )
    _add_llm_overrides(draft_localization_parser)
    draft_localization_parser.set_defaults(func=run_factory_draft_localization)

    draft_nativeness_parser = factory_sub.add_parser(
        "draft-nativeness",
        parents=[common],
        help="Select closed-catalog nativeness factors and author ONLY their "
        "language-specific agent-speech rules/examples into a review packet. "
        "Does not modify pack_draft.yaml; review and approve the packet before "
        "running apply-nativeness",
    )
    _add_llm_overrides(draft_nativeness_parser)
    draft_nativeness_parser.set_defaults(func=run_factory_draft_nativeness)

    apply_nativeness_parser = factory_sub.add_parser(
        "apply-nativeness",
        parents=[common],
        help="Apply an approved _factory/<lang>/nativeness_review.yaml to "
        "pack_draft.yaml. Refuses pending/stale packets and never overwrites "
        "existing nativeness content",
    )
    apply_nativeness_parser.set_defaults(func=run_factory_apply_nativeness)

    add_experiment_parser = factory_sub.add_parser(
        "add-experiment",
        parents=[common],
        help="Append the deterministic `experiments` entry for one domain to "
        "the SHIPPED pack.yaml (idempotent: a pack that already has the "
        "domain is skipped). The rest of the file stays byte-identical; the "
        "merged pack is re-validated through the full guardrails.",
    )
    add_experiment_parser.add_argument(
        "--domain",
        default=DEFAULT_MULTILINGUAL_DOMAIN,
        help="Benchmark domain of the new experiments entry (must have a "
        "profile in tau2.multilingual.domain_profiles); default: %(default)s",
    )
    add_experiment_parser.add_argument(
        "--smoke-task-stem",
        default=None,
        help="Task id stem the smoke stage runs (smoke id = "
        "'<stem>_<suffix>'). Defaults to the domain profile's stem.",
    )
    add_experiment_parser.set_defaults(func=run_factory_add_experiment)

    seed_tasks_parser = factory_sub.add_parser(
        "seed-tasks",
        help="Materialize a GENERATED-POOL domain's multilingual seed split "
        "(the profile's source task file + provenance manifest) and emit the "
        "per-language '<domain>_tasks_<lang>' arm files (English prose, ids "
        "suffixed) + the English caller-gender sidecar. Curated-source "
        "domains use `arm-tasks`. Deterministic and idempotent.",
    )
    seed_tasks_parser.add_argument(
        "--domain",
        default=DEFAULT_MULTILINGUAL_DOMAIN,
        help="Domain to seed (must have a profile with a dedicated "
        "multilingual source file); default: %(default)s",
    )
    seed_tasks_parser.add_argument(
        "--split",
        default="base",
        help="Split of the domain's split_tasks.json to materialize (default 'base')",
    )
    seed_tasks_parser.add_argument(
        "--lang",
        action="append",
        default=None,
        help="Language(s) to emit an arm file for (repeatable; default 'en')",
    )
    seed_tasks_parser.set_defaults(func=run_factory_seed_tasks)

    arm_tasks_parser = factory_sub.add_parser(
        "arm-tasks",
        help="Emit a CURATED-source domain's per-language "
        "'<domain>_tasks_<lang>' arm files (English source prose, ids suffixed "
        "'_<lang>') + the English caller-gender sidecar. The complement of "
        "`seed-tasks`: the source set is the reviewed, checked-in tasks.json, "
        "so nothing is materialized. Deterministic and idempotent.",
    )
    arm_tasks_parser.add_argument(
        "--domain",
        default=DEFAULT_MULTILINGUAL_DOMAIN,
        help="Domain to emit arms for (must have a profile whose multilingual "
        "source is the curated tasks.json); default: %(default)s",
    )
    arm_tasks_parser.add_argument(
        "--lang",
        action="append",
        default=None,
        help="Language(s) to emit an arm file for (repeatable; default 'en')",
    )
    arm_tasks_parser.set_defaults(func=run_factory_arm_tasks)

    draft_continuers_parser = factory_sub.add_parser(
        "draft-continuers",
        parents=[common],
        help="Give a shipped pack's personas a PURE backchannel continuer "
        "(the 'mm-hmm' equivalent), selected from the pack's own authored "
        "material where one exists",
    )
    draft_continuers_parser.add_argument(
        "--force",
        action="store_true",
        help="Redraft a pack that already carries the continuer marker",
    )
    _add_llm_overrides(draft_continuers_parser)
    draft_continuers_parser.set_defaults(func=run_factory_draft_continuers)

    finalize_parser = factory_sub.add_parser(
        "finalize",
        parents=[common],
        help="Promote a guardrail-clean draft into data/tau2/multilingual/<lang>/",
    )
    finalize_parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing final pack.yaml",
    )
    finalize_parser.set_defaults(func=run_factory_finalize)

    generate_assets_parser = factory_sub.add_parser(
        "generate-assets",
        parents=[common],
        help="Design + auto-pin an ElevenLabs voice for every persona (voice "
        "generation only; locale acoustic bed production is retired) and "
        "re-check the pack",
    )
    generate_assets_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-design voices even for personas that already have a voice_id",
    )
    generate_assets_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be generated without calling the ElevenLabs API",
    )
    generate_assets_parser.set_defaults(func=run_factory_generate_assets)

    voice_samples_parser = factory_sub.add_parser(
        "voice-samples",
        parents=[common],
        help="Render audition MP3s for a pack's pinned persona voices (plus "
        "optional explicit voice_ids) into a folder for human codec audition",
    )
    voice_samples_parser.add_argument(
        "--out",
        required=True,
        help="Output directory for the rendered MP3s",
    )
    voice_samples_parser.add_argument(
        "--label",
        default="pinned",
        help="Filename label for the pack persona renders (default: pinned)",
    )
    voice_samples_parser.add_argument(
        "--persona",
        action="append",
        default=None,
        help="Restrict pack renders to this persona id (repeatable; "
        "default: all personas)",
    )
    voice_samples_parser.add_argument(
        "--extra",
        action="append",
        default=None,
        metavar="LABEL:GENDER:VOICE_ID",
        help="Also render this explicit ElevenLabs voice with the language's "
        "gender-matched audition script (repeatable)",
    )
    voice_samples_parser.add_argument(
        "--english",
        action="store_true",
        help="Render the English audition script instead of the native one — "
        "for judging delivery (pace, clarity, distance) without a language barrier",
    )
    voice_samples_parser.set_defaults(func=run_factory_voice_samples)

    delete_voices_parser = factory_sub.add_parser(
        "delete-voices",
        help="Delete rejected ElevenLabs voices (audition losers, superseded "
        "redesigns); refuses any voice_id pinned by a language pack",
    )
    delete_voices_parser.add_argument(
        "--voice-id",
        action="append",
        required=True,
        help="ElevenLabs voice id to delete (repeatable)",
    )
    delete_voices_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be deleted without calling the ElevenLabs API",
    )
    delete_voices_parser.set_defaults(func=run_factory_delete_voices)

    autoform_parser = factory_sub.add_parser(
        "autoform",
        parents=[common],
        help="LLM-generate an author form (the seed for `draft`) from a "
        "language identity, so no human writes the form",
    )
    autoform_parser.add_argument(
        "--display-name",
        required=True,
        help="Human-readable language name, e.g. 'Romanian'",
    )
    autoform_parser.add_argument(
        "--script",
        required=True,
        help="ISO 15924 script code, e.g. 'latn' (must be in "
        "tau2.multilingual.invariants.SCRIPT_RANGES)",
    )
    autoform_parser.add_argument(
        "--hints",
        default="",
        help="Optional freeform hints for the author-form generator",
    )
    autoform_parser.add_argument(
        "-o",
        "--out",
        default=None,
        help="Where to write the form (default: the factory workspace "
        "_factory/<lang>/autoform.yaml)",
    )
    _add_llm_overrides(autoform_parser)
    autoform_parser.set_defaults(func=run_factory_autoform)

    smoke_assets_parser = factory_sub.add_parser(
        "smoke-assets",
        help="Cheap ElevenLabs API smoke: one Voice Design + one Sound Effects "
        "call, PASS/FAIL per endpoint (run before generate-assets)",
    )
    smoke_assets_parser.set_defaults(func=run_factory_smoke_assets)

    noise_bank_parser = factory_sub.add_parser(
        "text-noise-bank",
        parents=[common],
        help="Render the noisy-text corrupted stimulus bank for one "
        "language's task set (pre-run human review; same code path as the "
        "runtime injection)",
    )
    noise_bank_parser.add_argument(
        "--domain",
        default=DEFAULT_MULTILINGUAL_DOMAIN,
        help=f"Benchmark domain (default: {DEFAULT_MULTILINGUAL_DOMAIN})",
    )
    noise_bank_parser.add_argument(
        "--task-set",
        default=None,
        help="Registered task set to scan (default: <domain>_<lang>)",
    )
    noise_bank_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Noise seed (default: tau2.config.DEFAULT_TEXT_NOISE_SEED). "
        "Must match the run's --text-noise-seed for the bank to be what runs.",
    )
    noise_bank_parser.add_argument(
        "--operators",
        default=None,
        help="Optional comma-separated RESTRICTION of the language's closed "
        "operator set (catalog ids in tau2.multilingual.text_noise)",
    )
    noise_bank_parser.add_argument(
        "--out",
        default=None,
        help="Output JSON path (default: data/annotation_packets/"
        "text_noise_banks/<task_set>_seed<seed>.json)",
    )
    noise_bank_parser.set_defaults(func=run_factory_text_noise_bank)

    subset_run_parser = factory_sub.add_parser(
        "subset-run",
        help="Deterministically draw a seeded subset of an existing dir-format "
        "run directory into a complete, self-consistent new run directory "
        "(pruned simulation_index, the drawn simulations/<id>.json files, and "
        "ONLY the drawn sims' artifacts/task_<task_id>/sim_<sim_id>/ "
        "subtrees). Copy-only — the source is never modified. Provenance "
        "(source, seed, count, draw rule, git sha) is stamped inside the "
        "output results.json info.subset_provenance",
    )
    subset_run_parser.add_argument(
        "--results",
        required=True,
        help="Source run directory (contains results.json + simulations/ + artifacts/)",
    )
    subset_run_parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Draw seed (rule 'run-subset-draw-v1': sort the index by "
        "(task_id, trial, id), then random.Random(seed).sample)",
    )
    subset_run_parser.add_argument(
        "--count",
        type=int,
        required=True,
        help="Number of sims to draw (must not exceed the source index size)",
    )
    subset_run_parser.add_argument(
        "--out",
        required=True,
        help="Output run directory (must not already exist; no silent overwrite)",
    )
    subset_run_parser.set_defaults(func=run_factory_subset_run)


def run_factory_subset_run(args) -> None:
    from pathlib import Path

    from tau2.multilingual.factory.author_form import FactoryDraftError
    from tau2.multilingual.factory.run_subset import build_run_subset

    try:
        outcome = build_run_subset(
            Path(args.results), Path(args.out), seed=args.seed, count=args.count
        )
    except FactoryDraftError as exc:
        print(f"FAILED — {exc}")
        sys.exit(1)

    provenance = outcome.provenance
    print(
        f"Subset run written: {len(outcome.drawn)}/"
        f"{provenance.source_index_size} sims "
        f"(rule {provenance.draw_rule}, seed {provenance.seed}):"
    )
    print(f"  source: {provenance.source}")
    print(f"  out:    {outcome.out_dir}")
    print(
        f"  copied: {len(outcome.drawn)} sim file(s), "
        f"{outcome.artifact_dirs_copied} artifact subtree(s)"
    )
    tasks = sorted({str(entry.task_id) for entry in outcome.drawn})
    shown = ", ".join(tasks[:8]) + (", ..." if len(tasks) > 8 else "")
    print(f"  tasks:  {len(tasks)} distinct ({shown})")


def run_factory_validate(args) -> None:
    from tau2.multilingual.factory.guardrails import (
        validate_final_pack,
        validate_pack_draft,
    )
    from tau2.multilingual.factory.state import factory_project_dir

    if args.draft:
        target = factory_project_dir(args.lang)
        report = validate_pack_draft(target)
        label = f"draft pack for '{args.lang}' ({target})"
    else:
        report = validate_final_pack(args.lang)
        label = f"final pack for '{args.lang}'"

    _print_guardrail_notes(report)
    if report.ok:
        print(f"OK — {label} passes all guardrails.")
        return
    print(f"FAILED — {len(report.problems)} problem(s) in {label}:\n")
    for problem in report.problems:
        print(f"  - {problem}")
    sys.exit(1)


def _print_guardrail_notes(report) -> None:
    """Surface non-fatal guardrail notes (e.g. locale beds pending generate-assets)."""
    if getattr(report, "notes", None):
        print(f"NOTE — {len(report.notes)} non-fatal note(s):")
        for note in report.notes:
            print(f"  - {note}")
        print()


def run_factory_draft_continuers(args) -> None:
    from tau2.multilingual.factory.author_form import FactoryDraftError
    from tau2.multilingual.factory.continuer_draft import run_draft_continuers

    try:
        pack_path, original = pack_snapshot(args.lang)
        outcome = run_draft_continuers(
            args.lang,
            force=args.force,
            model=args.model,
            reasoning_effort=args.reasoning,
        )
    except FactoryDraftError as exc:
        print(f"FAILED — {exc}")
        sys.exit(1)

    if not outcome.written:
        print(
            f"SKIPPED — pack '{args.lang}' continuers already drafted "
            "(--force to redo)."
        )
        return
    print(
        f"Drafted backchannel continuers for '{args.lang}' — prompt "
        f"{outcome.prompt_version} sha256:{outcome.prompt_sha256[:12]}, "
        f"model {outcome.model}:"
    )
    for persona_id, entry in outcome.personas.items():
        print(f"  {persona_id}: {entry.phrases} ({entry.provenance})")
        if entry.note:
            print(f"    {entry.note}")
    print(f"  pack: {outcome.pack_path}")
    print(
        "  NOTE — add the result to REVIEWED_CONTINUERS in "
        "tests/test_multilingual/test_pack_invariants.py once a native "
        "reviewer has signed it off, so it cannot drift."
    )
    validate_pack_write(args.lang, pack_path, original)


def pack_snapshot(lang: str):
    from tau2.multilingual.factory.backfill_lib import load_shipped_pack

    pack = load_shipped_pack(lang)
    return pack.pack_path, pack.original


def validate_pack_write(lang: str, pack_path, original: str) -> None:
    """Run full guardrails and restore a rejected pack mutation.

    Only problems the write INTRODUCED reject it. A shipped pack can carry a
    known, documented problem it is waiting on its native reviewer to
    resolve; holding an unrelated single-field verb hostage to that would
    make the pack un-editable without making the rail any stronger for the
    field actually being written. The pre-existing problems are reported
    alongside the success so they stay visible.
    """
    from tau2.multilingual.factory.guardrails import validate_final_pack

    try:
        report = validate_final_pack(lang)
    except BaseException:
        pack_path.write_text(original)
        print("FAILED — validation raised; pack rolled back to its pre-write text.")
        raise
    _print_guardrail_notes(report)
    if report.ok:
        print("OK — validate_final_pack passes.")
        return

    # Re-run against the pre-write text to separate what this write broke
    # from what it inherited.
    written = pack_path.read_text()
    pack_path.write_text(original)
    try:
        baseline = validate_final_pack(lang)
    except BaseException:
        print(
            "FAILED — the pre-write pack does not even validate; rolled back "
            "to it anyway."
        )
        raise
    inherited = set(baseline.problems)
    introduced = [p for p in report.problems if p not in inherited]

    if not introduced:
        pack_path.write_text(written)
        print(
            f"OK — validate_final_pack reports no NEW problems. "
            f"{len(report.problems)} pre-existing problem(s) remain, "
            "untouched by this write:\n"
        )
        for problem in report.problems:
            print(f"  - (pre-existing) {problem}")
        return

    print(
        f"FAILED — {len(introduced)} guardrail problem(s) introduced by this "
        "write; the pack was rolled back to its pre-write text:\n"
    )
    for problem in introduced:
        print(f"  - {problem}")
    sys.exit(1)


def backfill_action(outcome) -> str:
    """What the backfill did, for the console line.

    ``replaced`` (a --force re-roll of already-reviewed content) is reported
    ahead of ``upgraded`` (gap-filling), because a forced run merges into the
    existing block too — both flags are set and the re-roll is the news.
    """
    return (
        "Re-drafted"
        if outcome.replaced
        else "Upgraded"
        if outcome.upgraded
        else "Wrote"
    )


def run_factory_localize_entities(args) -> None:
    from tau2.multilingual.factory.entity_localization import (
        gender_sidecar_path,
        identity_manifest_path,
        identity_map_path,
        identity_task_set_path,
        localize_entities,
        native_identity_manifest_path,
        native_identity_map_path,
        native_identity_task_set_path,
    )

    native_script = bool(getattr(args, "native_script", False))
    try:
        identity_tasks, problems, identity_map = localize_entities(
            args.lang,
            args.domain,
            script_code=args.script_code,
            write=not args.dry_run,
            native_script=native_script,
        )
    except (FileNotFoundError, ValueError, KeyError) as exc:
        # Missing corpus / arm file, invalid corpus, unknown domain: expected
        # operator errors — the sibling verbs' FAILED shape, not a traceback.
        print(f"FAILED — {exc}")
        sys.exit(1)
    identity_count = len(identity_map)
    variant_label = "identity_native" if native_script else "identity"
    print(
        f"{args.lang}/{args.domain}: {identity_count} caller identities, "
        f"{len(identity_tasks)} {variant_label}-variant tasks derived"
    )
    if problems:
        print(f"NOT WRITTEN — {len(problems)} problem(s):\n")
        for problem in problems:
            print(f"  - {problem}")
        sys.exit(1)
    if args.dry_run:
        print("OK — dry run, nothing written.")
        return
    if native_script:
        print(
            f"Wrote native identity map to "
            f"{native_identity_map_path(args.lang, args.domain)}"
        )
        print(
            "Wrote native identity-variant tasks to "
            f"{native_identity_task_set_path(args.lang, args.domain)}"
        )
        print(
            "Wrote native identity-localization provenance to "
            f"{native_identity_manifest_path(args.lang, args.domain)}"
        )
        print(
            "No gender sidecar written: *_identity_native ids resolve through "
            "the plain sidecar (caller_gender_for_task strips the _native tail)."
        )
        return
    print(f"Wrote identity map to {identity_map_path(args.lang, args.domain)}")
    print(
        f"Wrote identity-variant tasks to {identity_task_set_path(args.lang, args.domain)}"
    )
    print(
        "Wrote caller-gender sidecar (plain + _identity task ids) to "
        f"{gender_sidecar_path(args.lang, args.domain)}"
    )
    print(
        "Wrote identity-localization provenance to "
        f"{identity_manifest_path(args.lang, args.domain)}"
    )
    print(
        f"The '{args.domain}_{args.lang}_identity' task set registers "
        "automatically by filename convention."
    )


def run_factory_autoform(args) -> None:
    from pathlib import Path

    import yaml

    from tau2.multilingual.factory.author_form import (
        FactoryDraftError,
        generate_author_form,
    )
    from tau2.multilingual.factory.state import factory_file_path
    from tau2.multilingual.invariants import SCRIPT_RANGES

    if args.script not in SCRIPT_RANGES:
        print(
            f"FAILED — script '{args.script}' has no registered Unicode range. "
            f"Known scripts: {sorted(SCRIPT_RANGES)}. Adding a new script is a "
            "one-line engineer edit to tau2.multilingual.invariants.SCRIPT_RANGES."
        )
        sys.exit(1)

    try:
        form = generate_author_form(
            args.lang,
            args.display_name,
            args.script,
            hints=args.hints,
            model=args.model,
            reasoning_effort=args.reasoning,
        )
    except FactoryDraftError as exc:
        print(f"FAILED — {exc}")
        sys.exit(1)

    out_path = (
        Path(args.out) if args.out else factory_file_path(args.lang, "autoform.yaml")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        yaml.safe_dump(
            form.model_dump(mode="json"), sort_keys=False, allow_unicode=True
        )
    )

    print(f"Author form written: {out_path}")
    print(f"  personas: {', '.join(p.name for p in form.personas)}")
    print(
        "\nReview/edit the form, then draft the pack:\n"
        f"  tau2 factory draft --lang {args.lang} --form {out_path} "
        "--max-repair-rounds 4"
    )


def run_factory_draft(args) -> None:
    from pathlib import Path

    from tau2.multilingual.factory.author_form import FactoryDraftError
    from tau2.multilingual.factory.drafting import run_draft

    try:
        outcome = run_draft(
            args.lang,
            Path(args.form),
            model=args.model,
            reasoning_effort=args.reasoning,
            max_repair_rounds=args.max_repair_rounds,
            domain=getattr(args, "domain", None),
            smoke_task_stem=getattr(args, "smoke_task_stem", None),
        )
    except FactoryDraftError as exc:
        print(f"FAILED — {exc}")
        sys.exit(1)

    print(f"Draft written (pack revision {outcome.pack_revision}):")
    print(f"  pack:       {outcome.pack_draft_path}")
    print(f"  guidelines: {outcome.guidelines_draft_path}")
    if outcome.stripped_voice_ids:
        print(
            "NOTE — stripped model-emitted voice_id(s) (voice selection is a "
            f"human bottleneck; pin after a codec audition): "
            f"{outcome.stripped_voice_ids}"
        )
    _print_guardrail_notes(outcome.report)
    if outcome.report.ok:
        suffix = (
            f" (after {outcome.repair_rounds} auto-repair round(s))"
            if outcome.repair_rounds
            else ""
        )
        print(f"OK — the draft passes all guardrails{suffix}.")
        return
    print(
        f"FAILED — {len(outcome.report.problems)} problem(s) remain after "
        f"{outcome.repair_rounds} auto-repair round(s):\n"
    )
    for problem in outcome.report.problems:
        print(f"  - {problem}")
    print(
        f"\nIterate by editing {outcome.pack_draft_path} directly, then re-run "
        f"`tau2 factory validate --draft --lang {args.lang}`."
    )
    sys.exit(1)


def run_factory_draft_localization(args) -> None:
    from tau2.multilingual.factory.author_form import FactoryDraftError
    from tau2.multilingual.factory.localization_drafting import (
        run_backfill_localization,
    )

    try:
        pack_path, original = pack_snapshot(args.lang)
        outcome = run_backfill_localization(
            args.lang,
            domain=args.domain,
            force=args.force,
            model=args.model,
            reasoning_effort=args.reasoning,
        )
    except FactoryDraftError as exc:
        print(f"FAILED — {exc}")
        sys.exit(1)

    if not outcome.written:
        print(
            f"SKIPPED — pack '{args.lang}' already has a complete "
            "localization block (--force to re-draft)."
        )
        return
    print(
        f"{backfill_action(outcome)} localization block for '{args.lang}' "
        f"({outcome.symbols} symbol readouts, {outcome.glossary_terms} "
        f"glossary terms; prompt {outcome.prompt_version} "
        f"sha256:{outcome.prompt_sha256[:12]}, model {outcome.model}):"
    )
    if outcome.upgraded and outcome.upgraded_fields:
        verb = "re-drafted" if outcome.replaced else "added"
        print(
            f"  {verb}: {', '.join(outcome.upgraded_fields)} "
            "(every other previously-reviewed field preserved verbatim)"
        )
    print(f"  pack: {outcome.pack_path}")
    validate_pack_write(args.lang, pack_path, original)


def run_factory_draft_nativeness(args) -> None:
    from tau2.multilingual.factory.author_form import FactoryDraftError
    from tau2.multilingual.factory.nativeness_drafting import (
        draft_nativeness_review,
    )

    try:
        outcome = draft_nativeness_review(
            args.lang,
            model=args.model,
            reasoning_effort=args.reasoning,
        )
    except FactoryDraftError as exc:
        print(f"FAILED — {exc}")
        sys.exit(1)

    action = "Wrote" if outcome.written else "Reused"
    print(
        f"{action} nativeness review packet for '{args.lang}' "
        f"({len(outcome.artifact.judge_factors)} selected factor(s)):"
    )
    print(f"  packet: {outcome.review_path}")
    print(
        "  PAUSE — review every factor id, rule, and target-language example. "
        "Set `review_status: approved` in the packet, then run "
        f"`tau2 factory apply-nativeness --lang {args.lang}`."
    )


def run_factory_apply_nativeness(args) -> None:
    from tau2.multilingual.factory.author_form import FactoryDraftError
    from tau2.multilingual.factory.nativeness_drafting import (
        apply_nativeness_review,
    )

    try:
        outcome = apply_nativeness_review(args.lang)
    except FactoryDraftError as exc:
        print(f"FAILED — {exc}")
        sys.exit(1)

    action = "Applied" if outcome.written else "Already applied"
    print(
        f"{action} approved nativeness rubrics for '{args.lang}' "
        f"({', '.join(outcome.factor_ids)}):"
    )
    print(f"  draft:  {outcome.pack_draft_path}")
    print(f"  packet: {outcome.review_path}")
    print(f"Next: `tau2 factory finalize --lang {args.lang}`.")


def run_factory_add_experiment(args) -> None:
    from tau2.multilingual.factory.author_form import FactoryDraftError
    from tau2.multilingual.factory.pack_assembly import run_add_experiment

    try:
        outcome = run_add_experiment(
            args.lang,
            args.domain,
            smoke_task_stem=args.smoke_task_stem,
        )
    except FactoryDraftError as exc:
        print(f"FAILED — {exc}")
        sys.exit(1)

    if not outcome.written:
        print(
            f"SKIPPED — pack '{args.lang}' already has an experiments entry "
            f"for domain '{args.domain}' (preset '{outcome.preset_name}')."
        )
        return
    print(
        f"Added '{args.domain}' experiments entry for '{args.lang}' "
        f"(preset '{outcome.preset_name}', arm '{outcome.main_arm_name}'):"
    )
    print(f"  pack: {outcome.pack_path}")
    validate_pack_write(args.lang, outcome.pack_path, outcome.original)


def run_factory_seed_tasks(args) -> None:
    from tau2.multilingual.factory.author_form import FactoryDraftError
    from tau2.multilingual.factory.seed_tasks import run_seed_tasks

    try:
        outcome = run_seed_tasks(args.domain, split=args.split, langs=args.lang)
    except FactoryDraftError as exc:
        print(f"FAILED — {exc}")
        sys.exit(1)

    print(
        f"{outcome.domain}/{outcome.split}: {outcome.task_count} seed tasks "
        f"-> {outcome.source_path}"
    )
    print(f"  manifest: {outcome.manifest_path}")
    for lang, path in outcome.emitted.items():
        print(f"  {lang} arm: {path}")
    if outcome.written:
        print(f"Wrote {len(outcome.written)} file(s).")
    else:
        print("OK — everything already up to date, nothing written.")
    print("Arm files register automatically by filename convention.")


def run_factory_arm_tasks(args) -> None:
    from tau2.multilingual.factory.author_form import FactoryDraftError
    from tau2.multilingual.factory.task_arms import run_arm_tasks

    try:
        outcome = run_arm_tasks(args.domain, langs=args.lang)
    except FactoryDraftError as exc:
        print(f"FAILED — {exc}")
        sys.exit(1)

    print(
        f"{outcome.domain}: {outcome.task_count} curated source tasks "
        f"<- {outcome.source_path}"
    )
    for lang, path in outcome.emitted.items():
        print(f"  {lang} arm: {path}")
    if outcome.written:
        print(f"Wrote {len(outcome.written)} file(s).")
    else:
        print("OK — everything already up to date, nothing written.")
    print("Arm files register automatically by filename convention.")


def run_factory_finalize(args) -> None:
    from tau2.multilingual.factory.author_form import FactoryDraftError
    from tau2.multilingual.factory.finalize import finalize

    try:
        outcome = finalize(args.lang, force=args.force)
    except FactoryDraftError as exc:
        print(f"FAILED — {exc}")
        sys.exit(1)

    print(f"Finalized pack for '{args.lang}':")
    print(f"  pack:            {outcome.pack_path}")
    print(f"  guidelines:      {outcome.guidelines_path}")
    print(f"  text guidelines: {outcome.text_guidelines_path}")
    _print_guardrail_notes(outcome.report)
    if not outcome.report.ok:
        print(
            f"FAILED — the promoted pack fails {len(outcome.report.problems)} "
            "final guardrail(s):\n"
        )
        for problem in outcome.report.problems:
            print(f"  - {problem}")
        sys.exit(1)
    print("OK — validate_final_pack passes.\n")
    for line in outcome.checklist_lines:
        print(line)


def run_factory_delete_voices(args) -> None:
    from tau2.multilingual.factory.voice_cleanup import delete_voices

    results = delete_voices(args.voice_id, dry_run=args.dry_run)
    failed = 0
    for r in results:
        marker = "ok  " if r.status in ("deleted", "dry_run") else "FAIL"
        if r.status not in ("deleted", "dry_run"):
            failed += 1
        print(f"  {marker} {r.status:14s} {r.voice_id:22s} {r.detail}")
    if failed:
        print(f"{failed} voice(s) not deleted (pinned or error).")
        sys.exit(1)


def run_factory_voice_samples(args) -> None:
    from pathlib import Path

    from tau2.multilingual.factory.voice_samples import (
        ExtraVoice,
        render_voice_samples,
    )

    try:
        extras = [ExtraVoice.parse(spec) for spec in (args.extra or [])]
    except ValueError as e:
        print(str(e))
        sys.exit(2)

    results = render_voice_samples(
        args.lang,
        out_dir=Path(args.out),
        label=args.label,
        personas=args.persona,
        extras=extras,
        english=args.english,
    )

    print(f"Voice samples for '{args.lang}':")
    failed = 0
    for r in results:
        if r.error:
            failed += 1
            print(f"  FAIL {r.label:34s} {r.voice_id or '-':22s} {r.error}")
        else:
            print(f"  ok   {r.label:34s} {r.voice_id:22s} {r.path}")
    if failed:
        print(f"FAILED — {failed} sample(s) did not render.")
        sys.exit(1)


def run_factory_generate_assets(args) -> None:
    from tau2.multilingual.factory.asset_generation import generate_assets

    outcome = generate_assets(args.lang, force=args.force, dry_run=args.dry_run)

    label = "Would generate" if args.dry_run else "Generated assets"
    print(f"{label} for '{args.lang}':")
    for line in outcome.lines:
        print(line)
    if outcome.voice_rationale_path:
        print(f"  rationale: {outcome.voice_rationale_path}")

    if outcome.guardrail_ok:
        print("OK — final-pack guardrails pass (voices pinned, noise files resolve).")
        return
    print(f"FAILED — {len(outcome.guardrail_problems)} guardrail problem(s):\n")
    for problem in outcome.guardrail_problems:
        print(f"  - {problem}")
    sys.exit(1)


def run_factory_smoke_assets(args) -> None:
    from tau2.multilingual.factory.asset_smoke import run_asset_smoke

    outcome = run_asset_smoke()
    print(f"Smoke outputs: {outcome.out_dir}\n")
    for check in outcome.checks:
        print(f"  {'PASS' if check.passed else 'FAIL'} — {check.name}")
        if check.detail:
            print(f"         {check.detail}")
    if not outcome.ok:
        failed = [c.name for c in outcome.checks if not c.passed]
        print(f"\nFAILED — {len(failed)} endpoint(s): {', '.join(failed)}")
        sys.exit(1)
    print("\nOK — all endpoints reachable.")


def run_factory_text_noise_bank(args) -> None:
    from pathlib import Path

    from tau2.config import DEFAULT_TEXT_NOISE_SEED
    from tau2.data_model.simulation import TextNoiseSettings
    from tau2.multilingual.text_noise import build_text_noise_bank
    from tau2.utils import DATA_DIR

    task_set = args.task_set or f"{args.domain}_{args.lang}"
    operators = (
        [op for op in args.operators.split(",") if op] if args.operators else None
    )
    settings = TextNoiseSettings(
        seed=args.seed if args.seed is not None else DEFAULT_TEXT_NOISE_SEED,
        operators=operators,
    )
    bank = build_text_noise_bank(args.lang, args.domain, task_set, settings)

    out = (
        Path(args.out)
        if args.out
        else DATA_DIR
        / "annotation_packets"
        / "text_noise_banks"
        / f"{task_set}_seed{settings.seed}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(bank.model_dump_json(indent=2))

    manifest = bank.manifest
    print(
        f"Text-noise bank — {manifest.language} / {manifest.task_set_name} "
        f"(catalog {manifest.catalog_version}, seed {manifest.seed})"
    )
    print(f"Operators: {', '.join(manifest.operators)}")
    print(
        f"{manifest.num_tasks} task(s), {manifest.num_corrupted} corrupted "
        f"entit(ies), {manifest.num_skipped} skipped\n"
    )
    for plan in bank.plans:
        rows = [
            f"    {e.kind:<8} {e.clean!r} -> {e.corrupted!r}  [{e.operator}]"
            for e in plan.info.entities
        ] + [f"    {s.kind:<8} {s.clean!r} (clean — {s.reason})" for s in plan.skipped]
        if rows:
            print(f"  {plan.task_id}")
            for row in rows:
                print(row)
    print(f"\nWrote {out}")
