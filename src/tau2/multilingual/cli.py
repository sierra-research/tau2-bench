"""Language discovery and validation without authoring or network calls."""

import argparse
import json
from pathlib import Path

from tau2.multilingual.runtime import (
    language_root,
    list_languages,
    load_language,
    load_pack,
)


def add_language_run_args(parser: argparse.ArgumentParser) -> None:
    """Register language selection on the ordinary run command."""
    from tau2.data_model.semantic_communication import CommunicateScorer

    parser.add_argument(
        "--communicate-scorer",
        choices=[scorer.value for scorer in CommunicateScorer],
        default=None,
        help="Communicate-info scorer: the pinned semantic LLM judge (default) or "
        "the original substring matching (custom runs only).",
    )
    parser.add_argument(
        "--communicate-judge",
        type=json.loads,
        default=None,
        metavar="JSON",
        help='Semantic judge overrides, e.g. {"model": "...", "model_args": {...}}.',
    )
    parser.add_argument(
        "--no-localize-db",
        dest="localize_db",
        action="store_const",
        const=False,
        default=None,
        help="Keep canonical databases, tasks and policies while speaking the selected language; voice runs require --benchmark none.",
    )
    parser.add_argument(
        "--language",
        default=None,
        action="append",
        help="Language pack (default: English). Repeatable for released voice "
        "runs; omit there to run all released languages.",
    )


def communicate_scoring_from_args(args: argparse.Namespace):
    """The explicit communicate-info scorer a run command requests, if any."""
    from tau2.data_model.semantic_communication import (
        CommunicateScorer,
        CommunicateScoring,
        SemanticCommunicateSettings,
    )

    if args.communicate_scorer is None and args.communicate_judge is None:
        return None
    scorer = CommunicateScorer(args.communicate_scorer or CommunicateScorer.SEMANTIC)
    if scorer == CommunicateScorer.SUBSTRING:
        if args.communicate_judge is not None:
            raise ValueError("--communicate-judge applies to the semantic scorer")
        return CommunicateScoring(scorer=scorer)
    return CommunicateScoring(
        judge=SemanticCommunicateSettings.model_validate(args.communicate_judge or {})
    )


def add_languages_args(parser: argparse.ArgumentParser) -> None:
    """Register the package-owned language inspection commands."""
    sub = parser.add_subparsers(dest="language_command", required=True)
    listing = sub.add_parser("list", help="List published language directories")
    listing.set_defaults(func=run_languages_list)
    rendering = sub.add_parser(
        "render-prompts", help="Render actual voice prompts offline for review"
    )
    rendering.add_argument("--output", required=True)
    rendering.add_argument(
        "--language", action="append", help="Language filter; repeatable"
    )
    rendering.add_argument(
        "--domain", action="append", help="Domain filter; repeatable"
    )
    rendering.add_argument("--seed", type=int, default=None)
    rendering.add_argument(
        "--no-localize-db",
        dest="localize_db",
        action="store_const",
        const=False,
        default=None,
        help="Review canonical scenarios in the selected language; cannot use --benchmark.",
    )
    rendering.add_argument(
        "--benchmark",
        help="Render the released benchmark using the same resolver/defaults as tau2 run --audio-native",
    )
    rendering.add_argument(
        "--config",
        type=Path,
        help="Ordinary VoiceRunConfig JSON used for prompt construction",
    )
    rendering.set_defaults(func=run_render_prompts)
    for verb in ("show", "validate"):
        command = sub.add_parser(
            verb, help=f"{verb.title()} a published pack and all its bindings"
        )
        command.add_argument("language")
        command.set_defaults(func=run_language_inspect)


def run_languages_list(args: argparse.Namespace) -> None:
    """List only published pack filenames; does not register task globs."""
    print(json.dumps(list_languages()))


def run_render_prompts(args: argparse.Namespace) -> None:
    """Render selected published packs with the shared runtime prompt functions."""
    from pathlib import Path

    from tau2.data_model.simulation import VoiceRunConfig
    from tau2.multilingual.prompt_review import render_review, write_review

    template = (
        VoiceRunConfig.model_validate_json(args.config.read_text())
        if args.config
        else None
    )
    if args.localize_db is not None:
        from tau2.data_model.run_config import AudioNativeConfig

        template = template or VoiceRunConfig(audio_native_config=AudioNativeConfig())
        template = template.model_copy(update={"localize_db": args.localize_db})
    review = render_review(
        languages=args.language,
        domains=args.domain,
        seed=args.seed,
        template=template,
        benchmark=args.benchmark,
    )
    output = Path(args.output)
    write_review(review, output)
    print(
        json.dumps({"pairs": len(review.pairs), "review": str(output / "index.html")})
    )


def run_language_inspect(args: argparse.Namespace) -> None:
    """Validate an entire pack before displaying its declared content."""
    pack = load_pack(args.language)
    modes = ("text", "voice") if pack.voice is not None else ("text",)
    loaded = [
        load_language(args.language, domain, mode=mode)
        for domain in pack.domains
        for mode in modes
    ]
    if args.language_command == "show":
        print(pack.model_dump_json(indent=2))
    else:
        print(
            json.dumps(
                {
                    "language": pack.language,
                    "revision": pack.revision,
                    "pack": str(language_root() / pack.language / "pack.yaml"),
                    "modes": list(modes),
                    "domains": {
                        f"{entry.domain}:{entry.mode}": {
                            "tasks": len(entry.tasks),
                            **entry.artifact_pins.model_dump(mode="json"),
                        }
                        for entry in loaded
                    },
                },
                indent=2,
            )
        )
