"""Publish explicit custom-run domain bindings without changing release packs."""

import argparse
import hashlib
import json
from pathlib import Path

from pydantic import RootModel, TypeAdapter

from tau2.data_model.language import DomainBinding, PackExtension, TaskMetadata
from tau2.domains.banking_knowledge.data_model import Document
from tau2.domains.banking_knowledge.localization import knowledge_digest
from tau2.domains.banking_knowledge.regional import BankingPublishedRegionalBundle
from tau2.multilingual.factory.localization_artifact import RegionalMapArtifact
from tau2.multilingual.factory.localization_publish import build_regional_bundle
from tau2.multilingual.localization import LocalizationSource
from tau2.multilingual.runtime import language_root, load_pack
from tau2.multilingual.scenario_dispatch import bundle_model
from tau2.utils.artifacts import ReviewApproval, content_digest, write_artifact

BANKING_VOICE_DATE_GUIDANCE_VERSION = "banking-spoken-dates-v1"
BANKING_DEBIT_SHIPPING_VERSION = "banking-domestic-debit-shipping-v1"
BANKING_DATE_AND_SHIPPING_VERSION = (
    f"{BANKING_VOICE_DATE_GUIDANCE_VERSION}+{BANKING_DEBIT_SHIPPING_VERSION}"
)
BANKING_ES_SCENARIO_TEXT_VERSION = "banking-es-scenario-text-v1"
BANKING_ES_EXTENSION_VERSION = (
    f"{BANKING_DATE_AND_SHIPPING_VERSION}+{BANKING_ES_SCENARIO_TEXT_VERSION}"
)
BANKING_PT_SCENARIO_TEXT_VERSION = "banking-pt-scenario-text-v1"
BANKING_PT_EXTENSION_VERSION = (
    f"{BANKING_DATE_AND_SHIPPING_VERSION}+{BANKING_PT_SCENARIO_TEXT_VERSION}"
)
BANKING_OTHER_SCENARIO_TEXT_VERSION = "banking-hi-ko-zh-scenario-text-v1"
BANKING_KO_NAME_ORTHOGRAPHY_VERSION = "ko-name-orthography-dedup-v1"
BANKING_KO_EXTENSION_VERSION = (
    f"{BANKING_DATE_AND_SHIPPING_VERSION}+{BANKING_KO_NAME_ORTHOGRAPHY_VERSION}"
    f"+{BANKING_OTHER_SCENARIO_TEXT_VERSION}"
)
BANKING_ZH_NAME_ORTHOGRAPHY_VERSION = "zh-name-orthography-dedup-v1"
BANKING_ZH_EXTENSION_VERSION = (
    f"{BANKING_DATE_AND_SHIPPING_VERSION}+{BANKING_ZH_NAME_ORTHOGRAPHY_VERSION}"
    f"+{BANKING_OTHER_SCENARIO_TEXT_VERSION}"
)
DEBIT_CARD_ORDER_DOCUMENT_ID = "doc_bank_accounts_bank_accounts_(general)_023"
_US_SHIPPING_REQUIREMENT = (
    "Customer's address on file must be a valid US domestic address "
    "(international shipping is not available)"
)
_BANKING_SHIPPING_COUNTRIES = {
    "hi": "India",
    "ko": "South Korea",
    "zh": "China",
    "es": "Spain",
    "pt": "Brazil",
}


# Banking scenarios retain canonical US-formatted dates. Only the caller's spoken
# rendering changes; tasks, database values, and tool arguments stay untouched.
BANKING_VOICE_DATE_GUIDELINES = {
    "hi": (
        "Banking scenario dates are provided in MM/DD/YYYY format. When saying a "
        "date aloud, use the native Indian DD/MM/YYYY order: day, month, year. "
        "Keep the calendar date unchanged; only its spoken order changes. For "
        "dates, this takes precedence over the general instruction to keep "
        "concrete values exactly as written."
    ),
    "es": (
        "Banking scenario dates are provided in MM/DD/YYYY format. When saying a "
        "date aloud, use the native Spanish DD/MM/YYYY order: day, month, year. "
        "Keep the calendar date unchanged; only its spoken order changes. For "
        "dates, this takes precedence over the general instruction to keep "
        "concrete values exactly as written."
    ),
    "pt": (
        "Banking scenario dates are provided in MM/DD/YYYY format. When saying a "
        "date aloud, use the native Brazilian DD/MM/YYYY order: day, month, year. "
        "Keep the calendar date unchanged; only its spoken order changes. For "
        "dates, this takes precedence over the general instruction to keep "
        "concrete values exactly as written."
    ),
    "ko": (
        "Banking scenario dates are provided in MM/DD/YYYY format. When saying a "
        "date aloud, use the native Korean year–month–day order. Keep the calendar "
        "date unchanged; only its spoken order changes. For dates, this takes "
        "precedence over the general instruction to keep concrete values exactly "
        "as written."
    ),
    "zh": (
        "Banking scenario dates are provided in MM/DD/YYYY format. When saying a "
        "date aloud, use the native Mainland Mandarin year–month–day order. Keep "
        "the calendar date unchanged; only its spoken order changes. For dates, "
        "this takes precedence over the general instruction to keep concrete "
        "values exactly as written."
    ),
}


def _with_regional_debit_shipping(
    bundle: BankingPublishedRegionalBundle, language: str
) -> BankingPublishedRegionalBundle:
    """Add the versioned domestic shipping rule to an opt-in banking bundle."""
    from tau2.utils import DATA_DIR

    country = _BANKING_SHIPPING_COUNTRIES[language]
    directory = DATA_DIR / "tau2/domains/banking_knowledge"
    if knowledge_digest(directory) != bundle.knowledge_base_sha256:
        raise ValueError("English banking KB changed since regional publication")
    source = Document.model_validate_json(
        (directory / "documents" / f"{DEBIT_CARD_ORDER_DOCUMENT_ID}.json").read_bytes()
    )
    if source.id != DEBIT_CARD_ORDER_DOCUMENT_ID:
        raise ValueError("Reviewed debit card shipping document ID changed")
    if source.content.count(_US_SHIPPING_REQUIREMENT) != 1:
        raise ValueError("Reviewed debit card shipping requirement changed")
    if DEBIT_CARD_ORDER_DOCUMENT_ID in bundle.knowledge_document_overrides:
        raise ValueError("Debit card shipping document already overridden")
    replacement = source.model_copy(
        update={
            "content": source.content.replace(
                _US_SHIPPING_REQUIREMENT,
                "Customer's address on file must be a valid domestic address "
                f"in {country} (international shipping is not available)",
                1,
            )
        }
    )
    overrides = dict(bundle.knowledge_document_overrides)
    overrides[source.id] = replacement
    return bundle.model_copy(
        update={
            "knowledge_document_overrides": overrides,
            "revision": f"{bundle.revision}+{BANKING_DEBIT_SHIPPING_VERSION}",
        }
    )


def _with_spanish_scenario_text(
    bundle: BankingPublishedRegionalBundle,
) -> BankingPublishedRegionalBundle:
    """Repair reviewed Spanish caller context without changing scored gold actions."""
    replacements = {
        "task_077_es": ("living in Chicago", "living in Barcelona"),
        "task_082_es": (
            "FitLife charged me",
            "Gimnasio Pulso Urbano Premium charged me",
        ),
        "task_091_es": (
            "I've been in Seattle all week",
            "I've been in Logroño all week",
        ),
        "task_097_es": ("living in Seattle", "living in Huelva"),
    }
    payload = bundle.model_dump(mode="json")
    seen: set[str] = set()
    for task in payload["tasks"]:
        task_id = task["id"]
        if task_id not in replacements:
            continue
        old, new = replacements[task_id]
        instruction = task["user_scenario"]["instructions"]
        if instruction.count(old) != 1:
            raise ValueError(f"Reviewed Spanish scenario changed: {task_id}")
        task["user_scenario"]["instructions"] = instruction.replace(old, new, 1)
        if task_id == "task_091_es":
            old_note = "customer is in Seattle"
            notes = task["description"]["notes"]
            if notes.count(old_note) != 1:
                raise ValueError("Reviewed Spanish task_091 notes changed")
            task["description"]["notes"] = notes.replace(
                old_note, "customer is in Logroño", 1
            )
        seen.add(task_id)
    if seen != replacements.keys():
        raise ValueError(
            f"Missing reviewed Spanish scenarios: {replacements.keys() - seen}"
        )
    payload["revision"] += f"+{BANKING_ES_SCENARIO_TEXT_VERSION}"
    return BankingPublishedRegionalBundle.model_validate(payload)


def _with_portuguese_scenario_text(
    bundle: BankingPublishedRegionalBundle,
) -> BankingPublishedRegionalBundle:
    """Repair reviewed Portuguese caller text without changing scored gold actions."""
    replacements = {
        "task_082_pt": (
            "FitLife charged me",
            "Academia Corpo Pleno Premium charged me",
        ),
        "task_097_pt": ("living in Seattle", "living in São Paulo"),
    }
    payload = bundle.model_dump(mode="json")
    seen: set[str] = set()
    for task in payload["tasks"]:
        task_id = task["id"]
        if task_id not in replacements:
            continue
        old, new = replacements[task_id]
        instruction = task["user_scenario"]["instructions"]
        if instruction.count(old) != 1:
            raise ValueError(f"Reviewed Portuguese scenario changed: {task_id}")
        task["user_scenario"]["instructions"] = instruction.replace(old, new, 1)
        seen.add(task_id)
    if seen != replacements.keys():
        raise ValueError(
            f"Missing reviewed Portuguese scenarios: {replacements.keys() - seen}"
        )
    payload["revision"] += f"+{BANKING_PT_SCENARIO_TEXT_VERSION}"
    return BankingPublishedRegionalBundle.model_validate(payload)


def _with_other_scenario_text(
    bundle: BankingPublishedRegionalBundle, language: str
) -> BankingPublishedRegionalBundle:
    """Align reviewed Hindi, Korean, and Mandarin caller prose with regional entities."""
    replacements = {
        "hi": {
            "task_082_hi": ("FitLife charged me", "गोल्ड्स जिम इंडिया charged me"),
        },
        "ko": {
            "task_082_ko": ("FitLife charged me", "밸런스핏 프리미엄 charged me"),
            "task_091_ko": (
                "I've been in Seattle all week",
                "I've been in 서울특별시 all week",
            ),
        },
        "zh": {
            "task_082_zh": ("FitLife charged me", "劲活健身会所 charged me"),
            "task_090_zh": ("my apartment in Boston", "my apartment in 西安市"),
            "task_097_zh": ("living in Seattle", "living in 兰州市"),
        },
    }[language]
    payload = bundle.model_dump(mode="json")
    seen: set[str] = set()
    for task in payload["tasks"]:
        task_id = task["id"]
        if task_id not in replacements:
            continue
        old, new = replacements[task_id]
        instruction = task["user_scenario"]["instructions"]
        if instruction.count(old) != 1:
            raise ValueError(f"Reviewed {language} scenario changed: {task_id}")
        task["user_scenario"]["instructions"] = instruction.replace(old, new, 1)
        seen.add(task_id)
    if seen != replacements.keys():
        raise ValueError(
            f"Missing reviewed {language} scenarios: {replacements.keys() - seen}"
        )
    payload["revision"] += f"+{BANKING_OTHER_SCENARIO_TEXT_VERSION}"
    return BankingPublishedRegionalBundle.model_validate(payload)


def compile_approved_regional(
    *,
    map_path: Path,
    source_path: Path,
    approval_path: Path,
    metadata_path: Path,
    output: Path,
    data_root: Path,
) -> None:
    """Compile an exact approved map for an opt-in extension, with no model calls."""
    artifact = RegionalMapArtifact.model_validate_json(map_path.read_bytes())
    source = LocalizationSource.model_validate_json(source_path.read_bytes())
    approval = ReviewApproval.model_validate_json(approval_path.read_bytes())
    approval.require(artifact)
    metadata = TypeAdapter(dict[str, TaskMetadata]).validate_json(
        metadata_path.read_bytes()
    )
    bundle = build_regional_bundle(artifact, source, metadata, data_root)
    write_artifact(bundle, output)
    print(
        json.dumps(
            {
                "stage": "compile-approved-regional",
                "output": str(output),
                "sha256": content_digest(bundle),
            }
        )
    )


def publish_pack_extension(
    *,
    name: str,
    language: str,
    revision: str,
    bundle_paths: list[Path],
    output: Path,
) -> PackExtension:
    """Publish already approved regional bundles for opt-in custom runs."""
    pack = load_pack(language)
    pack_bytes = (language_root() / language / "pack.yaml").read_bytes()
    bundles = {}
    for path in bundle_paths:
        raw = path.read_bytes()
        domain = json.loads(raw)["domain"]
        if domain in bundles or domain in pack.domains:
            raise ValueError(f"duplicate or released domain: {domain}")
        bundle = bundle_model(domain).model_validate_json(raw)
        if bundle.settings.language != language:
            raise ValueError(f"bundle language differs from {language}: {domain}")
        if name == "banking-dates-v1" and domain == "banking_knowledge":
            assert isinstance(bundle, BankingPublishedRegionalBundle)
            bundle = _with_regional_debit_shipping(bundle, language)
            if language == "es":
                bundle = _with_spanish_scenario_text(bundle)
            elif language == "pt":
                bundle = _with_portuguese_scenario_text(bundle)
            elif language in {"hi", "ko", "zh"}:
                bundle = _with_other_scenario_text(bundle, language)
        bundles[domain] = bundle
    if not bundles:
        raise ValueError("pack extension needs at least one regional bundle")
    bindings = {
        domain: DomainBinding(
            revision=(
                f"{bundle.revision}+{BANKING_VOICE_DATE_GUIDANCE_VERSION}"
                if name == "banking-dates-v1"
                and domain == "banking_knowledge"
                and language in BANKING_VOICE_DATE_GUIDELINES
                else bundle.revision
            ),
            task_metadata_file=f"tasks/{domain}.metadata.json",
            regional_bundle_file=f"regional/{domain}.json",
            voice_guidelines=(
                BANKING_VOICE_DATE_GUIDELINES.get(language, "")
                if name == "banking-dates-v1" and domain == "banking_knowledge"
                else ""
            ),
        )
        for domain, bundle in sorted(bundles.items())
    }
    extension = PackExtension(
        name=name,
        language=language,
        base_pack_sha256=hashlib.sha256(pack_bytes).hexdigest(),
        revision=revision,
        domains=bindings,
    )
    for domain, bundle in sorted(bundles.items()):
        binding = bindings[domain]
        write_artifact(
            RootModel[dict[str, TaskMetadata]](root=bundle.metadata),
            output / binding.task_metadata_file,
        )
        write_artifact(bundle, output / binding.regional_bundle_file)
    # Write the manifest last so a partially published extension is not loadable.
    write_artifact(extension, output / "extension.json")
    return extension


def _run_publish(args: argparse.Namespace) -> None:
    extension = publish_pack_extension(
        name=args.name,
        language=args.language,
        revision=args.revision,
        bundle_paths=args.bundle,
        output=args.output,
    )
    print(
        f"Published {extension.name}/{extension.language}: "
        f"{', '.join(sorted(extension.domains))}"
    )


def add_pack_extension_args(commands: argparse._SubParsersAction) -> None:
    """Register opt-in language-pack extension publication."""
    compile_parser = commands.add_parser(
        "compile-approved-regional",
        help="Compile a digest-approved regional map for a separate-run extension",
    )
    for flag in ("map", "source", "approval", "metadata", "output"):
        compile_parser.add_argument("--" + flag, type=Path, required=True)
    compile_parser.add_argument("--data-root", type=Path)

    def run_compile(args: argparse.Namespace) -> None:
        from tau2.utils import DATA_DIR

        compile_approved_regional(
            map_path=args.map,
            source_path=args.source,
            approval_path=args.approval,
            metadata_path=args.metadata,
            output=args.output,
            data_root=args.data_root or DATA_DIR,
        )

    compile_parser.set_defaults(func=run_compile)
    publish = commands.add_parser(
        "publish-pack-extension",
        help="Bind approved regional bundles for separate runs only",
    )
    publish.add_argument("--name", required=True)
    publish.add_argument("--language", required=True)
    publish.add_argument("--revision", required=True)
    publish.add_argument("--bundle", type=Path, action="append", required=True)
    publish.add_argument("--output", type=Path, required=True)
    publish.set_defaults(func=_run_publish)
