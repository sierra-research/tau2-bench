"""Apply approved banking entity names to English knowledge documents."""

import json
import re
from collections import defaultdict
from pathlib import Path

from tau2.domains.banking_knowledge.data_model import Document
from tau2.multilingual.factory.localization_artifact import RegionalMapArtifact

# Exact English-source occurrences reviewed as entity references. Values come
# from the localization map; this table identifies context, not translations.
_ENTITY_DOCUMENTS = {
    "Amazon": {
        "doc_credit_cards_credit_cards_(general)_008",
        "doc_credit_cards_ecocard_004",
    },
    "Starbucks": {"doc_credit_cards_credit_cards_(general)_008"},
    "Target": {
        "doc_credit_cards_credit_cards_(general)_008",
        "doc_credit_cards_ecocard_004",
    },
    "Salesforce": {"doc_business_credit_cards_business_bronze_rewards_card_003"},
    "Coursera": {"doc_business_credit_cards_business_silver_rewards_card_005"},
    "Tesla Supercharger": {"doc_credit_cards_ecocard_004"},
    "ThredUp": {"doc_credit_cards_ecocard_004"},
    "WeWork": {"doc_business_credit_cards_business_bronze_rewards_card_003"},
    "Dell": {"doc_business_credit_cards_business_silver_rewards_card_005"},
    "AWS": {"doc_business_checking_accounts_sky_blue_003"},
    "ATM WITHDRAWAL - CHASE BANK #2847 CHICAGO IL": {
        "doc_bank_accounts_bank_accounts_(general)_018"
    },
}

# Here "Target" means a threshold or an action, not the retailer.
_NON_ENTITY_DOCUMENTS = {
    "Target": {
        "doc_business_credit_cards_green_rewards_card_009",
        "doc_business_savings_accounts_automatic_sweep_program_003",
        "doc_business_savings_accounts_automatic_sweep_program_008",
    }
}

# The KB sometimes names a brand while transactions use a longer or adapted
# merchant name. These source relationships must be explicit in the same rule.
_BRONZE_ALIASES = {
    "Slack": "Slack Technologies",
    "Zoom": "Zoom Video",
}
_SILVER_ALIASES = {
    "Apple": "Apple Store",
    "Microsoft": "Microsoft 365",
    "Dell": "Dell Technologies",
}


def _one_native(translations: dict[str, set[str]], source: str) -> str | None:
    targets = translations.get(source, set())
    if len(targets) > 1:
        raise ValueError(f"Ambiguous banking KB entity {source!r}: {sorted(targets)}")
    return next(iter(targets)) if targets else None


def _replace_bullet(content: str, before: str, after: str) -> str:
    needle = f"- {before}\n"
    if content.count(needle) != 1:
        raise ValueError(f"Banking KB alias bullet changed: {before!r}")
    return content.replace(needle, f"- {after}\n", 1)


def _add_rule_aliases(
    identifier: str, content: str, translations: dict[str, set[str]]
) -> str:
    """Connect approved transaction names to their English source KB rules."""
    if identifier == "doc_credit_cards_ecocard_004":
        parent = _one_native(translations, "Target")
        collection = _one_native(translations, "Target - Eco Collection")
        if parent is not None and collection is not None:
            content = _replace_bullet(
                content,
                parent,
                f"{parent}\n- {collection} (eco collection sold by {parent})",
            )
    elif identifier == "doc_business_credit_cards_business_bronze_rewards_card_003":
        for brand, transaction in _BRONZE_ALIASES.items():
            native = _one_native(translations, transaction)
            if native is not None:
                content = _replace_bullet(content, brand, f"{native} ({brand})")
    elif identifier == "doc_business_credit_cards_business_silver_rewards_card_005":
        for brand, transaction in _SILVER_ALIASES.items():
            native = _one_native(translations, transaction)
            if native is not None:
                listed = _one_native(translations, brand) or brand
                content = _replace_bullet(
                    content, listed, f"{listed} (including {native})"
                )
    return content


def localize_knowledge_entities(
    artifact: RegionalMapArtifact, data_root: Path
) -> dict[str, Document]:
    """Return only documents changed by mapped entities, preserving English prose.

    The approved map supplies every replacement. Document IDs and unrelated words
    stay unchanged. Source inventory validation is performed by the caller.
    """
    directory = data_root / "tau2/domains/banking_knowledge/documents"
    documents = {}
    for path in sorted(directory.glob("*.json")):
        doc = Document.model_validate(json.loads(path.read_text()))
        documents[doc.id] = doc
    translations: dict[str, set[str]] = defaultdict(set)
    for entry in artifact.entries:
        source, native = entry.source, entry.forms.native
        if (
            entry.category in {"other", "prose"}
            and isinstance(source, str)
            and isinstance(native, str)
            and source != native
        ):
            translations[source].add(native)

    corpus = "\n".join(f"{doc.title}\n{doc.content}" for doc in documents.values())
    patterns = {}
    for source in translations:
        if source not in corpus:
            continue
        pattern = re.compile(
            r"(?<![A-Za-z0-9])" + re.escape(source) + r"(?![A-Za-z0-9])"
        )
        if pattern.search(corpus):
            patterns[source] = pattern
    for source, pattern in patterns.items():
        matching = {
            doc.id
            for doc in documents.values()
            if pattern.search(doc.title) or pattern.search(doc.content)
        }
        expected = _ENTITY_DOCUMENTS.get(source, set()) | _NON_ENTITY_DOCUMENTS.get(
            source, set()
        )
        if matching != expected:
            raise ValueError(f"Unreviewed banking KB entity context: {source!r}")
    changed = {}
    for doc in documents.values():
        title, content = doc.title, doc.content
        for source in sorted(patterns, key=len, reverse=True):
            if doc.id not in _ENTITY_DOCUMENTS[source]:
                continue
            pattern = patterns[source]
            if not pattern.search(title) and not pattern.search(content):
                continue
            targets = translations[source]
            if len(targets) != 1:
                raise ValueError(
                    f"Ambiguous banking KB entity {source!r}: {sorted(targets)}"
                )
            replacement = next(iter(targets))
            title = pattern.sub(lambda _: replacement, title)
            content = pattern.sub(lambda _: replacement, content)
        content = _add_rule_aliases(doc.id, content, translations)
        if title != doc.title or content != doc.content:
            changed[doc.id] = doc.model_copy(
                update={"title": title, "content": content}
            )
    return changed
