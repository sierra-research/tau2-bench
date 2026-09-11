# Copyright Sierra
"""The shared engine for the factory's surgical pack-block backfill verbs.

``tau2 factory draft-localization`` and the retired native-strings /
pragmatics backfills all follow the same shape — build a fixed prompt,
``json_call`` it, validate the reply, gate on content problems, then splice
ONLY the drafted block into the shipped ``pack.yaml``
(every other line byte-identical) after re-validating the merged pack. This
module owns that shape once:

- :func:`draft_validated` — the draft -> validate -> problems gate;
- :func:`load_shipped_pack` — open a shipped pack.yaml for surgical editing,
  with both the raw text and the loader-normalized data view;
- :func:`parse_existing_block` — the pack's existing block, validated (an
  invalid block is a loud error, never silently replaced);
- :func:`render_backfill_block` — the appended block text: marker header +
  provenance comment + the dumped top-level key;
- :func:`write_backfill_block` — strip-any-previous-block, splice, re-validate
  the merged pack (:func:`validate_merged_pack`, through
  ``loader.normalize_pack_data`` so the backfill validates exactly what the
  runtime will load), then write;
- :func:`write_pack_text` — the same validate-then-write gate for a text a
  verb edited surgically instead of re-dumping;
- :class:`BackfillOutcome` — the common outcome shape the verbs report.

A whole-block re-dump necessarily drops the comments inside the block it
rewrites, so it is only right for the FIRST write of a block. Editing one
field of a block that may already carry authored review notes goes through
``factory.pack_text_edit`` (``set_nested_block`` /
``replace_nested_scalar``) plus :func:`write_pack_text`.
"""

import re
from pathlib import Path
from typing import Callable, Optional, TypeVar

import yaml
from pydantic import BaseModel, Field, ValidationError

from tau2.multilingual.factory.author_form import FactoryDraftError
from tau2.multilingual.factory.llm import FactoryLLM
from tau2.multilingual.loader import normalize_pack_data
from tau2.multilingual.localize_lib import multilingual_data_dir
from tau2.multilingual.schema import LanguagePack

T = TypeVar("T")
M = TypeVar("M", bound=BaseModel)


def draft_validated(
    llm: FactoryLLM,
    model: str,
    *,
    system: str,
    user: str,
    call_name: str,
    parse: Callable[[object], T],
    problems: Callable[[T], list[str]],
    subject: str,
    failure_noun: str,
    reasoning_effort: Optional[str] = None,
) -> T:
    """One fixed-prompt LLM call -> a validated, problem-free ``T``.

    ``parse`` turns the raw JSON reply into the candidate; ``problems`` returns
    content-contract violations of a schema-valid candidate. Either failure
    raises a loud ``FactoryDraftError`` naming ``subject`` and ``failure_noun``.
    """
    data = llm.json_call(
        model,
        system,
        user,
        call_name=call_name,
        reasoning_effort=reasoning_effort,
    )
    try:
        candidate = parse(data)
    except ValidationError as exc:
        raise FactoryDraftError(
            f"LLM call '{call_name}' for '{subject}' returned an invalid "
            f"{failure_noun}: {exc}"
        ) from exc
    found = problems(candidate)
    if found:
        raise FactoryDraftError(
            f"LLM call '{call_name}' for '{subject}' returned an invalid "
            f"{failure_noun}: {'; '.join(found)}"
        )
    return candidate


class ShippedPack(BaseModel):
    """A shipped ``pack.yaml`` opened for surgical backfill editing.

    ``original`` is the raw file text (the splice base — backfills edit text,
    not re-dumped YAML, so untouched lines stay byte-identical). ``data`` is
    the loader-normalized mapping (``loader.normalize_pack_data``): the
    ``experiment`` sibling removed, guidelines paths absolute, and list-form
    ``personas`` / ``acoustic_presets`` keyed into dicts — so backfill reads
    see exactly the shape the runtime loads.
    """

    language: str
    pack_path: Path
    original: str
    data: dict

    @property
    def personas(self) -> dict[str, dict]:
        """The pack's persona mappings, keyed by persona id."""
        return {
            str(pid): p
            for pid, p in (self.data.get("personas") or {}).items()
            if isinstance(p, dict)
        }

    @property
    def display_name(self) -> str:
        return str(self.data.get("display_name") or self.language)

    @property
    def script(self) -> str:
        """The personas' (first, alphabetically) script code; 'latn' default."""
        scripts = {
            str(p.get("script")) for p in self.personas.values() if p.get("script")
        }
        return sorted(scripts)[0] if scripts else "latn"


def load_shipped_pack(lang: str) -> ShippedPack:
    """Open ``data/tau2/multilingual/<lang>/pack.yaml`` for backfilling.

    Raises ``FactoryDraftError`` when the pack does not exist or is not a
    YAML mapping (loud, before any LLM call is spent).
    """
    pack_path = multilingual_data_dir() / lang / "pack.yaml"
    if not pack_path.exists():
        raise FactoryDraftError(f"no shipped pack for '{lang}' at {pack_path}")
    original = pack_path.read_text()
    raw = yaml.safe_load(original)
    if not isinstance(raw, dict):
        raise FactoryDraftError(f"{pack_path} is not a YAML mapping")
    return ShippedPack(
        language=lang,
        pack_path=pack_path,
        original=original,
        data=normalize_pack_data(raw, pack_path.parent),
    )


class BackfillOutcome(BaseModel):
    """The common shape of what a backfill verb did, for the CLI and tests.

    Verb-specific outcomes subclass this with their own counts and their
    prompt version/sha defaults.
    """

    language: str
    pack_path: Path
    written: bool = Field(
        description="False when skipped (complete block already present)"
    )
    replaced: bool = Field(
        default=False,
        description="True when --force RE-ROLLED already-reviewed content "
        "(rather than only filling gaps). The re-roll is scoped to what the "
        "verb is scoped to — for draft-localization, the requested domain's "
        "glossary — so `replaced` never means the whole block was rewritten",
    )
    upgraded: bool = Field(
        default=False,
        description="True when an existing block was merged into rather than "
        "written fresh: the fields in `upgraded_fields` were drafted in while "
        "every other already-reviewed field kept its exact existing value",
    )
    upgraded_fields: list[str] = Field(
        default_factory=list,
        description="The fields the merge drafted in — gaps the block was "
        "missing, plus anything --force deliberately re-rolled",
    )
    prompt_version: str
    prompt_sha256: str
    model: str = ""


def parse_existing_block(
    pack: ShippedPack, *, key: str, model_cls: type[M]
) -> Optional[M]:
    """The pack's existing ``<key>`` block, validated — or None when absent.

    An invalid existing block is a loud ``FactoryDraftError`` (fix it first),
    never silently replaced.
    """
    raw = pack.data.get(key)
    if raw is None:
        return None
    try:
        return model_cls.model_validate(raw)
    except ValidationError as exc:
        raise FactoryDraftError(
            f"pack '{pack.language}' has an invalid {key} block; fix it "
            f"before backfilling: {exc}"
        ) from exc


def render_backfill_provenance(
    *,
    verb: str,
    prompt_version: str,
    sha256: str,
    model: str,
    upgraded_fields: Optional[list[str]] = None,
    upgrade_action: str = "drafted",
    field_noun: str = "fields",
) -> str:
    """The block's provenance comment line(s).

    ``upgraded_fields`` switches to the upgrade form (``upgrade_action`` names
    what happened to them, ``field_noun`` what the preserved rest is called);
    without it, the whole block was drafted at once.
    """
    if upgraded_fields:
        return (
            f"# Upgraded by `tau2 factory {verb}`: "
            f"{', '.join(upgraded_fields)} {upgrade_action} by prompt "
            f"{prompt_version} sha256:{sha256[:12]} (model:{model});\n"
            f"# all previously-reviewed {field_noun} preserved verbatim.\n"
        )
    return (
        f"# Drafted by `tau2 factory {verb}` (prompt {prompt_version} "
        f"sha256:{sha256[:12]}, model:{model}).\n"
    )


def render_backfill_block(
    key: str,
    config: BaseModel,
    *,
    header: str,
    verb: str,
    prompt_version: str,
    sha256: str,
    model: str,
    comment: str,
    upgraded_fields: Optional[list[str]] = None,
    upgrade_action: str = "drafted",
    field_noun: str = "fields",
) -> str:
    """The appended pack.yaml block: marker header + provenance + the key.

    ``comment`` is the verb's fixed explanatory comment lines (each
    ``#``-prefixed, newline-terminated); the provenance line comes from
    :func:`render_backfill_provenance`.
    """
    body = yaml.safe_dump(
        {key: config.model_dump(mode="json", exclude_none=True)},
        sort_keys=False,
        allow_unicode=True,
    )
    provenance = render_backfill_provenance(
        verb=verb,
        prompt_version=prompt_version,
        sha256=sha256,
        model=model,
        upgraded_fields=upgraded_fields,
        upgrade_action=upgrade_action,
        field_noun=field_noun,
    )
    return f"\n{header} ---\n{provenance}{comment}{body}"


def write_backfill_block(
    pack: ShippedPack, *, key: str, marker: str, block: str, replacing: bool
) -> None:
    """Splice ``block`` onto the pack text and write it, re-validating first.

    ``replacing`` strips the previous ``<key>`` block (and its ``marker``
    comment); every other line stays byte-identical. The merged pack must
    load as a ``LanguagePack`` before anything is written (loud, not
    best-effort).
    """
    base_text = (
        strip_top_level_block(pack.original, key=key, marker=marker)
        if replacing
        else pack.original
    )
    write_pack_text(pack, base_text.rstrip("\n") + "\n" + block)


def write_pack_text(pack: ShippedPack, new_text: str) -> None:
    """Write an edited pack.yaml text, re-validating the merged pack first.

    The shared write gate: whether the new text came from a whole-block
    re-dump (:func:`write_backfill_block`) or from a surgical
    ``pack_text_edit`` splice, nothing reaches disk unless it loads as a
    ``LanguagePack``.
    """
    validate_merged_pack(new_text, pack.pack_path.parent)
    pack.pack_path.write_text(new_text)


# A non-indented `key:` line — what ends an appended top-level block.
_TOP_LEVEL_KEY_RE = re.compile(r"^[A-Za-z_]+:")


def strip_top_level_block(text: str, *, key: str, marker: str) -> str:
    """Remove the top-level ``<key>:`` block (and its ``marker`` comment)
    from a pack.yaml text, leaving every other line untouched.

    The block is what the backfill verbs append: an optional marker-comment
    header, the bare top-level ``<key>:`` line, and its indented body up to
    the next non-blank column-0 line (a key OR a root-level comment) or EOF.
    """
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    in_header = False  # between the marker comment and `<key>:`
    in_body = False  # inside the `<key>:` mapping
    for line in lines:
        stripped = line.rstrip("\n")
        if in_header:
            # Header comment lines belong to the block; the header ends at
            # `<key>:` (drop and enter the body) or, defensively, at any
            # other top-level key (keep it).
            if stripped.startswith(f"{key}:"):
                in_header, in_body = False, True
                continue
            if _TOP_LEVEL_KEY_RE.match(stripped):
                in_header = False
                out.append(line)
            continue
        if in_body:
            # The body ends at the first non-blank column-0 line — a key OR a
            # root-level comment; only the block's own indented/blank lines
            # are dropped with it.
            if stripped and not stripped[0].isspace():
                in_body = False
                out.append(line)
            continue
        if stripped.startswith(marker):
            in_header = True
            continue
        if stripped.startswith(f"{key}:"):
            in_body = True
            continue
        out.append(line)
    return "".join(out)


def validate_merged_pack(new_text: str, pack_dir: Path) -> None:
    """The pre-write gate: the merged pack text must load as a ``LanguagePack``.

    Runs the loader's own ``normalize_pack_data`` first, so the backfill
    validates exactly what the runtime will load (loud, not best-effort —
    a validation error aborts before anything is written).
    """
    merged = yaml.safe_load(new_text)
    LanguagePack.model_validate(normalize_pack_data(merged, pack_dir))
