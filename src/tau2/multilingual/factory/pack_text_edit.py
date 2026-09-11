# Copyright Sierra
"""Shared locate-and-splice helpers for surgical pack.yaml TEXT edits.

The factory verbs that touch a shipped pack's ``personas:`` section
(``generate-assets`` voice pinning, ``draft-continuers``, and the retired
one-shot redraft verbs before them) edit the raw pack.yaml text
rather than re-dumping YAML, so every untouched line — comments included —
stays byte-identical and concurrent edits to other pack keys rebase
trivially. This module owns the primitives they share: the persona-id line
regex, the persona block locator/replacers, the per-persona list-key
replacer, the nested scalar/block replacers, and the provenance-marker
strip/insert helpers.

``backfill_lib`` writes a whole TOP-LEVEL key by re-dumping it, which drops
every comment inside that key. That is correct only for the first write of a
block nobody has annotated yet; once a block carries authored comments (a
measured-defect note, a PENDING NATIVE REVIEW marker), any later edit of one
field inside it must come through :func:`replace_nested_scalar` (one authored
string) or :func:`set_nested_block` (one nested mapping/list) instead.
"""

import re
import textwrap
from typing import Optional

import yaml

from tau2.multilingual.factory.author_form import FactoryDraftError

# A persona-id line under the top-level `personas:` key (2-space indent).
PERSONA_KEY_RE = re.compile(r"^ {2}(\S[^:]*):\s*$")

# Every provenance marker block opens with this, which is what lets one
# marker be replaced without eating the ones stacked below it.
MARKER_HEADER_PREFIX = "# --- "


def persona_block_bounds(lines: list[str], persona_id: str) -> tuple[int, int]:
    """Return (start, end) line indices of a persona block under ``personas:``.

    ``start`` is the persona-id line; ``end`` is exclusive (first line that
    leaves the block — either the next persona at 2-space indent or any line at
    0 indent). Raises loudly if the persona id is not found.
    """
    start = None
    for i, line in enumerate(lines):
        m = PERSONA_KEY_RE.match(line)
        if m and m.group(1) == persona_id:
            start = i
            break
    if start is None:
        raise FactoryDraftError(
            f"could not locate persona block '{persona_id}' in pack.yaml"
        )

    end = len(lines)
    for j in range(start + 1, len(lines)):
        line = lines[j]
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent <= 2:  # next persona (2) or a new top-level key (0)
            end = j
            break
    return start, end


def set_persona_voice_id(pack_text: str, persona_id: str, voice_id: str) -> str:
    """Set ``voice_id`` for one persona, editing only that line (or inserting it)."""
    lines = pack_text.splitlines(keepends=True)
    start, end = persona_block_bounds(lines, persona_id)

    for k in range(start + 1, end):
        if re.match(r"^    voice_id:", lines[k]):
            newline = "\n" if lines[k].endswith("\n") else ""
            lines[k] = f"    voice_id: {voice_id}{newline}"
            return "".join(lines)

    # No existing voice_id line — insert right after the persona-id line.
    lines.insert(start + 1, f"    voice_id: {voice_id}\n")
    return "".join(lines)


def replace_persona_block(text: str, persona_id: str, persona: dict) -> str:
    """Replace ONE persona's mapping block inside ``personas:``; every other
    line (including sibling personas) stays byte-identical."""
    lines = text.splitlines(keepends=True)
    start, end = persona_block_bounds(lines, persona_id)
    rendered = textwrap.indent(
        yaml.safe_dump({persona_id: persona}, sort_keys=False, allow_unicode=True),
        "  ",
    )
    return "".join(lines[:start]) + rendered + "".join(lines[end:])


def replace_persona_list_blocks(
    text: str, *, key: str, new_by_persona: dict[str, list[str]]
) -> str:
    """Replace ONE list-valued key inside each named persona's block.

    Used by the redraft verbs (``draft-continuers`` today; the retired
    one-shot redrafts before it), which rewrite a single persona field and must
    leave every other line — sibling personas, comments, other keys —
    byte-identical so concurrent pack edits rebase trivially.

    Raises when any named persona's block cannot be located: a partial
    rewrite is worse than no rewrite.
    """
    key_re = re.compile(rf"^( +){re.escape(key)}:\s*$")
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    replaced: set[str] = set()
    in_personas = False
    current_persona: Optional[str] = None
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.rstrip("\n")
        if not in_personas:
            if stripped == "personas:":
                in_personas = True
            out.append(line)
            i += 1
            continue
        if stripped and not stripped[0].isspace():
            # Next top-level key (or root comment) ends the personas block.
            in_personas = False
            current_persona = None
            out.append(line)
            i += 1
            continue
        persona_match = PERSONA_KEY_RE.match(stripped)
        if persona_match:
            current_persona = persona_match.group(1)
            out.append(line)
            i += 1
            continue
        key_match = key_re.match(stripped)
        if key_match and current_persona in new_by_persona:
            indent = key_match.group(1)
            # Consume the existing block: list items at this indent plus any
            # deeper-indented continuation lines.
            #
            # A blank line, or a comment indented at least as far as the key,
            # does NOT end the block — a list written with a blank line or a
            # per-item comment in the middle would otherwise leave the items
            # after it stranded, and YAML would fold them back into the key we
            # just rewrote. Those lines are only consumed if a real list item
            # follows, which is what ``end`` tracks: trailing blanks and
            # comments stay byte-identical outside the block.
            j = i + 1
            end = i + 1
            while j < n:
                nxt = lines[j].rstrip("\n")
                nxt_indent = len(nxt) - len(nxt.lstrip(" "))
                if not nxt.strip() or (
                    nxt.lstrip().startswith("#") and nxt_indent >= len(indent)
                ):
                    j += 1
                    continue
                if nxt.startswith(f"{indent}- ") or nxt_indent > len(indent):
                    j += 1
                    end = j
                    continue
                break
            rendered = yaml.safe_dump(
                {key: new_by_persona[current_persona]},
                sort_keys=False,
                allow_unicode=True,
            )
            out.append(textwrap.indent(rendered, indent))
            replaced.add(current_persona)
            i = end
            continue
        out.append(line)
        i += 1
    missing = sorted(set(new_by_persona) - replaced)
    if missing:
        raise FactoryDraftError(
            f"could not locate the {key} block for persona(s) {missing} in "
            "pack.yaml — refusing a partial rewrite"
        )
    return "".join(out)


# A `key:` line in a pack.yaml mapping (list items, which start with '- ',
# deliberately do not match — the nested-scalar path walks mappings only).
MAPPING_KEY_RE = re.compile(r"^( *)([A-Za-z_][\w-]*):(.*)$")


def _locate_mapping_key(
    lines: list[str], path: tuple[str, ...]
) -> Optional[tuple[int, int, str]]:
    """Find the single ``key:`` line at mapping ``path``: (index, indent, rest).

    Returns None when the path is absent. Raises when it matches more than
    once — an ambiguous path is never silently resolved to the first hit.
    """
    stack: list[tuple[int, str]] = []
    matches: list[tuple[int, int, str]] = []
    for i, raw in enumerate(lines):
        stripped = raw.rstrip("\n")
        if not stripped.strip() or stripped.lstrip().startswith("#"):
            continue
        m = MAPPING_KEY_RE.match(stripped)
        if m is None:
            continue
        indent, key, rest = len(m.group(1)), m.group(2), m.group(3)
        while stack and stack[-1][0] >= indent:
            stack.pop()
        stack.append((indent, key))
        if tuple(k for _, k in stack) == path:
            matches.append((i, indent, rest))
    if len(matches) > 1:
        raise FactoryDraftError(
            f"pack.yaml path {'.'.join(path)} matched {len(matches)} times — "
            "refusing an ambiguous edit"
        )
    return matches[0] if matches else None


def _value_block_end(lines: list[str], start: int, indent: int) -> int:
    """Exclusive end index of the value owned by the key line at ``start``.

    The value is the run of following lines that are more deeply indented, or
    are list items at the key's own indent (``yaml.safe_dump`` writes a block
    sequence at its key's indent). A blank line or a comment indented at least
    as far as the key does NOT end the value — it is absorbed only if real
    value content follows it, which is what ``end`` tracks. So a comment
    sitting between this value and the NEXT sibling key stays OUTSIDE the
    replaced span, where it belongs: it documents the sibling, not this value.
    """
    j = end = start + 1
    pad = " " * indent
    while j < len(lines):
        nxt = lines[j].rstrip("\n")
        nxt_indent = len(nxt) - len(nxt.lstrip(" "))
        if not nxt.strip() or (nxt.lstrip().startswith("#") and nxt_indent >= indent):
            j += 1
            continue
        if nxt_indent > indent or nxt.startswith(f"{pad}- "):
            j += 1
            end = j
            continue
        break
    return end


# ``yaml.safe_dump``'s default line width. A spliced key is dumped on its own
# and then indented, so the emitter must be told to fold that much earlier or
# the re-rendered lines wrap at a different column than the rest of the file.
_YAML_WIDTH = 80


def _render_mapping_key(key: str, value: object, indent: int) -> str:
    """``key: value`` dumped as YAML and indented to sit at ``indent``.

    Folding matches what a whole-file dump would produce at this depth: the
    emitter's width is reduced by ``indent``, since the indent is added after
    it has already chosen its wrap points.
    """
    return textwrap.indent(
        yaml.safe_dump(
            {key: value},
            sort_keys=False,
            allow_unicode=True,
            width=max(_YAML_WIDTH - indent, 20),
        ),
        " " * indent,
    )


def set_nested_block(text: str, *, path: tuple[str, ...], new_value: object) -> str:
    """Replace (or insert) ONE nested mapping/list value at ``path``, in place.

    ``path`` is the mapping key chain from the document root, e.g.
    ``("localization", "domain_glossaries", "airline")``. Only that key's own
    lines are re-rendered; every other line — sibling keys, and crucially the
    authored comments elsewhere in the enclosing block — stays byte-identical.
    Use this instead of a whole-block re-dump whenever the block being edited
    may carry human review notes.

    When ``path`` is absent but its parent exists, the key is INSERTED at the
    end of the parent's block (the gap-filling case: a pack that predates the
    field). A missing parent is a loud error, not an invented nesting.
    """
    lines = text.splitlines(keepends=True)
    found = _locate_mapping_key(lines, path)
    if found is not None:
        start, indent, _ = found
        end = _value_block_end(lines, start, indent)
        rendered = _render_mapping_key(path[-1], new_value, indent)
        return "".join(lines[:start]) + rendered + "".join(lines[end:])

    if len(path) == 1:
        raise FactoryDraftError(
            f"pack.yaml has no top-level `{path[0]}:` key to set — a whole "
            "top-level block is appended by backfill_lib, not spliced here"
        )
    parent = _locate_mapping_key(lines, path[:-1])
    if parent is None:
        raise FactoryDraftError(
            f"could not locate pack.yaml path {'.'.join(path[:-1])} to insert "
            f"'{path[-1]}' into — refusing to invent the nesting"
        )
    parent_start, parent_indent, parent_rest = parent
    inline = parent_rest.strip()
    if inline and inline != "{}":
        raise FactoryDraftError(
            f"pack.yaml path {'.'.join(path[:-1])} holds an inline "
            f"{'sequence' if inline == '[]' else 'scalar'}, so '{path[-1]}' "
            "cannot be nested under it"
        )
    if inline == "{}":
        # An empty inline mapping cannot take an indented child; reopen the
        # key as a block mapping first.
        lines[parent_start] = f"{' ' * parent_indent}{path[-2]}:\n"
    parent_end = _value_block_end(lines, parent_start, parent_indent)
    rendered = _render_mapping_key(path[-1], new_value, parent_indent + 2)
    return "".join(lines[:parent_end]) + rendered + "".join(lines[parent_end:])


def replace_nested_scalar(text: str, *, path: tuple[str, ...], new_value: str) -> str:
    """Replace ONE scalar value at a nested mapping ``path``, in place.

    ``path`` is the mapping key chain from the document root, e.g.
    ``("localization", "spelling_alphabet", "avoid")``. Only that key's line
    and its wrapped continuation lines are re-rendered (at the key's own
    indent, via ``yaml.safe_dump`` so quoting/folding match the rest of the
    file); every other line — sibling keys, the block's authored comments —
    stays byte-identical.

    Raises when the path cannot be located, is ambiguous (matched twice), or
    does not hold an inline scalar (a key introducing a nested block or list
    is a :func:`set_nested_block` edit, not a scalar one).
    """
    lines = text.splitlines(keepends=True)
    found = _locate_mapping_key(lines, path)
    if found is None:
        raise FactoryDraftError(
            f"could not locate pack.yaml path {'.'.join(path)} — refusing to guess"
        )
    start, indent, rest = found
    if not rest.strip():
        raise FactoryDraftError(
            f"pack.yaml path {'.'.join(path)} is not an inline scalar "
            "(the key introduces a nested block or list)"
        )
    # Consume the value's wrapped continuation lines: more deeply indented,
    # and not a list item of their own.
    end = start + 1
    while end < len(lines):
        nxt = lines[end].rstrip("\n")
        if not nxt.strip():
            break
        nxt_indent = len(nxt) - len(nxt.lstrip(" "))
        if nxt_indent <= indent or nxt.lstrip().startswith("- "):
            break
        end += 1
    rendered = _render_mapping_key(path[-1], new_value, indent)
    return "".join(lines[:start]) + rendered + "".join(lines[end:])


def strip_marker_block(text: str, header: str) -> str:
    """Remove one provenance marker comment block (the line starting with
    ``header`` plus its following comment lines) — idempotent re-runs.
    Marker blocks with a different header stay untouched.

    A pack accumulates one marker per verb that has written to it and they
    sit contiguously above ``personas:``, so "keep eating comment lines" would
    delete every marker BELOW the one being replaced — a ``--force`` re-run of
    any single verb would silently drop its siblings' provenance. The next
    marker header therefore ends the block.
    """
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    in_marker = False
    for line in lines:
        stripped = line.rstrip("\n")
        if in_marker:
            if stripped.startswith("#") and not stripped.startswith(
                MARKER_HEADER_PREFIX
            ):
                continue
            in_marker = False
        if stripped.startswith(header):
            in_marker = True
            continue
        out.append(line)
    return "".join(out)


def insert_marker_before_key(text: str, *, marker: str, key: str) -> str:
    """Insert a provenance marker comment block right above ``<key>:``."""
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    inserted = False
    for line in lines:
        if not inserted and line.rstrip("\n").rstrip() == f"{key}:":
            out.append(marker)
            inserted = True
        out.append(line)
    if not inserted:
        raise FactoryDraftError(f"pack.yaml has no top-level `{key}:` key")
    return "".join(out)


def insert_marker_before_personas(text: str, marker: str) -> str:
    """Insert a provenance marker comment block right above ``personas:``."""
    return insert_marker_before_key(text, marker=marker, key="personas")
