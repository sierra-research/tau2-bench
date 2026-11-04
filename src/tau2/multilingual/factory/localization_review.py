"""Standalone English review of source-checked regional localization proposals."""

import html
import json
from pathlib import Path

from pydantic import JsonValue, TypeAdapter

from tau2.data_model.tasks import Task
from tau2.multilingual.localization import (
    LocalizationEntry,
    LocalizationMap,
    ProseEntityBinding,
    SourceDocument,
    json_content_digest,
)


def _pre(value: JsonValue) -> str:
    text = (
        value
        if isinstance(value, str)
        else json.dumps(value, ensure_ascii=False, indent=2)
    )
    return f"<pre>{html.escape(text)}</pre>"


def _entry_row(
    entry: LocalizationEntry,
    mapping: LocalizationMap,
    documents: dict[str, SourceDocument],
    anchor: str,
) -> str:
    locations = []
    for occurrence in entry.occurrences:
        document = documents.get(occurrence.document_id)
        source_path = document.source_path if document else "Unknown source document"
        pointer = "".join(
            "/" + str(part).replace("~", "~0").replace("/", "~1")
            for part in occurrence.path
        )
        locations.append(
            f"<li><code>{html.escape(source_path)}</code>"
            f"<br><code>{html.escape(pointer or '(document root)')}</code>"
            f"<br><small>Document: {html.escape(occurrence.document_id)}</small></li>"
        )
    tool_note = (
        '<p class="warning">Tool-text evidence: this mapping alone does not establish '
        "that runtime tool messages or rules have been wired to it.</p>"
        if entry.category == "tool_text"
        or any(
            documents.get(item.document_id) is not None
            and documents[item.document_id].kind == "tool_text"
            for item in entry.occurrences
        )
        else ""
    )
    return (
        f'<tr class="entry" id="{anchor}" '
        f'data-domain="{html.escape(mapping.source.domain)}" '
        f'data-language="{html.escape(mapping.settings.language)}" '
        f'data-category="{html.escape(entry.category)}" '
        f'data-operation="{html.escape(entry.operation)}">'
        f'<th scope="row"><a href="#{anchor}">{html.escape(entry.id)}</a>'
        f"<p>{html.escape(mapping.settings.language)} · {html.escape(mapping.source.domain)}</p>"
        f"<p>{html.escape(entry.category.replace('_', ' '))} · {html.escape(entry.operation)}</p>"
        f"<p>{html.escape(entry.context)}</p>"
        f"<details><summary>Source paths ({len(locations)})</summary><ul>{''.join(locations)}</ul></details></th>"
        f"<td>{_pre(entry.source)}</td><td>{_pre(entry.forms.native)}</td>"
        f"<td>{_pre(entry.forms.romanized)}</td>"
        f"<td><p>{html.escape(entry.forms.english)}</p>"
        f"<h4>Rationale / review concerns</h4><p>{html.escape(entry.notes)}</p>{tool_note}</td></tr>"
    )


def _table(rows: list[str]) -> str:
    return (
        '<div class="table-scroll"><table><thead><tr><th>Entry and source paths</th>'
        "<th>Source value</th><th>Native form</th><th>Romanized form</th>"
        "<th>English meaning and rationale</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def _binding_row(
    binding: ProseEntityBinding, mapping: LocalizationMap, anchor: str
) -> str:
    """Expose supplemental source references as first-class pending review rows."""
    operation = "preserve" if binding.kind == "protected" else "adapt"
    parent = next(entry for entry in mapping.entries if entry.id == binding.entry_id)
    return (
        f'<tr class="entry" id="{anchor}" '
        f'data-domain="{html.escape(mapping.source.domain)}" '
        f'data-language="{html.escape(mapping.settings.language)}" '
        f'data-category="prose_reference" data-operation="{operation}">'
        f'<th scope="row"><a href="#{anchor}">Prose reference</a>'
        f"<p>{html.escape(binding.kind)} · {html.escape(mapping.settings.language)} · "
        f"{html.escape(mapping.source.domain)}</p>"
        f"<p>Within source entry <code>{html.escape(binding.entry_id)}</code>.</p>"
        f"<p>{html.escape(parent.context)}</p>"
        f"<details><summary>Containing source text</summary>{_pre(parent.source)}</details></th>"
        f"<td>{_pre(binding.source)}</td><td>{_pre(binding.forms.native)}</td>"
        f"<td>{_pre(binding.forms.romanized)}</td>"
        f"<td><p>{html.escape(binding.forms.english)}</p><h4>Rationale / review concerns</h4>"
        f"<p>{html.escape(binding.reason)}</p>"
        f"<details><summary>Reference lineage</summary>{_pre(binding.model_dump(mode='json', exclude={'source', 'forms', 'reason'}))}</details>"
        "</td></tr>"
    )


def _select(name: str, label: str, values: set[str]) -> str:
    options = "".join(
        f'<option value="{html.escape(value)}">{html.escape(value.replace("_", " "))}</option>'
        for value in sorted(values)
    )
    return f'<label>{label}<select id="{name}"><option value="">All</option>{options}</select></label>'


def _compiled_name_details(mapping: LocalizationMap, index: int) -> str:
    """Show the actual caller instructions emitted by the regional compiler."""
    if mapping.source.domain != "retail" or not any(
        document.kind == "tasks" for document in mapping.source.documents
    ):
        return ""
    from tau2.multilingual.factory.localization_compile import compile_localization

    adapter = TypeAdapter(list[Task])
    representations = []
    for representation in ("native", "romanized"):
        documents = compile_localization(mapping, representation)
        representations.append(
            [
                task
                for document in documents
                if document.kind == "tasks"
                for task in adapter.validate_python(document.content)
            ]
        )
    rows = []
    for native, romanized in zip(*representations, strict=True):
        if native.id != romanized.id:
            raise ValueError("Compiled task representations have different task IDs")
        rows.append(
            '<tr class="task-preview">'
            f'<th scope="row">{html.escape(native.id)}</th>'
            f"<td>{_pre(native.user_scenario.instructions if isinstance(native.user_scenario.instructions, str) else native.user_scenario.instructions.known_info)}</td>"
            f"<td>{_pre(romanized.user_scenario.instructions if isinstance(romanized.user_scenario.instructions, str) else romanized.user_scenario.instructions.known_info)}</td></tr>"
        )
    task_label = "task" if len(rows) == 1 else "tasks"
    return (
        f'<details id="compiled-name-details-{index}">'
        f"<summary>Compiled caller instructions ({len(rows)} {task_label})</summary>"
        "<p>Each task includes explicit given-name and family-name facts from its "
        "localized customer record. These are the compiled instructions for each "
        "representation, including the deterministic name clarification.</p>"
        '<div class="table-scroll"><table><thead><tr><th>Task ID</th>'
        "<th>Native caller instructions</th><th>Romanized caller instructions</th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div></details>"
    )


def write_localization_review(maps: list[LocalizationMap], output: Path) -> Path:
    """Write every proposed mapping to a deterministic, self-contained index.html.

    Preserved values are included in collapsed groups. Search and filters operate
    on every row, including source locations. No source content becomes executable
    HTML or JavaScript, and rendering never supplies human approval.
    """
    if not maps:
        raise ValueError("A localization review requires at least one map")
    maps = [LocalizationMap.model_validate(item.model_dump()) for item in maps]
    sections = []
    compiled_links = []
    expected_count = 0
    covered_count = 0
    occurrence_count = 0
    entry_count = 0
    for index, mapping in enumerate(maps):
        expected = {entry.id for entry in mapping.source.entries}
        actual = {entry.id for entry in mapping.entries}
        covered = len(expected & actual)
        expected_count += len(expected)
        covered_count += covered
        occurrences = sum(len(entry.occurrences) for entry in mapping.entries)
        occurrence_count += occurrences
        documents = {document.id: document for document in mapping.source.documents}
        rows = {"changes": [], "preserved": []}
        for entry in mapping.entries:
            group = "preserved" if entry.operation == "preserve" else "changes"
            rows[group].append(
                _entry_row(entry, mapping, documents, f"entry-{entry_count:06d}")
            )
            entry_count += 1
        binding_rows = []
        for binding in mapping.prose_bindings:
            binding_rows.append(
                _binding_row(binding, mapping, f"entry-{entry_count:06d}")
            )
            entry_count += 1
        gaps = []
        if expected - actual:
            gaps.append("Missing mappings: " + ", ".join(sorted(expected - actual)))
        if actual - expected:
            gaps.append(
                "Unknown source entries: " + ", ".join(sorted(actual - expected))
            )
        if len(actual) != len(mapping.entries):
            gaps.append("Duplicate mapping IDs are present.")
        gap_html = "".join(f'<p class="warning">{html.escape(gap)}</p>' for gap in gaps)
        source_rows = "".join(
            f"<tr><td>{html.escape(document.id)}</td><td>{html.escape(document.kind)}</td>"
            f"<td><code>{html.escape(document.source_path)}</code></td>"
            f"<td><code>{html.escape(document.sha256)}</code></td></tr>"
            for document in mapping.source.documents
        )
        notes = "".join(
            f"<li>{html.escape(note)}</li>" for note in mapping.settings.notes
        )
        settings = mapping.settings
        name_order = settings.model_dump(mode="json")["name_order"]
        changes = (
            f"<section data-entry-group><h3>Adapted / translated values ({len(rows['changes'])})</h3>{_table(rows['changes'])}</section>"
            if rows["changes"]
            else ""
        )
        preserved = (
            f'<details class="preserved" data-entry-group><summary>Preserved values ({len(rows["preserved"])}; expand to review)</summary>{_table(rows["preserved"])}</details>'
            if rows["preserved"]
            else ""
        )
        compiled_details = _compiled_name_details(mapping, index)
        bindings = (
            f"<section data-entry-group><h3>Prose references ({len(binding_rows)}; review individually)</h3>"
            "<p>These source-bound names, aliases, and protected text also require review. "
            "They do not add to the whole-entry source coverage count.</p>"
            f"{_table(binding_rows)}</section>"
            if binding_rows
            else ""
        )
        if compiled_details:
            compiled_links.append(
                f'<a href="#compiled-name-details-{index}">'
                f"{html.escape(settings.language)} / retail caller instructions</a>"
            )
        sections.append(
            f'<article id="map-{index}"><h2>{html.escape(settings.language)} / {html.escape(mapping.source.domain)} '
            '<span class="badge">Draft</span></h2>'
            f"<p>Revision: <code>{html.escape(mapping.revision)}</code>. "
            f"Source entries covered: <strong>{covered} / {len(expected)}</strong>; "
            f"{occurrences} mapped occurrences.</p>{gap_html}"
            f"<p>Country: {html.escape(settings.country_code)} · Currency: {html.escape(settings.currency_code)} "
            f"· Fixed benchmark scale: {html.escape(settings.currency_scale)} "
            f"· Time zone: {html.escape(settings.time_zone)} "
            f"· Name order: {html.escape(name_order.replace('_', ' '))}.</p>"
            f'<ul class="notes">{notes}</ul>'
            "<details><summary>Source snapshots and generation provenance</summary>"
            f"<p>Source digest: <code>{json_content_digest(mapping.source.model_dump(mode='json'))}</code></p>"
            f"<p>Map digest: <code>{json_content_digest(mapping.model_dump(mode='json'))}</code></p>"
            f"{_pre(mapping.provenance.model_dump(mode='json'))}"
            f"{_pre(mapping.source_refresh.model_dump(mode='json')) if mapping.source_refresh else ''}"
            f"{_pre(mapping.email_revision.model_dump(mode='json')) if mapping.email_revision else ''}"
            f"{_pre(mapping.user_id_revision.model_dump(mode='json')) if mapping.user_id_revision else ''}"
            f"{_pre(mapping.caller_name_revision.model_dump(mode='json')) if mapping.caller_name_revision else ''}"
            '<div class="table-scroll"><table><thead><tr><th>Document</th><th>Kind</th>'
            f"<th>Source path</th><th>Content SHA256</th></tr></thead><tbody>{source_rows}</tbody></table></div></details>"
            f"{compiled_details}{bindings}{changes}{preserved}</article>"
        )
    filters = (
        _select("language", "Language", {item.settings.language for item in maps})
        + _select("domain", "Domain", {item.source.domain for item in maps})
        + _select(
            "category",
            "Category",
            {entry.category for item in maps for entry in item.entries}
            | (
                {"prose_reference"}
                if any(item.prose_bindings for item in maps)
                else set()
            ),
        )
        + _select(
            "operation",
            "Change",
            {entry.operation for item in maps for entry in item.entries}
            | {
                "preserve" if binding.kind == "protected" else "adapt"
                for item in maps
                for binding in item.prose_bindings
            },
        )
    )
    locales = sorted(
        {f"{item.settings.language}-{item.settings.country_code}" for item in maps}
    )
    locale_label = (
        locales[0] if len(locales) == 1 else "Multiple locales: " + ", ".join(locales)
    )
    title = html.escape(f"{locale_label} — Regional localization review")
    compiled_navigation = (
        f"<p>Compiled task previews: {' · '.join(compiled_links)}</p>"
        if compiled_links
        else ""
    )
    page = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{title}</title><style>" + _STYLE + "</style></head><body>"
        f"<header><h1>{title}</h1>"
        '<p class="notice"><strong>Draft — awaiting human review.</strong> '
        "Generated proposals and coverage counts are not human approval or proof of native quality.</p>"
        f"<p><strong>{covered_count} / {expected_count}</strong> source entries covered · "
        f"{entry_count} mapping rows · {occurrence_count} mapped occurrences.</p>"
        "<p>Native values are intended for localized runtime data. Romanization records the same value in Latin script; "
        "English meaning explains it for review. Preserved groups are collapsed initially, with every row retained.</p>"
        f"{compiled_navigation}"
        '<div class="filters"><label>Search all values and source paths'
        '<input id="search" type="search" placeholder="Name, value, explanation, or path"></label>'
        f'{filters}<button type="button" id="reset">Reset filters</button></div>'
        f'<p id="matches" role="status">{entry_count} / {entry_count} matching entries.</p></header>'
        f"<main>{''.join(sections)}</main><script>{_SCRIPT}</script></body></html>"
    )
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "index.html"
    destination.write_text(page, encoding="utf-8")
    return destination


_STYLE = """
:root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#f4f6f8;color:#182333;font:15px/1.5 system-ui,sans-serif}
header{padding:24px 30px;background:#fff;border-bottom:1px solid #ccd3dc}h1{font-size:26px;margin:0 0 12px}h2{font-size:21px}h3{font-size:17px}h4{font-size:14px;margin:14px 0 4px}
main{padding:20px 30px}article{background:#fff;padding:22px;margin:0 0 24px;border:1px solid #d7dde5;border-radius:8px}p{margin:8px 0}.notice,.warning{background:#fff3d7;padding:10px;border-left:3px solid #bd7a16}.badge{font-size:13px;background:#fff0c9;border-radius:4px;padding:3px 8px;margin-left:8px}
.filters{display:flex;flex-wrap:wrap;gap:12px;align-items:end;margin-top:18px}label{display:flex;flex-direction:column;gap:4px;font-weight:600}input,select,button{font:inherit;padding:8px;border:1px solid #adb7c5;border-radius:4px;background:white}input{width:330px;max-width:100%}button{cursor:pointer}
.table-scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;table-layout:fixed;margin:12px 0;min-width:1050px}td,th{border:1px solid #dce1e8;padding:12px;vertical-align:top;text-align:left;overflow-wrap:anywhere}thead{background:#eef2f6}tbody th{font-weight:400}pre{white-space:pre-wrap;overflow-wrap:anywhere;margin:0;font:13px/1.6 ui-monospace,monospace}code{font:13px ui-monospace,monospace;overflow-wrap:anywhere}details{margin:14px 0}summary{cursor:pointer;font-weight:600}a{color:#155a9a}ul{padding-left:22px}li{margin:6px 0}[hidden]{display:none!important}.notes{max-width:1100px}
@media(max-width:700px){header,main{padding:16px}article{padding:12px}.filters label{width:100%}input{width:100%}}
"""

_SCRIPT = """
const rows = Array.from(document.querySelectorAll('tr.entry'));
const search = document.getElementById('search');
const selectors = ['language','domain','category','operation'].map(name => document.getElementById(name));
const searchable = new Map(rows.map(row => [row, row.textContent.toLocaleLowerCase()]));
function filterEntries() {
  const query = search.value.trim().toLocaleLowerCase();
  let count = 0;
  for (const row of rows) {
    const matches = selectors.every(select => !select.value || row.dataset[select.id] === select.value)
      && (!query || searchable.get(row).includes(query));
    row.hidden = !matches;
    count += Number(matches);
  }
  for (const group of document.querySelectorAll('[data-entry-group]')) {
    group.hidden = !Array.from(group.querySelectorAll('tr.entry')).some(row => !row.hidden);
    if (group.classList.contains('preserved') && (query || selectors.some(select => select.value))) {
      group.open = true;
    }
  }
  document.getElementById('matches').textContent = `${count} / ${rows.length} matching entries. Preserved groups can be expanded below.`;
}
search.addEventListener('input', filterEntries);
selectors.forEach(select => select.addEventListener('change', filterEntries));
document.getElementById('reset').addEventListener('click', () => {
  search.value = '';
  selectors.forEach(select => { select.value = ''; });
  document.querySelectorAll('details.preserved').forEach(group => { group.open = false; });
  filterEntries();
});
function revealEntry() {
  const target = document.getElementById(location.hash.slice(1));
  if (target?.matches('details[id^="compiled-name-details-"]')) {
    target.open = true;
    target.scrollIntoView({block:'start'});
    return;
  }
  if (!target || !target.classList.contains('entry')) return;
  search.value = '';
  selectors.forEach(select => { select.value = ''; });
  filterEntries();
  const group = target.closest('details.preserved');
  if (group) group.open = true;
  target.scrollIntoView({block:'start'});
}
window.addEventListener('hashchange', revealEntry);
revealEntry();
"""
