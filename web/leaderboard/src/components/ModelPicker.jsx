import { useEffect, useMemo, useRef, useState } from 'react'
import { DOMAINS, formatEffort } from '../data/visualizerIndex'
import './ModelPicker.css'

const SORTS = [
  { id: 'score', label: 'Score' },
  { id: 'date', label: 'Newest' },
  { id: 'name', label: 'A–Z' },
]

const sorters = {
  score: (a, b) => (b.score ?? -1) - (a.score ?? -1) || a.model.localeCompare(b.model),
  date: (a, b) => (b.date || '').localeCompare(a.date || '') || a.model.localeCompare(b.model),
  name: (a, b) => a.model.localeCompare(b.model) || (b.date || '').localeCompare(a.date || ''),
}

// Each search term must start a word of the name, organization, effort or
// type, so "realtime 2" finds gpt-realtime-2 without matching every date.
const searchWords = e => [e.model, e.org, e.effort, e.submissionType]
  .filter(Boolean).join(' ').toLowerCase().split(/[\s\-_.()/]+/).filter(Boolean)
const matches = (e, terms) => {
  const words = searchWords(e)
  return terms.every(t => words.some(w => w.startsWith(t)) || e.model.toLowerCase().includes(t))
}

// Effort, custom setup and user simulator: what tells apart same-named rows.
export function EntryChips({ entry }) {
  return (
    <>
      {entry.effort && <span className="mp-chip">{formatEffort(entry.effort)}</span>}
      {entry.submissionType === 'custom' && (
        <span className="mp-chip mp-chip-custom" title={entry.userSimulator ? `User simulator: ${entry.userSimulator}` : 'Custom setup'}>
          Custom{entry.userSimulator ? ` · sim ${entry.userSimulator.replace(/-\d{4}-\d{2}-\d{2}/, '')}` : ''}
        </span>
      )}
    </>
  )
}

function EntryRow({ entry, active, selected, onSelect, onHover }) {
  const domains = [...new Set(entry.cells.map(c => c.domain))]
  const languages = new Set(entry.cells.map(c => c.language)).size
  return (
    <li
      role="option"
      aria-selected={selected}
      className={`mp-row ${active ? 'is-active' : ''} ${selected ? 'is-selected' : ''}`}
      onMouseDown={e => { e.preventDefault(); onSelect(entry) }}
      onMouseEnter={onHover}
      data-entry={entry.id}
    >
      <div className="mp-row-main">
        <div className="mp-row-title">
          <span className="mp-model">{entry.model}</span>
          <EntryChips entry={entry} />
        </div>
        <div className="mp-row-meta">
          <span>{entry.org}</span>
          {entry.date && <span>{entry.date}</span>}
          <span className="mp-domains" title={domains.map(d => DOMAINS[d]?.label || d).join(', ')}>
            {domains.map(d => <span key={d}>{DOMAINS[d]?.icon || '•'}</span>)}
          </span>
          {languages > 1 && <span>{languages} languages</span>}
        </div>
      </div>
      <div className="mp-score" title={entry.scoreLabel}>
        {entry.score != null ? `${entry.score.toFixed(1)}%` : '—'}
      </div>
    </li>
  )
}

/**
 * Searchable, sortable list of a benchmark's submissions. `inline` renders the
 * list in place (the landing state); otherwise a button opens it as a popover.
 */
export default function ModelPicker({ entries, selected, onSelect, inline = false }) {
  const [open, setOpen] = useState(false)
  const [query, setQuery] = useState('')
  const [sort, setSort] = useState('score')
  const [active, setActive] = useState(0)
  const rootRef = useRef(null)
  const inputRef = useRef(null)
  const listRef = useRef(null)

  const shown = useMemo(() => {
    const terms = query.toLowerCase().split(/\s+/).filter(Boolean)
    return entries
      .filter(e => matches(e, terms))
      .sort(sorters[sort])
  }, [entries, query, sort])

  const expanded = inline || open
  useEffect(() => { setActive(0) }, [query, sort, expanded])

  useEffect(() => {
    if (!open) return
    inputRef.current?.focus()
    const onDown = e => { if (!rootRef.current?.contains(e.target)) setOpen(false) }
    document.addEventListener('mousedown', onDown)
    return () => document.removeEventListener('mousedown', onDown)
  }, [open])

  useEffect(() => {
    listRef.current?.querySelector('.mp-row.is-active')?.scrollIntoView({ block: 'nearest' })
  }, [active])

  const choose = entry => {
    onSelect(entry)
    setOpen(false)
    setQuery('')
  }

  const onKeyDown = e => {
    if (e.key === 'ArrowDown') { e.preventDefault(); setActive(i => Math.min(i + 1, shown.length - 1)) }
    else if (e.key === 'ArrowUp') { e.preventDefault(); setActive(i => Math.max(i - 1, 0)) }
    else if (e.key === 'Enter' && shown[active]) { e.preventDefault(); choose(shown[active]) }
    else if (e.key === 'Escape' && !inline) { setOpen(false) }
  }

  const panel = (
    <div className={`mp-panel ${inline ? 'mp-inline' : 'mp-popover'}`}>
      <div className="mp-toolbar">
        <input
          ref={inputRef}
          className="mp-search"
          type="search"
          placeholder={`Search ${entries.length} submissions — model, org, effort…`}
          value={query}
          onChange={e => setQuery(e.target.value)}
          onKeyDown={onKeyDown}
          aria-label="Search submissions"
        />
        <div className="mp-sort" role="group" aria-label="Sort by">
          {SORTS.map(s => (
            <button key={s.id} className={sort === s.id ? 'active' : ''} aria-pressed={sort === s.id} onClick={() => setSort(s.id)}>{s.label}</button>
          ))}
        </div>
      </div>
      <ul className="mp-list" role="listbox" ref={listRef}>
        {shown.map((entry, i) => (
          <EntryRow
            key={entry.id}
            entry={entry}
            active={i === active}
            selected={selected?.id === entry.id}
            onSelect={choose}
            onHover={() => setActive(i)}
          />
        ))}
        {!shown.length && <li className="mp-empty">No submissions match “{query}”.</li>}
      </ul>
    </div>
  )

  if (inline) return <div className="model-picker" ref={rootRef}>{panel}</div>

  return (
    <div className="model-picker" ref={rootRef}>
      <button className="mp-trigger" onClick={() => setOpen(o => !o)} aria-expanded={open} aria-haspopup="listbox">
        <span className="mp-trigger-label">
          {selected ? <><span className="mp-model">{selected.model}</span> <EntryChips entry={selected} /> <span className="mp-trigger-org">{selected.org}</span></> : 'Choose a model'}
        </span>
        <span className="mp-caret" aria-hidden="true">▾</span>
      </button>
      {open && panel}
    </div>
  )
}
