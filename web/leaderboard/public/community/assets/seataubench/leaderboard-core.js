export const MODELS = {
  'gpt-5-mini': 'GPT-5-mini',
  'qwen3-235b': 'Qwen3-235B',
  'kimi-k2.5': 'Kimi-K2.5',
}
export const LANGUAGES = {
  vi: 'Vietnamese',
  th: 'Thai',
  id: 'Indonesian',
  zh: 'Mandarin Chinese',
  tl: 'Filipino',
  en: 'English',
}
export const DOMAINS = {
  airline: 'Airline',
  retail: 'Retail',
  telecom: 'Telecom',
}
export const SETTINGS = {
  s1: 'English baseline',
  s2: 'L2 interaction',
  s3: 'L2 tool',
  s4: 'Fully localized',
}
export const METRICS = {
  pass1: 'pass@1',
  pass2: 'pass²',
  pass3: 'pass³',
  rho3: 'ρ³',
}
export const defaults = () => ({
  setting: 's4',
  language: 'all',
  domain: 'all',
  models: Object.keys(MODELS),
  metric: 'pass1',
})
export const escapeHtml = (value) =>
  String(value).replace(
    /[&<>"']/g,
    (c) =>
      ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[
        c
      ],
  )
export const percent = (value) => `${(value * 100).toFixed(1)}%`
export function readState(search) {
  const p = new URLSearchParams(search)
  const state = defaults()
  if (Object.hasOwn(SETTINGS, p.get('setting')))
    state.setting = p.get('setting')
  if (
    p.get('language') === 'all' ||
    (Object.hasOwn(LANGUAGES, p.get('language')) && p.get('language') !== 'en')
  )
    state.language = p.get('language')
  if (Object.hasOwn(DOMAINS, p.get('domain'))) state.domain = p.get('domain')
  if (Object.hasOwn(METRICS, p.get('metric'))) state.metric = p.get('metric')
  if (p.has('models'))
    state.models = [
      ...new Set(
        p
          .get('models')
          .split(',')
          .filter((m) => Object.hasOwn(MODELS, m)),
      ),
    ]
  return state
}
export function writeState(state, currentSearch = '') {
  const p = new URLSearchParams(currentSearch)
  for (const key of ['setting', 'language', 'domain', 'metric'])
    p.set(key, state[key])
  p.set('models', state.models.join(','))
  return p.toString()
}
export function selectRows(rows, state) {
  return rows.filter(
    (r) =>
      r.setting === state.setting &&
      state.models.includes(r.model) &&
      (state.setting === 's1'
        ? r.language === 'en'
        : state.language === 'all' || r.language === state.language) &&
      (state.domain === 'all' || r.domain === state.domain),
  )
}
export function aggregate(rows, metric = 'pass1') {
  const result = Object.keys(MODELS)
    .flatMap((model) => {
      const cells = rows.filter((r) => r.model === model)
      if (!cells.length) return []
      return [
        {
          model,
          cells: cells.length,
          ...Object.fromEntries(
            Object.keys(METRICS).map((key) => [
              key,
              cells.reduce((sum, r) => sum + r[key], 0) / cells.length,
            ]),
          ),
        },
      ]
    })
    .sort(
      (a, b) =>
        Math.round(b[metric] * 1000) - Math.round(a[metric] * 1000) ||
        MODELS[a.model].localeCompare(MODELS[b.model]),
    )
  let rank = 0
  result.forEach((row, index) => {
    if (
      index === 0 ||
      Math.round(row[metric] * 1000) !==
        Math.round(result[index - 1][metric] * 1000)
    )
      rank = index + 1
    row.rank = rank
  })
  return result
}
export function rankingHtml(rows, metric) {
  const results = aggregate(rows, metric)
  if (!results.length)
    return '<tr><td colspan="6" class="empty-state">Select at least one model to see results.</td></tr>'
  return results
    .map(
      (r) =>
        `<tr data-model="${r.model}"><td class="rank">${String(r.rank).padStart(2, '0')}</td><th scope="row"><span class="model-name"><i class="model-dot ${r.model}" aria-hidden="true"></i>${MODELS[r.model]}</span><span class="model-detail">${r.cells} published ${r.cells === 1 ? 'result' : 'results'}</span></th>${Object.keys(
          METRICS,
        )
          .map(
            (k) =>
              `<td class="score ${k === metric ? 'selected-metric' : ''}" ${k === metric ? `style="--score:${r[k] * 100}%"` : ''}>${percent(r[k])}</td>`,
          )
          .join('')}</tr>`,
    )
    .join('')
}
export function detailHtml(rows, mixed = false) {
  if (!rows.length)
    return '<tr><td colspan="8" class="empty-state">No models selected.</td></tr>'
  return rows
    .map(
      (r) =>
        `<tr><th scope="row">${MODELS[r.model]}</th><td>${mixed ? r.language.replace('mix-', 'Mix-') : LANGUAGES[r.language]}</td><td>${DOMAINS[r.domain]}</td>${Object.keys(
          METRICS,
        )
          .map((k) => `<td class="score">${percent(r[k])}</td>`)
          .join(
            '',
          )}<td><a href="https://arxiv.org/pdf/2606.28715v2#page=${r.sourceTable <= 19 ? 23 : 24}" target="_blank" rel="noopener noreferrer">Table ${r.sourceTable}</a></td></tr>`,
    )
    .join('')
}
export function csv(rows) {
  const keys = [
    'setting',
    'language',
    'domain',
    'model',
    'pass1',
    'pass2',
    'pass3',
    'rho3',
    'sourceTable',
  ]
  return (
    [
      keys.join(','),
      ...rows.map((r) =>
        keys
          .map((k) =>
            typeof r[k] === 'number' && k !== 'sourceTable'
              ? r[k].toFixed(3)
              : r[k],
          )
          .join(','),
      ),
    ].join('\r\n') + '\r\n'
  )
}
