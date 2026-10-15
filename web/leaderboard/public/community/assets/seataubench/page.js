import {
  defaults,
  LANGUAGES,
  SETTINGS,
  METRICS,
  readState,
  writeState,
  selectRows,
  rankingHtml,
  detailHtml,
  csv,
} from './leaderboard-core.js'

const toggle = document.querySelector('.mobile-menu-toggle')
const nav = document.querySelector('#site-nav')
toggle.addEventListener('click', () => {
  const open = toggle.getAttribute('aria-expanded') !== 'true'
  toggle.setAttribute('aria-expanded', String(open))
  nav.classList.toggle('open', open)
})
nav.addEventListener('click', (e) => {
  if (e.target.closest('a')) {
    nav.classList.remove('open')
    toggle.setAttribute('aria-expanded', 'false')
  }
})
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && nav.classList.contains('open')) {
    nav.classList.remove('open')
    toggle.setAttribute('aria-expanded', 'false')
    toggle.focus()
  }
})
// Keep shared anchors useful when their content lives inside a disclosure.
function revealHashTarget() {
  const target = document.getElementById(location.hash.slice(1))
  if (!target) return
  let element = target
  while (element) {
    if (element.tagName === 'DETAILS') element.open = true
    element = element.parentElement
  }
}
revealHashTarget()
window.addEventListener('hashchange', revealHashTarget)

async function initLeaderboard() {
  const response = await fetch(new URL('./results.json', import.meta.url))
  if (!response.ok) throw new Error('Results unavailable')
  const { rows } = await response.json()
  let state = readState(location.search)
  const form = document.querySelector('#leaderboard-controls')
  const language = form.elements.language
  const targetOptions = language.innerHTML
  const modelInputs = [...form.querySelectorAll('input[name="model"]')]
  function render(updateUrl = false) {
    const selected = selectRows(rows, state)
    form.elements.setting.value = state.setting
    form.elements.domain.value = state.domain
    form.elements.metric.value = state.metric
    language.innerHTML =
      state.setting === 's1'
        ? '<option value="en">English baseline</option>'
        : targetOptions
    language.disabled = state.setting === 's1'
    language.value = state.setting === 's1' ? 'en' : state.language
    modelInputs.forEach((input) => {
      input.checked = state.models.includes(input.value)
    })
    document.querySelector('#language-help').textContent =
      state.setting === 's1'
        ? 'S1 is English only. Your target-language selection is retained.'
        : 'Language of the localized surface; S3 dialogue remains English.'
    document.querySelector('#ranking-body').innerHTML = rankingHtml(
      selected,
      state.metric,
    )
    document.querySelector('#detail-body').innerHTML = detailHtml(selected)
    document
      .querySelectorAll('#ranking-table thead th[data-metric]')
      .forEach((th) => {
        th.classList.toggle(
          'selected-metric',
          th.dataset.metric === state.metric,
        )
      })
    const langLabel =
      state.setting === 's1'
        ? 'English'
        : state.language === 'all'
          ? '5 target languages'
          : LANGUAGES[state.language]
    const domainLabel =
      state.domain === 'all'
        ? '3 domains'
        : state.domain[0].toUpperCase() + state.domain.slice(1)
    document.querySelector('#selection-summary').textContent =
      `${SETTINGS[state.setting]} · ${langLabel} · ${domainLabel}`
    document.querySelector('#result-status').textContent =
      `${state.models.length} models · ${selected.length} published results · ranked by ${METRICS[state.metric]}`
    document.querySelector('#detail-count').textContent = selected.length
    document.querySelector('#s3-note').hidden = state.setting !== 's3'
    document.querySelector('#download-csv').disabled = selected.length === 0
    if (updateUrl)
      history.replaceState(
        null,
        '',
        `${location.pathname}?${writeState(state, location.search)}${location.hash}`,
      )
  }
  form.addEventListener('change', () => {
    // Keep the target language while the S1 control temporarily displays English.
    if (state.setting !== 's1') state.language = language.value
    state.setting = form.elements.setting.value
    state.domain = form.elements.domain.value
    state.metric = form.elements.metric.value
    state.models = modelInputs.filter((i) => i.checked).map((i) => i.value)
    render(true)
  })
  form.addEventListener('submit', (e) => e.preventDefault())
  document.querySelector('#reset-filters').addEventListener('click', () => {
    state = defaults()
    render(true)
  })
  document.querySelector('#download-csv').addEventListener('click', () => {
    const blob = new Blob([csv(selectRows(rows, state))], {
      type: 'text/csv;charset=utf-8;',
    })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `seataubench-${state.setting}-${state.language}-${state.domain}.csv`
    a.click()
    setTimeout(() => URL.revokeObjectURL(url), 1000)
  })
  window.addEventListener('popstate', () => {
    state = readState(location.search)
    render()
  })
  render()
  form.hidden = false
  document.querySelector('#download-csv').hidden = false
}
initLeaderboard().catch(() => {
  document.querySelector('#result-status').textContent =
    'Interactive filters could not load. The verified default ranking and published results remain available below.'
})
