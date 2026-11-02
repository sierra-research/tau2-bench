// τ-i18n view built from submitted voice-v2 packages: the 3D explorer, the
// language map, the leaderboard table and its breakdowns, filters and the
// homepage card. Packages are synthetic and served through request
// interception, so the committed manifest stays untouched.
import assert from 'node:assert/strict'
import { test, expect } from '@playwright/test'
import { readFile } from 'node:fs/promises'

const source = await readFile(new URL('../src/data/multilingual/metrics.js', import.meta.url), 'utf8')
const { frontier } = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`)

const languages = ['en', 'es', 'hi', 'ko', 'pt', 'zh']
const domains = ['airline', 'retail', 'telecom']
const components = ['nonresponse', 'interruption', 'selectivity', 'monologue', 'tool_misuse']
const mean = values => values.reduce((a, b) => a + b, 0) / values.length
const percent = value => `${value.toFixed(1)}%`

const metadata = (model, organization, provider, submission_type = 'standard') => ({
  schema_version: 'benchmark-metadata-v1', model_name: model, model_organization: organization, submitting_organization: organization,
  submission_date: '2026-10-20', contact_info: { email: 'a@example.org' }, submission_type, references: [], notes: null,
  agent_provider: provider, agent_model: model, evidence: 'matrix', trajectories: null,
})
const coverage = group => ({ expected_trials: group.length, scored_trials: group.length, error_trials: 0, unscored_trials: 0, missing_trials: 0, complete: true, score: mean(group.map(cell => cell.score)), observed_score: mean(group.map(cell => cell.score)) })
const submission = score => {
  const cells = languages.flatMap((language, l) => domains.map((domain, d) => ({ language, domain, score: score(l, d) })))
  return {
    frame: { benchmark_version: 'voice-v2', revision: 'voice-v2-rev', cells: cells.map(({ language, domain }) => ({ language, domain })) },
    summary: {
      benchmark_version: 'voice-v2', frame_revision: 'voice-v2-rev', complete: true, scored_trials: 50 * cells.length, score: mean(cells.map(cell => cell.score)),
      cells: cells.map(cell => ({ ...cell, complete: true })),
      by_language: Object.fromEntries(languages.map(language => [language, coverage(cells.filter(cell => cell.language === language))])),
      by_domain: Object.fromEntries(domains.map(domain => [domain, coverage(cells.filter(cell => cell.domain === domain))])),
    },
  }
}
// Each cell has 50 calls; `pass(l, d)` scored calls pass, two are unscored.
const quality = (interactionPass, experiencePass) => ({
  schema_version: 'benchmark-quality-v1', benchmark_version: 'voice-v2', frame_revision: 'voice-v2-rev',
  interaction_version: 'voice-v2-interaction-call-v1', experience_version: 'voice-v2-experience-call-v2',
  cells: languages.flatMap((language, l) => domains.map((domain, d) => ({
    language, domain,
    interaction: { passed: interactionPass(l, d), failed: 48 - interactionPass(l, d), unscored: 2, excluded: 0 },
    experience: { passed: experiencePass(l, d), failed: 48 - experiencePass(l, d), unscored: 2, excluded: 0 },
    interaction_components: Object.fromEntries(components.map((key, c) => [key, { flagged: c + l, known: 50 }])),
    calls: [],
  }))),
})
const packages = {
  fixture_a: { metadata: metadata('alpha-live', 'Alpha Labs', 'openai'), submission: submission((l, d) => (60 + 2 * l + 5 * d) / 100), quality: quality((l, d) => 30 + l + d, l => 40 - l) },
  fixture_b: { metadata: metadata('beta-live', 'Beta AI', 'gemini'), submission: submission((l, d) => (50 + l + d) / 100), quality: quality(() => 44, (l, d) => 20 + d) },
  fixture_c: { metadata: metadata('gamma-live', 'Gamma Inc', 'xai'), submission: submission(() => 0.4), quality: null },
  fixture_custom: { metadata: metadata('delta-live', 'Delta', 'xai', 'custom'), submission: submission(() => 0.9), quality: null },
}
const files = { 'benchmark-metadata.json': 'metadata', 'benchmark-submission.json': 'submission', 'benchmark-quality.json': 'quality' }
const pooled = (pkg, measure, keep = () => true) => {
  const cells = pkg.quality.cells.filter(keep)
  const passed = cells.reduce((sum, cell) => sum + cell[measure].passed, 0)
  return 100 * passed / cells.reduce((sum, cell) => sum + cell[measure].passed + cell[measure].failed, 0)
}

test.beforeEach(async ({ page }) => {
  await page.route('**/voice-leaderboard.json', route => route.fulfill({ status: 404, body: '' }))
  await page.route('**/submissions/manifest.json', async route => {
    const manifest = await (await route.fetch()).json()
    await route.fulfill({ json: { ...manifest, voice_v2_submissions: Object.keys(packages), voice_v2_domain_submissions: [] } })
  })
  await page.route('**/submissions/fixture_*/*', route => {
    const [dir, file] = route.request().url().split('/').slice(-2)
    const json = files[file] && packages[dir][files[file]]
    return json ? route.fulfill({ json }) : route.fulfill({ status: 404, body: '' })
  })
})
test.afterEach(async ({ page }) => {
  await page.unrouteAll({ behavior: 'ignoreErrors' })
})

test('Pareto frontier', () => {
  const points = [{ id: 'a', x: 1, y: 3, z: 1 }, { id: 'b', x: 2, y: 4, z: 0 }, { id: 'c', x: null, y: 5, z: 5 }, { id: 'tie', x: 2, y: 4, z: 0 }]
  assert.deepEqual(frontier(points, ['x', 'y']).map(p => p.id), ['b', 'tie'])
  assert.deepEqual(frontier(points, ['x', 'y', 'z']).map(p => p.id), ['a', 'b', 'tie'])
})

test('package-driven explorer, breakdowns, language map and homepage', async ({ page }, testInfo) => {
  test.setTimeout(90000)
  await page.setViewportSize({ width: 1440, height: 1100 })
  const errors = []
  page.on('pageerror', error => errors.push(error.message))
  await page.goto('/leaderboard?benchmark=voice', { waitUntil: 'networkidle' })
  await page.getByRole('heading', { name: /Three dimensions of a good conversation/ }).waitFor()
  const rows = page.locator('.mv-scores .model-row')
  await expect(rows).toHaveCount(3)
  assert.equal(await page.locator('.mv-scores > .metrics-table-container > table > thead th').count(), 7)
  assert.equal(await page.locator('.mv-score-button').count(), 9)
  // Only packages with all three scores enter the cube.
  assert.equal(await page.locator('.mv-cube-point').count(), 2)
  await page.locator('.mv-main-panel').screenshot({ path: testInfo.outputPath('tau-i18n-cube.png') })

  const a = packages.fixture_a
  const rowA = page.locator('tr[data-package="fixture_a"]')
  await expect(rowA.locator('td[data-metric="task"]')).toContainText(percent(100 * a.submission.summary.score))
  await expect(rowA.locator('td[data-metric="interaction"]')).toContainText(percent(pooled(a, 'interaction')))
  await expect(rowA.locator('td[data-metric="experience"]')).toContainText(percent(pooled(a, 'experience')))
  await expect(page.locator('tr[data-package="fixture_c"] td[data-metric="interaction"]')).toHaveText('—')

  // Cube gestures and focus.
  const cube = await page.locator('.mv-cube').boundingBox()
  const geometry = () => page.locator('.mv-cube polygon').first().getAttribute('points')
  const initialGeometry = await geometry()
  await page.mouse.move(cube.x + 30, cube.y + 30)
  await page.mouse.down()
  await page.mouse.move(cube.x + 100, cube.y + 30, { steps: 5 })
  await page.mouse.up()
  assert.notEqual(await geometry(), initialGeometry)
  await page.getByRole('button', { name: 'Reset view' }).click()
  assert.equal(await geometry(), initialGeometry)
  await page.getByRole('button', { name: /^alpha-live, task/ }).click()
  assert.equal(await page.locator('.mv-cube-point.is-muted').count(), 1)
  await page.getByRole('button', { name: 'Show all models', exact: true }).click()
  assert.equal(await page.locator('.mv-cube-point.is-muted').count(), 0)

  // Breakdown tabs.
  await page.getByRole('button', { name: 'alpha-live: Interaction breakdown', exact: true }).click()
  const detail = page.locator('.mv-detail')
  await expect(detail.locator('.mv-breakdown-tabs button')).toHaveCount(3)
  const overall = detail.locator('.mv-pass-heatmap tr[data-factor="all"]')
  await expect(overall.locator('td[data-language="all"]')).toHaveText(percent(pooled(a, 'interaction')))
  await expect(overall.locator('td[data-language="ko"]')).toHaveText(percent(pooled(a, 'interaction', cell => cell.language === 'ko')))
  await expect(detail.locator('.mv-pass-heatmap tr[data-factor="retail"] td[data-language="all"]')).toHaveText(percent(pooled(a, 'interaction', cell => cell.domain === 'retail')))
  await expect(detail.locator('.mv-failure-heatmap tr[data-factor="monologue"] td[data-language="hi"]')).toHaveText(percent(100 * (3 + 2) / 50))
  await detail.screenshot({ path: testInfo.outputPath('tau-i18n-interaction.png') })
  await detail.locator('.mv-breakdown-tabs').getByRole('button', { name: /Generation quality/ }).click()
  await expect(detail.locator('.mv-pass-heatmap tr[data-factor="all"] td[data-language="all"]')).toHaveText(percent(pooled(a, 'experience')))
  await detail.locator('.mv-breakdown-tabs').getByRole('button', { name: /Task completion/ }).click()
  await expect(detail.locator('.mv-task-heatmap tr[data-factor="all"] td[data-language="all"]')).toHaveText(percent(100 * a.submission.summary.score))
  await page.getByRole('button', { name: 'Close breakdown', exact: true }).click()
  await expect(page.locator('.mv-detail')).toHaveCount(0)

  // A domain scopes all three leaderboard scores; the 3D explorer stays on Overall.
  const pointA = page.locator('.mv-cube-point').first()
  const overallPoint = await pointA.getAttribute('aria-label')
  await page.locator('.mv-domain-toggle').getByRole('button', { name: /Airline/ }).click()
  await expect(rowA.locator('td[data-metric="interaction"]')).toContainText(percent(pooled(a, 'interaction', cell => cell.domain === 'airline')))
  await expect(page.locator('.mv-core-note')).toHaveText('Core domains only · Airline, Retail, Telecom')
  assert.equal(await page.locator('.mv-cube-point').count(), 2)
  assert.equal(await pointA.getAttribute('aria-label'), overallPoint)
  await page.locator('.mv-domain-toggle').getByRole('button', { name: /Banking/ }).click()
  await expect(page.locator('.mv-main-panel .mv-cube-empty')).toHaveCount(0)
  assert.equal(await page.locator('.mv-cube-point').count(), 2)
  // Banking is scored on task completion only: its other columns are hidden.
  await expect(page.locator('.mv-scores > .metrics-table-container > table > thead th')).toHaveCount(5)
  await expect(rowA.locator('td[data-metric]')).toHaveCount(1)
  await page.locator('.mv-domain-toggle').getByRole('button', { name: /Overall/ }).click()
  await expect(page.locator('.mv-scores > .metrics-table-container > table > thead th')).toHaveCount(7)
  await page.getByRole('button', { name: 'Generation quality ↑' }).click()
  await expect(rows.first()).toContainText('alpha-live')

  // Language map.
  await page.getByRole('button', { name: 'Language map', exact: true }).click()
  await expect(page.locator('.mv-heatmap tbody tr')).toHaveCount(4)
  assert.equal(await page.locator('.mv-heatmap thead th').count(), 8)
  await page.locator('.mv-map-metrics').getByRole('button', { name: 'Interaction', exact: true }).click()
  await expect(page.locator('.mv-heatmap [data-model="fixture_b"] [data-language="es"]')).toHaveText(percent(100 * 44 / 48))
  await expect(page.locator('.mv-heatmap [data-model="fixture_c"] [data-language="all"]')).toHaveText('—')
  await page.locator('.mv-main-panel').screenshot({ path: testInfo.outputPath('tau-i18n-language-map.png') })
  assert.deepEqual(await page.locator('.mv-paper-links a[href^="https://arxiv.org/abs/"]').allInnerTexts(), ['τ-i18n ↗', 'τ-Entity ↗', 'τ-Voice ↗'])

  // Mobile.
  await page.getByRole('button', { name: '3D explorer', exact: true }).click()
  await page.setViewportSize({ width: 390, height: 844 })
  await page.getByRole('button', { name: 'alpha-live: Generation quality breakdown', exact: true }).click()
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth))

  // Homepage card: standard packages by task completion.
  await page.setViewportSize({ width: 1440, height: 1100 })
  await page.goto('/')
  const card = page.locator('.preview-table-wrapper').filter({ hasText: 'All languages' })
  await expect(card.locator('.preview-model-name')).toHaveText(['alpha-live', 'beta-live', 'gamma-live'])
  await card.click()
  await page.getByRole('heading', { name: /Three dimensions of a good conversation/ }).waitFor()
  assert.deepEqual(errors, [])
})

test('multilingual submission filters and English progress remain available', async ({ page }) => {
  await page.goto('/leaderboard?benchmark=voice')
  await expect(page.getByRole('heading', { name: /Three dimensions of a good conversation/ })).toBeVisible()
  const filters = page.locator('.mv-leaderboard-heading')
  await expect(filters.getByRole('heading', { name: 'Leaderboard', exact: true })).toBeVisible()
  await expect(page.locator('.mv-scores .model-row')).toHaveCount(3)
  const toggle = name => filters.locator('.checkbox-container').filter({ hasText: new RegExp(`^${name}$`) }).click()
  await toggle('Standard')
  await expect(page.locator('.mv-scores .model-row')).toHaveCount(0)
  await expect(page.getByText('Select Standard or Custom to view results.')).toBeVisible()
  await toggle('Custom')
  await expect(page.locator('.mv-scores .model-row')).toHaveCount(1)
  await expect(page.locator('tr[data-package="fixture_custom"]')).toContainText('custom submission')
  await toggle('Standard')
  await expect(page.locator('.mv-scores .model-row')).toHaveCount(4)
  await page.getByRole('button', { name: 'English', exact: true }).click()
  await expect(page.locator('.mv')).toHaveCount(0)
  await expect(page.locator('.domain-toggle-switch')).toBeVisible()
  await page.goto('/progress?benchmark=voice')
  await expect(page.getByRole('button', { name: 'English', exact: true })).toHaveAttribute('aria-pressed', 'true')
  await expect(page.locator('.mv')).toHaveCount(0)
})
