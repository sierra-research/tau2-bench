// voice-v2-banking and voice-v2-entity packages listed under
// manifest.voice_v2_domain_submissions fill the Banking and Intake rows of the
// same agent's voice-v2 row without changing its Overall. Packages are
// synthetic and served through request interception, so the committed
// manifest stays untouched.
import { test, expect } from '@playwright/test'

const languages = ['en', 'es', 'hi', 'ko', 'pt', 'zh']
const mean = values => values.reduce((a, b) => a + b, 0) / values.length
const coverage = group => ({ expected_trials: group.length, scored_trials: group.length, error_trials: 0, unscored_trials: 0, missing_trials: 0, complete: true, score: mean(group.map(cell => cell.score)), observed_score: mean(group.map(cell => cell.score)) })
const metadata = (model, provider) => ({
  schema_version: 'benchmark-metadata-v1', model_name: model, model_organization: 'Fixture Labs', submitting_organization: 'Fixture Labs',
  submission_date: '2026-10-20', contact_info: { email: 'a@example.org' }, submission_type: 'standard', references: [], notes: null,
  agent_provider: provider, agent_model: model, evidence: 'matrix', trajectories: null,
})
const submission = (version, domains, score) => {
  const cells = languages.flatMap((language, l) => domains.map((domain, d) => ({ language, domain, score: score(l, d) })))
  return {
    frame: { benchmark_version: version, revision: `${version}-rev`, cells: cells.map(({ language, domain }) => ({ language, domain })) },
    summary: {
      benchmark_version: version, frame_revision: `${version}-rev`, complete: true, scored_trials: cells.length, score: mean(cells.map(cell => cell.score)),
      cells: cells.map(cell => ({ ...cell, complete: true })),
      by_language: Object.fromEntries(languages.map(language => [language, coverage(cells.filter(cell => cell.language === language))])),
      by_domain: Object.fromEntries(domains.map(domain => [domain, coverage(cells.filter(cell => cell.domain === domain))])),
    },
  }
}
// Crossed Pass^3 sidecar: Intake's leaderboard score, distinct from pass@1.
const pass3 = score => {
  const cells = languages.map((language, l) => ({ language, domain: 'intake', score: score(l) }))
  return { schema_version: 'benchmark-pass3-v1', benchmark_version: 'voice-v2-entity', frame_revision: 'voice-v2-entity-rev', cells, score: mean(cells.map(cell => cell.score)) }
}
const core = submission('voice-v2', ['airline', 'retail', 'telecom'], (l, d) => (40 + 5 * l + 10 * d) / 100)
const packages = {
  fixture_core: { 'benchmark-metadata.json': metadata('fixture-live', 'gemini'), 'benchmark-submission.json': core },
  // A second voice-v2 package for the same agent is dropped, not a second row.
  fixture_core_duplicate: { 'benchmark-metadata.json': metadata('fixture-live', 'gemini'), 'benchmark-submission.json': submission('voice-v2', ['airline', 'retail', 'telecom'], () => 0.99) },
  // Same agent as fixture_core: fills its Banking row.
  fixture_banking: { 'benchmark-metadata.json': metadata('fixture-live', 'gemini'), 'benchmark-submission.json': submission('voice-v2-banking', ['banking_knowledge'], l => (10 + l) / 100) },
  // No voice-v2 package for this agent: shown only under Intake.
  fixture_entity: { 'benchmark-metadata.json': metadata('fixture-entity-only', 'xai'), 'benchmark-submission.json': submission('voice-v2-entity', ['intake'], l => (60 + 2 * l) / 100), 'benchmark-pass3.json': pass3(l => (20 + l) / 100) },
}
const percent = value => `${(100 * value).toFixed(1)}%`

test.beforeEach(async ({ page }) => {
  // Exercise compatibility with deployments that do not have a snapshot.
  await page.route('**/voice-leaderboard.json', route => route.fulfill({ status: 404, body: '' }))
  await page.route('**/submissions/manifest.json', async route => {
    const manifest = await (await route.fetch()).json()
    await route.fulfill({ json: { ...manifest, voice_v2_submissions: ['fixture_core', 'fixture_core_duplicate'], voice_v2_domain_submissions: ['fixture_banking', 'fixture_entity'] } })
  })
  await page.route('**/submissions/fixture_*/*', route => {
    const [dir, file] = route.request().url().split('/').slice(-2)
    const json = packages[dir][file]
    return json ? route.fulfill({ json }) : route.fulfill({ status: 404, body: '' })
  })
})

// Views that refetch the manifest can leave an intercepted fetch in flight
// when a test ends; drop those instead of failing on the closed page.
test.afterEach(async ({ page }) => {
  await page.unrouteAll({ behavior: 'ignoreErrors' })
})

const selectDomain = (page, name) => page.locator('.mv-domain-toggle button', { hasText: name }).click()

test('domain and duplicate packages never change Overall or add Overall rows', async ({ page }) => {
  await page.goto('/leaderboard?benchmark=voice')
  const table = page.locator('.mv-leaderboard-table')
  await expect(table.locator('tr.mv-package-row')).toHaveCount(1)
  await expect(table.locator('tr[data-package="fixture_core"] td[data-metric="task"]')).toContainText(percent(core.summary.score))
})

test('banking fills the matching voice-v2 row', async ({ page }) => {
  await page.goto('/leaderboard?benchmark=voice')
  await selectDomain(page, 'Banking')
  const row = page.locator('.mv-leaderboard-table tr[data-package="fixture_core"]')
  await expect(row.locator('td[data-metric="task"]')).toContainText(percent(packages.fixture_banking['benchmark-submission.json'].summary.score))
  await row.locator('.mv-model-trigger').click()
  const detail = page.locator('.mv-leaderboard-table .mv-detail')
  await expect(detail.locator('tr[data-factor="banking_knowledge"] td[data-language="ko"]')).toHaveText('13.0%')
  await expect(detail.locator('tr[data-factor="intake"] td[data-language="ko"]')).toHaveText('—')
  await expect(detail.locator('tr[data-factor="all"] td[data-language="all"]')).toHaveText(percent(core.summary.score))
  await expect(detail).toContainText('Submitted 2 packages')
})

test('an entity package without voice-v2 appears only under Intake, scored by Pass^3', async ({ page }) => {
  await page.goto('/leaderboard?benchmark=voice')
  const table = page.locator('.mv-leaderboard-table')
  await expect(table.locator('tr[data-package="fixture_entity"]')).toHaveCount(0)
  await selectDomain(page, 'Intake')
  await expect(table.locator('thead .mv-metric-header')).toHaveText(['Task completion (pass^3) ↑ •'])
  const row = table.locator('tr[data-package="fixture_entity"]')
  await expect(row.locator('td[data-metric="task"]')).toContainText(percent(packages.fixture_entity['benchmark-pass3.json'].score))
  await expect(table.locator('tr[data-package="fixture_core"] td[data-metric="task"]')).toHaveText('—')
  await row.locator('.mv-model-trigger').click()
  const detail = page.locator('.mv-leaderboard-table .mv-detail')
  await expect(detail.locator('tr[data-factor="intake"] td[data-language="en"]')).toHaveText('20.0%')
  await expect(detail.locator('tr[data-factor="intake"] th')).toContainText('Intake (pass^3)')
  await expect(detail.locator('tr[data-factor="all"] td[data-language="all"]')).toHaveText('—')
  await selectDomain(page, 'Airline')
  await expect(row).toHaveCount(0)
})

test('moving to /progress shows the English progress view', async ({ page }) => {
  await page.goto('/leaderboard?benchmark=voice')
  await expect(page.locator('.mv-leaderboard-table')).toBeVisible()
  await page.evaluate(() => {
    window.history.pushState(null, '', '/progress?benchmark=voice')
    window.dispatchEvent(new PopStateEvent('popstate'))
  })
  await expect(page.locator('#progress')).toBeAttached()
  await page.goBack()
  await expect(page.locator('.mv-leaderboard-table')).toBeVisible()
})

test('a same-path popstate keeps the English toggle', async ({ page }) => {
  await page.goto('/leaderboard?benchmark=voice')
  await page.getByRole('button', { name: 'English', exact: true }).click()
  await page.evaluate(() => {
    window.history.pushState(null, '', '/leaderboard?benchmark=voice')
    window.dispatchEvent(new PopStateEvent('popstate'))
  })
  await expect(page.getByRole('button', { name: 'English', exact: true })).toHaveAttribute('aria-pressed', 'true')
})

test('the language map view survives a reload and leaves with the τ-i18n view', async ({ page }) => {
  await page.goto('/leaderboard?benchmark=voice&view=languages')
  const tab = name => page.locator('.mv-view-tabs button', { hasText: name })
  await expect(tab('Language map')).toHaveAttribute('aria-pressed', 'true')
  await expect(page).toHaveURL(/view=languages/)
  await page.reload()
  await expect(tab('Language map')).toHaveAttribute('aria-pressed', 'true')
  await tab('3D explorer').click()
  await expect(page).not.toHaveURL(/view=/)
  await tab('Language map').click()
  await expect(page).toHaveURL(/view=languages/)
  await page.getByRole('button', { name: 'English', exact: true }).click()
  await expect(page).not.toHaveURL(/view=/)
})

test('?domain=intake opens the Intake tab and the URL follows the tab', async ({ page }) => {
  await page.goto('/leaderboard?benchmark=voice&domain=intake')
  const table = page.locator('.mv-leaderboard-table')
  await expect(page.locator('.mv-domain-toggle button.active')).toContainText('Intake')
  await expect(table.locator('tr[data-package="fixture_entity"]')).toHaveCount(1)
  await selectDomain(page, 'Airline')
  await expect(page).toHaveURL(/[?&]domain=airline/)
  await selectDomain(page, 'Overall')
  await expect(page).not.toHaveURL(/domain=/)
})
