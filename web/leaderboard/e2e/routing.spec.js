// Routing + prerendering behavior tests. Run against the prerendered dist/
// served with GitHub Pages semantics (see playwright.config.js).
import { expect, test } from '@playwright/test'

// ---------------------------------------------------------------------------
// Direct loads: every route serves a real page with its own title and content.
// ---------------------------------------------------------------------------

test('direct load: homepage', async ({ page }) => {
  await page.goto('/')
  await expect(page).toHaveTitle(/τ-bench — Benchmarking AI Agents/)
  await expect(page.locator('.preview-table-wrapper')).toHaveCount(3)
  await expect(page.getByText('How τ-bench has evolved')).toBeVisible()
})

test('direct load: leaderboard defaults to τ³-Banking', async ({ page }) => {
  await page.goto('/leaderboard')
  await expect(page).toHaveTitle(/Leaderboard — τ-bench/)
  await expect(page.getByRole('heading', { name: 'τ³-Banking Leaderboard' })).toBeVisible()
})

test('direct load: leaderboard respects benchmark param', async ({ page }) => {
  await page.goto('/leaderboard?benchmark=voice')
  await expect(page.getByRole('heading', { name: 'τ³-Voice Leaderboard' })).toBeVisible()
})

test('direct load: /progress shows leaderboard with progress section', async ({ page }) => {
  await page.goto('/progress')
  await expect(page.locator('#progress')).toBeAttached()
})

test('direct load: blog and visualizer', async ({ page }) => {
  await page.goto('/blog')
  await expect(page).toHaveTitle(/Blog — τ-bench/)

  await page.goto('/trajectory-visualizer')
  await expect(page).toHaveTitle(/Visualizer — τ-bench/)
})

// ---------------------------------------------------------------------------
// Prerendered HTML: content and per-route meta exist without JavaScript.
// ---------------------------------------------------------------------------

test('prerendered leaderboard HTML contains content and meta', async ({ request }) => {
  const res = await request.get('/leaderboard')
  expect(res.status()).toBe(200)
  const html = await res.text()
  expect(html).toContain('<title>Leaderboard — τ-bench</title>')
  expect(html).toContain('τ³-Banking Leaderboard')
  expect(html).toContain('property="og:title"')
  expect(html).toContain('https://taubench.com/leaderboard')
})

test('prerendered homepage HTML contains preview cards', async ({ request }) => {
  const html = await (await request.get('/')).text()
  expect(html).toContain('preview-table-wrapper')
  expect(html).not.toContain('Loading leaderboard')
})

// ---------------------------------------------------------------------------
// Legacy hash links: every pre-path-routing URL shape redirects correctly.
// ---------------------------------------------------------------------------

test('legacy #leaderboard redirects with params intact', async ({ page }) => {
  await page.goto('/#leaderboard?benchmark=voice')
  await expect(page.getByRole('heading', { name: 'τ³-Voice Leaderboard' })).toBeVisible()
  await expect(page).toHaveURL(/\/leaderboard\?benchmark=voice/)
})

test('legacy benchmark=text maps to core', async ({ page }) => {
  await page.goto('/#leaderboard?benchmark=text')
  await expect(page.getByRole('heading', { name: 'τ²-bench Leaderboard' })).toBeVisible()
  await expect(page).toHaveURL(/benchmark=core/)
})

test('legacy #progress and deprecated #docs redirect', async ({ page }) => {
  await page.goto('/#progress')
  await expect(page).toHaveURL(/\/progress/)
  await expect(page.locator('#progress')).toBeAttached()

  await page.goto('/#docs')
  await expect(page).toHaveURL(/\/(\?.*)?$/)
  await expect(page.locator('.preview-table-wrapper')).toHaveCount(3)
})

test('legacy visualizer deep link preserves query', async ({ page }) => {
  await page.goto('/#trajectory-visualizer?view=tasks')
  await expect(page).toHaveURL(/\/trajectory-visualizer\?.*view=tasks/)
  await expect(page).toHaveTitle(/Visualizer — τ-bench/)
})

// ---------------------------------------------------------------------------
// Client-side navigation and history.
// ---------------------------------------------------------------------------

test('preview card navigates client-side; back returns home', async ({ page }) => {
  await page.goto('/')
  await page.locator('.preview-table-wrapper').first().click()
  await expect(page).toHaveURL(/\/leaderboard\?benchmark=knowledge/)
  await expect(page.getByRole('heading', { name: 'τ³-Banking Leaderboard' })).toBeVisible()

  await page.goBack()
  await expect(page).toHaveURL(/\/(\?.*)?$/)
  await expect(page.getByText('How τ-bench has evolved')).toBeVisible()
})

test('nav from /progress back to /leaderboard scrolls to top and keeps params', async ({ page }) => {
  await page.goto('/progress?benchmark=voice')
  // Wait for the auto-scroll down to the progress section to happen.
  await page.waitForFunction(() => window.scrollY > 0)

  await page.getByRole('button', { name: 'Leaderboard' }).click()
  await expect(page).toHaveURL(/\/leaderboard\?benchmark=voice/)
  await page.waitForFunction(() => window.scrollY === 0)
  await expect(page.getByRole('heading', { name: 'τ³-Voice Leaderboard' })).toBeVisible()
})

test('nav links update path and title', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('button', { name: 'Leaderboard' }).click()
  await expect(page).toHaveURL(/\/leaderboard/)
  await expect(page).toHaveTitle(/Leaderboard — τ-bench/)

  await page.getByRole('button', { name: 'Overview' }).click()
  await expect(page).toHaveURL(/\/(\?.*)?$/)
  await expect(page).toHaveTitle(/τ-bench — Benchmarking AI Agents/)
})

// ---------------------------------------------------------------------------
// Unknown paths: GitHub Pages serves 404.html, which boots the SPA.
// ---------------------------------------------------------------------------

test('unknown path returns 404 status but renders the app', async ({ page }) => {
  const response = await page.goto('/definitely-not-a-page')
  expect(response.status()).toBe(404)
  await expect(page.locator('.preview-table-wrapper')).toHaveCount(3)
})
