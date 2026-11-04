import { test, expect } from '@playwright/test'

// The visualizer lists submissions from visualizer-index.json alone; it loads
// no trajectory data until a model is chosen.

test('visualizer opens on a model list without loading trajectories', async ({ page }) => {
  const trajectoryRequests = []
  page.on('request', request => {
    if (request.url().includes('/trajectories/')) trajectoryRequests.push(request.url())
  })
  await page.goto('/trajectory-visualizer')

  const tabs = page.getByRole('tab')
  await expect(tabs).toHaveCount(4)
  await expect(page.getByRole('heading', { name: 'Choose a model' })).toBeVisible()
  await expect(page.locator('.mp-inline .mp-row').first()).toBeVisible()
  expect(trajectoryRequests).toEqual([])
})

test('visualizer search narrows the list by word prefix', async ({ page }) => {
  await page.goto('/trajectory-visualizer?bench=voice-en')
  await expect(page.getByRole('tab', { selected: true })).toContainText('English')
  await page.getByLabel('Search submissions').fill('realtime 2')
  const names = await page.locator('.mp-inline .mp-row .mp-model').allTextContents()
  expect(names.length).toBeGreaterThan(0)
  for (const name of names) expect(name).toBe('gpt-realtime-2')
})

test('visualizer choosing a model selects its benchmark, language and domain', async ({ page }) => {
  await page.goto('/trajectory-visualizer?bench=voice-ml')
  await page.locator('.mp-inline .mp-row').first().dispatchEvent('mousedown')
  await expect(page).toHaveURL(/bench=voice-ml&model=[^&]+&lang=en&domain=airline/)
  await expect(page.locator('.language-pill')).toHaveCount(6)
  await page.locator('.language-pill', { hasText: 'Korean' }).click()
  await expect(page).toHaveURL(/lang=ko/)
})

test('visualizer legacy leaderboard links resolve their benchmark', async ({ page }) => {
  const index = await (await page.request.get('/visualizer-index.json')).json()
  const voice = index.entries.find(e => e.benchmark === 'voice-en' && e.cells.some(c => c.domain === 'retail'))
  await page.goto(`/trajectory-visualizer?model=${voice.id}&domain=retail`)
  await expect(page.getByRole('tab', { selected: true })).toContainText('English')
  await expect(page).toHaveURL(new RegExp(`bench=voice-en&model=${voice.id}&domain=retail`))
  await expect(page.locator('.mp-trigger')).toContainText(voice.model)
})
