import { readFileSync } from 'node:fs'
import { test, expect } from '@playwright/test'
import { domainScore, packageScore } from '../src/data/voiceV2Rows.js'

const snapshot = JSON.parse(readFileSync(new URL('../public/voice-leaderboard.json', import.meta.url), 'utf8'))

test('snapshot renders scores and breakdowns without fetching S3 packages', async ({ page }) => {
  const packageRequests = []
  await page.route('**/submissions/**', route => {
    packageRequests.push(route.request().url())
    return route.abort()
  })
  await page.goto('/leaderboard?benchmark=voice')
  const core = snapshot.rows.filter(pkg => pkg.submission)
  const table = page.locator('.mv-leaderboard-table')
  await expect(table.locator('tr.mv-package-row')).toHaveCount(core.length)
  for (const pkg of core) {
    const row = table.locator(`tr[data-package="${pkg.dir}"]`)
    await expect(row.locator('td[data-metric="task"]')).toContainText(`${packageScore(pkg, 'all').toFixed(1)}%`)
  }
  const pkg = core[0]
  await table.locator(`tr[data-package="${pkg.dir}"] .mv-model-trigger`).click()
  await expect(table.locator('.mv-detail tr[data-factor="airline"] td[data-language="en"]')).toHaveText(`${domainScore(pkg, 'airline', 'en').toFixed(1)}%`)
  expect(packageRequests).toEqual([])
})

test('snapshot keeps aggregate evidence identical to submitted packages', () => {
  for (const row of snapshot.rows) {
    for (const pkg of [row.submission && row, ...Object.values(row.extras)].filter(Boolean)) {
      const read = file => JSON.parse(readFileSync(new URL(`../public/submissions/${pkg.dir}/${file}`, import.meta.url), 'utf8'))
      const source = read('benchmark-submission.json')
      expect(pkg.submission.summary).toEqual(source.summary)
      expect(pkg.submission.frame.cells).toEqual(source.frame.cells.map(({ language, domain }) => ({ language, domain })))
      expect(pkg.submission.trials).toBeUndefined()
      if (pkg.quality) {
        const quality = read('benchmark-quality.json')
        for (const [index, cell] of pkg.quality.cells.entries()) {
          const { calls, ...aggregate } = quality.cells[index]
          expect(calls.length).toBeGreaterThan(0)
          expect(cell).toEqual(aggregate)
        }
      }
      if (pkg.pass3) {
        const pass3 = read('benchmark-pass3.json')
        expect(pkg.pass3.score).toEqual(pass3.score)
        expect(pkg.pass3.cells).toEqual(pass3.cells.map(({ language, domain, score }) => ({ language, domain, score })))
      }
    }
  }
})
