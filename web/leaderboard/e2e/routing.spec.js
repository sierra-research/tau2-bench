// Routing + prerendering behavior tests. Run against the prerendered dist/
// served with GitHub Pages semantics (see playwright.config.js).
import { expect, test } from '@playwright/test'
import { HYPER_TAU_URL, LEADERBOARD_MENU } from '../src/routes.js'

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

test('direct load: community spotlight', async ({ page }) => {
  await page.goto('/community')
  await expect(page).toHaveTitle(/Community Spotlight — τ-bench/)
  await expect(page.getByRole('heading', { name: 'Community Spotlight' })).toBeVisible()

  const tauRecCard = page.locator('.community-card').filter({ hasText: 'τ-Rec' })
  await expect(tauRecCard).toHaveCount(1)
  await expect(tauRecCard).toContainText('τ-Rec')
  await expect(tauRecCard.locator('.blog-card-link')).toHaveAttribute('href', '/community/tau-rec.html')
  await expect(tauRecCard).toContainText('Bharath Sivaram Narasimhan')
  await expect(tauRecCard).toContainText('Karthik Narasimhan')
  await expect(tauRecCard).toContainText('Independent Researcher')
  await expect(tauRecCard).toContainText('Princeton University')
  await expect(tauRecCard.locator('.post-author-photo')).toHaveCount(2)
  await expect(tauRecCard.locator('.post-author-name').filter({ hasText: 'Bharath Sivaram Narasimhan' })).toHaveAttribute(
    'href',
    '/authors/bharath-narasimhan.html'
  )
})

test('τ-Rec community spotlight contains authors, highlights, results, and primary links', async ({ page }) => {
  await page.goto('/community/tau-rec.html')

  await expect(page).toHaveTitle(/τ-Rec: A Verifiable Benchmark/)
  await expect(page.getByRole('heading', { level: 1, name: /τ-Rec/ })).toBeVisible()
  await expect(page.getByText('Bharath Sivaram Narasimhan', { exact: true })).toBeVisible()
  await expect(page.getByText('Karthik Narasimhan', { exact: true })).toBeVisible()
  await expect(page.locator('.author-names a').filter({ hasText: 'Bharath Sivaram Narasimhan' })).toHaveAttribute(
    'href',
    '/authors/bharath-narasimhan.html'
  )
  await expect(page.getByRole('heading', { name: 'Three ideas τ-Rec brings together' })).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Verifiable by construction' })).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Reliability across trials' })).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Reveal-tagged elicitation' })).toBeVisible()
  await expect(page.getByRole('heading', { name: 'How a τ-Rec task works' })).toBeVisible()
  await expect(page.locator('#sample-conversation .demo-label')).toContainText('Sample conversation trace')
  await expect(page.getByText(/Task 040 · DeepSeek V4 Flash \(max thinking\) · passing trial/)).toBeVisible()
  await expect(page.getByText('5/5 constraints satisfied · no policy violations', { exact: true })).toBeVisible()
  await expect(page.getByRole('link', { name: 'public trace archive' })).toHaveAttribute(
    'href',
    'https://github.com/nbharaths/tau-rec/releases/tag/v1.1.0'
  )
  await expect(page.getByRole('heading', { name: 'Paper results' })).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Key findings' })).toBeVisible()

  const paperResults = page.locator('#paper-results')
  await expect(paperResults.locator('tbody tr')).toHaveCount(9)
  await expect(paperResults.locator('#results-metric-header')).toHaveText('pass^1')
  await expect(paperResults.locator('tbody tr').first()).toContainText('DeepSeek V4 Flash')
  await expect(paperResults.locator('tbody tr').first()).toContainText('Max thinking')
  await expect(paperResults.locator('tbody tr').first()).toContainText('57.1%')

  await paperResults.getByRole('button', { name: 'pass^4' }).click()
  await expect(paperResults.locator('#results-metric-header')).toHaveText('pass^4')
  await expect(paperResults.getByRole('button', { name: 'pass^4' })).toHaveAttribute('aria-pressed', 'true')
  await expect(paperResults.locator('tbody tr').first()).toContainText('DeepSeek V4 Flash')
  await expect(paperResults.locator('tbody tr').first()).toContainText('High thinking')
  await expect(paperResults.locator('tbody tr').first()).toContainText('38.3%')
  await expect(paperResults.getByRole('link', { name: 'View live leaderboard' })).toHaveAttribute(
    'href',
    'https://github.com/nbharaths/tau-rec/blob/main/LEADERBOARD.md'
  )

  await expect(page.getByRole('img', { name: /reliability falls across repeated trials/i })).toBeVisible()
  await expect(page.getByRole('img', { name: /hidden preferences make tasks harder/i })).toBeVisible()
  await expect(page.getByRole('link', { name: 'Read the paper' })).toHaveAttribute(
    'href',
    'https://doi.org/10.1145/3773078.3831847'
  )
  await expect(page.getByRole('link', { name: 'View on GitHub' })).toHaveAttribute(
    'href',
    'https://github.com/nbharaths/tau-rec'
  )
  await expect(page.getByRole('link', { name: 'Live leaderboard', exact: true })).toHaveAttribute(
    'href',
    'https://github.com/nbharaths/tau-rec/blob/main/LEADERBOARD.md'
  )

  const resultImages = await page.locator('.result-figure img').evaluateAll(
    (images) => images.map((image) => ({ complete: image.complete, naturalWidth: image.naturalWidth }))
  )
  expect(resultImages).toHaveLength(2)
  expect(resultImages.every((image) => image.complete && image.naturalWidth >= 1600)).toBe(true)

  const figureLinksAreIsolated = await page.locator('.result-figure a[target="_blank"]').evaluateAll(
    (links) => links.every((link) => link.relList.contains('noopener') && link.relList.contains('noreferrer'))
  )
  expect(figureLinksAreIsolated).toBe(true)
})

test('τ-Rec community spotlight stays responsive and its mobile nav works', async ({ page }) => {
  await page.goto('/community/tau-rec.html')

  for (const width of [390, 768, 769, 900, 1440]) {
    await page.setViewportSize({ width, height: 844 })
    const hasHorizontalOverflow = await page.evaluate(
      () => document.documentElement.scrollWidth > window.innerWidth
    )
    expect(hasHorizontalOverflow).toBe(false)
  }

  await page.setViewportSize({ width: 390, height: 844 })
  await page.locator('.mobile-menu-toggle').click()
  await expect(page.locator('#site-nav')).toHaveClass(/open/)
  await expect(page.getByRole('link', { name: 'Community' })).toBeVisible()
})

test('community spotlight page and nav stay responsive', async ({ page }) => {
  await page.goto('/community')

  for (const width of [769, 800, 900, 1100, 1101, 1150, 1440]) {
    await page.setViewportSize({ width, height: 800 })
    const layout = await page.evaluate(() => ({
      hasHorizontalOverflow: document.documentElement.scrollWidth > window.innerWidth,
      navItems: [...document.querySelectorAll('.nav-links > *')].map((item) => {
        const rect = item.getBoundingClientRect()
        return { left: rect.left, right: rect.right, height: rect.height }
      }),
    }))

    expect(layout.hasHorizontalOverflow).toBe(false)
    for (const item of layout.navItems) {
      expect(item.left).toBeGreaterThanOrEqual(0)
      expect(item.right).toBeLessThanOrEqual(width)
      expect(item.height).toBeLessThanOrEqual(25)
    }
  }
})

test('Bharath author page contains concise bio, paper, and community spotlight', async ({ page }) => {
  await page.goto('/authors/bharath-narasimhan.html')

  await expect(page).toHaveTitle(/Bharath Sivaram Narasimhan \| τ-bench/)
  await expect(page.getByRole('heading', { level: 1, name: 'Bharath Sivaram Narasimhan' })).toBeVisible()
  await expect(page.getByText('Independent Researcher', { exact: true })).toBeVisible()
  await expect(
    page.getByText(
      'Bharath Sivaram Narasimhan is an independent researcher focused on verifiable evaluation methods for reliable agentic AI.',
      { exact: true }
    )
  ).toBeVisible()
  await expect(page.getByRole('heading', { name: 'Profiles' })).toHaveCount(0)
  await expect(page.getByRole('link', { name: /τ-Rec: A Verifiable Benchmark/ })).toHaveAttribute(
    'href',
    'https://doi.org/10.1145/3773078.3831847'
  )
  await expect(page.locator('.blog-card-link')).toHaveAttribute('href', '/community/tau-rec.html')
  for (const width of [390, 640, 641, 900]) {
    await page.setViewportSize({ width, height: 844 })
    const hasHorizontalOverflow = await page.evaluate(
      () => document.documentElement.scrollWidth > window.innerWidth
    )
    expect(hasHorizontalOverflow).toBe(false)
  }
})

test('community spotlight is available from the mobile nav', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto('/')

  await page.locator('.mobile-menu-toggle').click()
  const communityLink = page.getByRole('button', { name: 'Community' })
  await expect(communityLink).toBeVisible()
  await communityLink.click()

  await expect(page).toHaveURL(/\/community$/)
  await expect(page).toHaveTitle(/Community Spotlight — τ-bench/)
  await expect(page.getByRole('heading', { name: 'Community Spotlight' })).toBeVisible()
  await expect(page.locator('.nav-links')).toHaveClass(/mobile-hidden/)

  await page.locator('.mobile-menu-toggle').click()
  await expect(page.getByRole('button', { name: 'Community' })).toHaveClass(/active/)
})

test('community spotlight footer reaches the bottom on a tall viewport', async ({ page }) => {
  await page.setViewportSize({ width: 900, height: 890 })
  await page.goto('/community')

  const footerBottom = await page.locator('.simple-footer').evaluate(
    (footer) => footer.getBoundingClientRect().bottom
  )
  expect(footerBottom).toBeGreaterThanOrEqual(889)
})

test('static author and published blog pages link to community spotlight', async ({ page }) => {
  for (const path of [
    '/authors/soham-ray.html',
    '/blog/tau-knowledge.html',
    '/blog/tau-voice-examples.html',
    '/blog/tau3-task-fixes.html',
  ]) {
    await page.goto(path)
    await expect(page.getByRole('link', { name: 'Community' })).toHaveAttribute('href', '/community')
  }
})

test('published static pages use the Sierra favicon', async ({ page }) => {
  for (const path of [
    '/authors/soham-ray.html',
    '/blog/tau-knowledge.html',
    '/blog/tau-voice-examples.html',
    '/blog/tau3-task-fixes.html',
    '/community/tau-rec.html',
    '/talks/stanford-aims-tau.html',
  ]) {
    await page.goto(path)
    await expect(page.locator('link[rel="icon"]')).toHaveAttribute('href', '/sierra-logo.png')
  }
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

test('prerendered community HTML contains the spotlight and meta', async ({ request }) => {
  const res = await request.get('/community')
  expect(res.status()).toBe(200)
  const html = await res.text()
  expect(html).toContain('<title>Community Spotlight — τ-bench</title>')
  expect(html).toContain('property="og:title"')
  expect(html).toContain('τ-Rec')
  expect(html).toContain('/community/tau-rec.html')
  expect(html).toContain('Bharath Sivaram Narasimhan')
  expect(html).toContain('https://taubench.com/community')
})

test('static τ-Rec spotlight HTML contains SEO, results, and canonical links', async ({ request }) => {
  const res = await request.get('/community/tau-rec.html')
  expect(res.status()).toBe(200)
  const html = await res.text()
  expect(html).toContain('<title>τ-Rec: A Verifiable Benchmark for Agentic Recommender Systems | τ-bench</title>')
  expect(html).toContain('property="og:title"')
  expect(html).toContain('https://taubench.com/community/tau-rec.html')
  expect(html).toContain('Three ideas τ-Rec brings together')
  expect(html).toContain('Reveal-tagged elicitation')
  expect(html).toContain('id="sample-conversation"')
  expect(html).toContain('5/5 constraints satisfied · no policy violations')
  expect(html).toContain('id="paper-results"')
  const resultsBody = html.match(/<tbody id="results-tbody"[^>]*>([\s\S]*?)<\/tbody>/)?.[1]
  expect(resultsBody).toBeDefined()
  expect((resultsBody?.match(/<tr>/g) ?? [])).toHaveLength(9)
  expect(resultsBody).toContain('<td class="mode-cell">High thinking</td>')
  expect(resultsBody).toContain('<td class="score-cell">57.1%</td>')
  expect(html).toContain('tau-rec-reliability.png')
  expect(html).toContain('tau-rec-hidden-intent.png')

  for (const asset of [
    '/community/assets/bharath-narasimhan.jpg',
    '/community/assets/tau-rec-reliability.png',
    '/community/assets/tau-rec-hidden-intent.png',
  ]) {
    expect((await request.get(asset)).status()).toBe(200)
  }

  const sitemap = await (await request.get('/sitemap.xml')).text()
  expect(sitemap).toContain('https://taubench.com/community/tau-rec.html')
  expect(sitemap).toContain('https://taubench.com/authors/bharath-narasimhan.html')
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

  await page.getByRole('button', { name: 'Leaderboards' }).click()
  await page.getByRole('menuitem', { name: /τ³-Voice/ }).click()
  await expect(page).toHaveURL(/\/leaderboard\?benchmark=voice/)
  await page.waitForFunction(() => window.scrollY === 0)
  await expect(page.getByRole('heading', { name: 'τ³-Voice Leaderboard' })).toBeVisible()
})

test('nav links update path and title', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('button', { name: 'Leaderboards' }).click()
  await page.getByRole('menuitem', { name: /τ³-Banking/ }).click()
  await expect(page).toHaveURL(/\/leaderboard\?benchmark=knowledge/)
  await expect(page).toHaveTitle(/Leaderboard — τ-bench/)

  await page.getByRole('button', { name: 'Overview' }).click()
  await expect(page).toHaveURL(/\/(\?.*)?$/)
  await expect(page).toHaveTitle(/τ-bench — Benchmarking AI Agents/)
})

// ---------------------------------------------------------------------------
// Leaderboards menu and the τ^τ-bench pointers.
// ---------------------------------------------------------------------------

test('Leaderboards menu lists every track and opens/closes', async ({ page }) => {
  await page.goto('/')
  const menu = page.getByRole('menu', { name: 'Leaderboards' })
  await expect(menu).toBeHidden()

  const trigger = page.getByRole('button', { name: 'Leaderboards' })
  await trigger.click()
  await expect(menu).toBeVisible()
  await expect(trigger).toHaveAttribute('aria-expanded', 'true')
  await expect(menu.getByRole('menuitem')).toHaveCount(LEADERBOARD_MENU.length)
  for (const item of LEADERBOARD_MENU) {
    await expect(menu.getByRole('menuitem', { name: new RegExp(item.label.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')) })).toBeVisible()
  }

  await page.keyboard.press('Escape')
  await expect(menu).toBeHidden()

  await trigger.click()
  await page.locator('.hero-description').click()
  await expect(menu).toBeHidden()
})

test('Leaderboards menu closes on browser back/forward', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('button', { name: 'Leaderboards' }).click()
  await page.getByRole('menuitem', { name: /τ³-Voice/ }).click()
  await expect(page).toHaveURL(/\/leaderboard\?benchmark=voice/)

  const trigger = page.getByRole('button', { name: 'Leaderboards' })
  await trigger.click()
  await expect(page.getByRole('menu', { name: 'Leaderboards' })).toBeVisible()
  await page.goBack()
  await expect(page).toHaveURL(/\/(\?.*)?$/)
  await expect(page.getByRole('menu', { name: 'Leaderboards' })).toBeHidden()
  await expect(trigger).toHaveAttribute('aria-expanded', 'false')
})

test('Leaderboards menu switches benchmark while already on the leaderboard', async ({ page }) => {
  await page.goto('/leaderboard?benchmark=core')
  await expect(page.getByRole('heading', { name: 'τ²-bench Leaderboard' })).toBeVisible()
  await page.getByRole('button', { name: 'Leaderboards' }).click()
  await page.getByRole('menuitem', { name: /τ³-Voice/ }).click()
  await expect(page).toHaveURL(/\/leaderboard\?benchmark=voice/)
  await expect(page.getByRole('heading', { name: 'τ³-Voice Leaderboard' })).toBeVisible()
})

test('τ^τ-bench entry opens the hyper-tau-bench site in a new tab', async ({ page }) => {
  await page.goto('/')
  await page.getByRole('button', { name: 'Leaderboards' }).click()
  const hyper = page.getByRole('menuitem', { name: /τ\^τ-bench/ })
  await expect(hyper).toHaveAttribute('href', HYPER_TAU_URL)
  await expect(hyper).toHaveAttribute('target', '_blank')
  await expect(hyper).toContainText('New')
})

test('announcement banner points at τ^τ-bench', async ({ page }) => {
  await page.goto('/')
  const banner = page.locator('.update-notification')
  await expect(banner).toContainText('-bench is here')
  await expect(banner.locator('.notification-link')).toHaveAttribute('href', HYPER_TAU_URL)
})

// ---------------------------------------------------------------------------
// Unknown paths: GitHub Pages serves 404.html, which boots the SPA.
// ---------------------------------------------------------------------------

test('unknown path returns 404 status but renders the app', async ({ page }) => {
  const response = await page.goto('/definitely-not-a-page')
  expect(response.status()).toBe(404)
  await expect(page.locator('.preview-table-wrapper')).toHaveCount(3)
})
