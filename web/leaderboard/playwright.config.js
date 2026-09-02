import { defineConfig } from '@playwright/test'

// E2E tests for routing + prerendering. They run against the *built and
// prerendered* dist/ served with GitHub Pages semantics (no SPA rewrites,
// 404.html for unknown paths) — `npm run build && npm run prerender` first.
export default defineConfig({
  testDir: './e2e',
  timeout: 30000,
  retries: process.env.CI ? 1 : 0,
  use: {
    baseURL: 'http://127.0.0.1:4173',
    // Use the system Chrome instead of downloading a Playwright browser;
    // it's preinstalled on GitHub runners and dev machines alike.
    channel: 'chrome',
  },
  webServer: {
    command: 'node scripts/prerender.mjs --serve',
    port: 4173,
    reuseExistingServer: !process.env.CI,
  },
})
