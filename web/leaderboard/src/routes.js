// Single source of truth for the site's routes and per-page metadata.
//
// Imported by the app (App.jsx) and by scripts/prerender.mjs (plain Node),
// so keep this file free of JSX and browser globals. Adding a page here is
// what makes it exist: the router resolves it, the prerender step emits a
// static HTML file for it, and the e2e suite picks it up.

export const SITE_ORIGIN = 'https://taubench.com'

// τ^τ-bench (hyper-tau-bench) is a sibling benchmark with its own site and
// leaderboard: coding harnesses building customer-service agents. Linked from
// the nav's Leaderboards menu and the announcement banner.
export const HYPER_TAU_URL = 'https://sierra-research.github.io/hyper-tau-bench/'

// Entries of the nav's Leaderboards menu, in display order. In-site entries
// carry an app path; the external entry carries an absolute href.
export const LEADERBOARD_MENU = [
  { key: 'knowledge', label: 'τ³-Banking', note: 'Text · knowledge retrieval', path: '/leaderboard?benchmark=knowledge' },
  { key: 'voice', label: 'τ³-Voice', note: 'Real-time voice', path: '/leaderboard?benchmark=voice' },
  { key: 'core', label: 'τ²-bench', note: 'Text · retail, airline, telecom', path: '/leaderboard?benchmark=core' },
  { key: 'hyper', label: 'τ^τ-bench', note: 'Agent-building agents', href: HYPER_TAU_URL, badge: 'New' },
]

// path → view name. Multiple paths may map to the same view ('/progress' is
// the leaderboard scrolled to the progress-over-time section).
export const ROUTES = {
  '/': 'home',
  '/leaderboard': 'leaderboard',
  '/progress': 'leaderboard',
  '/trajectory-visualizer': 'trajectory-visualizer',
  '/blog': 'blog',
}

// Canonical path for each view (used for navigation and canonical/og:url).
export const VIEW_PATHS = {
  home: '/',
  leaderboard: '/leaderboard',
  'trajectory-visualizer': '/trajectory-visualizer',
  blog: '/blog',
}

const SITE_TITLE = 'τ-bench — Benchmarking AI Agents on Real-World Tasks'
const SITE_DESCRIPTION =
  'Can AI agents reliably complete real-world tasks? τ-bench measures how well agents converse with users, call tools, retrieve knowledge, and follow policy across enterprise domains — in text and voice.'

export const PAGE_META = {
  home: {
    title: SITE_TITLE,
    description: SITE_DESCRIPTION,
  },
  leaderboard: {
    title: 'Leaderboard — τ-bench',
    description:
      'Model rankings on τ³-Banking, τ³-Voice, and τ²-bench: pass^k reliability for AI agents on enterprise customer-service tasks.',
  },
  'trajectory-visualizer': {
    title: 'Visualizer — τ-bench',
    description:
      'Explore τ-bench evaluation trajectories: full agent–user conversations, tool calls, and task definitions across domains.',
  },
  blog: {
    title: 'Blog — τ-bench',
    description: 'Research updates and release notes from the τ-bench team.',
  },
}

const stripTrailingSlash = (p) => (p.length > 1 && p.endsWith('/') ? p.slice(0, -1) : p)

// Resolve a pathname to a view; unknown paths render the homepage view
// (matching the old hash router's behavior for unknown hashes).
export const getViewFromPath = (pathname) => ROUTES[stripTrailingSlash(pathname)] || 'home'
