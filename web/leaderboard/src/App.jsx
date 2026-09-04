import { useState, useEffect } from 'react'
import './App.css'
import { getViewFromPath, HYPER_TAU_URL, LEADERBOARD_MENU, PAGE_META, SITE_ORIGIN, VIEW_PATHS } from './routes'
import TrajectoryVisualizer from './components/TrajectoryVisualizer'
import Leaderboard from './components/Leaderboard'
import LeaderboardPreview from './components/LeaderboardPreview'
import EvolutionTimeline from './components/EvolutionTimeline'
import Blog from './components/Blog'

// Update the document head to match the current view. The prerender step
// (scripts/prerender.mjs) snapshots the DOM after this runs, which is how
// each prerendered page gets its own title/description/canonical tags.
const setHeadContent = (selector, attr, value) => {
  const el = document.head.querySelector(selector)
  if (el) el.setAttribute(attr, value)
}

const applyPageMeta = (view) => {
  const meta = PAGE_META[view]
  if (!meta) return
  const url = `${SITE_ORIGIN}${VIEW_PATHS[view] || '/'}`
  document.title = meta.title
  setHeadContent('meta[name="description"]', 'content', meta.description)
  setHeadContent('link[rel="canonical"]', 'href', url)
  setHeadContent('meta[property="og:url"]', 'content', url)
  setHeadContent('meta[property="og:title"]', 'content', meta.title)
  setHeadContent('meta[property="og:description"]', 'content', meta.description)
  setHeadContent('meta[name="twitter:title"]', 'content', meta.title)
  setHeadContent('meta[name="twitter:description"]', 'content', meta.description)
}

// Benchmark names use a caret for a superscript in plain text ("τ^τ-bench");
// render it as one. Menu data stays JSX-free so the prerender script can
// import it.
const renderBenchName = (label) => {
  const caret = label.indexOf('^')
  if (caret === -1) return label
  const rest = label.slice(caret + 1)
  const supEnd = rest.search(/[^a-zA-Zα-ωΑ-Ω0-9]/)
  const sup = supEnd === -1 ? rest : rest.slice(0, supEnd)
  const tail = supEnd === -1 ? '' : rest.slice(supEnd)
  return (
    <>
      {label.slice(0, caret)}<sup>{sup}</sup>{tail}
    </>
  )
}

function App() {

  const [currentView, setCurrentView] = useState(() => getViewFromPath(window.location.pathname))
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const [leaderboardsOpen, setLeaderboardsOpen] = useState(false)

  // Handle navigation with URL updates
  const navigateTo = (view) => {
    setCurrentView(view)
    setMobileMenuOpen(false) // Close mobile menu when navigating
    const path = VIEW_PATHS[view]
    if (!path) return
    // Preserve existing query params when already on the target path (the
    // visualizer and leaderboard keep their state in the query string).
    if (window.location.pathname !== path) {
      // Keep the query string when moving between two routes of the same
      // view (e.g. /progress → /leaderboard both render the leaderboard,
      // and ?benchmark=… should survive the switch).
      const sameView = getViewFromPath(window.location.pathname) === view
      window.history.pushState(null, '', sameView ? `${path}${window.location.search}` : path)
    }
    // If the view didn't change, React won't re-render anything, so without
    // this a nav click from e.g. /progress (scrolled to the chart) back to
    // /leaderboard would visibly do nothing.
    window.scrollTo(0, 0)
  }

  // Navigate to an app-internal URL (path + query), e.g. from the homepage
  // preview cards: '/leaderboard?benchmark=voice'.
  const navigateToUrl = (url) => {
    window.history.pushState(null, '', url)
    setCurrentView(getViewFromPath(new URL(url, window.location.origin).pathname))
    // Views that keep state in the query string (the leaderboard's
    // ?benchmark=…) re-read it on popstate; fire one so switching benchmarks
    // from the nav menu works while that view is already mounted.
    window.dispatchEvent(new PopStateEvent('popstate'))
    setMobileMenuOpen(false)
    setLeaderboardsOpen(false)
    window.scrollTo(0, 0)
  }

  // Toggle mobile menu
  const toggleMobileMenu = () => {
    setMobileMenuOpen(!mobileMenuOpen)
  }



  // Scroll to a specific section if the path refers to one (/progress is the
  // leaderboard scrolled to the Progress-over-time panel). Tries a few times
  // with rAF + small timeouts so it works even if the target hasn't mounted
  // yet (data-loading async views).
  const scrollToSectionForPath = (pathname) => {
    const sectionId = pathname.replace(/\/$/, '') === '/progress' ? 'progress' : null
    if (!sectionId) return
    const tryScroll = (attemptsLeft) => {
      const el = document.getElementById(sectionId)
      if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'start' })
      } else if (attemptsLeft > 0) {
        setTimeout(() => tryScroll(attemptsLeft - 1), 100)
      }
    }
    requestAnimationFrame(() => tryScroll(20))
  }

  // Keep the document head (title, description, canonical, og:*) in sync
  // with the current view.
  useEffect(() => {
    applyPageMeta(currentView)
  }, [currentView])

  // Listen for browser back/forward button clicks and handle mobile menu
  useEffect(() => {
    const handlePopState = () => {
      setCurrentView(getViewFromPath(window.location.pathname))
      setLeaderboardsOpen(false)
      scrollToSectionForPath(window.location.pathname)
    }

    // Close mobile menu when clicking outside; same for the Leaderboards menu.
    const handleClickOutside = (event) => {
      if (mobileMenuOpen && !event.target.closest('.nav-container')) {
        setMobileMenuOpen(false)
      }
      if (leaderboardsOpen && !event.target.closest('.nav-dropdown')) {
        setLeaderboardsOpen(false)
      }
    }
    const handleKeyDown = (event) => {
      if (event.key === 'Escape') setLeaderboardsOpen(false)
    }

    // Listen to events
    window.addEventListener('popstate', handlePopState)
    document.addEventListener('click', handleClickOutside)
    document.addEventListener('keydown', handleKeyDown)

    // Honor an initial deep-link like /progress on first paint.
    scrollToSectionForPath(window.location.pathname)

    return () => {
      window.removeEventListener('popstate', handlePopState)
      document.removeEventListener('click', handleClickOutside)
      document.removeEventListener('keydown', handleKeyDown)
    }
  }, [mobileMenuOpen, leaderboardsOpen])

  return (
    <div className="App">
      {/* Navigation */}
      <nav className="navbar">
        <div className="nav-container">
          <div className="nav-logo">
            <div className="logo-main" onClick={() => navigateTo('home')}>
              <span className="tau-symbol">τ</span>
              <span className="bench-text">-bench</span>
            </div>
            <a href="https://sierra.ai" target="_blank" rel="noopener noreferrer" className="logo-attribution">
              <img src={`${import.meta.env.BASE_URL}sierra_logo.jpeg`} alt="Sierra" className="sierra-logo" />
              <span className="from-text">from Sierra</span>
            </a>
          </div>
          <button className="mobile-menu-toggle" onClick={toggleMobileMenu}>
            <span></span>
            <span></span>
            <span></span>
          </button>
          <div className={`nav-links ${mobileMenuOpen ? '' : 'mobile-hidden'}`}>
            <button onClick={() => navigateTo('home')} className={`nav-link ${currentView === 'home' ? 'active' : ''}`}>Overview</button>
            <div className={`nav-dropdown ${leaderboardsOpen ? 'open' : ''}`}>
              <button
                type="button"
                onClick={() => setLeaderboardsOpen((open) => !open)}
                className={`nav-link nav-dropdown-trigger ${currentView === 'leaderboard' ? 'active' : ''}`}
                aria-haspopup="menu"
                aria-expanded={leaderboardsOpen}
              >
                Leaderboards
                <span className="nav-dropdown-chevron" aria-hidden="true" />
              </button>
              <div className="nav-dropdown-menu" role="menu" aria-label="Leaderboards">
                {LEADERBOARD_MENU.map((item) =>
                  item.href ? (
                    <a
                      key={item.key}
                      role="menuitem"
                      className="nav-dropdown-item external"
                      href={item.href}
                      target="_blank"
                      rel="noopener noreferrer"
                      aria-label={item.label}
                      onClick={() => { setLeaderboardsOpen(false); setMobileMenuOpen(false) }}
                    >
                      <span className="nav-dropdown-item-label">
                        <span className="nav-dropdown-item-name">{renderBenchName(item.label)}</span>
                        {item.badge && <span className="nav-dropdown-badge">{item.badge}</span>}
                        <span className="nav-dropdown-external" aria-hidden="true">↗</span>
                      </span>
                      <span className="nav-dropdown-item-note">{item.note}</span>
                    </a>
                  ) : (
                    <button
                      key={item.key}
                      type="button"
                      role="menuitem"
                      className="nav-dropdown-item"
                      onClick={() => navigateToUrl(item.path)}
                    >
                      <span className="nav-dropdown-item-label">
                        <span className="nav-dropdown-item-name">{renderBenchName(item.label)}</span>
                      </span>
                      <span className="nav-dropdown-item-note">{item.note}</span>
                    </button>
                  )
                )}
              </div>
            </div>
            <button onClick={() => navigateTo('trajectory-visualizer')} className={`nav-link ${currentView === 'trajectory-visualizer' ? 'active' : ''}`}>Visualizer</button>
            <button onClick={() => navigateTo('blog')} className={`nav-link ${currentView === 'blog' ? 'active' : ''}`}>Blog</button>
            <a href="https://github.com/sierra-research/tau2-bench" target="_blank" rel="noopener noreferrer" onClick={() => setMobileMenuOpen(false)}>GitHub</a>
            <a href="https://github.com/sierra-research/tau2-bench/blob/main/docs/leaderboard-submission.md" target="_blank" rel="noopener noreferrer" onClick={() => setMobileMenuOpen(false)}>Submit Results</a>
          </div>
        </div>
      </nav>

      {/* Update Notification */}
      <div className="update-notification">
        <div className="notification-container">
          <span className="notification-badge">NEW</span>
          <span className="notification-text">
            τ<sup>τ</sup>-bench is here: can coding agents <em>build</em> the customer-service agents τ-bench evaluates?{' '}
            <a href={HYPER_TAU_URL} className="notification-link" target="_blank" rel="noopener noreferrer"><strong>Explore the τ<sup>τ</sup>-bench leaderboard →</strong></a>
          </span>
        </div>
      </div>

      {/* Conditional Content Rendering */}
      {currentView === 'home' ? (
        <>
          {/* Hero Section */}
          <section className="hero">
            <div className="hero-container-vertical">
              <div className="hero-content-vertical">
                <div className="hero-title-section">
                  <h1 className="hero-main-title">
                    <span className="tau-symbol">τ</span>
                    <span className="bench-text">-bench</span>
                  </h1>
                </div>

                <p className="hero-description">
                  Can AI agents reliably complete real-world tasks? 
                  τ-bench measures how well agents converse with users, call tools, 
                  retrieve knowledge, and follow policy across enterprise domains — in text and voice.
                </p>

                <LeaderboardPreview
                  onViewFullLeaderboard={() => navigateTo('leaderboard')}
                  onNavigate={navigateToUrl}
                />
              </div>
            </div>
          </section>

          <EvolutionTimeline />
        </>
      ) : currentView === 'leaderboard' ? (
        <Leaderboard />
      ) : currentView === 'trajectory-visualizer' ? (
        <TrajectoryVisualizer />
      ) : currentView === 'blog' ? (
        <Blog />
      ) : null}

      {/* Simple Footer */}
      <footer className="simple-footer">
        <div className="container">
          <p>
            For questions or feedback, contact{' '}
            <a href="mailto:research@sierra.ai" className="footer-email">
              research@sierra.ai
            </a>
          </p>
        </div>
      </footer>

    </div>
  )
}

export default App
