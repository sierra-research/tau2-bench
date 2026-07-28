import { useState, useEffect } from 'react'
import './App.css'
import { getViewFromPath, PAGE_META, SITE_ORIGIN, VIEW_PATHS } from './routes'
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

function App() {

  const [currentView, setCurrentView] = useState(() => getViewFromPath(window.location.pathname))
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

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
    setMobileMenuOpen(false)
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
      scrollToSectionForPath(window.location.pathname)
    }

    // Close mobile menu when clicking outside
    const handleClickOutside = (event) => {
      if (mobileMenuOpen && !event.target.closest('.nav-container')) {
        setMobileMenuOpen(false)
      }
    }

    // Listen to events
    window.addEventListener('popstate', handlePopState)
    document.addEventListener('click', handleClickOutside)

    // Honor an initial deep-link like /progress on first paint.
    scrollToSectionForPath(window.location.pathname)

    return () => {
      window.removeEventListener('popstate', handlePopState)
      document.removeEventListener('click', handleClickOutside)
    }
  }, [mobileMenuOpen])

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
            <button onClick={() => navigateTo('leaderboard')} className={`nav-link ${currentView === 'leaderboard' ? 'active' : ''}`}>Leaderboard</button>
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
            τ³-bench is here: <a href={`${import.meta.env.BASE_URL}blog/tau-knowledge.html`} className="notification-link"><strong>τ-knowledge</strong></a> evaluates
            agents on knowledge-intensive tasks, and{' '}
            <a href="https://sierra.ai/blog/tau-voice-benchmarking-real-time-voice-agents-on-real-world-tasks" className="notification-link"><strong>τ-voice</strong></a> benchmarks
            real-time voice agents.
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
