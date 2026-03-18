import { useState, useEffect } from 'react'
import './App.css'
import TrajectoryVisualizer from './components/TrajectoryVisualizer'
import Leaderboard from './components/Leaderboard'

function App() {
  
  // Initialize currentView based on URL hash (strip query params for view matching)
  const getViewFromHash = (hash) => {
    const base = hash.split('?')[0]
    if (base === 'leaderboard') return 'leaderboard'
    if (base === 'trajectory-visualizer') return 'trajectory-visualizer'
    if (base === 'results' || base === 'docs') return '__deprecated__'
    return 'home'
  }

  const getInitialView = () => {
    const hash = window.location.hash.slice(1)
    const view = getViewFromHash(hash)
    if (view === '__deprecated__') {
      window.history.replaceState(null, '', '#home')
      return 'home'
    }
    return view
  }
  
  const [currentView, setCurrentView] = useState(getInitialView())
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)
  const [blogDropdownOpen, setBlogDropdownOpen] = useState(false)

  // Handle navigation with URL updates
  const navigateTo = (view) => {
    setCurrentView(view)
    setMobileMenuOpen(false) // Close mobile menu when navigating
    if (view === 'home') {
      window.history.pushState(null, '', '#home')
    } else if (view === 'leaderboard') {
      window.history.pushState(null, '', '#leaderboard')
    } else if (view === 'trajectory-visualizer') {
      window.history.pushState(null, '', '#trajectory-visualizer')
    }
  }

  // Toggle mobile menu
  const toggleMobileMenu = () => {
    setMobileMenuOpen(!mobileMenuOpen)
  }



  // Listen for browser back/forward button clicks and handle mobile menu
  useEffect(() => {
    const handleHashChange = () => {
      const hash = window.location.hash.slice(1)
      const view = getViewFromHash(hash)
      if (view === '__deprecated__') {
        window.history.replaceState(null, '', '#home')
        setCurrentView('home')
      } else {
        setCurrentView(view)
      }
    }

    const handlePopState = () => {
      handleHashChange()
    }

    // Close mobile menu when clicking outside
    const handleClickOutside = (event) => {
      if (mobileMenuOpen && !event.target.closest('.nav-container')) {
        setMobileMenuOpen(false)
      }
    }

    // Listen to events
    window.addEventListener('hashchange', handleHashChange)
    window.addEventListener('popstate', handlePopState)
    document.addEventListener('click', handleClickOutside)

    // Set initial URL if none exists
    if (!window.location.hash) {
      window.history.replaceState(null, '', '#home')
    }

    return () => {
      window.removeEventListener('hashchange', handleHashChange)
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
            <div className="nav-dropdown" onMouseEnter={() => setBlogDropdownOpen(true)} onMouseLeave={() => setBlogDropdownOpen(false)}>
              <button className="nav-link nav-dropdown-trigger">
                Blog Posts <span className="dropdown-arrow">▾</span>
              </button>
              <div className={`nav-dropdown-menu ${blogDropdownOpen ? 'open' : ''}`}>
                <a href={`${import.meta.env.BASE_URL}blog/tau-knowledge.html`} onClick={() => { setMobileMenuOpen(false); setBlogDropdownOpen(false); }}>τ-knowledge</a>
                <a href={`${import.meta.env.BASE_URL}blog/tau-voice-examples.html`} onClick={() => { setMobileMenuOpen(false); setBlogDropdownOpen(false); }}>τ-voice examples</a>
                <a href={`${import.meta.env.BASE_URL}blog/tau3-task-fixes.html`} onClick={() => { setMobileMenuOpen(false); setBlogDropdownOpen(false); }}>τ³ Task Fixes</a>
              </div>
            </div>
            <a href="https://github.com/sierra-research/tau2-bench" target="_blank" rel="noopener noreferrer" onClick={() => setMobileMenuOpen(false)}>GitHub</a>
          </div>
        </div>
      </nav>

      {/* Update Notification */}
      <div className="update-notification">
        <div className="notification-container">
          <span className="notification-badge">NEW</span>
          <span className="notification-text">
            τ-bench now supports the <strong>banking domain</strong> and a <strong>voice mode</strong>, introduced by the{' '}
            <a href={`${import.meta.env.BASE_URL}blog/tau-knowledge.html`} className="notification-link">τ-knowledge</a> and{' '}
            <a href={`${import.meta.env.BASE_URL}blog/tau-voice-examples.html`} className="notification-link">τ-voice examples</a>.
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
                
                <div className="hero-image-section">
                  <img src={`${import.meta.env.BASE_URL}traj.png`} alt="Sample τ-bench Trajectories" className="trajectory-image" />
                </div>
                
                <div className="hero-description-section">
                  <p className="hero-description">
                    Benchmarking AI agents in collaborative real-world scenarios. 
                    τ-bench challenges agents to coordinate, guide, and assist users 
                    in achieving shared objectives across complex enterprise domains.
                  </p>
                  <div className="hero-actions">
                    <div className="button-row">
                      <a href="https://github.com/sierra-research/tau2-bench" target="_blank" rel="noopener noreferrer">
                        <button className="btn-primary">View on GitHub</button>
                      </a>
                      <button onClick={() => navigateTo('leaderboard')} className="btn-secondary">
                        View Leaderboard
                      </button>
                      <a href="https://github.com/sierra-research/tau2-bench?tab=readme-ov-file#leaderboard-submission" target="_blank" rel="noopener noreferrer">
                        <button className="btn-secondary">Submit Results</button>
                      </a>
                    </div>
                    <div className="button-row">
                      <a href="https://arxiv.org/abs/2506.07982" target="_blank" rel="noopener noreferrer">
                        <button className="btn-secondary">Read Paper</button>
                      </a>
                      <a href="https://sierra.ai/uk/blog/benchmarking-ai-agents" target="_blank" rel="noopener noreferrer">
                        <button className="btn-secondary">Read Blog Post</button>
                      </a>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </section>


        </>
      ) : currentView === 'leaderboard' ? (
        <Leaderboard />
      ) : currentView === 'trajectory-visualizer' ? (
        <TrajectoryVisualizer />
      ) : null}

      {/* Simple Footer */}
      <footer className="simple-footer">
        <div className="container">
          <p>
            For questions or feedback, contact{' '}
            <a href="mailto:victor@sierra.ai" className="footer-email">
              victor@sierra.ai
            </a>
            {' '}or{' '}
            <a href="mailto:ben.s@sierra.ai" className="footer-email">
              ben.s@sierra.ai
            </a>
          </p>
        </div>
      </footer>

    </div>
  )
}

export default App
