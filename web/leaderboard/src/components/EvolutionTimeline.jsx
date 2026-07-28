import './EvolutionTimeline.css'

// One entry per release, chronological (oldest first); rendered newest-first.
// All content is static and visible without interaction; the only links are
// the paper/blog references.
const MILESTONES = [
  {
    name: 'τ-bench',
    date: 'June 2024',
    links: [
      { label: 'Paper', href: 'https://arxiv.org/abs/2406.12045' },
      { label: 'Blog', href: 'https://sierra.ai/blog/benchmarking-ai-agents' },
    ],
    description: (
      <>
        The original benchmark for tool-agent-user interaction. Agents converse
        with a simulated user and call tools, scored against verifiable database
        outcomes with the pass<sup>k</sup> reliability metric.
      </>
    ),
    domainsLabel: 'Domains',
    domains: ['🛍️ Retail', '✈️ Airline'],
  },
  {
    name: 'τ²-bench',
    date: 'June 2025',
    links: [
      { label: 'Paper', href: 'https://arxiv.org/abs/2506.07982' },
      { label: 'Blog', href: 'https://sierra.ai/blog/benchmarking-agents-in-collaborative-real-world-scenarios' },
    ],
    description: (
      <>
        Dual control: the user can now act on the world too. Agents must guide
        users through steps only the user can perform, not just act on their
        behalf.
      </>
    ),
    domainsLabel: 'Adds domain',
    domains: ['📱 Telecom'],
  },
  {
    name: 'Task audit & fixes',
    date: 'February 2026',
    badge: 'τ³-bench',
    links: [
      { label: 'Blog', href: `${import.meta.env.BASE_URL}blog/tau3-task-fixes.html` },
    ],
    description: (
      <>
        Audited and fixed 50+ tasks across the airline and retail domains —
        correcting expected actions, ambiguous instructions, and impossible
        constraints.
      </>
    ),
    domainsLabel: 'Updates',
    domains: ['🛍️ Retail', '✈️ Airline'],
  },
  {
    name: 'τ-knowledge',
    hoverNote: 'Appears on the leaderboard as τ³-Banking',
    date: 'March 2026',
    badge: 'τ³-bench',
    links: [
      { label: 'Paper', href: 'https://arxiv.org/abs/2603.04370' },
      { label: 'Blog', href: `${import.meta.env.BASE_URL}blog/tau-knowledge.html` },
    ],
    description: (
      <>
        Knowledge-intensive tasks: agents retrieve and reason over a realistic
        knowledge base of ~700 documents, combining retrieval with policy
        application.
      </>
    ),
    domainsLabel: 'Adds domain',
    domains: ['🏦 Banking'],
  },
  {
    name: 'τ-voice',
    hoverNote: 'Appears on the leaderboard as τ³-Voice',
    date: 'March 2026',
    badge: 'τ³-bench',
    links: [
      { label: 'Paper', href: 'https://arxiv.org/abs/2603.13686' },
      { label: 'Blog', href: 'https://sierra.ai/blog/tau-voice-benchmarking-real-time-voice-agents-on-real-world-tasks' },
    ],
    description: (
      <>
        Real-time voice: full-duplex conversations with interruptions, accents,
        and background noise — the same task rigor, now in speech.
      </>
    ),
    domainsLabel: 'Voice mode',
    domains: ['🛍️ Retail', '✈️ Airline', '📱 Telecom'],
  },
]

function EvolutionTimeline() {
  return (
    <section className="evolution-section">
      <h2 className="evolution-title">How τ-bench has evolved</h2>
      <div className="evolution-timeline">
        {[...MILESTONES].reverse().map((m) => (
          <div className="evolution-entry" key={`${m.name}-${m.date}`}>
            <div className="evolution-marker">
              <span className="evolution-dot" />
            </div>
            <div className="evolution-content">
              <div className="evolution-header">
                <span className="evolution-name" title={m.hoverNote}>{m.name}</span>
                {m.badge && <span className="evolution-badge">{m.badge}</span>}
                <span className="evolution-date">{m.date}</span>
                <span className="evolution-links">
                  {m.links.map((l) => (
                    <a
                      className="evolution-link"
                      key={l.label}
                      href={l.href}
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      {l.label} →
                    </a>
                  ))}
                </span>
              </div>
              <p className="evolution-description">{m.description}</p>
              <div className="evolution-domains">
                <span className="evolution-domains-label">{m.domainsLabel}:</span>
                {m.domains.map((d) => (
                  <span className="evolution-domain-chip" key={d}>{d}</span>
                ))}
              </div>
            </div>
          </div>
        ))}
      </div>
    </section>
  )
}

export default EvolutionTimeline
