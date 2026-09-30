import './RewardDetails.css'

const labels = { DB: 'Database', COMMUNICATE: 'Communication', ACTION: 'Actions', ENV_ASSERTION: 'Environment assertions', NL_ASSERTION: 'Natural-language assertions' }
const display = value => typeof value === 'string' ? value : JSON.stringify(value, null, 2)

/** Display the recorded evaluator output, without regrading the trajectory. */
export default function RewardDetails({ rewardInfo, sourceUrl }) {
  const reward = rewardInfo || {}
  const basis = reward.reward_basis || []
  const breakdown = reward.reward_breakdown || {}
  const components = [...new Set([...basis, ...Object.keys(breakdown)])]
  const checks = []
  if (reward.db_check) checks.push({ label: 'Database state matches reference', met: reward.db_check.db_match })
  for (const check of reward.communicate_checks || []) checks.push({ label: `Communication: ${check.info}`, met: check.met, reason: check.justification })
  for (const check of reward.nl_assertions || []) checks.push({ label: check.nl_assertion, met: check.met, reason: check.justification })
  for (const check of reward.env_assertions || []) checks.push({ label: `Environment: ${display(check.env_assertion)}`, met: check.met })
  for (const check of reward.action_checks || []) checks.push({ label: `Action: ${check.action?.name || 'Unknown action'}`, met: check.action_match, reason: display(check.action?.arguments || {}) })
  const failed = checks.filter(check => check.met === false)
  const passed = checks.filter(check => check.met === true)
  const notes = Object.entries(reward.info || {}).filter(([, value]) => value != null)
  const hasDetails = components.length > 0 || checks.length > 0 || notes.length > 0

  const renderChecks = items => <ul className="reward-checks">{items.map((check, index) => (
    <li key={index}>
      <strong>{check.met ? 'Passed' : 'Failed'}: {check.label}</strong>
      {check.reason && <pre>{check.reason}</pre>}
    </li>
  ))}</ul>

  return (
    <details className="reward-details">
      <summary>Reward details{failed.length > 0 ? ` · ${failed.length} failed check${failed.length === 1 ? '' : 's'}` : ''}</summary>
      <p>Recorded grader output. Checks can include diagnostics that do not count toward the reward.</p>
      {components.length > 0 && <ul className="reward-components">{components.map(component => (
        <li key={component}>
          <strong>{labels[component] || component}</strong>: {breakdown[component] ?? 'Score not recorded'}
          {basis.length > 0 && <span> ({basis.includes(component) ? 'counts toward reward' : 'diagnostic only'})</span>}
        </li>
      ))}</ul>}
      {failed.length > 0 && <section aria-label="Failed reward checks"><h5>Failed checks</h5>{renderChecks(failed)}</section>}
      {passed.length > 0 && <details><summary>Passed checks ({passed.length})</summary>{renderChecks(passed)}</details>}
      {notes.length > 0 && <details><summary>Grader notes</summary>{notes.map(([name, value]) => <div key={name}><strong>{name}</strong><pre>{display(value)}</pre></div>)}</details>}
      {!hasDetails && <p>Detailed grading information was not recorded in this trajectory.</p>}
      {sourceUrl && <a href={sourceUrl} target="_blank" rel="noopener noreferrer">Raw trajectory JSON</a>}
    </details>
  )
}
