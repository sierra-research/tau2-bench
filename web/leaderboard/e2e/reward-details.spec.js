import { expect, test } from '@playwright/test'

const task = { id: '11', description: { purpose: 'Refund test' }, user_scenario: { instructions: { reason_for_call: 'Downgrade my reservation' } } }
const reward = {
  reward: 0, reward_basis: ['DB', 'COMMUNICATE'], reward_breakdown: { DB: 1, COMMUNICATE: 0 },
  db_check: { db_match: true, db_reward: 1 },
  communicate_checks: [{ info: '5244', met: false, justification: "Information '5244' not communicated." }],
  action_checks: [{ action: { name: 'update_reservation_flights', arguments: { cabin: 'basic_economy' } }, action_match: true }],
}

async function load(page, modality, rewardInfo = reward) {
  const simulation = { id: 'example', task_id: '11', trial: 0, reward_info: rewardInfo, messages: [], ticks: [], duration: 1, termination_reason: 'user_stop' }
  await page.route('**/submissions/**', async route => {
    const url = route.request().url()
    let body
    if (url.endsWith('/manifest.json')) body = { submissions: modality === 'text' ? ['example'] : [], voice_submissions: modality === 'voice' ? ['example'] : [] }
    else if (url.endsWith('/submission.json')) body = { model_name: 'Example', trajectories_available: true, trajectory_files: { airline: modality === 'voice' ? 'airline' : 'airline.json' } }
    else if (url.endsWith('/results.json')) body = { tasks: [task], simulation_index: [{ ...simulation, reward: rewardInfo?.reward }] }
    else if (url.endsWith('/simulations/example.json')) body = simulation
    else if (url.endsWith('/airline.json')) body = { tasks: [task], simulations: [simulation] }
    else return route.fulfill({ status: 404, body: '' })
    await route.fulfill({ json: body })
  })
  await page.goto('/trajectory-visualizer?model=example&domain=airline&task=11')
  const panel = page.locator('.reward-details')
  await expect(panel).toBeVisible()
  await panel.locator('summary').first().click()
  return panel
}

for (const modality of ['voice', 'text']) {
  test(`${modality}: explains failed reward and keeps passed checks collapsed`, async ({ page }) => {
    const panel = await load(page, modality)
    await expect(panel).toContainText('Database: 1 (counts toward reward)')
    await expect(panel).toContainText('Communication: 0 (counts toward reward)')
    await expect(panel.getByText("Information '5244' not communicated.")).toBeVisible()
    await expect(panel.getByText('Passed: Database state matches reference')).not.toBeVisible()
    await panel.getByText('Passed checks (2)', { exact: true }).click()
    await expect(panel.getByText('Passed: Database state matches reference')).toBeVisible()
    await expect(panel.getByText('Passed: Action: update_reservation_flights')).toBeVisible()
    const source = panel.getByRole('link', { name: 'Raw trajectory JSON' })
    await expect(source).toHaveAttribute('href', modality === 'voice' ? /\/simulations\/example.json$/ : /\/airline.json$/)
  })
}

test('summary-only reward does not fabricate component grades', async ({ page }) => {
  const panel = await load(page, 'voice', { reward: 0 })
  await expect(panel).toContainText('Detailed grading information was not recorded')
  await expect(panel.locator('.reward-components')).toHaveCount(0)
})

test('missing reward and grader notes are handled', async ({ page }) => {
  const panel = await load(page, 'text', null)
  await expect(panel).toContainText('Detailed grading information was not recorded')
})

test('recorded diagnostics and notes remain distinct from reward components', async ({ page }) => {
  const panel = await load(page, 'text', {
    ...reward, reward_breakdown: { DB: 1, COMMUNICATE: 0, ACTION: 1 },
    nl_assertions: [{ nl_assertion: 'Explain refund', met: false, justification: 'Explanation incomplete' }],
    env_assertions: [{ env_assertion: { field: 'cabin', value: 'economy' }, met: false }],
    info: { note: 'Simulation terminated prematurely' },
  })
  await expect(panel).toContainText('Actions: 1 (diagnostic only)')
  await expect(panel.getByText('Explanation incomplete')).toBeVisible()
  await panel.getByText('Grader notes', { exact: true }).click()
  await expect(panel.getByText('Simulation terminated prematurely')).toBeVisible()
})
