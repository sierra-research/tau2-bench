export const OUTCOME_COLORS = {
  pass: '#22c55e',
  fail: '#ef4444',
}

export const CLASS_COLORS = {
  stable_pass: '#22c55e',
  stable_fail: '#ef4444',
  bimodal: '#f59e0b',
  fragile: '#8b5cf6',
  moderate: '#6b7280',
}

export const TOOL_TYPE_COLORS = {
  READ: '#3b82f6',
  WRITE: '#ef4444',
  GENERIC: '#6b7280',
  THINK: '#8b5cf6',
  UNKNOWN: '#d1d5db',
}

export const ROLE_COLORS = {
  assistant: '#3b82f6',
  user: '#22c55e',
  tool: '#9ca3af',
  system: '#6b7280',
}

export const DIMENSION_COLORS = {
  outcome_consistency: '#22c55e',
  action_consistency: '#3b82f6',
  sequence_consistency: '#8b5cf6',
  cost_stability: '#f59e0b',
  policy_compliance: '#ef4444',
}

export function scoreColor(score) {
  if (score == null || isNaN(score)) return '#d1d5db'
  if (score >= 0.8) return '#22c55e'
  if (score >= 0.6) return '#84cc16'
  if (score >= 0.4) return '#f59e0b'
  if (score >= 0.2) return '#f97316'
  return '#ef4444'
}
