// Shared helpers for the τ-i18n voice views.
export const mean = values => {
  const present = values.filter(value => value !== null && value !== undefined)
  return present.length ? present.reduce((sum, value) => sum + value, 0) / present.length : null
}
export const percent = value => value == null ? '—' : `${value.toFixed(1)}%`
// A point is dominated only if another is at least as good on every axis,
// and strictly better on at least one. Missing measurements cannot compete.
export const frontier = (rows, axes) => rows.filter(row => axes.every(axis => row[axis] != null) && !rows.some(other =>
  axes.every(axis => other[axis] != null && other[axis] >= row[axis]) && axes.some(axis => other[axis] > row[axis])
))

// Heatmap shading relative to the values on screen, so small spreads stay
// visible: the lowest value gets the lightest green and the highest the darkest.
export const heatScale = values => {
  const present = values.filter(value => value != null)
  const low = Math.min(...present)
  const high = Math.max(...present)
  return value => {
    if (value == null) return '#f1f5f9'
    const intensity = high > low ? (value - low) / (high - low) : 0.5
    return `rgba(25, 121, 91, ${0.06 + 0.54 * intensity})`
  }
}
