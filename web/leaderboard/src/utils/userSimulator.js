/**
 * Voice user-simulator versions.
 *
 * Voice submissions must name a published version (`v1.0`, `v2.0`, …) rather than
 * an arbitrary model — see VOICE_USER_SIMULATOR_VERSIONS in src/tau2/config.py.
 * Each version pins the simulator's LLM, so the version string in the User Sim
 * column is itself the comparability signal.
 *
 * Text submissions are unconstrained and record their simulator model directly.
 */

/**
 * The version the leaderboard currently ranks on. Voice scores are only
 * comparable within a version (v2.0 is a stronger simulator than v1.0), so when
 * a model has runs on several versions the progress chart plots this one.
 * Bump alongside VOICE_USER_SIMULATOR_VERSION when the field is re-run.
 */
export const RANKING_VOICE_USER_SIM = 'v1.0'

/** True for a published voice version identifier, e.g. "v1.0". */
export const isVoiceUserSimVersion = userSimulator =>
  !!userSimulator && /^v\d/.test(userSimulator)

/**
 * Suffix for compact listings (dropdowns), where several voice rows can share a
 * model name and differ only by simulator version.
 */
export const voiceUserSimLabel = userSimulator =>
  userSimulator ? ` — user sim: ${userSimulator}` : ''
