# τ-bench 1.0.1 — banking_knowledge Grading Fixes

**⚠️ Grading change: `banking_knowledge` scores are not comparable across this release.** This release bundles the recent `banking_knowledge` task and grading fixes: a systematic penalty that zeroed rewards for correct-but-cautious agent behavior, plus task-data corrections. Under the grading-scheme fixes, re-grading the leaderboard trajectory sets moves scores **only upward** — no previously-passing simulation fails — by up to ~9 points pass^1 depending on the model. One task-data fix (task_074, #374) corrects a gold refund value and can move that task's trials in either direction. All other domains are unaffected.

## What was wrong

1. **Extra read calls zeroed the reward** (#329). Every `call_discoverable_agent_tool` call was logged to a DB table that participates in the reward's DB-hash comparison. One prudent verification read not present in the golden trajectory — e.g. listing a user's accounts right after opening one — failed the task, even though the knowledge base encourages such reads. Reads are now logged only when the task's golden trajectory requires them, so required-read assertions still discriminate while extra validation reads no longer poison the hash. This fix accounts for nearly all of the score movement.
2. **Int-vs-float argument spelling changed DB hashes** (#397). `25` and `25.0` are the same JSON number but produced different deterministic record IDs and DB states. Numeric tool arguments are now normalized.
3. **Tasks 077–086 gold trajectories were not agent-realizable** (#402). The lost/stolen card scenarios omitted reads any real agent must perform; the golden trajectories now include them.
4. **Bank account transactions were returned oldest-first, contradicting the docs** (#403). The knowledge base documents `get_bank_account_transactions_9173` as most-recent-first, but the tool returned raw DB insertion order. This made the "dispute the earliest duplicate" tie-breaker unresolvable from tool output on the duplicate-charge tasks (083–085). Output is now sorted by date descending (stable sort), and task_085's fixture rows were aligned to the same tie-break convention as 083/084. No gold actions changed.
5. **Contradictory cash-back rate in the Platinum Rewards knowledge doc** (#388). The document stated two different rates for the same card; the numbers now agree.
6. **task_074 under-refunded Light Blue ATM fees against its own policy docs** (#374). The Light Blue Account docs grant two free out-of-network and two free foreign ATM withdrawals per month, but the gold refund honored only one of each ($8.00 instead of $14.50). The fixture's fee descriptions also embedded the answer key — annotations like "(SHOULD BE FREE - 1ST OF 2)" and "(AFTER 2 FREE)" told the agent which fees were erroneous. The gold credit is now the policy-faithful $14.50 and all twelve annotated descriptions are scrubbed to plain statement text ("NON-RHO ATM FEE"/"FOREIGN ATM FEE"), so the agent must derive the free allowance from the policy docs; four "AFTER FREE ALLOWANCE" annotations of the same kind on the task_072/073 Light Green accounts are scrubbed too (description-only, no grading impact). This is a gold-value change: trajectories that reproduced the old $8.00 refund fail task_074 under 1.0.1, while policy-faithful $14.50 refunds now pass.

## Measured impact (leaderboard re-grade)

| Model | pass^1 (old) | pass^1 (new) | Δ pass^1 | pass^4 (old) | pass^4 (new) | flips |
|---|---|---|---|---|---|---|
| gpt-5-5 | 37.37 | 46.39 | +9.02 | 20.62 | 27.84 | +35/-0 |
| gpt-5-4 | 30.67 | 39.43 | +8.76 | 16.49 | 21.65 | +34/-0 |
| gpt-5-2 | 24.74 | 32.22 | +7.48 | 11.34 | 18.56 | +29/-0 |
| claude-opus-4-7 | 25.26 | 30.15 | +4.89 | 12.37 | 15.46 | +19/-0 |
| claude-opus-4-6 | 24.48 | 27.32 | +2.84 | 10.31 | 11.34 | +11/-0 |
| gemini-3-flash | 20.62 | 27.32 | +6.7 | 4.12 | 7.22 | +26/-0 |
| gemini-3-1-pro-preview | 22.54 | 26.03 | +3.49 | 8.42 | 9.28 | +14/-0 |
| claude-sonnet-4-5 | 22.42 | 25.26 | +2.84 | 10.31 | 10.31 | +11/-0 |
| claude-opus-4-5 | 21.39 | 24.74 | +3.35 | 8.25 | 11.34 | +13/-0 |
| gemini-3-pro | 15.72 | 18.04 | +2.32 | 4.12 | 4.12 | +9/-0 |
| grok-4-2 | 17.57 | 18.04 | +0.47 | 7.29 | 8.25 | +2/-0 |
| grok-4-fast | 14.18 | 15.72 | +1.54 | 4.12 | 4.12 | +6/-0 |
| gemini-2-5-pro | 12.76 | 13.66 | +0.9 | 1.08 | 1.03 | +4/-0 |
| grok-4-1-fast | 12.4 | 13.14 | +0.74 | 5.21 | 5.15 | +3/-0 |
| gpt-5-2-none | 11.08 | 12.63 | +1.55 | 4.12 | 4.12 | +1/-0 |

_No simulation flipped from pass to fail; every reward change is upward. pass^k values are recomputed counting infrastructure-error simulations as failed trials (the leaderboard convention); the sub-0.1-point pass^4 dips for gemini-2-5-pro and grok-4-1-fast come from that convention alignment, not from any score flip. glm-5-think's trajectory file contains only 3 trials for some tasks, so pass^4 is not recomputable; its previous value is retained. NL-assertion judgments are preserved from the original grading runs (the v1.0.1 fixes are all environment-side), so the re-grade is fully deterministic._

## Re-grading your own results

```bash
tau2 evaluate-trajs --fresh-tasks path/to/results.json -o regraded/
```

`--fresh-tasks` (new in 1.0.1) grades against the current task definitions instead of the ones embedded in your results file. Re-grading also now applies the same per-task read-log allowlist as live grading, and replays trajectories recorded before 1.0.1 leniently so cosmetic tool-output drift (e.g. `25` vs `25.0` argument echoes) no longer aborts the replay. Live evaluation remains strict.

## Reproducing pre-1.0.1 behavior

Most of these fixes were merged to `main` on July 14–15, 2026, ahead of this release and without a version bump, so recent installs from `main` already include them. To reproduce the pre-fix `banking_knowledge` behavior (e.g. to match scores published before this release), pin the `pre-v1.0.1` tag — the last commit before the fix series landed (`b51a6d6`):

```bash
pip install git+https://github.com/sierra-research/tau2-bench@pre-v1.0.1
```

Grading is identical at every earlier 2026 commit of `main` (verified back to `2be6916`, April 2026); changes in that window are simulation-side only (task_053 `user_tools` fix, default retrieval config, prompts), so any pin in that range re-grades recorded trajectories identically.

See [CHANGELOG.md](CHANGELOG.md) and [RELEASE_NOTES.md](RELEASE_NOTES.md) for details.
