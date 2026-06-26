# banking_knowledge in voice — verification & findings

**Date:** 2026-06-26 · **Branch:** `soham/banking-knowledge-voice`

First voice (audio-native, OpenAI `gpt-realtime-2`) run of the `banking_knowledge`
RAG domain. **Verified working with zero source-code changes** for two retrieval
configs — `golden_retrieval` (oracle: task docs inlined) and `terminal_use`
(agentic: sandboxed `shell` tool over the 698-doc KB). This doc captures how to run
it, the setup gotchas, and what we learned.

## TL;DR
- Knowledge injection into the voice model works for **both** configs — no code change needed.
  `golden_retrieval` inlines docs into the realtime session `instructions`; `terminal_use`
  serializes the `shell` tool into `tools` and the realtime agent really does `grep` the KB
  and act on results.
- **In voice, golden's oracle advantage disappears.** Text: golden 0.40 ≫ terminal 0.16.
  Voice: golden 0.20 ≈ terminal 0.20. The voice bottleneck is *reliably executing actions in
  a spoken multi-turn dialog*, not retrieval quality.
- DB reward is brittle (all-or-nothing whole-DB hash; 88/97 tasks DB-only) — interpret absolute
  numbers with care; even the oracle text ceiling is only 0.40.

## Setup (one-time, local environment — not repo changes)
- `npm i -g @anthropic-ai/sandbox-runtime@0.0.23` — provides `srt`, required by `terminal_use`.
- `brew install ripgrep` — must be a **real `rg` binary**. Gotcha: some environments expose `rg`
  only as a shell function (e.g. Claude Code bundles it), so `srt`'s `shutil.which("rg")` check
  fails with `Error: Sandbox dependencies are not available... Required: ripgrep (rg)`. A real
  binary on `PATH` fixes it. (`brew install` may need `HOMEBREW_NO_REQUIRE_TAP_TRUST=1` if an
  unrelated tap is untrusted.)
- `--save-to` is auto-prefixed with `data/simulations/` — pass a **bare** relative path.
- `TAU2_FORCE_LLM_COMMUNICATE_JUDGE=1` — for valid English `communicate_info` scoring (matches
  the repo's own English preset).

## How to run
- `scripts/banking_voice_run.sh` — text + voice × {golden_retrieval, terminal_use}, 25 tasks,
  seed 42. Idempotent (`--auto-resume`). `REAL_CFG` env overrides the non-golden config.
- `scripts/banking_terminal_retry.sh` — re-run only the `terminal_use` phases at low concurrency
  (`CONC=2`). `terminal_use` is fragile under load (see findings); use this if it stalls.
- `scripts/analyze_banking.py <results_dir> ...` — consolidated report: DB reward, pass^1,
  termination reasons, retrieval-tool-call evidence, sandbox-error + audio sanity.

## Results (25 tasks, seed 42, agent = `gpt-realtime-2` voice / `gpt-4.1` text)

| Retrieval config | Mode | Avg DB reward | Pass^1 | Action-match | KB shell calls | Audio |
|---|---|---|---|---|---|---|
| `golden_retrieval` | text | 0.400 | 0.40 | 0.478 | 0 (inlined) | — |
| `golden_retrieval` | voice | 0.200 | 0.20 | 0.315 | 0 (inlined) | 27/27 |
| `terminal_use` | text | 0.160 | 0.16 | — | 754 / 25 sims | — |
| `terminal_use` | voice | 0.200 | 0.20 | 0.307 | 280 / 23 sims | 26/26 |

## Findings
1. **Wiring verified in voice (both configs).** `terminal_use` voice issued 280 `shell` calls
   across 23/25 sims, grepping the KB and acting on real returned content; 0 sandbox errors.
   `golden` voice issued 0 retrieval calls (correct — docs already in context).
2. **Golden's oracle edge collapses in voice — and it's behavioral, not just DB-hash noise.**
   Action-match (did the agent take the correct actions, independent of the DB hash) falls from
   0.478 (golden text) to 0.315 (golden voice), landing on top of terminal voice's 0.307. Tasks
   golden nails in text but fails in voice show action-match collapsing (e.g. task_007 1.0→0.0,
   task_025 1.0→0.0). The agent has the docs; it just doesn't reliably execute multi-step actions
   during a messy spoken conversation.
3. **DB reward is brittle.** Exact hash of the entire DB (agent + user views); 88/97 tasks are
   `reward_basis=["DB"]`. Absolute rewards understate behavioral correctness; even the oracle
   text ceiling is 0.40.
4. **`terminal_use` is LLM-call-heavy (~30 agentic shell turns/sim) → fragile under concurrency.**
   First run at conc 6–8 (alongside a competing realtime run) stalled: realtime websocket drops
   (`code 1006` / "Not connected to API") in voice, and `gpt-4.1` RPM-limit `infrastructure_error`
   in text. Re-running at **conc 2** fixed it cleanly. Note: `--timeout` did not reliably kill hung
   realtime sessions under high concurrency.

## Caveats / next steps
- N=25, 1 trial. The golden≈terminal *equality* in voice is partly noise (SE ≈ 0.08); they pass
  different task sets (3 shared, 2 golden-only, 2 terminal-only). What's robust is that golden's
  clear text advantage is **gone** in voice. To actually resolve golden-vs-terminal in voice, run
  the full 97 tasks or ≥3 trials.
- Raw simulation outputs are kept on disk under `data/simulations/banking_{text,voice}/…`
  (gitignored, not part of this PR).
