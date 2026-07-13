# [Voice / Airline] Humains Cascaded Voice Agent — Pass^1 64.0 (humains.com)

## Summary

Voice (FULL_DUPLEX, audio-native) leaderboard submission for a **custom cascaded voice agent** on the **airline** domain. Single trial, **Pass^1 = 64.0 (32/50)**, regular speech complexity, all tasks evaluated.

This is a *cascaded* agent, not a native audio-realtime model:

> **Soniox STT (stt-rt-v5) → Humains bCortex multi-state orchestrator (gemini-3.5-flash on Vertex AI, thinkingLevel=low) → OpenAI TTS (gpt-4o-mini-tts, voice `alloy`)**, with a turn-taking politeness adapter between the bench and the orchestrator.
>
> *Verified against logs: STT/brain/agent-TTS/user-TTS/tick/complexity all match. Vertex `thinkingLevel=low` was primary; a same-model AI-Studio fallback engaged on ~1% of inferences under transient Vertex 429 rate-limiting (that fallback path runs uncapped thinking).*

Per the voice guide, I'm opening this PR and contacting you to coordinate a re-run with Sierra's held-out voices. Trajectory data (results.json + simulations/ + audio for all 50 tasks) is hosted for verification — link below.

## What's in the PR

- `web/leaderboard/public/submissions/humains-cascaded-voice-agent_humains-com_2026-07-13/submission.json`
- `manifest.json` updated (`voice_submissions` array)

## Hosted trajectories (for verification)

- **Bundle** (results.json + `simulations/` ×50 + `artifacts/*/audio/` ×50, tau2 directory format, ~190 MB gz):
  `https://github.com/InprisAI/humain-studio/releases/download/tau-voice-lean-low-64-2026-07-13/tau_voice_lean_low_64_bundle.tar.gz`
- The merged `results.json` passes `Results.model_validate` (50 sims).

## Methodology & required disclosures

- **Submission type: custom.** Modified scaffold (cascaded pipeline), custom multi-state prompts with airline doctrine, and a custom turn-taking adapter (dead-air fillers, liveness/ear watchdogs, stale-drop / soft-yield / yield-truncate, 2-second collision-protection rule, uninterruptible reassurance, backchannel gate, fuzzy dedupe; barge-in disabled). `modified_prompts: true`.
- **Reproducibility caveat (important).** These trajectories were produced against a **locally-patched bCortex core** — two conversation-history changes (interrupted-COT tool-result carryover, and a "lean-trail" history that keeps tool results but drops intermediate chain-of-thought hops). The submitted trajectories faithfully reflect this patched agent, but the score is **not reproducible on the stock/deployed platform** without those edits. Flagging up front so you can decide how you want to treat it.
- **User-sim TTS substitution.** Held-out ElevenLabs voices were substituted with Soniox persona-equivalents (ElevenLabs credits exhausted). Expecting Sierra to re-run with the held-out voices per the case-by-case voice policy.
- **Single trial, Pass^1 only** (audio-native multi-trial is expensive); higher Pass^k are null.
- **Infra re-runs.** 9 tasks hit a transient OpenAI user-simulator connection error and were cleanly re-run; only the clean re-runs are in the board.

## Results

| Domain | Pass^1 | Trials | Complexity |
|---|---|---|---|
| airline | 64.0 (32/50) | 1 | regular |

### Attribution
Same submitting entity as PR #368 (text / airline). The submitted system is a **Humains agent** (cognitive architecture by **humains.com**) running on the base LLM **gemini-3.5-flash, developed by Google**. The base model is Google's; the agent / cascaded voice layer is Humains'.

Contact: **Nissan Yaron** <nissan@humains.com> (GitHub `Nisyron`).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
