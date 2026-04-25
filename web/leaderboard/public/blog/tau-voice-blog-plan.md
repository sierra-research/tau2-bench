# τ-voice Blog Plan

Goal: write an engaging technical blog post about τ-voice in `web/leaderboard/public/blog/tau-voice-wip.html` (suffixed `-wip` so it ships with the leaderboard but is not linked from anywhere — share the URL directly).

## Status

- **Done.** Self-contained `tau-voice-wip.html` shipped, modeled on `tau-knowledge.html`.
- Reuses two existing assets:
  - `tau-voice-figure.html` — embedded as iframe (Speech Activity Timeline).
  - `tau-voice-examples.html` — linked from a CTA banner + the closing CTA.
- Nav dropdown updated everywhere the blog dropdown lives:
  - `web/leaderboard/src/App.jsx` (React app shell)
  - `web/leaderboard/public/blog/tau-knowledge.html`
  - `web/leaderboard/public/blog/tau-voice-examples.html`
  - `web/leaderboard/public/blog/tau3-task-fixes.html`
- Validated by serving via `npm run dev` (HTTP 200, ~71 KB) and capturing headless-Chrome screenshots of the full page.

## Source material gathered

- arXiv paper (`/Users/victorbarres/code/tau-voice-paper/tau-voice_arXiv/`): abstract, intro, methods, experiments, results, conclusion, related work.
- Sierra blog announcement on τ³-bench (knowledge + voice).
- Beamer template (visual style cues only).
- Audio examples already in `web/leaderboard/public/blog/audio/{clean,realistic}/`.
- Voice submission JSONs in `web/leaderboard/public/submissions/` — these give the timeline of voice models.

## Voice models on the leaderboard (release date → average pass@1 across 3 domains)

| Date | Provider | Model | Retail | Airline | Telecom | Avg |
|------|----------|-------|--------|---------|---------|-----|
| 2025-08-28 | OpenAI | gpt-realtime-1.0 | 36.0 | 36.0 | 19.3 | 30.4 |
| 2025-12-12 | Google | gemini-live-2.5-flash | 29.8 | 30.0 | 17.5 | 25.8 |
| 2025-12-17 | xAI    | grok-voice-fast-1.0 | 38.6 | 36.0 | 40.4 | 38.3 |
| 2026-02-23 | OpenAI | gpt-realtime-1.5 | 44.7 | 40.0 | 21.1 | 35.3 |
| 2026-03-26 | Google | gemini-3.1-flash-live (high) | 45.6 | 64.0 | 21.9 | 43.8 |
| 2026-03-26 | Google | gemini-3.1-flash-live (minimal) | 26.3 | 42.0 | 17.5 | 28.6 |
| **2026-04-23** | **xAI** | **grok-voice-think-fast-1.0** | **62.3** | **66.0** | **73.7** | **67.3** |

→ A ~23 pp jump in <1 month with xAI's reasoning-enabled voice model. Highlight this.

## Headline numbers from the paper

- Text reasoning ceiling (GPT-5): 85% pass@1.
- Text non-reasoning (GPT-4.1): 54% pass@1.
- Voice **Clean** (paper-era providers): 31–51% (best xAI 51%).
- Voice **Realistic**: 26–38% (best xAI 38%).
- Voice retains only 30–45% of text capability.
- Accents are the most damaging factor on average (–10 pp); xAI loses 38% of clean perf to accents while Google is essentially unaffected.
- Failure analysis: 79–90% of failures are agent errors, not simulator artifacts.
- Voice quality: each provider is best on a different dimension (OpenAI = latency/responsiveness, xAI = selectivity, Google = lowest interrupt rate).

## Blog structure (Anthropic-style: rigorous, didactic, real-world focused)

1. **Header + TL;DR box** — research badge, "April 2026"; TL;DR summarises voice-text gap, rapid progress, open source.
2. **Why voice agents need end-to-end evaluation** — accessibility, telephony, real customer service, the "spell my name in a noisy café" example (paraphrased from the paper's intro vignette).
3. **What's missing in existing benchmarks** — task completion vs conversational dynamics evaluated separately. Visual comparison table (related-work table from paper, but simplified).
4. **Introducing τ-voice** — three contributions: voice agent benchmark on grounded tasks, controllable & realistic user simulator, empirical findings.
5. **Embedded Speech Activity Timeline figure** (iframe of `tau-voice-figure.html`) with caption — this anchors the whole concept of full-duplex realism.
6. **The voice user simulator (anatomy)** — speech generation → TTS persona → audio environment → turn-taking policy. Inline static SVG/CSS diagram.
7. **Headline finding: voice-text gap** — custom HTML bar chart from `combined_comparison_table` data (text vs Clean vs Realistic, by provider, all-domains average).
8. **Realistic conditions matter — what hurts most** — ablation chart (Noise / Accents / Turn-taking) on Retail.
9. **No provider masters both task completion and turn-taking** — voice-quality metric panel (Latency / Responsiveness / Interrupt / Selectivity), short verbal summary per provider.
10. **What's actually going wrong: failure modes** — error-source/type breakdown panel (Voice-Fragile vs Noise-Fragile cohorts), with examples (authentication transcription, hallucinated completions, lost-track-of-multi-step requests).
11. **Listen for yourself** — promote `tau-voice-examples.html` as the place to hear failures and explore the interactive STV viewer.
12. **Models are improving fast** — line chart of voice frontier over time (built from the submissions table above), call out the xAI Grok Voice Think Fast 1.0 jump (67% avg, 73% on telecom). CTA → Progress view in the leaderboard.
13. **Wide adoption** — short paragraph: working with all major audio-native providers (OpenAI, Google, xAI), with logos.
14. **Limitations & what's next** — English-only, TTS-mediated accents, simulator vs real users, future cascaded baselines, accessibility.
15. **Open and reproducible** — links to repo, leaderboard, paper.

## Visualisations to author (HTML/CSS/SVG, reproducible, not screenshots)

- A. Headline 4-bar chart: Text-reasoning (GPT-5) | Text-NR (GPT-4.1) | Best-Voice-Clean (xAI 51%) | Best-Voice-Realistic (xAI 38%).
- B. Per-provider grouped chart (Clean vs Realistic) for Google / OpenAI / xAI on the All-domains row.
- C. Ablation chart on Retail: Clean vs +Noise / +Accents / +Turn-taking / Realistic per provider.
- D. Voice quality 4-metric panel (Latency / Responsiveness / Interrupt / Selectivity) per provider.
- E. Error-source/type stacked breakdown for the two cohorts.
- F. Voice frontier-over-time chart (mini Progress view): line chart with provider-coloured dots from the submissions table.

## Deliverables

- Single-file `tau-voice-wip.html` with everything inline (CSS, SVG, JS) — no extra build step. ✅
- Updates to nav dropdown links across the existing blog pages so the new page is discoverable. ✅

## Implementation notes

- Charts are all hand-rolled (HTML/CSS for bars/grids, inline SVG for the progress chart and the simulator pipeline) — zero JS frameworks, zero image assets, fully reproducible from the data tables in this plan.
- **Progress chart now mirrors the leaderboard's `ProgressView.jsx` exactly**: 1080×560 viewBox, dashed `#047857` frontier with linear-gradient area fill, white-circle dots with provider-tinted ring + provider logo image inside, frontier-only labels with white halo strokes, legend chips with provider logos, footnote referencing `model_release.release_date`. Reads as a self-contained, identical view of what the live Progress page would show if filtered to voice.
- Error-analysis stacked bars use the cohort decomposition from §6 of the paper (Voice-Fragile vs Noise-Fragile, by error source and error type).
- All paper-era figures are tagged `paper · Feb 2026`; the headline chart explicitly contrasts paper-era voice (38%) vs today's voice (67%) on a single axis with a `LEADERBOARD · THIS WEEK` tag, so the progress story is legible at a glance.
- The "Introducing τ-voice" bullet on Verifiable, grounded tasks now spells out the apples-to-apples comparison: "byte-for-byte identical" tasks/tools/policy/evaluator vs the text leaderboard → full-duplex voice numbers can be read directly against half-duplex text numbers on the same task.
- Limitations cleaned up: removed the patient-simulator bullet, removed the "recorded human accent data is on the roadmap" promise — τ-voice is now scoped explicitly to English + TTS-driven personas.

## Number audit (latest, against `web/leaderboard/public/submissions/*.json`)

Voice (release_date · retail / airline / telecom / **avg**):

| Date | Provider | Model | Retail | Airline | Telecom | **Avg** |
|------|----------|-------|--------|---------|---------|---------|
| 2025-08-28 | OpenAI | gpt-realtime-1.0 | 36.0 | 36.0 | 19.3 | **30.4** |
| 2025-12-12 | Google | gemini-live-2.5-flash-native-audio | 29.8 | 30.0 | 17.5 | **25.8** |
| 2025-12-17 | xAI | grok-voice-fast-1.0 | 38.6 | 36.0 | 40.4 | **38.3** |
| 2026-02-23 | OpenAI | gpt-realtime-1.5 | 44.7 | 40.0 | 21.1 | **35.3** |
| 2026-03-26 | Google | gemini-3.1-flash-live (thinking-high) | 45.6 | 64.0 | 21.9 | **43.8** |
| 2026-03-26 | Google | gemini-3.1-flash-live (thinking-minimal) | 26.3 | 42.0 | 17.5 | **28.6** |
| 2026-04-23 | xAI | grok-voice-think-fast-1.0 | 62.3 | 66.0 | 73.7 | **67.3** |

Text (retail + airline + telecom average, current frontier):

| Model | Retail | Airline | Telecom | **Avg** |
|-------|--------|---------|---------|---------|
| Gemini 3.0 Pro (Google sub) | 85.3 | 73.0 | 98.0 | **85.4** |
| Claude Opus 4.5 | 79.6 | 84.0 | 92.3 | **85.3** |
| Claude Sonnet 4.5 | 86.2 | 70.0 | 98.0 | **84.7** |
| GPT-5.2 (Sierra sub, full reasoning) | 81.6 | 83.0 | 89.7 | **84.8** |
| GPT-5 | 81.6 | 62.5 | 95.8 | **80.0** |
| GPT-4.1 (no reasoning) | 74.0 | 56.0 | 34.0 | **54.7** |

Headline numbers as cited in the blog:
- Text reasoning ceiling: ~85% (matches Gemini 3 Pro / Opus 4.5 / GPT-5.2).
- Text non-reasoning baseline: 54% (GPT-4.1).
- Voice paper-era best (xAI grok-voice-fast-1.0): **38%** (= 38.3 rounded).
- Voice today's best (xAI grok-voice-think-fast-1.0): **67%** (= 67.3 rounded).
- Retention: 38/85 ≈ 45%; 67/85 ≈ 79%.
- Frontier delta over Feb→Apr 2026: 67.3 − 38.3 = +29 pp.

All chart values were re-derived from the JSONs above; the per-provider, ablation, voice-quality, and error-analysis numbers come from the paper and are tagged accordingly in the page UI.
