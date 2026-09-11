# Annotation Packets

Export simulation results to standalone HTML pages for expert human review of
voice calls, via `tau2 annotate packets`.

## Form types (`--form`)

| Form | Contents | Use |
|------|----------|-----|
| `error_analysis` | error source / type / notes | classify the first critical error in failed calls |
| `user_realism` | 8 audio dimensions (1–4) | rate how realistic the synthetic caller is |
| `voice_review` | **combined**: findings **+** audio, one pass | listen once, record user-simulator errors **and** rate the audio |

`voice_review` is the recommended single-pass mode: each call page has a **Findings**
section (error source / type / notes — primarily for user-simulator errors) followed by an
**Audio** section that rates all 8 realism dimensions 1–4, including:

- **Speech Accuracy** — was the text *actually said*? (synthesis fidelity vs. the transcript)
- **Phrasing Naturalness** — was the *wording itself* natural, judged as text? This separates a
  **phrasing** problem (what the simulator wrote) from a **synthesis** problem (how it sounded).

Dimensions live in `voice_user_simulator_rubric.json`; columns auto-track the
rubric (`RealismRow` / `VoiceReviewRow` in `tau2.annotation.models` — a guard
test keeps them in lockstep).

## Quick Start

```bash
# Combined voice-call review (recommended)
uv run tau2 annotate packets --form voice_review \
  --batch-name round1_retail \
  --results data/simulations/my_experiment/results.json \
  --max-items 50

# Error-analysis only (failed calls)
uv run tau2 annotate packets --form error_analysis \
  --batch-name round1_retail \
  --results data/simulations/my_experiment/results.json \
  --filter-reward "< 1"

# Open in browser
open data/annotations/round1_retail/index.html
```

## Filtering & sampling

```bash
--filter-reward "< 1"          # by reward (supports <, <=, >, >=, ==, !=)
--filter-tasks 9 16 31         # by task ID
--shuffle --seed 42            # seeded random sample before --max-items
--max-items 50                 # cap number of exports
--append                       # add new sims to an existing packet (dedupes by sim id)
```

Run `tau2 annotate packets --help` for the full CLI reference.

## Output

```
data/annotations/round1_retail/
├── index.html              # Landing page with progress tracking
├── manifest.json           # Provenance + per-conversation entries (the batch's source of truth)
├── task_9_sim_f61e6c15/
│   ├── index.html           # Annotation page
│   └── audio.wav            # Audio (if available)
└── ...
```

## Annotator Workflow

> Full annotator-facing guide (task, error source/type categories, the 8 audio
> dimensions, workflow, edge cases): [`VOICE_REVIEW_INSTRUCTIONS.md`](VOICE_REVIEW_INSTRUCTIONS.md)
> (also embedded in-app via the **📖 Guidelines** button).

1. Open `index.html` in a browser, enter your name (annotations are stored per rater)
2. Click a simulation to open its review page
3. Review the conversation transcript, listen to the audio, check the expected actions /
   reward / LLM-judge review
4. Fill in the form. For `voice_review`: record any error in **Findings**, then rate all 8
   **Audio** dimensions (1–4; scores ≤2 require a one-line note)
5. Mark as Complete — this also auto-downloads a backup CSV
6. When finished, click "Export All to CSV" on the index page

Annotations are saved in browser localStorage and survive reloads. Use **Import CSV** on the
index page to restore from a backup, so a crash never loses more than the current call.
