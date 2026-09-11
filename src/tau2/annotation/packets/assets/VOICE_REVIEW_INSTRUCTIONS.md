# Voice Call Review — Annotator Instructions

Thank you for reviewing these voice calls! This guide explains the task, the
categories you'll use, and the workflow. The same content is available in-app
via the **📖 Guidelines** button (or press **G**) on any review page.

---

## 1. What you're reviewing

Each call is one conversation between:

- **The caller** — a *simulated* customer (an AI user simulator with a voice).
  Synthetic, but meant to behave like a real person on a phone call.
- **The agent** — an AI customer-service agent that answers questions and takes
  actions (look up a reservation, cancel an order, etc.) under a fixed policy.

Your job has **two parts per call**, both filled on the same page:

1. **Findings** — identify the first thing that went *wrong* in the call
   (we care most about **user-simulator** problems, but record agent/system
   problems too).
2. **Audio** — rate how *realistic* the caller sounded on eight 1–4 dimensions.

Always **listen to the audio** — the text transcript alone is not enough. Use
the player at the top of the page; click any transcript row (or an error tick)
to jump the audio to that moment. Keyboard: **Space** play/pause, **←/→** seek 5s.

Each page also shows the **User Instructions** (what the caller was trying to do),
**Agent Policy**, **Expected Actions**, and the **Reward Details** (how the run
was auto-scored) — all collapsible. Open them when you need context for whether
something was actually an error.

---

## 2. Part 1 — Findings (what went wrong)

Pick the **first critical** problem in the call. "Critical" = it prevented the
task from succeeding, derailed the conversation, or made the caller unrealistic
in a way that would break a real call. If the call went fine, leave Findings blank.

### Error Source — *who* caused it

| Value | Use when |
|---|---|
| **Agent Error** | The agent misbehaved — wrong tool call, ignored policy, hallucinated, didn't respond. |
| **User Simulator Error** | The *caller* misbehaved — said the wrong thing, broke character, leaked info it shouldn't know, hung up early, repeated itself, behaved unnaturally. **This is our main focus.** |
| **System Error** | Framework/infrastructure — audio cut out, the call broke, a turn was dropped by the harness rather than by either participant. |

### Error Type — *what kind* of problem

| Value | Meaning |
|---|---|
| **Transcription** | ASR / speech-to-text mistake — what was said vs. what was understood diverged. |
| **VAD** | Turn-taking / interruption — talking over each other, cutting off, or failing to take a turn. |
| **Logical** | Reasoning / tool-call / instruction-following mistake. |
| **Hallucination** | Made-up information not supported by the task, tools, or what was actually said. |
| **Unresponsive** | No response, or a long latency/silence that breaks the call. |
| **Early Termination** | The call ended prematurely (e.g. caller hung up before the task was done). |

Add a short **Summary Note** describing what happened and where (a tick/time
reference helps).

---

## 3. Part 2 — Audio (caller realism)

Rate the **caller** (not the agent) on each dimension, **1–4**:

- **4** = indistinguishable from a real human caller on a phone line.
- **3** = mostly convincing, a few off moments.
- **2** = more artificial than natural.
- **1** = clearly robotic / a recording.

For any score of **1 or 2**, a one-line note is required (what made it unnatural).

| Dimension | What it asks |
|---|---|
| **Voice & Prosody Quality** | Does the voice sound natural in timbre, intonation, pacing, and expressiveness? |
| **Audio Environment Realism** | Does the background/line quality sound like a real phone call? |
| **Turn-Taking Naturalness** | Is the timing of when the caller starts/stops speaking natural? |
| **Backchannel Naturalness** | Are listener signals ("mm-hmm", "haan", "okay") natural and well-timed? |
| **Interruption Behavior** | Are the caller's interruptions natural and well-motivated? |
| **Behavioral Plausibility** | Does the caller behave like a real person (patience, emotion, engagement)? |
| **Speech Accuracy** | Was the text *actually said*? Does the audio faithfully render the caller's transcript (no dropped/garbled words, mispronounced IDs/numbers/names, truncation)? |
| **Phrasing Naturalness** | Is the *wording itself* what a real person would say — judged as text, independent of how it was voiced? |

**Speech Accuracy vs. Phrasing Naturalness vs. Voice & Prosody** — these three
let us separate *where* an unnatural moment comes from:

- **Voice & Prosody / Speech Accuracy** = a **synthesis** problem (how the text
  was turned into audio: robotic delivery, or words mangled/dropped by TTS).
- **Phrasing Naturalness** = a **phrasing** problem (the words the simulator
  *chose* read wrong — stilted, over-formal, or chatbot-like — even if voiced
  perfectly).

When a line sounds off, ask: *would a real person have said these words?*
(phrasing) vs. *did it sound/say them right?* (synthesis).

---

## 4. Workflow

1. Open `index.html`, enter your name (annotations are stored per rater).
2. Click a call. **Listen to the whole thing**, following the transcript.
3. Skim the User Instructions / Expected Actions / Reward Details if you need
   context for whether something counts as an error.
4. Fill **Findings** (or leave blank if the call was clean).
5. Rate all **eight Audio** dimensions; add notes for any 1–2.
6. Check **Mark as Complete** — this also downloads a backup CSV.
7. When done with the batch, click **Export All to CSV** on the index page.

**Your work is saved automatically** in the browser as you go, and again on each
"Mark as Complete." If something goes wrong, use **Import CSV** on the index page
to restore from a backup — you'll never lose more than the call you're on.

---

## 5. Tips & edge cases

- **Transcription vs. Logical** — if the agent acted on a *misheard* word, that's
  Transcription; if it heard correctly but reasoned wrong, that's Logical.
- **VAD vs. Unresponsive** — overlapping/cut-off speech is VAD; dead air with no
  reply is Unresponsive.
- **User hung up vs. Agent failed** — if the caller ended the call before the task
  was done, that's a User Simulator / Early Termination finding, even if the agent
  was slow.
- **Backchannels** are expected during *long* agent turns, not after every short
  reply — absence during a long explanation is more of a problem than silence
  during quick exchanges.
- When unsure between two error types, pick the one closest to the **root cause**
  and explain in the note.
