# Conversation judges

`tau2 judges legacy conversation` runs the LLM conversation judges (the
EVA-ported conversation-progression and conciseness judges) over stored calls
into a typed, resumable sidecar artifact; it does not overwrite the run's
reward, nativeness, delivery, or legacy `quality_info` fields. The verb lives
under the demoted `tau2 judges legacy` group: fully usable, explicitly
invoked, out of the standard judging flow.

```bash
tau2 judges legacy conversation data/simulations/my-run \
  --out artifacts/conversation.json
```

The organizing rule (binding decision, 2026-08-18): **deterministic
computation lives under `tau2 metrics`; LLM-judged computation lives under
`tau2 judges`.** The deterministic interaction facts — latencies, missed
responses, interruption behavior, route mix, selectivity, dead air — are the
paper's headline and come from `tau2 metrics interaction-facts` (see
[Voice Interaction Metrics](interaction-metrics.md)). EVA-Bench-style
thresholds and composite scores are arbitrary and lossy, especially across
languages, so **every composite is a permanent shadow column**: still
computed and version-stamped for comparability, labeled
`uncalibrated-shadow`, and never ranked or headlined.

## Headline: per-dimension and per-turn

The artifact's per-call headline is:

- `progression_dimensions` — the four closed EVA conversation-progression
  dimensions (unnecessary tool calls, information loss, redundant statements,
  question quality), each with `flagged`, a 1–3 severity rating, and brief
  evidence. The rule-derived progression composite is floor-saturated on the
  paper corpus (means 0.00–0.14), which is exactly why the dimensions are the
  headline and the composite is shadow-only.
- `conciseness_turns` — one verdict per spoken agent turn with a 1–3 rating
  and closed-set failure modes (`verbosity_or_filler`,
  `excess_information_density`, `over_enumeration_or_list_exhaustion`,
  `contextually_disproportionate_detail`).

In-verb cell aggregates (`cells`, per language × domain) report per-dimension
flag rates and mean ratings plus per-failure-mode turn rates, so downstream
tables need no ad-hoc scripts. The CLI report prints these headline
aggregates first and the shadow means under a separate, labeled section.

## Shadow: the demoted composites

`shadow_scores` (per call, `label: "uncalibrated-shadow"`) carries:

- `eva_x` — the complete versioned `eva-x-tau-v3` adaptation of ServiceNow
  [EVA](https://github.com/ServiceNow/eva), computed exactly as before the
  facts-first conversion: the three component scales (turn-taking ≥ 0.8,
  conversation progression ≥ 0.5, conciseness ≥ 0.5) and the conjunctive
  pass rule. The call passes only if every component with evidence passes; a
  component with no evidence scores `None` (N/A) and is excluded — zero
  evidence is never a fail — and the conjunction is `None` only when every
  component is N/A. The embedded turn-taking component is the canonical
  deterministic scorer from `tau2.metrics.turn_taking` (typed reasoned N/A
  included: `no_scoreable_turns`, `agent_mute`).
- `ours` — the τ universal quality composite (`QualityInfo`), on by default
  (`--no-ours` to skip; `--no-ours-llm-judge` for its deterministic-only
  diagnostic).

Both are stamped: EVA source version, τ adaptation version, prompt version,
multilingual-policy version, model settings, git commit, input paths, and a
hash of every input simulation.

## Judges

Conversation progression retains the four EVA dimensions; the overall 1–3
rating is derived from the dimension flags, then normalized to 0, 0.5, or 1
(shadow only). Conciseness is judged per spoken agent turn with the full
conversation in context; ratings 1–3 normalize to 0, 0.5, and 1, then average
across turns (shadow only). The judge explicitly considers information
density, over-enumeration, filler, and contextual proportionality.

Contradictory judge output (a flagged dimension rated clean, or the reverse)
is a validation error that lands on the per-call error channel, never a
silent downgrade. A call with no spoken agent turns has no conciseness
evidence: N/A, never a zero-fail.

### Multilingual policy

`eva-x-multilingual-v1` keeps EVA's numeric formulas and thresholds fixed
while making the semantic interpretation language-aware. Judges evaluate the
language actually spoken and may not impose English brevity or
turn-organization norms. They must not treat obligatory honorifics,
politeness, discourse particles, grammatical morphology,
language-appropriate confirmation, or context-required repetition and
explicitness as filler merely because English would express the same content
more compactly.

## Caching and the error channel

Paid judge units are cached beside the output by default; pass `--cache-dir`
for a stable shared location or `--force` to deliberately re-judge. Every
conciseness turn, progression call, and τ quality result has an independently
validated cache. The unit digest covers exactly the judge-relevant inputs
(trajectory, ticks, effective policy, task contract, domain, language) so an
in-place re-evaluation that rewrites rewards cannot torch it, and the cache
namespace is keyed to the run-dir basename plus run metadata — never the
absolute path — so relocating run directories keeps the cache. The cache
envelope, component ids, and versions are unchanged from the retired
`tau2 judges experience` verb, so judgments paid for under the old verb are
reused, never re-bought.

Broken inputs never abort the batch: a simulation with a missing task
contract, a duplicate identity, or a multilingual persona whose language
cannot be resolved is skipped with a warning and recorded on the per-call
error channel (`suite: "input"`); mislabeled multilingual calls are never
silently judged under English norms.

Nativeness and acoustic delivery remain separate axes and are not folded in.
