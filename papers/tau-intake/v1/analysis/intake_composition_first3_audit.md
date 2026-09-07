# Composition-run audit: balanced first-three-trial analysis

Date: 2026-09-07

## Decision

The paper uses trials 0, 1, and 2 for both flat composition arms. This is a
balanced, post-hoc inclusion rule: the n=2 root was launched with three trials,
whereas the n=3 root was launched with four. Timestamps show that all four n=3
workers ran concurrently, so trial 3 was not added after observing an earlier
result.

## Frozen sources

- n=2: `data/simulations/paper_runs/tau-elicit/ablations/entity_composition/intake_ecomp_n2/results.json`
- n=3: `data/simulations/paper_runs/tau-elicit/ablations/entity_composition/intake_ecomp_n3/results.json`
- single-field reference: `data/simulations/paper_runs/tau-elicit/main_runs/intake_m_openai_xhigh_regular/results.json`
- authoritative compose manifest: `data/tau2/domains/intake/bands/compose.manifest.json`
  at run commit `ce74dff8f099fb8560b814592d76e3847ca37ee6`

Both composition roots record seed 42, GPT Realtime 2 with xhigh reasoning,
the regular speech environment, complication rate 0, and a `gpt-5.5`
low-reasoning caller. The only root-level difference is the task split and the
configured trial count (3 for n=2, 4 for n=3).

The 200-task single-field row is a contextual scaffolded reference (154/200),
not a matched count-effect arm. It uses a higher caller reasoning setting and a
shorter duration cap than the composition roots. Accordingly, the defensible
composition result is the within-arm separation between field accuracy and
all-fields-correct task accuracy; the n=1-to-n=3 slope is descriptive.

## Included results

| Arm | Trial | Task passes | Task Pass@1 | Correct fields | Field Pass@1 |
|---|---:|---:|---:|---:|---:|
| n=2 | 0 | 36/60 | .600 | 89/120 | .742 |
| n=2 | 1 | 44/60 | .733 | 100/120 | .833 |
| n=2 | 2 | 45/60 | .750 | 101/120 | .842 |
| **n=2 pooled** | **0--2** | **125/180** | **.694** | **290/360** | **.806** |
| n=3 | 0 | 19/30 | .633 | 73/90 | .811 |
| n=3 | 1 | 18/30 | .600 | 71/90 | .789 |
| n=3 | 2 | 11/30 | .367 | 58/90 | .644 |
| **n=3 pooled** | **0--2** | **48/90** | **.533** | **202/270** | **.748** |

Field outcomes were reconstructed from the actual `submit_fields` payloads in
each simulation and fold-compared with the seeded gold values. Whether all
fields were correct agreed with the stored task reward for every included
call.

## Quality checks

- Every included trial has every intended task exactly once: 60 tasks for n=2
  and 30 for n=3. There are no null rewards or infrastructure-error
  terminations.
- The low n=3 trial 2 is not a worker failure: errors are spread across
  strata, with the largest drop on hard tasks. Its caller-voice draw is
  unusually Arjun-heavy (10/30), but all per-call settings match the other
  trials.
- The n=2 root contains one scored `max_steps` termination. Four initial
  caller-hallucination attempts were discarded and replaced; the frozen result
  set is complete.
- The current checked-in compose manifest is not used for this audit. A later
  catalog-2.4 redraw changed `intake_c2_auto_hard_10` from a VIN parent to a
  license-plate parent. The task embedded in the run matches the manifest at
  `ce74dff8`.

## Excluded fourth n=3 trial and sensitivity

Trial 3 has 21/30 task passes (.700) and 78/90 correct fields (.867). Pooling
all four n=3 trials would produce 69/120 task passes (.575) and 280/360 correct
fields (.778). The balanced first-three rule lowers the point estimates but
does not change the finding that whole-task accuracy degrades faster than
per-field accuracy as fields are composed.

## Run and PR history

PR #1006 introduced the composition task generator and metric but explicitly
states that it launched no experiment runs. The frozen results were produced
later at commit `ce74dff8`. Repository history contains no launch command or PR
record explaining why n=3 requested four trials while n=2 requested three;
the best-supported explanation is an invocation-level trial-count mismatch.
