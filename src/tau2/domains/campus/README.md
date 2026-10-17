# campus — University Academic-Affairs Service Domain (τ²-bench first native Chinese domain)

> Domain package for [τ²-bench](https://github.com/sierra-research/tau2-bench) (MIT), pinned to **v1.0.1** scoring.
> All institutions, people, and records in this domain are **fictional** (the university is "Qingchuan University", a made-up school); policy material is rewritten and de-identified from public regulations. No real personal data exists anywhere in this package.
> **Unofficial**: this is a community-built domain extension — **not affiliated with Sierra Research** (the maintainers of τ²-bench); it is contributed upstream as a pull request.

## 1. Domain overview

`campus` is a natively designed **Chinese** domain for university academic-affairs service (course add/drop, exam deferrals, grade review & appeals, scholarships and aid, certificate issuing, student-status and tickets). It is *not* a translation: policy, database, tasks, and user personas are all written for the Chinese higher-education setting, while the scoring protocol stays exactly on the official τ²-bench v1.0.1 track (`reward_basis = [DB, COMMUNICATE]` + `env_assertions`; `RewardType.ACTION` not used), so scores follow the official v1.0.1 scoring contract; absolute levels are not comparable across different task sets.

- **Policy**: `data/tau2/domains/campus/policy.md` — a 36-article service regulation for the fictional university, deliberately packed with time-boundary traps (the add-drop window, deferral deadlines, the maintenance window, month-end settlement, tiered appeals…).
- **Tasks**: 50 tasks (`tasks.json` + `split_tasks.json`, splits `base` / `easy` / `medium` / `hard`), difficulty 15 / 20 / 15; **29 zero-write tasks (of which 20 carry explicit refusal semantics)** (privacy, skipping appeal levels, out-of-scope requests); every task carries scripted user personas with non-cooperative behavior.
- **Data**: `db.json` (11 business tables + `env`) + `user_db.json` (3 tables), 14 tables in total (students act through their own app).
- **Tests**: `tests/test_domains/test_campus/` — 187 pytest cases (tools, replay, contract, full 50-task gold-replay CI, task self-consistency lint, upstream-contract pins, deadline-guard matrix, state-machine invariants).

## 2. Dual-control design

Following the τ²-bench dual-control protocol, **both** sides hold tools:

| Side | Tools | Role |
|---|---|---|
| Agent (customer-service) | **15** tools (`tools.py`) | look up, apply, withdraw, create tickets… must guide the student to complete multi-step procedures |
| Student (simulated user) | **4** user tools (`user_tools.py`) | upload materials, confirm/sign, and **refuse** — the agent cannot do these *for* the student |

Hard tasks are multi-chain: the student must perform 3–4 actions in their own app while the agent keeps the procedure consistent (e.g. withdrawal requires agent request + student signature). The user simulator is driven by per-task scripted personas drawn from **seven base archetypes** — cooperative, persistent-questioner, change-of-mind, forgetful, emotional, vague-describer, entrenched change-of-mind — **plus composite personas** (e.g. forgetful × change-of-mind), each with an explicit compliance dimension (whether the student complies with agent-proposed proxy actions); the cooperation level is backstopped by the per-task refusal constraints in `task_instructions` (38/50 tasks explicitly refuse agent-proposed proxy actions), so an over-cooperative user cannot silently complete the task for the agent.

## 3. Leaderboard (5 rows / 4 vendors, by pass^4)

Pass^1 = share of passing trials (of 200); pass^4 = tasks passing **4/4** (of 50); 4 trials/task, seed=20261004, temperature=0, max_steps=60; user simulator + NL justification model = DeepSeek-V4.1-Flash.

| # | Model | pass^1 | pass^4 | avg turns | easy / medium / hard |
|---|---|---|---|---|---|
| 1 | DeepSeek V4.1-Flash (anchor) | 0.975 | 0.900 | 6.07 | 1.000 / 0.963 / 0.967 |
| 2 | Qwen3.8-Flash | 0.860 | 0.720 | 5.41 | 0.967 / 0.800 / 0.833 |
| 3 | GLM-5.3 | 0.805 | 0.680 | 5.17 | 0.833 / 0.825 / 0.750 |
| 4 | GLM-5.3-Flash | 0.790 | 0.680 | 5.12 | 0.783 / 0.813 / 0.767 |
| 5 | MiMo-V2.6-Pro | 0.855 | 0.660 | 5.19 | 0.883 / 0.875 / 0.800 |

> **v1.1.1 patch (2026-10-05)**: tasks H01 (two-document illness deferrals, art. 12(2)) and H06 (special-channel 10-workday submission window, the 2026-03-20 anchor being a deliberate design re-anchor — H06's scenario opens while the special-channel window is still live) were updated for policy↔code↔gold consistency and re-run on all rows (40-sim overlay; per-row pre-patch values are preserved in `leaderboard-final.json`). Rows 3/4 swap places at equal pass^4 (0.680), decided by pass^1. See §8 Changelog.

All numbers are recomputed programmatically from `results.json` (never hand-copied). Row 5 includes 1 infrastructure-failed trial in the pre-patch as-run (upstream outage): excluding it gives 0.840 / 0.660 pre-patch and 0.860 / 0.680 on the current v1.1.1 overlay — both calibers are documented. Closed-source flagship models were intentionally not run (cost vs. information gain); the anchor row carries the ceiling reference.

Difficulty tiers are design-time labels (15 easy / 20 medium / 15 hard); observed per-tier pass rates are shown in the table above, and between-tier differences are not statistically separable at this sample size (60–80 trials per tier).

Task-era disclosure: the anchor row was collected on the first generation of assertion strings (r1); the other four rows were collected after the assertion strings were revised. The embedded task copies inside each published run are preserved verbatim from run time apart from the v2.0.0 QA-metadata cleanup; they keep each row's as-run assertion strings and differ from the current published `tasks.json` at statement level (task-era disclosure) and are the primary evidence of each row's grading caliber.

Full disclosure set (every figure recomputed programmatically from the published run assets; the Chinese half of this README carries identical content):

**Task-era detail.**
 The embedded task copies inside the published run files (as-run evidence, frozen) and the currently published task set belong to two different eras. At the statement layer, **50/50 tasks differ**. At the scoring layer (anchor row vs the current release): evaluation_criteria differs on **12 tasks** (E12, H01, H05, H08, H13, H14, H15, M05, M06, M08, M12, M19), **14 including initial_state** (+ H06, M03). **Shared by all five rows: 5 tasks** (M03/H06 = initial_state; M05/H01/H15 = four actions[].info text sites):

| Task | Differs in |
|---|---|
| M03 | initial_state (seed/time-anchor difference) |
| H06 | initial_state (seed/time-anchor difference) |
| M05 | evaluation_criteria.actions[].info text |
| H01 | evaluation_criteria.actions[].info text |
| H15 | evaluation_criteria.actions[].info text |

**Anchor-only: 9 tasks** (the other four rows' embedded scoring fields match the current release; the anchor carries the first-generation assertion strings):

| Task | Differs in |
|---|---|
| E12 | communicate_info assertion string |
| M06 | actions 2->0 |
| M08 | communicate_info assertion string |
| M12 | communicate_info assertion string |
| M19 | communicate_info assertion string |
| H05 | communicate_info assertion string |
| H08 | communicate_info assertion string |
| H13 | communicate_info assertion string |
| H14 | communicate_info assertion string |

The scoring contract itself (deterministic substring matching over COMMUNICATE assertions + DB terminal-state hash + env_assertions as diagnostics) does not change across eras.

**Termination disclosure.**
 Exactly 5 of the 1,000 leaderboard sims ended with a reason other than `user_stop`:

| Row | trial | Reason |
|---|---|---|
| Qwen3.8-Flash | M16 t2 | `too_many_errors` |
| Qwen3.8-Flash | M16 t3 | `too_many_errors` |
| GLM-5.3 | M16 t0 | `too_many_errors` |
| GLM-5.3 | M16 t3 | `too_many_errors` |
| MiMo-V2.6-Pro | E14 t1 | `infrastructure_error` |

Task M16 alone absorbs all four `too_many_errors` terminations (GLM-5.3 t0/t3, Qwen3.8-Flash t2/t3 — tool-error limit reached); MiMo E14 t1 is an `infrastructure_error` (upstream outage, retries exhausted, reward_info=None scored 0; engineering noise). The dual caliber for E14 t1 is documented in the leaderboard page's "two scoring calibers" section.

**M16 root cause (trial level).** All four `too_many_errors` runs trace to a single mechanism: the harness counts every tool-execution error cumulatively and ends the run when the count reaches the limit recorded in the run configuration (`max_errors=10`). On M16 the catalog number the agent needs (`AW-008`, temporary hardship grant) is held by the simulated user by design — the user-side script records it as a fact from the counselor's flowchart — while the agent has no read surface for it: the policy text contains no `AW-` identifiers, no read tool lists the award catalog, and this student has no prior application whose record would reveal the format, so the intended path is to ask the user. The four failing runs instead kept issuing guessed write calls to `submit_scholarship_app` until the budget ran out — error mixes of 10 catalog-not-found + 1 eligibility refusal (GLM-5.3 t0), 8 + 2 (GLM-5.3 t3), 2 + 8 (Qwen3.8-Flash t2), and 3 + 7 (Qwen3.8-Flash t3; that run had already submitted `AW-008` successfully before spending its last two error slots on nonexistent `AW-009`/`AW-010`). Of the 20 M16 runs across the five rows, 16 ended normally: 9 obtained the number verbatim from the user and 7 found it through a short identifier sweep — the busiest sweeps stopped at exactly 9 errors, one short of the limit. Assessment: the dominant cause is model behavior — repeated guessed writes against a state-changing tool instead of asking the dialogue partner who holds the value — not a defect in the task or scoring contract (reference actions, assertion strings, and DB checks are untouched, and the same task completed normally in the other runs). Two friction points are recorded here for a future iteration rather than changed now: the not-found error does not enumerate valid catalog numbers, and the upstream error budget charges policy-refusal errors — whose messages carry corrective guidance — at the same weight as invalid identifiers; exposing the catalog through a read-only tool would remove the guesswork but would change the tool contract, so it is left unchanged for now.

**Tier breakdown.**
| Row | easy | medium | hard | Note |
|---|---|---|---|---|
| DeepSeek V4.1-Flash | 1.0000 | 0.9625 | 0.9667 | hard > medium（非单调 / non-monotonic） |
| Qwen3.8-Flash | 0.9667 | 0.8000 | 0.8333 | hard > medium（非单调 / non-monotonic） |
| GLM-5.3 | 0.8333 | 0.8250 | 0.7500 |  |
| GLM-5.3-Flash | 0.7833 | 0.8125 | 0.7667 |  |
| MiMo-V2.6-Pro | 0.8833 | 0.8750 | 0.8000 |  |

Tiers are static id-prefix labels, not empirically calibrated; the anchor and Qwen rows are non-monotonic (hard > medium), so the tiers carry no statistical discriminability within this sample.

**Run-code provenance.**
 The anchor row and the anchor's second-round batch record `info.git_commit` = `d9960762`; the other four rows record `7d6cae5e`; the release tag `campus-v2.1.0` = `72f430a` postdates all runs (as does the current release tag). Changes between the run code and the release tag touched only task-statement text, guard tests and file encodings — scoring semantics are unchanged, evidenced by the reference-action replay passing 50/50 with per-task dual hashes identical to the pre-cleanup baseline.

**Cost caliber note.**
 GOAT credits **include prompt-cache effects**; credits-per-million-tokens density varies with upstream cache hit rates and is **not directly comparable across rows**: GLM-5.3 bills at fresh rates (**1.63 cr/M**) while MiMo shows a measured **98.9%** cache hit rate (**0.122 cr/M**) — the 13× density gap is itself evidence of differing cache behavior on equivalent workload. The DeepSeek side (user+judge) uses converted prices; the anchor row's 8.71 is a 200-sim subset reference value. Any cross-row cost comparison must quote this caliber note alongside the numbers.

## 4. Protocol & grading declarations

1. Tasks with multi-chain procedures open with the user stating their student ID (hard beat; omitting it provokes ID hallucination).
2. Free-text tickets (M20) are graded on exact DB hashes — rewording loses points; single-trial variance is disclosed as-is.
3. Anchor row and user/NL-justification backend are the same DeepSeek-V4.1-Flash (legacy alias `deepseek-chat` vs canonical `deepseek-flash`, same `system_fingerprint`).
4. Agents occasionally answer in English and "translate away" Chinese policy strings → COMMUNICATE misses; assertion strings are tool-guaranteed entities, but the mechanism is disclosed.
5. Non-cooperative personas are fixed across all tested models (per-task scripted in `tasks.json` + `personas`).
6. seed=20261004, temperature=0 (agent/user/judge), max_steps=60 for every run.

Zero-signal disclosure: 24 of the 50 tasks carry neither gold actions nor `env_assertions` — this is by design under the v1.0.1 contract (the DB component punishes spurious writes; process quality is not scored).

Scoring-contract note: all 50 tasks ship `reward_basis=[DB, COMMUNICATE]` (the upstream default); the 54 `env_assertions` across 25 tasks are diagnostic outputs (reported in `RewardInfo.env_assertions`) and do not gate the reward, per the official v1.0.1 contract (docs/evaluation.md).

**Grading (judge) statement**: COMMUNICATE items use **deterministic substring matching** against the agent's full reply text (no whitespace/full-width normalization); DB items use terminal-state hash comparison; both must pass. The `deepseek-flash` model only writes justification text — the met/not-met decision is fully reproducible by rule (530/530 in calibration). Paraphrases may therefore score as misses; ~60% of misses in the audited population are such string-level engineering noise rather than capability failures. The two headline validation figures must not be conflated: **530/530 = rule-reproducibility** (the deterministic matcher reproduces every calibration decision), while **78.0% = human agreement on a constructively sampled calibration set** — two different things.

As-run policy disclosure: the policy text embedded in each run (per-sim) is the pre-v1.4 original; it has wording-level deviations from the implementation in force at the time, and grading semantics follow the implementation — which is the released semantics. Policy v1.4 was the text-alignment round that brought the text in line with the implementation. The run-level policy field has been normalized to the released version.

**Three policy-text eras.**
 Three policy texts coexist in this release: the per-simulation embedded as-run text (5,663 chars each, 13 internal pitfall markers per copy, 200 sims per file — the text the leaderboard numbers were actually scored against), the run-level policy field (5,515 chars = published v1.4.1, normalized across all five rows), and the currently published policy.md (5,541 chars, v1.4.2 = v1.4.1 + version footer). Two **semantic** differences are called out explicitly: clause 2 — the as-run text says working days are "Monday–Friday (**excluding statutory holidays**)" while the published text says "(**this document applies no statutory-holiday adjustment** to deadline computation)"; the two are semantically opposed, and the implementation follows the published wording (scoring semantics = published semantics). Clause 13 — the as-run text required students to "confirm the make-up exam schedule **in the mini-program**", a requirement deleted from the published text (no system support). The as-run text is retained verbatim as evidence and never rewritten.

**Limitations (complete list).**
1. **Anchor dual role**: the anchor-row model also serves as the user simulator and the NL judge backend (a cost-driven design trade-off); sensitivity analysis is future work.
2. **Single seed**: one 4-trial run per model with a fixed seed; the binomial standard error of pass^1≈0.8 is about **2.8 percentage points** (excluding within-task clustering, so the true uncertainty is larger); read row gaps of 0.01–0.03 at that scale.
3. **Noise-level tiebreak**: ranks 3 and 4 share pass^4 = 0.680 and differ by **0.015** in pass^1 (3 trials out of 200) — inside sampling noise.
4. **No cross-domain calibration**: absolute scores here are not comparable with the official leaderboard's absolute scores.
5. **Zero-signal tasks 24/50**: 24 tasks have neither reference actions (actions=[]) nor env_assertions — process is unscored (v1.0.1 contract design); list: E01, E04, E05, E07, E08, E09, E10, E12, E13, E14, M01, M03, M06, M07, M08, M09, M15, M17, M18, M19, H05, H11, H13, H14.
6. **Weak assertion discriminability**: 98 COMMUNICATE assertions across 50 tasks, **55 of them ≤3 characters** (56.1%), 9 ISO dates, 2 record ids (DF-007 / EN-0383).
7. **Two rates, two meanings**: **530/530 = rule reproducibility** (a deterministic substring matcher reproduced every calibration verdict); **78.0% = human agreement on a constructively sampled calibration set (132 items)** — different quantities; about 60% of population-level MISSes are literal-format engineering noise.
8. **Agent-side authorization asymmetry**: agent tools accept any student_id (same shape as the upstream retail domain); only the student side is bound — impersonation-style tasks test policy compliance, not tool enforcement.

## 5. Reproduce

> Pin a revision: the default branch (`tau2-zh`) is a showcase snapshot, not the benchmark code.

```sh
git clone --branch campus-v2.3.0 https://github.com/Zitrack/tau2-bench
# latest pinned release: https://github.com/Zitrack/tau2-bench/releases/latest
cd tau2-bench

# install (Python >=3.12,<3.14)
uv sync

# domain tests (187 cases)
uv run pytest tests/test_domains/test_campus

# validate data
uv run tau2 check-data

# run the domain with any OpenAI-compatible agent
uv run tau2 run --domain campus --agent-llm <model> --user-llm deepseek/deepseek-flash \
  --num-trials 4 --task-split-name base

# re-score an existing trajectory dump against current tasks
uv run tau2 evaluate-trajs <results.json> --fresh-tasks
```

Leaderboard rows were produced with a dual-channel harness (agent on an OpenAI-compatible endpoint, user simulator + justification model pinned to `deepseek-flash`) writing `results.json` + `meta.json` per model; full protocol notes, cost disclosures, and the row-addition guide are published with the [campus-v2.3.0 release assets](https://github.com/Zitrack/tau2-bench/releases/tag/campus-v2.3.0) (`leaderboard-page.md`).

## 6. Citation

```bibtex
@misc{tau2-zh-campus,
  title        = {Tau2-ZH: A Native Chinese Campus Domain for $\tau^2$-bench},
  author       = {{Tau2-ZH Project (Zitrack)}},
  year         = {2026},
  howpublished = {\url{https://github.com/Zitrack/tau2-bench} (dev/campus branch)},
  note         = {50 tasks; scoring pinned to tau2-bench v1.0.1}
}
```

Please also cite the benchmark itself: τ²-bench — Si et al., arXiv:2506.07982.

Questions, bug reports, and contributions: **via GitHub Issues** on this fork ([Zitrack/tau2-bench Issues](https://github.com/Zitrack/tau2-bench/issues)).

## 7. Known limitations (modeling boundaries)

- **Subject authorization is a policy-compliance test surface, same as upstream**: agent-side tools accept an arbitrary `student_id` (the upstream retail domain behaves identically); campus goes one step further on the student side with `bind_student` initialization binding and upload ownership checks ("records not belonging to you", policy art. 4). **Attribution caveat**: because the agent side is enforced by policy rather than by tooling, tasks built on cross-student access (e.g. H05) measure policy compliance — a pass shows the agent declined a lookup it *could* have performed, a fail reflects policy reasoning or instruction-following rather than a security control's strength, and such scores must not be read as authorization-security results. An audit-log / authorization-violation dimension is listed as future work.
- **Not modeled**: cross-college approval routing, status propagation after a leave of absence, the major-change workflow, scholarship ranking tables (rank-based eligibility is not tool-verifiable), and statutory public holidays (aligned in policy v1.4: no holiday adjustments in deadline calculations).
- **Certificate / medical field notes**: the certificate progress query (`get_service_requests`) returns `deadline_at` and a detail string but not `ready_at` or proxy-authorization remaining validity (the apply response does carry `ready_at`); `hospital_level` accepts only the canonical values "top-tier" (三甲) and "campus-hospital designated clinic" (校医院指定门诊) — variant spellings are not normalized.
- **Dependence on upstream environment internals** (listed here so an upstream change surfaces in this document): the `set_state` override (replay of `initialization_actions` + one final sync), `sync_tools` timing (`make_tool_call` does not sync — campus settles on the tool execution path), the double-DB pointer re-share after `set_state`, and the `initialization_data.user_data` non-DB key (`student_id`) convention. All four are pinned by `tests/test_domains/test_campus/test_upstream_contract_campus.py`, so upstream drift fails the suite instead of silently shifting scores.

## 8. Changelog

> This table is the **single source of version narrative** for the project: release notes, the dataset card, the project homepage, and pull-request updates reference it rather than restating version history; where channels disagree, this table governs.

- **v2.3.0 (2026-10-09) — prompt-surface cleanup, disclosure hardening, code-tag binding.** Task statements: **16 evaluation-register terms** across 14 string leaves naturalized to domain wording; model-visible tool descriptions: the six descriptions that still carried authoring labels (internal version tags, escalation codes, pitfall codes, archive pointers, evaluation-design phrasing) were rewritten to self-contained wording with policy-clause and behavioral facts preserved, and code comments swept the same way. Scoring contract fields are byte-identical to v2.2.0 (`communicate_info`, `env_assertions`, `initial_state`, per-task `user_tools`, gold-action fields); per-task double-hash replay unchanged. New guards: model-visible tool-description forbidden-token lint (with counterexamples), task-text forbidden-token lint, semantic snapshot tests pinning inline guidance to tool behavior, and a `code_tag` version twin in `manifest.json` (placeholder text removed; concrete commit sha and dataset revision are recorded in the matching Release asset manifest). Release assets: Wilson 95% confidence intervals for every leaderboard row and difficulty tier, an explicit statistical-indistinguishability note for the tied ranks, in-table as-run policy-text and anchor-row (task-era) qualifiers, a unified cost-caliber table, calibration-item locatability disclosure (43 of 132 items not directly re-verifiable), and a downstream training-corpus caution. Data files updated: `tasks.json` (statement text only). pytest **173→187**. **Leaderboard numbers remain the v2.0.0-era as-run results** (no re-runs).
- **v2.2.0 (2026-10-08) — task-statement self-containment, disclosure set, data hygiene.** Task statements: **68 string leaves** rewritten to self-contained wording — the 15 tasks that cited an unpublished internal tool-specification document now carry inline operational guidance (tool-chain timing, preconditions and parameter semantics preserved), authoring terms were removed from a persona and a purpose, evaluation-jargon vocabulary was naturalized, and one purpose's quoted rejection note was rephrased as indirect speech (it paraphrases the runtime rejection wording rather than quoting it). Scoring contract fields are byte-identical to v2.1.0: `communicate_info`, `env_assertions`, `initial_state`, per-task `user_tools`, and the core fields of every gold action (`action_id`/`requestor`/`name`/`arguments`); per-task double-hash replay unchanged. Disclosure set (§3/§4 above): the three policy-text eras (per-sim as-run 5,663 chars / run-level v1.4.1 5,515 / published v1.4.2 5,541), task-era detail (the anchor row carries first-generation assertion strings), limitations (24/50 zero-signal tasks, assertion discriminability, rule-reproducibility vs human-agreement distinction, agent-side authorization asymmetry), trial-level terminations with the M16 root-cause note (two friction points recorded for a future iteration; the tool contract is unchanged), the difficulty-tier table with its non-monotonic rows, run-code provenance, and cost caliber. Data hygiene: student profile fields rectified (gender aligned to name semantics; 40 unique masked phone/ID values with disjoint tails — display fields only, no grading surface reads them); `manifest.json` binds code to data (per-file sha256 + version twins). Evidence & governance: asset manifest with platform upload record and a freeze policy (old-tag assets immutable from this release), harness endpoint/UA metadata scrubbed from the run files (model reasoning traces retained with a redistribution note), the second-round calibration batch published, a task-statement cleanup record (marker-occurrence accounting), and a public data validator (`validate-published-tasks.py`, PASS 2063). pytest **167→173**. **Leaderboard numbers remain the v2.0.0-era as-run results** (no re-runs); the per-task statement deltas are itemized in the release cleanup record.

- **v2.1.0 (2026-10-07) — task-statement cleanup, guard expansion (policy v1.4.2).** Every string leaf of the published task set was swept for authoring markers: **101 sites** cleaned — leading test-point fragments (50), internal decision/version tags (`D-S…`/`v1.x`) across purposes, relevant-policy notes, and gold-action info strings (ticket-spec references genericized), plus two author-version annotations inside user-scenario scripts (cleaned under an explicit two-site exception to the "instructions byte-identical" rule). M06's task statement was rewritten to match its actual zero-write refusal semantics and E12's stale expectation dropped. Scoring contract fields are byte-identical to v2.0.0: `communicate_info`, `env_assertions`, `initial_state`, per-task `user_tools`, and the core fields of every gold action (`action_id`/`requestor`/`name`/`arguments`). Four gold-action **info** text sites were cleaned (M05 ×1, H01 ×2, H15 ×1) — free-text annotations only, never read by the evaluator. Per-task double-hash replay is identical to v2.0.0. Policy **v1.4.2** adds a single version footer line (article text unchanged). New guards: full-leaf marker scan, task self-consistency lint (tool-surface reachability, refusal semantics for zero-action tasks, user-side gold actions), three upstream-contract pins; `solo_mode` now raises per upstream convention; `io_utils` gains UTF-8 on write paths and trajectory readers. pytest **157→167**. **Leaderboard numbers remain the v2.0.0-era as-run results** (no re-runs); the release notes disclose the task-statement deltas.
- **v2.0.0 (2026-10-06) — data-contract release.** Internal QA metadata fully removed from the public dataset: `description.notes` dropped across all 50 tasks, `issues` keys removed, and P0x hint fragments scrubbed from the agent-visible `purpose` / `relevant_policies` fields — embedded task copies inside the published raw runs normalized to the same data-contract (QA metadata stripped: notes/issues/P0x; assertion strings and trajectory bodies preserved). `policy.md` meta-annotations (13 grading-marker tags, internal version labels, header changelog) removed with article text unchanged → policy **v1.4.1**; the as-run policy text inside historical trajectories is preserved verbatim. Calibers redefined & recomputed: **54** env_assertions (25 tasks), **29 zero-write tasks (20 with explicit refusal semantics)**, pytest **147→157** (81 test functions, incl. the 50-task gold-replay parametrization). Release engineering: fork-side CI, UTF-8 fix in `io_utils` (Windows GBK), root-README banner, runtime benchmark metadata, raw runs & judge calibration published as release assets. Gold-replay CI 50/50 maintained at every step.
- **v1.1.5 (2026-10-06) — release discoverability & runtime version metadata.** The root `README.md` gains a fork banner (latest tag / domain README / dataset / protocol / upstream-PR links) so release tags are self-explanatory when cloned; `tau2.domains.campus` now exposes runtime-readable `__version__` (benchmark axis), `POLICY_VERSION`, and `SCORING_PROTOCOL` — the engine package version (`tau2 == 1.0.1`) stays pinned in `pyproject.toml` for leaderboard comparability. Code-only; dataset files unchanged. Tag `campus-v1.1.5`. 147 tests green.
- **v1.1.4 (2026-10-06) — art. 33 closure details (proxy-pickup).** Validity of the proxy-pickup authorization code now anchors to the **signing time** (`auth_sig.acted_at + 30d`), not the later document-upload time, when the "sign first, upload later" path completes the loop; `get_service_requests` distinguishes the two causes of pending-signature — unsigned → 缺签署（代领授权书）, signed-but-missing-ID → 缺受托人证件影像（第 33 条）— so agents are no longer steered to re-sign. Zero gold impact (no gold task contains the affected strings or the sign-first path); gold-replay CI 50/50, 146 tests green. Fix commit lands after tag `campus-v1.1.3`; use tag `campus-v1.1.4` for the corrected pin.
- **v1.1.3 (2026-10-06) — proxy-pickup closure & ticket field slimming (art. 33 / art. 16).** `confirm_action` on a proxy-pickup authorization now requires a bound validly-uploaded proxy ID document before the certificate enters production (art. 33: signature + valid document, in either order — a later valid upload completes the loop once the signature is confirmed); confirming without the document holds the certificate at pending-signature instead of silently entering production. `create_ticket`'s optional `target_grade_id` (art. 16 appeal window) is now window-input only, no longer persisted on the ticket row — agents passing it explicitly get byte-identical terminal states to the default path. Zero impact on the published suite: H07's gold actions already upload the proxy document before confirming, and no gold task passes `target_grade_id`. Gold-replay CI 50/50, 144 tests green.
- **v1.1.2 (2026-10-06) — medical-document handling (art. 12).** A medical document uploaded without declaring `hospital_level` (empty or "none") is now marked **returned-for-supplement** instead of invalid; only a declared other-institution (or other non-canonical value) constitutes invalid material → rejection per art. 12, so an incomplete illness-deferral set settles back to pending-materials instead of being rejected and discarding already-valid documents. Zero impact on the published suite: all medical uploads among the 50 tasks declare compliant levels (M05/H01 = top-tier; M16/H07 are non-medical), gold-replay CI 50/50, 141 tests green. Follow-up: `confirm_action` now decides "enter pending-materials" by usable (non-returned) uploads, so a confirmation receipt no longer misreports submitted-for-review when only returned materials exist — response and settled state agree. Variant-level normalization remains disclosed future work.
- **v1.1.1 (2026-10-06) — hardening.** Pre-settle before every tool call (agent & user sides); user-side deadline guards on signature confirmation and material upload (art. 8/12); special-channel stuck-state terminal transition (confirmed-but-overdue → revert + expire); monotonic waitlist positions (max+1, no position reuse after abandonment). Zero live-impact on the published suite proven by the 50-task gold-replay CI (per-task DB/user-DB hashes byte-identical before/after the change).
- **v1.1 (2026-10-05) — consistency fixes (PR #596 updates, part 1/2).** Full gold-replay CI (exceptions-as-failures + terminal-state invariants; countermeasure aligned with upstream issue #499); optional `target_grade_id` appeal binding (art. 16); deferral↔enrollment binding (art. 12/10); illness deferrals restricted to post-exam filing (art. 12); waitlist ACTIVE-status / suspended-offering guards (art. 10/8); `copy_count ≥ 1` (art. 32); authoritative `enrolled_count` recount in special-channel settlement; illness two-document requirement + H01 gold update (art. 12(2)); special-channel 10-workday submission window + H06 re-anchor (art. 9); policy v1.4 (art. 2/13). Scoring-contract note: `env_assertions` are diagnostic outputs and do not gate the reward. Leaderboard re-anchored via a 40-sim overlay rerun (H01/H06, all 5 rows).
- **v1.0.1 (2026-10-03) — initial release.** Campus domain, 50 tasks (easy 15 / medium 20 / hard 15, incl. 17 refusal tasks), dual-control, policy v1.3, 5-row leaderboard.

## 9. Policy material provenance

The 36-article regulation is fictional in identity and realistic in shape. Sources, stated at **file granularity**:

- **Primary source**: Ministry of Education Order No. 41 (*Regulations on Student Management in Higher Education*, 2017) — the national frame for student status, course registration, deferrals, grade review and appeals, awards and aid, and certification.
- **Secondary sources**: publicly published university academic-affairs rules (student-status management, add/drop and deferral procedures, grade-review and appeal channels, scholarship administration, certificate issuance) as published on Chinese universities' public websites.

Rewriting & de-identification boundary: deadlines, windows, quotas, counts, and amounts are **re-parameterized** across topics; clause numbering and wording are original; the institution ("Qingchuan University") and every person, record, and number are invented. **No clause-by-clause mapping to any real regulation or real university is published — deliberately**: such a mapping would allow reverse inference of real institutions' rules, so the provenance statement stops at file level.

## 10. Language quality (structural self-audit)

The corpus is written natively in Chinese (not translated). A structural self-audit was run over all 350 conversational/task string leaves of `tasks.json` (persona / purpose / relevant_policies / instructions) plus `policy.md`, along the measures validated for Chinese in the upstream multilingual study (naturalness, modal particles, name/address conventions, counting units, speech fidelity, tool use):

- **Naturalness proxies**: 0 English function-word intrusions and 0 English-only prose leaves in the task text; full-width vs half-width punctuation ≈ 45:1 (half-width appears only inside identifiers and numerals).
- **Modal particles**: 10 tokens (吧 / 嘛 / 呀) in student-facing task instructions; **0** in `policy.md` — the written-regulation register correctly carries no conversational particles.
- **Name/address conventions**: roles are addressed as 老师 (×7) and 同学 (×2); dialogue uses the plain spoken 你 (×327); `policy.md` uses no person-to-person address (institutional register).
- **Counting units**: 32 checked measure-word collocations present (门课 ×21, 份材料 ×2, 张单 ×1, …); 8 deliberately impossible pairings (一门材料 / 一份课程 / …) all count 0.
- **Speech fidelity / tool use**: not applicable to static text (written chat corpus; tool use is a runtime measure).

**Honest disclosure**: this is an **author-run self-audit of structural proxies only — no third-party native-speaker review, no external raters, and no human quality rating** were used, unlike the human-validated measures of the upstream multilingual study. The figures above are consistency checks, not quality scores, and must not be read as such.

---

# campus — 高校教务办事域（τ²-bench 首个原生中文域）

> [τ²-bench](https://github.com/sierra-research/tau2-bench)（MIT 许可）的域扩展，计分钉死官方 **v1.0.1**。
> 本域全部机构、人物、记录均**虚构**（学校为"青川大学"，虚构校名）；政策文本由公开规章改写脱敏，包内不含任何真实个人数据。
> **非官方声明**：本域为社区构建的扩展，**与 Sierra Research（τ²-bench 维护方）无隶属/关联关系**（unofficial / not affiliated with Sierra Research），经 pull request 向上游贡献。

## 1. 域概要

`campus` 是为中文高校教务场景**原生设计**（非翻译）的域：选课/补退选、考试缓考、成绩查分与申诉、奖助学金、证明开具、学籍与工单。政策（36 条，含补退选窗口、缓考时限、维护窗口、月末结账、逐级申诉等时限坑）、数据库（`db.json`＝11 业务表＋`env`、`user_db.json`＝3 表，合计 14 表）、50 题任务（easy 15 / medium 20 / hard 15，含 29 道零写任务，其中 20 道具有明确拒绝语义）与 persona 全部中文原生；判分口径与官方同构（`[DB, COMMUNICATE]` + `env_assertions`，不用 ACTION），分数遵循官方 v1.0.1 判分契约；绝对水平不可跨任务集比较。

## 2. 双控设计

遵循 τ²-bench 双控协议，**两侧**都有工具：

| 侧 | 工具 | 职责 |
|---|---|---|
| Agent（客服侧） | **15** 个工具（`tools.py`） | 查询/申请/撤回/建单……必须引导学生走完多步流程 |
| 学生（模拟用户） | **4** 个用户工具（`user_tools.py`） | 上传材料、确认签署、**拒绝**——这些动作 agent 无法代做 |

难题为多链并发：学生须在自己的 App 里完成 3–4 个动作，同时 agent 保持流程一致（如撤回＝agent 发起＋学生签署）。user simulator 由逐题钉死脚本驱动，persona 取自**七种基础型**（配合、较真、改主意、忘事、情绪化、绕弯、深度改主意）**加复合型**（如忘事×改主意），每题 persona 另带显式顺从维度（是否顺从 agent 提议的代办）；配合度由逐题 `task_instructions` 的拒绝约束兜底（38/50 题显式拒绝 agent 提议的代办操作），防止过度合作的模拟用户替 agent 悄悄完成任务。

## 3. 榜单（5 行 / 4 家厂商，按 pass^4 降序）

Pass^1＝通过 trial 占比（共 200）；pass^4＝4/4 全过的题数（共 50）；每题 4 trials、seed=20261004、temperature=0、max_steps=60；user simulator 与 NL 判定模型钉在 DeepSeek-V4.1-Flash。

| # | 模型 | pass^1 | pass^4 | 平均轮次 | 易 / 中 / 难 |
|---|---|---|---|---|---|
| 1 | DeepSeek V4.1-Flash（锚点） | 0.975 | 0.900 | 6.07 | 1.000 / 0.963 / 0.967 |
| 2 | Qwen3.8-Flash | 0.860 | 0.720 | 5.41 | 0.967 / 0.800 / 0.833 |
| 3 | GLM-5.3 | 0.805 | 0.680 | 5.17 | 0.833 / 0.825 / 0.750 |
| 4 | GLM-5.3-Flash | 0.790 | 0.680 | 5.12 | 0.783 / 0.813 / 0.767 |
| 5 | MiMo-V2.6-Pro | 0.855 | 0.660 | 5.19 | 0.883 / 0.875 / 0.800 |

> **v1.1.1 补丁（2026-10-05）**：H01（因病缓考双材料，第 12 条二）与 H06（特别通道十工作日提交窗，2026-03-20 为设计性重锚——其情景在特别通道窗仍开放时开始）为政策↔代码↔金标一致性而更新，并对全部 5 行重跑（40 sims overlay；各行 pre-patch 原值保留于 `leaderboard-final.json`）。pass^4 同为 0.680 的第 3/4 名按 pass^1 排序互换。详见 §8 变更记录。

全部数字均由 `results.json` 程序化重算（绝不手抄）。第 5 行在修复前 as-run 中含 1 个基础设施故障 trial（上游故障）：剔除后修复前为 0.840 / 0.660、现行 v1.1.1 overlay 为 0.860 / 0.680——两种口径均已披露。闭源旗舰模型有意未跑（成本 vs 信息增益）；锚点行承担天花板参照。

难度分层为设计期标签（易 15 / 中 20 / 难 15）；上表给出各层实测通过率，在本样本量下（每层 60–80 trials）层间差异不具统计可分性。

任务时代披露：锚点行采集于断言串第一代（r1），其余四行采集于换串后；已发布跑批中的内嵌任务副本除 v2.0.0 的 QA 元数据清理外自跑批时点逐字保留——保留各行 as-run 断言串，语句层与现行公布 `tasks.json` 不同（任务时代披露），即各行判分口径的原始凭证。

完整披露集（全部数字由已发布跑批资产程序化重算；本 README 中文半区内容相同）：

**任务时代明细。**
已发布跑批内嵌的任务副本（as-run 凭证，冻结）与现行发布任务集属两个时代。语句层 **50/50 题不同**。评分层（与现行发布比对，锚点行）：evaluation_criteria 差异 **12 题**（E12, H01, H05, H08, H13, H14, H15, M05, M06, M08, M12, M19），含 initial_state 共 **14 题**（＋H06, M03）。**五行共有 5 题**（M03/H06＝initial_state 差异；M05/H01/H15＝actions[].info 4 处文本）：

| 题 | 差异面 |
|---|---|
| M03 | initial_state（种子/时间锚差异） |
| H06 | initial_state（种子/时间锚差异） |
| M05 | evaluation_criteria.actions[].info 文本 |
| H01 | evaluation_criteria.actions[].info 文本 |
| H15 | evaluation_criteria.actions[].info 文本 |

**锚点独有 9 题**（其余四行的内嵌评分字段与现行发布一致；锚点持第一代断言串）：

| 题 | 差异面 |
|---|---|
| E12 | communicate_info 断言串 |
| M06 | actions 2→0 |
| M08 | communicate_info 断言串 |
| M12 | communicate_info 断言串 |
| M19 | communicate_info 断言串 |
| H05 | communicate_info 断言串 |
| H08 | communicate_info 断言串 |
| H13 | communicate_info 断言串 |
| H14 | communicate_info 断言串 |

**终止披露。**
全量 1,000 场榜单 sims 中非 `user_stop` 终止恰 5 场：

| 行 | trial | 终止原因 |
|---|---|---|
| Qwen3.8-Flash | M16 t2 | `too_many_errors` |
| Qwen3.8-Flash | M16 t3 | `too_many_errors` |
| GLM-5.3 | M16 t0 | `too_many_errors` |
| GLM-5.3 | M16 t3 | `too_many_errors` |
| MiMo-V2.6-Pro | E14 t1 | `infrastructure_error` |

M16 一题吸收全部 4 次 `too_many_errors`（GLM-5.3 t0/t3、Qwen3.8-Flash t2/t3，工具调用错误数达上限）；MiMo E14 t1 为 `infrastructure_error`（上游瞬断重试耗尽，reward_info=None 计 0，工程噪声）。E14 t1 的双口径见 leaderboard-page「两套计分口径」。

**M16 成因（trial 级）。** 四次 `too_many_errors` 指向同一机制：harness 对工具执行错误**累计计数**，达到运行配置记录的上限（`max_errors=10`）即终止该场 run。M16 所需的目录号（`AW-008`，临时困难补助）按双控设计由模拟用户掌握——用户侧脚本把它记作「辅导员流程图上的事实」——而 agent 侧没有读取面：政策全文无任何 `AW-` 标识符，读取类工具不列出奖项目录，该生也没有可供反推格式的既往申请记录，因此预期路径是**向用户询问**。四个失败 run 反复以猜测值调用写工具 `submit_scholarship_app` 直到预算耗尽——错误构成为：10 次「目录不存在」＋1 次资格拒绝（GLM-5.3 t0）、8＋2（GLM-5.3 t3）、2＋8（Qwen3.8-Flash t2）、3＋7（Qwen3.8-Flash t3；该场曾成功提交过 `AW-008`，随后把最后两个错误名额花在不存在的 `AW-009`/`AW-010` 上）。五行合计 20 场 M16 run 中 16 场正常结束：9 场由用户直接报出目录号，7 场经短程编号试探在预算内命中——试探最多的两场恰好停在 9 个错误，距上限一步之遥。判定：**主因是模型行为**——对着有状态写工具反复猜测，而不向持有该值的对话方询问——而非任务或判分契约缺陷（参考动作、断言串、DB 检查均未受影响，同题在其余 run 正常完成）。两个摩擦点记录备后续迭代、现在不改：其一，「目录不存在」报错不列举有效目录号；其二，上游错误预算把带纠正指引的政策拒绝错误与非法标识符错误同权重计数；为目录增加只读查询工具可以消除猜测，但那会改变工具契约，故暂不改动。

**难度分层。**
| Row | easy | medium | hard | Note |
|---|---|---|---|---|
| DeepSeek V4.1-Flash | 1.0000 | 0.9625 | 0.9667 | hard > medium（非单调 / non-monotonic） |
| Qwen3.8-Flash | 0.9667 | 0.8000 | 0.8333 | hard > medium（非单调 / non-monotonic） |
| GLM-5.3 | 0.8333 | 0.8250 | 0.7500 |  |
| GLM-5.3-Flash | 0.7833 | 0.8125 | 0.7667 |  |
| MiMo-V2.6-Pro | 0.8833 | 0.8750 | 0.8000 |  |

难度标签＝题集 id 前缀静态标注，未经难度实证标定；锚点与 Qwen 两行 hard>medium 非单调，分层在样本内不具统计区分力。

**运行代码版本。**
锚点行与锚点第二轮批次 `info.git_commit` = `d9960762`；其余四行 = `7d6cae5e`；发布 tag `campus-v2.1.0` = `72f430a`（跑批之后；现行发布 tag 同样晚于全部跑批）。run-code 与发布 tag 之间的变更仅涉及任务陈述文本、守卫测试与文件编码，判分语义零变化——以参考动作回放 50/50 逐题双哈希（重放终态哈希与清洗前逐题相等）为证。

**成本口径注。**
GOAT credits **含 prompt 缓存效应**，密度（cr/M tok）随上游缓存命中率波动，**跨行不可直接比较**：GLM-5.3 无缓存收益按 fresh 计费 **1.63 cr/M**，MiMo 实测缓存命中 **98.9%** 密度仅 **0.122 cr/M**——同一量级 workload 下 13× 密度差本身即缓存行为差异的证据；DeepSeek 侧（user+judge）为折算价，锚点行 8.71 为 200-sim 子集参考值。引用于任何对比时须连同本口径注一并引用。

## 4. 协议与判分声明

1. 多链程序任务以用户报学号开场（硬节拍；省略会诱发学号幻觉）。
2. 自由文本工单（M20）按精确 DB 哈希判分——改述即失分；单 trial 方差如实披露。
3. 锚点行与 user/判定后端为同一 DeepSeek-V4.1-Flash（遗留别名 `deepseek-chat` 与 canonical `deepseek-flash` 同一 `system_fingerprint`）。
4. Agent 偶尔以英文作答并把中文政策串"翻译掉"→ COMMUNICATE MISS；断言串均为工具必现实体，该机理已披露。
5. 非合作 persona 在全部被测模型间固定（`tasks.json` + persona 逐题脚本）。
6. 每次运行均为 seed=20261004、temperature=0（agent/user/judge）、max_steps=60。

零信号披露：50 题中 24 题既无金标 actions 也无 env_assertions——系 v1.0.1 契约设计（DB 分量罚乱写，过程不评分）。

判分契约说明：50 题 reward_basis 均为 [DB, COMMUNICATE]（上游默认）；25 题的 54 条 env_assertions 为诊断性输出（见 RewardInfo.env_assertions），不计入 reward 判分，与官方 v1.0.1 文档契约一致。

**判分（judge）声明**：COMMUNICATE 项对 agent 全程回复文本做**确定性子串匹配**（空格/全半角不归一化）；DB 项为终态哈希比对；两项须同时通过。deepseek-flash 只产出判定文本——met/not-met 完全可由规则复现（校准 530/530）。改述因此可能记 MISS；母体约 60% MISS 属字面工程噪声而非能力失败。两个头条数字不可混读：**530/530＝规则可复现性**（确定性匹配器复现了全部校准判定），**78.0%＝构造性抽样校准集上的人工一致率**——两件事。

as-run 政策披露：各次运行内嵌的政策文本（per-sim）为 v1.4 前原文，与当时实现存在措辞级偏差；判分语义以实现为准，即发布版语义。政策 v1.4 即把文本对齐到实现的那一轮。run-level 政策字段已归一为发布版。

**三时代政策文本。**
同一发布物中并存三份政策文本：逐 sim 内嵌的运行时原文（5,663 字符/份，每份含"坑点"标记 13 处，200 sims/文件——榜单数字实际读取的文本）、run-level 政策字段（5,515 字符，= v1.4.1 发布版，五行已归一）、现行发布 policy.md（5,541 字符，v1.4.2＝v1.4.1＋版本脚注）。两处**语义级**差异点名：第 2 条——运行时原文写"工作日指周一至周五（**法定节假日除外**）"，发布版写"（**本规程时限计算不引入法定节假日调整**）"，两者语义相反，实现按发布版（判分语义＝发布版语义）；第 13 条——运行时原文含"补缓考…**学生须在小程序确认补缓考安排**"要求，发布版已删除该要求（系统无承载）。运行时原文作为 as-run 凭证如实保留、不再改写。

**局限披露（完整清单）。**
1. **锚点双角色**：锚点行模型同时是 user simulator 与 NL judge 的后端（成本设计取舍）；敏感性分析列为 future work。
2. **单 seed**：每模型单次 4-trial 采样（seed 固定），pass^1≈0.8 时二项式标准误约 **2.8 个百分点**（未计题内聚类，实际更大）；行间 0.01–0.03 差异按此量级解读。
3. **噪声级决胜**：第 3/4 名 pass^4 同为 0.680、pass^1 相差 **0.015**（200 trial 中 3 个 trial），在抽样噪声量级内。
4. **无跨域标定**：本页绝对分数与官方榜单绝对分数不可直接比较。
5. **零信号题 24/50**：24 题既无参考动作（actions=[]）也无 env_assertions，过程不评分（v1.0.1 契约设计）；题单：E01, E04, E05, E07, E08, E09, E10, E12, E13, E14, M01, M03, M06, M07, M08, M09, M15, M17, M18, M19, H05, H11, H13, H14。
6. **断言判别力弱**：50 题共 98 条 COMMUNICATE 断言，**55 条 ≤3 字**（56.1%）、9 条 ISO 日期、2 条记录号（DF-007 / EN-0383）。
7. **两率区分**：**530/530＝确定性子串匹配器对全部校准判定的规则复现率**；**78.0%＝构造性抽样校准集（132 条）上的人工一致率**——两件事，不可混读；母体层面约 60% 的 MISS 属字面工程噪声。
8. **agent 侧授权不对称**：agent 工具接受任意 student_id（与上游 retail 域同构），仅学生侧经 bind 绑定——代查类题考政策遵从而非工具强制。

## 5. 复现

> 请基于钉定修订跑基准：默认分支（tau2-zh）为展示快照，非基准代码。

```sh
git clone --branch campus-v2.3.0 https://github.com/Zitrack/tau2-bench
# latest pinned release: https://github.com/Zitrack/tau2-bench/releases/latest
cd tau2-bench

# 安装（Python >=3.12,<3.14）
uv sync

# 域测试（187 项）
uv run pytest tests/test_domains/test_campus

# 数据校验
uv run tau2 check-data

# 以任意 OpenAI 兼容 agent 运行本域
uv run tau2 run --domain campus --agent-llm <model> --user-llm deepseek/deepseek-flash \
  --num-trials 4 --task-split-name base

# 对既有轨迹结果按现行任务重新判分
uv run tau2 evaluate-trajs <results.json> --fresh-tasks
```

榜单各行由双通道 harness 产出（agent 走 OpenAI 兼容端点，user simulator＋判定模型钉在 `deepseek-flash`），逐模型落盘 `results.json`＋`meta.json`；完整协议说明、成本披露与加行指南见 [campus-v2.3.0 Release assets](https://github.com/Zitrack/tau2-bench/releases/tag/campus-v2.3.0) 中的 `leaderboard-page.md`。

## 6. 引用格式

```bibtex
@misc{tau2-zh-campus,
  title        = {Tau2-ZH: A Native Chinese Campus Domain for $\tau^2$-bench},
  author       = {{Tau2-ZH Project (Zitrack)}},
  year         = {2026},
  howpublished = {\url{https://github.com/Zitrack/tau2-bench} (dev/campus branch)},
  note         = {50 tasks; scoring pinned to tau2-bench v1.0.1}
}
```

同时请引用基准本体：τ²-bench — Si et al., arXiv:2506.07982。

问题反馈与贡献请经 **GitHub Issues** 提出（[Zitrack/tau2-bench Issues](https://github.com/Zitrack/tau2-bench/issues)）。

## 7. 已知限制（建模边界）

- **主体授权＝政策遵从测试面，与上游同构**：agent 侧工具接受任意 `student_id`（上游 retail 同构行为）；campus 的差异化在学生端——`bind_student` 初始化绑定 + 上传属主校验（"记录不属于本人"，政策第 4 条）。**归因注意**：agent 侧由政策而非工具强制，故跨学生访问类任务（如 H05）考的是政策遵从——通过＝agent 拒绝了它*本可以*执行的查询，失败反映政策理解或指令遵循而非安全控制强度；此类分数不得解读为授权安全性结果。audit-log / 越权维度列为 future work。
- **未建模清单**：跨学院审批流转、休学后的状态传播、转专业工作流、奖学金排名表（排名类资格不经工具校验）。法定节假日已随政策 v1.4 对齐口径（时限计算不引入节假日调整），不再是待建模项。
- **证书 / 医院字段说明**：证书进度查询（`get_service_requests`）返回 `deadline_at` 与明细串，但不返回 `ready_at` 与代领授权余期（申请响应含 `ready_at`）；`hospital_level` 仅收 `三甲` / `校医院指定门诊` canonical 值，变体写法不归一。
- **对上游环境内部契约的依赖**（在此列明，上游一旦改动即在此暴露）：`set_state` 覆写（重放 `initialization_actions`＋末次一次性 sync）、`sync_tools` 时机（`make_tool_call` 不做 sync——campus 在工具执行路径上结算）、`set_state` 后双库指针的重新共享，以及 `initialization_data.user_data` 的非库键（`student_id`）约定。以上四项均由 `tests/test_domains/test_campus/test_upstream_contract_campus.py` 钉住，上游漂移会让测试套件变红，而不是静默改变判分。

## 8. 变更记录

> 本变更记录是项目的**版本叙事唯一来源**：Release notes、数据集卡、项目首页与 PR 更新均引用本表而非各自重述版本史；各渠道表述不一致时以本表为准。

- **v2.3.0（2026-10-09）——提示词面清扫、披露强化与 code_tag 绑定。** 任务陈述：**16 处评测语域词**（14 个字符串叶）自然化为领域用语；模型可见的工具描述：6 个仍带作者标注（内部版本标签、越级码、坑点码、档案指针、评测设计措辞）的描述改写为自足表述（政策条款号与行为事实保持），源码注释同法清扫。判分契约字段与 v2.2.0 逐字节一致（communicate_info、env_assertions、initial_state、逐题 user_tools、金标动作字段）；逐题双哈希重放不变。新增守卫：模型可见工具描述禁词 lint（含反例）、题面文本禁词 lint、内联指引与工具行为的语义快照测试，以及 `manifest.json` 的 `code_tag` 版本孪生（占位文本移除；具体 commit sha 与数据修订见对应 Release 资产清单）。Release 资产新增：逐行与逐难度层的 Wilson 95% 置信区间、并列名次的统计不可分说明、表内 as-run 政策文本与锚点行（任务时代）限定、统一成本口径表、校准条目可定位性披露（132 条中 43 条不可直接复核）、下游训练语料警示。数据文件更新：`tasks.json`（仅陈述文本）。pytest **173→187**。**榜单数字仍为 v2.0.0 时代 as-run 结果**（未重跑）。
- **v2.2.0（2026-10-08）——任务陈述自足化、披露集与数据整饰。** 任务陈述：**68 个字符串叶**改写为自足表述——原引用未公开内部《工具规格》文档的 15 题改为内联操作指引（工具链时序、预检条件、参数语义保持），persona 与 purpose 中的作者用语移除、评测语域词汇自然化，另有一处 purpose 的引号化提示语改为转述（转述运行时拒绝文案而非引用）。判分契约字段与 v2.1.0 逐字节一致：communicate_info、env_assertions、initial_state、逐题 user_tools 与金标动作全部核心字段（action_id / requestor / name / arguments）；逐题双哈希重放不变。披露集（见上文 §3/§4）：三时代政策文本（per-sim as-run 5,663 字符 / run 级 v1.4.1 5,515 / 发布 v1.4.2 5,541）、任务时代明细（锚点行持第一代断言串）、局限（24/50 零信号题、断言判别力、规则可复现率与人工一致率的区分、agent 侧授权不对称）、trial 级终止披露与 M16 成因注记（两处摩擦点留待后续版本，工具契约未动）、难度分层表（含非单调行）、run-code 溯源与成本口径。数据整饰：学生档案字段校正（性别对齐姓名语义；40 组唯一脱敏电话/证件号、尾号互斥——均为展示字段，判分面不读取）；`manifest.json` 绑定代码与数据（逐件 sha256＋版本孪生）。证据与治理：资产清单（含平台上传记录与冻结纪律——自本版起旧 tag 资产不可变）、运行文件中 harness 端点/UA 元数据洗刷（模型思维链保留并附再分发说明）、校准第二轮批次发布、任务陈述清洗记录（标记出现次数口径）、公开数据校验器（`validate-published-tasks.py`，PASS 2063）。pytest **167→173**。**榜单数字仍为 v2.0.0 时代任务的 as-run 结果**（未重跑）；逐题陈述差异在发布清洗记录中逐条列出。

- **v2.1.0（2026-10-07）——任务陈述清洗与守卫扩展（政策 v1.4.2）**：公开任务集全部字符串叶清扫作者标记：**101 处**——前导"考点"残迹（50 处）、purpose/相关条款/金标动作 info 中的内部决策与版本标注（`D-S…`/`v1.x`）删除、工具规格引用改中性通称，以及两处 user-scenario 剧本中的作者版本标注（经明示两站点例外——"instructions 逐字节不变"条款恰对此两点修订）。M06 任务陈述重写为零写拒绝语义、E12 陈旧期望句删除。判分契约字段与 v2.0.0 逐字节一致：communicate_info、env_assertions、initial_state、逐题 user_tools，以及金标动作全部核心字段（action_id / requestor / name / arguments）；金标动作的 info 自由文本清理了 4 处（M05×1、H01×2、H15×1），evaluator 不读该文本。逐题双哈希重放与 v2.0.0 完全一致。政策 **v1.4.2** 增加单行版本脚注（条款正文不变）。新增守卫：全叶标记扫描、任务自洽 lint（工具面可达性、零动作题拒绝语义、user 侧金标动作）、三项上游契约钉；`solo_mode` 按上游规范改为抛错；`io_utils` 写侧与轨迹读取补 UTF-8。pytest **157→167**。**榜单数字仍为 v2.0.0 时代任务的 as-run 结果**（未重跑）；任务陈述差异在 Release notes 中披露。
- **v2.0.0（2026-10-06）——数据契约版本**：公开数据集内部 QA 元数据全量移除——50 题 `description.notes` 剥离、`issues` 键删除、agent 可见 `purpose`/`relevant_policies` 中 P0x 提示片段清除（已发布原始跑批中的内嵌任务副本归一到同一数据契约——QA 元数据剥离：notes/issues/P0x；断言串与轨迹本体保留）；`policy.md` 元注记清除（13 个判分标记＋内部版本标签＋头部变更日志）——条款正文零触碰，政策版本 **v1.4.1**；历史轨迹中的 as-run 政策文本按原样保留。口径重定义并重算：**54** 条 env_assertions（25 题）、**29 道零写任务（其中 20 道具明确拒绝语义）**、pytest **147→157**（81 个测试函数，含 50 题金标重放参数化）。发布工程：fork-side CI、`io_utils` UTF-8 修复（Windows GBK）、根 README 横幅、运行时基准元数据、原始跑批与判分校准作为 Release assets 公开。全程金标重放 CI 50/50。
- **v1.1.5（2026-10-06）——发布可发现性与运行时版本元数据**：根 `README.md` 顶部增加 fork 横幅（最新 tag／域 README／数据集／协议／上游 PR 链接），clone 发布 tag 后即自解释；`tau2.domains.campus` 暴露运行时可读的 `__version__`（基准轴）、`POLICY_VERSION` 与 `SCORING_PROTOCOL`——引擎包版本（`tau2 == 1.0.1`）仍钉在 `pyproject.toml` 以保证榜单可比性。仅代码变更，数据集文件零变化。tag `campus-v1.1.5`。147 项测试全绿。
- **v1.1.4（2026-10-06）——第 33 条闭环细节（代领授权）**：先签后传路径闭环推进时，代领授权码有效期锚定为**签署时刻**（`auth_sig.acted_at + 30 日`），不再按后置上传时刻起算；`get_service_requests` 对"待签署"按成因分流——未签署 → 缺签署（代领授权书），已签署缺证件 → 缺受托人证件影像（第 33 条）——不再误导 agent 重新签署。对金标零影响（受影响字符串与先签后传路径均不在金标中）；金标重放 CI 50/50，146 项测试全绿。修复提交位于 tag `campus-v1.1.3` 之后，修正版请用 tag `campus-v1.1.4`。
- **v1.1.3（2026-10-06）——代领闭环与工单字段瘦身（第 33 条 / 第 16 条）**：代领授权的 `confirm_action` 现要求已绑定有效受托人证件影像方可让证明进入制作（第 33 条：签署＋有效证件，两序皆达——后传的有效证件在上传侧闭环推进）；无证件的签署确认将证明保持在"待签署"，不再静默进入制作。`create_ticket` 的可选 `target_grade_id`（第 16 条申诉窗）改为仅作判窗输入、不再写入工单行——显式传参的 agent 终态与缺省路径逐字节一致。对现役 50 题零影响（H07 金标本就先传证件后签署；金标零传参）。金标重放 CI 50/50，144 项测试全绿。
- **v1.1.2（2026-10-06）——医疗材料处理（第 12 条）**：医疗类材料未申报医院等级（空串/"无"）改为"已退回补报"而非"无效材料"；仅声明"其他机构"（或其他非 canonical 声明值）才构成无效驳回（第 12 条原文）；`confirm_action` 按可用材料（非退回）判定"进入待材料"，回执不再误报"已提交待审"。对现役 50 题零影响（金标重放 CI 50/50，141 项测试全绿）。变体等级归一化仍为已披露 future work。
- **v1.1.1（2026-10-06）——加固**：所有工具调用前置结算（pre-settle，agent/学生两侧）；用户侧签署确认与材料上传 deadline 守卫（第 8/12 条）；特别通道"已确认但逾期"终态迁移（回退＋过期）；候补位次单调（max+1，放弃后不复用位次）。50 题双库哈希逐字节一致证明零现役影响。
- **v1.1（2026-10-05）——一致性修复（PR #596 Updates part 1/2）**：50 题全量金标重放 CI（异常即失败＋终态不变量，对齐上游 #499）；申诉可选绑定目标成绩（第 16 条）；缓考↔选课绑定（第 12/10 条）；因病限考后补办（第 12 条）；候补在读/停开守卫（第 10/8 条）；开具份数≥1（第 32 条）；特别通道结算计数权威重算；因病双材料＋H01 金标更新（第 12 条二）；特别通道十工作日提交窗＋H06 重锚（第 9 条）；政策 v1.4（第 2/13 条）。判分契约说明：`env_assertions` 为诊断性输出、不计入 reward。榜单经 40 sims overlay 重跑重锚（H01/H06，全部 5 行）。
- **v1.0.1（2026-10-03）——首次发布**：campus 域、50 题（易 15/中 20/难 15，含 17 道拒绝题）、双控、政策 v1.3、5 行榜单。

## 9. 政策素材来源

36 条规程在身份上虚构、在形态上写实。来源只列到**文件粒度**：

- **主要素材**：教育部第 41 号令《普通高等学校学生管理规定》（2017）——学籍、选课、缓考、成绩复查与申诉、奖助与证照的国家级框架。
- **辅助素材**：中国高校网站公开发布的教务制度（学籍管理、补退选与缓考流程、成绩复查与申诉渠道、奖助管理、证明开具）。

改写与脱敏边界：时限、窗口、额度、计数、金额等参数跨主题**重新配比**；条款编号与措辞均为原创；机构（"青川大学"）与全部人物、记录、数字均为虚构。**不发布与任何真实规章或真实高校的逐条映射——这是有意为之**：该映射可反推真实机构的制度，因此来源声明止于文件粒度。

## 10. 语言质量（结构化自审）

语料为中文原生写作（非翻译）。对 `tasks.json` 的全部 350 个对话/任务字符串叶（persona / purpose / relevant_policies / instructions）加 `policy.md` 做了结构化自审，维度对齐上游多语研究为中文验证过的度量（naturalness、modal_particles、称谓、量词、speech fidelity、tool use）：

- **自然度代理指标**：任务文本中英文功能词渗入 0 处、纯英文散文叶 0 个；全角 vs 半角标点 ≈ 45:1（半角仅出现在标识符与数字内）。
- **语气词**：学生侧任务指令中 10 个（吧 / 嘛 / 呀）；`policy.md` 中 **0** 个——成文规程语域正确地不带对话语气词。
- **称谓**：角色称谓为 老师（×7）、同学（×2）；对白使用口语 你（×327）；`policy.md` 无人对人称谓（机构语域）。
- **量词**：抽查 32 处量词搭配在位（门课 ×21、份材料 ×2、张单 ×1……）；8 组刻意构造的不可能搭配（一门材料 / 一份课程……）全部为 0。
- **speech fidelity / tool use**：不适用于静态文本（书面聊天语料；tool use 为运行时度量）。

**诚实披露**：这是**作者本人对结构化代理指标的自审——未做任何第三方母语者审校，无外部标注员，无人工质量评分**，不同于上游多语研究的人工验证度量。以上数字是一致性检查，不是质量分数，不得作如是解读。

