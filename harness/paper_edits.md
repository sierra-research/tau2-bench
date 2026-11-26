# Paper edits after the v2 rerun

Do not change Table 4 or Table 5 until `harness/summarize.py` has produced the new
numbers. The old counts came from a shared Twenty instance and cannot be reused.

## Abstract

- Replace "ten customer lifecycle workflows" with "ten CRM dialogue scenarios
  spanning the customer lifecycle".
- After the reset is verified, keep "deterministic post-interaction database
  assertions" but add: "Each trial started from a clean CRM state; task fixtures
  were seeded through MCP and mirrored into the agent's local CRM view."
- If the reward is ENV-only, keep "no judge model scored the response text." If a
  separate dialogue-quality judge is reported, add: "A separate LLM rubric was used
  only as a diagnostic and did not gate the reward."
- Replace all pass^k values and intervals with the new summary.
- Delete "Both sub-4B models failed the threshold" or replace "the threshold" with
  the actual threshold and its source. If only the two leaders are rerun, remove
  the sub-4B sentences from the abstract.

## Section 1.4 (Method)

Replace the sentence that describes the benchmark with:

> Each of the ten CRM dialogue scenarios was executed across eight repeated
> stochastic trials per model. Every trial began from a clean CRM state: Twenty
> CRM, PostgreSQL and the task fixtures were reset through the MCP layer, and the
> agent's local CRM view was seeded with the same record identifiers. The reward
> was computed only from deterministic post-interaction database assertions.
> Reliability is reported as pass^1 and pass^4 with 95 percent task-cluster
> bootstrap intervals. The two leading models are compared task by task with a
> paired sign-flip test and a task-cluster bootstrap interval for the difference.

Remove any claim that trust or commitment levels are persisted CRM fields unless
that is implemented and asserted.

## Section 3 (Design)

- The physics-cognition boundary applies to people, companies, opportunities and
  tasks. Trust and commitment are local annotations in the current toolkit, not
  canonical Twenty fields. Either say this explicitly or remove the trust tool from
  the evaluated workflow.
- Replace "the study did not test it against an active adversary" only if the audit
  log was active in the rerun. If not, remove the audit-log claim.

## Section 4 (Protocol)

### Trial reset

Replace the current paragraph with:

> Trial reset. Each trial began from a known state. The harness deleted all people,
> companies, opportunities and tasks from Twenty, then verified that the four
> `find_many_*` calls returned zero records. It seeded the task's company and, for
> existing-customer scenarios, the person and opportunity fixtures through MCP. The
> returned Twenty UUIDs were written into the agent's local CRM view, so the read
> tools and the live CRM referred to the same records. A trial was aborted if the
> clean-state check failed.

### Success rule

Replace the current paragraph with:

> Success rule. A trial passed only when every deterministic environment assertion
> passed against the live Twenty CRM. Assertions checked the person's name and
> company link, the opportunity name and stage, the task title, and the total
> record counts. A missing record, a wrong company link, a duplicate record or an
> extra record counted as failure. No LLM judge gated the reward.

### Task set

Add:

> The ten scenarios are: new lead (Sarah Chen), trust-building follow-up (Mark
> Reynolds), pilot (Priya Sharma), trial (David Kim), small pilot with follow-up
> (Emma Rodriguez), strategic partnership (James Wilson), complaint and stage
> change (Maria Garcia), referral (Alex Thompson), renewal (Rachel Brown), and
> recovery (Chris Martinez). The earlier labels "customer information retrieval"
> and "CRM data maintenance" are removed because the domain does not contain those
> tasks.

### Simulator

Add:

> The customer was played by `<model>`, which was not one of the evaluated agents.
> The simulator was pinned to `<version>`, with a fixed persona prompt per task and
> seed `<seed>`. If a second simulator was run, report both.

If the simulator was Qwen 3.5 9B, add a limitation: the simulator and the best agent
shared a model family, so the comparison may be biased in Qwen's favour.

## Table 5

Rename the rows to the v2 tasks in `harness/task_map.md`. Replace the old
"Customer information retrieval" and "CRM data maintenance" rows with the actual
scenarios. Do not keep the old row labels if the artifact does not contain those
tasks.

## Section 5.2 (Limitations)

Add these limitations:

- Only ten dialogue scenarios and eight trials per task; the task-cluster intervals
  are wide and the paired test has low power.
- The scenarios are synthetic and were not validated by external CRM experts.
- One CRM schema (Twenty v2.31) and one hardware profile.
- The user simulator is an LLM; its behaviour is part of the measurement. If it
  shares a model family with an evaluated agent, the comparison is confounded.
- The reward checks final database state only. It does not check intermediate
  policy adherence, audit-log quality or the efficiency of the tool trajectory.
- The fixtures are task-specific; results may not transfer to other schemas or
  industry processes.

## Section 6 (Conclusions)

- Replace "all canonical mutations" with the stores that are actually canonical in
  the rerun. If trust/commitment remain local, say so.
- Replace strong ranking claims with the new intervals and the paired result.
- If the sub-4B models were not rerun with the fixed harness, do not present their
  old counts as comparable. Remove them or label them as a pilot.
- Fix the pass^k definition: "pass^k is the probability that all k of k sampled
  runs of a task succeed", not "four consecutive successes".

## Artifact availability

Add a short statement with:

- the tau2-bench-copy repository and commit hash;
- `tasks_v2.json`, `fixtures_v2.json` and `split_tasks.json`;
- the per-trial logs, including seeds, assertions and transcripts;
- the summarizer version and bootstrap seed.

## What not to claim after the rerun

- Do not claim that the old 320-trial table is the result of the fixed harness.
- Do not claim that the model ranking is conclusive.
- Do not claim that the failure taxonomy is measured unless the transcripts were
  coded with a published codebook and inter-rater agreement.
- Do not claim that trust was measured through the CRM database.
