# Frozen prompts and prompt inputs

This directory is the authoritative prompt archive for the reported
τ-Multilingual runs. It contains the exact rendered agent and user system
prompts for all 10,140 scored simulations. The cohort spans multiple repository
revisions, so the current language packs are not used as substitutes for
historical runs.

`manifest.json` maps every reported result file to:

- the exact agent and user system-prompt hashes for every simulation;
- the resolved persona, task, and trial for every simulation;
- the revision recorded by that run;
- the user-guideline snapshot recorded in `results.json`;
- the agent policy and task payload recorded in `results.json`;
- the prompt-affecting resolved run configuration; and
- the runtime language-pack values extracted from that recorded revision.

Retained request logs are the primary source when present: 9,895 user prompts
and 2,700 agent prompts are checked directly. For simulations without a
retained prompt request, the exporter executes the exact recorded source
revision with the recorded run inputs. This also handles early result schemas
that did not serialize the language override even though the task set and run
logs record it.

The archive covers 90 main voice cells, 36 text cells, and eight retail
localization-ablation cells. Objects are deduplicated by SHA-256 under
`objects/`; `manifest.json` records both every object hash and a combined prompt
profile hash for each result cell.

Verify the checked-in archive itself:

```bash
tau2 paper multilingual-verify-prompts \
  --root papers/tau-multilingual/reproduction/prompts
```

With the frozen result bundle and original repository objects available, also
prove every archived input against its source:

```bash
tau2 paper multilingual-verify-prompts \
  --root papers/tau-multilingual/reproduction/prompts \
  --evidence-root /path/to/tau-multi \
  --repo-root /path/to/tau2-bench
```

The archive covers prompts and prompt-affecting local configuration supplied by
this codebase. It cannot capture provider-side instructions that were not
returned to the client or later changes to hosted model implementations.
