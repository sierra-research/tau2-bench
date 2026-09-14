# τ-Banking agent harness: technical handoff pack

**Goal:** Find the knowledge representations, retrieval strategies, harness maintenance approaches, and balance between exploration and exploitation that best enable agents to work effectively. Exploration means seeking new information, resolving uncertainty, and investigating alternatives; exploitation means using established knowledge, procedures, and lessons from previous executions. The harness should help the model decide when to rely on what it knows and when to investigate further, while keeping its knowledge and memory current. The goal is not to beat the benchmark. Use τ-Banking as a limited test environment to develop and assess these approaches.

**Assignment:** Develop a harness that helps the agent use information effectively, exploring knowledge representation, retrieval, maintenance, and the balance between exploration and exploitation.

**Three-hour exercise:** The main goal is to see how you iterate on DEV: inspect failures, choose an improvement, try it, and use the results to guide the next iteration. The success criteria below are pointers, not a checklist; you are not expected to cover them all. Focus on the areas that seem most useful based on what you observe. Test runs can continue beyond the three-hour window.

API credentials will be shared separately through 1Password.

## 1. What is τ-bench?

τ-bench evaluates an agent interacting with a simulated user and domain-specific tools under policy instructions. The agent must gather information, converse, and act to complete a customer request. The original release covered retail and airline tasks; later releases expanded the benchmark family. [1, 2]

The relevant target here is **τ-Banking, the banking domain introduced by τ-Knowledge**, available as `banking_knowledge`. It combines a banking knowledge base with tools that change account-related state. It is distinct from the original τ-bench release. [2, 3]

## 2. What does it measure?

The original benchmark evaluates outcomes by comparing the resulting database state against the task's expected outcome. Its reliability metric, **pass^k**, measures success across all k repeated attempts, rather than whether any one attempt succeeds. Pass^1 is single-attempt success; pass^k is different from best-of-k/pass@k. [1]

τ-Banking tests whether an agent can retrieve and apply information while conducting a conversation and producing the required state changes. Its corpus contains 698 documents across 21 product categories, approximately 195,000 tokens. The challenge combines document retrieval, policy reasoning, tool discovery and execution, and dependencies between steps. Some tools must first be discovered through documentation. [3]

A task is successful when the agent's actions result in a final database state that matches the task's expected database state. [5, Section 3]

### Current benchmark harness

The τ-Banking harness runs a conversation between an agent and a simulated customer. The agent receives instructions and tool access, retrieves banking policies and procedures from the knowledge base, and uses tool responses and conversation history to decide what to do next. Some tools must be located in the documentation and unlocked before use. Tool calls can update the simulated banking environment; the evaluator checks the resulting outcome against the task requirements. [1-3]

The framework supports configurable knowledge-access strategies, including embedding-based retrieval, keyword search, long-context processing, and filesystem exploration. The selected retrieval strategy, agent model, prompts, simulator, and execution budgets together define the baseline configuration. [2, 4]

In the **terminal setup**, the knowledge base is available as files, and the agent uses a shell tool with commands such as `grep`, `cat`, and `find` to locate policies and tool documentation. Banking operations still happen through the environment's banking tools; the terminal provides a way to find the information needed to use them. [5, Section 5]

**Providing all tools upfront** would mean putting every banking tool's definition directly in the agent's available tool list. In the paper's setup, some tools are available initially, while others must be discovered through the knowledge base before they can be invoked. The agent therefore has to find both the applicable procedure and the capability needed to carry it out. Terminal versus retrieval search concerns how the agent accesses knowledge; all-tools-upfront versus discoverable tools concerns which capabilities are exposed initially. [5, Section 3]

## 3. What we need you to develop

Build a harness that helps the agent answer: What do I know? What is still unknown? Which rules apply? What can I do next? What will that change? How will I verify the result?

Start with a reproducible baseline and use its failures to decide what to build. The table below offers possible directions and success criteria to guide your iterations. Choose the areas that address the failures you observe; completing every row is not required.

| Possible focus | Success criteria (guidance) |
|---|---|
| Information representation | Representations are inspectable, linked to source material, and can be updated without task-specific patches. |
| Partial observability | The agent verifies decision-critical claims, resolves missing prerequisites before acting, and preserves uncertainty when verification is unavailable. Measure unsupported assumptions and missed verification steps. |
| Consequence prediction | Predictions are recorded before execution and checked against observable outcomes. Measure prediction accuracy and whether dependency-related ordering failures decrease relative to baseline. |
| Workflow and state changes | The agent detects observable prediction mismatches, updates affected facts and prerequisites, and replans accordingly. Measure stale-state errors and successful recovery after unexpected results. |
| Execution traces and audit logs | Audit logs are transformed into reusable representations that capture actions, state changes, and outcomes, helping the agent use information from previous executions. |
| Previous-execution memory | Retrieved lessons have traceable origins and applicability conditions and can be revised or invalidated. Use DEV comparisons to identify repeated failures, benefits, and regressions; freeze memory before milestone TEST runs. |
| Failure analysis | Each failed run receives a supported diagnosis or an explicit unresolved label. Representative cases identify the failed requirement, responsible step, and proposed correction; reruns assess whether the correction addresses that failure. |
| Harness maintenance | Changes to representations, policies, memory, and configuration are tracked. A source or rule change can be traced to affected representations and lessons, updated or invalidated, and checked on the relevant DEV cases without rewriting task-specific logic. |
| Changes to workflow rules over time | The agent uses the updated rules, invalidates conflicting stored lessons, and avoids applying obsolete procedures. Report adaptation performance separately from standard benchmark results. |

An agent trace records what the agent observed and did. A backend audit log may expose additional state changes the agent never saw. Preserve that distinction: evaluator-only database access must not silently become agent input.

For consequence prediction, evaluate only changes that can actually be verified. Mark unobserved effects as unknown. Store concise decision summaries and evidence rather than requiring private chain-of-thought.

### Failure categories

Classify failures as retrieval omission, incorrect policy interpretation, missing prerequisite or verification, tool discovery/calling error, wrong action order, stale or incorrect state, unsupported success claim, failure to recover, or evaluation ambiguity. Allow multiple labels but identify the earliest supported cause. Separate model/harness errors from simulator and infrastructure errors.

## 4. Limited evaluation plan and deliverables

Use the following project splits. SMOKE is a subset of DEV, not a separate held-out set. Iterate on DEV; keep TEST frozen and use it only at milestones. Do not run the full benchmark or launch broad repeated evaluations as part of routine development.

| Split | Size | Purpose and use |
|---|---|---|
| SMOKE | 6 | Pipeline check: confirm setup, retrieval, tool execution, logging, and grading work end to end. |
| DEV | 25 | Failure analysis and unrestricted iteration on representations and the harness. Use targeted reruns for the failure being addressed. |
| TEST | 37 | Frozen evaluation at milestones only. Freeze the harness, representations, and memory before each run. |

**SMOKE (6): pipeline check**

task_014 task_024 task_057 task_072 task_081 task_087

**DEV (25): failure analysis, iterate freely**

task_014 task_015 task_016 task_018 task_024 task_026 task_028 task_048 task_057 task_066 task_070 task_071 task_072 task_074 task_075 task_078 task_081 task_082 task_083 task_084 task_087 task_098 task_099 task_101 task_102

**TEST (37): frozen, milestones only**

task_001 task_002 task_003 task_004 task_007 task_010 task_025 task_027 task_031 task_033 task_034 task_035 task_036 task_038 task_039 task_040 task_043 task_045 task_047 task_051 task_052 task_053 task_054 task_056 task_058 task_059 task_061 task_064 task_067 task_068 task_069 task_088 task_089 task_093 task_095 task_096 task_097

### Evaluation procedure

1. **Record the baseline configuration.** Capture the domain, split IDs, agent and user-simulator models, prompts, retrieval configuration, and budgets. The TEST list contains 37 unique tasks and has no overlap with DEV.

2. **Check the pipeline with SMOKE.** Establish that the pipeline runs and produces usable traces before undertaking DEV analysis.

3. **Iterate on DEV.** Diagnose a concrete failure, change the relevant representation or harness component, and rerun affected cases. Use component comparisons only when they answer a specific design question. Build reusable memory from DEV executions.

4. **Evaluate TEST at milestones only.** Record the milestone and run budget in advance. Freeze the harness and all derived representations and memory before execution. Keep TEST cases out of development memory and task-specific tuning; do not rerun TEST after each change. Keep hidden user instructions, expected actions, and evaluator-only state out of agent context.

5. **Report representation and maintenance quality first.** Assess source grounding, state consistency, dependency handling, uncertainty, update/invalidation behavior, trace completeness, and the ability to diagnose failures. Include task outcomes, costs, and latency as supporting diagnostics. Label results by the actual split; do not present subset results as full-benchmark scores. Repeated-run reliability is optional and limited to a planned milestone budget.

6. **Keep adaptation experiments on DEV.** If changing policies or testing cross-execution learning, specify the changes, task order, reset rules, and available feedback. Freeze the resulting memory for TEST. Independent repeated trials must not learn from one another.

For the three-hour exercise, share the harness changes you made and a brief account of your DEV iterations: what failed, what you changed, what you observed, and what you would try next. Include available results and identify any test runs still in progress. Those runs can finish after the three-hour window.

**Success for this exercise:** show thoughtful iteration across DEV, using observed failures to improve how the harness supports the agent. Explain your choices and what you learned. The broader criteria are pointers for this work, not required deliverables within three hours.

## Sources

1. Original τ-bench paper.
2. Official τ-bench code and documentation.
3. Official τ-Knowledge overview and banking walkthrough.
4. Sierra: τ³-Bench knowledge and voice release.
5. τ-Knowledge paper.
