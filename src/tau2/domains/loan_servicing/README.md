# Loan servicing domain (`loan_servicing`, `loan_servicing_fr`)

A customer-service domain for consumer loans, in English and Quebec French. The
agent supports borrowers of **Boréal Finance**, a fictional lender in Montréal
serving Quebec and Ontario. It follows a written policy to verify callers, take
payments, quote payoffs, waive fees, enroll hardship plans and more. All names,
records and data are fictional.

The domain is designed for **cross-lingual policy compliance**: every task has
an English and a French variant with identical evaluation criteria, and the
policy exists in both languages. Only the language changes, so the same
tasks can measure how much an agent loses when the customer, or the policy,
is not in English.

## Policy

`data/tau2/domains/loan_servicing/policy.md` (English, about 1,600 words) and
`policy_fr.md` (Quebec French, same rules). The rules cover:

- **Verification:** full name, date of birth and postal code must all match one
  record before any information or action; two failed attempts lead to a
  transfer.
- **Authorization:** borrowers and co-borrowers may act; authorized third
  parties get general information only; nobody else gets anything.
- **Confirmation:** explicit confirmation before every write, re-confirmed if
  the request changes; reference numbers are read out after each action.
- **Payments, payoff quotes, due-date changes, late-fee waivers, hardship
  plans, autopay, documents and contact updates**, each with dated eligibility
  rules (for example a waiver must happen *before* a payment, because payments
  pay open fees first).
- **Transfers:** only for listed reasons, each with a reason code.

The current date is fixed (2026-03-16), so every date rule has a single
correct answer.

## Tools

19 tools (`tools.py`). No tool has optional parameters, and write tools store
only structured values, so the database hash check is exact.

| Type | Tools |
|---|---|
| Read | `find_borrower_by_email`, `find_borrower_by_phone`, `find_borrower_by_name_dob`, `get_borrower_details`, `get_loan_details`, `list_payments`, `calculate_payoff` |
| Generic | `calculate` |
| Write | `make_payment`, `cancel_scheduled_payment`, `enable_autopay`, `disable_autopay`, `change_due_date`, `waive_late_fee`, `enroll_hardship_plan`, `update_email`, `update_phone`, `send_document`, `transfer_to_human_agents` |

`transfer_to_human_agents` is a write tool here: it stores a transfer record
(`{transfer_id, reason}`), so a missed, unneeded or wrongly coded transfer is
caught by the database check. Reference ids (`PM-…`, `DR-…`, `HP-…`, `TR-…`)
are deterministic, so tasks can require the agent to read one out.

The database (`db.json`) has 60 borrowers, 90 loans and 1,886 payments,
generated with a fixed seed, with planted edge cases (past-due loans, recent
due-date changes, prior waivers, co-borrowers, third parties, charged-off
loans).

## Tasks

48 tasks (`tasks.json`, ids `ls_001_en` to `ls_048_en` and `ls_001_fr` to
`ls_048_fr`):

| Category | Tasks | Category | Tasks |
|---|---|---|---|
| Payments | 7 | Payoff | 3 |
| Hardship | 7 | Autopay | 3 |
| Authorization | 7 | Multi-request | 3 |
| Late fee waiver | 5 | Information | 2 |
| Must transfer | 5 | Documents | 2 |
| Due date change | 4 | | |

- 13 tasks expect no database change (refusals), 6 expect a transfer, and the
  rest expect one or more actions. 17 tasks are marked hard: the customer
  pushes back after a correct refusal, or the request combines several rules.
- 27 tasks also require a reference id to be read out (`DB` + `COMMUNICATE`);
  21 are scored on the database only.
- **Language-neutral scoring.** `communicate_info` holds only codes (such as
  `PM-31887`) or whole numbers under 1,000, so the substring check works the
  same in both languages. Each EN/FR pair has byte-identical
  `evaluation_criteria`; only the user scenario is translated.
- **Do-nothing baseline:** an agent that never writes scores up to 27% (13 of
  48), or 23% if it also never reads out an id.

## Splits and conditions

`split_tasks.json` has three splits of 48 tasks each: `base` and `en_user`
(English customers) and `fr_user` (French customers). Two domains share the
same tasks, tools and database and differ only in the policy language:
`loan_servicing` (English policy) and `loan_servicing_fr` (French policy).
This gives three paired conditions:

| Condition | Policy | Customer | Command |
|---|---|---|---|
| C1 | English | English | `--domain loan_servicing --task-split-name en_user` |
| C2 | English | French | `--domain loan_servicing --task-split-name fr_user` |
| C3 | French | French | `--domain loan_servicing_fr --task-split-name fr_user` |

The agent's first message is tau2's fixed English greeting in every
condition.

## Running

```bash
tau2 run --domain loan_servicing --task-split-name fr_user \
  --agent-llm gpt-5.4-mini --user-llm gpt-5.4-mini --num-trials 1
```

Solo mode is not supported. On Windows, set `PYTHONUTF8=1` so French text
can be printed to the console.

## Baseline results

From an independent study of this domain (5 agent models, 3 conditions,
4 trials per task, user simulator `gpt-5.4-mini`, default tau2 settings, runs
of 2026-10-01 to 2026-10-08). pass^1 with 95% paired-bootstrap intervals over
tasks for three of the models:

| Agent | C1 (EN/EN) | C2 (EN policy, FR user) | C3 (FR/FR) |
|---|---|---|---|
| `openrouter/deepseek/deepseek-v4-flash` | 97.9% [94.3, 100.0] | 95.8% [90.1, 100.0] | 96.4% [91.7, 100.0] |
| `vertex_ai/gemini-3.1-flash-lite` | 92.2% [86.5, 96.9] | 90.1% [81.8, 96.9] | 84.4% [74.0, 93.2] |
| `vertex_ai/gemini-3.1-pro-preview` | 92.7% [84.9, 99.0] | 93.8% [85.4, 100.0] | 92.2% [84.9, 97.9] |

These runs used an earlier wording of `ls_041_fr` that made the French
customer raise the fee waiver only after the payment, which caused most of
that task's C2 and C3 failures; the text is now fixed (the customer raises
both requests in the first message, as in English) and has not been re-run. Full results,
trajectories and the analysis:
[tau2-loan-servicing-study](https://github.com/robertsolomon002/tau2-loan-servicing-study).

## Tests

`tests/test_domains/test_loan_servicing/`: database validation (ids, links,
amortization, balances against posted payments, status consistency), every
tool's success and error paths, replay and hash stability, and checks that the
two domains share tools, database, tasks and splits, with the same codes,
numbers and headings in both policies.
