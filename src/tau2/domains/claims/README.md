# Claims Domain for TauBench

## Overview

This folder contains the **Claims Domain implementation** for **TauBench**, a benchmarking framework for multi-turn reasoning and tool usage in AI agents.  

The domain simulates a realistic **insurance claims workflow**, including:

- Claim reporting, evaluation, and submission
- Document verification and handling
- Claim approval, rejection, and settlement
- Payment processing (total loss, minor liabilities, subrogation)
- Fraud detection and human escalation
- Recovery and subrogation management

The domain is designed to benchmark AI Agents in a controlled, realistic insurance environment.

---

## Features

- **Claims Management Tools**: Fully implemented `ClaimsTools` toolkit with typed READ/WRITE tool actions.
- **Policy and Claim Validation**: Automatic checks for active policies, mandatory documents, and coverage limits.
- **Explicit User Confirmation**: All WRITE actions require explicit user confirmation to mimic safe operational procedures.
- **Fraud & Escalation Handling**: Supports automated fraud detection, human transfers, and recovery actions.
- **Audit Trail**: Maintains a detailed timeline of claim events, status changes, payments, and recoveries.

---

## Domain Objects

### Users 
- Policyholders, beneficiaries, third-party claimants
- Attributes: user ID, full name, email, addresses, date of birth, policy numbers, payment methods

### Insurance Policies
- Attributes: policy ID, type (motor, property, health, life, travel), effective/expiry dates, sum insured, premium, status

### Claims
- Attributes: claim ID, policy ID, claimant info, type of loss, cover type, dates, status, loss and coverage details, payments, documents, timeline, assignments, audit trail, fraud assessment, subrogation/recovery info

---

## Workflow

### Report a Claim
1. Validate that the policy exists and is active
2. Collect mandatory claim details (user ID, policy ID, type, date of loss, supporting documents)
3. Request missing documents via `request_documents` (WRITE tool, with confirmation)
4. Submit claim for review using `submit_claim`

### Evaluate a Claim
- Verify coverage against policy terms
- Check documents via `list_claim_documents` and `verify_documents`
- Confirm claimant identity and third-party authorizations
- Update claim status with `review_claim`

### Approve / Reject a Claim
- Approve if all mandatory documents are verified and claim is within coverage
- Reject if documents are missing, coverage is invalid, or fraud is suspected
- Use appropriate WRITE tools (`approve_claim`, `approve_partial_reimbursement`, `reject_claim`) with explicit confirmation
- Rejections may trigger human review via `transfer_to_human_agents`

### Payments / Settlements
- Process total loss payments (`process_total_loss_payment`) or minor liabilities (`process_liability_minor`)
- Payments follow coverage, deductibles, and policy limits
- Update claim payment history automatically

### Fraud & Recovery
- Escalate suspicious claims using `escalate_claim`
- Identify subrogation opportunities with `identify_subrogation`
- Initiate recoveries via `initiate_recovery` and track with `update_recovery_status`

---

## Tool Usage Policy

- **One tool per response:** Agents may invoke only one tool per interaction
- **Explicit confirmation required:** All WRITE operations must be confirmed by the user
- **Read vs Write:** READ tools may be called without confirmation
- **Human escalation:** Requests outside tool scope must be transferred via `transfer_to_human_agents`

---

## Data Model
The domain structure and database storage follows similar implementation as the rest of the other domains but with complex tasks.

## Pre-requisites
$env:TAU2_STRICT_REPLAY = "1"
$env:OPENAI_API_KEY = "<your_api_key_here>"  # for agents requiring OpenAI access

## Build docker image
docker build -t tau2-claims .


## Run Simulations/Evaluations(Minimal Test run)

## Running locally for specific tasks
tau2 run `
 --domain claims `
 --agent-llm gpt-4.1 `
 --user-llm gpt-4.1 `
 --num-trials 1 `
 --task-ids LIFE-2 LIFE-4 LIFE-9 TR-1 TR-5 TR-7 HOME-0 HOME-6 HOME-9 MT-0 MT-3 MT-7 MED-0 MED-1 MED-3 `
 --save-to [filename] 

## Running locally for all tasks
tau2 run `
 --domain claims `
 --agent-llm gpt-4.1 `
 --user-llm gpt-4.1 `
 --num-trials 1 `
 --save-to [filename]

<!-- ### output file inside container
docker run --rm `
  tau2-claims tau2 run `
  --domain claims `
  --agent-llm gpt-4.1 `
  --user-llm gpt-4.1 `
  --num-trials 2 `
  --task-ids LIFE-9 `
  --max-concurrency 3 `
  --save-to "claims_gpt-4.1_benchmark_LIFE-9_v1.json"

### output persists in the host directory
docker run --rm `
  -v $(pwd):/outputs `
  tau2-claims tau2 run `
  --domain claims `
  --agent-llm gpt-4.1 `
  --user-llm gpt-4.1 `
  --num-trials 2 `
  --task-ids LIFE-9 `
  --max-concurrency 3 `
  --save-to /outputs/claims_gpt-4.1_benchmark_LIFE-9_v1.json


 -->
