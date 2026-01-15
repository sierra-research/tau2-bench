# Vacation Rental Domain

The Vacation Rental domain extends tau2-bench evaluation beyond rule application into **preference adaptation**. By introducing host profiles as a decision layer, we test whether agents truly understand and act on human preferences—not just follow policies.

## The Three-Layer Decision Model

This domain introduces a three-layer decision model that goes beyond simple policy lookup:

```mermaid
flowchart TB
    subgraph L1["LAYER 1: DOMAIN POLICY"]
        P1["Platform rules that apply universally"]
        P2["Cancellation windows • Grace periods • Refund rules"]
    end

    subgraph L2["LAYER 2: HOST PROFILE"]
        H1["Individual host philosophy"]
        H2["Primary focus: reviews | revenue | relationships"]
        H3["Risk tolerance • Soft spots • Hard limits"]
    end

    subgraph L3["LAYER 3: GUEST CONTEXT"]
        G1["Situational factors"]
        G2["Repeat guest status • Evidence quality • Circumstances"]
    end

    L1 --> L2 --> L3
```

The Vacation Rental domain uniquely tests **Layer 2**: can agents adapt to different human preferences when given the same situation?

## Host Profile Archetypes

The domain includes three distinct host profiles that represent real-world hosting philosophies:

| Host | Philosophy | Risk Tolerance | Max Goodwill Refund | Key Soft Spots |
|------|------------|----------------|---------------------|----------------|
| **Ibrahim** | Reviews-first | High | 50% | Medical emergencies, repeat guests, honest communication |
| **Alessia** | Revenue-first | Low | 0% | Repeat guests only |
| **Pierre** | Relationships | Medium | 25% | Military service, first responders, honest mistakes |

These profiles create **divergent correct answers** for identical scenarios.

## Grounding Evaluation with Pre-Computed Host Decisions

To ensure deterministic evaluation, host decisions are pre-computed and stored as ground truth:

```mermaid
sequenceDiagram
    participant Agent
    participant Tool as request_host_decision()
    participant DB as Pre-computed Decisions

    Agent->>Tool: host_id, situation_type, guest_context
    Tool->>DB: Lookup matching decision
    DB-->>Tool: decision, approved_amount_pct, reasoning
    Tool-->>Agent: HostDecision object

    Note over DB: Deterministic ground truth<br/>enables reproducible scoring
```

This design ensures:
- **Deterministic scoring:** Expected outcomes are pre-defined and grounded
- **Psychology simulation:** The reasoning field captures different host preferences
- **Testable preference adaptation:** Same input + different host = different correct answer

## Example: Same Request, Different Hosts, Different Outcomes

### Setup: Early Check-In Request

| Variable | Value |
|----------|-------|
| Guest | Aoife Ryan |
| Request | Early check-in (11 AM instead of 3 PM) |
| Reason | "My flight arrives early, I'd rather not wait" |

### Divergent Outcomes

```mermaid
flowchart TB
    Request["Aoife requests early check-in"]

    Request --> Task32
    Request --> Task33

    subgraph Task32["Task 32: Ibrahim's Property"]
        I1["Guest History: 4 stays with Ibrahim"]
        I2["Host: Reviews-focused, high risk tolerance"]
        I3["Soft spot: repeat_guests ✓"]
        I4["Decision: APPROVE"]
    end

    subgraph Task33["Task 33: Alessia's Property"]
        A1["Guest History: 0 stays with Alessia"]
        A2["Host: Revenue-focused, low risk tolerance"]
        A3["Soft spot: repeat_guests ✗"]
        A4["Decision: DENY"]
    end
```

### Expected Agent Tool Chain

**Task 32 (Ibrahim):**
1. `get_reservation_details(RES016)`
2. `get_listing_details(LST002)` → identifies host
3. `get_guest_history(aoife_ryan_3456)` → discovers 4 stays with Ibrahim
4. `get_host_profile(host_ibrahim_...)` → sees soft spot for repeat guests
5. `request_host_decision(..., guest_context="repeat_guest")` → **APPROVE**

**Task 33 (Alessia):**
1. `get_reservation_details(RES005)`
2. `get_listing_details(LST006)` → identifies host
3. `get_guest_history(aoife_ryan_3456)` → finds 0 stays with Alessia
4. `get_host_profile(host_alessia_...)` → sees low flexibility
5. `request_host_decision(..., guest_context=null)` → **DENY**

### Why This Matters for Evaluation

| Failure Mode | Task 32 | Task 33 |
|--------------|---------|---------|
| Agent always approves | Pass | Fail |
| Agent always denies | Fail | Pass |
| Agent skips host profile lookup | Fail | Fail |
| Agent ignores guest history | Fail | Fail |
| **Agent adapts to host preferences** | **Pass** | **Pass** |

## Disputed Evidence Scenario

This scenario shows how host psychology affects ambiguous situations:

```mermaid
flowchart TB
    Issue["Issue: Carpet stain reported<br/>Evidence: INCONCLUSIVE<br/>Guest photos vs Host turnover photos"]

    Issue --> Ibrahim
    Issue --> Alessia

    subgraph Ibrahim["Ibrahim (Reviews-focused)"]
        ID["Decision: APPROVE 25%"]
        IR["'A negative review costs me<br/>far more than a partial refund'"]
        IC["Compensation: $131.25"]
    end

    subgraph Alessia["Alessia (Revenue-focused)"]
        AD["Decision: DENY"]
        AR["'Burden of proof is<br/>on the guest'"]
        AC["Compensation: $0"]
    end
```

Same evidence. Same platform policy. **Different correct answers based on host.**

## Domain Statistics

- **Users:** 8 (5 guests, 3 hosts)
- **Listings:** 8 properties across different cancellation policies
- **Reservations:** 20 bookings
- **Tasks:** 35 evaluation scenarios
- **Host Decisions:** 13 pre-computed decisions
- **Task Splits:** train (21), test (14), eval (16), base (35)

## Tools

The domain provides the following tools:

### Core Tools
- `get_user_details` - Retrieve user profile and reservations
- `get_reservation_details` - Get reservation information
- `get_listing_details` - Get listing and cancellation policy
- `get_current_time` - Get current datetime for policy calculations
- `get_cancellation_policy_rules` - Get refund calculation rules
- `calculate` - Safe arithmetic calculator
- `cancel_reservation` - Cancel with policy-based refund validation

### Host Consideration Tools
- `get_host_profile` - Retrieve host preferences and philosophy
- `get_guest_history` - Check guest's stay history by host
- `request_host_decision` - Get pre-computed host decision for situation

### Issue Handling Tools
- `submit_issue_report` - Create issue record
- `get_issue_details` - Retrieve issue information
- `validate_issue_evidence` - Check evidence validation status
- `process_goodwill_refund` - Issue refund beyond policy (with host limits)
- `apply_service_credit` - Apply credit to guest account
- `add_reservation_note` - Document decisions

### Escalation
- `transfer_to_human_agents` - Escalate to human support
