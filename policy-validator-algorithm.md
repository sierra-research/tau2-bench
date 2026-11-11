# Algorithm: τ²-bench Retail Policy → Node/Edge Validator Spec

This is the construction procedure that turns `data/tau2/domains/retail/policy.md`
into a runnable, graph-structured legality checker, plus the checking algorithm
that replays a trajectory against it. The companion file
`retail_policy_validator.py` is the executable output of running this algorithm on
the real retail policy, validated against tau2’s shipped result trajectories.

-----

## 0. Output model — what the spec *is*

The validator is a **guarded labeled transition system** (an automaton augmented
with a key/value store), not a plain graph, because the policy mixes four kinds of
constraint that a flat rule list can’t express together:

```
ValidatorSpec = (Σ, S, Store, Δ)
  Σ      : the action alphabet           = one NODE per agent tool
  S      : session automaton state        = {authenticated?, bound_user, ...}
  Store  : a data store                   = {seen_ids, locked_orders, last_confirm}
  Δ      : the transition relation        = legal EDGE firings
```

Each tool call is a candidate **edge firing**. It is legal iff *every* obligation
attached to that action’s node holds:

|obligation  |meaning                                                           |
|------------|------------------------------------------------------------------|
|`requires`  |automaton precondition (e.g. AUTHENTICATED before the firing)     |
|`cross_user`|the action targets only the session’s bound user                  |
|`provenance`|every ID argument was *seen* earlier (tool result or user message)|
|`confirm`   |a DB-write is immediately preceded by explicit user consent       |
|`guard`     |per-action predicate over args + grounded environment state       |

A successful write also carries **effects** (status transition, one-shot lock).
A trajectory is **policy-legal** iff replaying it produces zero failed obligations.

-----

## 1. The construction algorithm  (policy.md → ValidatorSpec)

The policy is prose, so the algorithm is a **clause-classification + extraction**
pass. For each policy sentence, classify it into one of six clause types and emit
the corresponding spec element. The classification keywords below are what you
grep for; the retail column is the worked result.

### Step 1 — Inventory the action surface  →  Σ (nodes)

Parse the domain’s tool signatures. Partition into **READ** (no DB mutation) and
**WRITE** (mutators). One node per tool.

- *Retail:* 7 WRITE nodes (`cancel_pending_order`, `modify_pending_order_address`,
  `modify_pending_order_payment`, `modify_pending_order_items`,
  `modify_user_address`, `return_delivered_order_items`,
  `exchange_delivered_order_items`); the rest READ. Catalog reads
  (`get_product_details`, `get_item_details`, `list_all_product_types`) and
  `find_user_id_*`, `calculate`, `think` are auth-exempt.

### Step 2 — Precedence language  →  automaton states + `requires`

Scan for ordering words: *“at the beginning”*, *“before”*, *“once … you can”*,
*“after confirmation”*. Each yields a state variable and a gate.

- *“authenticate … before [info/actions], even when the user provides the id”* →
  state `AUTHENTICATED`; set it only when a `find_user_id_*` call **succeeds**
  (its tool result is non-error). Attach `requires=AUTHENTICATED` to every
  non-exempt node. → emits code **action_before_auth**.

### Step 3 — Scope language  →  `cross_user` + Store.bound_user

Scan for *“one user per conversation”*, *“deny … any other user”*.

- Bind `Store.bound_user` to the first successfully resolved user id. Any
  user-scoped or order-scoped action whose target user ≠ bound_user is illegal.
  (For order-scoped tools, resolve the order’s owner from the grounded DB.) →
  emits **cross_user_action**.

### Step 4 — Authorization rules  →  `guard` predicates  (per WRITE node)

For each WRITE node, read the policy section under its matching heading and
translate three sub-patterns:

- **status preconditions** — *“can only be X if status is ‘pending’/‘delivered’,
  check status before acting”* → `guard: env.status(order_id) == <required>`.
  *Retail:* cancel/modify_* ⇒ `pending`; return/exchange ⇒ `delivered`; generic
  rule ⇒ status ∈ {pending, delivered}.
- **enumerated-value constraints** — *“reason must be ‘no longer needed’ or
  ‘ordered by mistake’”*, *“payment source ∈ {gift_card, paypal, credit_card}”*,
  *“single payment method different from the original”* → membership / inequality
  guards.
- **structural item rules** — *“same product, different option, no product-type
  change, must be available”* → for each `(item_id → new_item_id)`:
  `product(old)==product(new) ∧ available(new)`. Ownership: `payment_method_id ∈ user.payment_methods`. → all emit **guard_violation**.

### Step 5 — ID arguments  →  `provenance` (the grounding gate)

Mark every ID-typed argument (`order_id`, `item_ids`, `new_item_ids`,
`payment_method_id`, `user_id`) provenance-required. Maintain `Store.seen_ids`,
populated from **prior tool-result contents *and* prior user messages** (a user
may legitimately type an order id). An ID arg absent from `seen_ids` is fabricated.
→ emits **ungrounded_id**. *(Stricter variant — write ids must come from a tool
call, not user text — is a one-line change; see caveats.)*

### Step 6 — One-shot / irreversibility  →  `effects` + lock guard

Scan for *“can only be called once per order”*, *“will not be able to modify or
cancel anymore”*. The action sets `Store.locked[order_id]=True`; add
`not locked[order_id]` to its guard and to sibling mutators on that order.

- *Retail:* `modify_pending_order_items` (→ status `pending (item modified)`) and
  `exchange_delivered_order_items` lock the order.

### Step 7 — Confirmation language  →  `confirm` obligation

*“list the action details and obtain explicit user confirmation (yes) before any
DB-updating action”* → every WRITE requires that the most recent user message
before the call expresses consent. Consent matcher = affirmatives (*yes, ok,
confirm, go ahead…*) **and** imperative confirmations (*“please cancel”, “cancel
it”, “return them”*). → emits **missing_confirmation** *(heuristic — see caveats)*.

Also emit two **turn-shape invariants** from *“at most one tool call at a time;
never a tool call and a user reply in the same turn”*:
**multiple_tool_calls** (>1 call in one assistant message) and
**msg_and_toolcall_same_turn**.

-----

## 2. Instantiated retail spec (the output of Steps 1–7)

|node (tool)                                                      |requires|confirm|provenance args                                 |guard                                                         |effects                                   |
|-----------------------------------------------------------------|:------:|:-----:|------------------------------------------------|--------------------------------------------------------------|------------------------------------------|
|`cancel_pending_order`                                           |auth    |✔      |order_id                                        |status==pending; reason∈{no longer needed, ordered by mistake}|status→cancelled                          |
|`modify_pending_order_address`                                   |auth    |✔      |order_id                                        |status==pending; ¬locked                                      |status stays pending                      |
|`modify_pending_order_payment`                                   |auth    |✔      |order_id, payment_method_id                     |status==pending; ¬locked; pm owned by user                    |status stays pending                      |
|`modify_pending_order_items`                                     |auth    |✔      |order_id, item_ids, new_item_ids, payment_method|status==pending; ¬locked; same-product/avail; len match       |status→pending(item modified); **lock**   |
|`modify_user_address`                                            |auth    |✔      |user_id                                         |target==bound_user                                            |—                                         |
|`return_delivered_order_items`                                   |auth    |✔      |order_id, item_ids, payment_method_id           |status==delivered; refund target valid                        |status→return requested                   |
|`exchange_delivered_order_items`                                 |auth    |✔      |order_id, item_ids, new_item_ids, payment_method|status==delivered; same-product/avail; len match              |status→exchange requested; **lock**       |
|`get_user_details`                                               |auth    |—      |user_id                                         |(cross_user)                                                  |—                                         |
|`get_order_details`                                              |auth    |—      |order_id                                        |(cross_user)                                                  |—                                         |
|`get_product_details`/`get_item_details`/`list_all_product_types`|—       |—      |(catalog id)                                    |—                                                             |—                                         |
|`find_user_id_by_email`/`find_user_id_by_name_zip`               |—       |—      |—                                               |—                                                             |sets AUTHENTICATED + bound_user on success|

-----

## 3. The checking algorithm (replay)

```
function CHECK(messages, env_from_db):
    index tool-results by call-id  ->  (content, error)        # for effect/auth gating
    state  = {auth:false, bound:null}
    store  = {seen:∅, locked:{}, last_confirm:false}
    violations = []
    for m in messages (in order):
        if m.role == user:
            store.last_confirm = consent_matcher(m.content)     # Step 7
            store.seen ∪= extract_ids(m.content)                # Step 5 (user-grounded)
        elif m.role == tool:
            store.seen ∪= extract_ids(m.content)                # Step 5 (tool-grounded)
        elif m.role == assistant:
            if m.content and m.tool_calls: emit msg_and_toolcall_same_turn   # Step 7
            if len(m.tool_calls) > 1:      emit multiple_tool_calls          # Step 7
            for call in m.tool_calls:
                ok = not result_error(call.id)
                if call is find_user_id_* and ok:
                    state.auth=true; state.bound ?= result_content(call.id)  # Steps 2,3
                if node not auth-exempt and not state.auth: emit action_before_auth
                spec = SPEC[call.name];  if not spec: continue
                for a in spec.provenance:                                     # Step 5
                    for id in ids(call.args[a]):
                        if id ∉ store.seen: emit ungrounded_id
                if state.bound and target_user(call) ≠ state.bound: emit cross_user_action  # Step 3
                if spec.write and not store.last_confirm: emit missing_confirmation         # Step 7
                for e in spec.guard(call.args, env, store): emit guard_violation            # Step 4
                if spec.write and ok:                                                       # Step 6
                    env.apply(call); if spec.locks: store.locked[order_id]=true
                if spec.write: store.last_confirm=false      # consume the confirmation
    return violations            # legal  ⇔  violations == []
```

Two grounding modes for `env`:

- **Grounded (recommended):** initialize order statuses, item→product/availability,
  and user payment methods from `db.json`; update statuses by replaying *successful*
  writes. Gives true status/ownership/product guards.
- **Transcript-only:** skip `db.json`; status guards are inferred from observed
  `get_order_details` results (weaker, but no env needed).

-----

## 4. Running it + validated results

```bash
python retail_policy_validator.py <results.json> --db <retail/db.json> [--strict] [--show K]
```

Run against tau2’s shipped `gpt-4.1 … retail_default … 4trials.json` (456 sims):

```
reward==1 (task pass)      : 338
** CORRUPT SUCCESSES **    : 240   (reward==1 yet policy-violating, --strict)
   corrupt rate among passes: 71.0%
violation histogram (strict):
   488  multiple_tool_calls        (144 batch a WRITE; 344 are read-only)
    85  guard_violation
    18  ungrounded_id
     9  msg_and_toolcall_same_turn
     1  action_before_auth
```

This is the corrupt-success phenomenon directly: ~71% of *reward=1* retail
trajectories made at least one policy-illegal decision the benchmark reward
ignores. (Matches the 27–78% range reported by the procedure-aware-evaluation
literature.)

-----

## 5. Caveats & precision tiers

- **High-precision codes** (`--strict` keeps only these): `multiple_tool_calls`,
  `msg_and_toolcall_same_turn`, `action_before_auth`, `ungrounded_id`,
  `cross_user_action`, `guard_violation`. These are structural or computed against
  the grounded DB — low false-positive rate.
- **`missing_confirmation` is heuristic.** Regex consent detection has false
  positives/negatives (it already mis-passed *“please cancel it because it’s no
  longer needed”* until the matcher was broadened). For production, replace the
  matcher with a small LLM judge over the 2–3 turns preceding each write.
- **Tier `multiple_tool_calls` by severity.** Read-only batches are arguably
  benign; a batch containing a WRITE is serious. Filter accordingly rather than
  treating all 488 equally.
- **Effect timing.** Effects apply only when the matching tool result is non-error,
  which removed spurious downstream guard violations from replayed failed/duplicate
  writes (164→85 here).
- **Stricter provenance.** To enforce the paper’s “Strict Grounding Gate” (write
  ids must originate from a *tool call*, not user text), seed `seen` for write-id
  args from tool results only — a one-line change in the user-message branch.
- **Gift-card balance / cumulative-total guards** are not yet wired (would need
  payment-history replay); add as additional `guard` predicates if needed.

-----

## 6. Using this as an SFT data filter

This validator is the upgrade to the data-quality stage of the fine-tuning
pipeline. Reward-only filtering keeps the ~71% corrupt successes, so an SFT set
built on “reward==1” alone teaches Qwen to reach correct end-states via illegal
paths — the opposite of what you want for pass^k reliability.

Pipeline integration:

1. Generate teacher rollouts (Phase 2 of the SFT plan).
1. Keep a trajectory only if `reward==1` **and** `CHECK()` returns no high-precision
   violation (run in `--strict`, plus an LLM-judged confirmation pass).
1. Bonus: `CHECK()` emits **per-step legality labels**, the exact signal for later
   step-level process supervision or a verifier-shaped RL reward.