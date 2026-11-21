# Food (`food`) Domain

Restaurant ordering and table reservations. One customer contacts one store, either at the drive-thru (pickup now) or by ordering ahead (mobile: pickup or delivery, now or later today), and/or books a table. The agent finds items, applies customizations, discloses requests the menu can't satisfy, checks times and delivery rules, places the order, and books, changes, or cancels reservations after the customer confirms. There are three stores:

| Store | `store_id` | Menu |
|---|---|---|
| Corner Cafe | `cafe` | Espresso drinks, teas, blended drinks, bakery, breakfast, lunch combos |
| City Pizza | `pizza` | Pizzas with portioned toppings and halves, deals, wings, sandwiches, pastas |
| Sunset Burger | `burger` | Burgers, chicken & fish, combos and value boxes, salads, shakes, drinks |

## Data provenance

All stores, menus, prices, seed reservations, and tasks are fictional and were authored for this benchmark. Phone numbers use the reserved `555-01XX` range. The JSON files under `data/tau2/domains/food/` are the source of truth; `tests/test_domains/test_food/` checks their schema, references, and integrity.

## Menu data and lookup tools

`get_environment(task=...)` builds the environment; the agent receives both:
- **Concise `## Menu Data` in the system prompt** (`menu_prompt.py`, ~5k-13k tokens per store by chars/4): lists available item names, categories, base prices, availability schedules, delivery restrictions, and modifier group structure (`### Menu Items`, `### Modifier Groups`, and `### Unavailable Items`), while omitting item descriptions, default choices/amounts (e.g. default toppings, sauces, espresso shots, syrup pumps), and modifier extra prices.
- **18 agent tools**:
  - **13 ordering and reservation tools**: `get_cart`, `add_to_cart`, `update_cart_item`, `remove_from_cart`, `clear_cart`, `set_fulfillment`, `place_order`, `check_table_availability`, `make_reservation`, `find_reservations`, `update_reservation`, `cancel_reservation`, and `transfer_to_human_agents`.
  - **5 read-only menu and store lookup tools**: `get_store_info`, `list_menu_categories`, `search_menu`, `get_item_details`, and `get_modifier_group`.

When `## Menu Data` already provides the item and modifier names needed to fulfill an order, the agent can call `add_to_cart` or `update_cart_item` directly without redundant lookup tool calls. When the customer asks what categories/items exist, what comes in or on an item (`description` or default recipe), default shot/pump counts, or modifier extra prices, the agent must call `get_item_details`, `search_menu`, `list_menu_categories`, or `get_modifier_group`.

The policy is built per task: `tau2 run` passes the task to `get_environment`. Callers that build the environment without a task (the Gymnasium interface, `tau2 domain`) get the policy without the Current Store and Menu Data sections, so the food domain is not supported there yet.

## Order channels

The task's initial cart sets `channel`:
- `drive_thru` (default): pickup now (`scheduled_time` "ASAP"); name and phone are optional.
- `mobile`: ordering ahead. `place_order` requires the full name and a 10-digit phone number (e.g. `415-555-0199`). The order is pickup or delivery, ASAP or "HH:MM" later today, within store hours and at least the store's lead time from now (pickup 20 min, delivery 45 min). Delivery needs an address and the store's minimum subtotal (cafe $12, pizza $15, burger $10).

`set_fulfillment(order_type, scheduled_time)` sets method and time on the cart; `add_to_cart`, lookups, and `place_order` then check item availability (schedules, `allowed_fulfillment_methods`) at that time and for that method. The **Current Store** block shows the channel, weekday and time, store hours, lead times, delivery terms, and whether the store takes reservations.

## Table reservations

Corner Cafe and City Pizza take reservations (`Store.reservation_policy`): parties of 1-8, 30-minute slots from opening + 30 min to closing - 60 min, up to 30 days ahead, a number of tables per slot and seating area (cafe: 4 indoor + 2 outdoor; pizza: 6 indoor), and high chairs per slot (cafe 2, pizza 4). One booking uses one table. Sunset Burger takes none. `FoodDB.reservations` holds seeded bookings (`RES-0001`..`RES-0020`): cafe Friday 2026-06-26 19:00 full, Friday 18:30 outdoor full, Saturday 06-27 12:00 without free high chairs; pizza Saturday 18:00 full and 18:30 with one table left. Tasks can add bookings through `initial_state.initialization_data.agent_data.reservations` (use ids like `RES-T001`). New bookings get `RES-%04d` ids.

## Data (`data/tau2/domains/food/`)

| File | Content |
|---|---|
| `menus/*.json` | Static per-store menu catalogs (`cafe.json`, `pizza.json`, `burger.json`: items, modifier groups, schedules, store rules) |
| `db.json` | Transactional initial state (`cart`, `orders`, and the 20 seed `reservations`); `FoodDB.load()` automatically merges `menus/*.json` |
| `policy.md` | Agent policy rendered with **Current Store** and concise **Menu Data** |
| `tasks.json` | 62 tasks (cafe 20, pizza 20, burger 22) |
| `split_tasks.json` | `base` (62), `cafe` (20), `pizza` (20), `burger` (22), `small` (6) |
| `tasks_voice.json` | Pre-sampled voice personas per task and speech complexity |

The JSON files are the source of truth; no generator script is part of the repo. `tau2 check-data` does not inspect them; `tests/test_domains/test_food/` checks id formats, references, reachability, that every item can be added at every size, fictional phone numbers, and that every task's gold actions replay. When task ids change, regenerate `tasks_voice.json` with `python -m tau2.user_simulation_voice_presets food` (default `--seed 42`). Per-task seeds use Python's `hash()`, so set `PYTHONHASHSEED=0` for a reproducible file.

## Grading

- **DB (every task).** `FoodTools.get_db_hash` hashes the active `store_id`, **every submitted order in order** (items, `order_type`, `scheduled_time`, channel, and for mobile orders the normalized name, phone digits, and delivery address), `transferred` (whether `transfer_to_human_agents` was called), the current cart's items (ignored when `transferred` is `True`, so partial carts before an escalation do not fail), and all **confirmed reservations** by content (date, time, party size, normalized name, phone digits, seating, high chairs; ids are ignored, so a booking made and cancelled in the same conversation counts as never made). Line items are canonicalized, so equivalent constructions hash the same: identical lines are merged, modifier quantities are summed, half-pizza mirrors are swapped, a default removed with `quantity: 0` equals choosing its "No ..." option, explicit `$0` standard toppings in optional groups that include a matching `"No ..."` removal option (on stores that do not expand default ingredients) are stripped, and a default applied twice equals an explicit quantity of 2. Not calling `place_order`, placing two orders, or leaving items in the cart after ordering all differ from the gold outcome.
- **NL assertions (19 of 62 tasks).** Used only for what the DB can't see, i.e. what the agent said: price quotes, counts it confirmed, and false claims. Never for cart contents. These tasks depend on the upstream NL-assertion evaluator, which `docs/evaluation.md` marks experimental, and on its judge model (`DEFAULT_LLM_NL_ASSERTIONS` in `src/tau2/config.py`).

## Running

```bash
# Text mode
tau2 run --domain food --agent-llm <model> --user-llm <model> --task-split-name base

# Voice (full-duplex): report both speech conditions so audio and turn-taking effects
# (control vs regular) can be separated from reasoning errors.
tau2 run --domain food --benchmark none --audio-native --audio-native-provider gemini --speech-complexity control
tau2 run --domain food --benchmark none --audio-native --audio-native-provider gemini --speech-complexity regular
```

Re-grade saved runs with `tau2 evaluate-trajs`.

## Tests

`tests/test_domains/test_food/`:
- `test_tools_food.py`: the tools (menu lookups, cart edits, fulfillment, ordering, reservations) and their validation errors.
- `test_food_grading.py`: golden replays, `get_db_hash` equivalences, and session isolation.
- `test_food_environment.py`: registration, environment construction, and the agent prompt (Current Store and Menu Data).
- `test_food_menus.py`: menu data integrity (id formats, references, reachability, unique names, every item orderable at every size).
- `test_food_tasks.py`: task data (splits, grading design, user script conventions, fictional contact details).
