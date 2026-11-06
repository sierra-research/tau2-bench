"""Store- and channel-matched voice presets for food domain tasks."""

import json
import random
import re
from collections.abc import Sequence
from functools import lru_cache
from typing import Literal

from tau2.data_model.tasks import Task
from tau2.data_model.voice import SampledVoiceConfig
from tau2.domains.food.data_model import CartState
from tau2.domains.food.environment import get_tasks
from tau2.user_simulation_voice_presets import (
    COMPLEXITY_SEED_OFFSETS,
    EffectsMode,
)

FoodStoreId = Literal["cafe", "pizza", "burger"]
FoodChannel = Literal["drive_thru", "mobile"]

FOOD_BACKGROUND_SPEECH_EVENTS_PER_MINUTE = 2.0
FOOD_HEAVY_BACKGROUND_SPEECH_EVENTS_PER_MINUTE = 3.0
FOOD_OOD_SPEECH_SLOTS = 3
FOOD_IN_DOMAIN_SPEECH_SLOTS = 7

FOOD_DRIVE_THRU_BG_FILES: tuple[str, ...] = (
    "busy_street_iphone_mic.wav",
    "food/car_cabin_radio_music.wav",
    "food/car_cabin_radio_talk.wav",
)

FOOD_MOBILE_BG_FILES: tuple[str, ...] = (
    "people_talking.wav",
    "medium_size_room_tv_news_iphone_mic.wav",
    "food/living_room_tv_broadcast.wav",
)

# Pure non-speech environmental acoustic bursts (mixed via burst_noise at -5..10 dB).
FOOD_DRIVE_THRU_ACOUSTIC_BURSTS: tuple[str, ...] = (
    "car_horn.wav",
    "engine_idling.wav",
    "siren.wav",
)

FOOD_MOBILE_ACOUSTIC_BURSTS: tuple[str, ...] = (
    "ringing_phone.wav",
    "dog_bark.wav",
)

# Out-of-domain secondary-speaker background chatter (zero menu/booking commands;
# mixed via background_speech at 6..12 dB across 3 of 10 pool slots = 30%).
FOOD_OUT_OF_DOMAIN_DRIVE_THRU_SPEECH: tuple[str, ...] = (
    "food/out_of_domain/side_speech_seatbelt.wav",
    "food/out_of_domain/side_speech_radio_down.wav",
    "food/out_of_domain/side_speech_car_chat_exit.wav",
)

FOOD_OUT_OF_DOMAIN_MOBILE_SPEECH: tuple[str, ...] = (
    "food/out_of_domain/side_speech_sit_down.wav",
    "food/out_of_domain/side_speech_quiet.wav",
    "food/out_of_domain/side_speech_family_chat.wav",
    "food/out_of_domain/side_speech_roommate_plates.wav",
)

_RESERVATION_CONFLICTS: tuple[str, ...] = (
    "check table availability",
    "reservation",
    "reservations",
    "book a table",
    "reserve a table",
    "table for",
)

# In-domain store- and channel-matched secondary-speaker distractors: (burst_subpath, conflict_phrases).
# 5 drive-thru cross-lane clips + 4 mobile household clips per store (zero brand trademarks).
FOOD_IN_DOMAIN_DISTRACTORS: dict[
    tuple[FoodStoreId, FoodChannel], tuple[tuple[str, tuple[str, ...]], ...]
] = {
    ("cafe", "drive_thru"): (
        ("food/in_domain/other_lane_cafe_mocha.wav", ("Mocha", "Croissant")),
        ("food/in_domain/other_lane_cafe_panini.wav", ("Panini", "Lemonade")),
        (
            "food/in_domain/other_lane_cafe_matcha.wav",
            ("Matcha", "Oat Milk", "Croissant"),
        ),
        ("food/in_domain/other_lane_cafe_cold_brew.wav", ("Cold Brew", "Muffin")),
        ("food/in_domain/other_lane_cafe_window.wav", ()),
    ),
    ("cafe", "mobile"): (
        ("food/in_domain/household_cafe_mocha.wav", ("Mocha",)),
        ("food/in_domain/household_cafe_turkey.wav", ("Turkey Club", "Chips")),
        ("food/in_domain/household_cafe_chai.wav", ("Chai", "Cookie")),
        (
            "food/in_domain/household_cafe_table.wav",
            (*_RESERVATION_CONFLICTS, "outdoor table", "outdoor seating"),
        ),
    ),
    ("pizza", "drive_thru"): (
        ("food/in_domain/other_lane_pizza_sticks.wav", ("Mozzarella Sticks", "Cola")),
        ("food/in_domain/other_lane_pizza_ziti.wav", ("Ziti", "Cannoli")),
        (
            "food/in_domain/other_lane_pizza_classic.wav",
            ("Classic Pie", "Extra Cheese", "Garlic Knots"),
        ),
        ("food/in_domain/other_lane_pizza_wings.wav", ("Wings", "Ranch")),
        ("food/in_domain/other_lane_pizza_window.wav", ()),
    ),
    ("pizza", "mobile"): (
        ("food/in_domain/household_pizza_ziti.wav", ("Ziti", "Cannoli")),
        ("food/in_domain/household_pizza_calzone.wav", ("Calzone", "Marinara")),
        ("food/in_domain/household_pizza_hero.wav", ("Chicken Parm", "Lemonade")),
        (
            "food/in_domain/household_pizza_table.wav",
            (*_RESERVATION_CONFLICTS, "indoor table", "indoor seating"),
        ),
    ),
    ("burger", "drive_thru"): (
        ("food/in_domain/other_lane_burger_pier.wav", ("Pier Burger", "Banana Shake")),
        ("food/in_domain/other_lane_burger_wrap.wav", ("Chicken Wrap", "Nuggets")),
        (
            "food/in_domain/other_lane_burger_cheeseburgers.wav",
            ("Cheeseburger", "Cheeseburgers", "Fries"),
        ),
        ("food/in_domain/other_lane_burger_shake.wav", ("Vanilla Shake", "Nuggets")),
        ("food/in_domain/other_lane_burger_window.wav", ()),
    ),
    ("burger", "mobile"): (
        (
            "food/in_domain/household_burger_fish.wav",
            ("Fish Sandwich", "Apple Slices"),
        ),
        ("food/in_domain/household_burger_sundae.wav", ("Sundae", "Fries")),
        ("food/in_domain/household_burger_wrap.wav", ("Chicken Wrap", "Root Beer")),
        ("food/in_domain/household_burger_ranch.wav", ("Onion", "Onions", "Ranch")),
    ),
}


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def get_food_task_store_and_channel(task: Task) -> tuple[FoodStoreId, FoodChannel]:
    """Resolve a food task's store ('cafe'|'pizza'|'burger') and channel ('drive_thru'|'mobile')."""
    init = task.initial_state.initialization_data if task.initial_state else None
    cart_data = ((init.agent_data or {}) if init else {}).get("cart") or {}
    cart = CartState.model_validate(cart_data)
    if cart.store_id not in ("cafe", "pizza", "burger"):
        raise ValueError(
            f"Unexpected food task store_id={cart.store_id!r} on {task.id}"
        )
    return cart.store_id, cart.channel


def get_food_in_domain_distractor_files(task: Task) -> list[str]:
    """Return in-domain store- and channel-matched speech distractors disjoint from the task's order."""
    store_id, channel = get_food_task_store_and_channel(task)
    actions = (
        task.evaluation_criteria.actions if task.evaluation_criteria else None
    ) or []
    haystack = _norm(
        " ".join(
            [
                *(f"{a.name} {json.dumps(a.arguments)}" for a in actions),
                str(task.user_scenario or ""),
            ]
        )
    )
    return [
        subpath
        for subpath, conflicts in FOOD_IN_DOMAIN_DISTRACTORS[(store_id, channel)]
        if not any(
            re.search(rf"\b{re.escape(_norm(c))}\b", haystack) for c in conflicts
        )
    ]


def _expand_slot_bucket(
    items: Sequence[str],
    slots: int,
    rng: random.Random,
) -> list[str]:
    """Deterministically fill `slots` entries from `items` with minimal variance."""
    bucket = list(items)
    full, rem = divmod(slots, len(bucket))
    return bucket * full + rng.sample(bucket, rem)


def build_food_background_speech_pool(
    task: Task,
    *,
    seed: int = 42,
) -> list[str]:
    """Build a deterministic 10-slot `background_speech_files` pool (3 OOD + 7 in-domain)."""
    _, channel = get_food_task_store_and_channel(task)
    ood_files = (
        FOOD_OUT_OF_DOMAIN_DRIVE_THRU_SPEECH
        if channel == "drive_thru"
        else FOOD_OUT_OF_DOMAIN_MOBILE_SPEECH
    )
    in_domain_files = get_food_in_domain_distractor_files(task)
    rng = random.Random(seed)
    return [
        *_expand_slot_bucket(ood_files, FOOD_OOD_SPEECH_SLOTS, rng),
        *_expand_slot_bucket(in_domain_files, FOOD_IN_DOMAIN_SPEECH_SLOTS, rng),
    ]


@lru_cache(maxsize=1)
def _food_task_map() -> dict[str, Task]:
    return {task.id: task for task in get_tasks()}


def apply_food_voice_preset(
    task_id: str,
    sampled: SampledVoiceConfig,
    *,
    seed: int,
    channel_effects_mode: EffectsMode = "regular",
) -> SampledVoiceConfig:
    """Apply store- and channel-matched food acoustic presets."""
    task = _food_task_map().get(task_id)
    if task is None or sampled.environment is None:
        return sampled

    store_id, channel = get_food_task_store_and_channel(task)
    source_cfg = sampled.source_effects_config
    complexity_offset = COMPLEXITY_SEED_OFFSETS.get(sampled.complexity, 0)

    bg_candidates = (
        FOOD_DRIVE_THRU_BG_FILES if channel == "drive_thru" else FOOD_MOBILE_BG_FILES
    )
    bg_noise_file = (
        random.Random(seed + complexity_offset + 1).choice(bg_candidates)
        if source_cfg.enable_background_noise
        else None
    )
    burst_files = (
        list(
            FOOD_DRIVE_THRU_ACOUSTIC_BURSTS
            if channel == "drive_thru"
            else FOOD_MOBILE_ACOUSTIC_BURSTS
        )
        if source_cfg.enable_burst_noise
        else []
    )
    bg_speech_files = build_food_background_speech_pool(
        task, seed=seed + complexity_offset + 2
    )
    bg_speech_rate = (
        FOOD_HEAVY_BACKGROUND_SPEECH_EVENTS_PER_MINUTE
        if channel_effects_mode == "heavy"
        else FOOD_BACKGROUND_SPEECH_EVENTS_PER_MINUTE
    )
    updated_source = source_cfg.model_copy(
        update={
            "enable_background_speech": True,
            "background_speech_events_per_minute": bg_speech_rate,
        }
    )
    return sampled.model_copy(
        update={
            "environment": f"food_{store_id}_{channel}",
            "background_noise_file": bg_noise_file,
            "burst_noise_files": burst_files,
            "background_speech_files": bg_speech_files,
            "source_effects_config": updated_source,
        }
    )
