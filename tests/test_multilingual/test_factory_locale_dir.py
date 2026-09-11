# Copyright Sierra
"""Tests for locale-dir derivation + bed-path localization.

Locale beds live under a ``language_COUNTRY`` dir (es_ES,
ar_EG) and the pack's ``background_noise_files`` must point there. The dir is
derived from the personas' ISO 3166-2 ``locale`` codes (no hardcoded table), and
the assembler rewrites the bare ``<lang>/`` bed paths to it once personas exist.
"""

from __future__ import annotations

from tau2.multilingual.factory.pack_assembly import (
    localize_acoustic_bed_paths,
    reassign_personas_to_locale_beds,
)
from tau2.multilingual.factory.paths import derive_locale_dir


def _bed_presets():
    return {
        "x_outdoor_traffic": {
            "background_noise_files": ["x_XX/busy_street_iphone_mic.wav"]
        },
        "x_office": {"background_noise_files": ["people_talking.wav"]},
        "x_household_multi_gen": {
            "background_noise_files": ["x_XX/medium_size_room_tv_news_iphone_mic.wav"]
        },
    }


def test_reassign_moves_persona_off_shared_office_bed():
    # Model parked the formal persona on office (shared English bed), leaving the
    # locale household bed unused -> reassign so both personas are on locale beds.
    personas = {
        "formal": {"acoustic_preset_id": "x_office"},
        "casual": {"acoustic_preset_id": "x_outdoor_traffic"},
    }
    reassign_personas_to_locale_beds(personas, _bed_presets())
    assert personas["formal"]["acoustic_preset_id"] == "x_household_multi_gen"
    assert personas["casual"]["acoustic_preset_id"] == "x_outdoor_traffic"


def test_reassign_noop_when_both_personas_already_on_locale_beds():
    personas = {
        "a": {"acoustic_preset_id": "x_household_multi_gen"},
        "b": {"acoustic_preset_id": "x_outdoor_traffic"},
    }
    reassign_personas_to_locale_beds(personas, _bed_presets())
    assert personas["a"]["acoustic_preset_id"] == "x_household_multi_gen"
    assert personas["b"]["acoustic_preset_id"] == "x_outdoor_traffic"


def test_derive_locale_dir_from_persona_country():
    # Single-country packs -> language_COUNTRY.
    assert derive_locale_dir("ar", ["EG-C"]) == "ar_EG"
    assert derive_locale_dir("ru", ["RU-MOW"]) == "ru_RU"
    assert derive_locale_dir("ko", ["KR-11"]) == "ko_KR"
    assert derive_locale_dir("zh", ["CN-BJ"]) == "zh_CN"


def test_derive_locale_dir_reproduces_merged_convention():
    # The two merged packs were hand-mapped (hi->hi_IN, ro->ro_RO); derivation
    # must reproduce them from the persona locales so they need no special case.
    assert derive_locale_dir("hi", ["IN-MH", "IN-TG"]) == "hi_IN"
    assert derive_locale_dir("ro", ["RO-B"]) == "ro_RO"


def test_derive_locale_dir_ties_anchor_on_first_persona():
    # Mixed-locale pack (Spain + Argentina): the first persona's country anchors.
    assert derive_locale_dir("es", ["ES-MD", "AR-C"]) == "es_ES"


def test_derive_locale_dir_falls_back_to_bare_language():
    assert derive_locale_dir("xx", [None, "", "garbage"]) == "xx"
    assert derive_locale_dir("xx", []) == "xx"


def test_localize_acoustic_bed_paths_only_rewrites_locale_beds():
    presets = {
        "es_outdoor_traffic": {
            "background_noise_files": ["es/busy_street_iphone_mic.wav"],
            "burst_noise_files": ["car_horn.wav"],
        },
        "es_office": {"background_noise_files": ["people_talking.wav"]},
        "es_household_multi_gen": {
            "background_noise_files": ["es/medium_size_room_tv_news_iphone_mic.wav"],
        },
    }
    localize_acoustic_bed_paths(presets, "es", "es_ES")
    assert presets["es_outdoor_traffic"]["background_noise_files"] == [
        "es_ES/busy_street_iphone_mic.wav"
    ]
    assert presets["es_household_multi_gen"]["background_noise_files"] == [
        "es_ES/medium_size_room_tv_news_iphone_mic.wav"
    ]
    # Shared English office bed and bursts are untouched.
    assert presets["es_office"]["background_noise_files"] == ["people_talking.wav"]
    assert presets["es_outdoor_traffic"]["burst_noise_files"] == ["car_horn.wav"]


def test_localize_acoustic_bed_paths_noop_when_bare():
    presets = {
        "x_outdoor": {"background_noise_files": ["xx/busy_street_iphone_mic.wav"]}
    }
    localize_acoustic_bed_paths(presets, "xx", "xx")
    assert presets["x_outdoor"]["background_noise_files"] == [
        "xx/busy_street_iphone_mic.wav"
    ]
