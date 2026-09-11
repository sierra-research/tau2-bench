# Copyright Sierra
"""Design ElevenLabs persona voices and auto-pin them into a language pack.

Programmatic core behind ``tau2 factory generate-assets`` (voices): designs a
voice for each persona from its ``tts_voice_prompt``, saves it to the ElevenLabs
account, and writes the ``voice_id`` back into the pack.yaml. The generic
Voice Design wrappers live in :mod:`tau2.voice.utils.voice_design`.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Optional

import yaml
from loguru import logger
from pydantic import BaseModel, Field

from tau2.multilingual.factory.pack_text_edit import set_persona_voice_id
from tau2.utils import DATA_DIR
from tau2.voice.utils.voice_design import (
    MIN_AUDITION_CHARS,
    VOICE_DESIGN_MODEL,
    VOICE_DESIGN_SEED,
    design_and_create_voice,
)

VOICE_NAME_PREFIX = "tau2"

# Sentinel a pack author writes into voice_id to request generation; treated the
# same as an empty/missing id by generate_pack_voices.
PENDING_VOICE_ID = "PENDING_GENERATION"


def _needs_voice(voice_id) -> bool:
    """True if a persona's voice_id is missing or a generation placeholder."""
    return not voice_id or str(voice_id).strip() in ("", PENDING_VOICE_ID)


# Per-language audition scripts (≥100 chars; the API rejects shorter `text`).
# These are native ~30s audition scripts. The pt script uses Brazilian Portuguese
# forms such as gerunds, 'pra', 'você', and 'dar uma conferida'. A language maps to either a single
# gender-neutral script or, for languages with gendered first-person grammar
# (e.g. Hindi "जा रहा हूँ" vs "जा रही हूँ"), a {gender: script} map so each
# persona is designed and auditioned in its own grammatical gender. Falls back
# to the English sample for unlisted languages.
_AUDITION_TEXT: dict[str, str | dict[str, str]] = {
    "hi": {
        "male": (
            "तो, मैं अगले महीने अपनी बहन से मिलने फ्लाइट से जा रहा हूँ, और "
            "अभी-अभी पता चला कि मेरा वर्क शेड्यूल बदल गया है। मैं सोच रहा "
            "था कि क्या मेरी फ्लाइट दो-तीन दिन आगे कर सकते हैं, बेहतर होगा "
            "अगर शाम की हो। और मैं ये भी चेक करना चाहूँगा कि मेरी सीट में "
            "अभी भी एक्स्ट्रा लेगरूम है या नहीं, क्योंकि आजकल मेरे घुटने "
            "में थोड़ी दिक्कत चल रही है। ओह, और पिछली बार मील ऑप्शन कन्फर्म "
            "नहीं हुआ था, तो अगर आप वो भी डबल-चेक कर दें, तो बहुत अच्छा "
            "होगा। सच कहूँ तो, मैं बस चाहता हूँ कि सब कुछ सेट हो जाए, ताकि "
            "इसकी टेंशन लेना बंद कर सकूँ।"
        ),
        "female": (
            "तो, मैं अगले महीने अपनी बहन से मिलने फ्लाइट से जा रही हूँ, और "
            "अभी-अभी पता चला कि मेरा वर्क शेड्यूल बदल गया है। मैं सोच रही "
            "थी कि क्या मेरी फ्लाइट दो-तीन दिन आगे कर सकते हैं, बेहतर होगा "
            "अगर शाम की हो। और मैं ये भी चेक करना चाहूँगी कि मेरी सीट में "
            "अभी भी एक्स्ट्रा लेगरूम है या नहीं, क्योंकि आजकल मेरे घुटने "
            "में थोड़ी दिक्कत चल रही है। ओह, और पिछली बार मील ऑप्शन कन्फर्म "
            "नहीं हुआ था, तो अगर आप वो भी डबल-चेक कर दें, तो बहुत अच्छा "
            "होगा। सच कहूँ तो, मैं बस चाहती हूँ कि सब कुछ सेट हो जाए, ताकि "
            "इसकी टेंशन लेना बंद कर सकूँ।"
        ),
    },
    "zh": (
        "你好，我下个月要飞去看我姐姐，然后我刚发现我的工作排班改了。我想把航班往后改个两天，最好是晚上的。另外我也想确认一下，我的座位是不是还有加大腿部空间，因为我最近膝盖有点不太舒服。哦对了，上次我的餐食选项好像没弄成功，所以如果你能帮我再确认一下就太好了。说实话，我就是想在放心之前把这些都处理好。"
    ),
    "ko": (
        "그러니까, 다음 달에 여동생 보러 비행기 타고 가는데요, 방금 회사 근무 일정이 바뀐 걸 알게 됐어요. 그래서 "
        "항공편을 며칠 뒤로 미룰 수 있으면 좋겠고, 가능하면 저녁 시간대로요. 그리고 제 좌석이 아직 다리 공간이 좀 "
        "더 넓은 자리인지도 확인하고 싶어요. 요즘 무릎이 좀 안 좋아서요. 아, 그리고 지난번엔 기내식 옵션이 제대로 "
        "적용이 안 됐거든요. 그것도 한 번만 다시 확인해 주시면 좋겠습니다. 솔직히 말하면, 괜히 계속 신경 쓰이기 "
        "전에 다 정리해 두고 싶어요."
    ),
    "es": {
        "male": (
            "Bueno, vuelo para ver a mi hermana el mes que viene y acabo de "
            "enterarme de que me cambiaron el horario en el trabajo. Quería "
            "ver si puedo mover mi vuelo a un par de días después, "
            "idealmente por la tarde-noche. También me gustaría comprobar "
            "si mi asiento sigue teniendo espacio extra para las piernas, "
            "porque últimamente la rodilla me está dando guerra. Ah, y la "
            "última vez no quedó registrada la opción de comida, así que si "
            "pudiera revisarlo también, sería genial. La verdad, solo "
            "quiero dejarlo todo arreglado para poder quedarme tranquilo."
        ),
        "female": (
            "Bueno, vuelo para ver a mi hermana el mes que viene y acabo de "
            "enterarme de que me cambiaron el horario en el trabajo. Quería "
            "ver si puedo mover mi vuelo a un par de días después, "
            "idealmente por la tarde-noche. También me gustaría comprobar "
            "si mi asiento sigue teniendo espacio extra para las piernas, "
            "porque últimamente la rodilla me está dando guerra. Ah, y la "
            "última vez no quedó registrada la opción de comida, así que si "
            "pudiera revisarlo también, sería genial. La verdad, solo "
            "quiero dejarlo todo arreglado para poder quedarme tranquila."
        ),
    },
    "pt": (
        "Então, mês que vem eu vou viajar de avião pra visitar minha "
        "irmã, e acabei de descobrir que meu horário de trabalho mudou. "
        "Eu queria ver se dá pra empurrar meu voo uns dois ou três "
        "dias, de preferência à noite. Também queria confirmar se meu "
        "assento ainda tem espaço extra pras pernas, porque meu joelho "
        "anda me incomodando ultimamente. Ah, e da última vez a opção "
        "de refeição não ficou confirmada, então se você puder dar uma "
        "conferida nisso também, seria ótimo. Sinceramente, eu só quero "
        "deixar tudo resolvido pra parar de me preocupar com isso."
    ),
}
_DEFAULT_AUDITION_TEXT = (
    "Hi, I'm calling about an issue with my account. I tried logging in earlier "
    "today but it keeps saying my password is wrong. I'm pretty sure I haven't "
    "changed it recently, so I'm not sure what's going on. Could you help me "
    "figure this out?"
)


def audition_text_for(language: str, gender: Optional[str] = None) -> str:
    """A locale-appropriate Voice Design audition line (guaranteed ≥100 chars).

    For languages whose audition line is gender-specific (gendered first-person
    grammar), ``gender`` ("male"/"female") selects the matching variant so a
    persona is never auditioned in the wrong grammatical gender. It is ignored
    for languages with a single neutral line. Falls back to the male variant
    (then the English sample) when a gender is missing or unrecognized.
    """
    entry = _AUDITION_TEXT.get(language, _DEFAULT_AUDITION_TEXT)
    if isinstance(entry, dict):
        text = (
            entry.get(gender or "") or entry.get("male") or next(iter(entry.values()))
        )
    else:
        text = entry
    if len(text) < MIN_AUDITION_CHARS:  # defensive — keep the API happy
        text = (text + " " + _DEFAULT_AUDITION_TEXT)[: max(120, len(text))]
    return text


# --------------------------------------------------------------------------- #
# Pack-level orchestration
# --------------------------------------------------------------------------- #


class PersonaVoiceStatus(str, Enum):
    """Per-persona voice-design outcome."""

    CREATED = "created"
    NO_PREVIEWS = "no_previews"
    ERROR = "error"
    SKIPPED = "skipped"


class PersonaVoiceResult(BaseModel):
    """One persona's voice-design outcome (``generate_pack_voices``)."""

    persona_id: str = Field(description="Pack persona id")
    display_name: str = Field(description="Persona display name")
    voice_id: Optional[str] = Field(
        description="The pinned ElevenLabs voice id (None unless created/skipped)"
    )
    status: PersonaVoiceStatus
    detail: str = Field(default="", description="Human-readable status detail")


def _pack_path(language: str) -> Path:
    return DATA_DIR / "tau2" / "multilingual" / language / "pack.yaml"


def _persona_gender(persona: dict) -> Optional[str]:
    """Read a persona's gender from its ``tags.gender`` (``male``/``female``)."""
    tags = persona.get("tags") or {}
    g = tags.get("gender")
    return str(g).lower() if g else None


def validate_one_male_one_female(personas: dict) -> list[str]:
    """Return problems if the pack is not exactly one male + one female persona."""
    problems: list[str] = []
    genders = {pid: _persona_gender(p) for pid, p in personas.items()}
    missing = [pid for pid, g in genders.items() if g not in ("male", "female")]
    if missing:
        problems.append(
            "personas missing a tags.gender of male/female: " + ", ".join(missing)
        )
        return problems
    males = [pid for pid, g in genders.items() if g == "male"]
    females = [pid for pid, g in genders.items() if g == "female"]
    if len(males) != 1 or len(females) != 1:
        problems.append(
            f"expected exactly 1 male + 1 female persona, got "
            f"male={males} female={females}"
        )
    return problems


def generate_pack_voices(
    language: str,
    *,
    client=None,
    model_id: str = VOICE_DESIGN_MODEL,
    seed: int = VOICE_DESIGN_SEED,
    force: bool = False,
    dry_run: bool = False,
) -> list[PersonaVoiceResult]:
    """Design + pin a voice for every persona in ``<lang>/pack.yaml``.

    Enforces the one-male/one-female invariant, then for each persona designs a
    voice from its ``tts_voice_prompt`` and writes the saved ``voice_id`` back
    into the pack (one line per persona, minimal diff).

    By default a persona that already has a real ``voice_id`` is left untouched
    (only missing / ``PENDING_GENERATION`` ones are generated); pass
    ``force=True`` to re-design every persona.
    """
    pack_path = _pack_path(language)
    pack_text = pack_path.read_text()
    data = yaml.safe_load(pack_text)
    personas: dict = data.get("personas") or {}

    problems = validate_one_male_one_female(personas)
    if problems:
        raise ValueError(
            "pack persona-gender invariant failed:\n  - " + "\n  - ".join(problems)
        )

    results: list[PersonaVoiceResult] = []

    if dry_run:
        for pid, p in personas.items():
            results.append(
                PersonaVoiceResult(
                    persona_id=pid,
                    display_name=p.get("display_name", pid),
                    voice_id=None,
                    status=PersonaVoiceStatus.SKIPPED,
                    detail="dry-run",
                )
            )
        return results

    if client is None:
        from elevenlabs import ElevenLabs

        client = ElevenLabs()

    for pid, p in personas.items():
        display = p.get("display_name", pid)
        prompt = p.get("tts_voice_prompt") or ""
        short = p.get("short_description") or display
        voice_name = f"{VOICE_NAME_PREFIX}_{pid}"
        if not force and not _needs_voice(p.get("voice_id")):
            results.append(
                PersonaVoiceResult(
                    persona_id=pid,
                    display_name=display,
                    voice_id=p.get("voice_id"),
                    status=PersonaVoiceStatus.SKIPPED,
                    detail="already pinned",
                )
            )
            continue
        if not prompt:
            results.append(
                PersonaVoiceResult(
                    persona_id=pid,
                    display_name=display,
                    voice_id=None,
                    status=PersonaVoiceStatus.ERROR,
                    detail="empty tts_voice_prompt",
                )
            )
            continue
        # Audition each persona in its own grammatical gender (matters for
        # languages with gendered first-person verb agreement, e.g. Hindi).
        audition = audition_text_for(language, _persona_gender(p))
        try:
            voice_id = design_and_create_voice(
                client,
                voice_description=prompt,
                voice_name=voice_name,
                short_description=short,
                audition_text=audition,
                model_id=model_id,
                seed=seed,
            )
        except Exception as e:  # noqa: BLE001
            logger.error(f"voice design failed for {pid}: {e}")
            results.append(
                PersonaVoiceResult(
                    persona_id=pid,
                    display_name=display,
                    voice_id=None,
                    status=PersonaVoiceStatus.ERROR,
                    detail=str(e),
                )
            )
            continue

        if not voice_id:
            results.append(
                PersonaVoiceResult(
                    persona_id=pid,
                    display_name=display,
                    voice_id=None,
                    status=PersonaVoiceStatus.NO_PREVIEWS,
                )
            )
            continue

        # Auto-pin: rewrite just this persona's voice_id line.
        pack_text = set_persona_voice_id(pack_text, pid, voice_id)
        pack_path.write_text(pack_text)
        results.append(
            PersonaVoiceResult(
                persona_id=pid,
                display_name=display,
                voice_id=voice_id,
                status=PersonaVoiceStatus.CREATED,
                detail=voice_name,
            )
        )
        logger.info(f"pinned {pid} -> {voice_id}")

    return results
