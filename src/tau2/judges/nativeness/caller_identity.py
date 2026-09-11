# Copyright Sierra
"""Typed caller-name context for precision-first nativeness checks.

Korean identity tasks store reviewed locale names in Romanized form because
those values must also work as user ids and email handles.  The judge needs the
spoken Hangul form, but must never invent it from arbitrary transcript text.
This module therefore owns a closed, reviewed alias table for the fixed Korean
locale corpus and resolves names only from an identity task's structured patch.

WHERE that patch carries the name is not this module's knowledge: extraction
dispatches on the domain profile's ``CallerIdentityKind``
(``tau2.multilingual.domain_profiles``), so an unregistered domain raises at
the profile lookup and a registered domain whose identity kind has no
extraction here raises ``NotImplementedError`` — never a silent judge skip.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tau2.data_model.tasks import Task
from tau2.multilingual.domain_profiles import CallerIdentityKind, get_domain_profile
from tau2.multilingual.names import NameOrder, compose_full_name, split_full_name

KOREAN_CALLER_NAME_PROVENANCE = "ko_locale_corpus_reviewed_spoken_aliases_v1"

KOREAN_GIVEN_NAME_ALIASES: dict[str, str] = {
    "Seoyeon": "서연",
    "Jiwoo": "지우",
    "Minseo": "민서",
    "Haeun": "하은",
    "Yuna": "유나",
    "Jiyoon": "지윤",
    "Soyeon": "소연",
    "Eunji": "은지",
    "Hyejin": "혜진",
    "Sujin": "수진",
    "Minji": "민지",
    "Jieun": "지은",
    "Yerin": "예린",
    "Chaewon": "채원",
    "Jiyeon": "지연",
    "Sumin": "수민",
    "Dahye": "다혜",
    "Hyunji": "현지",
    "Seulgi": "슬기",
    "Nayeon": "나연",
    "Minjun": "민준",
    "Seojun": "서준",
    "Jihoon": "지훈",
    "Doyun": "도윤",
    "Hajun": "하준",
    "Joonho": "준호",
    "Sungmin": "성민",
    "Hyunwoo": "현우",
    "Jaehyun": "재현",
    "Woojin": "우진",
    "Donghyun": "동현",
    "Seungmin": "승민",
    "Junseo": "준서",
    "Yejun": "예준",
    "Siwoo": "시우",
    "Jinwoo": "진우",
    "Kyungho": "경호",
    "Sangwoo": "상우",
    "Taehyun": "태현",
    "Dongwook": "동욱",
}

KOREAN_FAMILY_NAME_ALIASES: dict[str, str] = {
    "Kim": "김",
    "Lee": "이",
    "Park": "박",
    "Choi": "최",
    "Jung": "정",
    "Kang": "강",
    "Cho": "조",
    "Yoon": "윤",
    "Jang": "장",
    "Lim": "임",
    "Han": "한",
    "Oh": "오",
    "Seo": "서",
    "Shin": "신",
    "Kwon": "권",
    "Hwang": "황",
    "Ahn": "안",
    "Song": "송",
    "Yoo": "유",
    "Hong": "홍",
    "Jeon": "전",
    "Moon": "문",
    "Baek": "백",
    "Heo": "허",
}


class CallerNameContext(BaseModel):
    """Authoritative localized caller name with role-preserving spoken aliases."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    language: Literal["ko"] = "ko"
    given_romanized: str
    family_romanized: str
    full_romanized: str
    given_spoken: str
    family_spoken: str
    full_spoken: str
    provenance: str = Field(default=KOREAN_CALLER_NAME_PROVENANCE)

    @model_validator(mode="after")
    def _roles_and_display_order_are_consistent(self) -> "CallerNameContext":
        expected_romanized = compose_full_name(
            self.given_romanized, self.family_romanized, NameOrder.FAMILY_FIRST
        )
        expected_spoken = f"{self.family_spoken}{self.given_spoken}"
        if self.full_romanized != expected_romanized:
            raise ValueError("Korean caller full_romanized must be family-first")
        if self.full_spoken != expected_spoken:
            raise ValueError("Korean caller full_spoken must be family-first")
        return self

    @property
    def reversed_spoken(self) -> str:
        """The exact given-first form that the deterministic checker rejects."""
        return f"{self.given_spoken} {self.family_spoken}"

    @property
    def protected_name_literals(self) -> frozenset[str]:
        """Known name values that generic task-literal stripping must preserve."""
        return frozenset(
            {
                self.given_romanized,
                self.family_romanized,
                self.full_romanized,
                self.given_spoken,
                self.family_spoken,
                self.full_spoken,
            }
        )


def korean_caller_name(
    given_romanized: str, family_romanized: str
) -> Optional[CallerNameContext]:
    """Resolve one reviewed corpus name; unknown aliases remain unresolved."""
    given = KOREAN_GIVEN_NAME_ALIASES.get(given_romanized)
    family = KOREAN_FAMILY_NAME_ALIASES.get(family_romanized)
    if given is None or family is None:
        return None
    return CallerNameContext(
        given_romanized=given_romanized,
        family_romanized=family_romanized,
        full_romanized=compose_full_name(
            given_romanized, family_romanized, NameOrder.FAMILY_FIRST
        ),
        given_spoken=given,
        family_spoken=family,
        full_spoken=f"{family}{given}",
    )


def _patched_name_roles(
    task: Task, identity_kind: CallerIdentityKind
) -> Optional[tuple[str, str]]:
    state = task.initial_state
    initialization = state.initialization_data if state is not None else None
    agent_data = initialization.agent_data if initialization is not None else None
    if not isinstance(agent_data, dict):
        return None

    if identity_kind in (
        CallerIdentityKind.USER_ID_HANDLE,
        CallerIdentityKind.PROSE_NAME_ZIP,
    ):
        # Both kinds patch the caller as the sole record in agent_data["users"].
        users = agent_data.get("users")
        if not isinstance(users, dict) or len(users) != 1:
            return None
        record = next(iter(users.values()))
        name = record.get("name") if isinstance(record, dict) else None
        if not isinstance(name, dict):
            return None
        given, family = name.get("first_name"), name.get("last_name")
        return (
            (given, family)
            if isinstance(given, str) and isinstance(family, str)
            else None
        )

    if identity_kind is CallerIdentityKind.STRUCTURED_NAME_PHONE:
        actions = state.initialization_actions if state is not None else None
        set_user_names = {
            action.arguments.get("name")
            for action in (actions or [])
            if action.env_type == "user"
            and action.func_name == "set_user_info"
            and isinstance(action.arguments.get("name"), str)
        }
        customers = agent_data.get("customers")
        if len(set_user_names) != 1 or not isinstance(customers, list):
            return None
        full_name = next(iter(set_user_names))
        matching = [
            customer
            for customer in customers
            if isinstance(customer, dict) and customer.get("full_name") == full_name
        ]
        if len(matching) != 1:
            return None
        return split_full_name(full_name, NameOrder.FAMILY_FIRST)

    raise NotImplementedError(
        f"Caller-identity kind {identity_kind!r} has no Korean name-roles "
        "extraction. A domain registered with this kind cannot be judged "
        "silently: teach tau2.judges.nativeness.caller_identity where its "
        "identity patch carries the name."
    )


def resolve_korean_caller_name(
    task: Task, language: Optional[str], domain: Optional[str]
) -> Optional[CallerNameContext]:
    """Resolve a name only from a Korean identity task's structured locale patch."""
    language = (language or "").lower()
    domain = (domain or "").lower()
    if language != "ko" or not domain or not task.id.endswith("_ko_identity"):
        return None
    identity_kind = get_domain_profile(domain).caller_identity
    roles = _patched_name_roles(task, identity_kind)
    return korean_caller_name(*roles) if roles is not None else None
