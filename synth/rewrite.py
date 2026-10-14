"""LLM scenario rewriter + alignment gate for the failure-mode tasks.

The gold actions (the verifiable DB target) are ALWAYS derived in code; this
module only turns the deterministic `change_summary` into natural customer
phrasing. It never invents ids/quantities/orders and never decides the answer.
After rewriting, an alignment check (D4d) confirms the prose entails exactly the
coded goal; on mismatch it retries once, else raises.

Needs an API key (ANTHROPIC_API_KEY / OPENAI_API_KEY). Model via env
SYNTH_REWRITE_MODEL (default 'gpt-4.1'). Used by build_augmented.py --llm.
"""

from __future__ import annotations

import copy
import json
import os

import litellm

from patterns import TransformSpec
from seeds import Seed

MODEL = os.environ.get("SYNTH_REWRITE_MODEL", "gpt-4.1")

_REWRITE_SYS = (
    "You rewrite a retail customer-service scenario so a user-simulator can act "
    "it out naturally. You are given the ORIGINAL request and a STRUCTURED CHANGE "
    "to fold in. Rules: keep the customer's identity facts verbatim; do NOT change "
    "or invent any order ids, item ids, product names, quantities, or prices; state "
    "the new requirement explicitly and naturally; output ONLY JSON with keys "
    "'reason_for_call' and 'task_instructions'."
)

_ALIGN_SYS = (
    "You are a strict checker. Given a customer's request text and a STRUCTURED GOAL "
    "(the verifiable outcome), answer whether the request unambiguously asks for that "
    "goal and nothing contradictory. Output ONLY JSON {\"aligned\": true|false, "
    "\"why\": \"...\"}."
)


def _chat(system: str, user: str) -> str:
    resp = litellm.completion(
        model=MODEL, temperature=0.4,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
    )
    return resp["choices"][0]["message"]["content"]


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1].lstrip("json").strip()
    return json.loads(text)


def _aligned(reason: str, spec: TransformSpec) -> tuple[bool, str]:
    out = _parse_json(_chat(_ALIGN_SYS, json.dumps({
        "request": reason, "structured_goal": spec.change_summary,
        "behavioral_script": spec.behavioral_script})))
    return bool(out.get("aligned")), out.get("why", "")


def rewrite_scenario(seed: Seed, spec: TransformSpec) -> dict:
    instr = copy.deepcopy(seed.instructions)
    payload = json.dumps({
        "pattern": spec.pattern,
        "original_request": seed.reason_for_call,
        "identity_keep_verbatim": instr.get("known_info", ""),
        "structured_change": spec.change_summary,
        "mind_change_script": spec.behavioral_script,
    }, indent=2)

    for _ in range(2):
        out = _parse_json(_chat(_REWRITE_SYS, payload))
        reason = out.get("reason_for_call", "").strip()
        ok, _why = _aligned(reason, spec)
        if ok and reason:
            instr["reason_for_call"] = reason
            if out.get("task_instructions"):
                instr["task_instructions"] = out["task_instructions"].strip()
            elif spec.behavioral_script:
                instr["task_instructions"] = (
                    (instr.get("task_instructions", "") or "") + " " + spec.behavioral_script
                ).strip()
            return instr
    raise RuntimeError(f"rewrite failed alignment for {seed.id}/{spec.pattern}")
