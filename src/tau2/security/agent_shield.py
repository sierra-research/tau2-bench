"""AgentShield — Defense module for conversational agents.

A 5-stage defense pipeline that protects agents against prompt injection,
policy violations, tool misuse, and information leakage in multi-turn
conversations.

Designed for integration with τ²-bench agent evaluation. Each stage
runs independently and produces a verdict (safe/caution/block).

Architecture:
    Input → [Sanitize] → [Classify] → [Verify Policy] → [Validate Output] → [Decide]

References:
    - OWASP Top 10 for LLM Applications (2023)
    - Yi et al. (2023) "Benchmarking and Defending Against Indirect Prompt Injection"
    - Jain et al. (2023) "Baseline Defenses for Adversarial Attacks Against Aligned LLMs"
    - Zheng et al. (2023) "Judging LLM-as-a-Judge"

License: MIT
"""

import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Verdict(str, Enum):
    """Defense verdict for each stage."""

    SAFE = "safe"
    CAUTION = "caution"
    BLOCK = "block"


@dataclass
class StageResult:
    """Result from a single defense stage."""

    stage: str
    verdict: Verdict
    reason: str = ""
    details: dict = field(default_factory=dict)


@dataclass
class ShieldResult:
    """Aggregated result from all 5 stages."""

    final_verdict: Verdict
    stages: list[StageResult]
    blocked: bool = False
    block_reason: str = ""

    @property
    def is_safe(self) -> bool:
        return self.final_verdict == Verdict.SAFE

    def summary(self) -> dict:
        return {
            "verdict": self.final_verdict.value,
            "blocked": self.blocked,
            "block_reason": self.block_reason,
            "stages": [
                {"stage": s.stage, "verdict": s.verdict.value, "reason": s.reason}
                for s in self.stages
            ],
        }


# ═══════════════════════════════════════════════════════════════════
# STAGE 1: INPUT SANITIZATION
# ═══════════════════════════════════════════════════════════════════


def sanitize_input(text: str) -> tuple[str, StageResult]:
    """Clean and normalize input text.

    Applies Unicode normalization (NFKC) and strips control characters
    that could be used to hide malicious content.

    Returns:
        Tuple of (sanitized_text, stage_result).
    """
    if not text:
        return text, StageResult(
            stage="sanitize", verdict=Verdict.SAFE, reason="Empty input"
        )

    original_len = len(text)

    # Unicode NFKC normalization (standard, publicly documented)
    cleaned = unicodedata.normalize("NFKC", text)

    # Strip zero-width characters that hide content
    zero_width = {
        "\u200b",  # zero-width space
        "\u200c",  # zero-width non-joiner
        "\u200d",  # zero-width joiner
        "\u2060",  # word joiner
        "\ufeff",  # BOM / zero-width no-break space
    }
    suspicious_chars = sum(1 for c in cleaned if c in zero_width)
    cleaned = "".join(c for c in cleaned if c not in zero_width)

    # Strip ANSI escape sequences
    cleaned = re.sub(r"\x1b\[[0-9;]*[a-zA-Z]", "", cleaned)

    # Strip control characters (except newline, tab)
    cleaned = "".join(
        c
        for c in cleaned
        if c in ("\n", "\t", "\r") or not unicodedata.category(c).startswith("C")
    )

    verdict = Verdict.SAFE
    reason = "Input clean"

    if suspicious_chars > 3:
        verdict = Verdict.CAUTION
        reason = f"Stripped {suspicious_chars} zero-width characters"
    elif len(cleaned) < original_len * 0.8:
        verdict = Verdict.CAUTION
        reason = f"Significant content removed during sanitization ({original_len} → {len(cleaned)})"

    return cleaned, StageResult(
        stage="sanitize",
        verdict=verdict,
        reason=reason,
        details={"original_length": original_len, "cleaned_length": len(cleaned)},
    )


# ═══════════════════════════════════════════════════════════════════
# STAGE 2: THREAT CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════

# Threat categories based on OWASP Top 10 for LLM (2023)
THREAT_CHECKS = [
    {
        "id": "instruction_override",
        "category": "prompt_injection",
        "patterns": [
            re.compile(
                r"ignore\s+(all\s+)?(previous|prior|above)\s+(instructions|rules|guidelines)",
                re.I,
            ),
            re.compile(
                r"disregard\s+(your|the|all)\s+(policy|rules|instructions)", re.I
            ),
            re.compile(
                r"forget\s+(everything|all|what)\s+(you|i)\s+(told|said|know)", re.I
            ),
            re.compile(r"you\s+are\s+now\s+(a|an|my)\s+", re.I),
            re.compile(r"new\s+instructions?\s*[:=]", re.I),
            re.compile(r"system\s*:\s*you\s+are", re.I),
        ],
        "severity": "high",
    },
    {
        "id": "data_extraction",
        "category": "information_leakage",
        "patterns": [
            re.compile(
                r"(reveal|show|display|print|output)\s+(your|the|all)\s+(system|internal|hidden)\s+(prompt|instructions|rules)",
                re.I,
            ),
            re.compile(
                r"what\s+(are|is)\s+your\s+(system|internal|original)\s+(prompt|instructions|rules)",
                re.I,
            ),
            re.compile(
                r"repeat\s+(the|your)\s+(above|system|initial)\s+(text|prompt|message)",
                re.I,
            ),
        ],
        "severity": "high",
    },
    {
        "id": "role_hijack",
        "category": "prompt_injection",
        "patterns": [
            re.compile(r"pretend\s+(to\s+be|you\s+are)\s+", re.I),
            re.compile(r"act\s+as\s+(if|though)\s+you", re.I),
            re.compile(r"from\s+now\s+on\s+(you|your)\s+(are|role|job)", re.I),
            re.compile(
                r"switch\s+(to|into)\s+(a\s+)?(different|new)\s+(mode|role|persona)",
                re.I,
            ),
        ],
        "severity": "medium",
    },
    {
        "id": "encoding_evasion",
        "category": "obfuscation",
        "patterns": [
            re.compile(r"base64\s*(decode|encode|convert)", re.I),
            re.compile(r"\\x[0-9a-fA-F]{2}(\\x[0-9a-fA-F]{2}){3,}", re.I),
            re.compile(r"&#x?[0-9a-fA-F]+;(&#x?[0-9a-fA-F]+;){3,}", re.I),
            re.compile(r"%[0-9a-fA-F]{2}(%[0-9a-fA-F]{2}){5,}", re.I),
        ],
        "severity": "medium",
    },
    {
        "id": "tool_abuse",
        "category": "tool_misuse",
        "patterns": [
            re.compile(r"call\s+(all|every)\s+(tool|function|api)", re.I),
            re.compile(
                r"execute\s+(this|the\s+following)\s+(command|code|script)", re.I
            ),
            re.compile(r"run\s+(rm|del|drop|truncate|delete\s+from)", re.I),
        ],
        "severity": "high",
    },
    {
        "id": "social_engineering",
        "category": "manipulation",
        "patterns": [
            re.compile(
                r"(my|the)\s+(boss|manager|ceo|supervisor)\s+(told|said|wants|asked)\s+(me|you)\s+to",
                re.I,
            ),
            re.compile(
                r"(urgent|emergency|critical)\s*[!:]\s*(override|skip|bypass)", re.I
            ),
            re.compile(r"(i\s+will|gonna)\s+(sue|report|fire)\s+(you|this)", re.I),
        ],
        "severity": "low",
    },
    {
        "id": "account_impersonation",
        "category": "identity_fraud",
        "patterns": [
            re.compile(
                r"(access|modify|cancel|change)\s+(another|someone\s+else|my\s+friend|my\s+wife|my\s+husband)('?s?)\s+(account|reservation|order|plan|subscription)",
                re.I,
            ),
            re.compile(
                r"(i\s+am|i'm)\s+(calling|writing)\s+(on\s+behalf|for)\s+(another|someone|a\s+friend)",
                re.I,
            ),
        ],
        "severity": "high",
    },
    {
        "id": "bulk_action_abuse",
        "category": "tool_misuse",
        "patterns": [
            re.compile(
                r"(cancel|delete|modify|change|update)\s+(all|every|each)\s+(of\s+)?(my\s+)?(reservations?|orders?|accounts?|plans?|bookings?)\s+(at\s+once|simultaneously|in\s+one\s+go)",
                re.I,
            ),
            re.compile(r"(batch|bulk)\s+(cancel|delete|modify|update|process)", re.I),
        ],
        "severity": "medium",
    },
]


def classify_threats(text: str) -> StageResult:
    """Classify input text against known threat patterns.

    Checks input against OWASP-aligned threat categories.
    Returns the highest severity match found.
    """
    matches = []

    for check in THREAT_CHECKS:
        for pattern in check["patterns"]:
            if pattern.search(text):
                matches.append(
                    {
                        "id": check["id"],
                        "category": check["category"],
                        "severity": check["severity"],
                    }
                )
                break  # One match per check is enough

    if not matches:
        return StageResult(
            stage="classify", verdict=Verdict.SAFE, reason="No threats detected"
        )

    high_count = sum(1 for m in matches if m["severity"] == "high")
    medium_count = sum(1 for m in matches if m["severity"] == "medium")

    if high_count >= 2 or (high_count >= 1 and medium_count >= 2):
        verdict = Verdict.BLOCK
        reason = f"Multiple high-severity threats: {[m['id'] for m in matches]}"
    elif high_count >= 1:
        verdict = Verdict.CAUTION
        reason = f"High-severity threat: {matches[0]['id']}"
    elif medium_count >= 2:
        verdict = Verdict.CAUTION
        reason = f"Multiple medium threats: {[m['id'] for m in matches]}"
    else:
        verdict = Verdict.SAFE
        reason = f"Low-severity only: {[m['id'] for m in matches]}"

    return StageResult(
        stage="classify",
        verdict=verdict,
        reason=reason,
        details={"matches": matches, "total": len(matches)},
    )


# ═══════════════════════════════════════════════════════════════════
# STAGE 3: POLICY VERIFICATION
# ═══════════════════════════════════════════════════════════════════


@dataclass
class PolicyRule:
    """A single policy rule to verify against."""

    id: str
    description: str
    check: str  # "require" or "forbid"
    pattern: re.Pattern


# Default rules based on common customer service policies
DEFAULT_POLICY_RULES = [
    PolicyRule(
        "POL-001",
        "Agent must not promise specific outcomes",
        "forbid",
        re.compile(r"i\s+(guarantee|promise|assure)\s+(you|that)", re.I),
    ),
    PolicyRule(
        "POL-002",
        "Agent must not share other customer data",
        "forbid",
        re.compile(
            r"(another|other|different)\s+customer'?s?\s+(account|data|info)", re.I
        ),
    ),
    PolicyRule(
        "POL-003",
        "Agent must not bypass confirmation for write actions",
        "forbid",
        re.compile(
            r"(skip|bypass|ignore)\s+(the\s+)?(confirmation|approval|consent)", re.I
        ),
    ),
    PolicyRule(
        "POL-004",
        "Agent must not make medical/legal/financial diagnoses",
        "forbid",
        re.compile(
            r"(you\s+(have|suffer|got)\s+(a|from)\s+)|(my\s+diagnosis\s+is)", re.I
        ),
    ),
    PolicyRule(
        "POL-005",
        "Agent must not disclose internal system details",
        "forbid",
        re.compile(
            r"(our|the)\s+(internal|backend|database|system)\s+(uses|runs|has)", re.I
        ),
    ),
    PolicyRule(
        "POL-006",
        "Agent must not guarantee specific refund amounts or timelines",
        "forbid",
        re.compile(
            r"(you\s+will\s+(definitely|certainly)\s+(get|receive)\s+(a\s+)?(full\s+)?refund|refund\s+will\s+(be\s+)?processed?\s+within\s+\d+)",
            re.I,
        ),
    ),
    PolicyRule(
        "POL-007",
        "Agent must not provide definitive medical diagnoses",
        "forbid",
        re.compile(
            r"(you\s+(definitely|clearly)\s+have|this\s+is\s+(definitely|clearly)\s+a\s+case\s+of|my\s+diagnosis\s+is\s+that)",
            re.I,
        ),
    ),
]


def verify_policy(text: str, rules: Optional[list[PolicyRule]] = None) -> StageResult:
    """Verify agent output against policy rules.

    Checks if the agent's response violates any configured policy rules.
    """
    if rules is None:
        rules = DEFAULT_POLICY_RULES

    violations = []
    for rule in rules:
        if rule.check == "forbid" and rule.pattern.search(text):
            violations.append({"rule": rule.id, "description": rule.description})
        elif rule.check == "require" and not rule.pattern.search(text):
            violations.append({"rule": rule.id, "description": rule.description})

    if not violations:
        return StageResult(
            stage="verify_policy", verdict=Verdict.SAFE, reason="No policy violations"
        )

    verdict = Verdict.BLOCK if len(violations) >= 2 else Verdict.CAUTION
    return StageResult(
        stage="verify_policy",
        verdict=verdict,
        reason=f"{len(violations)} policy violation(s): {[v['rule'] for v in violations]}",
        details={"violations": violations},
    )


# ═══════════════════════════════════════════════════════════════════
# STAGE 4: OUTPUT VALIDATION
# ═══════════════════════════════════════════════════════════════════

# Sensitive data patterns to catch before sending to user
SENSITIVE_PATTERNS = [
    {
        "id": "api_key",
        "pattern": re.compile(r"(sk-|pk-|api[_-]?key)[a-zA-Z0-9]{20,}", re.I),
    },
    {
        "id": "email_bulk",
        "pattern": re.compile(
            r"[\w.+-]+@[\w-]+\.[\w.]+.*[\w.+-]+@[\w-]+\.[\w.]+", re.I
        ),
    },
    {"id": "ssn", "pattern": re.compile(r"\b\d{3}-\d{2}-\d{4}\b")},
    {
        "id": "credit_card",
        "pattern": re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"),
    },
    {
        "id": "internal_path",
        "pattern": re.compile(r"(/home/|/var/|/etc/|C:\\Users\\)[^\s]{10,}", re.I),
    },
    {
        "id": "connection_string",
        "pattern": re.compile(r"(postgres|mysql|mongodb)://[^\s]{10,}", re.I),
    },
    {
        "id": "raw_json_db_dump",
        "pattern": re.compile(
            r'"(user_id|customer_id|reservation_id|order_id|account_id)"\s*:\s*"[^"]{5,}".*"(password|token|secret|ssn|credit_card)',
            re.I,
        ),
    },
    {
        "id": "jwt_token",
        "pattern": re.compile(r"eyJ[a-zA-Z0-9_-]{20,}\.eyJ[a-zA-Z0-9_-]{20,}", re.I),
    },
]


def validate_output(text: str) -> StageResult:
    """Validate agent output for sensitive data leakage.

    Scans output for credentials, PII, and internal system information
    that should never be exposed to users.
    """
    found = []
    for check in SENSITIVE_PATTERNS:
        if check["pattern"].search(text):
            found.append(check["id"])

    if not found:
        return StageResult(
            stage="validate_output",
            verdict=Verdict.SAFE,
            reason="No sensitive data detected",
        )

    verdict = Verdict.BLOCK if len(found) >= 2 else Verdict.CAUTION
    return StageResult(
        stage="validate_output",
        verdict=verdict,
        reason=f"Sensitive data detected: {found}",
        details={"sensitive_types": found},
    )


# ═══════════════════════════════════════════════════════════════════
# STAGE 5: DECISION GATE
# ═══════════════════════════════════════════════════════════════════


def decide(stages: list[StageResult]) -> ShieldResult:
    """Aggregate stage results into a final verdict.

    Decision logic:
    - Any BLOCK → final BLOCK
    - 2+ CAUTION → final BLOCK
    - 1 CAUTION → final CAUTION
    - All SAFE → final SAFE
    """
    blocks = [s for s in stages if s.verdict == Verdict.BLOCK]
    cautions = [s for s in stages if s.verdict == Verdict.CAUTION]

    if blocks:
        return ShieldResult(
            final_verdict=Verdict.BLOCK,
            stages=stages,
            blocked=True,
            block_reason=blocks[0].reason,
        )

    if len(cautions) >= 2:
        reasons = [c.reason for c in cautions]
        return ShieldResult(
            final_verdict=Verdict.BLOCK,
            stages=stages,
            blocked=True,
            block_reason=f"Multiple cautions escalated: {reasons}",
        )

    if cautions:
        return ShieldResult(
            final_verdict=Verdict.CAUTION,
            stages=stages,
            blocked=False,
            block_reason="",
        )

    return ShieldResult(
        final_verdict=Verdict.SAFE,
        stages=stages,
        blocked=False,
        block_reason="",
    )


# ═══════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════


def shield_input(text: str) -> ShieldResult:
    """Run the full 5-stage defense pipeline on input text.

    Usage:
        result = shield_input(user_message)
        if result.blocked:
            return "I cannot process that request."
    """
    sanitized, sanitize_result = sanitize_input(text)
    classify_result = classify_threats(sanitized)
    # Policy verification on input (check if user is requesting policy violations)
    policy_result = verify_policy(sanitized)

    stages = [sanitize_result, classify_result, policy_result]
    return decide(stages)


def shield_output(
    text: str, policy_rules: Optional[list[PolicyRule]] = None
) -> ShieldResult:
    """Run defense pipeline on agent output before sending to user.

    Usage:
        result = shield_output(agent_response)
        if result.blocked:
            return regenerate_response()
    """
    policy_result = verify_policy(text, rules=policy_rules)
    output_result = validate_output(text)

    stages = [policy_result, output_result]
    return decide(stages)


def shield_conversation(
    user_input: str,
    agent_output: str,
    policy_rules: Optional[list[PolicyRule]] = None,
) -> dict:
    """Run full bidirectional defense on a conversation turn.

    Checks both the user input and the agent output.

    Returns:
        Dict with input_result, output_result, and overall_safe boolean.
    """
    input_result = shield_input(user_input)
    output_result = shield_output(agent_output, policy_rules=policy_rules)

    return {
        "input": input_result.summary(),
        "output": output_result.summary(),
        "overall_safe": input_result.is_safe and output_result.is_safe,
        "blocked": input_result.blocked or output_result.blocked,
    }
