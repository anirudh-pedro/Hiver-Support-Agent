"""
src/route.py

Hybrid escalation/routing policy for the Hiver-AmazonHelp support agent.

Architecture (three layers, evaluated in order):
  1. Deterministic rule layer  — safety-critical rules always win. Each rule
     has a stable name so triggered_rule is auditable in production logs.
  2. Low-confidence override   — if the intent classifier itself was unsure,
     we never let an ambiguous intent drive the routing decision.
  3. LLM judgment layer        — for cases that rules do not cover, ask the
     model to decide with a one-sentence reason.

Design rationale (from LABELING_GUIDE.md Section 5):
  - Financial actions, account security, and PII leakage require human
    intervention regardless of anything else the thread contains.
  - Escalation decisions must be auditable — that is why rules run first and
    always record triggered_rule so callers know exactly why.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants — tune these in Phase 7 for the precision/recall trade-off curve
# ─────────────────────────────────────────────────────────────────────────────

# If the intent classifier's confidence falls below this, route to human.
LOW_CONFIDENCE_THRESHOLD: float = 0.60

# Model used in the LLM judgment layer.
ROUTING_LLM_MODEL: str = "groq/compound"

# Minimum turn count that, combined with frustration phrases, triggers
# rule_repeated_failure.  (3 turns means at least 2 customer messages.)
REPEATED_FAILURE_MIN_TURNS: int = 3


# ─────────────────────────────────────────────────────────────────────────────
# Output dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RoutingDecision:
    decision: str               # "auto_handle" | "escalate"
    reason: str                 # human-readable explanation
    triggered_rule: Optional[str]  # rule name if a deterministic rule fired
    confidence: float           # 0.0–1.0; 1.0 for hard rules, model-reported for LLM
    source: str                 # "rule" | "low_confidence" | "llm" | "deferred_llm"
    model: str = "deterministic_rule"  # exact model string or rule identifier


# ─────────────────────────────────────────────────────────────────────────────
# Helper: extract full thread text as a single lowercase string for matching
# ─────────────────────────────────────────────────────────────────────────────

def _thread_text(thread: Dict[str, Any]) -> str:
    """
    Returns the full thread text as one lowercase string.
    Prefers the pre-built 'full_thread_text' field; falls back to
    reconstructing from 'turns'.
    """
    if thread.get("full_thread_text"):
        return str(thread["full_thread_text"]).lower()
    turns = thread.get("turns", [])
    return " ".join(t.get("text", "") for t in turns).lower()


def _customer_text(thread: Dict[str, Any]) -> str:
    """
    Returns only the customer-side text (lowercase), for rules that should
    not be triggered by AmazonHelp's own replies.
    """
    turns = thread.get("turns", [])
    if turns:
        customer_turns = [
            t.get("text", "")
            for t in turns
            if t.get("author_id", "") != "AmazonHelp"
        ]
        if customer_turns:
            return " ".join(customer_turns).lower()
    # Fall back: parse full_thread_text for [Customer]: lines
    full = str(thread.get("full_thread_text", ""))
    customer_lines = [
        line.replace("[Customer]:", "").strip()
        for line in full.splitlines()
        if line.startswith("[Customer]:")
    ]
    return " ".join(customer_lines).lower()


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 1 — Deterministic rule functions
# Each returns (escalate: bool, reason: str) or None if the rule does not fire.
# ─────────────────────────────────────────────────────────────────────────────

def rule_security(intent: str, thread: Dict[str, Any]):
    """
    rule_security: account_access threads where the thread text indicates an
    active compromise, lockout, or unauthorised change.

    Fires on: hacked, unauthorized change, locked out, account on hold,
    account suspended, orders being placed, someone else logged in.
    Does NOT fire on: routine password reset help or 2FA setup questions
    where there is no signal of active compromise.
    """
    if intent != "account_access":
        return None

    text = _thread_text(thread)
    compromise_patterns = [
        r"\bhack(ed|er|ing)?\b",
        r"\bunauthori[sz]ed\b",
        r"\blocked\s*out\b",
        r"\baccount\s*(on\s*hold|suspended|frozen|compromised)\b",
        r"\bsomeone\s*(else|other)\s*(has\s*)?(logged|signed|got)\s*in\b",
        r"\border[s]?\s*(being|were|was|are)\s*placed\b",
        r"\bpassword\s*changed\b",
        r"\bemail\s*(address\s*)?(was\s*|has\s+been\s*)?changed\b",
        r"\bphone\s*(number\s*)?(was\s*|has\s+been\s*)?changed\b",
    ]
    for pat in compromise_patterns:
        if re.search(pat, text):
            return (
                True,
                "Active account compromise or lockout detected — requires urgent "
                "security team verification and account freeze.",
            )
    return None


def rule_financial_action(intent: str, thread: Dict[str, Any]):
    """
    rule_financial_action: billing_dispute threads where the customer is
    requesting a concrete monetary action (refund, charge reversal, missing refund status,
    unauthorised charge, payment dispute), NOT merely asking how billing/refunds work in general.

    Fires on:  refund me / give me a refund / charged me / unauthorised charge /
               reversal / dispute this charge / i want my money back / missing refund /
               not reflected in bank / return my money.
    Does NOT fire on: pure general policy inquiries ("how do refunds work?").
    """
    if intent != "billing_dispute":
        return None

    text = _customer_text(thread)

    # Negative signal: pure hypothetical policy inquiry without a personal claim
    policy_only_patterns = [
        r"\bhow\s+(do(es)?|can|long|to)\b.*\brefund\b",
        r"\brefund\s*policy\b",
        r"\bwhat\s+is\s+(the\s+|your\s+)?refund\b",
        r"\beligible\s+for\s+a?\s*refund\b",
    ]
    has_personal_claim = any(k in text for k in [
        "refund me", "my refund", "my money", "charged me", "my card", "my account",
        "returned the", "i returned", "not received", "haven't received", "not rcvd", "still no"
    ])
    if any(re.search(p, text) for p in policy_only_patterns) and not has_personal_claim:
        return None

    # Positive action signals
    action_patterns = [
        r"\brefund\s*(me|my|this|it|the|now|asap|not|pending|amount|status|success)?\b",
        r"\bi\s*(need|want|require|expect|demand|await|awaiting)\s+a?\s*refund\b",
        r"\bwhen\s+will\s+(u|you|i)\s+(get\s+a\s+|receive\s+a\s+)?refund\b",
        r"\b(return|give|get)\s+(back\s+)?my\s+money\b",
        r"\bmoney\s*back\b",
        r"\b(reimburse|reimbursed|reimbursement)\b",
        r"\bunauthori[sz]ed\s+(charge|payment|transaction|debit|purchase)\b",
        r"\b(charged|billed|deducted|debited)\s+(me|my|without|twice|double|extra|wrong|incorrectly|first|again)\b",
        r"\bdouble\s*charg(e|ed|ing)\b",
        r"\bcharge\s*reversal\b",
        r"\bdispute\s*(this\s*)?(charge|payment|transaction)\b",
        r"\b(took|taken|stole)\s+(my\s+|the\s+)?money\b",
        r"\b(not\s+reflected|hasn'?t\s+reflected)\s+in\s+(my\s+)?(bank|account|card)\b",
        r"\b(refund\s+has\s+not\s+been|not\s+initiated)\b",
    ]

    for pat in action_patterns:
        if re.search(pat, text):
            return (
                True,
                "Customer is requesting a monetary action (refund, charge reversal, missing refund status, or "
                "payment dispute) — requires human billing specialist with payment system access.",
            )

    return None


def rule_pii_exposure(intent: str, thread: Dict[str, Any]):
    """
    rule_pii_exposure: customer has posted PII (order number, card digits,
    phone number, full address) in a public tweet.

    This is both a privacy risk and requires human scrubbing of the post.
    Fires regardless of intent.
    """
    text = _customer_text(thread)

    pii_patterns = [
        # Order number: 3-digit + 7-digit + 7-digit (Amazon format) or 10-17 digit strings
        (r"\b\d{3}-\d{7}-\d{7}\b", "Amazon order number"),
        # Card digits: 4+ digit groups, especially last 4 after card/ending
        (r"\b(card|ending|debit|credit)\s*[:\-]?\s*\d{4}\b", "card digits"),
        # Standalone 16-digit card number
        (r"\b(?:\d{4}[\s\-]?){3}\d{4}\b", "card number"),
        # Phone numbers: US/international formats
        (r"\b(\+\d{1,3}[\s\-]?)?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}\b", "phone number"),
        # Street address: number + 1-3 street name words + street type abbreviation
        (r"\b\d{1,5}\s+(?:[A-Za-z0-9\.\-]+\s+){1,3}(?:st|street|ave|avenue|rd|road|blvd|boulevard|ln|lane|dr|drive|ct|court|way|pkwy|parkway)\b",
         "street address"),
    ]

    for pattern, pii_type in pii_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return (
                True,
                f"Customer appears to have shared {pii_type} in a public tweet — "
                "requires human agent to scrub PII and continue via secure DM.",
            )
    return None


def rule_repeated_failure(intent: str, thread: Dict[str, Any]):
    """
    rule_repeated_failure: thread shows clear evidence of prior failed contacts,
    broken callback promises, or multiple failed delivery/return attempts.

    Detection heuristic:
      - Thread has >= REPEATED_FAILURE_MIN_TURNS AND
      - Customer text contains frustration phrases indicating this is not the
        first contact / a promise was broken.
    """
    # Count turns
    n_turns = thread.get("n_turns", None)
    if n_turns is None:
        turns = thread.get("turns", [])
        if turns:
            n_turns = len(turns)
        else:
            # Parse from full_thread_text
            full = str(thread.get("full_thread_text", ""))
            n_turns = full.count("[Customer]:") + full.count("[AmazonHelp]:")

    if int(n_turns) < REPEATED_FAILURE_MIN_TURNS:
        return None

    text = _customer_text(thread)
    frustration_patterns = [
        r"\bthird\s*(time|attempt|try)\b",
        r"\bfourth\s*(time|attempt|try)\b",
        r"\bfifth\s*(time|attempt|try)\b",
        r"\bmultiple\s*(time|attempt|contact|call|agent)s?\b",
        r"\bstill\s+no\b",
        r"\bstill\s+(waiting|haven|not|haven'?t)\b",
        r"\b(promised|told)\s+(me\s+)?(it\s+)?(would|will|should)\b",
        r"\bpromise\s+was\s+(broken|not\s+kept)\b",
        r"\bkeep\s+(calling|contacting|trying|reaching)\b",
        r"\b(no\s+one\s+called|never\s+called|never\s+got)\b",
        r"\bagain\s+and\s+again\b",
        r"\bfor\s+the\s+(second|third|fourth|nth|last|past\s+\d+)\s+(time|week|day)\b",
        r"\bprevious\s+(agent|rep|support|contact|case)\b",
        r"\bescalat(e|ed|ing)\b",
    ]
    for pat in frustration_patterns:
        if re.search(pat, text):
            return (
                True,
                "Thread shows evidence of repeated prior failed contacts or broken "
                "agent promises — requires human specialist to break the cycle.",
            )
    return None


def rule_legal_or_churn_threat(intent: str, thread: Dict[str, Any]):
    """
    rule_legal_or_churn_threat: customer makes explicit legal threats or
    announces intent to close long-standing Prime membership after failures.

    Fires on: consumer court, legal action, sue, complaint to regulator,
              cancel my prime, closing my account (after negative context).
    """
    text = _customer_text(thread)

    legal_patterns = [
        r"\bsue\b",
        r"\blawsuit\b",
        r"\blegal\s+action\b",
        r"\bconsumer\s+(court|forum|protection)\b",
        r"\b(report|complaint)\s+to\s+(bbb|ftc|cfpb|trading\s+standards|regulator)\b",
        r"\battorney\b",
        r"\bsolicit?or\b",
        r"\bclass\s+action\b",
        r"\bsmall\s+claims\b",
        r"\bchargeback\b",      # financial chargeback = legal/financial escalation
    ]
    churn_patterns = [
        r"\bcancel(l?ing|l?ed)?\s+(my\s+)?(prime|membership|subscription|account)\b",
        r"\bclos(e|ing)\s+(my\s+)?(prime|account)\b",
        r"\bdelete\s+(my\s+)?account\b",
        r"\bleaving\s+amazon\b",
        r"\bswitch(ing)?\s+to\s+(walmart|ebay|target|etsy)\b",
    ]

    for pat in legal_patterns:
        if re.search(pat, text):
            return (
                True,
                "Customer has made an explicit legal threat — requires immediate "
                "senior specialist review to avoid liability escalation.",
            )
    for pat in churn_patterns:
        if re.search(pat, text):
            return (
                True,
                "Customer is threatening to cancel their Prime membership or close "
                "their account — requires human retention specialist intervention.",
            )
    return None


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 2 — Low-confidence override
# ─────────────────────────────────────────────────────────────────────────────

def _low_confidence_override(intent_confidence: float) -> Optional[RoutingDecision]:
    """
    If the intent classifier itself was uncertain, we cannot trust any
    intent-conditioned routing logic downstream.  Route to human to be safe.
    """
    if intent_confidence < LOW_CONFIDENCE_THRESHOLD:
        return RoutingDecision(
            decision="escalate",
            reason=(
                f"Intent confidence ({intent_confidence:.2f}) is below the routing "
                f"threshold ({LOW_CONFIDENCE_THRESHOLD}) — routing to human to avoid "
                "mishandling an ambiguous case."
            ),
            triggered_rule="rule_low_confidence",
            confidence=1.0,  # the override itself is certain
            source="low_confidence",
            model="rule_low_confidence",
        )
    return None


# ─────────────────────────────────────────────────────────────────────────────
# LAYER 3 — LLM judgment layer
# ─────────────────────────────────────────────────────────────────────────────

_ROUTING_SYSTEM_PROMPT = """\
You are a routing specialist for an Amazon customer support AI agent.
Your task: decide whether a support thread should be handled autonomously
by the AI ("auto_handle") or escalated to a human specialist ("escalate").

ESCALATE when the thread involves:
- Financial actions: refund, charge reversal, investigating unauthorized charges.
- Security/Privacy: account takeover, hacked credentials, unauthorized changes, PII shared publicly.
- Carrier escalation: high-value missing package marked delivered, driver misconduct.
- Physical defect replacement: damaged expensive/hazardous items needing immediate replacement.
- High agitation / churn risk: legal threats, social media escalation, or closing Prime after multiple failures.

AUTO_HANDLE when the thread involves:
- Factual questions about policies, specs, release dates, compatibility.
- Tracking/return guidance using standard self-service tools.
- Phishing confirmation (is this Amazon email real?).
- Positive feedback / non-support chatter.
- Directing to standard account recovery pages when no active compromise is evident.

Respond ONLY with valid JSON, no markdown, no commentary:
{"decision": "auto_handle" | "escalate", "reason": "<one sentence>", "confidence": <0.0-1.0>}
"""

def _build_routing_prompt(thread: Dict[str, Any], intent: str, confidence: float) -> str:
    full_text = str(thread.get("full_thread_text", "")).strip()
    return (
        f"Predicted intent: {intent} (classifier confidence: {confidence:.2f})\n\n"
        f"Full conversation thread:\n{full_text}\n\n"
        "Based on the thread and intent above, output your routing decision as JSON."
    )


def _llm_route(
    thread: Dict[str, Any],
    intent: str,
    intent_confidence: float,
    client: Any,
    model: str = ROUTING_LLM_MODEL,
    max_retries: int = 5,
) -> RoutingDecision:
    """
    Ask the LLM to decide routing for a single thread.
    Returns a RoutingDecision with source="llm" and exact model string.
    """
    prompt = _build_routing_prompt(thread, intent, intent_confidence)
    raw = ""
    candidate_models = [model, "groq/compound", "qwen/qwen3.8-27b"]
    # Deduplicate candidate models while preserving order
    seen_models = set()
    models_to_try = []
    for m in candidate_models:
        if m and m not in seen_models:
            seen_models.add(m)
            models_to_try.append(m)

    for attempt in range(max_retries):
        cur_model = models_to_try[attempt % len(models_to_try)]
        try:
            resp = client.chat.completions.create(
                model=cur_model,
                messages=[
                    {"role": "system", "content": _ROUTING_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=120,
                temperature=0.0,
            )
            raw = (resp.choices[0].message.content or "").strip()
            # Extract JSON — handle LLMs that wrap in markdown
            match = re.search(r"\{.*?\}", raw, re.DOTALL)
            if not match:
                raise ValueError(f"No JSON found in: {raw!r}")
            parsed = json.loads(match.group())
            decision = parsed.get("decision", "escalate")
            if decision not in ("auto_handle", "escalate"):
                decision = "escalate"
            reason = str(parsed.get("reason", "LLM routing decision."))
            conf = float(parsed.get("confidence", 0.7))
            return RoutingDecision(
                decision=decision,
                reason=reason,
                triggered_rule=None,
                confidence=conf,
                source="llm",
                model=cur_model,
            )
        except Exception as exc:
            wait = 4.0 * (attempt + 1)
            logger.warning(
                "LLM routing attempt %d/%d (model=%s) failed: %s — retrying in %.0fs",
                attempt + 1, max_retries, cur_model, exc, wait,
            )
            time.sleep(wait)

    # Safe fallback: if LLM fails entirely, escalate
    logger.error("LLM routing failed after %d attempts; defaulting to escalate.", max_retries)
    return RoutingDecision(
        decision="escalate",
        reason="LLM routing layer failed after all retries — defaulting to escalate for safety.",
        triggered_rule="rule_llm_failure_fallback",
        confidence=0.0,
        source="llm",
        model="llm_failure_fallback",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API: route_thread
# ─────────────────────────────────────────────────────────────────────────────

# Ordered list of deterministic rules; evaluated left-to-right, first match wins.
_RULES = [
    ("rule_security",            rule_security),
    ("rule_financial_action",    rule_financial_action),
    ("rule_pii_exposure",        rule_pii_exposure),
    ("rule_repeated_failure",    rule_repeated_failure),
    ("rule_legal_or_churn_threat", rule_legal_or_churn_threat),
]


def route_thread(
    thread: Dict[str, Any],
    intent: str,
    intent_confidence: float,
    client: Any,
    model: str = ROUTING_LLM_MODEL,
    low_confidence_threshold: float = LOW_CONFIDENCE_THRESHOLD,
    rules_only: bool = False,
) -> RoutingDecision:
    """
    Main routing entry point.  Applies layers in order:
      1. Deterministic rules  (short-circuit on first match, source="rule")
      2. Low-confidence override  (source="low_confidence")
      3. LLM judgment  (source="llm"), unless rules_only=True

    Args:
        thread: thread dict with 'full_thread_text', 'turns', 'n_turns'.
        intent: predicted intent string from classify_intent().
        intent_confidence: confidence score from classify_intent().
        client: OpenAI-compatible client for the LLM layer.
        model: model ID for the LLM layer.
        low_confidence_threshold: override threshold (tunable for Phase 7).
        rules_only: if True, skip the LLM layer and return source="deferred_llm"
            for threads that do not match any deterministic rule. Use this when
            the LLM API quota is exhausted; re-run with rules_only=False to
            backfill those decisions later.

    Returns:
        RoutingDecision dataclass.
    """
    # ── Layer 1: deterministic rules ─────────────────────────────────────────
    for rule_name, rule_fn in _RULES:
        result = rule_fn(intent, thread)
        if result is not None:
            escalate_flag, reason = result
            decision = "escalate" if escalate_flag else "auto_handle"
            logger.debug("Rule %s fired: %s", rule_name, decision)
            return RoutingDecision(
                decision=decision,
                reason=reason,
                triggered_rule=rule_name,
                confidence=1.0,
                source="rule",
                model=rule_name,
            )

    # ── Layer 2: low-confidence override ─────────────────────────────────────
    override = _low_confidence_override(intent_confidence)
    if override is not None:
        logger.debug("Low-confidence override fired (conf=%.2f)", intent_confidence)
        return override

    # ── Layer 3: LLM judgment (skipped when rules_only=True) ─────────────────
    if rules_only:
        logger.debug("rules_only=True — deferring LLM decision for this thread.")
        return RoutingDecision(
            decision="auto_handle",  # conservative default: no rule fired, likely safe
            reason=(
                "No deterministic rule matched and intent confidence is sufficient; "
                "LLM judgment deferred pending API quota reset."
            ),
            triggered_rule=None,
            confidence=intent_confidence,
            source="deferred_llm",
            model="deferred_llm",
        )

    logger.debug("No rule fired — delegating to LLM layer.")
    return _llm_route(thread, intent, intent_confidence, client, model)
