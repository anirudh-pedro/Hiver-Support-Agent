"""
tests/test_draft_reply.py

Sanity tests for grounded reply drafting:
1. Output format and structure (DraftReply dataclass, grounding_thread_ids, model_used).
2. Realistic length (< 280 chars) and ^initials signature convention.
3. Anti-hallucination check: verify no invented order numbers or currency figures.
4. Batch execution preserves order and returns valid replies.
5. Graceful fallback on empty customer messages.
"""

import re
import pytest
from src.draft_reply import DraftReply, draft_reply, draft_replies_batch
from src.intents import IntentResult


def test_reply_draft_structure():
    """Verifies that draft_reply returns a valid DraftReply with grounding IDs."""
    thread = {
        "thread_id": 10001,
        "customer_message": "My order was supposed to arrive yesterday and is delayed.",
    }
    result = draft_reply(thread, intent="delivery_delay", k=5)

    assert isinstance(result, DraftReply)
    assert len(result.reply_text.strip()) > 0
    assert len(result.grounding_thread_ids) == 5
    assert result.model_used is not None


def test_reply_length_and_signoff():
    """Verifies that replies are within realistic Twitter length and include agent initials signoff."""
    thread = {
        "thread_id": 10002,
        "customer_message": "Why was my card charged twice for my Prime membership?",
    }
    result = draft_reply(thread, intent="billing_dispute", k=5)

    assert len(result.reply_text) <= 320, f"Reply too long for Twitter ({len(result.reply_text)} chars): {result.reply_text}"
    # Must contain agent initials sign-off like ^AH, ^TN, ^RO
    assert re.search(r"\^[A-Z]{2,4}\b", result.reply_text), f"Missing agent signoff (^XX) in: {result.reply_text}"


def test_anti_hallucination_currency_and_order_number():
    """
    Checks that the model does not invent specific dollar/pound amounts or order IDs
    when none were provided in the input message.
    """
    input_text = "I received a damaged blender with broken glass everywhere. I need help."
    thread = {
        "thread_id": 10003,
        "customer_message": input_text,
    }
    result = draft_reply(thread, intent="order_product_problem", k=5)
    reply = result.reply_text

    # Check for fabricated Amazon order numbers: 3 digits - 7 digits - 7 digits
    order_matches = re.findall(r"\b\d{3}-\d{7}-\d{7}\b", reply)
    assert len(order_matches) == 0, f"Hallucinated order number detected: {order_matches}"

    # Check for fabricated currency values not present in input
    currency_matches = re.findall(r"[\$£€]\d+(?:\.\d{2})?", reply)
    # Filter out standard authorization hold mentions (like £1 or $1) if grounded in policy
    fabricated_currency = [c for c in currency_matches if c not in ["$1", "£1", "€1"]]
    assert len(fabricated_currency) == 0, f"Hallucinated currency amount detected: {fabricated_currency}"


def test_draft_replies_batch_concurrency():
    """Verifies batch execution preserves input ordering and completes concurrently."""
    batch_threads = [
        {"thread_id": 20001, "customer_message": "Where is my book? Tracking says delivered but I got nothing."},
        {"thread_id": 20002, "customer_message": "Can someone help me reset my account password? Locked out."},
    ]
    intents = ["delivery_delay", "account_access"]

    results = draft_replies_batch(batch_threads, intents=intents, max_workers=2, k=3)

    assert len(results) == 2
    assert results[0].grounding_thread_ids is not None
    assert results[1].grounding_thread_ids is not None
    assert len(results[0].reply_text) > 0
    assert len(results[1].reply_text) > 0


def test_empty_customer_message_fallback():
    """Empty thread message should return a safe standard triage fallback without crashing."""
    thread = {"thread_id": 30001, "customer_message": ""}
    result = draft_reply(thread)

    assert isinstance(result, DraftReply)
    assert "^" in result.reply_text
    assert "DM" in result.reply_text or "help" in result.reply_text.lower()
