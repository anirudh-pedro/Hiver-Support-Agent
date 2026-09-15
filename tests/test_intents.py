"""
test_intents.py - Regression and unit tests for intent classification.

Covers:
- At least one clear example per intent category across all 8 draft intents.
- One deliberately ambiguous case.
- Single classification and batch classification.
- Inclusion of brand reply context.
"""

import os
import sys
import pytest

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.intents import (
    IntentResult,
    classify_intent,
    classify_intents_batch,
    DEFAULT_MODEL,
)
from src.prompts import VALID_INTENTS

# 9 Test cases: 8 clear categories + 1 deliberately ambiguous case
TEST_THREADS = [
    {
        "id": "test_delivery_delay",
        "expected_intent": "delivery_delay",
        "thread": {
            "thread_id": 101,
            "turns": [
                {
                    "tweet_id": 101,
                    "author_id": "cust_1",
                    "inbound": True,
                    "created_at": "Wed Nov 01 10:00:00 +0000 2017",
                    "text": "@AmazonHelp where is my package? The tracking says delivered but it has not arrived!",
                },
                {
                    "tweet_id": 102,
                    "author_id": "AmazonHelp",
                    "inbound": False,
                    "created_at": "Wed Nov 01 10:05:00 +0000 2017",
                    "text": "We are sorry to hear that! Please DM us your tracking number.",
                },
            ],
            "n_turns": 2,
            "last_turn_author": "AmazonHelp",
        },
    },
    {
        "id": "test_billing_dispute",
        "expected_intent": "billing_dispute",
        "thread": {
            "thread_id": 102,
            "turns": [
                {
                    "tweet_id": 201,
                    "author_id": "cust_2",
                    "inbound": True,
                    "created_at": "Wed Nov 01 11:00:00 +0000 2017",
                    "text": "@AmazonHelp I was charged twice for Prime membership this month and haven't received my refund.",
                }
            ],
            "n_turns": 1,
            "last_turn_author": "cust_2",
        },
    },
    {
        "id": "test_order_product_problem",
        "expected_intent": "order_product_problem",
        "thread": {
            "thread_id": 103,
            "turns": [
                {
                    "tweet_id": 301,
                    "author_id": "cust_3",
                    "inbound": True,
                    "created_at": "Wed Nov 01 12:00:00 +0000 2017",
                    "text": "@AmazonHelp my package arrived today but the coffee maker is broken and damaged.",
                }
            ],
            "n_turns": 1,
            "last_turn_author": "cust_3",
        },
    },
    {
        "id": "test_account_access",
        "expected_intent": "account_access",
        "thread": {
            "thread_id": 104,
            "turns": [
                {
                    "tweet_id": 401,
                    "author_id": "cust_4",
                    "inbound": True,
                    "created_at": "Wed Nov 01 13:00:00 +0000 2017",
                    "text": "@AmazonHelp I cannot sign in to my account, it says locked out and I am not getting the 2FA OTP password reset email.",
                }
            ],
            "n_turns": 1,
            "last_turn_author": "cust_4",
        },
    },
    {
        "id": "test_product_info_question",
        "expected_intent": "product_info_question",
        "thread": {
            "thread_id": 105,
            "turns": [
                {
                    "tweet_id": 501,
                    "author_id": "cust_5",
                    "inbound": True,
                    "created_at": "Wed Nov 01 14:00:00 +0000 2017",
                    "text": "@AmazonHelp is the new Echo Show compatible with 5GHz Wi-Fi networks?",
                }
            ],
            "n_turns": 1,
            "last_turn_author": "cust_5",
        },
    },
    {
        "id": "test_service_complaint_vague",
        "expected_intent": "service_complaint_vague",
        "thread": {
            "thread_id": 106,
            "turns": [
                {
                    "tweet_id": 601,
                    "author_id": "cust_6",
                    "inbound": True,
                    "created_at": "Wed Nov 01 15:00:00 +0000 2017",
                    "text": "@AmazonHelp awful customer service experience today. Rude representative hung up on me. Terrible support!",
                }
            ],
            "n_turns": 1,
            "last_turn_author": "cust_6",
        },
    },
    {
        "id": "test_scam_phishing_check",
        "expected_intent": "scam_phishing_check",
        "thread": {
            "thread_id": 107,
            "turns": [
                {
                    "tweet_id": 701,
                    "author_id": "cust_7",
                    "inbound": True,
                    "created_at": "Wed Nov 01 16:00:00 +0000 2017",
                    "text": "@AmazonHelp I received an email saying my account was suspended and asking me to click a link. Is this legit or a phishing scam?",
                }
            ],
            "n_turns": 1,
            "last_turn_author": "cust_7",
        },
    },
    {
        "id": "test_non_support",
        "expected_intent": "non_support",
        "thread": {
            "thread_id": 108,
            "turns": [
                {
                    "tweet_id": 801,
                    "author_id": "cust_8",
                    "inbound": True,
                    "created_at": "Wed Nov 01 17:00:00 +0000 2017",
                    "text": "@AmazonHelp thank you guys for the great service and quick response!",
                }
            ],
            "n_turns": 1,
            "last_turn_author": "cust_8",
        },
    },
    {
        # Deliberately ambiguous case: mentions both service frustration and a late shipment
        "id": "test_ambiguous_case",
        "expected_intent": "delivery_delay",
        "thread": {
            "thread_id": 109,
            "turns": [
                {
                    "tweet_id": 901,
                    "author_id": "cust_9",
                    "inbound": True,
                    "created_at": "Wed Nov 01 18:00:00 +0000 2017",
                    "text": "@AmazonHelp worst service ever! My order has been delayed for 5 days and nobody is helping me.",
                }
            ],
            "n_turns": 1,
            "last_turn_author": "cust_9",
        },
    },
]


@pytest.mark.parametrize("case", TEST_THREADS)
def test_classify_intent_individual(case):
    """Verifies that each category produces a valid intent and non-empty reasoning."""
    thread = case["thread"]
    result = classify_intent(thread)

    assert isinstance(result, IntentResult)
    assert result.intent in VALID_INTENTS
    assert 0.0 <= result.confidence <= 1.0
    assert len(result.reasoning) > 0

    # Verify matching intent
    assert result.intent == case["expected_intent"]


def test_classify_intent_with_brand_reply():
    """Verifies context toggle with brand reply included."""
    thread = TEST_THREADS[0]["thread"]
    res_without = classify_intent(thread, include_brand_reply=False)
    res_with = classify_intent(thread, include_brand_reply=True)

    assert res_without.intent in VALID_INTENTS
    assert res_with.intent in VALID_INTENTS
    assert res_with.intent == "delivery_delay"


def test_classify_intents_batch(monkeypatch):
    """Verifies batch execution preserves input ordering and thread count with mock client."""
    from unittest.mock import MagicMock
    import json

    mock_client = MagicMock()
    # Create responses matching each expected case
    def mock_create(*args, **kwargs):
        messages = kwargs.get("messages", [])
        prompt_content = messages[1]["content"] if len(messages) > 1 else ""
        
        # Match test thread contents to expected intent
        matched_intent = "delivery_delay"
        for case in TEST_THREADS:
            txt = case["thread"]["turns"][0]["text"]
            if txt in prompt_content:
                matched_intent = case["expected_intent"]
                break
        
        mock_resp = MagicMock()
        mock_resp.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps({
                        "assigned_intent": matched_intent,
                        "confidence": 0.95,
                        "justification": "Mocked test reasoning",
                    })
                )
            )
        ]
        return mock_resp

    mock_client.chat.completions.create.side_effect = mock_create

    threads = [case["thread"] for case in TEST_THREADS]
    results = classify_intents_batch(threads, max_workers=2, client=mock_client)

    assert len(results) == len(threads)
    for res, case in zip(results, TEST_THREADS):
        assert res.intent in VALID_INTENTS
        assert res.intent == case["expected_intent"]
        assert 0.0 <= res.confidence <= 1.0


def test_empty_thread():
    """Verifies handling of pathological empty thread."""
    empty_thread = {"thread_id": 999, "turns": []}
    res = classify_intent(empty_thread)
    assert res.intent == "other"
    assert res.confidence == 0.0

