"""
tests/test_route.py

Unit tests for the deterministic rule layer in src/route.py.
Each rule has:
  - A positive test: clear-trigger input -> rule fires (escalate).
  - A near-miss negative test: similar-but-not-matching input -> rule does NOT fire.

Tests use pytest.  Run with:
    pytest tests/test_route.py -v
"""

import pytest
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.route import (
    rule_financial_action,
    rule_legal_or_churn_threat,
    rule_pii_exposure,
    rule_repeated_failure,
    rule_security,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_thread(text: str, n_turns: int = 1) -> dict:
    """Minimal thread dict for testing the rule functions."""
    return {
        "full_thread_text": text,
        "turns": [],
        "n_turns": n_turns,
    }


# ─────────────────────────────────────────────────────────────────────────────
# rule_security
# ─────────────────────────────────────────────────────────────────────────────

class TestRuleSecurity:

    def test_fires_on_active_compromise(self):
        thread = make_thread(
            "[Customer]: @AmazonHelp My account was hacked! Someone changed my email and "
            "is placing orders right now. I'm locked out completely."
        )
        result = rule_security("account_access", thread)
        assert result is not None
        escalate, reason = result
        assert escalate is True
        assert "security" in reason.lower() or "compromise" in reason.lower()

    def test_fires_on_unauthorized_change(self):
        thread = make_thread(
            "[Customer]: My phone number was changed on my Amazon account without my permission."
        )
        result = rule_security("account_access", thread)
        assert result is not None
        escalate, _ = result
        assert escalate is True

    def test_does_not_fire_on_wrong_intent(self):
        """Same text but different intent — rule must only fire on account_access."""
        thread = make_thread(
            "[Customer]: My account was hacked and someone placed orders!"
        )
        result = rule_security("billing_dispute", thread)
        assert result is None

    def test_does_not_fire_on_routine_password_reset(self):
        """Simple password reset request with no compromise signal."""
        thread = make_thread(
            "[Customer]: @AmazonHelp I forgot my password. How do I reset it?"
        )
        result = rule_security("account_access", thread)
        assert result is None

    def test_does_not_fire_on_2fa_setup_question(self):
        thread = make_thread(
            "[Customer]: How do I enable two-factor authentication on my account?"
        )
        result = rule_security("account_access", thread)
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# rule_financial_action
# ─────────────────────────────────────────────────────────────────────────────

class TestRuleFinancialAction:

    def test_fires_on_explicit_refund_request(self):
        thread = make_thread(
            "[Customer]: @AmazonHelp You charged my card $99 for Prime even though I "
            "cancelled. Refund me now."
        )
        result = rule_financial_action("billing_dispute", thread)
        assert result is not None
        escalate, reason = result
        assert escalate is True
        assert "monetary" in reason.lower() or "refund" in reason.lower()

    def test_fires_on_unauthorized_charge(self):
        thread = make_thread(
            "[Customer]: There is an unauthorized charge on my account — I never made this purchase."
        )
        result = rule_financial_action("billing_dispute", thread)
        assert result is not None

    def test_fires_on_money_back(self):
        thread = make_thread(
            "[Customer]: I want my money back. I returned the item 2 weeks ago and still nothing."
        )
        result = rule_financial_action("billing_dispute", thread)
        assert result is not None

    def test_fires_on_double_charge(self):
        thread = make_thread(
            "[Customer]: You double charged me this month. This is unacceptable!"
        )
        result = rule_financial_action("billing_dispute", thread)
        assert result is not None

    def test_does_not_fire_on_policy_question(self):
        """Asking how refunds work is NOT a financial action request."""
        thread = make_thread(
            "[Customer]: How do refunds work for digital purchases on Amazon?"
        )
        result = rule_financial_action("billing_dispute", thread)
        assert result is None

    def test_does_not_fire_on_eligibility_question(self):
        thread = make_thread(
            "[Customer]: Am I eligible for a refund on a gift item?"
        )
        result = rule_financial_action("billing_dispute", thread)
        assert result is None

    def test_does_not_fire_on_wrong_intent(self):
        thread = make_thread(
            "[Customer]: Refund me now! I want my money back."
        )
        result = rule_financial_action("delivery_delay", thread)
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# rule_pii_exposure
# ─────────────────────────────────────────────────────────────────────────────

class TestRulePIIExposure:

    def test_fires_on_amazon_order_number(self):
        thread = make_thread(
            "[Customer]: @AmazonHelp My order 114-7890123-4567890 has not arrived!"
        )
        result = rule_pii_exposure("delivery_delay", thread)
        assert result is not None
        escalate, reason = result
        assert escalate is True
        assert "order" in reason.lower() or "pii" in reason.lower()

    def test_fires_on_phone_number(self):
        thread = make_thread(
            "[Customer]: @AmazonHelp Please call me at 555-867-5309 to resolve this."
        )
        result = rule_pii_exposure("service_complaint_vague", thread)
        assert result is not None

    def test_fires_on_street_address(self):
        thread = make_thread(
            "[Customer]: My package was delivered to 123 Maple Street but I live at "
            "456 Oak Avenue."
        )
        result = rule_pii_exposure("delivery_delay", thread)
        assert result is not None

    def test_fires_on_card_last_four_with_keyword(self):
        thread = make_thread(
            "[Customer]: The charge was made to my card ending 4321 and I didn't authorize it."
        )
        result = rule_pii_exposure("billing_dispute", thread)
        assert result is not None

    def test_does_not_fire_on_clean_complaint(self):
        """A complaint with no PII should not trigger the rule."""
        thread = make_thread(
            "[Customer]: @AmazonHelp My package is late and I am very frustrated!"
        )
        result = rule_pii_exposure("delivery_delay", thread)
        assert result is None

    def test_does_not_fire_on_bare_numbers(self):
        """Random short numbers should not be treated as order/card numbers."""
        thread = make_thread(
            "[Customer]: I ordered 2 items and received 1. Missing 1 item!"
        )
        result = rule_pii_exposure("order_product_problem", thread)
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# rule_repeated_failure
# ─────────────────────────────────────────────────────────────────────────────

class TestRuleRepeatedFailure:

    def test_fires_on_third_time_with_enough_turns(self):
        thread = make_thread(
            "[Customer]: This is the third time I'm contacting you about this missing package. "
            "[AmazonHelp]: We understand. [Customer]: Still no resolution! Promises were made.",
            n_turns=4,
        )
        result = rule_repeated_failure("delivery_delay", thread)
        assert result is not None
        escalate, reason = result
        assert escalate is True
        assert "repeated" in reason.lower() or "failed" in reason.lower()

    def test_fires_on_broken_promise(self):
        thread = make_thread(
            "[Customer]: You promised me a callback yesterday and no one called. "
            "Still waiting for resolution.",
            n_turns=5,
        )
        result = rule_repeated_failure("service_complaint_vague", thread)
        assert result is not None

    def test_fires_on_still_no_with_enough_turns(self):
        thread = make_thread(
            "[Customer]: Still no package after 3 weeks.",
            n_turns=4,
        )
        result = rule_repeated_failure("delivery_delay", thread)
        assert result is not None

    def test_does_not_fire_on_first_contact(self):
        """Single-turn thread with no repeat language should not fire."""
        thread = make_thread(
            "[Customer]: My package hasn't arrived. Please help.",
            n_turns=1,
        )
        result = rule_repeated_failure("delivery_delay", thread)
        assert result is None

    def test_does_not_fire_below_turn_threshold(self):
        """Even with 'still no' language, too few turns should not fire."""
        thread = make_thread(
            "[Customer]: Still no package.",
            n_turns=2,
        )
        result = rule_repeated_failure("delivery_delay", thread)
        assert result is None

    def test_does_not_fire_on_high_turns_without_frustration(self):
        """Many turns but no repeat/failure signal should not fire."""
        thread = make_thread(
            "[Customer]: Thanks for the update! [AmazonHelp]: Happy to help! "
            "[Customer]: Great service.",
            n_turns=6,
        )
        result = rule_repeated_failure("delivery_delay", thread)
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# rule_legal_or_churn_threat
# ─────────────────────────────────────────────────────────────────────────────

class TestRuleLegalOrChurnThreat:

    def test_fires_on_legal_action(self):
        thread = make_thread(
            "[Customer]: @AmazonHelp If this is not resolved today I will take legal action!"
        )
        result = rule_legal_or_churn_threat("billing_dispute", thread)
        assert result is not None
        escalate, reason = result
        assert escalate is True
        assert "legal" in reason.lower()

    def test_fires_on_consumer_court(self):
        thread = make_thread(
            "[Customer]: I will file a complaint in consumer court if my refund is not processed."
        )
        result = rule_legal_or_churn_threat("billing_dispute", thread)
        assert result is not None

    def test_fires_on_chargeback(self):
        thread = make_thread(
            "[Customer]: I'm going to file a chargeback with my bank. This is ridiculous."
        )
        result = rule_legal_or_churn_threat("billing_dispute", thread)
        assert result is not None

    def test_fires_on_cancel_prime(self):
        thread = make_thread(
            "[Customer]: I'm cancelling my Prime membership. This is the last straw."
        )
        result = rule_legal_or_churn_threat("service_complaint_vague", thread)
        assert result is not None
        escalate, reason = result
        assert escalate is True
        assert "prime" in reason.lower() or "retention" in reason.lower()

    def test_fires_on_sue(self):
        thread = make_thread(
            "[Customer]: I'll sue Amazon if this package is not found!"
        )
        result = rule_legal_or_churn_threat("delivery_delay", thread)
        assert result is not None

    def test_does_not_fire_on_asking_about_legal_rights(self):
        """Asking a question about rights is not a threat."""
        thread = make_thread(
            "[Customer]: What are my legal rights if my package doesn't arrive?"
        )
        result = rule_legal_or_churn_threat("delivery_delay", thread)
        # 'legal' alone without 'action' should not fire — confirm
        # (may or may not fire depending on pattern; the important case is below)
        # This test mainly ensures vague mentions don't trigger
        # The rule_legal patterns require "legal action" specifically
        assert result is None

    def test_does_not_fire_on_upgrade_mention(self):
        """Positive account talk should not trigger churn detection."""
        thread = make_thread(
            "[Customer]: I'm thinking of upgrading my Prime membership."
        )
        result = rule_legal_or_churn_threat("product_info_question", thread)
        assert result is None
