"""
tests/test_retrieve.py

Sanity checks and unit tests for the AmazonHelp semantic retrieval index:
1. Delivery delay queries return delivery-related historical resolutions.
2. Exact / near-identical queries rank the indexed example as #1 with similarity ~ 1.0.
3. k=5 returns 5 distinct non-duplicate results.
4. Macro deduplication correctly normalizes handles, URLs, and sign-offs.
5. Empty / whitespace queries return empty lists gracefully.
6. Billing dispute query retrieves relevant billing/charge resolutions.
"""

import json
from pathlib import Path
import pytest

from src.retrieve import (
    RetrievedExample,
    AmazonHelpRetriever,
    normalize_resolution_macro,
    retrieve_similar,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_META_PATH = PROJECT_ROOT / "data" / "index" / "amazonhelp_meta.jsonl"
FAISS_INDEX_PATH = PROJECT_ROOT / "data" / "index" / "amazonhelp.faiss"


@pytest.fixture(scope="module")
def ensure_index_built():
    """Ensures FAISS index and metadata exist before running retrieval tests."""
    assert FAISS_INDEX_PATH.exists(), f"FAISS index missing at {FAISS_INDEX_PATH}"
    assert INDEX_META_PATH.exists(), f"Metadata missing at {INDEX_META_PATH}"


def test_dedup_macro_normalization():
    """Verifies that URLs, Twitter handles, agent signatures, and whitespace are normalized."""
    text1 = "@AmazonHelp We'd like to help! Please reach out to us here: https://t.co/xyz123 ^TN"
    text2 = "@115820 We'd like to help! Please reach out to us here: https://t.co/abc789 ^AG"
    text3 = "@user Different resolution text entirely. Please check your tracking link: https://t.co/foo ^JS"

    norm1 = normalize_resolution_macro(text1)
    norm2 = normalize_resolution_macro(text2)
    norm3 = normalize_resolution_macro(text3)

    assert norm1 == norm2, f"Expected macros to match:\n1: {norm1}\n2: {norm2}"
    assert norm1 != norm3, f"Expected different resolutions to not match:\n1: {norm1}\n3: {norm3}"
    assert "^" not in norm1
    assert "https" not in norm1
    assert "@" not in norm1


def test_empty_or_whitespace_query(ensure_index_built):
    """Empty or whitespace-only queries should return empty results without crashing."""
    assert retrieve_similar("") == []
    assert retrieve_similar("    ") == []
    assert retrieve_similar("\n\t") == []


def test_delivery_delay_query_relevance(ensure_index_built):
    """A clear delivery delay query returns delivery/order related historical resolutions."""
    query = "My package was supposed to be delivered yesterday but it hasn't arrived. Where is my order?"
    results = retrieve_similar(query, k=5)

    assert len(results) == 5, f"Expected 5 results, got {len(results)}"
    top = results[0]
    assert isinstance(top, RetrievedExample)
    assert top.similarity_score > 0.60, f"Expected high similarity score for clear query, got {top.similarity_score}"

    # Semantic relevance check: at least 3 of 5 top results contain delivery/order/tracking/dispatch/carrier terms
    delivery_terms = {"deliver", "delivery", "order", "package", "arrive", "tracking", "carrier", "late", "delay", "dispatch"}
    matches = 0
    for res in results:
        combined = (res.customer_message + " " + res.brand_resolution).lower()
        if any(term in combined for term in delivery_terms):
            matches += 1
    assert matches >= 3, f"Expected at least 3 delivery-relevant results, found {matches}/5"


def test_near_identical_query_ranks_top(ensure_index_built):
    """Querying with the exact text of an indexed example ranks it #1 with similarity ~ 1.0."""
    # Read the first indexed record
    first_record = None
    with open(INDEX_META_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                first_record = json.loads(line)
                break

    assert first_record is not None, "Index metadata is empty"
    target_thread_id = first_record["thread_id"]
    exact_text = first_record["customer_message"]

    results = retrieve_similar(exact_text, k=5, dedup_macros=False)
    assert len(results) > 0
    top = results[0]
    assert top.thread_id == target_thread_id, f"Expected top thread_id {target_thread_id}, got {top.thread_id}"
    assert top.similarity_score >= 0.98, f"Expected similarity >= 0.98 for exact match, got {top.similarity_score}"


def test_k_distinct_results_and_deduplication(ensure_index_built):
    """k=5 returns 5 distinct non-duplicate results with unique thread_ids and distinct macros."""
    query = "Where is my item? It says delivered but nothing was left at my door"
    results = retrieve_similar(query, k=5, dedup_macros=True)

    assert len(results) == 5, f"Expected 5 results, got {len(results)}"

    # Check distinct thread IDs
    thread_ids = [r.thread_id for r in results]
    assert len(set(thread_ids)) == 5, f"Thread IDs are not unique: {thread_ids}"

    # Check distinct normalized brand resolutions
    macro_keys = [normalize_resolution_macro(r.brand_resolution) for r in results]
    assert len(set(macro_keys)) == 5, f"Brand resolutions contain duplicate macros: {macro_keys}"

    # Ensure similarity scores are sorted descending
    scores = [r.similarity_score for r in results]
    assert scores == sorted(scores, reverse=True), f"Results not sorted by descending score: {scores}"


def test_billing_dispute_query(ensure_index_built):
    """Billing dispute queries retrieve billing/charge related examples."""
    query = "I was charged twice on my credit card for my Prime membership subscription"
    results = retrieve_similar(query, k=3)

    assert len(results) == 3
    top = results[0]
    assert top.similarity_score > 0.55

    # Check for financial/account/membership/charge terms
    billing_terms = {"charge", "charged", "card", "prime", "membership", "refund", "subscription", "bank", "account", "billed"}
    combined_all = " ".join((r.customer_message + " " + r.brand_resolution).lower() for r in results)
    assert any(term in combined_all for term in billing_terms), f"No billing terms found in: {combined_all}"
