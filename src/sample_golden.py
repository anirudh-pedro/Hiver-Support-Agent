"""
sample_golden.py - Build the 220-thread Golden Evaluation Set.

This script creates the ground-truth evaluation benchmark for the AmazonHelp support agent:
1. Excludes all threads used in previous taxonomy validation (300 threads) and spot-check (60 threads)
   to ensure the golden evaluation set is strictly untouched/unseen data.
2. Applies STRATIFIED SAMPLING across the 8 locked intent categories plus "other":
   - Base distribution: delivery_delay (~35%), service_complaint_vague (~20%),
     product_info_question (~13%), non_support (~10%), order_product_problem (~10%),
     billing_dispute (~6%), account_access (~2%), scam_phishing_check (~2%), other (~3%).
   - Oversampling of rare classes: account_access (14), scam_phishing_check (14), other (14)
     to at least 12-15 examples each so per-class F1/accuracy can be reliably measured.
   - Proportional allocation of remaining 178 slots across dominant classes:
     delivery_delay: 66, service_complaint_vague: 38, product_info_question: 25,
     order_product_problem: 19, non_support: 19, billing_dispute: 11.
     Total = 66 + 38 + 25 + 19 + 19 + 11 + 14 + 14 + 14 = 220 threads.
3. Deterministic sampling via fixed random seed (SEED_STRATIFIED = 42).
4. Deterministic shuffle via fixed seed (SEED_SHUFFLE = 777) so human annotation is completely blind
   to stratum grouping.
5. Formats full thread text with explicit speaker labels ([Customer] / [AmazonHelp]).
6. Emits:
   - data/golden/golden_set_unlabeled.csv (columns: thread_id, full_thread_text, n_turns)
   - data/golden/golden_set_template.csv (columns: thread_id, full_thread_text, n_turns,
     gold_intent, gold_escalate, gold_reason, gold_reply_notes)
"""

from __future__ import annotations

import csv
import json
import logging
import os
import random
import re
import sys
from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple

# Ensure UTF-8 output encoding on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Fixed deterministic seeds
SEED_STRATIFIED = 42
SEED_SHUFFLE = 777

# Target sample counts per stratum (Total = 220)
STRATUM_TARGETS: Dict[str, int] = {
    "delivery_delay": 66,           # ~35% base proportion
    "service_complaint_vague": 38,  # ~20% base proportion
    "product_info_question": 25,    # ~13% base proportion
    "order_product_problem": 19,    # ~10% base proportion
    "non_support": 19,              # ~10% base proportion
    "billing_dispute": 11,          # ~6% base proportion
    "account_access": 14,           # Oversampled rare class (target >= 12-15)
    "scam_phishing_check": 14,      # Oversampled rare class (target >= 12-15)
    "other": 14,                    # Oversampled rare class (target >= 12-15)
}

# Regex patterns for stratum candidate filtering
PATTERNS = {
    "scam_phishing_check": re.compile(
        r"\b(phish|phishing|scam|scams|scammer|spoof|spoofed|fake email|fake order|fake message|suspicious email|fraudulent email|is this email legit|is this real|is this legit|fake prize)\b",
        re.IGNORECASE,
    ),
    "account_access": re.compile(
        r"\b(locked out|lock out|lockout|can't sign in|cant sign in|cannot sign in|unable to sign in|can't log in|cant log in|cannot log in|unable to log in|login issue|password reset|reset password|forgot password|2fa|otp|verification code|two-factor|two factor|authenticator|hacked|account locked|account on hold|unauthorized login)\b",
        re.IGNORECASE,
    ),
    "billing_dispute": re.compile(
        r"\b(charged twice|double charged|unauthorized charge|wrong charge|refund|refunded|rebill|overcharged|billing|charged me|charged my card|charged my account|cashback|promotional credit|subscription charge|membership fee charged)\b",
        re.IGNORECASE,
    ),
    "order_product_problem": re.compile(
        r"\b(damaged|broken|defective|faulty|wrong item|different item|missing item|missing parts|crushed|shattered|scratched|ripped|torn|leak|leaking|app crash|app crashing|fire stick crash|poor quality|box was empty)\b",
        re.IGNORECASE,
    ),
    "service_complaint_vague": re.compile(
        r"\b(worst customer service|horrible service|awful service|terrible service|useless customer service|rude representative|bad experience with customer service|waiting on hold|customer service is a joke|shocking service|disgusting service)\b",
        re.IGNORECASE,
    ),
    "delivery_delay": re.compile(
        r"\b(late|delay|delayed|not arrived|haven't received|hasn't arrived|where is my|tracking|carrier|package was supposed to|guaranteed delivery|delivered but|delivered to wrong|lost package|dispatch|courier)\b",
        re.IGNORECASE,
    ),
    "product_info_question": re.compile(
        r"\b(compatible with|compatibility|does it come with|when will it be in stock|release date|specifications|how do i|how to use|can i use|warranty|how much is|available to let me|is there an app)\b",
        re.IGNORECASE,
    ),
    "non_support": re.compile(
        r"\b(thank you so much|thanks amazon|great job|kudos|love amazon|best customer service ever|shout out to|you guys rock|happy holidays|merry christmas)\b",
        re.IGNORECASE,
    ),
    "other": re.compile(
        r"\b(unsubscribe|promotional email|marketing email|stop sending me email|amazon locker|return code|locker code|return window|gift registry|wedding registry|baby registry|trade-in|trade in|wishlist|wish list|affiliate|associates|cancel my order|cancel order|how do i return|drop off|concert ticket|amazon tickets|feedback on your website)\b",
        re.IGNORECASE,
    ),
}

# Negative pattern to prevent obvious misclassifications into 'other'
CORE_EXCLUDE_FOR_OTHER = re.compile(
    r"\b(late|delay|not arrived|haven't received|tracking|carrier|where is my|charge|refund|bill|broken|damaged|wrong item|defective|password|login|2fa|hacked|scam|phishing)\b",
    re.IGNORECASE,
)


def load_excluded_thread_ids(
    val_path: str = "data/processed/taxonomy_validation_sample.jsonl",
    spot_path: str = "data/processed/taxonomy_spotcheck.csv",
) -> Set[int]:
    """Collects all thread IDs from prior validation and spot-check to exclude."""
    excluded: Set[int] = set()
    if os.path.exists(val_path):
        with open(val_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        d = json.loads(line)
                        excluded.add(int(d["thread_id"]))
                    except (ValueError, KeyError):
                        pass
        logger.info("Loaded %d thread IDs from validation sample %s", len(excluded), val_path)

    if os.path.exists(spot_path):
        count_before = len(excluded)
        with open(spot_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("thread_id"):
                    try:
                        excluded.add(int(row["thread_id"]))
                    except ValueError:
                        pass
        logger.info("Total unique excluded threads after spotcheck %s: %d (+%d)", spot_path, len(excluded), len(excluded) - count_before)

    return excluded


def format_full_thread_text(turns: List[Dict[str, Any]]) -> str:
    """
    Concatenates all turns in a thread chronologically with speaker labels.
    Uses clean speaker markers: [Customer] or [AmazonHelp].
    """
    lines = []
    for turn in turns:
        author = str(turn.get("author_id", ""))
        speaker = "AmazonHelp" if author == "AmazonHelp" else "Customer"
        text = str(turn.get("text", "")).strip()
        lines.append(f"[{speaker}]: {text}")
    return "\n".join(lines)


def build_candidate_pools(
    input_jsonl: str,
    excluded_ids: Set[int],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Scans threads_amazonhelp.jsonl and partitions non-excluded threads into
    stratum candidate pools using high-precision lexical heuristics.
    """
    pools: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    total_processed = 0
    total_skipped_excluded = 0

    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            thread = json.loads(line)
            tid = int(thread["thread_id"])
            if tid in excluded_ids:
                total_skipped_excluded += 1
                continue

            total_processed += 1
            turns = thread.get("turns", [])
            first_text = turns[0]["text"] if turns else ""

            # Stratum assignment with priority given to rare classes
            if PATTERNS["scam_phishing_check"].search(first_text):
                pools["scam_phishing_check"].append(thread)
            elif PATTERNS["account_access"].search(first_text):
                pools["account_access"].append(thread)
            elif PATTERNS["billing_dispute"].search(first_text):
                pools["billing_dispute"].append(thread)
            elif PATTERNS["order_product_problem"].search(first_text):
                pools["order_product_problem"].append(thread)
            elif PATTERNS["service_complaint_vague"].search(first_text):
                pools["service_complaint_vague"].append(thread)
            elif PATTERNS["delivery_delay"].search(first_text):
                pools["delivery_delay"].append(thread)
            elif PATTERNS["product_info_question"].search(first_text):
                pools["product_info_question"].append(thread)
            elif PATTERNS["non_support"].search(first_text):
                pools["non_support"].append(thread)
            elif PATTERNS["other"].search(first_text) and not CORE_EXCLUDE_FOR_OTHER.search(first_text):
                pools["other"].append(thread)

    logger.info("Scanned %d eligible threads (skipped %d excluded).", total_processed, total_skipped_excluded)
    for stratum, candidates in sorted(pools.items()):
        logger.info("  Stratum '%s': %d available candidates", stratum, len(candidates))

    return pools


def sample_golden_set(
    pools: Dict[str, List[Dict[str, Any]]],
    targets: Dict[str, int],
    seed_stratified: int = SEED_STRATIFIED,
    seed_shuffle: int = SEED_SHUFFLE,
) -> List[Dict[str, Any]]:
    """
    Pulls exact target count from each stratum deterministically, then shuffles
    the entire set so rows are presented blind to human annotators.
    """
    rng_sample = random.Random(seed_stratified)
    sampled_threads: List[Dict[str, Any]] = []

    for stratum, target_count in targets.items():
        available = pools.get(stratum, [])
        if len(available) < target_count:
            raise ValueError(
                f"Stratum '{stratum}' only has {len(available)} candidates, but target is {target_count}!"
            )
        # Deterministic sample from pool
        selected = rng_sample.sample(available, target_count)
        sampled_threads.extend(selected)
        logger.info("Sampled %d / %d threads for stratum '%s'", len(selected), target_count, stratum)

    assert len(sampled_threads) == sum(targets.values()), (
        f"Expected {sum(targets.values())} sampled threads, got {len(sampled_threads)}"
    )

    # Deterministic shuffle to eliminate sequential stratum bias during human labeling
    rng_shuffle = random.Random(seed_shuffle)
    rng_shuffle.shuffle(sampled_threads)
    logger.info("Shuffled %d total golden set threads with seed %d.", len(sampled_threads), seed_shuffle)

    return sampled_threads


def export_golden_files(
    threads: List[Dict[str, Any]],
    output_dir: str = "data/golden",
) -> Tuple[str, str]:
    """
    Exports:
    1. golden_set_unlabeled.csv (thread_id, full_thread_text, n_turns)
    2. golden_set_template.csv (thread_id, full_thread_text, n_turns, gold_intent, gold_escalate, gold_reason, gold_reply_notes)
    """
    os.makedirs(output_dir, exist_ok=True)
    unlabeled_path = os.path.join(output_dir, "golden_set_unlabeled.csv")
    template_path = os.path.join(output_dir, "golden_set_template.csv")

    # 1. Write golden_set_unlabeled.csv
    with open(unlabeled_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["thread_id", "full_thread_text", "n_turns"])
        for t in threads:
            tid = t["thread_id"]
            full_text = format_full_thread_text(t.get("turns", []))
            n_turns = t.get("n_turns", len(t.get("turns", [])))
            writer.writerow([tid, full_text, n_turns])

    logger.info("Exported unlabeled golden set to: %s (%d rows)", unlabeled_path, len(threads))

    # 2. Write golden_set_template.csv (completely empty gold_* columns)
    with open(template_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "thread_id",
            "full_thread_text",
            "n_turns",
            "gold_intent",
            "gold_escalate",
            "gold_reason",
            "gold_reply_notes",
        ])
        for t in threads:
            tid = t["thread_id"]
            full_text = format_full_thread_text(t.get("turns", []))
            n_turns = t.get("n_turns", len(t.get("turns", [])))
            writer.writerow([
                tid,
                full_text,
                n_turns,
                "",  # gold_intent (empty for human ground truth)
                "",  # gold_escalate (empty for human ground truth)
                "",  # gold_reason (empty for human ground truth)
                "",  # gold_reply_notes (empty for human ground truth)
            ])

    logger.info("Exported labeling template to: %s (%d rows)", template_path, len(threads))
    return unlabeled_path, template_path


def main() -> None:
    logger.info("=== Starting Golden Evaluation Set Sampling ===")
    jsonl_path = "data/processed/threads_amazonhelp.jsonl"
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"Missing input threads file: {jsonl_path}")

    # Step 1: Collect excluded thread IDs (prior validation & spotcheck)
    excluded_ids = load_excluded_thread_ids()

    # Step 2: Build candidate stratum pools from unseen threads
    pools = build_candidate_pools(jsonl_path, excluded_ids)

    # Step 3: Deterministic stratified sampling & blind shuffle
    golden_threads = sample_golden_set(pools, STRATUM_TARGETS, seed_stratified=SEED_STRATIFIED, seed_shuffle=SEED_SHUFFLE)

    # Step 4: Export unlabeled CSV and labeling template CSV
    unlabeled_csv, template_csv = export_golden_files(golden_threads)

    logger.info("=== Golden Evaluation Set Sampling Complete ===")
    logger.info("Total Golden Threads: %d", len(golden_threads))
    logger.info("Target distribution across strata: %s", json.dumps(STRATUM_TARGETS, indent=2))


if __name__ == "__main__":
    main()
