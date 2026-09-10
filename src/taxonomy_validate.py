"""
taxonomy_validate.py - Validate the 8 draft intent taxonomy over 300 sampled threads.

Performs Step 1 (Validation at scale) & Step 2 (Human spot-check prep):
- Samples 300 threads from data/processed/threads_amazonhelp.jsonl (fixed seed 42).
- Runs LLM intent classification on the customer's first message.
- Outputs data/processed/taxonomy_validation_sample.jsonl.
- Outputs data/processed/taxonomy_spotcheck.csv (60 random threads, fixed seed 123).
- Prints intent distribution summary, 'other' count, and the 15 lowest-confidence cases.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import random
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Optional

# Ensure UTF-8 output encoding on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from src.intents import DEFAULT_MODEL, IntentResult, classify_intents_batch, get_llm_client
except ImportError:
    from intents import DEFAULT_MODEL, IntentResult, classify_intents_batch, get_llm_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Seeds
SEED_SAMPLE_300 = 42
SEED_SPOTCHECK_60 = 123


def load_all_threads(jsonl_path: str) -> List[Dict[str, Any]]:
    """Loads all threads from JSONL file."""
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"Thread file not found: {jsonl_path}")
    threads = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                threads.append(json.loads(line))
    return threads


def run_taxonomy_validation(
    input_jsonl: str = "data/processed/threads_amazonhelp.jsonl",
    output_validation_jsonl: str = "data/processed/taxonomy_validation_sample.jsonl",
    output_spotcheck_csv: str = "data/processed/taxonomy_spotcheck.csv",
    sample_size: int = 300,
    spotcheck_size: int = 60,
    model: Optional[str] = None,
    workers: int = 3,
) -> None:
    logger.info("Starting Taxonomy Validation Stage...")
    client, detected_model = get_llm_client()
    selected_model = model or detected_model
    if client is None:
        raise RuntimeError("No LLM API key detected. Please configure 'groq_api' in .env.")

    all_threads = load_all_threads(input_jsonl)
    logger.info("Loaded %d total processed threads from %s.", len(all_threads), input_jsonl)

    # Step 1: Deterministic random sample of 300 threads
    rng_sample = random.Random(SEED_SAMPLE_300)
    if len(all_threads) > sample_size:
        sampled_threads = rng_sample.sample(all_threads, sample_size)
    else:
        sampled_threads = all_threads
    logger.info("Sampled %d threads with seed %d.", len(sampled_threads), SEED_SAMPLE_300)

    # Classify sampled threads via real LLM API (allow_fallback=False strictly enforced)
    logger.info("Running real LLM intent classification using model '%s' (concurrency=%d)...", selected_model, workers)
    t0 = time.time()
    classification_results: List[IntentResult] = classify_intents_batch(
        threads=sampled_threads,
        include_brand_reply=False,
        model=selected_model,
        max_workers=workers,
        client=client,
        allow_fallback=False,  # Enforces NO silent fallback
    )
    duration = time.time() - t0
    logger.info("Classified %d threads in %.2f seconds.", len(sampled_threads), duration)

    # Prepare output records
    labeled_records = []
    for thread, res in zip(sampled_threads, classification_results):
        first_msg = thread["turns"][0]["text"] if thread.get("turns") else ""
        record = {
            "thread_id": thread["thread_id"],
            "first_message": first_msg,
            "assigned_intent": res.intent,
            "confidence": round(res.confidence, 4),
            "justification": res.reasoning,
        }
        labeled_records.append(record)

    # Write Step 1 output: data/processed/taxonomy_validation_sample.jsonl
    os.makedirs(os.path.dirname(os.path.abspath(output_validation_jsonl)), exist_ok=True)
    with open(output_validation_jsonl, "w", encoding="utf-8") as f:
        for rec in labeled_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Wrote %d validation records to %s.", len(labeled_records), output_validation_jsonl)

    # Step 2: Human spot-check prep: 60 random threads from the 300 (fixed seed 123)
    rng_spotcheck = random.Random(SEED_SPOTCHECK_60)
    spotcheck_records = rng_spotcheck.sample(
        labeled_records, min(spotcheck_size, len(labeled_records))
    )

    os.makedirs(os.path.dirname(os.path.abspath(output_spotcheck_csv)), exist_ok=True)
    with open(output_spotcheck_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["thread_id", "first_message", "llm_assigned_intent", "human_label"])
        for rec in spotcheck_records:
            writer.writerow([
                rec["thread_id"],
                rec["first_message"],
                rec["assigned_intent"],
                "",  # Empty human_label column for manual spreadsheet review
            ])
    logger.info("Wrote %d spot-check records to %s.", len(spotcheck_records), output_spotcheck_csv)

    # Print summary statistics
    intents = [r["assigned_intent"] for r in labeled_records]
    intent_counts = Counter(intents)
    total_samples = len(labeled_records)

    print("\n" + "=" * 80)
    print("                      TAXONOMY VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Total Sampled Threads Analyzed: {total_samples}")
    print(f"Model Used: {model}")
    print(f"Seed (300 sample): {SEED_SAMPLE_300} | Seed (60 spot-check): {SEED_SPOTCHECK_60}")
    print("-" * 80)
    print(f"{'Intent Category':<28} | {'Count':>6} | {'Percentage':>10}")
    print("-" * 80)

    for intent, count in intent_counts.most_common():
        pct = (count / total_samples) * 100.0
        print(f"{intent:<28} | {count:>6} | {pct:>9.2f}%")

    other_count = intent_counts.get("other", 0)
    other_pct = (other_count / total_samples) * 100.0
    print("-" * 80)
    print(f"--> 'other' Rate (Taxonomy Gap Indicator): {other_count}/{total_samples} ({other_pct:.2f}%)")
    print("=" * 80)

    # Print the 15 lowest-confidence examples for manual review
    sorted_by_conf = sorted(labeled_records, key=lambda x: x["confidence"])
    lowest_15 = sorted_by_conf[:15]

    print("\n" + "=" * 80)
    print("             15 LOWEST-CONFIDENCE EXAMPLES (BORDERLINE / AMBIGUOUS)")
    print("=" * 80)

    for i, item in enumerate(lowest_15, 1):
        clean_msg = item["first_message"].replace("\n", " ")
        if len(clean_msg) > 75:
            clean_msg = clean_msg[:72] + "..."
        print(f"[{i:02d}] Thread ID: {item['thread_id']} | Conf: {item['confidence']:.2f} | Intent: {item['assigned_intent']}")
        print(f"     Reason:  {item['justification']}")
        print(f"     Message: \"{clean_msg}\"")
        print("-" * 80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate 8-intent taxonomy across 300 sampled threads."
    )
    parser.add_argument(
        "--input-jsonl",
        type=str,
        default="data/processed/threads_amazonhelp.jsonl",
        help="Path to processed threads JSONL",
    )
    parser.add_argument(
        "--output-validation",
        type=str,
        default="data/processed/taxonomy_validation_sample.jsonl",
        help="Path for 300-thread validation JSONL",
    )
    parser.add_argument(
        "--output-spotcheck",
        type=str,
        default="data/processed/taxonomy_spotcheck.csv",
        help="Path for 60-thread spotcheck CSV",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=300,
        help="Sample size for validation (default: 300)",
    )
    parser.add_argument(
        "--spotcheck-size",
        type=int,
        default=60,
        help="Spotcheck sample size (default: 60)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"LLM model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Number of concurrent workers for LLM API calls (default: 3)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_taxonomy_validation(
        input_jsonl=args.input_jsonl,
        output_validation_jsonl=args.output_validation,
        output_spotcheck_csv=args.output_spotcheck,
        sample_size=args.sample_size,
        spotcheck_size=args.spotcheck_size,
        model=args.model,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
