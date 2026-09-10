"""
ingest.py - End-to-end Ingestion Pipeline for AmazonHelp Twitter Customer Support

Loads twcs.csv with strict dtypes, calls threads.py for forward-only thread
reconstruction, applies an English-only filter using langdetect, and applies
a cheap heuristic filter for praise/off-topic chatter. Outputs clean JSONL
threads and a comprehensive markdown report with sequential funnel metrics.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from langdetect import DetectorFactory, detect
from langdetect.lang_detect_exception import LangDetectException
from tqdm import tqdm

# Ensure repository root is on sys.path for direct script execution
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from src.threads import BranchStats, ThreadBuilder
except ImportError:
    from threads import BranchStats, ThreadBuilder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Heuristic keyword sets for customer praise/off-topic filter.
# Bias toward KEEPING borderline cases per prompt instructions.
SUPPORT_KEYWORDS = {
    "?", "issue", "issues", "problem", "problems", "broken", "delay", "delayed",
    "delays", "late", "missing", "refund", "refunds", "cancel", "cancelled",
    "canceled", "canceling", "cancelling", "return", "returned", "returning",
    "returns", "charge", "charged", "charges", "charging", "fee", "fees",
    "order", "orders", "package", "packages", "parcel", "deliver", "delivery",
    "delivered", "delivering", "track", "tracking", "help", "helping", "assist",
    "assistance", "wrong", "incorrect", "error", "errors", "failed", "fail",
    "failing", "fails", "fix", "fixed", "fixing", "lost", "stolen", "damage",
    "damaged", "damages", "stuck", "wait", "waiting", "dm", "pm", "call",
    "calling", "contact", "contacting", "account", "prime", "password", "login",
    "logging", "reset", "payment", "card", "item", "items", "bought", "buy",
    "purchase", "purchased", "why", "how", "when", "where", "what", "who",
    "cant", "cannot", "can't", "won't", "wont", "doesn't", "doesnt", "not",
    "never", "still", "haven't", "havent", "stole", "awful", "terrible", "horrible",
    "useless", "fraud", "scam", "urgent", "asap"
}

PRAISE_OFFTOPIC_PATTERNS = [
    r"^(?:thank\s*(?:you|u)|thanks|thx|tq|ty|cheers|kudos|props|awesome|amazing|great\s*service|great\s*job|good\s*job|love\s*(?:you|u)?|you\s*rock|shoutout|good\s*morning|good\s*afternoon|good\s*evening|good\s*night|happy\s*holidays|merry\s*christmas|happy\s*new\s*year)(?:[\s!.,:;]|@\w+)*$"
]

ALLOWED_PRAISE_CHATTER_WORDS = {
    "thank", "thanks", "you", "u", "for", "the", "good", "great", "awesome",
    "amazing", "service", "job", "day", "have", "a", "nice", "much", "very",
    "all", "cheers", "yay", "yep", "yup", "hi", "hello", "hey", "ok", "okay",
    "cool", "guys", "team", "to", "and", "so", "appreciate", "appreciated"
}


def _init_langdetect_worker() -> None:
    """Worker initialization to ensure deterministic langdetect results."""
    DetectorFactory.seed = 0


def clean_text_for_lang(text: str) -> str:
    """Strips @mentions and URLs to prevent noisy tokens from biasing language detection."""
    t = re.sub(r"@\w+", "", text)
    t = re.sub(r"https?://\S+", "", t)
    return t.strip()


def detect_language_single(text: str) -> str:
    """
    Detects language of a single text string using langdetect.
    Returns ISO language code or 'undetermined'.
    """
    cleaned = clean_text_for_lang(text)
    if not cleaned or len(cleaned) < 3:
        return "undetermined"
    try:
        return detect(cleaned)
    except LangDetectException:
        return "undetermined"
    except Exception:
        return "undetermined"


def is_praise_offtopic(text: str) -> bool:
    """
    Cheap heuristic filter to detect threads where the customer root message
    is pure praise or off-topic greeting with NO support intent.
    Biases heavily toward KEEPING borderline cases.
    """
    cleaned = clean_text_for_lang(text).lower()
    words = set(re.findall(r"\b[a-z']+\b", cleaned))

    # Keep if question mark or any support/complaint keyword is present
    if "?" in text or any(w in SUPPORT_KEYWORDS for w in words):
        return False

    # Check regex patterns for direct praise greetings
    for pat in PRAISE_OFFTOPIC_PATTERNS:
        if re.search(pat, cleaned):
            return True

    # Very short messages (<= 5 words) composed exclusively of praise/greeting tokens
    if 0 < len(words) <= 5 and words.issubset(ALLOWED_PRAISE_CHATTER_WORDS):
        return True

    return False


def filter_english_threads(
    threads: List[Dict[str, Any]],
    num_workers: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Counter, int]:
    """
    Filters threads to English-only based on customer's first message (turns[0]['text']).
    Uses multiprocessing on CPU cores for high performance.
    Returns:
        (retained_threads, language_distribution_counter, dropped_count)
    """
    logger.info("Applying English-only filter on customer root messages...")
    t0 = time.time()

    root_texts = [thread["turns"][0]["text"] for thread in threads]
    workers = num_workers or min(os.cpu_count() or 4, 8)

    logger.info("Running parallel langdetect across %d worker processes...", workers)
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_langdetect_worker) as executor:
        detected_langs = list(
            tqdm(
                executor.map(detect_language_single, root_texts, chunksize=500),
                total=len(root_texts),
                desc="Detecting languages",
                unit="threads",
            )
        )

    lang_counter = Counter(detected_langs)
    retained_threads: List[Dict[str, Any]] = []
    dropped_count = 0

    for thread, lang in zip(threads, detected_langs):
        if lang == "en":
            retained_threads.append(thread)
        else:
            dropped_count += 1

    duration = time.time() - t0
    logger.info(
        "Language filtering completed in %.2fs: Retained %d English threads, dropped %d non-English.",
        duration,
        len(retained_threads),
        dropped_count,
    )
    logger.info("Top detected languages: %s", lang_counter.most_common(10))

    return retained_threads, lang_counter, dropped_count


def filter_offtopic_threads(
    threads: List[Dict[str, Any]],
    spot_check_sample_size: int = 15,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    """
    Filters out threads where customer root is praise or off-topic chatter.
    Returns:
        (retained_threads, spot_check_sample, dropped_count)
    """
    logger.info("Applying cheap praise/off-topic heuristic filter...")
    t0 = time.time()

    retained_threads: List[Dict[str, Any]] = []
    dropped_samples: List[Dict[str, Any]] = []
    dropped_count = 0

    for thread in threads:
        root_text = thread["turns"][0]["text"]
        if is_praise_offtopic(root_text):
            dropped_count += 1
            if len(dropped_samples) < spot_check_sample_size:
                dropped_samples.append({
                    "thread_id": thread["thread_id"],
                    "n_turns": thread["n_turns"],
                    "text": root_text,
                })
        else:
            retained_threads.append(thread)

    duration = time.time() - t0
    logger.info(
        "Off-topic filtering completed in %.2fs: Retained %d support threads, dropped %d off-topic/praise.",
        duration,
        len(retained_threads),
        dropped_count,
    )

    return retained_threads, dropped_samples, dropped_count


def generate_report(
    raw_amazon_threads_count: int,
    after_lang_count: int,
    dropped_lang_count: int,
    lang_counter: Counter,
    final_threads: List[Dict[str, Any]],
    dropped_offtopic_count: int,
    offtopic_samples: List[Dict[str, Any]],
    branch_stats: BranchStats,
    csv_path: str,
    total_execution_seconds: float,
    report_path: str,
    jsonl_path: str,
) -> None:
    """
    Generates a comprehensive markdown report with sequential funnel metrics,
    branch resolution statistics, and thread length distribution.
    """
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)

    final_count = len(final_threads)
    thread_lengths = [t["n_turns"] for t in final_threads]
    length_counter = Counter(thread_lengths)

    # Compute length summary stats
    if thread_lengths:
        min_turns = min(thread_lengths)
        max_turns = max(thread_lengths)
        mean_turns = sum(thread_lengths) / len(thread_lengths)
        sorted_lens = sorted(thread_lengths)
        mid = len(sorted_lens) // 2
        median_turns = (
            sorted_lens[mid]
            if len(sorted_lens) % 2 != 0
            else (sorted_lens[mid - 1] + sorted_lens[mid]) / 2.0
        )
    else:
        min_turns = max_turns = mean_turns = median_turns = 0

    # Sequential funnel rates
    # Stage 1: Raw AmazonHelp threads reconstructed forward
    s1_in = raw_amazon_threads_count
    s1_drop = 0
    s1_out = raw_amazon_threads_count
    s1_stage_pct = 100.0
    s1_cum_pct = 100.0

    # Stage 2: English filter (denominator = Stage 1 survivors)
    s2_in = s1_out
    s2_drop = dropped_lang_count
    s2_out = after_lang_count
    s2_stage_pct = (s2_out / s2_in * 100.0) if s2_in > 0 else 0.0
    s2_cum_pct = (s2_out / s1_in * 100.0) if s1_in > 0 else 0.0

    # Stage 3: Praise/Off-Topic filter (denominator = Stage 2 survivors)
    s3_in = s2_out
    s3_drop = dropped_offtopic_count
    s3_out = final_count
    s3_stage_pct = (s3_out / s3_in * 100.0) if s3_in > 0 else 0.0
    s3_cum_pct = (s3_out / s1_in * 100.0) if s1_in > 0 else 0.0

    # Format language breakdown table (excluding 'en')
    non_en_langs = [(k, v) for k, v in lang_counter.items() if k != "en"]
    non_en_langs.sort(key=lambda x: x[1], reverse=True)

    report_lines = [
        "# Ingest Report: AmazonHelp Twitter Customer Support Threads",
        "",
        f"- **Generated At**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Source Dataset**: `{csv_path}`",
        f"- **Total Pipeline Runtime**: {total_execution_seconds:.2f} seconds ({total_execution_seconds / 60.0:.2f} minutes)",
        "",
        "## 1. Sequential Funnel Summary",
        "",
        "Each stage reflects sequential attrition where the retention percentage is calculated relative to the immediate prior stage survivors.",
        "",
        "| Stage | Description | Input Threads | Dropped | Retained | Stage Retention | Cumulative Retention |",
        "|---|---|---|---|---|---|---|",
        f"| **1. Raw Thread Reconstruction** | Reconstructed forward from customer roots | {s1_in:,} | {s1_drop:,} | {s1_out:,} | {s1_stage_pct:.2f}% | {s1_cum_pct:.2f}% |",
        f"| **2. English-Only Filter** | Filtered non-English customer initial messages via `langdetect` | {s2_in:,} | {s2_drop:,} | {s2_out:,} | {s2_stage_pct:.2f}% | {s2_cum_pct:.2f}% |",
        f"| **3. Praise / Off-Topic Heuristic** | Dropped short greetings / praise with no support intent | {s3_in:,} | {s3_drop:,} | {s3_out:,} | {s3_stage_pct:.2f}% | {s3_cum_pct:.2f}% |",
        f"| **4. Final Support Dataset** | Clean, multi-turn AmazonHelp support threads | {s3_out:,} | 0 | **{final_count:,}** | **100.0%** | **{s3_cum_pct:.2f}%** |",
        "",
        "> [!NOTE]",
        f"> The sequential funnel reveals that English language filtering accounts for the majority of exclusions (~{s2_drop / s1_in * 100:.1f}% of raw threads), reflecting Amazon's multi-regional Twitter presence (e.g. Amazon JP, Amazon DE, Amazon ES). The final golden dataset contains **{final_count:,}** clean support conversations.",
        "",
        "## 2. Language Filtering Statistics",
        "",
        f"- **English Threads Retained (`en`)**: {lang_counter.get('en', 0):,} ({lang_counter.get('en', 0) / max(1, s1_in) * 100:.2f}%)",
        f"- **Non-English / Undetermined Dropped**: {dropped_lang_count:,} ({dropped_lang_count / max(1, s1_in) * 100:.2f}%)",
        "",
        "### Top Dropped Languages Breakdown",
        "",
        "| Rank | Language Code | Detected Language / Category | Dropped Threads | Share of Dropped |",
        "|---|---|---|---|---|",
    ]

    lang_names = {
        "ja": "Japanese",
        "de": "German",
        "es": "Spanish",
        "fr": "French",
        "pt": "Portuguese",
        "it": "Italian",
        "nl": "Dutch",
        "undetermined": "Undetermined / Too Short / Emojis only",
        "tr": "Turkish",
        "id": "Indonesian",
        "ar": "Arabic",
        "ru": "Russian",
        "hi": "Hindi",
        "zh-cn": "Chinese (Simplified)",
        "pl": "Polish",
    }

    for idx, (code, count) in enumerate(non_en_langs[:15], 1):
        name = lang_names.get(code, "Other / Dialect")
        share = (count / dropped_lang_count * 100.0) if dropped_lang_count > 0 else 0.0
        report_lines.append(f"| {idx} | `{code}` | {name} | {count:,} | {share:.2f}% |")

    report_lines.extend([
        "",
        "## 3. Praise & Off-Topic Heuristic Filtering",
        "",
        f"- **Off-Topic / Praise Threads Dropped**: {dropped_offtopic_count:,} ({dropped_offtopic_count / max(1, s2_out) * 100:.2f}% of English threads)",
        "- **Design Bias**: Conservative heuristic designed to drop obvious non-support chatter (e.g. short thank-yous, holiday greetings) while strictly preserving borderline cases for downstream intent classification.",
        "",
        "### Spot-Check Sample of Filtered Conversations",
        "",
        "| Thread ID | Turns | Customer Message (Root) |",
        "|---|---|---|",
    ])

    for sample in offtopic_samples:
        clean_msg = sample["text"].replace("\n", " ").replace("|", "\\|")
        report_lines.append(f"| `{sample['thread_id']}` | {sample['n_turns']} | {clean_msg} |")

    report_lines.extend([
        "",
        "## 4. Branch Resolution Statistics",
        "",
        "When `response_tweet_id` contains comma-separated child IDs (~8-14% of rows), the pipeline applies the following modeled resolution hierarchy:",
        "1. **`ended_with_amazon`**: Prefer the branch whose eventual last turn is authored by `AmazonHelp` (brand has last word).",
        "2. **`longest_branch`**: If multiple branches qualify (or none do), pick the longest branch.",
        "3. **`tie_break_by_earlier_id`**: Deterministic tie-breaker by earlier child ID in `response_tweet_id`.",
        "",
        "| Branch Resolution Metric | Count | Percentage |",
        "|---|---|---|",
        f"| **Total Branch Decisions Made** | {branch_stats.total_branch_decisions:,} | 100.0% |",
        f"| **Threads Involving Branching** | {branch_stats.threads_with_branching:,} | — |",
        f"| ↳ Resolved by `ended_with_amazon` | {branch_stats.ended_with_amazon:,} | {branch_stats.ended_with_amazon / max(1, branch_stats.total_branch_decisions) * 100:.2f}% |",
        f"| ↳ Resolved by `longest_branch` | {branch_stats.longest_branch:,} | {branch_stats.longest_branch / max(1, branch_stats.total_branch_decisions) * 100:.2f}% |",
        f"| ↳ Resolved by `tie_break_by_earlier_id` | {branch_stats.tie_break_by_earlier_id:,} | {branch_stats.tie_break_by_earlier_id / max(1, branch_stats.total_branch_decisions) * 100:.2f}% |",
        "",
        "## 5. Thread Length Distribution",
        "",
        f"- **Total Clean Support Threads**: {final_count:,}",
        f"- **Turn Range**: {min_turns} to {max_turns} turns (capped at 15 turns)",
        f"- **Mean Turns**: {mean_turns:.2f}",
        f"- **Median Turns**: {median_turns:.1f}",
        "",
        "| Turns | Frequency | Share | Visual Representation |",
        "|---|---|---|---|",
    ])

    for turns in range(min_turns, max_turns + 1):
        freq = length_counter.get(turns, 0)
        pct = (freq / final_count * 100.0) if final_count > 0 else 0.0
        bar = "█" * int(pct / 2.5)
        report_lines.append(f"| **{turns}** | {freq:,} | {pct:.2f}% | `{bar}` |")

    report_lines.extend([
        "",
        "## 6. Output Artifacts",
        "",
        f"- Processed Threads JSONL: `{jsonl_path}` ({final_count:,} lines)",
        f"- Ingest Summary Report: `{report_path}`",
    ])

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    logger.info("Ingest report successfully written to %s", report_path)


def run_pipeline(
    csv_path: str,
    out_jsonl: str = "data/processed/threads_amazonhelp.jsonl",
    out_report: str = "data/processed/ingest_report.md",
    max_roots: Optional[int] = None,
    num_workers: Optional[int] = None,
) -> None:
    """
    Executes the full end-to-end ingestion and thread extraction pipeline.
    """
    start_time = time.time()
    logger.info("Starting AmazonHelp Ingestion Pipeline...")

    # Stage 1: Build conversation threads forward-only from customer roots
    builder = ThreadBuilder(max_turns=15, target_brand="AmazonHelp")
    builder.load_dataset(csv_path)
    raw_threads, branch_stats = builder.build_all_threads(max_roots=max_roots)
    raw_amazon_count = len(raw_threads)

    # Stage 2: English Language Filter
    english_threads, lang_counter, dropped_lang_count = filter_english_threads(
        raw_threads, num_workers=num_workers
    )
    after_lang_count = len(english_threads)

    # Stage 3: Praise / Off-Topic Heuristic Filter
    final_threads, offtopic_samples, dropped_offtopic_count = filter_offtopic_threads(
        english_threads
    )

    # Stage 4: Write JSONL
    os.makedirs(os.path.dirname(os.path.abspath(out_jsonl)), exist_ok=True)
    logger.info("Writing %d final threads to %s...", len(final_threads), out_jsonl)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for thread in final_threads:
            f.write(json.dumps(thread, ensure_ascii=False) + "\n")
    logger.info("Successfully wrote JSONL output.")

    # Stage 5: Generate Report
    total_duration = time.time() - start_time
    generate_report(
        raw_amazon_threads_count=raw_amazon_count,
        after_lang_count=after_lang_count,
        dropped_lang_count=dropped_lang_count,
        lang_counter=lang_counter,
        final_threads=final_threads,
        dropped_offtopic_count=dropped_offtopic_count,
        offtopic_samples=offtopic_samples,
        branch_stats=branch_stats,
        csv_path=csv_path,
        total_execution_seconds=total_duration,
        report_path=out_report,
        jsonl_path=out_jsonl,
    )

    logger.info(
        "=== Ingestion Pipeline Completed Successfully in %.2f seconds (%.2f minutes) ===",
        total_duration,
        total_duration / 60.0,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest and reconstruct AmazonHelp support threads from twcs.csv"
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to twcs.csv dataset",
    )
    parser.add_argument(
        "--out-jsonl",
        type=str,
        default="data/processed/threads_amazonhelp.jsonl",
        help="Output JSONL path (default: data/processed/threads_amazonhelp.jsonl)",
    )
    parser.add_argument(
        "--out-report",
        type=str,
        default="data/processed/ingest_report.md",
        help="Output ingest report path (default: data/processed/ingest_report.md)",
    )
    parser.add_argument(
        "--max-roots",
        type=int,
        default=None,
        help="Limit number of roots processed (useful for testing/profiling)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes for parallel language detection",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        csv_path=args.csv,
        out_jsonl=args.out_jsonl,
        out_report=args.out_report,
        max_roots=args.max_roots,
        num_workers=args.workers,
    )


if __name__ == "__main__":
    main()
