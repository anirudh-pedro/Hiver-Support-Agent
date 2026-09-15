"""
src/ngram_diagnostics.py

Computes secondary n-gram lexical overlap diagnostics (BLEU and ROUGE-L)
between generated drafts and historical AmazonHelp tweets.

IMPORTANT NOTICE:
  This is explicitly labeled as a SECONDARY DIAGNOSTIC, NOT A PRIMARY QUALITY METRIC.
  Historical Twitter replies in the dataset are predominantly generic macro redirects
  (e.g., "Please reach out to us via DM") rather than grounded autonomous resolutions.
  High BLEU against historical tweets measures boilerplate imitation, not solution correctness.
"""

import json
import logging
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import nltk
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

REPO_ROOT = Path(__file__).resolve().parent.parent

DRAFTS_PATH = REPO_ROOT / "data" / "results" / "golden_drafts.jsonl"
FALLBACK_MARKER = "I'm sorry for the inconvenience! Please contact us via DM"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def lcs_length(seq1: List[str], seq2: List[str]) -> int:
    """Computes the Longest Common Subsequence (LCS) length between two token lists."""
    m, n = len(seq1), len(seq2)
    if m == 0 or n == 0:
        return 0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def compute_rouge_l(candidate: str, reference: str) -> Tuple[float, float, float]:
    """
    Computes ROUGE-L (Precision, Recall, F1) using LCS on whitespace tokens.
    """
    cand_tokens = candidate.lower().strip().split()
    ref_tokens = reference.lower().strip().split()

    if not cand_tokens or not ref_tokens:
        return 0.0, 0.0, 0.0

    lcs = lcs_length(cand_tokens, ref_tokens)
    prec = lcs / len(cand_tokens) if cand_tokens else 0.0
    rec = lcs / len(ref_tokens) if ref_tokens else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return prec, rec, f1


def compute_bleu(candidate: str, reference: str) -> Tuple[float, float, float]:
    """
    Computes BLEU-1, BLEU-2, and BLEU-4 with Chen & Cherry smoothing method 1.
    """
    cand_tokens = candidate.lower().strip().split()
    ref_tokens = [reference.lower().strip().split()]

    if not cand_tokens or not ref_tokens[0]:
        return 0.0, 0.0, 0.0

    sf = SmoothingFunction().method1
    b1 = sentence_bleu(ref_tokens, cand_tokens, weights=(1.0, 0, 0, 0), smoothing_function=sf)
    b2 = sentence_bleu(ref_tokens, cand_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=sf)
    b4 = sentence_bleu(ref_tokens, cand_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=sf)

    return b1, b2, b4


def main():
    if not DRAFTS_PATH.exists():
        logger.error("No golden drafts found at %s", DRAFTS_PATH)
        return

    records = []
    with open(DRAFTS_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # Filter out fallback records for quality measurement
    valid_records = [r for r in records if FALLBACK_MARKER not in r.get("drafted_reply", "")]
    total_valid = len(valid_records)

    bleu1_list, bleu2_list, bleu4_list = [], [], []
    rouge_p_list, rouge_r_list, rouge_f1_list = [], [], []
    by_intent: Dict[str, List[Dict[str, float]]] = {}

    for r in valid_records:
        cand = r.get("drafted_reply", "")
        ref = r.get("actual_amazon_reply", "")
        intent = r.get("predicted_intent", "other")

        b1, b2, b4 = compute_bleu(cand, ref)
        rp, rr, rf1 = compute_rouge_l(cand, ref)

        bleu1_list.append(b1)
        bleu2_list.append(b2)
        bleu4_list.append(b4)

        rouge_p_list.append(rp)
        rouge_r_list.append(rr)
        rouge_f1_list.append(rf1)

        by_intent.setdefault(intent, []).append({
            "bleu1": b1, "bleu4": b4, "rouge_f1": rf1
        })

    def stats_summary(arr: List[float]) -> Tuple[float, float, float]:
        if not arr:
            return 0.0, 0.0, 0.0
        mean = sum(arr) / len(arr)
        sorted_arr = sorted(arr)
        med = sorted_arr[len(arr) // 2]
        var = sum((x - mean) ** 2 for x in arr) / len(arr)
        std = math.sqrt(var)
        return mean, med, std

    sep = "=" * 85
    print(f"\n{sep}")
    print(f"      SECONDARY N-GRAM DIAGNOSTICS: BLEU & ROUGE-L (Valid Drafts N={total_valid})")
    print(f"{sep}")
    print(
        "\n  DISCLAIMER: Non-headline metric. Historical AmazonHelp replies are mostly generic\n"
        "  redirects (e.g. 'contact us via DM'). Lexical overlap measures brand-voice phrasing\n"
        "  similarity rather than task completion or resolution correctness.\n"
    )

    print("-- Overall Corpus Metrics -----------------------------------------------------------")
    b1_m, b1_med, b1_s = stats_summary(bleu1_list)
    b2_m, b2_med, b2_s = stats_summary(bleu2_list)
    b4_m, b4_med, b4_s = stats_summary(bleu4_list)
    rf_m, rf_med, rf_s = stats_summary(rouge_f1_list)
    rp_m, rp_med, rp_s = stats_summary(rouge_p_list)
    rr_m, rr_med, rr_s = stats_summary(rouge_r_list)

    print(f"  BLEU-1 (Unigram)     : Mean = {b1_m*100:>5.2f}%  | Median = {b1_med*100:>5.2f}%  | Std = {b1_s*100:>5.2f}%")
    print(f"  BLEU-2 (Bigram)      : Mean = {b2_m*100:>5.2f}%  | Median = {b2_med*100:>5.2f}%  | Std = {b2_s*100:>5.2f}%")
    print(f"  BLEU-4 (4-Gram)      : Mean = {b4_m*100:>5.2f}%  | Median = {b4_med*100:>5.2f}%  | Std = {b4_s*100:>5.2f}%")
    print(f"  ROUGE-L (F1 Score)   : Mean = {rf_m*100:>5.2f}%  | Median = {rf_med*100:>5.2f}%  | Std = {rf_s*100:>5.2f}%")
    print(f"  ROUGE-L (Precision)  : Mean = {rp_m*100:>5.2f}%  | Median = {rp_med*100:>5.2f}%  | Std = {rp_s*100:>5.2f}%")
    print(f"  ROUGE-L (Recall)     : Mean = {rr_m*100:>5.2f}%  | Median = {rr_med*100:>5.2f}%  | Std = {rr_s*100:>5.2f}%")

    print("\n-- Per-Intent Lexical Overlap Breakdown ---------------------------------------------")
    print(f"  {'Intent Category':<26} | {'Count':<6} | {'BLEU-1':<10} | {'BLEU-4':<10} | {'ROUGE-L F1':<10}")
    print("  " + "-" * 72)
    for intent, items in sorted(by_intent.items(), key=lambda x: -len(x[1])):
        ib1 = sum(x["bleu1"] for x in items) / len(items) * 100
        ib4 = sum(x["bleu4"] for x in items) / len(items) * 100
        irf = sum(x["rouge_f1"] for x in items) / len(items) * 100
        print(f"  {intent:<26} | {len(items):>5}  | {ib1:>8.2f}%  | {ib4:>8.2f}%  | {irf:>8.2f}%")

    print(sep + "\n")


if __name__ == "__main__":
    main()
