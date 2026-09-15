"""
src/compute_judge_agreement.py

Computes inter-rater agreement and correlation between Human Judge scores
and LLM Judge scores on the stratified 30-thread evaluation sample.

Metrics Computed:
  1. Spearman Rank Correlation (rho): Primary metric (justified for ordinal 1-5 Likert scales).
  2. Pearson Correlation (r): Secondary linear correlation diagnostic.
  3. Exact Match Rate (%): Percentage of items where human_score == judge_score.
  4. Adjacent Agreement Rate (%): Percentage of items where |human_score - judge_score| <= 1.
  5. Mean Absolute Error (MAE): Average absolute score discrepancy.

Statistical Rationale:
  Likert scales (1-5 integers) are ordinal rather than interval measurements;
  the perceptual difference between 1 and 2 is not mathematically guaranteed to equal
  the difference between 4 and 5. Therefore, Spearman's rank correlation (rho) is
  the mathematically rigorous primary measure over Pearson (r), as it tests monotonic
  agreement without imposing Gaussian or linear interval assumptions.
"""

import csv
import json
import logging
import math
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import stats
warnings.filterwarnings("ignore", category=stats.ConstantInputWarning)

REPO_ROOT = Path(__file__).resolve().parent.parent

HUMAN_SCORES_CSV = REPO_ROOT / "data" / "golden" / "human_judge_30.csv"
HUMAN_SCORES_JSONL = REPO_ROOT / "data" / "results" / "human_judged_30.jsonl"
JUDGED_RESULTS_PATH = REPO_ROOT / "data" / "results" / "golden_judged.jsonl"


def load_human_scores() -> Dict[int, Dict[str, Any]]:
    """Loads human evaluation scores."""
    scores = {}
    if HUMAN_SCORES_CSV.exists():
        with open(HUMAN_SCORES_CSV, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                tid = int(row["thread_id"])
                scores[tid] = {
                    "tone_empathy": int(row["tone_empathy"]),
                    "factual_correctness": int(row["factual_correctness"]),
                    "completeness": int(row["completeness"]),
                    "safety_privacy": int(row["safety_privacy"]),
                    "overall_score": float(row.get("overall_score", 0.0)),
                }
    elif HUMAN_SCORES_JSONL.exists():
        with open(HUMAN_SCORES_JSONL, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    scores[int(r["thread_id"])] = r
    return scores


def load_llm_judge_scores() -> Dict[int, Dict[str, Any]]:
    """Loads LLM judge scores from golden_judged.jsonl."""
    scores = {}
    if JUDGED_RESULTS_PATH.exists():
        with open(JUDGED_RESULTS_PATH, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    tid = int(r["thread_id"])
                    scores[tid] = {
                        "tone_empathy": int(r["tone_empathy"]["score"]),
                        "factual_correctness": int(r["factual_correctness"]["score"]),
                        "completeness": int(r["completeness"]["score"]),
                        "safety_privacy": int(r["safety_privacy"]["score"]),
                        "overall_score": float(r["overall_score"]),
                        "overall_flag": r.get("overall_flag", "needs_review"),
                    }
    return scores


def compute_metrics_for_dimension(
    human_vals: List[float], judge_vals: List[float]
) -> Dict[str, float]:
    """Computes Spearman, Pearson, Exact Match, Adjacent Match, and MAE."""
    n = len(human_vals)
    if n == 0:
        return {"spearman": 0.0, "pearson": 0.0, "exact_match": 0.0, "adjacent_match": 0.0, "mae": 0.0}

    # Spearman rank correlation
    spearman_res = stats.spearmanr(human_vals, judge_vals)
    spearman_rho = spearman_res.statistic if not math.isnan(spearman_res.statistic) else 0.0
    spearman_p = spearman_res.pvalue if not math.isnan(spearman_res.pvalue) else 1.0

    # Pearson correlation
    pearson_res = stats.pearsonr(human_vals, judge_vals)
    pearson_r = pearson_res.statistic if not math.isnan(pearson_res.statistic) else 0.0

    exact_matches = sum(1 for h, j in zip(human_vals, judge_vals) if round(h) == round(j))
    adjacent_matches = sum(1 for h, j in zip(human_vals, judge_vals) if abs(round(h) - round(j)) <= 1)
    mae = sum(abs(h - j) for h, j in zip(human_vals, judge_vals)) / n

    return {
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,
        "pearson_r": pearson_r,
        "exact_match_pct": (exact_matches / n) * 100.0,
        "adjacent_match_pct": (adjacent_matches / n) * 100.0,
        "mae": mae,
    }


def main():
    human_map = load_human_scores()
    judge_map = load_llm_judge_scores()

    common_ids = sorted(set(human_map.keys()).intersection(set(judge_map.keys())))

    sep = "=" * 88
    print(f"\n{sep}")
    print(f"       HUMAN-JUDGE AGREEMENT & CORRELATION ANALYSIS (Sampled N={len(common_ids)} / 30)")
    print(f"{sep}")

    if len(common_ids) < 5:
        print(f"\n[!] Only {len(common_ids)} threads with overlapping human and judge scores found.")
        print("Please complete human labeling in `streamlit run src/human_judge_app.py` or run `src/judge_replies.py`.")
        print(sep + "\n")
        return

    dimensions = [
        ("Tone & Empathy", "tone_empathy"),
        ("Factual Correctness", "factual_correctness"),
        ("Completeness", "completeness"),
        ("Safety & Privacy", "safety_privacy"),
        ("Composite Score", "overall_score"),
    ]

    header = f"{'Evaluation Dimension':<24} | {'Spearman rho':<14} | {'Pearson r':<11} | {'Exact Match':<13} | {'Off<=1 Match':<13} | {'MAE':<6}"
    print("\n" + header)
    print("-" * len(header))

    for label, key in dimensions:
        h_vals = [human_map[tid][key] for tid in common_ids]
        j_vals = [judge_map[tid][key] for tid in common_ids]
        m = compute_metrics_for_dimension(h_vals, j_vals)

        rho_str = f"{m['spearman_rho']:.3f} (p={m['spearman_p']:.3f})" if m['spearman_p'] < 0.05 else f"{m['spearman_rho']:.3f} (n.s.)"
        print(
            f"{label:<24} | {rho_str:<14} | {m['pearson_r']:<11.3f} | {m['exact_match_pct']:>5.1f}%       | {m['adjacent_match_pct']:>5.1f}%       | {m['mae']:.2f}"
        )

    print("-" * len(header))
    print(
        "\n* Metric Justification: Spearman's rho is the headline correlation metric because\n"
        "  1-5 Likert ratings are ordinal non-interval measures with non-normal distributions.\n"
        "  Adjacent agreement (Off<=1) measures standard inter-rater grading tolerance."
    )
    print(sep + "\n")


if __name__ == "__main__":
    main()
