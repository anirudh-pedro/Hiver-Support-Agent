"""
src/route_baselines.py

Evaluates routing baselines on the 220-thread golden set:
  1. Trivial Baseline: Predicts 'auto_handle' for all threads (majority class).
  2. Simple Baseline (Rules-Only): Runs ONLY the 5 deterministic safety rules
     (rule_security, rule_financial_action, rule_pii_exposure, rule_repeated_failure,
     rule_legal_or_churn_threat) with no LLM judgment and no low-confidence override.
  3. Full Hybrid Policy: The 3-layer architecture (Deterministic rules +
     Low-confidence override + LLM judgment).

Outputs side-by-side comparison of accuracy, precision, recall, F1, and confusion matrices.
"""

import csv
import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.route import (
    _RULES,
    rule_financial_action,
    rule_legal_or_churn_threat,
    rule_pii_exposure,
    rule_repeated_failure,
    rule_security,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

GOLDEN_CSV_PATH = REPO_ROOT / "data" / "golden" / "golden_set_working.csv"
DRAFTS_PATH = REPO_ROOT / "data" / "results" / "golden_drafts.jsonl"
ROUTING_RESULTS_PATH = REPO_ROOT / "data" / "results" / "golden_routing.jsonl"


@dataclass
class Metrics:
    name: str
    total: int
    correct: int
    accuracy: float
    tp: int
    fp: int
    fn: int
    tn: int
    precision: float
    recall: float
    f1: float
    specificity: float  # auto_handle recall


def calculate_metrics(name: str, predictions: List[str], gold_labels: List[str]) -> Metrics:
    """Computes binary classification metrics treating 'escalate' as positive class."""
    assert len(predictions) == len(gold_labels), "Length mismatch"
    total = len(gold_labels)
    correct = sum(1 for p, g in zip(predictions, gold_labels) if p == g)
    accuracy = correct / total if total > 0 else 0.0

    tp = sum(1 for p, g in zip(predictions, gold_labels) if p == "escalate" and g == "escalate")
    fp = sum(1 for p, g in zip(predictions, gold_labels) if p == "escalate" and g == "auto_handle")
    fn = sum(1 for p, g in zip(predictions, gold_labels) if p == "auto_handle" and g == "escalate")
    tn = sum(1 for p, g in zip(predictions, gold_labels) if p == "auto_handle" and g == "auto_handle")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return Metrics(
        name=name,
        total=total,
        correct=correct,
        accuracy=accuracy,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        precision=precision,
        recall=recall,
        f1=f1,
        specificity=specificity,
    )


def load_golden_threads() -> List[Dict[str, Any]]:
    """Loads threads aligned with gold labels and predicted intents."""
    # 1. Load gold labels and thread metadata from CSV
    meta_by_id = {}
    with open(GOLDEN_CSV_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t_id = int(row["thread_id"])
            meta_by_id[t_id] = {
                "thread_id": t_id,
                "gold_escalate": row.get("my_escalate", "").strip(),
                "gold_intent": row.get("my_intent", "").strip(),
                "n_turns": int(row.get("n_turns", 1)),
                "full_thread_text": row.get("full_thread_text", ""),
            }

    # 2. Load predicted intents from drafts
    drafts_by_id = {}
    if DRAFTS_PATH.exists():
        with open(DRAFTS_PATH, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    drafts_by_id[int(r["thread_id"])] = r

    threads = []
    for t_id, meta in meta_by_id.items():
        draft = drafts_by_id.get(t_id, {})
        pred_intent = draft.get("predicted_intent", meta["gold_intent"])
        threads.append({
            "thread_id": t_id,
            "full_thread_text": draft.get("full_thread_text", meta["full_thread_text"]),
            "turns": [],
            "n_turns": meta["n_turns"],
            "gold_escalate": meta["gold_escalate"],
            "predicted_intent": pred_intent,
        })
    return threads


def run_trivial_baseline(threads: List[Dict[str, Any]]) -> List[str]:
    """Baseline 1: Always predict 'auto_handle' (majority class)."""
    return ["auto_handle" for _ in threads]


def run_naive_heuristic_baseline(threads: List[Dict[str, Any]], min_turns: int = 3) -> List[str]:
    """
    Baseline 2 (Simple Non-ML Heuristic): Escalate if thread has >= min_turns,
    otherwise auto_handle. Represents what a simple rules engineer might deploy first.
    """
    preds = []
    for t in threads:
        n_turns = int(t.get("n_turns", 1))
        preds.append("escalate" if n_turns >= min_turns else "auto_handle")
    return preds


def run_rules_only_baseline(threads: List[Dict[str, Any]]) -> Tuple[List[str], Counter]:
    """
    Baseline 3 (Ablation: 5 Deterministic Rules Only): Run ONLY deterministic rules from route.py.
    If no rule fires, default to 'auto_handle'.
    (No LLM judgment layer and no low-confidence override).
    """
    preds = []
    rule_counts = Counter()
    for t in threads:
        matched_rule = None
        decision = "auto_handle"
        for rule_name, rule_fn in _RULES:
            res = rule_fn(t["predicted_intent"], t)
            if res is not None:
                escalate_flag, _ = res
                decision = "escalate" if escalate_flag else "auto_handle"
                matched_rule = rule_name
                break
        if matched_rule:
            rule_counts[matched_rule] += 1
        preds.append(decision)
    return preds, rule_counts


def load_hybrid_predictions(threads: List[Dict[str, Any]]) -> List[str]:
    """Loads the actual full hybrid policy predictions from golden_routing.jsonl."""
    preds_by_id = {}
    with open(ROUTING_RESULTS_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                preds_by_id[int(r["thread_id"])] = r["decision"]

    preds = []
    for t in threads:
        tid = t["thread_id"]
        preds.append(preds_by_id.get(tid, "auto_handle"))
    return preds


def print_comparison_table(metrics_list: List[Metrics], rule_counts: Counter) -> None:
    sep = "=" * 98
    print(f"\n{sep}")
    print("           ROUTING POLICY EVALUATION: 4-TIER PROGRESSION VS. FULL HYBRID SYSTEM")
    print(f"{sep}")

    print("\n-- Metrics Summary Table (Positive Class = 'escalate') ---------------------------------------")
    header = f"{'Metric / Dimension':<26} | {'1. Trivial (Majority)':<20} | {'2. Naive (Turns>=3)':<19} | {'3. Rules Ablation':<18} | {'4. Full Hybrid':<16}"
    print(header)
    print("-" * len(header))

    def row_fmt(label: str, v1: str, v2: str, v3: str, v4: str) -> str:
        return f"{label:<26} | {v1:<20} | {v2:<19} | {v3:<18} | {v4:<16}"

    m1, m2, m3, m4 = metrics_list[0], metrics_list[1], metrics_list[2], metrics_list[3]

    print(row_fmt("Architecture", "All 'auto_handle'", "Turns >= 3 heuristic", "5 Rules + Default", "Rules + LowConf + LLM"))
    print(row_fmt("Evaluated (N)", str(m1.total), str(m2.total), str(m3.total), str(m4.total)))
    print(row_fmt("Accuracy", f"{m1.accuracy*100:.1f}% ({m1.correct}/{m1.total})", f"{m2.accuracy*100:.1f}% ({m2.correct}/{m2.total})", f"{m3.accuracy*100:.1f}% ({m3.correct}/{m3.total})", f"{m4.accuracy*100:.1f}% ({m4.correct}/{m4.total})"))
    print(row_fmt("Precision (escalate)", f"{m1.precision:.3f}", f"{m2.precision:.3f}", f"{m3.precision:.3f}", f"{m4.precision:.3f}"))
    print(row_fmt("Recall (escalate)", f"{m1.recall:.3f} ({m1.tp}/{m1.tp+m1.fn})", f"{m2.recall:.3f} ({m2.tp}/{m2.tp+m2.fn})", f"{m3.recall:.3f} ({m3.tp}/{m3.tp+m3.fn})", f"{m4.recall:.3f} ({m4.tp}/{m4.tp+m4.fn})"))
    print(row_fmt("F1 Score (escalate)", f"{m1.f1:.3f}", f"{m2.f1:.3f}", f"{m3.f1:.3f}", f"{m4.f1:.3f}"))
    print(row_fmt("False Negatives (Misses)", f"{m1.fn} ({m1.fn/73*100:.1f}%)", f"{m2.fn} ({m2.fn/73*100:.1f}%)", f"{m3.fn} ({m3.fn/73*100:.1f}%)", f"{m4.fn} ({m4.fn/73*100:.1f}%)"))
    print(row_fmt("False Positives (Esc)", f"{m1.fp}", f"{m2.fp}", f"{m3.fp}", f"{m4.fp}"))
    print(row_fmt("Specificity (auto recall)", f"{m1.specificity*100:.1f}%", f"{m2.specificity*100:.1f}%", f"{m3.specificity*100:.1f}%", f"{m4.specificity*100:.1f}%"))
    print("-" * len(header))

    print("\n-- Confusion Matrices ------------------------------------------------------------------------")
    print("1. TRIVIAL BASELINE (All auto_handle):")
    print(f"                       Pred: auto_handle    Pred: escalate")
    print(f"   Gold: auto_handle         {m1.tn:>4} (TN)              {m1.fp:>4} (FP)")
    print(f"   Gold: escalate            {m1.fn:>4} (FN)              {m1.tp:>4} (TP)")

    print("\n2. NAIVE SINGLE-FEATURE HEURISTIC (Thread Turns >= 3):")
    print(f"                       Pred: auto_handle    Pred: escalate")
    print(f"   Gold: auto_handle         {m2.tn:>4} (TN)              {m2.fp:>4} (FP)")
    print(f"   Gold: escalate            {m2.fn:>4} (FN)              {m2.tp:>4} (TP)")

    print("\n3. RULES-ONLY ABLATION (5 Deterministic Rules):")
    print(f"                       Pred: auto_handle    Pred: escalate")
    print(f"   Gold: auto_handle         {m3.tn:>4} (TN)              {m3.fp:>4} (FP)")
    print(f"   Gold: escalate            {m3.fn:>4} (FN)              {m3.tp:>4} (TP)")

    print("\n4. FULL HYBRID SYSTEM (Rules + Low-Conf Override + LLM):")
    print(f"                       Pred: auto_handle    Pred: escalate")
    print(f"   Gold: auto_handle         {m4.tn:>4} (TN)              {m4.fp:>4} (FP)")
    print(f"   Gold: escalate            {m4.fn:>4} (FN)              {m4.tp:>4} (TP)")

    print(f"\n{sep}\n")


def main() -> None:
    threads = load_golden_threads()
    gold_labels = [t["gold_escalate"] for t in threads]

    # 1. Trivial Baseline
    trivial_preds = run_trivial_baseline(threads)
    m_trivial = calculate_metrics("Trivial (Majority Class)", trivial_preds, gold_labels)

    # 2. Naive Single-Feature Heuristic Baseline
    naive_preds = run_naive_heuristic_baseline(threads, min_turns=3)
    m_naive = calculate_metrics("Naive Heuristic (Turns>=3)", naive_preds, gold_labels)

    # 3. Simple Rules-Only Ablation Baseline
    rules_preds, rule_counts = run_rules_only_baseline(threads)
    m_rules = calculate_metrics("Simple (Rules-Only)", rules_preds, gold_labels)

    # 4. Full Hybrid Policy
    hybrid_preds = load_hybrid_predictions(threads)
    m_hybrid = calculate_metrics("Full Hybrid System", hybrid_preds, gold_labels)

    print_comparison_table([m_trivial, m_naive, m_rules, m_hybrid], rule_counts)


if __name__ == "__main__":
    main()
