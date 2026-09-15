"""
src/run_golden_routing.py

Runs the hybrid routing policy over all 220 golden evaluation threads
and saves results to data/results/golden_routing.jsonl.

Evaluation uses PREDICTED intent (not gold_intent) — this is an end-to-end
evaluation that measures the full pipeline's accuracy, not an oracle.

Gold labels come from the 'my_escalate' column in golden_set_working.csv
(the human annotations stored as my_* columns by the labeling app).
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.intents import get_all_llm_clients
from src.route import ROUTING_LLM_MODEL, LOW_CONFIDENCE_THRESHOLD, route_thread

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

GOLDEN_CSV_PATH = REPO_ROOT / "data" / "golden" / "golden_set_working.csv"
DRAFTS_PATH = REPO_ROOT / "data" / "results" / "golden_drafts.jsonl"
OUTPUT_PATH = REPO_ROOT / "data" / "results" / "golden_routing.jsonl"


def load_golden_meta() -> Dict[int, Dict[str, str]]:
    """Load gold labels (my_escalate, my_intent) keyed by thread_id."""
    meta = {}
    with open(GOLDEN_CSV_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t_id = int(row["thread_id"])
            meta[t_id] = {
                "gold_escalate": row.get("my_escalate", "").strip(),
                "gold_intent":   row.get("my_intent", "").strip(),
                "n_turns":       row.get("n_turns", "1"),
                "full_thread_text": row.get("full_thread_text", ""),
            }
    return meta


def load_drafts() -> List[Dict[str, Any]]:
    """Load the golden_drafts.jsonl (contains predicted_intent + intent_confidence)."""
    records = []
    with open(DRAFTS_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Run routing over golden set.")
    parser.add_argument("--rules-only", action="store_true", help="Run deterministic rules only (defer LLM)")
    parser.add_argument("--clean", action="store_true", help="Clean/overwrite existing results")
    parser.add_argument("--model", type=str, default=ROUTING_LLM_MODEL, help="Model for LLM routing")
    parser.add_argument("--sleep", type=float, default=1.5, help="Sleep time between LLM calls in seconds")
    args = parser.parse_args()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if args.clean and OUTPUT_PATH.exists():
        logger.info("Cleaning existing output file: %s", OUTPUT_PATH)
        OUTPUT_PATH.unlink()

    # Load existing results into a map
    existing_records: Dict[int, Dict[str, Any]] = {}
    done_ids: set = set()
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        t_id = int(rec["thread_id"])
                        existing_records[t_id] = rec
                        # If running full mode, we do NOT treat 'deferred_llm' or 'error' or 'rule_llm_failure_fallback' as done
                        if not args.rules_only and (
                            rec.get("source") in ("deferred_llm", "error")
                            or rec.get("triggered_rule") == "rule_llm_failure_fallback"
                        ):
                            continue
                        done_ids.add(t_id)
                    except Exception:
                        pass
        logger.info("Resuming: %d threads already reliably routed (%d total in file).", len(done_ids), len(existing_records))

    gold_meta = load_golden_meta()
    draft_records = load_drafts()

    # Build thread dicts from draft records, enriched with n_turns from CSV
    threads_to_route = []
    for rec in draft_records:
        t_id = int(rec["thread_id"])
        if t_id in done_ids:
            continue
        meta = gold_meta.get(t_id, {})
        thread = {
            "thread_id": t_id,
            "full_thread_text": rec.get("full_thread_text", meta.get("full_thread_text", "")),
            "turns": [],  # turns not needed; we use full_thread_text via _thread_text()
            "n_turns": int(meta.get("n_turns", 1)),
        }
        threads_to_route.append((
            thread,
            rec.get("predicted_intent", "other"),
            float(rec.get("intent_confidence", 0.5)),
            meta.get("gold_escalate", ""),
        ))

    logger.info(
        "Routing %d threads (model=%s, low_conf_threshold=%.2f, rules_only=%s) ...",
        len(threads_to_route), args.model, LOW_CONFIDENCE_THRESHOLD, args.rules_only,
    )

    # Prepare LLM client pool
    all_clients = get_all_llm_clients()
    clients = [c[0] for c in all_clients]
    logger.info("LLM clients available: %d", len(clients))

    def save_all_records():
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f_out:
            for rec_id in sorted(existing_records.keys()):
                f_out.write(json.dumps(existing_records[rec_id], ensure_ascii=False) + "\n")

    for idx, (thread, pred_intent, conf, gold_label) in enumerate(threads_to_route, 1):
        t_id = thread["thread_id"]
        client = clients[idx % len(clients)] if clients else None

        try:
            decision = route_thread(
                thread=thread,
                intent=pred_intent,
                intent_confidence=conf,
                client=client,
                model=args.model,
                low_confidence_threshold=LOW_CONFIDENCE_THRESHOLD,
                rules_only=args.rules_only,
            )
            record = {
                "thread_id": t_id,
                "predicted_intent": pred_intent,
                "intent_confidence": conf,
                "gold_escalate": gold_label,
                "decision": decision.decision,
                "reason": decision.reason,
                "triggered_rule": decision.triggered_rule,
                "confidence": decision.confidence,
                "source": decision.source,
                "model": decision.model,
                "correct": (decision.decision == gold_label) if gold_label else None,
            }
            existing_records[t_id] = record
            save_all_records()

            total_done = len(done_ids) + idx
            logger.info(
                "[%d/%d] #%d -> %s (src=%s, rule=%s, model=%s) | gold=%s correct=%s",
                total_done, len(gold_meta),
                t_id, decision.decision, decision.source,
                decision.triggered_rule or "-",
                decision.model,
                gold_label, record["correct"],
            )

        except Exception as exc:
            logger.error("Error routing thread #%d: %s", t_id, exc)
            record = {
                "thread_id": t_id,
                "predicted_intent": pred_intent,
                "intent_confidence": conf,
                "gold_escalate": gold_label,
                "decision": "escalate",  # safe default on error
                "reason": f"Routing pipeline error: {exc}",
                "triggered_rule": "rule_pipeline_error",
                "confidence": 0.0,
                "source": "error",
                "model": "error",
                "correct": ("escalate" == gold_label) if gold_label else None,
            }
            existing_records[t_id] = record
            save_all_records()

        # Brief pause between LLM calls to stay under rate limits
        if decision.source == "llm" and args.sleep > 0:
            time.sleep(args.sleep)

    # ── Print summary ─────────────────────────────────────────────────────────
    print_summary()


def print_summary() -> None:
    """Load golden_routing.jsonl and print a full summary."""
    if not OUTPUT_PATH.exists():
        logger.error("No output file found at %s", OUTPUT_PATH)
        return

    records = []
    with open(OUTPUT_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if not records:
        print("No records to summarize.")
        return

    total = len(records)
    decisions = Counter(r["decision"] for r in records)
    sources = Counter(r["source"] for r in records)
    models = Counter(r.get("model", r["source"]) for r in records)
    rule_counts = Counter(
        r["triggered_rule"]
        for r in records
        if r.get("triggered_rule") and r["source"] in ("rule", "low_confidence")
    )

    # Accuracy (only over records with a valid gold label)
    labeled = [r for r in records if r.get("gold_escalate") in ("auto_handle", "escalate")]
    correct = sum(1 for r in labeled if r.get("correct"))

    # Confusion matrix
    tp = sum(1 for r in labeled if r["decision"] == "escalate" and r["gold_escalate"] == "escalate")
    fp = sum(1 for r in labeled if r["decision"] == "escalate" and r["gold_escalate"] == "auto_handle")
    fn = sum(1 for r in labeled if r["decision"] == "auto_handle" and r["gold_escalate"] == "escalate")
    tn = sum(1 for r in labeled if r["decision"] == "auto_handle" and r["gold_escalate"] == "auto_handle")

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    sep = "=" * 65

    print(f"\n{sep}")
    print(f"  ROUTING SUMMARY  (N={total}, gold-labeled={len(labeled)})")
    print(sep)

    print("\n-- Decision Split ------------------------------------------")
    for decision, cnt in decisions.most_common():
        print(f"  {decision:<16}: {cnt:>4}  ({100*cnt/total:.1f}%)")

    print("\n-- Decision Source -----------------------------------------")
    for src, cnt in sources.most_common():
        print(f"  {src:<20}: {cnt:>4}  ({100*cnt/total:.1f}%)")

    print("\n-- Model / Attribution Breakdown ---------------------------")
    for mdl, cnt in models.most_common():
        print(f"  {mdl:<30}: {cnt:>4}  ({100*cnt/total:.1f}%)")

    print("\n-- Rule Firing Counts --------------------------------------")
    if rule_counts:
        for rule, cnt in rule_counts.most_common():
            print(f"  {rule:<35}: {cnt:>4}")
    else:
        print("  (no deterministic rules fired)")

    print(f"\n-- Raw Accuracy vs. gold_escalate (N={len(labeled)}) ------")
    print(f"  Correct: {correct} / {len(labeled)}  ({100*correct/len(labeled):.1f}%)")

    print("\n-- Confusion Matrix (escalate = positive class) ------------")
    print(f"  {'':20s}  Pred: auto  Pred: escalate")
    print(f"  Gold: auto_handle    {tn:>8}  {fp:>14}")
    print(f"  Gold: escalate       {fn:>8}  {tp:>14}")

    print("\n-- Metrics -------------------------------------------------")
    print(f"  Precision (escalate): {precision:.3f}")
    print(f"  Recall    (escalate): {recall:.3f}")
    print(f"  F1        (escalate): {f1:.3f}")
    print(sep)


if __name__ == "__main__":
    main()
