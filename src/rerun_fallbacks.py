"""
src/rerun_fallbacks.py

Re-runs the 18 fallback threads from golden_drafts.jsonl that received
the canned fallback reply due to rate-limit exhaustion during the main run.

Strategy:
  - Uses llama-3.1-8b-instant (small, fast, lower token cost) as primary.
  - Processes one thread at a time with SLEEP_BETWEEN seconds gap.
  - Rotates both API keys on every 429.
  - Updates records in-place so the file stays at exactly 220 records.
"""
import json
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.draft_reply import draft_reply
from src.intents import classify_intent, get_all_llm_clients
from src.retrieve import get_retriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DRAFTS_PATH = REPO_ROOT / "data" / "results" / "golden_drafts.jsonl"
THREADS_JSONL = REPO_ROOT / "data" / "processed" / "threads_amazonhelp.jsonl"
GOLDEN_CSV = REPO_ROOT / "data" / "golden" / "golden_set_working.csv"
FALLBACK_MARKER = "sorry for the inconvenience! Please contact us via DM"
PRIMARY_MODEL = "openai/gpt-oss-20b"
SLEEP_BETWEEN = 12.0   # seconds between threads
SLEEP_ON_429 = 15.0    # seconds to wait after a rate-limit hit


def load_all_records():
    with open(DRAFTS_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_all_records(records):
    with open(DRAFTS_PATH, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_thread_data(target_ids):
    import pandas as pd

    df = pd.read_csv(GOLDEN_CSV)
    gold_meta = {
        int(row["thread_id"]): str(row["full_thread_text"])
        for _, row in df.iterrows()
    }
    threads = {}
    with open(THREADS_JSONL, encoding="utf-8") as f:
        for raw_line in f:
            if not raw_line.strip():
                continue
            item = json.loads(raw_line)
            t_id = int(item["thread_id"])
            if t_id not in target_ids:
                continue
            actual_reply = ""
            for turn in item.get("turns", []):
                if turn.get("author_id") == "AmazonHelp":
                    actual_reply = turn.get("text", "").strip()
                    break
            threads[t_id] = {
                "thread_id": t_id,
                "turns": item.get("turns", []),
                "full_thread_text": gold_meta.get(t_id, ""),
                "actual_amazon_reply": actual_reply,
            }
    # Fallback for any IDs not found in the JSONL corpus
    for t_id in target_ids:
        if t_id not in threads:
            threads[t_id] = {
                "thread_id": t_id,
                "turns": [],
                "full_thread_text": gold_meta.get(t_id, ""),
                "actual_amazon_reply": "",
            }
    return threads


def main():
    records = load_all_records()
    fallback_ids = {
        r["thread_id"]
        for r in records
        if FALLBACK_MARKER in r.get("drafted_reply", "")
    }
    logger.info("Found %d fallback records to re-run: %s", len(fallback_ids), sorted(fallback_ids))
    if not fallback_ids:
        logger.info("No fallbacks found. Nothing to do.")
        return

    logger.info("Loading thread data from JSONL corpus...")
    thread_data = load_thread_data(fallback_ids)
    logger.info("Loading FAISS retriever (this may take a moment)...")
    retriever = get_retriever()
    retriever._load_resources()
    logger.info("Retriever ready.")

    all_clients = get_all_llm_clients()
    clients = [c[0] for c in all_clients]
    logger.info("LLM clients available: %d", len(clients))

    id_to_idx = {r["thread_id"]: i for i, r in enumerate(records)}
    success_count = 0
    fail_count = 0

    for run_num, t_id in enumerate(sorted(fallback_ids), 1):
        thread = thread_data[t_id]
        logger.info("[%d/%d] Thread #%d ...", run_num, len(fallback_ids), t_id)

        # Rotate client order per thread for load balancing
        n = len(clients)
        ordered = [clients[run_num % n]] + [clients[i % n] for i in range(1, n)]

        # ── Step A: Intent classification ─────────────────────────────────────
        intent_res = None
        for attempt in range(5):
            for client in ordered:
                try:
                    intent_res = classify_intent(thread, model=PRIMARY_MODEL, client=client, max_retries=3)
                    break
                except Exception as exc:
                    wait = SLEEP_ON_429 if "429" in str(exc) else 3
                    logger.warning("  classify err (att %d): %s  sleep=%.0fs", attempt + 1, exc, wait)
                    time.sleep(wait)
            if intent_res is not None:
                break
            time.sleep(5)

        if intent_res is None:
            logger.error("  classify FAILED for thread #%d — skipping.", t_id)
            fail_count += 1
            time.sleep(SLEEP_BETWEEN)
            continue

        pred_intent = intent_res.intent
        logger.info("  intent=%s (conf=%.2f)", pred_intent, intent_res.confidence)
        time.sleep(4.0)  # breathing room between classify and draft

        # ── Step B: Reply drafting ─────────────────────────────────────────────
        draft_res = None
        for attempt in range(5):
            for client in ordered:
                try:
                    draft_res = draft_reply(
                        thread=thread,
                        intent=pred_intent,
                        model=PRIMARY_MODEL,
                        client=client,
                        k=5,
                        max_retries=3,
                    )
                    break
                except Exception as exc:
                    wait = SLEEP_ON_429 if "429" in str(exc) else 3
                    logger.warning("  draft err (att %d): %s  sleep=%.0fs", attempt + 1, exc, wait)
                    time.sleep(wait)
            if draft_res is not None and FALLBACK_MARKER not in draft_res.reply_text:
                break
            time.sleep(5)

        if draft_res is None or FALLBACK_MARKER in draft_res.reply_text:
            logger.error("  draft still returned fallback for #%d — skipping.", t_id)
            fail_count += 1
            time.sleep(SLEEP_BETWEEN)
            continue

        # ── Update record in-place ─────────────────────────────────────────────
        new_record = {
            "thread_id": t_id,
            "predicted_intent": pred_intent,
            "intent_confidence": intent_res.confidence,
            "intent_reasoning": intent_res.reasoning,
            "retrieved_thread_ids": draft_res.grounding_thread_ids,
            "drafted_reply": draft_res.reply_text,
            "actual_amazon_reply": thread.get("actual_amazon_reply", ""),
            "full_thread_text": thread.get("full_thread_text", ""),
        }
        records[id_to_idx[t_id]] = new_record
        save_all_records(records)

        preview = draft_res.reply_text[:90].replace("\n", " ")
        logger.info("  OK thread #%d: %s", t_id, preview)
        success_count += 1

        logger.info("  Sleeping %.0fs before next thread...", SLEEP_BETWEEN)
        time.sleep(SLEEP_BETWEEN)

    logger.info("=" * 60)
    logger.info("Re-run complete: %d succeeded, %d still failed.", success_count, fail_count)

    final_records = load_all_records()
    remaining = sum(1 for r in final_records if FALLBACK_MARKER in r.get("drafted_reply", ""))
    logger.info("Remaining fallbacks in file: %d / %d", remaining, len(final_records))


if __name__ == "__main__":
    main()
