"""
src/run_golden_drafts.py

Runs the complete end-to-end grounded reply-drafting pipeline across all 220 golden evaluation threads:
  1. Classifies each thread with classify_intent(thread) (using its OWN predicted intent, NOT gold_intent).
  2. Retrieves top-5 historical precedents via FAISS (pure semantic similarity with macro deduplication).
  3. Drafts a grounded response via draft_reply().
  4. Saves results incrementally to data/results/golden_drafts.jsonl.
  5. Prints 10 random examples to console for manual quality inspection.
"""

import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set

import pandas as pd
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.draft_reply import DraftReply, draft_reply
from src.intents import IntentResult, classify_intent, get_llm_client, get_all_llm_clients
from src.retrieve import get_retriever, retrieve_similar

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

GOLDEN_CSV_PATH = REPO_ROOT / "data" / "golden" / "golden_set_working.csv"
THREADS_JSONL_PATH = REPO_ROOT / "data" / "processed" / "threads_amazonhelp.jsonl"
RESULTS_DIR = REPO_ROOT / "data" / "results"
DRAFTS_OUTPUT_PATH = RESULTS_DIR / "golden_drafts.jsonl"


def load_golden_threads_with_turns() -> List[Dict[str, Any]]:
    """
    Loads the 220 golden threads from golden_set_working.csv and enriches them
    with turns and the actual historical AmazonHelp reply from threads_amazonhelp.jsonl.
    """
    df_gold = pd.read_csv(GOLDEN_CSV_PATH)
    golden_ids = set(df_gold["thread_id"].astype(int))
    gold_meta_by_id = {
        int(row["thread_id"]): {
            "full_thread_text": str(row["full_thread_text"]),
            "my_intent": str(row.get("my_intent", "")),
        }
        for _, row in df_gold.iterrows()
    }

    logger.info("Scanning threads_amazonhelp.jsonl for %d golden thread records...", len(golden_ids))
    threads_by_id: Dict[int, Dict[str, Any]] = {}

    with open(THREADS_JSONL_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            t_id = int(item["thread_id"])
            if t_id in golden_ids:
                # Extract first AmazonHelp reply turn as the actual historical reply
                actual_reply = ""
                for turn in item.get("turns", []):
                    if turn.get("author_id") == "AmazonHelp":
                        actual_reply = turn.get("text", "").strip()
                        break

                threads_by_id[t_id] = {
                    "thread_id": t_id,
                    "turns": item.get("turns", []),
                    "full_thread_text": gold_meta_by_id[t_id]["full_thread_text"],
                    "actual_amazon_reply": actual_reply,
                }

    # Ensure all golden rows are present in original order
    ordered_threads: List[Dict[str, Any]] = []
    for _, row in df_gold.iterrows():
        t_id = int(row["thread_id"])
        if t_id in threads_by_id:
            ordered_threads.append(threads_by_id[t_id])
        else:
            # Fallback using CSV full_thread_text
            lines = str(row["full_thread_text"]).splitlines()
            first_reply = ""
            for l in lines:
                if l.startswith("[AmazonHelp]:"):
                    first_reply = l.replace("[AmazonHelp]:", "").strip()
                    break
            ordered_threads.append({
                "thread_id": t_id,
                "full_thread_text": str(row["full_thread_text"]),
                "actual_amazon_reply": first_reply,
            })

    logger.info("Loaded %d golden threads ready for pipeline execution.", len(ordered_threads))
    return ordered_threads


def run_pipeline_for_golden_set(threads: List[Dict[str, Any]], max_workers: int = 3) -> None:
    """
    Executes intent classification, retrieval, and reply drafting for all threads.
    Supports incremental resumption if partially completed.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Check existing results for resumption
    processed_ids: Set[int] = set()
    if DRAFTS_OUTPUT_PATH.exists():
        with open(DRAFTS_OUTPUT_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        processed_ids.add(int(record["thread_id"]))
                    except Exception:
                        pass
        logger.info("Found %d already processed threads in %s", len(processed_ids), DRAFTS_OUTPUT_PATH.name)

    remaining_threads = [t for t in threads if t["thread_id"] not in processed_ids]
    logger.info("Remaining threads to process: %d / %d", len(remaining_threads), len(threads))

    if not remaining_threads:
        logger.info("All 220 threads already drafted! Proceeding to display inspection sample.")
        return

    # 2. Pre-initialize resources
    logger.info("Initializing FAISS retriever...")
    retriever = get_retriever()
    retriever._load_resources()

    logger.info("Initializing LLM clients with multi-key failover...")
    all_clients = get_all_llm_clients()
    clients_pool = [c[0] for c in all_clients]
    logger.info("Loaded %d active LLM clients from environment.", len(clients_pool))

    # Model choice
    model_name = "openai/gpt-oss-20b"
    logger.info("Starting processing with model %s across %d clients...", model_name, len(clients_pool))
    start_time = time.time()

    with open(DRAFTS_OUTPUT_PATH, "a", encoding="utf-8") as f_out:
        for idx, thread in enumerate(remaining_threads, 1):
            t_id = thread["thread_id"]
            step_start = time.time()

            # Rotate clients so request load is balanced across keys
            client_candidates = [clients_pool[(idx + o) % len(clients_pool)] for o in range(len(clients_pool))]

            try:
                # Step A: Intent Classification
                intent_res = None
                for c in client_candidates:
                    try:
                        intent_res = classify_intent(thread, model=model_name, client=c, max_retries=5)
                        break
                    except Exception as e:
                        if "429" in str(e):
                            logger.warning("Classification 429 on client, failing over to next key...")
                            continue
                        raise

                if intent_res is None:
                    time.sleep(6.0)
                    intent_res = classify_intent(thread, model=model_name, client=client_candidates[0], max_retries=5)

                pred_intent = intent_res.intent

                # Space out requests by 2.0s to strictly stay under 30 RPM
                time.sleep(2.0)

                # Step B & C: Retrieval & Reply Drafting
                draft_res = None
                for c in client_candidates:
                    try:
                        draft_res = draft_reply(
                            thread=thread,
                            intent=pred_intent,
                            model=model_name,
                            client=c,
                            k=5,
                            max_retries=5,
                        )
                        break
                    except Exception as e:
                        if "429" in str(e):
                            logger.warning("Drafting 429 on client, failing over to next key...")
                            continue
                        raise

                if draft_res is None:
                    time.sleep(6.0)
                    draft_res = draft_reply(
                        thread=thread,
                        intent=pred_intent,
                        model=model_name,
                        client=client_candidates[0],
                        k=5,
                        max_retries=5,
                    )

                record = {
                    "thread_id": t_id,
                    "predicted_intent": pred_intent,
                    "intent_confidence": intent_res.confidence,
                    "intent_reasoning": intent_res.reasoning,
                    "retrieved_thread_ids": draft_res.grounding_thread_ids,
                    "drafted_reply": draft_res.reply_text,
                    "actual_amazon_reply": thread.get("actual_amazon_reply", ""),
                    "full_thread_text": thread.get("full_thread_text", ""),
                }

                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_out.flush()

                elapsed = time.time() - step_start
                total_done = len(processed_ids) + idx
                logger.info(
                    "[%d/%d] Thread #%d -> intent=%s (conf=%.2f) in %.1fs",
                    total_done,
                    len(threads),
                    t_id,
                    pred_intent,
                    intent_res.confidence,
                    elapsed,
                )

                # Space out between threads to maintain safe rate
                time.sleep(2.0)

            except Exception as exc:
                logger.error("Error processing thread #%d: %s", t_id, exc)
                # Fallback record
                record = {
                    "thread_id": t_id,
                    "predicted_intent": "other",
                    "intent_confidence": 0.0,
                    "intent_reasoning": f"Exception: {exc}",
                    "retrieved_thread_ids": [],
                    "drafted_reply": "I'm sorry for the inconvenience! Please contact us via DM so we can look into this for you. ^AH",
                    "actual_amazon_reply": thread.get("actual_amazon_reply", ""),
                    "full_thread_text": thread.get("full_thread_text", ""),
                }
                f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                f_out.flush()

    total_duration = time.time() - start_time
    logger.info("Finished drafting all remaining threads in %.2fs (%.2f min).", total_duration, total_duration / 60)


def display_sample_drafts(sample_size: int = 10, seed: int = 42) -> None:
    """
    Randomly selects 10 examples from golden_drafts.jsonl and prints them to console:
      - Full customer thread
      - Predicted intent
      - Actual historical AmazonHelp reply from the same thread
      - Drafted reply
    """
    if not DRAFTS_OUTPUT_PATH.exists():
        print(f"Error: {DRAFTS_OUTPUT_PATH} not found.")
        return

    records: List[Dict[str, Any]] = []
    with open(DRAFTS_OUTPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if not records:
        print("No records found in golden_drafts.jsonl.")
        return

    rng = random.Random(seed)
    chosen = rng.sample(records, min(sample_size, len(records)))

    print("\n" + "=" * 95)
    print(f"SPOT-CHECK INSPECTION: {len(chosen)} RANDOM EXAMPLES FROM GOLDEN DRAFTS (Seed={seed})")
    print("=" * 95)

    for i, rec in enumerate(chosen, 1):
        print(f"\n[{i:02d}/10] THREAD #{rec['thread_id']} | PREDICTED INTENT: [{rec['predicted_intent'].upper()}] (Conf: {rec.get('intent_confidence', 1.0):.2f})")
        print("-" * 95)
        print("FULL CONVERSATION THREAD:")
        print(rec.get("full_thread_text", "").strip())
        print("-" * 95)
        print("ACTUAL HISTORICAL AMAZONHELP REPLY (Ground Truth Brand Response):")
        print(rec.get("actual_amazon_reply", "[No historical reply found]").strip())
        print("-" * 95)
        print("DRAFTED REPLY (Model Generated with Grounding):")
        print(rec.get("drafted_reply", "").strip())
        print(f"Grounding Precedent Threads: {rec.get('retrieved_thread_ids', [])}")
        print("=" * 95)


if __name__ == "__main__":
    threads = load_golden_threads_with_turns()
    run_pipeline_for_golden_set(threads)
    display_sample_drafts(sample_size=10, seed=42)
