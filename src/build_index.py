"""
src/build_index.py

Builds the FAISS semantic retrieval index over historical AmazonHelp customer-brand interactions.
Grounds reply drafting in real, historically resolved support conversations.

Leakage Prevention:
Strictly excludes all thread_ids present in:
  1. The 300-sample taxonomy validation set (data/processed/taxonomy_validation_sample.jsonl)
  2. The 60-row human spot-check set (data/processed/taxonomy_spotcheck.csv)
  3. The 220-row golden evaluation set (data/golden/golden_set_working.csv / data/golden/golden_set_template.csv)
Total excluded threads: 520.

Resolution Extraction & Known Limitation:
- customer_message: First customer turn text in the thread.
- brand_resolution: LAST AmazonHelp turn text in the thread.
- KNOWN LIMITATION: In public social-support datasets (like Twitter Customer Support),
  many complete resolutions take place off-Twitter via Private Messages / Direct Messages (DM).
  Therefore, the last public AmazonHelp turn often includes routing instructions (e.g.,
  "Please send us a DM / visit link so we can look into your account details"). However,
  it still provides the exact brand voice, recommended immediate action, macro framing,
  and official Amazon policy resolution path.
"""

import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 256
CHUNK_SIZE = 5000  # Checkpoint every 5,000 embeddings to prevent data loss

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
GOLDEN_DIR = DATA_DIR / "golden"
INDEX_DIR = DATA_DIR / "index"
CHUNKS_DIR = INDEX_DIR / ".chunks"

THREADS_PATH = PROCESSED_DIR / "threads_amazonhelp.jsonl"
VAL_SAMPLE_PATH = PROCESSED_DIR / "taxonomy_validation_sample.jsonl"
SPOTCHECK_PATH = PROCESSED_DIR / "taxonomy_spotcheck.csv"
GOLDEN_WORKING_PATH = GOLDEN_DIR / "golden_set_working.csv"
GOLDEN_TEMPLATE_PATH = GOLDEN_DIR / "golden_set_template.csv"

FAISS_INDEX_PATH = INDEX_DIR / "amazonhelp.faiss"
INDEX_META_PATH = INDEX_DIR / "amazonhelp_meta.jsonl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_leakage_exclusion_ids() -> Set[int]:
    """
    Collects all thread_ids from the validation set, spot-check set, and golden evaluation set.
    These threads must never leak into the retrieval corpus.
    """
    excluded_ids: Set[int] = set()

    # 1. Validation sample (300 rows)
    if VAL_SAMPLE_PATH.exists():
        with open(VAL_SAMPLE_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    excluded_ids.add(int(item["thread_id"]))
        logger.info(f"Loaded validation IDs: {len(excluded_ids)} from {VAL_SAMPLE_PATH.name}")

    # 2. Spot-check sample (60 rows, subset of validation)
    if SPOTCHECK_PATH.exists():
        df_spot = pd.read_csv(SPOTCHECK_PATH)
        spot_ids = set(df_spot["thread_id"].astype(int))
        excluded_ids.update(spot_ids)
        logger.info(f"Verified spot-check IDs ({len(spot_ids)} rows) included in exclusions")

    # 3. Golden eval set (220 rows)
    golden_ids: Set[int] = set()
    for g_path in [GOLDEN_WORKING_PATH, GOLDEN_TEMPLATE_PATH]:
        if g_path.exists():
            df_gold = pd.read_csv(g_path)
            if "thread_id" in df_gold.columns:
                golden_ids.update(df_gold["thread_id"].astype(int))
    excluded_ids.update(golden_ids)
    logger.info(f"Loaded golden eval IDs: {len(golden_ids)} rows; total unique exclusions: {len(excluded_ids)}")

    return excluded_ids


def extract_thread_pair(thread_data: dict) -> Tuple[str, str]:
    """
    Extracts (customer_message, brand_resolution) from a thread:
    - customer_message: text of first customer turn
    - brand_resolution: text of LAST AmazonHelp turn

    Note: Many real resolutions conclude in private DMs or off-Twitter portal links.
    The last AmazonHelp turn represents the final public brand action/macro guidance.
    """
    turns = thread_data.get("turns", [])

    # First customer turn
    customer_message = ""
    for turn in turns:
        author = turn.get("author_id", "")
        inbound = turn.get("inbound", False)
        if author != "AmazonHelp" or inbound:
            customer_message = turn.get("text", "").strip()
            break

    # Last AmazonHelp turn
    brand_resolution = ""
    for turn in reversed(turns):
        author = turn.get("author_id", "")
        if author == "AmazonHelp":
            brand_resolution = turn.get("text", "").strip()
            break

    return customer_message, brand_resolution


def build_retrieval_index() -> Dict[str, any]:
    """
    Executes the full indexing pipeline with chunked checkpointing:
    1. Loads leakage exclusion set.
    2. Reads all AmazonHelp threads and filters out exclusions.
    3. Extracts (customer_message, brand_resolution) pairs.
    4. Computes normalized dense embeddings with SentenceTransformer in resumable chunks.
    5. Builds and saves FAISS IndexFlatIP (cosine similarity).
    6. Saves metadata JSONL mapping index position -> {thread_id, customer_message, brand_resolution}.
    """
    start_time = time.time()
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Leakage Exclusions
    excluded_ids = load_leakage_exclusion_ids()
    logger.info(f"Total thread IDs marked for leakage prevention: {len(excluded_ids)}")

    # Step 2: Extract pairs from threads
    logger.info(f"Scanning raw threads from {THREADS_PATH}...")
    corpus_pairs: List[dict] = []
    total_threads_read = 0
    excluded_count = 0
    skipped_empty = 0

    with open(THREADS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total_threads_read += 1
            thread = json.loads(line)
            thread_id = int(thread["thread_id"])

            if thread_id in excluded_ids:
                excluded_count += 1
                continue

            cust_msg, brand_res = extract_thread_pair(thread)
            if not cust_msg or not brand_res:
                skipped_empty += 1
                continue

            corpus_pairs.append({
                "thread_id": thread_id,
                "customer_message": cust_msg,
                "brand_resolution": brand_res,
            })

    total_eligible = len(corpus_pairs)
    logger.info(
        f"Read {total_threads_read} threads. Excluded {excluded_count} leakage threads. "
        f"Skipped {skipped_empty} malformed. Eligible corpus size: {total_eligible}"
    )

    # Step 3: Embed customer messages with chunked checkpointing
    customer_texts = [pair["customer_message"] for pair in corpus_pairs]
    num_chunks = (total_eligible + CHUNK_SIZE - 1) // CHUNK_SIZE
    logger.info(f"Embedding pipeline: {total_eligible} items in {num_chunks} chunks of ~{CHUNK_SIZE}")

    embedder = None
    chunk_files: List[Path] = []

    encode_start = time.time()
    for chunk_idx in range(num_chunks):
        c_start = chunk_idx * CHUNK_SIZE
        c_end = min(c_start + CHUNK_SIZE, total_eligible)
        chunk_file = CHUNKS_DIR / f"chunk_{chunk_idx:03d}_{c_start}_{c_end}.npy"
        chunk_files.append(chunk_file)

        if chunk_file.exists():
            logger.info(f"Chunk {chunk_idx + 1}/{num_chunks} [{c_start}:{c_end}] already cached at {chunk_file.name}")
            continue

        if embedder is None:
            logger.info(f"Loading embedding model: {MODEL_NAME}...")
            embedder = SentenceTransformer(MODEL_NAME)

        logger.info(f"Encoding chunk {chunk_idx + 1}/{num_chunks} ({c_end - c_start} texts)...")
        sub_texts = customer_texts[c_start:c_end]
        sub_emb = embedder.encode(
            sub_texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,  # Crucial for IndexFlatIP to equate to Cosine Similarity
            convert_to_numpy=True,
        )
        np.save(str(chunk_file), sub_emb)
        logger.info(f"Saved chunk {chunk_idx + 1}/{num_chunks} to {chunk_file.name}")

    encode_duration = time.time() - encode_start
    logger.info(f"All {num_chunks} chunks ready in {encode_duration:.2f}s")

    # Load and concatenate all chunk embeddings
    logger.info("Assembling all chunk embeddings...")
    all_embeddings_list = [np.load(str(cf)) for cf in chunk_files]
    embeddings_np = np.vstack(all_embeddings_list)
    embeddings_np = np.ascontiguousarray(embeddings_np, dtype=np.float32)

    dim = embeddings_np.shape[1]
    logger.info(f"Assembled embeddings array: shape {embeddings_np.shape}, dtype {embeddings_np.dtype}")

    # Step 4: Build FAISS IndexFlatIP
    logger.info(f"Building FAISS IndexFlatIP ({dim} dimensions)...")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings_np)
    logger.info(f"FAISS index built with {index.ntotal} vectors")

    # Step 5: Persist FAISS index and metadata
    logger.info(f"Saving FAISS index to {FAISS_INDEX_PATH}...")
    faiss.write_index(index, str(FAISS_INDEX_PATH))

    logger.info(f"Saving metadata to {INDEX_META_PATH}...")
    with open(INDEX_META_PATH, "w", encoding="utf-8") as f_meta:
        for idx, item in enumerate(corpus_pairs):
            meta_record = {
                "index_pos": idx,
                "thread_id": item["thread_id"],
                "customer_message": item["customer_message"],
                "brand_resolution": item["brand_resolution"],
            }
            f_meta.write(json.dumps(meta_record, ensure_ascii=False) + "\n")

    # Clean up chunk files after successful index build
    shutil.rmtree(CHUNKS_DIR, ignore_errors=True)
    logger.info("Cleaned up temporary chunk files")

    total_duration = time.time() - start_time
    logger.info(f"Index build complete in {total_duration:.2f}s! ({total_duration / 60:.2f} min)")

    return {
        "total_threads_read": total_threads_read,
        "leakage_excluded_count": excluded_count,
        "indexed_pairs_count": len(corpus_pairs),
        "embedding_model": MODEL_NAME,
        "embedding_dimension": dim,
        "encode_duration_s": round(encode_duration, 2),
        "total_build_time_s": round(total_duration, 2),
        "faiss_index_path": str(FAISS_INDEX_PATH),
        "meta_path": str(INDEX_META_PATH),
    }


if __name__ == "__main__":
    stats = build_retrieval_index()
    print("\n--- Indexing Summary ---")
    for k, v in stats.items():
        print(f"{k}: {v}")
