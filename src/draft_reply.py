"""
src/draft_reply.py

Production module for grounded customer support reply drafting in AmazonHelp's voice.
Grounds drafting in historical brand triage and response patterns retrieved via FAISS.

Pipeline:
  1. Extract customer message from thread.
  2. Classify intent via classify_intent(thread) (if not pre-computed).
  3. Retrieve top-k precedent threads via retrieve_similar(customer_message, k=5, dedup_macros=True).
  4. Construct few-shot grounded prompt (src/prompts.py: build_reply_draft_prompt_v1).
  5. LLM drafts concise, empathetic, non-hallucinatory Twitter reply with ^initials sign-off.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.intents import IntentResult, classify_intent, get_llm_client
from src.prompts import (
    REPLY_DRAFT_SYSTEM_PROMPT_V1,
    build_reply_draft_prompt_v1,
)
from src.retrieve import RetrievedExample, get_retriever, retrieve_similar

logger = logging.getLogger(__name__)

# Default model selection: fast, instruction-following openai/gpt-oss-20b on Groq
DEFAULT_DRAFT_MODEL: str = (
    os.getenv("DRAFT_MODEL")
    or ("openai/gpt-oss-20b" if (os.getenv("GROQ_API") or os.getenv("groq_api") or os.getenv("GROQ_API_KEY")) else "gpt-4o-mini")
)


@dataclass
class DraftReply:
    """Standard output schema for a drafted support reply."""
    reply_text: str
    grounding_thread_ids: List[int]
    model_used: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def extract_customer_message(thread: Dict[str, Any]) -> str:
    """
    Extracts the customer's opening message from various thread representations:
    - Raw thread dict with 'turns' list
    - Dict with 'customer_message' field
    - Dict with 'full_thread_text' string
    """
    if "customer_message" in thread and thread["customer_message"]:
        return str(thread["customer_message"]).strip()

    # 1. From 'turns' list
    turns = thread.get("turns", [])
    if turns:
        for turn in turns:
            author = turn.get("author_id", "")
            inbound = turn.get("inbound", False)
            if author != "AmazonHelp" or inbound:
                text = turn.get("text", "").strip()
                if text:
                    return text

    # 2. From 'full_thread_text' string
    full_text = thread.get("full_thread_text", "")
    if full_text:
        lines = full_text.splitlines()
        for line in lines:
            if line.startswith("[Customer]:"):
                return line.replace("[Customer]:", "").strip()
        # Fallback to first line if no [Customer] tag
        if lines:
            return lines[0].strip()

    return ""


def clean_drafted_reply(raw_text: str) -> str:
    """
    Cleans up the raw LLM response:
    - Strips wrapping quotation marks.
    - Strips markdown formatting or prefixes like 'AmazonHelp: ' or 'Reply: '.
    - Normalizes multiple spaces and newlines.
    """
    if not raw_text:
        return ""

    text = raw_text.strip()

    # Strip thinking tags if present
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Strip common prefixes
    prefixes = [
        "AmazonHelp:",
        "Reply:",
        "Drafted Reply:",
        "Draft Reply:",
        "Amazon:",
        "Tweet:",
    ]
    for p in prefixes:
        if text.lower().startswith(p.lower()):
            text = text[len(p):].strip()

    # Strip wrapping quotes
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()

    # Collapse internal whitespace/newlines to single line
    text = " ".join(text.split())
    return text


def draft_reply(
    thread: Dict[str, Any],
    intent: Optional[IntentResult | str] = None,
    retrieved_examples: Optional[List[RetrievedExample]] = None,
    model: Optional[str] = None,
    client: Optional[Any] = None,
    k: int = 5,
    max_retries: int = 4,
) -> DraftReply:
    """
    Drafts a grounded AmazonHelp reply for a customer thread.

    Args:
        thread: Thread dict (containing 'turns', 'full_thread_text', or 'customer_message').
        intent: Optional IntentResult or intent category string (classified automatically if None).
        retrieved_examples: Optional pre-fetched few-shot precedents (retrieved via FAISS if None).
        model: Optional LLM model identifier.
        client: Optional OpenAI/Groq client instance.
        k: Number of historical precedents to retrieve (default: 5).
        max_retries: Maximum number of retries on API rate limits.

    Returns:
        DraftReply {reply_text, grounding_thread_ids, model_used}
    """
    # 1. Extract customer message
    cust_msg = extract_customer_message(thread)
    if not cust_msg:
        logger.warning("Thread %s has empty customer message", thread.get("thread_id"))
        return DraftReply(
            reply_text="We'd like to help you with this! Please send us a DM with your details so we can investigate. ^AH",
            grounding_thread_ids=[],
            model_used=model or DEFAULT_DRAFT_MODEL,
        )

    # 2. Resolve Intent
    intent_str = "other"
    if intent is not None:
        if isinstance(intent, IntentResult):
            intent_str = intent.intent
        elif isinstance(intent, str):
            intent_str = intent
    else:
        try:
            intent_res = classify_intent(thread, client=client)
            intent_str = intent_res.intent
        except Exception as exc:
            logger.warning("Intent classification failed in draft_reply: %s. Defaulting to 'other'", exc)
            intent_str = "other"

    # 3. Retrieve grounding precedents
    if retrieved_examples is None:
        try:
            retrieved_examples = retrieve_similar(cust_msg, k=k, dedup_macros=True)
        except Exception as exc:
            logger.error("Retrieval failed in draft_reply: %s", exc)
            retrieved_examples = []

    grounding_ids = [ex.thread_id for ex in retrieved_examples]

    # 4. Initialize LLM Client
    llm_client = client
    selected_model = model or DEFAULT_DRAFT_MODEL
    if llm_client is None:
        detected_client, detected_model = get_llm_client()
        llm_client = detected_client
        if not model:
            # If default detected model is gpt-oss-120b but Groq key is present, use DEFAULT_DRAFT_MODEL
            selected_model = DEFAULT_DRAFT_MODEL

    if llm_client is None:
        raise RuntimeError("No LLM API credentials found for reply drafting. Set groq_api in .env.")

    # 5. Build Grounded Prompt
    user_prompt = build_reply_draft_prompt_v1(
        customer_message=cust_msg,
        intent=intent_str,
        retrieved_examples=retrieved_examples,
    )

    # 6. Call LLM with Retry Logic
    backoff = 2.0
    for attempt in range(max_retries):
        try:
            response = llm_client.chat.completions.create(
                model=selected_model,
                messages=[
                    {"role": "system", "content": REPLY_DRAFT_SYSTEM_PROMPT_V1},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=220,
                temperature=0.3,
            )

            raw_reply = response.choices[0].message.content or ""
            # print debug
            # logger.info("RAW REPLY FROM MODEL: %r", raw_reply)
            if not raw_reply.strip():
                logger.warning("Empty raw_reply from model %s", selected_model)
            reply_text = clean_drafted_reply(raw_reply)

            # Ensure signoff exists; if model omitted ^XX, append default ^AH
            if not re.search(r"\^[A-Z]{2,4}\b", reply_text):
                reply_text += " ^AH"

            return DraftReply(
                reply_text=reply_text,
                grounding_thread_ids=grounding_ids,
                model_used=selected_model,
            )

        except Exception as exc:
            err_msg = str(exc)
            if "429" in err_msg or "rate_limit" in err_msg.lower():
                # Extract wait time if Groq returns suggested retry duration
                wait_match = re.search(r"try again in ([\d\.]+)s", err_msg)
                sleep_time = float(wait_match.group(1)) + 0.5 if wait_match else backoff
                logger.warning(
                    "Rate limit on draft_reply (attempt %d/%d). Sleeping %.1fs...",
                    attempt + 1,
                    max_retries,
                    sleep_time,
                )
                time.sleep(sleep_time)
                backoff = min(backoff * 2.0, 30.0)
            else:
                logger.error(
                    "API call error on draft_reply (thread %s, attempt %d/%d): %s",
                    thread.get("thread_id"),
                    attempt + 1,
                    max_retries,
                    exc,
                )
                time.sleep(backoff)
                backoff *= 1.5

    raise RuntimeError(
        f"Reply drafting failed after {max_retries} attempts on thread {thread.get('thread_id')}"
    )


def draft_replies_batch(
    threads: List[Dict[str, Any]],
    intents: Optional[List[IntentResult | str]] = None,
    model: Optional[str] = None,
    max_workers: int = 4,
    client: Optional[Any] = None,
    k: int = 5,
) -> List[DraftReply]:
    """
    Drafts replies for a batch of customer threads concurrently while preserving order.

    Args:
        threads: List of thread dictionaries.
        intents: Optional list of corresponding IntentResult or intent strings (matching threads order).
        model: Optional LLM model identifier.
        max_workers: Thread pool concurrency limit (default: 4 to respect rate limits).
        client: Optional shared LLM client.
        k: Number of historical precedents to retrieve.

    Returns:
        List of DraftReply matching the input list order.
    """
    if not threads:
        return []

    # Pre-initialize shared retriever resources to prevent race condition
    retriever = get_retriever()
    retriever._load_resources()

    llm_client = client
    selected_model = model or DEFAULT_DRAFT_MODEL
    if llm_client is None:
        detected_client, _ = get_llm_client()
        llm_client = detected_client

    results: List[Optional[DraftReply]] = [None] * len(threads)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {}
        for idx, t in enumerate(threads):
            item_intent = intents[idx] if (intents and idx < len(intents)) else None
            future = executor.submit(
                draft_reply,
                thread=t,
                intent=item_intent,
                retrieved_examples=None,
                model=selected_model,
                client=llm_client,
                k=k,
            )
            future_to_idx[future] = idx

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                t_id = threads[idx].get("thread_id", idx)
                logger.error("Batch drafting failed on thread %s (idx %d): %s", t_id, idx, exc)
                results[idx] = DraftReply(
                    reply_text=f"I'm sorry for the trouble with your order! Please reach out to us via DM so we can assist. ^AH",
                    grounding_thread_ids=[],
                    model_used=selected_model,
                )

    return [
        r or DraftReply(
            reply_text="I'm sorry for the inconvenience! Please contact us via DM so we can help. ^AH",
            grounding_thread_ids=[],
            model_used=selected_model,
        )
        for r in results
    ]
