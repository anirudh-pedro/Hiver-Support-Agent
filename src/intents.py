"""
intents.py - Production Intent Classification module for AmazonHelp support threads.
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
from typing import Any, Dict, List, Optional, Tuple

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
    from src.prompts import (
        INTENT_TAXONOMY,
        SYSTEM_PROMPT,
        VALID_INTENTS,
        build_classification_prompt,
    )
except ImportError:
    from prompts import (
        INTENT_TAXONOMY,
        SYSTEM_PROMPT,
        VALID_INTENTS,
        build_classification_prompt,
    )

logger = logging.getLogger(__name__)

# Configurable default model
DEFAULT_MODEL: str = os.getenv("LLM_MODEL") or ("openai/gpt-oss-20b" if (os.getenv("GROQ_API") or os.getenv("groq_api") or os.getenv("GROQ_API_KEY")) else "gpt-4o-mini")


@dataclass
class IntentResult:
    """Standard output schema for intent classification."""
    intent: str
    confidence: float
    reasoning: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_all_llm_clients() -> List[Tuple[Any, str]]:
    """
    Instantiates all configured LLM clients (supporting primary and secondary Groq keys).
    Returns list of (client, model) tuples.
    """
    clients: List[Tuple[Any, str]] = []
    groq_keys: List[str] = []
    for k in ["groq_api", "GROQ_API", "GROQ_API_KEY", "groq_api_2", "GROQ_API_2", "GROQ_API_KEY_2"]:
        val = os.getenv(k)
        if val and val not in groq_keys:
            groq_keys.append(val)

    if groq_keys:
        try:
            from openai import OpenAI
            base_url = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
            for key in groq_keys:
                c = OpenAI(api_key=key, base_url=base_url, max_retries=0)
                model = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
                clients.append((c, model))
        except Exception as exc:
            logger.error("Failed to initialize Groq client: %s", exc)

    # Check OpenAI credentials as fallback
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            from openai import OpenAI
            base_url = os.getenv("OPENAI_BASE_URL")
            c = OpenAI(api_key=openai_key, base_url=base_url) if base_url else OpenAI(api_key=openai_key)
            model = os.getenv("LLM_MODEL", "gpt-4o-mini")
            clients.append((c, model))
        except Exception as exc:
            logger.error("Failed to initialize OpenAI client: %s", exc)

    return clients


def get_llm_client() -> Tuple[Optional[Any], str]:
    """
    Instantiates an OpenAI-compatible client, detecting Groq or OpenAI credentials.
    Returns:
        (client, default_model_name)
    """
    all_clients = get_all_llm_clients()
    if all_clients:
        return all_clients[0]
    return None, DEFAULT_MODEL


def _heuristic_classify_fallback(first_message: str) -> IntentResult:
    """
    Offline heuristic classifier used ONLY when explicitly permitted for offline
    testing where no API credentials are configured.
    """
    t = first_message.lower()

    # 1. Billing dispute
    if any(k in t for k in ["charged", "charge", "refund", "billing", "unauthorized", "cashback", "double bill", "subscription fee", "membership fee"]):
        return IntentResult(
            intent="billing_dispute",
            confidence=0.91,
            reasoning="Customer is disputing an unexpected charge, billing issue, or requesting a refund."
        )

    # 2. Delivery delay
    if any(k in t for k in ["delay", "delayed", "late", "where is my", "tracking", "not arrived", "haven't received my order", "haven't received my package", "delivered but"]):
        return IntentResult(
            intent="delivery_delay",
            confidence=0.92,
            reasoning="Customer indicates their package is delayed or has not arrived."
        )

    # 3. Order product problem
    if any(k in t for k in ["damaged", "broken", "defective", "wrong item", "different item", "missing item", "scratched"]):
        return IntentResult(
            intent="order_product_problem",
            confidence=0.93,
            reasoning="Customer received a damaged, defective, or incorrect physical item."
        )

    # 4. Account access
    if any(k in t for k in ["login", "sign in", "password", "locked out", "hacked", "2fa", "otp", "account locked"]):
        return IntentResult(
            intent="account_access",
            confidence=0.95,
            reasoning="Customer is reporting account lockout or login credential difficulty."
        )

    # 5. Product info question
    if any(k in t for k in ["compatible with", "specifications", "is this supported", "release date", "warranty period"]):
        return IntentResult(
            intent="product_info_question",
            confidence=0.88,
            reasoning="Customer is asking a factual question about product specifications."
        )

    # 6. Scam / phishing check
    if any(k in t for k in ["scam", "phishing", "is this email legit", "is this legit", "fake email"]):
        return IntentResult(
            intent="scam_phishing_check",
            confidence=0.96,
            reasoning="Customer is inquiring whether a received communication is a legitimate Amazon notice or phishing."
        )

    # 7. Non support
    if any(k in t for k in ["thank you", "thanks", "great job", "love you guys", "kudos"]):
        return IntentResult(
            intent="non_support",
            confidence=0.90,
            reasoning="Customer is expressing praise or general social chatter without an active support issue."
        )

    # 8. Service complaint vague
    if any(k in t for k in ["worst service", "awful service", "terrible support", "useless support", "horrible service"]):
        return IntentResult(
            intent="service_complaint_vague",
            confidence=0.84,
            reasoning="Customer is venting frustration regarding customer support quality without an actionable product request."
        )

    return IntentResult(
        intent="other",
        confidence=0.50,
        reasoning="Message does not clearly align with standard predefined intent keywords."
    )


def classify_intent(
    thread: Dict[str, Any],
    include_brand_reply: bool = False,
    model: Optional[str] = None,
    client: Optional[Any] = None,
    allow_fallback: bool = False,
    max_retries: int = 5,
) -> IntentResult:
    """
    Classifies the intent of a customer support thread using a real LLM API.

    Args:
        thread: Thread dictionary containing 'turns'.
        include_brand_reply: If True, includes the first AmazonHelp reply for context.
        model: Optional LLM model identifier.
        client: Optional pre-configured client.
        allow_fallback: If True and NO API key is available, uses offline heuristic.
                        Defaults to False to prevent silently swallowing errors.
        max_retries: Maximum retries on rate limits or transient errors.

    Returns:
        IntentResult(intent, confidence, reasoning)

    Raises:
        RuntimeError: If LLM call fails or rate limits persist without silent fallback.
    """
    turns = thread.get("turns", [])
    first_message = ""
    if turns:
        first_message = turns[0].get("text", "").strip()
    elif "customer_message" in thread and thread["customer_message"]:
        first_message = str(thread["customer_message"]).strip()
    elif "full_thread_text" in thread and thread["full_thread_text"]:
        for line in thread["full_thread_text"].splitlines():
            if line.startswith("[Customer]:"):
                first_message = line.replace("[Customer]:", "").strip()
                break
        if not first_message and thread["full_thread_text"].splitlines():
            first_message = thread["full_thread_text"].splitlines()[0].strip()

    if not first_message:
        return IntentResult(intent="other", confidence=0.0, reasoning="Empty message content or no turns.")

    brand_reply: Optional[str] = None
    if include_brand_reply and len(turns) > 1:
        for turn in turns[1:]:
            if not turn.get("inbound", True) and turn.get("author_id") == "AmazonHelp":
                brand_reply = turn.get("text")
                break

    llm_client = client
    selected_model = model
    if llm_client is None:
        detected_client, detected_model = get_llm_client()
        llm_client = detected_client
        if not selected_model:
            selected_model = detected_model

    # If no API client exists
    if llm_client is None:
        if allow_fallback:
            return _heuristic_classify_fallback(first_message)
        raise RuntimeError(
            "No active LLM API credentials found. Please set 'groq_api' / 'GROQ_API_KEY' or 'OPENAI_API_KEY' in .env."
        )

    user_content = build_classification_prompt(first_message, brand_reply)

    last_error: Optional[Exception] = None
    backoff_delay = 2.0

    for attempt in range(max_retries):
        try:
            create_kwargs = {
                "model": selected_model,
                "temperature": 0.0,
                "max_tokens": 250,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
            }
            if "qwen" in selected_model.lower():
                create_kwargs["response_format"] = {"type": "json_object"}

            response = llm_client.chat.completions.create(**create_kwargs)
            raw_text = response.choices[0].message.content or ""
            if not raw_text.strip():
                raise ValueError("Received empty response content from LLM.")

            # Robust JSON extraction
            match = re.search(r"\{.*?\}", raw_text, re.DOTALL)
            if match:
                parsed = json.loads(match.group(0))
            else:
                parsed = json.loads(raw_text)

            assigned_intent = parsed.get("assigned_intent")
            if not assigned_intent:
                raise ValueError(f"Missing 'assigned_intent' key in LLM response: {raw_text}")

            if assigned_intent not in VALID_INTENTS:
                logger.warning(
                    "LLM returned invalid intent '%s' (not in taxonomy). Coercing to 'other'.",
                    assigned_intent,
                )
                assigned_intent = "other"

            confidence = float(parsed.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))
            justification = str(parsed.get("justification", "")).strip()

            return IntentResult(
                intent=assigned_intent,
                confidence=confidence,
                reasoning=justification,
            )

        except Exception as exc:
            last_error = exc
            error_str = str(exc).lower()

            # Check for Rate Limit (HTTP 429)
            if "rate limit" in error_str or "429" in error_str:
                # Groq rate limit reset can take several seconds
                sleep_time = backoff_delay
                if "retry after" in error_str:
                    try:
                        m = re.search(r"retry after\s+([0-9.]+)", error_str)
                        if m:
                            sleep_time = float(m.group(1)) + 0.5
                    except Exception:
                        pass
                logger.warning(
                    "Rate limit encountered on thread %s. Backing off %.1fs (attempt %d/%d)...",
                    thread.get("thread_id"),
                    sleep_time,
                    attempt + 1,
                    max_retries,
                )
                time.sleep(sleep_time)
                backoff_delay = min(backoff_delay * 2.0, 30.0)
                continue

            # Check for JSON decode errors
            elif isinstance(exc, (json.JSONDecodeError, ValueError)):
                logger.warning(
                    "Malformed JSON from LLM on thread %s: %s. Retrying (attempt %d/%d)...",
                    thread.get("thread_id"),
                    exc,
                    attempt + 1,
                    max_retries,
                )
                time.sleep(1.0)
                continue

            else:
                logger.error(
                    "API call error on thread %s: %s. Retrying in %.1fs...",
                    thread.get("thread_id"),
                    exc,
                    backoff_delay,
                )
                time.sleep(backoff_delay)
                backoff_delay *= 1.5
                continue

    # If all retries fail, do NOT silently swallow into fallback!
    raise RuntimeError(
        f"Intent classification failed after {max_retries} attempts on thread {thread.get('thread_id')}: {last_error}"
    )


def classify_intents_batch(
    threads: List[Dict[str, Any]],
    include_brand_reply: bool = False,
    model: Optional[str] = None,
    max_workers: int = 10,
    client: Optional[Any] = None,
    allow_fallback: bool = False,
) -> List[IntentResult]:
    """
    Classifies a list of threads concurrently while maintaining the original order.

    Args:
        threads: List of thread dictionaries.
        include_brand_reply: Whether to include the brand reply for context.
        model: Optional model name.
        max_workers: Thread pool concurrency limit.
        client: Optional client instance.
        allow_fallback: Whether to permit offline fallback if no API credentials exist.

    Returns:
        List of IntentResult matching the input list order.
    """
    if not threads:
        return []

    llm_client = client
    selected_model = model
    if llm_client is None:
        detected_client, detected_model = get_llm_client()
        llm_client = detected_client
        if not selected_model:
            selected_model = detected_model

    results: List[Optional[IntentResult]] = [None] * len(threads)

    # If no live client, check allow_fallback
    if llm_client is None:
        if allow_fallback:
            return [
                classify_intent(t, include_brand_reply, model=selected_model, client=None, allow_fallback=True)
                for t in threads
            ]
        raise RuntimeError("No active LLM API credentials found. Please set 'groq_api' in .env.")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(
                classify_intent,
                thread=t,
                include_brand_reply=include_brand_reply,
                model=selected_model,
                client=llm_client,
                allow_fallback=allow_fallback,
            ): idx
            for idx, t in enumerate(threads)
        }

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                logger.error("Batch classification failed on index %d: %s", idx, exc)
                results[idx] = IntentResult(
                    intent="other",
                    confidence=0.0,
                    reasoning=f"Worker exception: {exc}",
                )

    return [res or IntentResult(intent="other", confidence=0.0, reasoning="Unresolved") for res in results]
