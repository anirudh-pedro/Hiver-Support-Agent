"""
src/retrieve.py

Semantic retrieval module grounding support agent replies in historically resolved AmazonHelp threads.
Uses FAISS IndexFlatIP (cosine similarity over normalized sentence-transformers embeddings).

Deduplication:
Support agents frequently reuse macro templates (e.g., standard DM links, apology phrasing).
The retriever detects near-duplicate boilerplate brand resolutions by normalizing out:
  - Twitter handle mentions (@user)
  - Shortened URLs (https://t.co/...)
  - Agent signature sign-offs (^XX)
  - Whitespace and casing
If multiple top candidates use the same underlying macro, only the highest-scoring candidate is kept,
and the retriever backfills from the candidate pool to ensure k distinct resolution strategies.

Intent Filtering Strategy Decision (v1):
- Chosen Approach: Pure semantic similarity for v1 (intent_filter=None by default).
- Rationale:
  1. Zero Corpus Classification Overhead: Running LLM or zero-shot classifiers across 60,355
     corpus threads is cost-prohibitive and computationally unnecessary.
  2. Dense Semantic Clustering: all-MiniLM-L6-v2 embeddings inherently cluster domain intents
     (e.g., delivery delay complaints naturally retrieve delivery resolution threads with >0.75 similarity).
  3. Avoids Cascade Misclassification: Hard filtering on predicted intent would discard relevant
     analogous resolutions whenever an upstream classifier mislabels nuanced cross-category queries.
  4. Extensibility: The parameter `intent_filter` is preserved in the API signature for future metadata filtering.
"""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = PROJECT_ROOT / "data" / "index"
FAISS_INDEX_PATH = INDEX_DIR / "amazonhelp.faiss"
INDEX_META_PATH = INDEX_DIR / "amazonhelp_meta.jsonl"
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


@dataclass
class RetrievedExample:
    thread_id: int
    customer_message: str
    brand_resolution: str
    similarity_score: float

    def to_dict(self) -> dict:
        return {
            "thread_id": self.thread_id,
            "customer_message": self.customer_message,
            "brand_resolution": self.brand_resolution,
            "similarity_score": round(float(self.similarity_score), 4),
        }


def normalize_resolution_macro(text: str) -> str:
    """
    Normalizes resolution text to identify identical/near-duplicate macro templates.
    Strips @mentions, URLs, agent sign-offs (^XX), and non-alphanumeric punctuation.
    """
    if not text:
        return ""
    # Strip URLs
    clean = re.sub(r"https?://\S+", "", text)
    # Strip @mentions
    clean = re.sub(r"@[A-Za-z0-9_]+", "", clean)
    # Strip agent initials signoffs like ^TN, ^AG, ^J
    clean = re.sub(r"\^[A-Za-z]{1,4}\b", "", clean)
    # Lowercase and clean non-alphanumeric
    clean = re.sub(r"[^a-z0-9\s]", " ", clean.lower())
    # Collapse multiple whitespaces
    clean = " ".join(clean.split())
    return clean


class AmazonHelpRetriever:
    """
    Singleton-style retriever managing the FAISS index, metadata mapping, and embedding model.
    """

    def __init__(
        self,
        index_path: Path = FAISS_INDEX_PATH,
        meta_path: Path = INDEX_META_PATH,
        model_name: str = DEFAULT_MODEL_NAME,
    ):
        self.index_path = Path(index_path)
        self.meta_path = Path(meta_path)
        self.model_name = model_name

        self._index: Optional[faiss.IndexFlatIP] = None
        self._meta: Optional[List[dict]] = None
        self._embedder: Optional[SentenceTransformer] = None

    def _load_resources(self):
        """Loads FAISS index, metadata records, and embedding model on demand."""
        if self._index is None:
            if not self.index_path.exists():
                raise FileNotFoundError(
                    f"FAISS index not found at {self.index_path}. Run `python src/build_index.py` first."
                )
            logger.info(f"Loading FAISS index from {self.index_path}...")
            self._index = faiss.read_index(str(self.index_path))

        if self._meta is None:
            if not self.meta_path.exists():
                raise FileNotFoundError(
                    f"Index metadata not found at {self.meta_path}. Run `python src/build_index.py` first."
                )
            logger.info(f"Loading metadata from {self.meta_path}...")
            records = []
            with open(self.meta_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        records.append(json.loads(line))
            self._meta = records
            logger.info(f"Loaded {len(self._meta)} metadata records")

        if self._embedder is None:
            logger.info(f"Loading embedding model {self.model_name}...")
            self._embedder = SentenceTransformer(self.model_name)

    def retrieve(
        self,
        query_text: str,
        k: int = 5,
        intent_filter: Optional[str] = None,
        dedup_macros: bool = True,
    ) -> List[RetrievedExample]:
        """
        Retrieves top-k relevant resolution examples for a given customer query text.

        Args:
            query_text: Incoming customer query text.
            k: Number of distinct examples to return.
            intent_filter: Optional intent string filter (v1 defaults to pure semantic similarity).
            dedup_macros: Whether to suppress identical/near-duplicate brand resolution macros.

        Returns:
            List of RetrievedExample instances sorted by descending similarity score.
        """
        if not query_text or not query_text.strip():
            return []

        self._load_resources()

        # Query embedding normalized for cosine similarity with IndexFlatIP
        query_vec = self._embedder.encode(
            [query_text.strip()],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        query_vec = np.ascontiguousarray(query_vec, dtype=np.float32)

        # Over-fetch to allow deduplication backfilling
        fetch_k = max(k * 5, 30) if dedup_macros else k
        fetch_k = min(fetch_k, self._index.ntotal)

        scores, indices = self._index.search(query_vec, fetch_k)
        scores = scores[0]
        indices = indices[0]

        results: List[RetrievedExample] = []
        seen_macro_keys: Set[str] = set()

        for score, idx in zip(scores, indices):
            if idx < 0 or idx >= len(self._meta):
                continue

            meta_item = self._meta[idx]
            brand_res = meta_item["brand_resolution"]

            if dedup_macros:
                macro_key = normalize_resolution_macro(brand_res)
                # If the normalized text is non-empty and already seen, skip as duplicate macro
                if macro_key and macro_key in seen_macro_keys:
                    continue
                if macro_key:
                    seen_macro_keys.add(macro_key)

            results.append(
                RetrievedExample(
                    thread_id=int(meta_item["thread_id"]),
                    customer_message=meta_item["customer_message"],
                    brand_resolution=brand_res,
                    similarity_score=float(score),
                )
            )

            if len(results) >= k:
                break

        return results


# Module-level singleton instance for zero-reloading efficiency
_DEFAULT_RETRIEVER: Optional[AmazonHelpRetriever] = None


def get_retriever() -> AmazonHelpRetriever:
    """Returns the shared AmazonHelpRetriever instance."""
    global _DEFAULT_RETRIEVER
    if _DEFAULT_RETRIEVER is None:
        _DEFAULT_RETRIEVER = AmazonHelpRetriever()
    return _DEFAULT_RETRIEVER


def retrieve_similar(
    query_text: str,
    k: int = 5,
    intent_filter: Optional[str] = None,
    dedup_macros: bool = True,
) -> List[RetrievedExample]:
    """
    Standard entrypoint for retrieval grounding.

    Args:
        query_text: Customer message to find historical precedent for.
        k: Number of results to return (default: 5).
        intent_filter: Optional intent filter (None for pure semantic similarity in v1).
        dedup_macros: If True, filters out near-duplicate brand resolution macros.

    Returns:
        List of RetrievedExample {thread_id, customer_message, brand_resolution, similarity_score}
    """
    retriever = get_retriever()
    return retriever.retrieve(
        query_text=query_text,
        k=k,
        intent_filter=intent_filter,
        dedup_macros=dedup_macros,
    )
