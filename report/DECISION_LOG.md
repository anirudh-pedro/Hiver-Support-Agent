# Architecture & Engineering Decision Log: AmazonHelp AI Customer Support Agent

This log records the foundational architectural, methodological, and modeling decisions made across the lifecycle of the project, including the non-obvious trade-offs and operational rationales.

---

1. **Target Brand Selection (`@AmazonHelp`)**:
   We chose `@AmazonHelp` over other brands (e.g., AppleSupport, Delta, Uber) because it represents the highest volume, most operationally diverse multi-turn customer support ecosystem in the Kaggle dataset, spanning physical logistics, digital services, billing, and account access.

2. **Strict English-Only Ingestion Filtering**:
   We deterministically filtered inbound tweets to English (`lang == 'en'`), discarding ~25% of raw threads; this prevented multilingual semantic drift in the single-vector embedding space and kept intent definitions linguistically unambiguous.

3. **Forward-Only Graph Thread Reconstruction with Tie-Breaking Hierarchy**:
   To convert disconnected Twitter reply trees into clean dialogue threads, we traversed `in_reply_to_tweet_id` forward from root customer tweets and resolved branch conflicts using a deterministic hierarchy: *Ended-with-Brand First $\to$ Longest Branch $\to$ Lowest Tweet ID Tie-break*.

4. **Gold Labels as "What *Should* Happen" vs. "What the Brand Did"**:
   Golden ground-truth labels were annotated based on normative operational support policy rather than actual historical brand actions, because Twitter support agents frequently copy-paste generic macro redirects even when immediate auto-handling or critical escalation was warranted.

5. **Strict Contamination Prevention & Corpus Exclusion**:
   All 220 golden evaluation threads, the 30 validation sample threads, and the 12 spot-check threads were programmatically excluded from the 60,355-pair FAISS retrieval index, preventing data leakage and guaranteeing out-of-sample retrieval evaluation.

6. **Pure Dense Semantic Retrieval vs. Corpus-Wide Intent Tagging**:
   We indexed historical QA pairs using pure semantic embeddings (`all-MiniLM-L6-v2`) without pre-classifying all 60k pairs with an LLM, avoiding prohibitive API cost, rate-limit bottlenecks, and cascading classification errors into retrieval.

7. **Embedding Architecture Selection (FAISS `IndexFlatIP` + Cosine Normalization)**:
   We selected normalized inner-product search with `all-MiniLM-L6-v2` (384 dimensions) because it runs in sub-millisecond latency on CPU, fits entirely in memory (~93MB index), and eliminates GPU infrastructure dependencies for retrieval.

8. **Three-Layer Hybrid Routing Policy (Safety Rules $\to$ Confidence $\to$ LLM)**:
   We structured routing hierarchically so that deterministic regex rules execute first, catching high-liability security and PII violations with 100% determinism before any LLM call, reducing API latency and preventing prompt jailbreaks from suppressing escalations.

9. **Low-Confidence Intent Override Threshold ($\le 0.65$)**:
   If the intent classifier outputs a confidence score $\le 0.65$, the system automatically routes the thread to human review (`rule_low_confidence`), ensuring the agent defaults to human judgment whenever classification boundary uncertainty is high.

10. **Cross-Model LLM-as-a-Judge to Mitigate Self-Preference Bias**:
    We used an independent model family (`qwen/qwen3.8-27b`) for automated response evaluation rather than the drafting model (`openai/gpt-oss-20b`), preventing self-preference bias where models systematically award higher quality marks to their own outputs.

11. **Subordinating Surface N-Gram Metrics (BLEU/ROUGE) to Multi-Dimensional Rubrics**:
    We designated BLEU/ROUGE as secondary sanity checks only, recognizing that reference Twitter replies vary widely in wording (e.g. agent names, link formats), and relied on 4-dimensional Likert evaluation (Tone, Correctness, Completeness, Safety) as primary.

12. **AI-Assisted Prefill with Mandatory Human Verification for Golden Set Annotation**:
    To construct the 220-thread golden set efficiently, an LLM generated initial draft labels with rationale, followed by exhaustive human review in a custom Streamlit labeling tool (`src/label_app.py`) to audit, correct, and finalize all 220 annotations.

13. **Strict 280-Character Twitter Length Enforcement**:
    We embedded hard length constraints into the drafting prompt and added character counters to the UI to ensure generated replies comply with native Twitter post constraints rather than generating verbose email-style paragraphs.

14. **Deterministic Routing Auditability over Pure Black-Box Predictions**:
    Even though a simple heuristic (`turns >= 3`) yielded slightly higher raw F1, we retained the hybrid rule-and-policy engine because 28% of decisions resolve through named, inspectable safety rules (e.g., `rule_security`, `rule_pii_exposure`), providing enterprise auditability that simple heuristics lack.
