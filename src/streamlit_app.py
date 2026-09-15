"""
src/streamlit_app.py

Interactive Demo UI for the AmazonHelp AI Customer Support Agent.
Reuses existing project modules:
  - src.intents: Intent classification
  - src.retrieve: FAISS dense semantic retrieval
  - src.draft_reply: RAG grounded reply generation
  - src.route: 3-layer hybrid escalation routing policy

Run with:
    streamlit run src/streamlit_app.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

from src.draft_reply import DEFAULT_DRAFT_MODEL, draft_reply
from src.intents import IntentResult, classify_intent, get_all_llm_clients, get_llm_client
from src.prompts import INTENT_TAXONOMY, VALID_INTENTS
from src.retrieve import (
    FAISS_INDEX_PATH,
    INDEX_META_PATH,
    AmazonHelpRetriever,
    RetrievedExample,
    get_retriever,
)
from src.route import (
    _RULES,
    LOW_CONFIDENCE_THRESHOLD,
    ROUTING_LLM_MODEL,
    RoutingDecision,
    route_thread,
)

# Page configuration
st.set_page_config(
    page_title="AI Customer Support Agent | AmazonHelp",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling for clean, professional cards and badges
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .badge-auto {
        background-color: #d4edda;
        color: #155724;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        border: 1px solid #c3e6cb;
    }
    .badge-escalate {
        background-color: #f8d7da;
        color: #721c24;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        border: 1px solid #f5c6cb;
    }
    .reply-box {
        background-color: #e8f4fd;
        border-left: 5px solid #1a73e8;
        padding: 16px;
        border-radius: 6px;
        font-size: 1.05rem;
        line-height: 1.5;
        color: #1a202c;
    }
    .pipeline-step {
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 6px;
        padding: 8px 12px;
        text-align: center;
        font-size: 0.85rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# CACHED RESOURCE LOADERS
# ==============================================================================

@st.cache_resource(show_spinner="Loading FAISS semantic index & embedding model...")
def load_cached_retriever() -> Optional[AmazonHelpRetriever]:
    """Loads and caches the FAISS retriever singleton."""
    if not FAISS_INDEX_PATH.exists() or not INDEX_META_PATH.exists():
        return None
    try:
        retriever = get_retriever()
        retriever._load_resources()
        return retriever
    except Exception as exc:
        logging.error("Failed to load retriever: %s", exc)
        return None


def check_api_status() -> Tuple[bool, List[Any], str]:
    """Verifies LLM API key availability without exposing secrets."""
    clients_pool = get_all_llm_clients()
    if not clients_pool:
        return False, [], "No active LLM API credentials found in environment or .env."
    clients = [c[0] for c in clients_pool]
    return True, clients, f"{len(clients)} LLM client(s) active"


# ==============================================================================
# PRESET EXAMPLES
# ==============================================================================

PRESET_EXAMPLES = {
    "delivery": {
        "message": "Hi, my order #171-5604465 was supposed to arrive yesterday by 8pm but tracking shows it is still in transit and delayed. Can you check where it is?",
        "context": "",
    },
    "security": {
        "message": "I received an email stating my account password and email were changed, and now I'm locked out and cannot log in. Please help me secure my account!",
        "context": "[Customer]: I got an email saying my login details were updated but I didn't do it.\n[AmazonHelp]: We take security seriously. Have you tried resetting via the help page?",
    },
    "billing": {
        "message": "I was just charged $149.99 on my credit card for an annual Prime membership that I never authorized or signed up for. I need an immediate refund and cancellation.",
        "context": "",
    },
}


# ==============================================================================
# MAIN APPLICATION
# ==============================================================================

def main():
    # Header
    st.title("📦 AI Customer Support Agent")
    st.subheader("AmazonHelp Support Triage & Response Assistant")
    st.caption("End-to-end intelligent triage: Intent Classification → Semantic Grounding → Draft Generation → Safety Routing.")

    # Check dependencies and system status
    retriever = load_cached_retriever()
    has_api, clients, api_status_msg = check_api_status()

    # ── SIDEBAR CONFIGURATION ────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API status badge
        if has_api:
            st.success(f"🟢 LLM API: {api_status_msg}")
        else:
            st.error("🔴 LLM API key is not configured.")
            st.info("Set `groq_api` or `OPENAI_API_KEY` in your `.env` file.")

        # FAISS Index status badge
        if retriever is not None:
            st.success("🟢 FAISS Index: 60,355 QA pairs loaded")
        else:
            st.error("🔴 FAISS Index missing.")
            st.code("python src/build_index.py", language="bash")

        st.divider()
        st.subheader("Parameters")
        
        selected_model = st.selectbox(
            "Model Architecture",
            options=["openai/gpt-oss-20b", "groq/compound-mini", "qwen/qwen3.8-27b", "groq/compound"],
            index=0,
            help="LLM model used for classification, drafting, and policy routing.",
        )

        top_k = st.slider(
            "Retrieval Precedents (Top-k)",
            min_value=1,
            max_value=5,
            value=3,
            help="Number of historical resolution pairs retrieved to ground the draft.",
        )

        debug_mode = st.checkbox("Enable Debug Inspector", value=False, help="Show raw JSON objects and confidence internals.")

        st.divider()
        st.markdown(
            """
            **Taxonomy Intents:**
            - `delivery_delay`
            - `order_product_problem`
            - `billing_dispute`
            - `account_access`
            - `product_info_question`
            - `scam_phishing_check`
            - `service_complaint_vague`
            - `non_support`
            - `other`
            """
        )

    # ── SECTION 1: CUSTOMER MESSAGE & INPUT ──────────────────────────────────
    st.markdown("### 💬 Customer Inquiry")

    # Ensure session state variables exist
    if "customer_msg_input" not in st.session_state:
        st.session_state["customer_msg_input"] = ""
    if "conv_context_input" not in st.session_state:
        st.session_state["conv_context_input"] = ""

    def set_preset_example(key: Optional[str]):
        if key and key in PRESET_EXAMPLES:
            st.session_state["customer_msg_input"] = PRESET_EXAMPLES[key]["message"]
            st.session_state["conv_context_input"] = PRESET_EXAMPLES[key]["context"]
        else:
            st.session_state["customer_msg_input"] = ""
            st.session_state["conv_context_input"] = ""

    # Quick example pill buttons
    st.caption("✨ **Quick Example Presets** *(click to load, or type custom message below)*:")
    p_col1, p_col2, p_col3, p_col4 = st.columns([1.2, 1.2, 1.2, 0.6])
    with p_col1:
        if st.button("📦 Delivery Delay", use_container_width=True, help="Load delivery delay example"):
            set_preset_example("delivery")
            st.rerun()
    with p_col2:
        if st.button("🔒 Account Security", use_container_width=True, help="Load account security & compromised credentials example"):
            set_preset_example("security")
            st.rerun()
    with p_col3:
        if st.button("💳 Billing Dispute", use_container_width=True, help="Load unauthorized billing charge example"):
            set_preset_example("billing")
            st.rerun()
    with p_col4:
        if st.button("🧹 Clear", use_container_width=True, help="Clear text areas"):
            set_preset_example(None)
            st.rerun()

    # Dynamic text inputs bound directly to session_state
    col_in1, col_in2 = st.columns([3, 2], gap="medium")
    with col_in1:
        customer_msg = st.text_area(
            "Customer message *",
            key="customer_msg_input",
            placeholder="Type or paste any customer tweet / inquiry here (e.g., 'My package arrived damaged, can I get a replacement?')...",
            height=135,
            help="The primary inbound customer tweet or question to triage.",
        )
    with col_in2:
        conv_context = st.text_area(
            "Conversation context (optional)",
            key="conv_context_input",
            placeholder="Optional multi-turn history...\n[Customer]: Earlier tweet...\n[AmazonHelp]: Prior response...",
            height=135,
            help="Optional multi-turn conversation history preceding this message.",
        )

    col_btn, _ = st.columns([1, 2])
    with col_btn:
        analyze_btn = st.button("🚀 Analyze Request", type="primary", use_container_width=True)

    if not analyze_btn:
        if not customer_msg.strip():
            st.info("💡 Type any customer message above or click one of the quick preset buttons, then click **🚀 Analyze Request**.")
        else:
            st.info("💡 Ready! Click **🚀 Analyze Request** to process your inquiry through the AI agent pipeline.")
        st.stop()

    if not customer_msg.strip():
        st.warning("⚠️ Please enter a customer message to analyze.")
        st.stop()

    # Verify prerequisites before execution
    if not has_api:
        st.error("❌ LLM API key is not configured. Please add `groq_api` or `OPENAI_API_KEY` to `.env` to run the agent.")
        st.stop()

    if retriever is None:
        st.error("❌ Required FAISS index files are missing (`data/index/amazonhelp.faiss`). Please run `python src/build_index.py` first.")
        st.stop()

    # ── PIPELINE EXECUTION ───────────────────────────────────────────────────
    client = clients[0] if clients else None

    # Reconstruct thread structure
    full_thread_text = customer_msg.strip()
    if conv_context.strip():
        full_thread_text = f"{conv_context.strip()}\n[Customer]: {customer_msg.strip()}"

    thread_obj = {
        "thread_id": 999001,
        "customer_message": customer_msg.strip(),
        "full_thread_text": full_thread_text,
        "turns": [
            {"author_id": "Customer", "text": customer_msg.strip(), "inbound": True}
        ],
        "n_turns": full_thread_text.count("[Customer]:") + full_thread_text.count("[AmazonHelp]:"),
    }

    with st.spinner("Processing through support pipeline (Classification → Grounding → Drafting → Routing)..."):
        # 1. Intent Classification
        try:
            intent_res = classify_intent(thread_obj, model=selected_model, client=client)
        except Exception as exc:
            st.error(f"Intent Classification failed: {exc}")
            intent_res = IntentResult(intent="other", confidence=0.50, reasoning=f"Fallback due to API error: {exc}")

        # 2. Semantic Retrieval
        try:
            retrieved_examples = retriever.retrieve(customer_msg.strip(), k=top_k, dedup_macros=True)
        except Exception as exc:
            st.error(f"Semantic Retrieval failed: {exc}")
            retrieved_examples = []

        # 3. Grounded Reply Drafting
        try:
            draft_res = draft_reply(
                thread=thread_obj,
                intent=intent_res,
                retrieved_examples=retrieved_examples,
                model=selected_model,
                client=client,
                k=top_k,
            )
        except Exception as exc:
            st.error(f"Reply Generation failed: {exc}")
            draft_res = None

        # 4. Hybrid Escalation Routing
        try:
            route_res = route_thread(
                thread=thread_obj,
                intent=intent_res.intent,
                intent_confidence=intent_res.confidence,
                client=client,
                model=selected_model,
            )
        except Exception as exc:
            st.error(f"Routing evaluation failed: {exc}")
            route_res = RoutingDecision(
                decision="escalate",
                reason=f"Routing error: {exc}",
                triggered_rule="rule_error_fallback",
                confidence=0.0,
                source="error",
            )

    st.divider()

    # ── SECTION 2 & SECTION 5: INTENT & ROUTING HIGHLIGHTS ───────────────────
    col_out1, col_out2 = st.columns([1, 1], gap="large")

    with col_out1:
        st.markdown("### 🏷️ Intent Classification")
        with st.container():
            col_m1, col_m2 = st.columns([3, 2])
            with col_m1:
                st.metric("Predicted Intent", intent_res.intent.replace("_", " ").title())
            with col_m2:
                conf_pct = int(round(intent_res.confidence * 100))
                st.metric("Confidence", f"{conf_pct}%")

            st.markdown(f"**Reasoning:** *{intent_res.reasoning}*")

    with col_out2:
        st.markdown("### 🚦 Escalation & Routing Decision")
        with st.container():
            col_r1, col_r2 = st.columns([3, 2])
            with col_r1:
                if route_res.decision == "escalate":
                    st.markdown('<div class="badge-escalate">🚨 ESCALATE TO HUMAN</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="badge-auto">✅ AUTO-HANDLE BY AI</div>', unsafe_allow_html=True)
            with col_r2:
                r_conf_pct = int(round(route_res.confidence * 100)) if route_res.confidence else 100
                st.metric("Routing Confidence", f"{r_conf_pct}%")

            st.markdown(f"**Policy Reason:** {route_res.reason}")
            if route_res.triggered_rule:
                st.caption(f"🔒 Triggered Deterministic Safety Rule: `{route_res.triggered_rule}`")

    st.divider()

    # ── SECTION 4: DRAFTED REPLY ─────────────────────────────────────────────
    st.markdown("### ✍️ Suggested Agent Reply (Grounded)")
    if draft_res:
        reply_text = draft_res.reply_text
        char_count = len(reply_text)
        
        st.markdown(
            f"""<div class="reply-box">{reply_text}</div>""",
            unsafe_allow_html=True,
        )
        
        col_c1, col_c2 = st.columns([3, 1])
        with col_c1:
            if char_count <= 280:
                st.caption(f"📏 Character Count: **{char_count} / 280** (Complies with standard Twitter limit)")
            else:
                st.caption(f"⚠️ Character Count: **{char_count} / 280** (Exceeds 280-char limit)")
        with col_c2:
            st.caption(f"🤖 Model: `{draft_res.model_used}`")
    else:
        st.warning("Could not generate drafted reply.")

    st.divider()

    # ── SECTION 3: HISTORICAL RETRIEVAL PRECEDENTS ───────────────────────────
    st.markdown("### 🔍 Historical Context & Grounding Precedents")
    st.caption(f"Top {len(retrieved_examples)} deduplicated historical AmazonHelp conversations retrieved via FAISS dense search:")

    if retrieved_examples:
        for idx, ex in enumerate(retrieved_examples, 1):
            with st.expander(f"Precedent #{idx} — Thread #{ex.thread_id} (Cosine Similarity: {ex.similarity_score:.4f})", expanded=(idx == 1)):
                col_ex1, col_ex2 = st.columns([1, 1], gap="medium")
                with col_ex1:
                    st.markdown("**👤 Historical Customer Query:**")
                    st.info(ex.customer_message)
                with col_ex2:
                    st.markdown("**📦 AmazonHelp Verified Resolution:**")
                    st.success(ex.brand_resolution)
    else:
        st.write("No historical precedents retrieved.")

    st.divider()

    # ── SECTION 6: SAFETY / RISK SIGNALS ─────────────────────────────────────
    st.markdown("### 🛡️ Safety & Risk Signal Evaluation")
    
    # Audit risk dimensions
    risk_checks = [
        ("Account Security & Credentials", "rule_security", "Detects hacked, stolen, or unauthorized account modifications."),
        ("Financial Action & Billing", "rule_financial_action", "Detects unauthorized charges, bank deductions, or cash disputes."),
        ("PII & Private Data Exposure", "rule_pii_exposure", "Flags public phone numbers, emails, or order IDs."),
        ("Repeated Service Failure", "rule_repeated_failure", "Detects multi-turn customer frustration and broken promises."),
        ("Legal or Churn Threat", "rule_legal_or_churn_threat", "Detects consumer court threats, lawyer mentions, or cancellation rants."),
    ]

    col_risk_list = st.columns(len(risk_checks))
    for col, (label, rule_key, desc) in zip(col_risk_list, risk_checks):
        with col:
            is_active = (route_res.triggered_rule == rule_key)
            if is_active:
                st.error(f"🚨 **{label}**\n\n*Triggered*")
            else:
                st.success(f"✅ **{label}**\n\n*Clear*")
            st.caption(desc)

    st.divider()

    # ── SECTION 7: PIPELINE SUMMARY & FLOW ───────────────────────────────────
    st.markdown("### 🔄 End-to-End Pipeline Trace")
    
    col_p1, col_p2, col_p3, col_p4, col_p5, col_p6 = st.columns(6)
    with col_p1:
        st.markdown('<div class="pipeline-step">1. Customer Request<br>📥 Input</div>', unsafe_allow_html=True)
    with col_p2:
        st.markdown(f'<div class="pipeline-step">2. Intent Classify<br>🏷️ {intent_res.intent}</div>', unsafe_allow_html=True)
    with col_p3:
        st.markdown(f'<div class="pipeline-step">3. FAISS Retrieval<br>🔍 Top-{len(retrieved_examples)} QA</div>', unsafe_allow_html=True)
    with col_p4:
        st.markdown('<div class="pipeline-step">4. Grounded Draft<br>✍️ Twitter Reply</div>', unsafe_allow_html=True)
    with col_p5:
        st.markdown(f'<div class="pipeline-step">5. Safety Rules<br>🛡️ {route_res.source.upper()}</div>', unsafe_allow_html=True)
    with col_p6:
        badge_style = "color:#721c24; background:#f8d7da;" if route_res.decision == "escalate" else "color:#155724; background:#d4edda;"
        st.markdown(f'<div class="pipeline-step" style="{badge_style}">6. Outcome<br><b>{route_res.decision.upper()}</b></div>', unsafe_allow_html=True)

    if debug_mode:
        st.divider()
        st.subheader("🛠️ Debug Inspector (Raw Payloads)")
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.write("**Intent Result JSON:**")
            st.json(intent_res.to_dict())
            st.write("**Routing Decision JSON:**")
            st.json({
                "decision": route_res.decision,
                "reason": route_res.reason,
                "triggered_rule": route_res.triggered_rule,
                "confidence": route_res.confidence,
                "source": route_res.source,
                "model": route_res.model,
            })
        with col_d2:
            st.write("**Draft Reply JSON:**")
            st.json(draft_res.to_dict() if draft_res else {})
            st.write("**Retrieved Precedents JSON:**")
            st.json([ex.to_dict() for ex in retrieved_examples])

    # Footer disclaimer
    st.caption("ℹ️ *Disclaimer: AI-generated support assistance. Escalate sensitive or unresolved cases to a human agent.*")


if __name__ == "__main__":
    main()
