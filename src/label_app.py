"""
label_app.py - AI-Assisted Golden Set Labeling Tool for Streamlit.

Run with:
    streamlit run src/label_app.py

Features:
- AI-Assisted Pre-fill: Pre-populates fields with AI draft suggestions from data/golden/_ai_draft_labels.csv.
- Suggestion Badge & Banner: Clear yellow visual callout that values are draft suggestions, not saved answers.
- Active Touch Tracking: Tracks whether intent, escalate, and reason have been actively inspected/interacted with.
  "Save & Next" is disabled until required fields are touched.
- Audit Logging: Tracks per-thread {thread_id, ai_suggested_intent, my_final_intent, intent_was_changed,
  ai_suggested_escalate, my_final_escalate, escalate_was_changed, time_spent_seconds} in data/golden/labeling_audit_log.csv.
- Incremental Persistence: Auto-saves to data/golden/golden_set_working.csv immediately on each Save.
- Completion Screen: Computes change rates, total/average review time, Cohen's kappa vs AI suggestions,
  and deep-dives into all threads where AI suggestions were changed.
"""

import csv
import os
import re
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# Paths
WORKING_PATH = "data/golden/golden_set_working.csv"
TEMPLATE_PATH = "data/golden/golden_set_template.csv"
AI_DRAFT_PATH = "data/golden/_ai_draft_labels.csv"
AUDIT_LOG_PATH = "data/golden/labeling_audit_log.csv"

# Valid Taxonomy Options
INTENT_OPTIONS = [
    "delivery_delay",
    "billing_dispute",
    "order_product_problem",
    "account_access",
    "product_info_question",
    "service_complaint_vague",
    "scam_phishing_check",
    "non_support",
    "other",
]

ESCALATE_OPTIONS = [
    "auto_handle",
    "escalate",
]


def ensure_data_setup():
    """Ensures working file, AI draft file, and audit log exist."""
    os.makedirs("data/golden", exist_ok=True)

    # 1. Check AI draft reference
    if not os.path.exists(AI_DRAFT_PATH):
        if not os.path.exists(TEMPLATE_PATH):
            st.error(f"Missing required file: {TEMPLATE_PATH}")
            st.stop()
        with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
        with open(AI_DRAFT_PATH, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["thread_id", "gold_intent", "gold_escalate", "gold_reason", "gold_reply_notes"])
            for r in reader:
                writer.writerow([
                    r["thread_id"],
                    r.get("gold_intent", ""),
                    r.get("gold_escalate", ""),
                    r.get("gold_reason", ""),
                    r.get("gold_reply_notes", ""),
                ])

    # 2. Check working file
    if not os.path.exists(WORKING_PATH):
        with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
        with open(WORKING_PATH, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["thread_id", "full_thread_text", "n_turns", "my_intent", "my_escalate", "my_reason", "my_reply_notes"])
            for r in reader:
                writer.writerow([
                    r["thread_id"],
                    r["full_thread_text"],
                    r["n_turns"],
                    "",
                    "",
                    "",
                    "",
                ])

    # 3. Check audit log
    if not os.path.exists(AUDIT_LOG_PATH):
        with open(AUDIT_LOG_PATH, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "thread_id",
                "ai_suggested_intent",
                "my_final_intent",
                "intent_was_changed",
                "ai_suggested_escalate",
                "my_final_escalate",
                "escalate_was_changed",
                "time_spent_seconds",
            ])


def load_datasets() -> Tuple[pd.DataFrame, Dict[str, Dict[str, str]]]:
    ensure_data_setup()
    working_df = pd.read_csv(WORKING_PATH, dtype=str, keep_default_na=False)
    
    # Load AI suggestions as a dictionary keyed by thread_id
    ai_draft_df = pd.read_csv(AI_DRAFT_PATH, dtype=str, keep_default_na=False)
    ai_suggestions = {}
    for _, r in ai_draft_df.iterrows():
        ai_suggestions[str(r["thread_id"])] = {
            "intent": r.get("gold_intent", ""),
            "escalate": r.get("gold_escalate", ""),
            "reason": r.get("gold_reason", ""),
            "reply_notes": r.get("gold_reply_notes", ""),
        }

    return working_df, ai_suggestions


def save_single_row(
    working_df: pd.DataFrame,
    idx: int,
    tid: str,
    final_intent: str,
    final_escalate: str,
    final_reason: str,
    final_reply_notes: str,
    ai_sug: Dict[str, str],
    time_spent: float,
):
    """Saves to golden_set_working.csv and appends/updates labeling_audit_log.csv."""
    # 1. Update working DataFrame and CSV
    working_df.at[idx, "my_intent"] = final_intent
    working_df.at[idx, "my_escalate"] = final_escalate
    working_df.at[idx, "my_reason"] = final_reason.strip()
    working_df.at[idx, "my_reply_notes"] = final_reply_notes.strip()
    working_df.to_csv(WORKING_PATH, index=False, encoding="utf-8")

    # 2. Update audit log
    intent_changed = (final_intent != ai_sug.get("intent", ""))
    esc_changed = (final_escalate != ai_sug.get("escalate", ""))

    audit_rows = []
    if os.path.exists(AUDIT_LOG_PATH):
        with open(AUDIT_LOG_PATH, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            audit_rows = list(reader)

    # Filter out prior entry for this thread_id if re-labeling
    audit_dict = {r["thread_id"]: r for r in audit_rows}
    audit_dict[tid] = {
        "thread_id": tid,
        "ai_suggested_intent": ai_sug.get("intent", ""),
        "my_final_intent": final_intent,
        "intent_was_changed": str(intent_changed),
        "ai_suggested_escalate": ai_sug.get("escalate", ""),
        "my_final_escalate": final_escalate,
        "escalate_was_changed": str(esc_changed),
        "time_spent_seconds": f"{time_spent:.1f}",
    }

    with open(AUDIT_LOG_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "thread_id",
            "ai_suggested_intent",
            "my_final_intent",
            "intent_was_changed",
            "ai_suggested_escalate",
            "my_final_escalate",
            "escalate_was_changed",
            "time_spent_seconds",
        ])
        for entry in audit_dict.values():
            writer.writerow([
                entry["thread_id"],
                entry["ai_suggested_intent"],
                entry["my_final_intent"],
                entry["intent_was_changed"],
                entry["ai_suggested_escalate"],
                entry["my_final_escalate"],
                entry["escalate_was_changed"],
                entry["time_spent_seconds"],
            ])


def render_chat_bubbles(full_text: str):
    """Renders thread turns in readable chat-bubble style."""
    turns = re.split(r"\n(?=\[(?:Customer|AmazonHelp)\]:)", full_text.strip())
    
    for turn in turns:
        t = turn.strip()
        if not t:
            continue
        if t.startswith("[Customer]:"):
            body = t[len("[Customer]:"):].strip()
            st.markdown(
                f"""
                <div style="background-color: #f0f4f8; color: #1e293b; padding: 12px 16px; 
                            border-radius: 12px 12px 12px 2px; margin-bottom: 10px; 
                            border-left: 5px solid #3b82f6; max-width: 88%;">
                    <div style="font-size: 0.78rem; font-weight: 700; color: #1d4ed8; margin-bottom: 4px;">👤 CUSTOMER</div>
                    <div style="font-size: 0.95rem; line-height: 1.45; white-space: pre-wrap;">{body}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif t.startswith("[AmazonHelp]:"):
            body = t[len("[AmazonHelp]:"):].strip()
            st.markdown(
                f"""
                <div style="background-color: #fff7ed; color: #1e293b; padding: 12px 16px; 
                            border-radius: 12px 12px 2px 12px; margin-bottom: 10px; margin-left: auto;
                            border-right: 5px solid #f97316; max-width: 88%;">
                    <div style="font-size: 0.78rem; font-weight: 700; color: #c2410c; margin-bottom: 4px; text-align: right;">🏢 AMAZONHELP</div>
                    <div style="font-size: 0.95rem; line-height: 1.45; white-space: pre-wrap;">{body}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.text(t)


def render_completion_view(working_df: pd.DataFrame):
    st.success("🎉 All 220 threads in the Golden Evaluation Set have been labeled!")
    st.markdown("## 📊 Ground Truth & AI Agreement Audit")

    if not os.path.exists(AUDIT_LOG_PATH):
        st.warning("Audit log not found yet.")
        return

    audit_df = pd.read_csv(AUDIT_LOG_PATH)
    if audit_df.empty:
        st.warning("Audit log is empty.")
        return

    # Total and average time spent
    audit_df["time_spent_seconds"] = pd.to_numeric(audit_df["time_spent_seconds"], errors="coerce").fillna(0.0)
    total_time_sec = audit_df["time_spent_seconds"].sum()
    avg_time_sec = audit_df["time_spent_seconds"].mean()
    total_time_min = total_time_sec / 60.0

    # Changes from AI suggestion
    intent_changed_count = (audit_df["intent_was_changed"].astype(str).str.lower() == "true").sum()
    intent_changed_pct = (intent_changed_count / len(audit_df)) * 100

    esc_changed_count = (audit_df["escalate_was_changed"].astype(str).str.lower() == "true").sum()
    esc_changed_pct = (esc_changed_count / len(audit_df)) * 100

    # Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Review Time", f"{total_time_min:.1f} min", f"{total_time_sec:.0f}s total")
    m2.metric("Avg Time / Thread", f"{avg_time_sec:.1f}s", "per thread")
    m3.metric("Intent Changed from AI", f"{intent_changed_pct:.1f}%", f"{intent_changed_count} / {len(audit_df)}")
    m4.metric("Escalate Changed from AI", f"{esc_changed_pct:.1f}%", f"{esc_changed_count} / {len(audit_df)}")

    # Cohen's Kappa
    try:
        from sklearn.metrics import cohen_kappa_score
        kappa_intent = cohen_kappa_score(audit_df["my_final_intent"], audit_df["ai_suggested_intent"])
        kappa_esc = cohen_kappa_score(audit_df["my_final_escalate"], audit_df["ai_suggested_escalate"])
    except Exception:
        kappa_intent = None
        kappa_esc = None

    k1, k2 = st.columns(2)
    k1.metric("Intent Cohen's Kappa (Final vs AI)", f"{kappa_intent:.4f}" if kappa_intent is not None else "N/A", "Agreement beyond chance")
    k2.metric("Escalate Cohen's Kappa (Final vs AI)", f"{kappa_esc:.4f}" if kappa_esc is not None else "N/A", "Routing agreement")

    # Distributions
    colA, colB = st.columns(2)
    with colA:
        st.subheader("Your Intent Distribution")
        icounts = Counter(audit_df["my_final_intent"])
        idf = pd.DataFrame([
            {"Intent": k, "Count": v, "Percentage": f"{(v / len(audit_df)) * 100:.1f}%"}
            for k, v in sorted(icounts.items(), key=lambda x: -x[1])
        ])
        st.dataframe(idf, use_container_width=True, hide_index=True)

    with colB:
        st.subheader("Your Escalation Split")
        ecounts = Counter(audit_df["my_final_escalate"])
        edf = pd.DataFrame([
            {"Decision": k, "Count": v, "Percentage": f"{(v / len(audit_df)) * 100:.1f}%"}
            for k, v in sorted(ecounts.items(), key=lambda x: -x[1])
        ])
        st.dataframe(edf, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("🔍 Intent Confusion Matrix (Rows: Your Ground Truth, Columns: AI Suggestion)")
    ct = pd.crosstab(audit_df["my_final_intent"], audit_df["ai_suggested_intent"], margins=True)
    st.dataframe(ct, use_container_width=True)

    st.markdown("---")
    st.subheader("🚩 Disagreements Breakdown (Threads where you changed the AI's suggestion)")
    audit_df["thread_id"] = audit_df["thread_id"].astype(str)
    changed_df = audit_df[
        (audit_df["intent_was_changed"].astype(str).str.lower() == "true") |
        (audit_df["escalate_was_changed"].astype(str).str.lower() == "true")
    ].copy()
    st.write(f"Total threads with changes: **{len(changed_df)}**")

    w_df = working_df[["thread_id", "full_thread_text", "my_reason"]].copy()
    w_df["thread_id"] = w_df["thread_id"].astype(str)

    # Join back with full thread text from working_df for review
    detailed_changed = pd.merge(changed_df, w_df, on="thread_id", how="left")

    for _, row in detailed_changed.iterrows():
        intent_diff = f"Intent: {row['ai_suggested_intent']} ➔ **{row['my_final_intent']}**" if row['ai_suggested_intent'] != row['my_final_intent'] else "Intent: Unchanged"
        esc_diff = f"Escalate: {row['ai_suggested_escalate']} ➔ **{row['my_final_escalate']}**" if row['ai_suggested_escalate'] != row['my_final_escalate'] else "Escalate: Unchanged"
        
        with st.expander(f"Thread {row['thread_id']} &nbsp;|&nbsp; {intent_diff} &nbsp;|&nbsp; {esc_diff}"):
            st.markdown(f"**Your Ground Truth Reason**: {row.get('my_reason', '')}")
            st.markdown(f"**Time Spent**: {row['time_spent_seconds']}s")
            st.markdown("**Conversation:**")
            st.text(row["full_thread_text"])


def main():
    st.set_page_config(
        page_title="AI-Assisted Golden Set Labeler",
        page_icon="🏷️",
        layout="wide",
    )

    working_df, ai_suggestions = load_datasets()
    total_threads = len(working_df)

    # Manage Session State
    if "current_idx" not in st.session_state:
        # Start at first unlabeled thread
        unlabeled_indices = working_df.index[working_df["my_intent"] == ""].tolist()
        st.session_state.current_idx = unlabeled_indices[0] if unlabeled_indices else 0

    if st.session_state.current_idx >= total_threads:
        st.session_state.current_idx = total_threads - 1
    if st.session_state.current_idx < 0:
        st.session_state.current_idx = 0

    idx = st.session_state.current_idx
    row = working_df.iloc[idx]
    tid = str(row["thread_id"])
    ai_sug = ai_suggestions.get(tid, {
        "intent": "delivery_delay",
        "escalate": "auto_handle",
        "reason": "",
        "reply_notes": "",
    })

    # Track time per thread
    thread_time_key = f"time_start_{tid}"
    if thread_time_key not in st.session_state:
        st.session_state[thread_time_key] = time.time()

    # Field touch tracking state per thread
    touch_key = f"touched_state_{tid}"
    if touch_key not in st.session_state:
        # If thread was already labeled in a prior session, mark touched
        already_saved = bool(row["my_intent"])
        st.session_state[touch_key] = {
            "intent": already_saved,
            "escalate": already_saved,
            "reason": already_saved,
            "notes": already_saved,
        }

    touched = st.session_state[touch_key]

    # Progress stats
    labeled_mask = working_df["my_intent"] != ""
    completed_count = int(labeled_mask.sum())
    remaining_count = total_threads - completed_count
    progress_val = completed_count / total_threads if total_threads > 0 else 0.0

    # --- SIDEBAR ---
    with st.sidebar:
        st.title("🏷️ Golden Set Labeler")
        st.metric("Progress", f"{completed_count} / {total_threads}", f"{remaining_count} remaining")
        st.progress(progress_val)

        options = [
            f"{i + 1}. Thread {working_df.iloc[i]['thread_id']} {'✓' if working_df.iloc[i]['my_intent'] else '○'}"
            for i in range(total_threads)
        ]
        selected_option = st.selectbox(
            "Jump to Thread:",
            options=range(total_threads),
            format_func=lambda i: options[i],
            index=idx,
            key="jump_selector",
        )
        if selected_option != idx:
            st.session_state.current_idx = selected_option
            st.session_state[f"time_start_{working_df.iloc[selected_option]['thread_id']}"] = time.time()
            st.rerun()

        unlabeled_indices = working_df.index[working_df["my_intent"] == ""].tolist()
        if unlabeled_indices:
            if st.button("⏩ Jump to Next Unlabeled", use_container_width=True):
                st.session_state.current_idx = unlabeled_indices[0]
                st.session_state[f"time_start_{working_df.iloc[unlabeled_indices[0]]['thread_id']}"] = time.time()
                st.rerun()

        st.markdown("---")
        st.caption(f"Working File: `{WORKING_PATH}`")
        st.caption(f"Audit Log: `{AUDIT_LOG_PATH}`")
        st.caption("AI-assisted pre-fill active with touch verification.")

    # --- COMPLETION VIEW ---
    if completed_count == total_threads:
        show_completion = st.checkbox("Show Final Evaluation & Benchmark Audit", value=True)
        if show_completion:
            render_completion_view(working_df)
            st.markdown("---")
            st.markdown("### Edit Specific Thread")

    # --- TOP PROGRESS BAR ---
    st.markdown(
        f"### Thread **{idx + 1}** of **{total_threads}** &nbsp;•&nbsp; `ID: {tid}` &nbsp;•&nbsp; "
        f"<span style='color: #059669; font-weight: 600;'>{completed_count} labeled</span>, "
        f"<span style='color: #dc2626; font-weight: 600;'>{remaining_count} remaining</span>",
        unsafe_allow_html=True,
    )
    st.progress(progress_val)

    # --- MAIN TWO-COLUMN LAYOUT ---
    left_col, right_col = st.columns([1.05, 0.95])

    # Left Column: Conversation Thread
    with left_col:
        st.subheader("💬 Full Conversation Thread")
        render_chat_bubbles(row["full_thread_text"])

    # Right Column: AI-Assisted Pre-fill Form
    with right_col:
        st.subheader("📝 Assign Ground Truth")

        # 1. AI Suggestion Yellow Callout Banner
        st.markdown(
            """
            <div style="background-color: #fefce8; border: 1.5px solid #facc15; border-left: 6px solid #eab308; 
                        padding: 10px 14px; border-radius: 8px; margin-bottom: 14px;">
                <div style="font-weight: 700; color: #854d0e; font-size: 0.88rem; display: flex; align-items: center; gap: 6px;">
                    ⚠️ AI SUGGESTION — REVIEW BEFORE SAVING
                </div>
                <div style="color: #713f12; font-size: 0.82rem; margin-top: 3px; line-height: 1.35;">
                    The fields below are pre-populated from the AI draft baseline. You must actively inspect and confirm/edit each field before saving.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Quick action: Accept All AI Suggestions in 1-click
        col_acc1, col_acc2 = st.columns([1.5, 1])
        with col_acc1:
            if st.button("⚡ Accept All AI Suggestions for this Thread", use_container_width=True):
                touched["intent"] = True
                touched["escalate"] = True
                touched["reason"] = True
                touched["notes"] = True
                st.session_state[f"val_intent_{tid}"] = ai_sug["intent"]
                st.session_state[f"val_escalate_{tid}"] = ai_sug["escalate"]
                st.session_state[f"val_reason_{tid}"] = ai_sug["reason"]
                st.session_state[f"val_notes_{tid}"] = ai_sug["reply_notes"]
                st.rerun()

        # Initial values: check if user already saved or use AI suggestion
        initial_intent = row["my_intent"] if row["my_intent"] in INTENT_OPTIONS else ai_sug["intent"]
        initial_esc = row["my_escalate"] if row["my_escalate"] in ESCALATE_OPTIONS else ai_sug["escalate"]
        initial_reason = row["my_reason"] if row["my_reason"] else ai_sug["reason"]
        initial_notes = row["my_reply_notes"] if row["my_reply_notes"] else ai_sug["reply_notes"]

        if f"val_intent_{tid}" not in st.session_state:
            st.session_state[f"val_intent_{tid}"] = initial_intent
        if f"val_escalate_{tid}" not in st.session_state:
            st.session_state[f"val_escalate_{tid}"] = initial_esc
        if f"val_reason_{tid}" not in st.session_state:
            st.session_state[f"val_reason_{tid}"] = initial_reason
        if f"val_notes_{tid}" not in st.session_state:
            st.session_state[f"val_notes_{tid}"] = initial_notes

        # Field 1: Intent
        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 10px; margin-bottom: 2px;">
                <span style="font-weight: 700; font-size: 0.92rem;">1. Primary Opening Intent</span>
                <span style="font-size: 0.78rem; font-weight: 600; padding: 2px 8px; border-radius: 4px; 
                             background-color: {'#dcfce7; color: #15803d;' if touched['intent'] else '#fee2e2; color: #b91c1c;'}">
                    {'🟢 Confirmed' if touched['intent'] else '🔴 Untouched'}
                </span>
            </div>
            <div style="font-size: 0.78rem; color: #a16207; background-color: #fef08a; padding: 2px 6px; border-radius: 4px; display: inline-block; margin-bottom: 6px;">
                AI Suggested: <b>{ai_sug['intent']}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

        def on_intent_change():
            st.session_state[touch_key]["intent"] = True

        intent_idx = INTENT_OPTIONS.index(st.session_state[f"val_intent_{tid}"]) if st.session_state[f"val_intent_{tid}"] in INTENT_OPTIONS else 0
        selected_intent = st.radio(
            "Intent Radio",
            options=INTENT_OPTIONS,
            index=intent_idx,
            key=f"radio_intent_{tid}",
            label_visibility="collapsed",
            on_change=on_intent_change,
        )
        st.session_state[f"val_intent_{tid}"] = selected_intent

        col_c1, _ = st.columns([1.2, 1.8])
        with col_c1:
            if not touched["intent"]:
                if st.button(f"✓ Confirm: {selected_intent}", key=f"btn_confirm_intent_{tid}"):
                    touched["intent"] = True
                    st.rerun()

        # Field 2: Escalation
        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 16px; margin-bottom: 2px;">
                <span style="font-weight: 700; font-size: 0.92rem;">2. Routing Decision (Full Context)</span>
                <span style="font-size: 0.78rem; font-weight: 600; padding: 2px 8px; border-radius: 4px; 
                             background-color: {'#dcfce7; color: #15803d;' if touched['escalate'] else '#fee2e2; color: #b91c1c;'}">
                    {'🟢 Confirmed' if touched['escalate'] else '🔴 Untouched'}
                </span>
            </div>
            <div style="font-size: 0.78rem; color: #a16207; background-color: #fef08a; padding: 2px 6px; border-radius: 4px; display: inline-block; margin-bottom: 6px;">
                AI Suggested: <b>{ai_sug['escalate']}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

        def on_esc_change():
            st.session_state[touch_key]["escalate"] = True

        esc_idx = ESCALATE_OPTIONS.index(st.session_state[f"val_escalate_{tid}"]) if st.session_state[f"val_escalate_{tid}"] in ESCALATE_OPTIONS else 0
        selected_escalate = st.radio(
            "Escalate Radio",
            options=ESCALATE_OPTIONS,
            index=esc_idx,
            key=f"radio_esc_{tid}",
            label_visibility="collapsed",
            on_change=on_esc_change,
        )
        st.session_state[f"val_escalate_{tid}"] = selected_escalate

        col_c2, _ = st.columns([1.2, 1.8])
        with col_c2:
            if not touched["escalate"]:
                if st.button(f"✓ Confirm: {selected_escalate}", key=f"btn_confirm_esc_{tid}"):
                    touched["escalate"] = True
                    st.rerun()

        # Field 3: Reason
        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 16px; margin-bottom: 2px;">
                <span style="font-weight: 700; font-size: 0.92rem;">3. Escalation Reason (Required)</span>
                <span style="font-size: 0.78rem; font-weight: 600; padding: 2px 8px; border-radius: 4px; 
                             background-color: {'#dcfce7; color: #15803d;' if touched['reason'] else '#fee2e2; color: #b91c1c;'}">
                    {'🟢 Confirmed' if touched['reason'] else '🔴 Untouched'}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        def on_reason_change():
            st.session_state[touch_key]["reason"] = True

        selected_reason = st.text_area(
            "Reason Text Area",
            value=st.session_state[f"val_reason_{tid}"],
            key=f"text_reason_{tid}",
            height=85,
            label_visibility="collapsed",
            on_change=on_reason_change,
        )
        st.session_state[f"val_reason_{tid}"] = selected_reason

        col_c3, _ = st.columns([1.2, 1.8])
        with col_c3:
            if not touched["reason"]:
                if st.button("✓ Confirm Reason", key=f"btn_confirm_reason_{tid}"):
                    touched["reason"] = True
                    st.rerun()

        # Field 4: Reply Notes
        st.markdown(
            """
            <div style="font-weight: 700; font-size: 0.92rem; margin-top: 14px; margin-bottom: 2px;">
                4. Reply Notes (Optional Guidance)
            </div>
            """,
            unsafe_allow_html=True,
        )

        def on_notes_change():
            st.session_state[touch_key]["notes"] = True

        selected_notes = st.text_area(
            "Notes Text Area",
            value=st.session_state[f"val_notes_{tid}"],
            key=f"text_notes_{tid}",
            height=70,
            label_visibility="collapsed",
            on_change=on_notes_change,
        )
        st.session_state[f"val_notes_{tid}"] = selected_notes

        # Validation Check: are all required fields touched and non-empty?
        is_ready_to_save = (
            touched["intent"] and
            touched["escalate"] and
            touched["reason"] and
            bool(selected_reason.strip())
        )

        st.markdown("---")

        # Action Buttons
        btn_col1, btn_col2, btn_col3 = st.columns([1.2, 0.9, 1.0])

        with btn_col1:
            if is_ready_to_save:
                if st.button("💾 Save & Next", type="primary", use_container_width=True):
                    time_spent = time.time() - st.session_state.get(thread_time_key, time.time())
                    save_single_row(
                        working_df=working_df,
                        idx=idx,
                        tid=tid,
                        final_intent=selected_intent,
                        final_escalate=selected_escalate,
                        final_reason=selected_reason,
                        final_reply_notes=selected_notes,
                        ai_sug=ai_sug,
                        time_spent=time_spent,
                    )
                    st.toast(f"Saved Thread {tid}! ✨", icon="✅")

                    # Advance to next unlabeled or next sequence
                    next_unlabeled = working_df.index[(working_df.index > idx) & (working_df["my_intent"] == "")].tolist()
                    if next_unlabeled:
                        st.session_state.current_idx = next_unlabeled[0]
                    elif idx + 1 < total_threads:
                        st.session_state.current_idx = idx + 1

                    # Reset timer for next thread
                    new_tid = str(working_df.iloc[st.session_state.current_idx]["thread_id"])
                    st.session_state[f"time_start_{new_tid}"] = time.time()
                    st.rerun()
            else:
                st.button("💾 Save & Next", disabled=True, use_container_width=True, help="Review and confirm all fields to enable.")
                missing_items = []
                if not touched["intent"]:
                    missing_items.append("Intent")
                if not touched["escalate"]:
                    missing_items.append("Escalation")
                if not touched["reason"]:
                    missing_items.append("Reason")
                elif not selected_reason.strip():
                    missing_items.append("Reason text cannot be empty")
                
                st.caption(f"⚠️ Untouched fields: **{', '.join(missing_items)}** (click radio or ✓ Confirm button)")

        with btn_col2:
            if st.button("⬅️ Back", use_container_width=True):
                if idx > 0:
                    st.session_state.current_idx = idx - 1
                    prev_tid = str(working_df.iloc[idx - 1]["thread_id"])
                    st.session_state[f"time_start_{prev_tid}"] = time.time()
                    st.rerun()
                else:
                    st.info("Already at first thread.")

        with btn_col3:
            if st.button("➡️ Skip to Next", use_container_width=True):
                if idx + 1 < total_threads:
                    st.session_state.current_idx = idx + 1
                    next_tid = str(working_df.iloc[idx + 1]["thread_id"])
                    st.session_state[f"time_start_{next_tid}"] = time.time()
                    st.rerun()
                else:
                    st.info("Already at last thread.")

        # Interactive DOM event listeners for radio clicks and textarea focus
        components.html(
            f"""
            <script>
            const doc = window.parent.document;
            function attachListeners() {{
                const radios = doc.querySelectorAll('div[data-testid="stRadio"]');
                if (radios.length >= 2) {{
                    radios[0].addEventListener('click', () => {{
                        for (const b of doc.querySelectorAll('button')) {{
                            if (b.innerText && b.innerText.startsWith('✓ Confirm:') && !b.innerText.includes('escalate') && !b.innerText.includes('auto_handle')) {{
                                b.click();
                                break;
                            }}
                        }}
                    }});
                    radios[1].addEventListener('click', () => {{
                        for (const b of doc.querySelectorAll('button')) {{
                            if (b.innerText && (b.innerText.includes('auto_handle') || b.innerText.includes('escalate'))) {{
                                b.click();
                                break;
                            }}
                        }}
                    }});
                }}
                const textareas = doc.querySelectorAll('textarea');
                if (textareas.length >= 1) {{
                    textareas[0].addEventListener('focus', () => {{
                        for (const b of doc.querySelectorAll('button')) {{
                            if (b.innerText && b.innerText.includes('Confirm Reason')) {{
                                b.click();
                                break;
                            }}
                        }}
                    }});
                }}
            }}
            setTimeout(attachListeners, 250);
            </script>
            """,
            height=0,
        )

    # --- COLLAPSIBLE REFERENCE GUIDE ---
    st.markdown("---")
    with st.expander("📖 Reference: Locked Boundary Rules & Operational Routing"):
        st.markdown(
            """
            ### ⚠️ Locked Boundary Rules
            - **Rule 1: Concrete Defects vs. Vague Complaints**:
              If a customer expresses anger, venting, or harsh emotion, but names a **concrete broken thing** (e.g. app crash, video playback failure, crushed packaging, defective device), classify as **`order_product_problem`**, **NOT** `service_complaint_vague`. `service_complaint_vague` is reserved *strictly* for complaints about service/process quality where no specific defect, order, or product is named.
            - **Rule 2: Humor & Sarcasm vs. Non-Support**:
              The presence of emojis (e.g. 😂, 📦, 🙄), sarcasm, irony, or jokes does **NOT** make a message `non_support` if it still describes a real delivery, defect, or billing issue.
            - **Primary Stated Issue vs. Drift**:
              If a thread's issue drifts or a second distinct problem emerges after the opening message, label `gold_intent` based on the customer's primary/opening stated issue only. Note any secondary issue in `gold_reply_notes`.

            ### 🚦 Operational Routing (Full Thread Context)
            - **`auto_handle`**:
              - Factual policy questions, specs, compatibility, release dates.
              - Guiding to self-service tracking, carrier links, or Returns Center (`amazon.com/returns`).
              - Phishing warning (`stop-spoofing@amazon.com`).
              - Closing positive praise / banter with brand gratitude.
            - **`escalate`**:
              - **Financial**: Issuing refunds, fee reversals, investigating unauthorized card charges.
              - **Security & Privacy**: Account takeover, locked out >2 weeks, hacked credentials, customer posted card/order info publicly.
              - **Logistics Failure**: Marked as delivered but missing, driver misconduct, spoiled perishables, repeated courier pickup no-shows.
              - **High Agitation / Service Breakdown**: Repeated broken promises (e.g. promised phone callback not delivered), 3+ failed contacts.
            """
        )


if __name__ == "__main__":
    main()
