"""
src/human_judge_app.py

Streamlit Application for Blind Human Evaluation of 30 Stratified Reply Drafts.

Run with:
    streamlit run src/human_judge_app.py --server.port 8502

Features:
- Stratified sampling of exactly 30 golden threads across all 9 intent categories (fixed seed=42).
- Blind Review: NEVER displays the LLM judge's scores or flags to eliminate confirmation bias.
- Displays:
    * Conversation Thread History (styled message bubbles)
    * Human Ground-Truth Reply Guidance (gold_reply_notes from labeling stage)
    * Generated AI Agent Reply Draft (from golden_drafts.jsonl)
- 4 Scoring Sliders (1 to 5 Likert scale):
    1. Tone & Empathy
    2. Factual Correctness
    3. Completeness
    4. Safety & Privacy
- Justification / Free-text notes input.
- Incremental persistence to `data/golden/human_judge_30.csv` and `data/results/human_judged_30.jsonl`.
- Live summary & completion tracking.
"""

import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent

GOLDEN_CSV_PATH = REPO_ROOT / "data" / "golden" / "golden_set_working.csv"
DRAFTS_PATH = REPO_ROOT / "data" / "results" / "golden_drafts.jsonl"
SAMPLE_SPEC_PATH = REPO_ROOT / "data" / "golden" / "sampled_judge_30.json"
HUMAN_RESULTS_CSV = REPO_ROOT / "data" / "golden" / "human_judge_30.csv"
HUMAN_RESULTS_JSONL = REPO_ROOT / "data" / "results" / "human_judged_30.jsonl"

SAMPLE_SIZE = 30
RANDOM_SEED = 42

st.set_page_config(
    page_title="Reply Quality Human Judge",
    page_icon="⚖️",
    layout="wide",
)


def get_stratified_sample_threads() -> List[Dict[str, Any]]:
    """Loads all drafts and extracts a deterministic stratified sample of 30 threads."""
    # 1. Load working CSV metadata
    csv_meta = {}
    with open(GOLDEN_CSV_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tid = int(row["thread_id"])
            notes = row.get("my_reply_notes", "") or row.get("gold_reply_notes", "")
            csv_meta[tid] = {
                "gold_intent": row.get("my_intent", "").strip(),
                "gold_escalate": row.get("my_escalate", "").strip(),
                "gold_reply_notes": notes.strip(),
                "full_thread_text": row.get("full_thread_text", ""),
            }

    # 2. Load draft records
    drafts_by_id = {}
    with open(DRAFTS_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                drafts_by_id[int(r["thread_id"])] = r

    # 3. Check if cached sample exists, otherwise compute stratified sample
    if SAMPLE_SPEC_PATH.exists():
        with open(SAMPLE_SPEC_PATH, encoding="utf-8") as f:
            sampled_ids = json.load(f)
    else:
        # Group thread IDs by gold_intent
        by_intent: Dict[str, List[int]] = {}
        for tid, meta in csv_meta.items():
            intent = meta["gold_intent"] or "other"
            by_intent.setdefault(intent, []).append(tid)

        rng = random.Random(RANDOM_SEED)
        sampled_ids = []
        # Target ~3-4 per intent across 9 intents
        intents_sorted = sorted(by_intent.keys())
        # First pass: at least 2 from each intent
        for intent in intents_sorted:
            pool = sorted(by_intent[intent])
            k = min(len(pool), 3)
            picked = rng.sample(pool, k=k)
            sampled_ids.extend(picked)

        # Fill remaining slots up to 30 from largest pools
        all_remaining = [tid for tid in csv_meta.keys() if tid not in sampled_ids]
        rng.shuffle(all_remaining)
        needed = SAMPLE_SIZE - len(sampled_ids)
        if needed > 0:
            sampled_ids.extend(all_remaining[:needed])
        sampled_ids = sampled_ids[:SAMPLE_SIZE]

        SAMPLE_SPEC_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(SAMPLE_SPEC_PATH, "w", encoding="utf-8") as f:
            json.dump(sampled_ids, f, indent=2)

    # Build list of 30 thread items
    items = []
    for tid in sampled_ids:
        meta = csv_meta.get(tid, {})
        draft = drafts_by_id.get(tid, {})
        items.append({
            "thread_id": tid,
            "gold_intent": meta.get("gold_intent", "other"),
            "gold_escalate": meta.get("gold_escalate", "auto_handle"),
            "gold_reply_notes": meta.get("gold_reply_notes", ""),
            "full_thread_text": draft.get("full_thread_text", meta.get("full_thread_text", "")),
            "drafted_reply": draft.get("drafted_reply", ""),
            "predicted_intent": draft.get("predicted_intent", ""),
        })
    return items


def load_human_scores() -> Dict[int, Dict[str, Any]]:
    """Loads existing human scores from CSV."""
    scores = {}
    if HUMAN_RESULTS_CSV.exists():
        with open(HUMAN_RESULTS_CSV, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                tid = int(row["thread_id"])
                scores[tid] = {
                    "thread_id": tid,
                    "tone_empathy": int(row.get("tone_empathy", 3)),
                    "factual_correctness": int(row.get("factual_correctness", 3)),
                    "completeness": int(row.get("completeness", 3)),
                    "safety_privacy": int(row.get("safety_privacy", 3)),
                    "notes": row.get("notes", "").strip(),
                    "overall_score": float(row.get("overall_score", 3.0)),
                }
    return scores


def save_human_score(record: Dict[str, Any], all_records: Dict[int, Dict[str, Any]]):
    """Saves human score record to CSV and JSONL."""
    all_records[record["thread_id"]] = record

    HUMAN_RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    HUMAN_RESULTS_JSONL.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "thread_id",
        "tone_empathy",
        "factual_correctness",
        "completeness",
        "safety_privacy",
        "overall_score",
        "notes",
    ]
    with open(HUMAN_RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for tid in sorted(all_records.keys()):
            r = all_records[tid]
            writer.writerow({
                "thread_id": r["thread_id"],
                "tone_empathy": r["tone_empathy"],
                "factual_correctness": r["factual_correctness"],
                "completeness": r["completeness"],
                "safety_privacy": r["safety_privacy"],
                "overall_score": r["overall_score"],
                "notes": r.get("notes", ""),
            })

    with open(HUMAN_RESULTS_JSONL, "w", encoding="utf-8") as f:
        for tid in sorted(all_records.keys()):
            f.write(json.dumps(all_records[tid], ensure_ascii=False) + "\n")


# ── MAIN STREAMLIT APP ────────────────────────────────────────────────────────

def main():
    st.title("⚖️ Reply-Quality Human Evaluation (Blind Sample N=30)")
    st.caption("Human audit of drafted support replies against ground-truth reply guidance.")

    threads = get_stratified_sample_threads()
    human_scores = load_human_scores()

    if "current_idx" not in st.session_state:
        st.session_state.current_idx = 0

    total_threads = len(threads)
    completed_count = len(human_scores)

    # Sidebar
    with st.sidebar:
        st.header("📊 Progress")
        pct = completed_count / total_threads
        st.progress(pct)
        st.write(f"**{completed_count} / {total_threads}** threads evaluated ({pct*100:.0f}%)")

        st.subheader("Jump to Thread")
        thread_labels = [
            f"#{t['thread_id']} ({'✅' if t['thread_id'] in human_scores else '⏳'}) - {t['gold_intent']}"
            for t in threads
        ]
        selected_label = st.selectbox(
            "Select thread:",
            thread_labels,
            index=st.session_state.current_idx,
        )
        new_idx = thread_labels.index(selected_label)
        if new_idx != st.session_state.current_idx:
            st.session_state.current_idx = new_idx
            st.rerun()

        st.divider()
        st.markdown(
            """
            **Rubric Dimensions (1–5 scale):**
            - **Tone & Empathy**: Warm, polite, brand-safe (1=rude, 5=exemplary).
            - **Factual Correctness**: Accurate Amazon policy/steps (1=wrong, 5=flawless).
            - **Completeness**: Covers all points in `gold_reply_notes` (1=misses core issue, 5=complete).
            - **Safety & Privacy**: Never leaks/asks PII publicly (1=PII breach, 5=secure).
            """
        )

    current_thread = threads[st.session_state.current_idx]
    t_id = current_thread["thread_id"]
    existing = human_scores.get(t_id, {})

    # Top banner info
    col_t1, col_t2, col_t3 = st.columns([2, 2, 2])
    with col_t1:
        st.subheader(f"Thread #{t_id} `[{st.session_state.current_idx + 1}/{total_threads}]`")
    with col_t2:
        st.info(f"🏷️ **Intent**: `{current_thread['gold_intent']}`")
    with col_t3:
        st.info(f"🚦 **Gold Escalate**: `{current_thread['gold_escalate']}`")

    col_left, col_right = st.columns([1, 1], gap="large")

    # Left Column: Conversation & Ground-Truth Notes
    with col_left:
        st.markdown("### 💬 Conversation Thread")
        raw_text = current_thread["full_thread_text"]
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        for line in lines:
            if line.startswith("[Customer]:"):
                msg = line.replace("[Customer]:", "").strip()
                st.markdown(
                    f"""<div style="background-color:#F0F2F6; padding:10px 14px; border-radius:10px; margin-bottom:8px; border-left:4px solid #FF9900;">
                    <b>👤 Customer:</b> {msg}</div>""",
                    unsafe_allow_html=True,
                )
            elif line.startswith("[AmazonHelp]:"):
                msg = line.replace("[AmazonHelp]:", "").strip()
                st.markdown(
                    f"""<div style="background-color:#E8F0FE; padding:10px 14px; border-radius:10px; margin-bottom:8px; border-left:4px solid #1A73E8;">
                    <b>📦 AmazonHelp:</b> {msg}</div>""",
                    unsafe_allow_html=True,
                )
            else:
                st.text(line)

        st.markdown("### 📝 Ground-Truth Reply Guidance (`gold_reply_notes`)")
        st.success(current_thread["gold_reply_notes"] or "*(No specific gold notes provided)*")

    # Right Column: AI Draft & Scoring Sliders
    with col_right:
        st.markdown("### 🤖 Generated AI Agent Reply Draft")
        draft_text = current_thread["drafted_reply"]
        st.markdown(
            f"""<div style="background-color:#E6F4EA; padding:14px; border-radius:10px; margin-bottom:14px; border-left:4px solid #34A853; font-size:1.05rem;">
            {draft_text}</div>""",
            unsafe_allow_html=True,
        )

        st.markdown("### ⚖️ Human Quality Scoring (Blind)")
        with st.form(key=f"score_form_{t_id}"):
            tone_val = st.slider(
                "1. Tone & Empathy (1 = Rude/Dismissive, 5 = Warm & Empathetic)",
                min_value=1,
                max_value=5,
                value=existing.get("tone_empathy", 4),
            )
            fact_val = st.slider(
                "2. Factual Correctness (1 = Factually Wrong / Hallucinated, 5 = Completely Accurate)",
                min_value=1,
                max_value=5,
                value=existing.get("factual_correctness", 4),
            )
            comp_val = st.slider(
                "3. Completeness (1 = Misses Main Issue, 5 = Fully Resolves / Guides)",
                min_value=1,
                max_value=5,
                value=existing.get("completeness", 4),
            )
            safe_val = st.slider(
                "4. Safety & Privacy (1 = Requests/Exposes PII Publicly, 5 = Flawless Privacy)",
                min_value=1,
                max_value=5,
                value=existing.get("safety_privacy", 5),
            )

            notes_val = st.text_area(
                "Human Auditor Notes / Justification (Optional):",
                value=existing.get("notes", ""),
                height=70,
            )

            mean_calc = round((tone_val + fact_val + comp_val + safe_val) / 4.0, 2)
            st.caption(f"Composite Score: **{mean_calc} / 5.00**")

            col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
            with col_btn1:
                prev_btn = st.form_submit_button("⬅️ Previous")
            with col_btn2:
                save_btn = st.form_submit_button("💾 Save Score", type="primary")
            with col_btn3:
                next_btn = st.form_submit_button("Next ➡️")

            if save_btn or next_btn:
                record = {
                    "thread_id": t_id,
                    "tone_empathy": tone_val,
                    "factual_correctness": fact_val,
                    "completeness": comp_val,
                    "safety_privacy": safe_val,
                    "overall_score": mean_calc,
                    "notes": notes_val,
                }
                save_human_score(record, human_scores)
                st.toast(f"Saved Thread #{t_id} (Score: {mean_calc})!", icon="✅")
                if next_btn and st.session_state.current_idx < total_threads - 1:
                    st.session_state.current_idx += 1
                    st.rerun()
                elif save_btn and st.session_state.current_idx < total_threads - 1:
                    st.session_state.current_idx += 1
                    st.rerun()

            if prev_btn and st.session_state.current_idx > 0:
                st.session_state.current_idx -= 1
                st.rerun()


if __name__ == "__main__":
    main()
