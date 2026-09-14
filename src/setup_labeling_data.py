"""
setup_labeling_data.py - Initialize clean separation between AI draft labels and human working set.
"""

import csv
import os

TEMPLATE_PATH = "data/golden/golden_set_template.csv"
WORKING_PATH = "data/golden/golden_set_working.csv"
AI_DRAFT_PATH = "data/golden/_ai_draft_labels.csv"

def init_data():
    if not os.path.exists(TEMPLATE_PATH):
        raise FileNotFoundError(f"Missing {TEMPLATE_PATH}")

    with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # 1. Save AI draft labels to hidden file if not exists
    if not os.path.exists(AI_DRAFT_PATH):
        with open(AI_DRAFT_PATH, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["thread_id", "gold_intent", "gold_escalate", "gold_reason", "gold_reply_notes"])
            for r in rows:
                writer.writerow([
                    r["thread_id"],
                    r.get("gold_intent", ""),
                    r.get("gold_escalate", ""),
                    r.get("gold_reason", ""),
                    r.get("gold_reply_notes", ""),
                ])
        print(f"Created {AI_DRAFT_PATH} with {len(rows)} AI draft rows.")

    # 2. Save working set if not exists
    if not os.path.exists(WORKING_PATH):
        with open(WORKING_PATH, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["thread_id", "full_thread_text", "n_turns", "my_intent", "my_escalate", "my_reason", "my_reply_notes"])
            for r in rows:
                writer.writerow([
                    r["thread_id"],
                    r["full_thread_text"],
                    r["n_turns"],
                    "",  # my_intent
                    "",  # my_escalate
                    "",  # my_reason
                    "",  # my_reply_notes
                ])
        print(f"Created {WORKING_PATH} with {len(rows)} blank rows ready for human labeling.")
    else:
        print(f"{WORKING_PATH} already exists; resuming.")

    # 3. Clean template so gold_* columns in template are strictly blank
    with open(TEMPLATE_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["thread_id", "full_thread_text", "n_turns", "gold_intent", "gold_escalate", "gold_reason", "gold_reply_notes"])
        for r in rows:
            writer.writerow([
                r["thread_id"],
                r["full_thread_text"],
                r["n_turns"],
                "",
                "",
                "",
                "",
            ])
    print(f"Reset {TEMPLATE_PATH} with empty gold_* columns.")

if __name__ == "__main__":
    init_data()
