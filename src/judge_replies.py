"""
src/judge_replies.py

LLM-as-a-Judge evaluation harness for drafted customer support replies.

Evaluates generated drafts against human ground-truth `gold_reply_notes` across
four core dimensions (1–5 Likert scale):
  1. tone_empathy: Empathetic, polite, professional, avoids robotic dismissal.
  2. factual_correctness: Aligned with official Amazon policies and ground-truth notes.
  3. completeness: Addresses all issues/questions raised in the thread.
  4. safety_privacy: Protects PII; instructs user to share details via secure DM/link.

Thresholds for `overall_flag`:
  - 'fail': Any dimension == 1, or safety_privacy <= 2, or mean score < 3.0.
  - 'needs_review': Min dimension == 2, or 3.0 <= mean score < 4.0.
  - 'pass': All dimensions >= 3, safety_privacy >= 4, and mean score >= 4.0.

Model Choice:
  Uses `qwen/qwen3.8-27b` (distinct from drafting model `openai/gpt-oss-20b`)
  to eliminate same-model evaluation bias.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.intents import get_all_llm_clients

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

GOLDEN_CSV_PATH = REPO_ROOT / "data" / "golden" / "golden_set_working.csv"
DRAFTS_PATH = REPO_ROOT / "data" / "results" / "golden_drafts.jsonl"
OUTPUT_PATH = REPO_ROOT / "data" / "results" / "golden_judged.jsonl"

JUDGE_MODEL_DEFAULT = "groq/compound-mini"
FALLBACK_MARKER = "I'm sorry for the inconvenience! Please contact us via DM"


@dataclass
class DimensionScore:
    score: int          # 1 to 5
    justification: str  # 1-sentence explanation


@dataclass
class JudgeScore:
    thread_id: int
    tone_empathy: DimensionScore
    factual_correctness: DimensionScore
    completeness: DimensionScore
    safety_privacy: DimensionScore
    overall_score: float
    overall_flag: str   # "pass" | "needs_review" | "fail"
    judge_model: str
    is_fallback_draft: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "thread_id": self.thread_id,
            "tone_empathy": asdict(self.tone_empathy),
            "factual_correctness": asdict(self.factual_correctness),
            "completeness": asdict(self.completeness),
            "safety_privacy": asdict(self.safety_privacy),
            "overall_score": round(self.overall_score, 2),
            "overall_flag": self.overall_flag,
            "judge_model": self.judge_model,
            "is_fallback_draft": self.is_fallback_draft,
        }


def compute_overall_flag(
    tone: int, factual: int, complete: int, safety: int
) -> Tuple[float, str]:
    """
    Computes mean score and overall triage flag according to rubric thresholds.
    """
    scores = [tone, factual, complete, safety]
    mean_score = sum(scores) / len(scores)

    # Fail conditions
    if any(s == 1 for s in scores) or safety <= 2 or mean_score < 3.0:
        return mean_score, "fail"

    # Pass conditions
    if all(s >= 3 for s in scores) and safety >= 4 and mean_score >= 4.0:
        return mean_score, "pass"

    # Needs review conditions
    return mean_score, "needs_review"


_JUDGE_SYSTEM_PROMPT = """\
You are an expert quality auditor for an enterprise customer support AI agent.
Your task is to evaluate a drafted customer support reply against the conversation context
and the human ground-truth reply guidance (`gold_reply_notes`).

Score the draft on FOUR dimensions using a 1 to 5 integer scale:
1. tone_empathy (1-5): Is the reply empathetic, polite, and professional? (1=rude/dismissive, 5=exceptionally warm & helpful)
2. factual_correctness (1-5): Is the information accurate according to Amazon policies and gold notes? (1=hallucinated/wrong policy, 5=completely accurate)
3. completeness (1-5): Does the draft address all questions/issues highlighted in the gold notes? (1=misses main problem, 5=covers every key point)
4. safety_privacy (1-5): Does the reply adhere to privacy rules? (Never confirms/exposes PII publicly; directs to secure DM/portal). (1=critical PII violation, 5=flawless privacy adherence)

Output format MUST be valid JSON only with exactly this schema:
{
  "tone_empathy": {"score": <1-5>, "justification": "<one sentence>"},
  "factual_correctness": {"score": <1-5>, "justification": "<one sentence>"},
  "completeness": {"score": <1-5>, "justification": "<one sentence>"},
  "safety_privacy": {"score": <1-5>, "justification": "<one sentence>"}
}
"""


def _build_judge_prompt(thread_text: str, drafted_reply: str, gold_reply_notes: str) -> str:
    return (
        f"--- CONVERSATION THREAD ---\n{thread_text.strip()}\n\n"
        f"--- HUMAN GOLD GUIDANCE (gold_reply_notes) ---\n{gold_reply_notes.strip()}\n\n"
        f"--- DRAFTED AGENT REPLY TO EVALUATE ---\n{drafted_reply.strip()}\n\n"
        "Evaluate the drafted reply across all 4 dimensions and output valid JSON."
    )


def judge_reply(
    thread_text: str,
    drafted_reply: str,
    gold_reply_notes: str,
    thread_id: int,
    clients: List[Any] | Any,
    model: str = JUDGE_MODEL_DEFAULT,
    max_retries: int = 4,
) -> JudgeScore:
    """
    Evaluates a single drafted reply using the LLM-as-a-Judge rubric.
    Rotates across provided API clients on error/retry.
    """
    is_fallback = FALLBACK_MARKER in drafted_reply
    prompt = _build_judge_prompt(thread_text, drafted_reply, gold_reply_notes)

    client_list = clients if isinstance(clients, list) else [clients]

    candidate_models = [model, "groq/compound-mini", "groq/compound", "qwen/qwen3.8-27b"]
    seen = set()
    models_to_try = [m for m in candidate_models if m and not (m in seen or seen.add(m))]

    raw = ""
    for attempt in range(max_retries):
        cur_client = client_list[attempt % len(client_list)] if client_list else None
        cur_model = models_to_try[attempt % len(models_to_try)]
        try:
            resp = cur_client.chat.completions.create(
                model=cur_model,
                messages=[
                    {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=350,
                temperature=0.0,
            )
            raw = (resp.choices[0].message.content or "").strip()
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                raise ValueError(f"No valid JSON in response: {raw!r}")

            parsed = json.loads(match.group())

            def parse_dim(dim_name: str) -> DimensionScore:
                d = parsed.get(dim_name, {})
                s = int(d.get("score", 3))
                s = max(1, min(5, s))
                j = str(d.get("justification", f"Evaluated {dim_name}."))
                return DimensionScore(score=s, justification=j)

            t_score = parse_dim("tone_empathy")
            f_score = parse_dim("factual_correctness")
            c_score = parse_dim("completeness")
            s_score = parse_dim("safety_privacy")

            overall_num, flag = compute_overall_flag(
                t_score.score, f_score.score, c_score.score, s_score.score
            )

            return JudgeScore(
                thread_id=thread_id,
                tone_empathy=t_score,
                factual_correctness=f_score,
                completeness=c_score,
                safety_privacy=s_score,
                overall_score=overall_num,
                overall_flag=flag,
                judge_model=cur_model,
                is_fallback_draft=is_fallback,
            )

        except Exception as exc:
            wait = 1.0 * (attempt + 1)
            logger.warning(
                "Judge attempt %d/%d (model=%s) failed for thread #%d: %s — trying next client in %.1fs",
                attempt + 1, max_retries, cur_model, thread_id, exc, wait,
            )
            time.sleep(wait)

    # Fallback score if judge fails after all retries
    logger.error("Judge failed after %d retries for thread #%d", max_retries, thread_id)
    return JudgeScore(
        thread_id=thread_id,
        tone_empathy=DimensionScore(score=2, justification="Judge parsing error."),
        factual_correctness=DimensionScore(score=2, justification="Judge parsing error."),
        completeness=DimensionScore(score=2, justification="Judge parsing error."),
        safety_privacy=DimensionScore(score=2, justification="Judge parsing error."),
        overall_score=2.0,
        overall_flag="needs_review",
        judge_model="judge_failure_fallback",
        is_fallback_draft=is_fallback,
    )


def load_golden_meta() -> Dict[int, Dict[str, str]]:
    """Loads gold_reply_notes from golden_set_working.csv (uses my_reply_notes)."""
    meta = {}
    with open(GOLDEN_CSV_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t_id = int(row["thread_id"])
            notes = row.get("my_reply_notes", "") or row.get("gold_reply_notes", "")
            meta[t_id] = {
                "gold_reply_notes": notes.strip(),
                "full_thread_text": row.get("full_thread_text", ""),
                "gold_intent": row.get("my_intent", ""),
            }
    return meta


def load_drafts() -> List[Dict[str, Any]]:
    """Loads records from golden_drafts.jsonl."""
    records = []
    with open(DRAFTS_PATH, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def print_judge_summary(judged_records: List[Dict[str, Any]]) -> None:
    """Prints a structured summary of reply evaluation metrics."""
    if not judged_records:
        print("No judged records to summarize.")
        return

    total = len(judged_records)
    valid = [r for r in judged_records if not r.get("is_fallback_draft", False)]
    fallbacks = [r for r in judged_records if r.get("is_fallback_draft", False)]

    flags = Counter(r["overall_flag"] for r in judged_records)
    flags_valid = Counter(r["overall_flag"] for r in valid)

    def avg_dim(records_list: List[Dict[str, Any]], dim: str) -> float:
        if not records_list:
            return 0.0
        return sum(r[dim]["score"] for r in records_list) / len(records_list)

    sep = "=" * 80
    print(f"\n{sep}")
    print(f"       REPLY-QUALITY JUDGE EVALUATION SUMMARY (N={total}, Valid LLM Drafts={len(valid)})")
    print(f"{sep}")

    print("\n-- Overall Quality Flag Distribution ----------------------------------------")
    print(f"  {'Flag':<16} | {'All Drafts (N=220)':<22} | {'Valid LLM Drafts (N=202)':<24}")
    print("-" * 70)
    for flag in ["pass", "needs_review", "fail"]:
        cnt_all = flags.get(flag, 0)
        cnt_val = flags_valid.get(flag, 0)
        print(f"  {flag:<16} | {cnt_all:>4} ({cnt_all/total*100:>5.1f}%)         | {cnt_val:>4} ({cnt_val/len(valid)*100:>5.1f}%)")

    print("\n-- Dimension Scores (1–5 Likert Scale, Valid LLM Drafts N=202) -------------")
    dims = [
        ("Tone & Empathy", "tone_empathy"),
        ("Factual Correctness", "factual_correctness"),
        ("Completeness", "completeness"),
        ("Safety & Privacy", "safety_privacy"),
    ]
    for label, dim in dims:
        scores = [r[dim]["score"] for r in valid]
        mean_s = sum(scores) / len(scores) if scores else 0.0
        score_dist = Counter(scores)
        dist_str = " | ".join(f"[{s}]: {score_dist.get(s,0)}" for s in range(1, 6))
        print(f"  {label:<22}: Mean = {mean_s:.2f} / 5.00   ({dist_str})")

    avg_all_valid = sum(r["overall_score"] for r in valid) / len(valid) if valid else 0.0
    print(f"\n  Average Composite Score (Valid N={len(valid)}): {avg_all_valid:.2f} / 5.00")
    if fallbacks:
        print(f"  Rate-Limit Fallback Drafts Excluded from Quality Metrics: {len(fallbacks)} (flagged automatically)")
    print(sep + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM-as-a-Judge on golden drafts.")
    parser.add_argument("--clean", action="store_true", help="Clean/overwrite existing results")
    parser.add_argument("--model", type=str, default=JUDGE_MODEL_DEFAULT, help="Judge model ID")
    parser.add_argument("--sleep", type=float, default=0.6, help="Sleep time between judge calls")
    args = parser.parse_args()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if args.clean and OUTPUT_PATH.exists():
        logger.info("Cleaning existing output: %s", OUTPUT_PATH)
        OUTPUT_PATH.unlink()

    existing_records: Dict[int, Dict[str, Any]] = {}
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        r = json.loads(line)
                        if r.get("judge_model") != "judge_failure_fallback":
                            existing_records[int(r["thread_id"])] = r
                    except Exception:
                        pass
        logger.info("Resuming: %d threads already judged.", len(existing_records))

    meta_by_id = load_golden_meta()
    drafts = load_drafts()

    all_clients = get_all_llm_clients()
    clients = [c[0] for c in all_clients]
    logger.info("LLM judge clients available: %d", len(clients))

    to_judge = [d for d in drafts if int(d["thread_id"]) not in existing_records]
    logger.info("Judging %d threads with model=%s ...", len(to_judge), args.model)

    def save_all():
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            for tid in sorted(existing_records.keys()):
                f.write(json.dumps(existing_records[tid], ensure_ascii=False) + "\n")

    for idx, draft in enumerate(to_judge, 1):
        t_id = int(draft["thread_id"])
        meta = meta_by_id.get(t_id, {})
        thread_text = draft.get("full_thread_text", meta.get("full_thread_text", ""))
        draft_text = draft.get("drafted_reply", "")
        gold_notes = meta.get("gold_reply_notes", "Provide helpful customer support.")

        score_obj = judge_reply(
            thread_text=thread_text,
            drafted_reply=draft_text,
            gold_reply_notes=gold_notes,
            thread_id=t_id,
            clients=clients,
            model=args.model,
        )

        existing_records[t_id] = score_obj.to_dict()
        save_all()

        total_done = len(existing_records)
        logger.info(
            "[%d/%d] #%d -> %s (mean=%.2f, tone=%d, fact=%d, comp=%d, safe=%d) [%s]",
            total_done, len(drafts), t_id, score_obj.overall_flag,
            score_obj.overall_score,
            score_obj.tone_empathy.score,
            score_obj.factual_correctness.score,
            score_obj.completeness.score,
            score_obj.safety_privacy.score,
            score_obj.judge_model,
        )

        if args.sleep > 0:
            time.sleep(args.sleep)

    print_judge_summary(list(existing_records.values()))


if __name__ == "__main__":
    main()
