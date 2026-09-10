"""
threads.py - Conversation Thread Reconstruction for Twitter Customer Support (TWCS)

Reconstructs multi-turn conversation threads forward-only starting from customer
root tweets (inbound tweets with no in_response_to_tweet_id) and extracts threads
involving AmazonHelp as a responder.

Handles branch resolution when a tweet has multiple child replies (response_tweet_id)
according to domain-specific modeling decisions.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


@dataclass
class BranchStats:
    """Tracks statistics on branch resolution rules during thread reconstruction."""
    total_branch_decisions: int = 0
    threads_with_branching: int = 0
    ended_with_amazon: int = 0
    longest_branch: int = 0
    tie_break_by_earlier_id: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "total_branch_decisions": self.total_branch_decisions,
            "threads_with_branching": self.threads_with_branching,
            "ended_with_amazon": self.ended_with_amazon,
            "longest_branch": self.longest_branch,
            "tie_break_by_earlier_id": self.tie_break_by_earlier_id,
        }


class ThreadBuilder:
    """
    Builds conversation threads forward from customer root tweets in twcs.csv.
    """

    def __init__(self, max_turns: int = 15, target_brand: str = "AmazonHelp") -> None:
        self.max_turns = max_turns
        self.target_brand = target_brand
        self.tweet_data: Dict[int, Dict[str, Any]] = {}
        self.children_map: Dict[int, List[int]] = {}
        self.customer_root_ids: List[int] = []
        self.branch_stats = BranchStats()

    def load_dataset(self, csv_path: str) -> None:
        """
        Loads twcs.csv with strict dtypes and constructs fast in-memory lookup indexes.
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Input CSV not found at: {csv_path}")

        logger.info("Loading %s with strict dtypes...", csv_path)
        t0 = time.time()
        df = pd.read_csv(
            csv_path,
            dtype={
                "tweet_id": "int64",
                "in_response_to_tweet_id": "Int64",
                "response_tweet_id": "string",
                "author_id": "string",
                "inbound": "bool",
                "created_at": "string",
                "text": "string",
            },
        )
        load_time = time.time() - t0
        logger.info("Loaded %d rows in %.2f seconds.", len(df), load_time)

        logger.info("Building in-memory tweet indexes...")
        t0 = time.time()

        tweet_ids = df["tweet_id"].to_numpy()
        author_ids = df["author_id"].to_numpy()
        inbound_flags = df["inbound"].to_numpy()
        created_ats = df["created_at"].to_numpy()
        texts = df["text"].to_numpy()
        response_tweets = df["response_tweet_id"].to_numpy()
        in_response_tos = df["in_response_to_tweet_id"].to_numpy()

        # Build tweet_data dictionary for fast attribute lookup
        self.tweet_data = {}
        self.children_map = {}
        self.customer_root_ids = []

        for i in range(len(tweet_ids)):
            tid = int(tweet_ids[i])
            aid = str(author_ids[i])
            inb = bool(inbound_flags[i])
            cat = str(created_ats[i])
            txt = str(texts[i]) if pd.notna(texts[i]) else ""

            self.tweet_data[tid] = {
                "tweet_id": tid,
                "author_id": aid,
                "inbound": inb,
                "created_at": cat,
                "text": txt,
            }

            resp = response_tweets[i]
            if pd.notna(resp):
                # Parse comma-separated child IDs
                cids = [int(x.strip()) for x in str(resp).split(",") if x.strip().isdigit()]
                if cids:
                    self.children_map[tid] = cids

            # A customer root tweet is inbound with no in_response_to_tweet_id
            if inb and pd.isna(in_response_tos[i]):
                self.customer_root_ids.append(tid)

        index_time = time.time() - t0
        logger.info(
            "Built indexes for %d tweets (%d customer roots) in %.2f seconds.",
            len(self.tweet_data),
            len(self.customer_root_ids),
            index_time,
        )

    def _evaluate_branch_continuation(
        self,
        node_id: int,
        remaining_depth: int,
        memo: Dict[Tuple[int, int], Tuple[List[int], bool, int]],
    ) -> Tuple[List[int], bool, int]:
        """
        Recursively finds the best continuation path from node_id up to remaining_depth.
        Returns:
            (best_path, ends_with_target_brand, path_length)
        """
        state_key = (node_id, remaining_depth)
        if state_key in memo:
            return memo[state_key]

        author = self.tweet_data[node_id]["author_id"]
        ends_with_target = (author == self.target_brand)

        if remaining_depth <= 1:
            res = ([node_id], ends_with_target, 1)
            memo[state_key] = res
            return res

        raw_children = self.children_map.get(node_id, [])
        valid_children = [c for c in raw_children if c in self.tweet_data]

        if not valid_children:
            res = ([node_id], ends_with_target, 1)
            memo[state_key] = res
            return res

        if len(valid_children) == 1:
            sub_path, sub_ends, sub_len = self._evaluate_branch_continuation(
                valid_children[0], remaining_depth - 1, memo
            )
            res = ([node_id] + sub_path, sub_ends, 1 + sub_len)
            memo[state_key] = res
            return res

        # Multiple valid child branches: evaluate all candidate continuations
        candidate_results = []
        for child_id in valid_children:
            sub_path, sub_ends, sub_len = self._evaluate_branch_continuation(
                child_id, remaining_depth - 1, memo
            )
            candidate_results.append({
                "child_id": child_id,
                "full_path": [node_id] + sub_path,
                "ends_with_target": sub_ends,
                "length": 1 + sub_len,
            })

        # MODELING DECISION (not a ground-truth fact):
        # When response_tweet_id contains multiple ids (branching, ~8% of rows),
        # we pick the branch whose eventual last turn is authored by AmazonHelp
        # (preferring conversations where the brand has the last word/resolution).
        # If multiple branches qualify (or none do), we take the longest branch.
        # If still tied, we break ties deterministically by earlier child ID
        # in the original response_tweet_id list.

        target_ending_cands = [c for c in candidate_results if c["ends_with_target"]]
        if target_ending_cands:
            pool = target_ending_cands
        else:
            pool = candidate_results

        max_len = max(c["length"] for c in pool)
        longest_pool = [c for c in pool if c["length"] == max_len]

        best_cand = longest_pool[0]  # first in valid_children order (earlier id)
        res = (best_cand["full_path"], best_cand["ends_with_target"], best_cand["length"])
        memo[state_key] = res
        return res

    def _resolve_child_branch(
        self,
        current_node_id: int,
        valid_children: List[int],
        remaining_depth: int,
        memo: Dict[Tuple[int, int], Tuple[List[int], bool, int]],
    ) -> Tuple[int, str]:
        """
        Chooses the best child among valid_children and logs which rule resolved the choice.
        Returns:
            (chosen_child_id, rule_name)
        """
        candidates = []
        for child_id in valid_children:
            sub_path, sub_ends, sub_len = self._evaluate_branch_continuation(
                child_id, remaining_depth - 1, memo
            )
            candidates.append({
                "child_id": child_id,
                "ends_with_target": sub_ends,
                "length": sub_len,
            })

        amz_cands = [c for c in candidates if c["ends_with_target"]]
        non_amz_cands = [c for c in candidates if not c["ends_with_target"]]

        # Check if resolved strictly by ended_with_amazon
        # (i.e. at least one candidate ended with AmazonHelp while other(s) did not,
        # and exactly one candidate was amz, or among amz cands there's a clear pool)
        if amz_cands and non_amz_cands and len(amz_cands) == 1:
            return amz_cands[0]["child_id"], "ended_with_amazon"

        if amz_cands:
            pool = amz_cands
            used_amazon_filter = bool(non_amz_cands)
        else:
            pool = non_amz_cands
            used_amazon_filter = False

        max_len = max(c["length"] for c in pool)
        longest_cands = [c for c in pool if c["length"] == max_len]

        if len(longest_cands) == 1:
            if used_amazon_filter and len(pool) > 1:
                # Filtered to Amazon-ending, then picked the strictly longest
                return longest_cands[0]["child_id"], "longest_branch"
            elif not used_amazon_filter:
                return longest_cands[0]["child_id"], "longest_branch"
            else:
                return longest_cands[0]["child_id"], "ended_with_amazon"
        else:
            # Tied on length and amazon status -> tie-break by earlier id in response_tweet_id
            return longest_cands[0]["child_id"], "tie_break_by_earlier_id"

    def reconstruct_single_thread(
        self,
        root_id: int,
        memo: Dict[Tuple[int, int], Tuple[List[int], bool, int]],
    ) -> Tuple[Optional[Dict[str, Any]], bool]:
        """
        Walks forward from a single customer root tweet up to max_turns.
        Returns:
            (thread_dict, had_branching)
        """
        path: List[int] = [root_id]
        curr_id = root_id
        had_branching = False

        while len(path) < self.max_turns:
            raw_children = self.children_map.get(curr_id, [])
            valid_children = [c for c in raw_children if c in self.tweet_data]

            if not valid_children:
                break

            if len(valid_children) == 1:
                next_id = valid_children[0]
            else:
                # Branching encountered
                had_branching = True
                self.branch_stats.total_branch_decisions += 1
                rem_depth = self.max_turns - len(path) + 1
                chosen_id, rule_name = self._resolve_child_branch(
                    curr_id, valid_children, rem_depth, memo
                )
                if rule_name == "ended_with_amazon":
                    self.branch_stats.ended_with_amazon += 1
                elif rule_name == "longest_branch":
                    self.branch_stats.longest_branch += 1
                elif rule_name == "tie_break_by_earlier_id":
                    self.branch_stats.tie_break_by_earlier_id += 1
                next_id = chosen_id

            path.append(next_id)
            curr_id = next_id

        # Verify that AmazonHelp appears at least once as a responder (turn index >= 1)
        contains_target_responder = any(
            self.tweet_data[tid]["author_id"] == self.target_brand for tid in path[1:]
        )

        if not contains_target_responder:
            return None, had_branching

        turns = [self.tweet_data[tid] for tid in path]
        thread_obj = {
            "thread_id": root_id,
            "turns": turns,
            "n_turns": len(turns),
            "last_turn_author": turns[-1]["author_id"],
        }
        return thread_obj, had_branching

    def build_all_threads(
        self,
        max_roots: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], BranchStats]:
        """
        Walks forward-only from all customer root tweets and filters to threads
        containing target_brand as a responder.
        """
        logger.info("Reconstructing threads forward-only from customer roots...")
        t0 = time.time()

        roots_to_process = self.customer_root_ids
        if max_roots is not None:
            roots_to_process = roots_to_process[:max_roots]

        reconstructed_threads: List[Dict[str, Any]] = []
        memo: Dict[Tuple[int, int], Tuple[List[int], bool, int]] = {}
        threads_with_branching_count = 0

        for root_id in tqdm(roots_to_process, desc="Reconstructing threads", unit="roots"):
            thread_obj, had_branching = self.reconstruct_single_thread(root_id, memo)
            if had_branching and thread_obj is not None:
                threads_with_branching_count += 1
            if thread_obj is not None:
                reconstructed_threads.append(thread_obj)

        self.branch_stats.threads_with_branching = threads_with_branching_count
        duration = time.time() - t0

        logger.info(
            "Reconstruction complete in %.2fs: Found %d %s threads out of %d customer roots.",
            duration,
            len(reconstructed_threads),
            self.target_brand,
            len(roots_to_process),
        )
        logger.info("Branch resolution stats: %s", self.branch_stats.to_dict())

        return reconstructed_threads, self.branch_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct conversation threads from twcs.csv forward-only."
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to twcs.csv dataset",
    )
    parser.add_argument(
        "--out-jsonl",
        type=str,
        default="data/processed/threads_amazonhelp.jsonl",
        help="Output JSONL path (default: data/processed/threads_amazonhelp.jsonl)",
    )
    parser.add_argument(
        "--out-report",
        type=str,
        default="data/processed/ingest_report.md",
        help="Output ingest report path (default: data/processed/ingest_report.md)",
    )
    parser.add_argument(
        "--max-roots",
        type=int,
        default=None,
        help="Maximum customer roots to process (for debugging/testing)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # If run standalone, execute the full pipeline to produce both the JSONL and markdown report
    # per prompt instructions.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    try:
        from src.ingest import run_pipeline
        run_pipeline(
            csv_path=args.csv,
            out_jsonl=args.out_jsonl,
            out_report=args.out_report,
            max_roots=args.max_roots,
        )
    except ImportError:
        try:
            from ingest import run_pipeline
            run_pipeline(
                csv_path=args.csv,
                out_jsonl=args.out_jsonl,
                out_report=args.out_report,
                max_roots=args.max_roots,
            )
        except ImportError:
            builder = ThreadBuilder(max_turns=15, target_brand="AmazonHelp")
            builder.load_dataset(args.csv)
            threads, stats = builder.build_all_threads(max_roots=args.max_roots)
            os.makedirs(os.path.dirname(os.path.abspath(args.out_jsonl)), exist_ok=True)
            with open(args.out_jsonl, "w", encoding="utf-8") as f:
                for thread in threads:
                    f.write(json.dumps(thread) + "\n")
            logger.info("Wrote %d threads to %s", len(threads), args.out_jsonl)


if __name__ == "__main__":
    main()
