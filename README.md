# AmazonHelp Twitter Customer Support Ingestion & Thread Reconstruction

Pipeline for ingesting and reconstructing multi-turn customer support conversation threads for **AmazonHelp** from Kaggle's *Customer Support on Twitter* (`twcs.csv`, ~2.8M rows).

## Features
- **Forward-Only Reconstruction**: Identifies all inbound customer root tweets (`inbound == True`, `in_response_to_tweet_id.isna()`) and walks reply chains forward.
- **Modeled Branch Resolution**: Resolves multi-reply branching (`response_tweet_id` containing comma-separated IDs) by prioritizing branches that end with `AmazonHelp`, then longest length, with deterministic tie-breaking.
- **Parallel English Language Filter**: Multiprocessed `langdetect` on customer root messages across CPU cores.
- **Praise / Off-Topic Filter**: Lightweight heuristic dropping non-support greetings and thank-yous while preserving borderline queries.
- **Comprehensive Reporting**: Generates a sequential funnel markdown report (`ingest_report.md`) with attrition rates, language breakdown, branch stats, and turn distributions.

## Setup
```bash
pip install -r requirements.txt
```

## Usage

### Run Full Ingestion Pipeline
```bash
python src/ingest.py --csv data/twcs.csv
```
Options:
- `--csv`: Path to `twcs.csv` (required)
- `--out-jsonl`: Output path for JSONL threads (default: `data/processed/threads_amazonhelp.jsonl`)
- `--out-report`: Output path for ingest report (default: `data/processed/ingest_report.md`)
- `--max-roots`: Optional limit on customer roots to process (for debugging)
- `--workers`: Number of worker processes for parallel language detection

### Run Thread Reconstruction Standalone
```bash
python src/threads.py --csv data/twcs.csv
```

## Output Format
Each thread is formatted as a single JSON object per line in `data/processed/threads_amazonhelp.jsonl`:
```json
{
  "thread_id": 115712,
  "turns": [
    {
      "tweet_id": 115712,
      "author_id": "115712",
      "inbound": true,
      "created_at": "Tue Oct 31 22:10:47 +0000 2017",
      "text": "@AmazonHelp why is my order delayed?"
    },
    {
      "tweet_id": 115713,
      "author_id": "AmazonHelp",
      "inbound": false,
      "created_at": "Tue Oct 31 22:15:10 +0000 2017",
      "text": "Hello, we would be happy to check on this for you. Please DM us your order ID."
    }
  ],
  "n_turns": 2,
  "last_turn_author": "AmazonHelp"
}
```

See [data/processed/ingest_report.md](data/processed/ingest_report.md) for the complete attrition funnel and statistics.
