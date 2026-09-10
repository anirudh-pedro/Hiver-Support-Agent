# Ingest Report: AmazonHelp Twitter Customer Support Threads

- **Generated At**: 2026-09-10 09:59:59
- **Source Dataset**: `data/twcs.csv`
- **Total Pipeline Runtime**: 356.76 seconds (5.95 minutes)

## 1. Sequential Funnel Summary

Each stage reflects sequential attrition where the retention percentage is calculated relative to the immediate prior stage survivors.

| Stage | Description | Input Threads | Dropped | Retained | Stage Retention | Cumulative Retention |
|---|---|---|---|---|---|---|
| **1. Raw Thread Reconstruction** | Reconstructed forward from customer roots | 81,329 | 0 | 81,329 | 100.00% | 100.00% |
| **2. English-Only Filter** | Filtered non-English customer initial messages via `langdetect` | 81,329 | 20,437 | 60,892 | 74.87% | 74.87% |
| **3. Praise / Off-Topic Heuristic** | Dropped short greetings / praise with no support intent | 60,892 | 17 | 60,875 | 99.97% | 74.85% |
| **4. Final Support Dataset** | Clean, multi-turn AmazonHelp support threads | 60,875 | 0 | **60,875** | **100.0%** | **74.85%** |

> [!NOTE]
> The sequential funnel reveals that English language filtering accounts for the majority of exclusions (~25.1% of raw threads), reflecting Amazon's multi-regional Twitter presence (e.g. Amazon JP, Amazon DE, Amazon ES). The final golden dataset contains **60,875** clean support conversations.

## 2. Language Filtering Statistics

- **English Threads Retained (`en`)**: 60,892 (74.87%)
- **Non-English / Undetermined Dropped**: 20,437 (25.13%)

### Top Dropped Languages Breakdown

| Rank | Language Code | Detected Language / Category | Dropped Threads | Share of Dropped |
|---|---|---|---|---|
| 1 | `ja` | Japanese | 6,502 | 31.81% |
| 2 | `es` | Spanish | 4,116 | 20.14% |
| 3 | `fr` | French | 3,295 | 16.12% |
| 4 | `de` | German | 2,538 | 12.42% |
| 5 | `pt` | Portuguese | 1,615 | 7.90% |
| 6 | `it` | Italian | 902 | 4.41% |
| 7 | `undetermined` | Undetermined / Too Short / Emojis only | 253 | 1.24% |
| 8 | `nl` | Dutch | 144 | 0.70% |
| 9 | `da` | Other / Dialect | 119 | 0.58% |
| 10 | `so` | Other / Dialect | 89 | 0.44% |
| 11 | `hu` | Other / Dialect | 88 | 0.43% |
| 12 | `ca` | Other / Dialect | 75 | 0.37% |
| 13 | `af` | Other / Dialect | 72 | 0.35% |
| 14 | `tl` | Other / Dialect | 67 | 0.33% |
| 15 | `hi` | Hindi | 59 | 0.29% |

## 3. Praise & Off-Topic Heuristic Filtering

- **Off-Topic / Praise Threads Dropped**: 17 (0.03% of English threads)
- **Design Bias**: Conservative heuristic designed to drop obvious non-support chatter (e.g. short thank-yous, holiday greetings) while strictly preserving borderline cases for downstream intent classification.

### Spot-Check Sample of Filtered Conversations

| Thread ID | Turns | Customer Message (Root) |
|---|---|---|
| `284827` | 2 | Thanks @115821 👍🏻 https://t.co/03DuNexkeH |
| `453306` | 4 | Great. Thanks @115821 https://t.co/GYDSNGks9P |
| `462555` | 4 | Thanks @115821  😒😒😒😒😒😒 https://t.co/yvCXs8nSHK |
| `490506` | 3 | Thanks @116875 @117795 @115821 https://t.co/1dJGnoHpIx |
| `613945` | 8 | @AmazonHelp Thank you @AmazonHelp |
| `1082218` | 2 | Hey, @AmazonHelp , thanks very much! |
| `1110241` | 2 | thanks @115821 https://t.co/HE4mDI7EEt |
| `1741705` | 2 | Thanks @115821 https://t.co/itwlWx3EF0 |
| `2130940` | 4 | Thanks @115821 https://t.co/4yRvOGakXY |
| `2135583` | 2 | @AmazonHelp thanks |
| `2389779` | 2 | Thanks @115850 📦 😊 https://t.co/68cQzkGlAi |
| `2539657` | 2 | @AmazonHelp Thanks! |
| `2632984` | 2 | Thanks @115830 😒 https://t.co/Q4Y7luv8SY |
| `2677842` | 7 | @115821 @AmazonHelp  Thanks so much https://t.co/ZMi4PGKCei |
| `2704984` | 7 | Thank you @115821 !!! Great job! 👏.       👏.       👏.       👏.      😒 https://t.co/3xOYNbdqEk |

## 4. Branch Resolution Statistics

When `response_tweet_id` contains comma-separated child IDs (~8-14% of rows), the pipeline applies the following modeled resolution hierarchy:
1. **`ended_with_amazon`**: Prefer the branch whose eventual last turn is authored by `AmazonHelp` (brand has last word).
2. **`longest_branch`**: If multiple branches qualify (or none do), pick the longest branch.
3. **`tie_break_by_earlier_id`**: Deterministic tie-breaker by earlier child ID in `response_tweet_id`.

| Branch Resolution Metric | Count | Percentage |
|---|---|---|
| **Total Branch Decisions Made** | 156,163 | 100.0% |
| **Threads Involving Branching** | 19,011 | — |
| ↳ Resolved by `ended_with_amazon` | 8,531 | 5.46% |
| ↳ Resolved by `longest_branch` | 86,202 | 55.20% |
| ↳ Resolved by `tie_break_by_earlier_id` | 61,430 | 39.34% |

## 5. Thread Length Distribution

- **Total Clean Support Threads**: 60,875
- **Turn Range**: 2 to 15 turns (capped at 15 turns)
- **Mean Turns**: 3.81
- **Median Turns**: 3.0

| Turns | Frequency | Share | Visual Representation |
|---|---|---|---|
| **2** | 26,196 | 43.03% | `█████████████████` |
| **3** | 7,968 | 13.09% | `█████` |
| **4** | 11,630 | 19.10% | `███████` |
| **5** | 3,766 | 6.19% | `██` |
| **6** | 4,635 | 7.61% | `███` |
| **7** | 1,653 | 2.72% | `█` |
| **8** | 1,835 | 3.01% | `█` |
| **9** | 736 | 1.21% | `` |
| **10** | 861 | 1.41% | `` |
| **11** | 303 | 0.50% | `` |
| **12** | 404 | 0.66% | `` |
| **13** | 170 | 0.28% | `` |
| **14** | 247 | 0.41% | `` |
| **15** | 471 | 0.77% | `` |

## 6. Output Artifacts

- Processed Threads JSONL: `data/processed/threads_amazonhelp.jsonl` (60,875 lines)
- Ingest Summary Report: `data/processed/ingest_report.md`
