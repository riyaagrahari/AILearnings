# Session 4 — Cleaning an India-First Corpus with Session 3's Strategies

This session takes the **data-cleaning strategies proposed in Session 3**
(`session3-report-training`), implements every one of them **from scratch**, and
runs them over a **real 10–100M-scale dataset** — then presents the measured
results as an interactive widget.

Everything on the site is computed by the pipeline; nothing is fabricated.

## The assignment, answered

### 1. How many strategies are listed in Session 3, and what are they?

Session 3's `Data Cleaning` section defines an **eight-stage pipeline** (see
`session3-report-training/src/data/cleaning.ts`). Two are ingest/ship bookends;
the **six in between are the active cleanups**:

| # | Stage | Kind |
|---|---|---|
| 1 | Raw Documents | ingest (bookend) |
| 2 | **Language Identification** | cleanup |
| 3 | **Unicode Normalization** | cleanup |
| 4 | **Deduplication** | cleanup |
| 5 | **Quality Scoring** | cleanup |
| 6 | **Safety & Toxicity Filtering** | cleanup |
| 7 | **Code-Repository Filtering** | cleanup |
| 8 | Training Corpus | assembly (bookend) |

**→ 8 strategies total, 6 of them active cleanups.**

### 2. What dataset was picked?

A genuine, open, **India-first multilingual** corpus:
[`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia)
(CC BY-SA 4.0), in **Hindi, Telugu and Marathi** — chosen because Session 3's
data mix leans on Wikipedia + Indic text, and because Hindi & Marathi share the
Devanagari script (so language-ID has to actually work, not just read a label).

Streaming a bounded budget out of the parquet shards lands the raw working set
at **62.1M characters across 30,208 real articles** — squarely inside the
requested 10–100M range.

### 3. What was cleaned, and why?

Running the eight stages in order took the corpus from **30,208 → 7,580
documents**:

| Stage | Removed | Why |
|---|---|---|
| Language ID | 240 | wrong dominant script (English/list stubs inside Indic dumps) |
| Normalization | 0 | transforms text, doesn't drop — 15.6K smart quotes fixed, **34.9K ZWJ/ZWNJ joiners preserved** |
| Deduplication | 1,335 | 980 exact + 355 near-dups (MinHash+LSH) — e.g. near-identical Indian *district* stub templates |
| Quality scoring | 11,762 | mostly too-short stubs; also low-alpha / high-symbol / repetitive |
| Safety | 2 dropped, 74 redacted | PII (email/phone/Aadhaar/IP/card) redacted across 55 docs; 2 toxic docs dropped |
| Code gate | 0 | CC-BY-SA is SPDX-allow-listed; 0 secrets found (repo gates N/A on prose) |
| Corpus assembly | 9,287 | rebalanced to a target language mix (hi 50 / mr 25 / te 25) |

### 4. Any other strategy or concern cleaned up?

Several strategies **route, redact or preserve** rather than delete:

- **Language review pool** — 876 docs (code-switch, low-confidence, Hindi/Marathi
  mismatch) are *flagged for audit, not discarded* — protecting Indic data.
- **PII redaction** — 74 spans masked in place instead of dropping documents.
- **Script integrity** — ZWJ/ZWNJ joiners preserved; NFC applied without
  collapsing semantically distinct sequences.
- **Code gate** — SPDX license allow-listing + secret scanning, the two gates
  that transfer from code to a prose corpus.

### 5. Final statistics

- **7,580** documents shipped · **43.4M** characters · **~6.88M** words ·
  **~10.9M** estimated tokens
- **69.9%** character retention · **25.1%** document retention
- Frozen, content-hashed, curriculum-ordered training snapshot

## The one bug worth calling out

The first run deleted 27,837 of 27,850 documents at the quality stage. Cause:
Python's stdlib `\w` **excludes Unicode combining marks (Mn/Mc)**, so it
shattered every Devanagari/Telugu word at each vowel-sign/virama — collapsing
"mean word length" below the threshold. This is *exactly* the trap Session 2's
README warned about. Fixed by tokenizing with Unicode properties
(`[\p{L}\p{N}\p{M}]`) via the `regex` module.

## Project structure

```
session4-data-cleaning/
├── pipeline/                     # the real cleaning pipeline (Python)
│   ├── download.py               # fetch hi/te/mr Wikipedia parquet shards
│   ├── strategies.py             # all 8 strategies, from scratch (stdlib + numpy + regex)
│   ├── run_pipeline.py           # orchestrates stages, writes artifacts/stats.json
│   ├── requirements.txt
│   ├── data/                     # downloaded shards (gitignored)
│   └── artifacts/stats.json      # canonical run output
└── src/                          # the widget (React 19 + TS + Tailwind + Framer Motion + Recharts)
    ├── data/stats.json           # widget's data source (synced from the run)
    ├── data/strategyInfo.ts      # Session-3 rationale (why/methods/trade-off) per stage
    └── components/sections/       # Hero, Strategies, Dataset, Pipeline, Concerns, FinalStats
```

## Reproduce it

```bash
# 1. Run the pipeline (downloads ~335 MB of parquet, then processes ~62M chars)
cd pipeline
pip install -r requirements.txt
python3 download.py
python3 run_pipeline.py          # writes artifacts/stats.json + syncs src/data/stats.json

# 2. Run the widget
cd ..
npm install
npm run dev                      # http://localhost:5173
npm run build                    # tsc + vite production build -> dist/
```

> Behind a TLS-intercepting corporate proxy, use
> `NODE_TLS_REJECT_UNAUTHORIZED=0 npm install --strict-ssl=false` and
> `pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org ...`.

## Notes on honesty

- The dedup, quality and language-ID thresholds are engineering choices, clearly
  surfaced in `strategies.py` and in the stage cards.
- `estimated_tokens` is a labelled `chars / 4` estimate, not a real tokenizer run.
- Toxicity hits are ~zero on curated Wikipedia by design — the guard matters far
  more on raw web crawl, which is honestly noted in the widget.
