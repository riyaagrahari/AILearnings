# Architecture

This document describes the system as actually implemented. For *why*
each algorithmic decision was made, see
[`BPE_ALGORITHM.md`](BPE_ALGORITHM.md). For how to interpret the numbers
`evaluate_tokenizer.py` produces, see
[`EVALUATION_METHODOLOGY.md`](EVALUATION_METHODOLOGY.md).

## Data flow, end to end

```
data/raw/<lang>/*.txt                          (user-provided, real text)
        │
        ▼
bpe.corpus.load_raw_corpus                     read + clean every file
        │  (calls bpe.preprocess.clean_text per file)
        ▼
bpe.corpus.build_multilingual_word_frequencies  per-language word counts
        │  (calls bpe.pretokenizer.default_pretokenize per document)
        │  (calls bpe.corpus.compute_language_weights, then combines)
        ▼
combined, weighted  word -> frequency  table
        │
        ▼
bpe.trainer.train_bpe                          learn merges greedily
        │
        ▼
vocab.json + merges.json + tokenizer_config.json   (backend/artifacts/)
        │
        ▼
bpe.tokenizer.BPETokenizer.from_files           load for inference
        │
        ├── .encode(text)  -> list[int]
        ├── .decode(ids)   -> str
        └── .encode_pretokens(text) -> per-word structure
        │
        ▼
bpe.evaluation.evaluate_tokenizer               measure against data/eval/
        │
        ▼
eval_report.json  (fertility, unk_rate, compression, roundtrip, vocab_utilization)
```

Two CLI entry points drive this: `scripts/train_tokenizer.py` runs
everything down to `vocab.json`/`merges.json`; `scripts/evaluate_tokenizer.py`
runs everything from loading those artifacts to `eval_report.json`.

## Module responsibilities

### `bpe/preprocess.py`
Cleans one raw text blob: NFC-normalizes, unescapes HTML entities,
strips comments/templates/tables/wiki-links/HTML tags, collapses
whitespace. Preserves all Unicode content, ZWJ/ZWNJ, punctuation and
numbers — it removes *markup*, never *content*. Pure function
(`clean_text(str) -> str`), no filesystem or training knowledge.

### `bpe/pretokenizer.py`
Splits cleaned text into pretokens: letter+combining-mark runs ("word"),
digit runs, and single punctuation characters — using Unicode general
categories (not `\w`/`\d`, which mishandle Indic combining marks). Also
owns the `▁` word-boundary-marker convention (prefixed onto any pretoken
that followed whitespace) that makes `tokenizer.py.decode()` possible at
all. This is the **one shared implementation** both `trainer.py` (at
training time) and `tokenizer.py` (at inference time) call — they must
split text identically, or inference wouldn't match training.

### `bpe/corpus.py`
The only module that touches `data/raw/`. Discovers `*.txt` files per
language folder, cleans each one (`preprocess.clean_text`), reduces each
language to a word-frequency table (`pretokenizer` + a thin wrapper in
`trainer.build_word_frequencies`), computes an exponential-smoothing
weight per language, and combines everything into the one shared
word-frequency table `trainer.train_bpe` consumes. Never downloads or
fabricates data — an empty/missing language folder just contributes zero
words.

### `bpe/trainer.py`
The training algorithm itself: `build_base_vocabulary` (special tokens +
every codepoint seen), then `train_bpe`'s greedy merge loop (an
incrementally-updated pair-frequency table + a lazy-invalidation max-heap
for efficiency — see `BPE_ALGORITHM.md` for why). Also owns
`save_vocab`/`save_merges` (the JSON schemas) and `merge_symbols`, the one
pure "apply a merge to a symbol list" primitive that `tokenizer.py`
reuses for inference rather than reimplementing.

### `bpe/tokenizer.py`
Inference only: `load_vocab`/`load_merges` (the inverse of
`trainer.save_*`) and `BPETokenizer.encode`/`.decode`/`.encode_pretokens`.
Deliberately does not depend on `trainer.py`'s training machinery (heap,
pair index) — only on the one shared `merge_symbols` primitive and the
shared `pretokenizer` module. Encoding applies the same greedy,
priority-ordered merge algorithm training used to produce the merges in
the first place; decoding reverses id→token and un-escapes the `▁`
marker back into spaces (see its module docstring for the exact,
honestly-documented whitespace-normalization contract this provides, and
the honest `<unk>` fallback for genuinely out-of-vocabulary content).

### `bpe/evaluation.py`
Measures, rather than assumes, tokenizer quality on held-out text:
fertility (tokens per real word, via `pretokenizer.is_word_like` to
exclude punctuation from the word count), UNK rate, compression ratio,
roundtrip correctness (against `tokenizer.normalize_for_roundtrip`'s
documented contract), and vocab utilization. Produces one
`LanguageEvalReport` per language plus a `fertility_balance_score`
(max − min fertility across languages).

### `scripts/train_tokenizer.py` / `scripts/evaluate_tokenizer.py`
Thin CLI wrappers: argument parsing, clear stdout/stderr messaging
(including refusing to proceed — with an explanatory error, not a crash —
if there is no real data to work with), and writing the JSON artifacts.
All actual logic lives in `bpe/`; the scripts contain no algorithmic code
of their own.

## Artifacts

| File | Written by | Schema owner |
|---|---|---|
| `vocab.json` | `trainer.save_vocab` | `trainer.py` |
| `merges.json` | `trainer.save_merges` | `trainer.py` |
| `tokenizer_config.json` | `scripts/train_tokenizer.py` | training config + per-language corpus stats |
| `eval_report.json` | `evaluation.save_report` | `evaluation.py` (`EvaluationReport`/`LanguageEvalReport`) |

## Why training and inference are decoupled

`tokenizer.py` never imports `trainer.py`'s heap/pair-index machinery —
only the one pure function (`merge_symbols`) and the shared
`pretokenizer` module. This mirrors how real tokenizer libraries ship:
inference code that only needs `vocab.json`/`merges.json` shouldn't have
to carry training-time data structures along with it. The one place
duplication was deliberately *avoided* (rather than accepted) was exactly
this shared "apply one merge" primitive, since two subtly different
reimplementations of it is precisely the kind of bug that would make
inference silently diverge from what was trained.
