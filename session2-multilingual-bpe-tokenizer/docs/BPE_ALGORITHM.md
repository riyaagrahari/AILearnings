# BPE algorithm and design decisions

This document is the record of *why* the tokenizer is built the way it
is — every nontrivial decision, with the reasoning and trade-offs, not
just the final choice. It doubles as the answer key for "explain your
design" in the assignment.

## 1. Base vocabulary: codepoints, not bytes

**Decision: Unicode codepoint-level base vocabulary** (every unique
character seen during training becomes a base token), **not** UTF-8
byte-level (which is what GPT-2/RoBERTa use).

Why this matters for *this* assignment specifically: Devanagari, Telugu
and Tamil characters are 3 bytes each in UTF-8. Under byte-level BPE,
every Indic character starts life as 3 separate base tokens that must
first be re-merged back into the character itself before any real
subword compression can happen — a "reconstruction tax" English text
never pays (ASCII is 1 byte/char). That tax directly inflates fertility
for exactly the three languages the assignment asks to keep balanced
with English. Codepoint-level vocabulary starts every language on equal
footing: one base token per character, regardless of script.

Trade-off accepted: codepoint-level has no universal fallback the way
byte-level does (256 byte values can represent literally anything).
Section 8 ("no-UNK guarantee") explains how this is handled honestly
instead of papered over.

## 2. Unicode normalization: NFC

**Decision: NFC (Canonical Composition)**, applied once in
`preprocess.clean_text` before anything else touches the text.

Devanagari/Telugu/Tamil text can represent the same visual character as
either one precomposed codepoint or a base+combining-mark sequence.
Without normalizing, the *same word* could hash to two different
strings depending on which form the source used, silently fragmenting
frequency counts and vocabulary entries. NFC (not NFD) was chosen because
it's the composed form and tends to minimize codepoint count.

## 3. Preprocessing: strip markup, not content

`preprocess.py` removes HTML/MediaWiki markup (comments, templates,
tables, `<ref>` tags, wiki-link/external-link syntax, residual HTML
tags) but never touches actual textual content, punctuation, numbers, or
any Unicode character (including ZWJ/ZWNJ). It runs on every input file
regardless of whether it actually contains markup — harmless on plain
prose, necessary on Wikipedia-style dumps.

## 4. Pretokenization: Unicode-category-based, not `\w`/`\d`

**Decision: classify every character by Unicode general category**
(`unicodedata.category`) into `word` (L or M), `digit` (N), `space` (Z or
ASCII whitespace), or `punct` (everything else), rather than using
Python's `\w`/`\d` regex classes directly.

This was verified empirically, not assumed: Python's `\w` **does not
include Unicode combining marks** (categories Mn/Mc) by default. For
Devanagari/Telugu/Tamil, vowel signs (matras) and virama are combining
marks — using `\w` to split words would sever every Indic word at each
vowel sign, e.g. "की" (base "क" + matra "ी") would be treated as a
character boundary, not folded into one word. Folding L and M categories
together into one "word" class fixes this for all three Indic scripts at
once, with no per-script special-casing.

ZWJ (U+200D) and ZWNJ (U+200C) are explicitly kept as part of a word (via
an exception list) since they control conjunct/ligature formation in
Indic scripts and are never markup or whitespace.

Digit runs are split from letter runs, and punctuation is emitted one
character at a time, both to avoid combinatorial vocabulary waste (a
"word+comma" or "word+digit" fused token wastes vocabulary slots that a
separated comma/digit token can serve for every word that precedes it).

## 5. The `▁` word-boundary marker

**Decision: prefix a `▁` (U+2581) marker onto any pretoken immediately
preceded by whitespace** (SentencePiece's convention), rather than
discarding whitespace with no trace.

This was a necessary addition once `tokenizer.py.decode()` needed to be
implemented: a flat list of token ids has no way to know whether two
adjacent tokens belonged to the same original word (e.g. "un"+"believ"+
"able", which must join with no spaces) or different words (e.g.
"the"+"cat", which must join with one) — that distinction cannot be
recovered from token identity alone. Marking word starts at
pretokenization time (the one place that information still exists)
solves this cleanly: decode just replaces `▁` with a space and
concatenates.

Documented limitation, by design: any run of one-or-more whitespace
characters (including newlines) collapses to exactly one space on
reconstruction, and trailing whitespace is dropped. Byte-exact whitespace
preservation was judged out of scope for an educational tokenizer whose
core goals are about subword compression and fertility, not typographic
fidelity — see `tokenizer.normalize_for_roundtrip` for the exact,
testable contract this provides.

## 6. Merge tie-breaking: deterministic lexicographic order

When multiple pairs share the same (maximum) frequency during training,
the pair chosen is the lexicographically smallest — enforced by pushing
`(-count, pair)` tuples onto a heap and letting ordinary Python tuple
comparison break ties on `pair`. This makes training fully deterministic:
the same corpus always produces the same `merges.json`, which matters
both for reproducibility and for debugging.

## 7. Number and punctuation handling

Digits and punctuation are never merged with adjacent letters as base
tokens (see §4) — this keeps the *initial* alphabet clean, but the BPE
merge process is still free to *learn* digit-digit or punctuation-adjacent
merges if they're frequent enough (e.g. "19" as one token, common years).
Nothing is hardcoded to prevent that; it's an emergent property of
whatever's actually frequent in the corpus.

## 8. Preventing English from dominating the shared vocabulary

If the four languages' raw corpora differ substantially in size (very
likely — English text is usually easiest to source in bulk), training
naively on the raw combined corpus would let whichever language
contributes the most *word occurrences* dominate merge selection, since
merges are chosen purely by frequency. The symptom would be Hindi/Telugu/
Tamil ending up with abnormally high fertility (few learned subwords,
mostly single-character tokens) while English enjoys disproportionately
low fertility.

**Decision: exponential-smoothing language weighting**
(`bpe.corpus.compute_language_weights`). Each language's raw word count
is smoothed by an exponent `alpha`:

```
target_share_l  =  count_l**alpha / sum(count_k**alpha for k)
weight_l        =  target_share_l / raw_share_l
```

- `alpha = 1` → weight is exactly 1.0 for every language (no correction;
  the combined corpus keeps its natural raw proportions).
- `alpha = 0` → every language's target share is `1/n` regardless of
  size (full equalization) — risky if a language's corpus is tiny and
  unrepresentative, since it gets amplified to match much larger,
  presumably more representative, corpora.
- **A middle value (0.3–0.5, the CLI default is 0.5)** softens the
  imbalance without fully erasing it: the larger corpus still gets a
  bigger share of the combined training signal, just a less dominant
  one. This is the standard multilingual-model recipe (e.g. mBERT/XLM-R
  use similar exponential smoothing for the same reason).

Each language's word-frequency table is scaled by its weight (rounded,
floored at 1 per nonzero word so no word is erased by rounding) before
all four are summed into the one shared table `train_bpe` actually
learns from.

## 9. Efficient, deterministic training

Naively rescanning every distinct pair's frequency to find the current
maximum, on every merge iteration, is `O(distinct_pairs)` per iteration —
prohibitive once the corpus is nontrivial. `trainer.train_bpe` instead:

- Maintains a running `pair_counts` table and a `pair_index` (which words
  currently contain each pair), updated incrementally (`_apply_word_delta`)
  only for the words actually touched by the most recent merge, not by
  rescanning the whole corpus.
- Uses a max-heap (`_CandidateHeap`) with **lazy invalidation**: instead
  of removing stale entries when a pair's count changes (an `O(n)`
  search), a fresh entry is pushed every time and popped entries are
  simply checked against the pair's current true count, discarding
  anything stale. This keeps each push/pop `O(log n)` without needing an
  indexed/decrease-key heap.

## 10. Training/inference symmetry

`trainer.py` and `tokenizer.py` share exactly two things, deliberately
kept as the *only* shared surface: `bpe.pretokenizer.default_pretokenize`
(so words are split identically at train and inference time — any
divergence here would silently break encoding) and
`trainer.merge_symbols` (the "apply a merge to a symbol list" primitive —
sharing it instead of reimplementing avoids two subtly different
merge-application algorithms drifting apart). Everything else in
`trainer.py` (the heap, the pair index, the training loop) is
training-only and `tokenizer.py` never imports it.

## 11. The "no-UNK" guarantee, stated honestly

The base vocabulary covers every codepoint seen during training. For
in-domain text in the same four languages training was performed on,
this makes `<unk>` usage rare-to-nonexistent in practice — but it is not
an absolute guarantee the way byte-level fallback would be: a genuinely
novel codepoint (a script never seen in training) has no vocabulary
entry and is mapped to `<unk>` rather than crashing or corrupting output.
`bpe.evaluation`'s `unk_rate` metric measures this empirically on real
held-out text, so the assignment's "no UNK for valid evaluation text"
goal is something the pipeline *checks*, not something the design merely
asserts.
