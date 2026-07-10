# Evaluation methodology

`scripts/evaluate_tokenizer.py` runs the trained tokenizer against
held-out text in `data/eval/<lang>/*.txt` and writes `eval_report.json`
(schema: `bpe.evaluation.EvaluationReport`). This document explains what
each field means, how it's computed, and how to read it against the
assignment's targets.

## Why held-out text, not the training corpus

Every metric below is only meaningful if measured on text the tokenizer
did **not** train on. Measuring against `data/raw/` would mostly measure
memorization (of course a word gets one token if it was frequent enough
in training to earn its own merge) rather than how the tokenizer
generalizes to new text in the same language. `data/eval/<lang>/` must be
disjoint from `data/raw/<lang>/` — see `data/README.md`.

## Per-language metrics (`LanguageEvalReport`)

### `fertility` — the primary target metric

```
fertility = num_tokens / num_words
```

Both counts are computed **only over word/digit-class pretokens**
(`bpe.pretokenizer.is_word_like`) — standalone punctuation marks are
excluded from both the numerator and denominator, so a sentence full of
commas doesn't artificially deflate the measured fertility. `num_words`
is the count of real-word pretokens; `num_tokens` is how many BPE
subword tokens those specific pretokens were split into.

**Assignment targets:** English fertility ≤ 1.2; fertility as close as
possible across en/hi/te/ta. A fertility of 1.0 means every real word
became exactly one token (maximum compression for that language); higher
values mean words are being split into multiple subwords more often.

`0.0` with `num_words == 0` means there was no eval data for that
language — not that fertility was perfect. Always check `num_documents`
alongside `fertility`.

### `unk_rate`

```
unk_rate = num_unk_tokens / num_tokens   (over ALL tokens, punctuation included)
```

The fraction of tokens that had to fall back to `<unk>` because no vocab
entry existed for them (see `BPE_ALGORITHM.md` §11). This is measured
over every token produced by `encode()`, not just word-class ones,
because "can this content be represented at all" is a broader question
than fertility. **Target: 0.0 for valid, in-domain evaluation text.** A
nonzero rate here on real eval data usually means either the training
corpus for that language was too small/unrepresentative, or the eval
text contains characters/scripts genuinely absent from training (e.g.
emoji, a fifth language, unusual symbols).

### `compression_ratio`

```
compression_ratio = num_tokens / num_chars   (over the whole document, all tokens)
```

Lower is "more compression" (fewer tokens needed per character of text).
Useful as a secondary sanity check alongside fertility, since it's
insensitive to how words are defined by the pretokenizer.

### `roundtrip_mismatches` / `roundtrip_ok`

For each document: `decode(encode(text)) == normalize_for_roundtrip(text)`
(see `bpe.tokenizer.normalize_for_roundtrip` — whitespace runs collapse
to one space, trailing whitespace drops; this is the tokenizer's actual,
documented contract, not a looser or stricter comparison). `roundtrip_ok`
is `True` only if every document in that language matched exactly.

A roundtrip failure almost always means at least one `<unk>`
substitution occurred (an `<unk>` token is inherently lossy — the
original character(s) it stood in for cannot be recovered on decode), so
`roundtrip_mismatches > 0` and `unk_rate > 0` typically point at the same
underlying gap in training-corpus coverage.

### `vocab_utilization`

```
vocab_utilization = (distinct token ids used encoding this language's eval text) / vocab_size
```

Shows whether a language is actually drawing on a meaningful slice of the
shared vocabulary, or barely touching it (a sign the shared vocabulary
may be secretly English-dominated even if fertility numbers look
acceptable — a useful cross-check, since fertility alone can't reveal
this).

## Report-level metric

### `fertility_balance_score`

```
fertility_balance_score = max(fertility) - min(fertility)
```

taken over languages that actually had eval data (`num_words > 0`; a
language with no eval text is excluded rather than counted as a spurious
0.0). **Lower is more balanced.** This is the single number that most
directly answers "did the language-weighting in `bpe.corpus` actually
work" — compare it across training runs with different `--alpha` values
to see the effect directly.

## Reading a report end to end

1. Check `per_language[*].num_documents` first — a language with 0
   eval documents makes every other metric for it meaningless (not a
   perfect score).
2. Check `per_language["en"].fertility <= 1.2` — the CLI prints this
   comparison explicitly.
3. Check `fertility_balance_score` — did weighting during training bring
   the languages close together, or is one language still an outlier?
4. Check `unk_rate` and `roundtrip_ok` per language — nonzero/`False`
   means real content is being lost, not just measured imprecisely.
5. Check `vocab_utilization` — a low value for one language alongside
   otherwise-fine fertility is a sign to inspect that language's training
   corpus size/weight in `tokenizer_config.json`.
