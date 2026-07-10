"""From-scratch Byte Pair Encoding (BPE) trainer.

Pipeline position
------------------
    clean text (bpe.preprocess)  --[this module]-->  vocab.json + merges.json

This module owns *training only* -- building the base vocabulary and
learning merge rules from a corpus. It deliberately does **not** implement
``encode``/``decode`` (that is ``tokenizer.py``'s responsibility, added in a
later step) and it is not a script/CLI -- it is a library of small,
independently-testable functions plus one orchestrating entry point,
:func:`train_bpe`.

Algorithm (see docs/BPE_ALGORITHM.md for the full design rationale)
---------------------------------------------------------------------
1. **Base vocabulary**: every unique Unicode *codepoint* present in the
   corpus (not bytes, not grapheme clusters -- see the design decision on
   base-vocabulary choice), plus a small set of reserved special tokens.
2. **Word frequencies**: the corpus is reduced to a `word string -> count`
   table via a pretokenizer. The default is ``bpe.pretokenizer.default_pretokenize``
   (Unicode-aware, shared with the inference path); a different splitting
   strategy can be injected via the ``pretokenize`` parameter without
   touching this file.
3. **Iterative merging**: repeatedly find the most frequent adjacent
   symbol pair across all words and merge it into a new symbol, until the
   vocabulary reaches ``vocab_size`` or no pair occurs more than once.
4. **Determinism**: ties in pair frequency are broken by picking the
   lexicographically smallest pair -- training is fully reproducible for
   a given corpus and vocab size, independent of dict/set iteration order.
5. **Efficiency**: a merge only touches the (typically small) subset of
   words that actually contain the winning pair, tracked via a
   ``pair -> word_ids`` index, and the next winning pair is found via a
   lazily-invalidated max-heap (O(log n) per candidate) rather than a full
   O(distinct_pairs) rescan every iteration.
"""

from __future__ import annotations

import heapq
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from bpe.pretokenizer import default_pretokenize

__all__ = [
    "DEFAULT_SPECIAL_TOKENS",
    "Merge",
    "TrainingResult",
    "build_word_frequencies",
    "build_base_vocabulary",
    "merge_symbols",
    "train_bpe",
    "train_bpe_from_texts",
    "save_vocab",
    "save_merges",
]

DEFAULT_SPECIAL_TOKENS: dict[str, int] = {
    "<pad>": 0,
    "<bos>": 1,
    "<eos>": 2,
    "<unk>": 3,
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Merge:
    """One learned merge rule, in the order it was learned.

    ``rank`` is that order (0 = learned first = highest priority at
    inference time) and is exactly the ``rank`` field written to
    ``merges.json``.
    """

    rank: int
    pair: tuple[str, str]
    result: str


@dataclass
class TrainingResult:
    """Everything :func:`train_bpe` produces."""

    vocab: dict[str, int]
    special_tokens: dict[str, int]
    merges: list[Merge] = field(default_factory=list)
    num_words: int = 0
    num_unique_words: int = 0

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


def build_word_frequencies(
    texts: Iterable[str],
    pretokenize: Callable[[str], list[str]] = default_pretokenize,
) -> dict[str, int]:
    """Reduce a corpus of clean text lines/documents to `word -> count`.

    Counting *unique* words (rather than keeping every occurrence) is
    what makes training tractable: the merge loop below operates on this
    table, so its cost depends on vocabulary diversity, not raw corpus
    size.
    """
    frequencies: dict[str, int] = {}
    for text in texts:
        for word in pretokenize(text):
            frequencies[word] = frequencies.get(word, 0) + 1
    return frequencies


# ---------------------------------------------------------------------------
# Base vocabulary
# ---------------------------------------------------------------------------


def build_base_vocabulary(
    word_freqs: dict[str, int],
    special_tokens: dict[str, int] = DEFAULT_SPECIAL_TOKENS,
) -> dict[str, int]:
    """Build the initial vocabulary: special tokens + every unique codepoint.

    Codepoints are assigned ids in sorted order so that, for a given
    corpus, the base vocabulary is identical no matter what order words
    were encountered in (part of the determinism guarantee).
    """
    vocab = dict(special_tokens)
    next_id = (max(special_tokens.values()) + 1) if special_tokens else 0

    codepoints = sorted({ch for word in word_freqs for ch in word})
    for ch in codepoints:
        if ch not in vocab:
            vocab[ch] = next_id
            next_id += 1
    return vocab


# ---------------------------------------------------------------------------
# Efficient, deterministic merge loop
# ---------------------------------------------------------------------------


class _CandidateHeap:
    """Max-heap over pair frequency with lazy invalidation.

    Rescanning the entire pair-frequency table for the maximum on every
    merge iteration is O(distinct_pairs) per iteration. Instead, every
    time a pair's count changes we push a fresh ``(-count, pair)`` entry;
    popping always yields the smallest tuple, i.e. the *largest* count,
    with ties broken by the lexicographically smallest pair (the
    determinism rule from the design doc) via ordinary tuple comparison.
    Stale entries (pushed before the pair's count last changed) are
    detected -- by comparing the popped count against the pair's current
    true count -- and simply discarded rather than eagerly removed from
    the heap, which would require O(n) search.
    """

    def __init__(self) -> None:
        self._heap: list[tuple[int, tuple[str, str]]] = []

    def push(self, pair: tuple[str, str], count: int) -> None:
        heapq.heappush(self._heap, (-count, pair))

    def pop_best(self, pair_counts: dict[tuple[str, str], int]) -> tuple[str, str] | None:
        while self._heap:
            neg_count, pair = heapq.heappop(self._heap)
            if pair_counts.get(pair) == -neg_count:
                return pair
        return None


def _pairs_of(symbols: list[str]) -> zip[tuple[str, str]]:
    return zip(symbols, symbols[1:])


def _apply_word_delta(
    word_id: int,
    symbols: list[str],
    freq: int,
    sign: int,
    pair_counts: dict[tuple[str, str], int],
    pair_index: dict[tuple[str, str], set[int]],
    heap: _CandidateHeap,
) -> None:
    """Add (``sign=+1``) or remove (``sign=-1``) one word's contribution to
    the global pair-frequency table, and keep the index/heap in sync.
    """
    touched_pairs: set[tuple[str, str]] = set()
    for pair in _pairs_of(symbols):
        new_count = pair_counts.get(pair, 0) + sign * freq
        if new_count > 0:
            pair_counts[pair] = new_count
            heap.push(pair, new_count)
        else:
            pair_counts.pop(pair, None)
        touched_pairs.add(pair)

    for pair in touched_pairs:
        if sign > 0:
            pair_index.setdefault(pair, set()).add(word_id)
        else:
            word_ids = pair_index.get(pair)
            if word_ids is not None:
                word_ids.discard(word_id)
                if not word_ids:
                    del pair_index[pair]


def merge_symbols(symbols: list[str], pair: tuple[str, str], merged: str) -> list[str]:
    """Replace every non-overlapping occurrence of ``pair`` in ``symbols``
    with the single symbol ``merged`` (standard greedy left-to-right BPE
    merge application).

    Public (not ``_``-prefixed) because ``tokenizer.py``'s inference path
    needs this exact same primitive to apply learned merges to new text --
    sharing it here avoids two subtly-different reimplementations of the
    one piece of logic that training and inference must agree on bit-for-
    bit. Everything else in this module (the heap, the pair index, the
    training loop) is training-only and stays private.
    """
    result: list[str] = []
    i = 0
    n = len(symbols)
    while i < n:
        if i < n - 1 and symbols[i] == pair[0] and symbols[i + 1] == pair[1]:
            result.append(merged)
            i += 2
        else:
            result.append(symbols[i])
            i += 1
    return result


def train_bpe(
    word_freqs: dict[str, int],
    vocab_size: int = 10_000,
    special_tokens: dict[str, int] = DEFAULT_SPECIAL_TOKENS,
) -> TrainingResult:
    """Learn BPE merges from a `word -> frequency` table.

    The base alphabet (special tokens + every unique codepoint in
    ``word_freqs``) is always included in full, even if that alone meets
    or exceeds ``vocab_size`` -- correctness of the base coverage takes
    priority over the nominal size target. Merging then proceeds greedily
    until ``vocab_size`` is reached or no pair occurs more than once
    (merging a pair that occurs only once would not compress anything).
    """
    vocab = build_base_vocabulary(word_freqs, special_tokens)
    next_id = max(vocab.values(), default=-1) + 1

    word_symbols: dict[int, list[str]] = {}
    word_count: dict[int, int] = {}
    pair_counts: dict[tuple[str, str], int] = {}
    pair_index: dict[tuple[str, str], set[int]] = {}
    heap = _CandidateHeap()

    for word_id, (word, freq) in enumerate(word_freqs.items()):
        symbols = list(word)
        word_symbols[word_id] = symbols
        word_count[word_id] = freq
        _apply_word_delta(word_id, symbols, freq, +1, pair_counts, pair_index, heap)

    merges: list[Merge] = []
    while len(vocab) < vocab_size:
        best_pair = heap.pop_best(pair_counts)
        if best_pair is None or pair_counts[best_pair] < 2:
            break  # no pair left worth merging

        merged_symbol = best_pair[0] + best_pair[1]
        affected_word_ids = list(pair_index.get(best_pair, ()))

        for word_id in affected_word_ids:
            symbols = word_symbols[word_id]
            freq = word_count[word_id]
            _apply_word_delta(word_id, symbols, freq, -1, pair_counts, pair_index, heap)
            new_symbols = merge_symbols(symbols, best_pair, merged_symbol)
            word_symbols[word_id] = new_symbols
            _apply_word_delta(word_id, new_symbols, freq, +1, pair_counts, pair_index, heap)

        merges.append(Merge(rank=len(merges), pair=best_pair, result=merged_symbol))
        vocab[merged_symbol] = next_id
        next_id += 1

    return TrainingResult(
        vocab=vocab,
        special_tokens=dict(special_tokens),
        merges=merges,
        num_words=sum(word_freqs.values()),
        num_unique_words=len(word_freqs),
    )


def train_bpe_from_texts(
    texts: Iterable[str],
    vocab_size: int = 10_000,
    special_tokens: dict[str, int] = DEFAULT_SPECIAL_TOKENS,
    pretokenize: Callable[[str], list[str]] = default_pretokenize,
) -> TrainingResult:
    """Convenience wrapper: clean texts -> word frequencies -> :func:`train_bpe`."""
    word_freqs = build_word_frequencies(texts, pretokenize=pretokenize)
    return train_bpe(word_freqs, vocab_size=vocab_size, special_tokens=special_tokens)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_vocab(result: TrainingResult, path: str | Path) -> None:
    """Write ``vocab.json`` -- see docs/ARCHITECTURE.md for the schema."""
    id_to_token = {str(idx): token for token, idx in result.vocab.items()}
    payload = {
        "vocab_size": result.vocab_size,
        "special_tokens": result.special_tokens,
        "tokens": id_to_token,
    }
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_merges(result: TrainingResult, path: str | Path) -> None:
    """Write ``merges.json`` -- an ordered list of learned merge rules."""
    payload = {
        "merges": [
            {"rank": m.rank, "pair": list(m.pair), "result": m.result} for m in result.merges
        ]
    }
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
