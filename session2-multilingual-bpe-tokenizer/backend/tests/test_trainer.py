"""Unit tests for bpe.trainer.

Covers: base-vocabulary construction, the merge loop's correctness and
determinism, the vocab-size stopping condition, and the vocab.json/merges.json
save format.

Pretokenizer-specific behavior (Unicode word/digit/punctuation splitting,
Indic combining marks, ZWJ/ZWNJ handling) is tested in test_pretokenizer.py,
since bpe.pretokenizer is now its own module shared by training and
inference -- trainer.py only consumes it as a dependency here.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bpe.trainer import (  # noqa: E402
    DEFAULT_SPECIAL_TOKENS,
    Merge,
    build_base_vocabulary,
    build_word_frequencies,
    save_merges,
    save_vocab,
    train_bpe,
    train_bpe_from_texts,
)


# ---------------------------------------------------------------------------
# build_word_frequencies
# ---------------------------------------------------------------------------


def test_build_word_frequencies_counts_across_documents():
    # "low" appears once at the very start of a text (no boundary marker,
    # since nothing precedes it) and, separately, after whitespace (with
    # a marker) -- these are intentionally distinct pretokens (mirrors
    # real BPE tokenizers, where " low" and "low" are different tokens).
    texts = ["low low low", "lower lower", "low"]
    freqs = build_word_frequencies(texts)
    assert freqs == {"low": 2, "\u2581low": 2, "lower": 1, "\u2581lower": 1}


# ---------------------------------------------------------------------------
# build_base_vocabulary
# ---------------------------------------------------------------------------


def test_base_vocabulary_includes_special_tokens_and_all_codepoints():
    word_freqs = {"ab": 3, "cd": 1}
    vocab = build_base_vocabulary(word_freqs)
    for token, idx in DEFAULT_SPECIAL_TOKENS.items():
        assert vocab[token] == idx
    for ch in "abcd":
        assert ch in vocab


def test_base_vocabulary_ids_are_deterministic_regardless_of_input_order():
    v1 = build_base_vocabulary({"ab": 1, "cd": 1})
    v2 = build_base_vocabulary({"cd": 1, "ab": 1})
    assert v1 == v2


def test_base_vocabulary_supports_custom_special_tokens():
    vocab = build_base_vocabulary({"a": 1}, special_tokens={"<foo>": 0})
    assert vocab["<foo>"] == 0
    assert vocab["a"] == 1


# ---------------------------------------------------------------------------
# train_bpe -- correctness on small, hand-verifiable examples
# ---------------------------------------------------------------------------


def test_train_bpe_merges_the_most_frequent_pair_first():
    # ('a','b') occurs 5 times, ('a','c') occurs 3 times -> ('a','b') wins.
    word_freqs = {"ab": 5, "ac": 3}
    result = train_bpe(word_freqs, vocab_size=100)
    assert result.merges[0].pair == ("a", "b")
    assert result.merges[0].result == "ab"
    assert result.merges[0].rank == 0


def test_train_bpe_produces_expected_merge_sequence_classic_example():
    # Loosely inspired by the canonical Sennrich et al. BPE walk-through:
    # a corpus dominated by "low"/"lower" vs. rarer "newest"/"widest"
    # should merge within "low"/"lower" first since their combined
    # frequency of shared pairs (7) is highest. After ('l','o') is merged,
    # the standalone 'o' symbol no longer exists, so the next winning
    # pair is ('lo','w') -- not ('o','w').
    word_freqs = {"low": 5, "lower": 2, "newest": 1, "widest": 1}
    result = train_bpe(word_freqs, vocab_size=100)
    first_two_pairs = [m.pair for m in result.merges[:2]]
    assert first_two_pairs == [("l", "o"), ("lo", "w")]


def test_train_bpe_stops_when_vocab_size_reached():
    word_freqs = {"aaaa": 10, "bbbb": 10, "cccc": 10}
    base_size = len(build_base_vocabulary(word_freqs))
    target = base_size + 2
    result = train_bpe(word_freqs, vocab_size=target)
    assert result.vocab_size == target
    assert len(result.merges) == 2


def test_train_bpe_stops_when_no_pair_occurs_more_than_once():
    # Every pair in this tiny corpus occurs exactly once -- nothing should
    # be merged even though vocab_size leaves plenty of headroom.
    word_freqs = {"ab": 1, "cd": 1}
    result = train_bpe(word_freqs, vocab_size=1000)
    assert result.merges == []


def test_train_bpe_includes_full_base_alphabet_even_if_it_exceeds_target():
    word_freqs = {"abcdef": 5}
    result = train_bpe(word_freqs, vocab_size=1)  # smaller than base alphabet
    for ch in "abcdef":
        assert ch in result.vocab
    for token in DEFAULT_SPECIAL_TOKENS:
        assert token in result.vocab


def test_train_bpe_deterministic_tie_break_picks_lexicographically_smaller_pair():
    # ('a','b') and ('c','d') both occur exactly 4 times -> ('a','b') wins
    # because it is lexicographically smaller.
    word_freqs = {"ab": 4, "cd": 4}
    result = train_bpe(word_freqs, vocab_size=100)
    assert result.merges[0].pair == ("a", "b")


def test_train_bpe_is_deterministic_regardless_of_word_insertion_order():
    freqs_a = {"low": 5, "lower": 2, "newest": 6, "widest": 3}
    freqs_b = {"widest": 3, "newest": 6, "lower": 2, "low": 5}
    result_a = train_bpe(freqs_a, vocab_size=50)
    result_b = train_bpe(freqs_b, vocab_size=50)
    assert [m.pair for m in result_a.merges] == [m.pair for m in result_b.merges]
    assert result_a.vocab == result_b.vocab


def test_train_bpe_records_merges_with_increasing_rank():
    word_freqs = {"aaaa": 10, "bbbb": 10, "cccc": 10}
    result = train_bpe(word_freqs, vocab_size=100)
    ranks = [m.rank for m in result.merges]
    assert ranks == list(range(len(result.merges)))


def test_train_bpe_final_vocab_contains_every_merge_result():
    word_freqs = {"aaaa": 10, "bbbb": 10}
    result = train_bpe(word_freqs, vocab_size=100)
    for merge in result.merges:
        assert merge.result in result.vocab


def test_train_bpe_word_and_unique_word_counts():
    word_freqs = {"low": 5, "lower": 2}
    result = train_bpe(word_freqs, vocab_size=100)
    assert result.num_words == 7
    assert result.num_unique_words == 2


# ---------------------------------------------------------------------------
# train_bpe_from_texts -- end-to-end convenience wrapper
# ---------------------------------------------------------------------------


def test_train_bpe_from_texts_end_to_end_english():
    texts = ["the cat sat on the mat", "the dog sat on the log"]
    result = train_bpe_from_texts(texts, vocab_size=60)
    assert result.vocab_size <= 60
    assert len(result.merges) > 0


def test_train_bpe_from_texts_multilingual_toy_corpus():
    texts = [
        "the price is high",  # English
        "मूल्य अधिक है",  # Hindi
        "ధర ఎక్కువ",  # Telugu
        "விலை அதிகம்",  # Tamil
    ]
    result = train_bpe_from_texts(texts, vocab_size=200)
    # Every character across all four languages must survive into the
    # base vocabulary -- this is the "shared vocab, no language ignored"
    # requirement showing up at the training-data-structures level.
    for text in texts:
        for ch in text:
            if ch != " ":
                assert ch in result.vocab or any(
                    ch in merge.result for merge in result.merges
                )


# ---------------------------------------------------------------------------
# save_vocab / save_merges -- JSON schema
# ---------------------------------------------------------------------------


def test_save_vocab_writes_expected_schema(tmp_path):
    result = train_bpe({"ab": 5, "ac": 3}, vocab_size=20)
    out = tmp_path / "vocab.json"
    save_vocab(result, out)
    data = json.loads(out.read_text(encoding="utf-8"))

    assert data["vocab_size"] == result.vocab_size
    assert data["special_tokens"] == DEFAULT_SPECIAL_TOKENS
    assert isinstance(data["tokens"], dict)
    # ids are stored as string keys (JSON object keys must be strings)
    for token, idx in result.vocab.items():
        assert data["tokens"][str(idx)] == token


def test_save_merges_writes_expected_schema_and_order(tmp_path):
    result = train_bpe({"aaaa": 10, "bbbb": 10}, vocab_size=100)
    out = tmp_path / "merges.json"
    save_merges(result, out)
    data = json.loads(out.read_text(encoding="utf-8"))

    assert list(data.keys()) == ["merges"]
    ranks = [entry["rank"] for entry in data["merges"]]
    assert ranks == sorted(ranks)
    for entry, merge in zip(data["merges"], result.merges):
        assert entry["pair"] == list(merge.pair)
        assert entry["result"] == merge.result


def test_save_vocab_roundtrips_non_ascii_tokens(tmp_path):
    result = train_bpe_from_texts(["ధర ఎక్కువ"], vocab_size=50)
    out = tmp_path / "vocab.json"
    save_vocab(result, out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "ధ" in data["tokens"].values()
