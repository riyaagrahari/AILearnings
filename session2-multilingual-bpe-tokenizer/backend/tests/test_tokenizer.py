"""Unit tests for bpe.tokenizer.

Covers: encode/decode round-tripping, the documented whitespace-
normalization contract, the honest <unk> fallback for out-of-vocabulary
content, the vocab.json/merges.json load functions, and multilingual
behavior across English, Hindi, Telugu and Tamil.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bpe.trainer import Merge, save_merges, save_vocab, train_bpe_from_texts  # noqa: E402
from bpe.tokenizer import (  # noqa: E402
    BPETokenizer,
    load_merges,
    load_vocab,
    normalize_for_roundtrip,
)


# ---------------------------------------------------------------------------
# normalize_for_roundtrip
# ---------------------------------------------------------------------------


def test_normalize_collapses_internal_whitespace_runs():
    assert normalize_for_roundtrip("hello   world") == "hello world"


def test_normalize_drops_trailing_whitespace():
    assert normalize_for_roundtrip("hello world   ") == "hello world"


def test_normalize_collapses_leading_whitespace_to_single_space():
    assert normalize_for_roundtrip("  hello") == " hello"


def test_normalize_is_idempotent():
    text = "  hello   world  "
    once = normalize_for_roundtrip(text)
    assert normalize_for_roundtrip(once) == once


# ---------------------------------------------------------------------------
# load_vocab / load_merges (inverse of trainer.save_vocab/save_merges)
# ---------------------------------------------------------------------------


def test_load_vocab_roundtrips_save_vocab(tmp_path):
    result = train_bpe_from_texts(["low lower lowest"], vocab_size=50)
    path = tmp_path / "vocab.json"
    save_vocab(result, path)

    vocab, special_tokens = load_vocab(path)
    assert vocab == result.vocab
    assert special_tokens == result.special_tokens


def test_load_merges_roundtrips_save_merges(tmp_path):
    result = train_bpe_from_texts(["low lower lowest"], vocab_size=50)
    path = tmp_path / "merges.json"
    save_merges(result, path)

    merges = load_merges(path)
    assert merges == result.merges


def test_load_merges_returns_merge_dataclass_instances(tmp_path):
    result = train_bpe_from_texts(["aaaa bbbb"], vocab_size=20)
    path = tmp_path / "merges.json"
    save_merges(result, path)
    merges = load_merges(path)
    assert all(isinstance(m, Merge) for m in merges)


# ---------------------------------------------------------------------------
# Helper to build a tokenizer directly from an in-memory TrainingResult
# ---------------------------------------------------------------------------


def _tokenizer_from_texts(texts, vocab_size=200):
    result = train_bpe_from_texts(texts, vocab_size=vocab_size)
    return BPETokenizer(
        vocab=result.vocab, special_tokens=result.special_tokens, merges=result.merges
    )


# ---------------------------------------------------------------------------
# encode / decode round-tripping
# ---------------------------------------------------------------------------


def test_encode_decode_roundtrip_on_training_text():
    texts = ["the cat sat on the mat", "the dog sat on the log"]
    tok = _tokenizer_from_texts(texts)
    for text in texts:
        assert tok.decode(tok.encode(text)) == normalize_for_roundtrip(text)


def test_encode_decode_roundtrip_on_unseen_but_in_vocabulary_text():
    texts = ["the cat sat on the mat", "the dog sat on the log"]
    tok = _tokenizer_from_texts(texts)
    unseen = "the cat sat on the log"  # new combination, same characters
    assert tok.decode(tok.encode(unseen)) == normalize_for_roundtrip(unseen)


def test_encode_is_deterministic():
    tok = _tokenizer_from_texts(["banana bandana"])
    assert tok.encode("banana") == tok.encode("banana")


def test_encode_matches_training_merge_priority():
    # Base alphabet for "aaaa"/"▁aaaa" is {'a', '▁'} + 4 special tokens = 6
    # ids; capping vocab_size at 7 allows exactly one merge (('a','a')),
    # so a fresh "aaaa" should encode as two "aa" tokens, not collapse
    # further (that would require a second merge this budget disallows).
    tok = _tokenizer_from_texts(["aaaa aaaa aaaa"], vocab_size=7)
    tokens = tok.encode_as_tokens("aaaa")
    assert tokens == ["aa", "aa"]


def test_decode_drops_structural_special_tokens():
    tok = _tokenizer_from_texts(["hello world"])
    pad_id = tok.special_tokens["<pad>"]
    bos_id = tok.special_tokens["<bos>"]
    eos_id = tok.special_tokens["<eos>"]
    real_ids = tok.encode("hello")
    decoded = tok.decode([bos_id, *real_ids, eos_id, pad_id])
    assert decoded == "hello"


# ---------------------------------------------------------------------------
# Honest <unk> fallback (no silent data loss, no crash)
# ---------------------------------------------------------------------------


def test_encode_falls_back_to_unk_for_unseen_characters():
    tok = _tokenizer_from_texts(["cat dog"], vocab_size=30)
    ids = tok.encode("zzz")  # 'z' never appeared in training
    assert tok.special_tokens["<unk>"] in ids


def test_decode_renders_unk_literally_not_silently():
    tok = _tokenizer_from_texts(["cat dog"], vocab_size=30)
    unk_id = tok.special_tokens["<unk>"]
    assert tok.decode([unk_id]) == "<unk>"


def test_encode_raises_if_unk_not_configured_and_token_missing():
    tok = _tokenizer_from_texts(["cat dog"], vocab_size=30)
    tok.special_tokens = {k: v for k, v in tok.special_tokens.items() if k != "<unk>"}
    tok._unk_id = None
    try:
        tok.encode("zzz")
    except KeyError:
        pass
    else:
        raise AssertionError("expected KeyError when <unk> is unavailable")


# ---------------------------------------------------------------------------
# Multilingual behavior
# ---------------------------------------------------------------------------


def test_roundtrip_across_english_hindi_telugu_tamil():
    texts = [
        "the price is high",
        "मूल्य अधिक है",
        "ధర ఎక్కువ",
        "விலை அதிகம்",
    ]
    tok = _tokenizer_from_texts(texts, vocab_size=400)
    for text in texts:
        decoded = tok.decode(tok.encode(text))
        assert decoded == normalize_for_roundtrip(text)


def test_fertility_is_low_for_in_vocabulary_repeated_word():
    # A word seen very frequently in training should end up as very few
    # tokens (ideally one merged token) -- a basic sanity check that
    # training + encoding actually compress repeated content.
    tok = _tokenizer_from_texts(["repeatedword " * 50], vocab_size=200)
    tokens = tok.encode_as_tokens("repeatedword")
    assert len(tokens) <= 3


# ---------------------------------------------------------------------------
# End-to-end: train -> save -> from_files -> encode/decode
# ---------------------------------------------------------------------------


def test_from_files_end_to_end(tmp_path):
    texts = ["the cat sat on the mat", "the dog sat on the log"]
    result = train_bpe_from_texts(texts, vocab_size=100)
    vocab_path = tmp_path / "vocab.json"
    merges_path = tmp_path / "merges.json"
    save_vocab(result, vocab_path)
    save_merges(result, merges_path)

    tok = BPETokenizer.from_files(vocab_path, merges_path)
    for text in texts:
        assert tok.decode(tok.encode(text)) == normalize_for_roundtrip(text)


def test_vocab_json_file_is_valid_json_with_expected_keys(tmp_path):
    result = train_bpe_from_texts(["hello world"], vocab_size=50)
    path = tmp_path / "vocab.json"
    save_vocab(result, path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert set(data.keys()) == {"vocab_size", "special_tokens", "tokens"}
