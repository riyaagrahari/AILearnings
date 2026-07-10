"""Unit tests for bpe.pretokenizer.

Covers Unicode-aware word/digit/punctuation splitting -- with particular
attention to Devanagari/Telugu/Tamil combining marks and ZWJ/ZWNJ, which a
naive ``\\w``/``\\d`` regex would mishandle (verified empirically -- see
docs/BPE_ALGORITHM.md) -- and the ``▁`` word-boundary marker convention
that makes ``tokenizer.py.decode()`` possible.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bpe.pretokenizer import (  # noqa: E402
    WORD_BOUNDARY_MARKER as M,
)
from bpe.pretokenizer import (  # noqa: E402
    ZWJ,
    ZWNJ,
    default_pretokenize,
)


def _reconstruct(tokens: list[str]) -> str:
    """Mirror tokenizer.py.decode()'s join/marker-replace logic, for
    testing the pretokenizer's boundary-marking contract in isolation.
    """
    return "".join(tokens).replace(M, " ")


# ---------------------------------------------------------------------------
# Basic splitting: words / digits / punctuation
# ---------------------------------------------------------------------------


def test_pretokenize_splits_english_words_punctuation_digits():
    assert default_pretokenize("Hello, world! 42") == [
        "Hello",
        ",",
        f"{M}world",
        "!",
        f"{M}42",
    ]


def test_pretokenize_separates_digit_runs_from_letters():
    assert default_pretokenize("abc123") == ["abc", "123"]


def test_pretokenize_emits_each_punctuation_character_separately():
    assert default_pretokenize("Wait...really?!") == [
        "Wait",
        ".",
        ".",
        ".",
        "really",
        "?",
        "!",
    ]


def test_pretokenize_handles_empty_string():
    assert default_pretokenize("") == []


def test_pretokenize_handles_pure_whitespace_string():
    assert default_pretokenize("   \t\n  ") == []


# ---------------------------------------------------------------------------
# Indic combining marks / ZWJ / ZWNJ
# ---------------------------------------------------------------------------


def test_pretokenize_keeps_devanagari_matras_attached_to_word():
    # "की" = "क" + vowel-sign "ी" (category Mc) -- must stay one word, not
    # be split at the combining mark like a naive \w-based regex would.
    assert default_pretokenize("की") == ["की"]
    assert default_pretokenize("रुपये") == ["रुपये"]


def test_pretokenize_keeps_telugu_virama_attached_to_word():
    assert default_pretokenize("త్ర") == ["త్ర"]


def test_pretokenize_keeps_tamil_vowel_signs_attached_and_splits_digits_punct():
    assert default_pretokenize("விலை42.") == ["விலை", "42", "."]


def test_pretokenize_preserves_zwnj_within_a_word():
    text = f"क्{ZWNJ}ष जैसे"
    tokens = default_pretokenize(text)
    assert tokens[0] == f"क्{ZWNJ}ष"
    assert ZWNJ in tokens[0]


def test_pretokenize_preserves_zwj_within_a_word():
    text = f"త్{ZWJ}ర పదం"
    tokens = default_pretokenize(text)
    assert tokens[0] == f"త్{ZWJ}ర"


def test_pretokenize_native_script_digits_treated_as_digit_class():
    # Devanagari digits (category Nd) should split from surrounding letters
    # exactly like Arabic digits do.
    text = "मूल्य\u0967\u0968\u0969रुपये"
    tokens = default_pretokenize(text)
    assert "\u0967\u0968\u0969" in tokens


# ---------------------------------------------------------------------------
# Word-boundary marker (▁)
# ---------------------------------------------------------------------------


def test_first_pretoken_has_no_marker_when_not_preceded_by_whitespace():
    assert default_pretokenize("hello world")[0] == "hello"


def test_pretoken_after_whitespace_gets_marker():
    assert default_pretokenize("hello world")[1] == f"{M}world"


def test_punctuation_gets_marker_only_when_preceded_by_whitespace():
    assert default_pretokenize("word!") == ["word", "!"]
    assert default_pretokenize("word !") == ["word", f"{M}!"]


def test_multiple_consecutive_whitespace_yields_a_single_marker():
    assert default_pretokenize("hello   world") == ["hello", f"{M}world"]


def test_leading_whitespace_produces_a_leading_marker():
    assert default_pretokenize("  hello") == [f"{M}hello"]


def test_marker_never_appears_mid_run():
    for token in default_pretokenize("the quick, brown42fox"):
        assert token.count(M) <= 1
        if M in token:
            assert token.startswith(M)


# ---------------------------------------------------------------------------
# Reconstruction contract (join + replace marker with space)
# ---------------------------------------------------------------------------


def test_reconstruction_matches_original_for_single_spaced_text():
    text = "Hello, world! 42"
    assert _reconstruct(default_pretokenize(text)) == text


def test_reconstruction_collapses_repeated_whitespace_to_single_space():
    tokens = default_pretokenize("hello   world")
    assert _reconstruct(tokens) == "hello world"


def test_reconstruction_of_multilingual_sentence():
    text = "The price is मूल्य ధర விலை."
    assert _reconstruct(default_pretokenize(text)) == text
