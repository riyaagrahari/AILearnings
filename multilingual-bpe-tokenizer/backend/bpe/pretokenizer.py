"""Unicode-aware pretokenizer, shared by training and inference.

Pipeline position
------------------
    clean text (bpe.preprocess)  --[this module]-->  word-level pretokens

Both ``trainer.py`` (to build the `word -> frequency` table used to learn
merges) and ``tokenizer.py`` (to segment new text the same way before
applying learned merges, and to reconstruct spacing on decode) must split
text into words *identically* -- otherwise inference would not match what
was trained on. This module is the single, shared implementation of that
split.

Design rules
------------
- **Unicode-aware, script-agnostic**: characters are classified by their
  Unicode general category, not by a fixed set of scripts, so English,
  Hindi, Telugu and Tamil (and anything else) all flow through the same
  logic.
- **Combining marks stay with their base letter.** Devanagari/Telugu/Tamil
  vowel signs and virama are Unicode category Mn/Mc, which Python's ``\\w``
  regex class does *not* include -- using ``\\w`` directly would shatter
  every Indic word at each vowel sign (verified empirically; see
  docs/BPE_ALGORITHM.md). This module folds L (letter) and M (mark)
  categories together into one "word" class instead.
- **ZWJ (U+200D) / ZWNJ (U+200C) are preserved** and treated as part of a
  word (they control conjunct/ligature formation and are never markup or
  whitespace).
- **Digit runs are split from letter runs** (any script's digits, Unicode
  category N), and **punctuation is emitted one character at a time** --
  both to avoid combinatorial word+digit/word+punctuation vocabulary waste
  during BPE training (see docs/BPE_ALGORITHM.md, "number/punctuation
  handling").
- **Whitespace is a separator, never emitted as its own token -- but it is
  not discarded either.** Whenever a pretoken is immediately preceded by
  whitespace (or is the very first pretoken of the input), it is prefixed
  with the boundary marker ``▁`` (U+2581, the same convention SentencePiece
  uses). This is what makes ``tokenizer.py.decode()`` possible at all: a
  flat list of token ids has no other way to know where one original word
  ends and the next begins (e.g. "un"+"believ"+"able" must join with *no*
  spaces, while "the"+"cat" must join *with* a space -- that distinction
  cannot be recovered from token identity alone without a marker).
  Concatenating decoded tokens and replacing ``▁`` with a single space
  reconstructs the original spacing, up to one documented simplification:
  any run of one-or-more whitespace characters (including newlines)
  collapses to exactly one space. Paragraph breaks and repeated spaces are
  therefore not preserved byte-for-byte; word content, punctuation and
  single-space boundaries are.
"""

from __future__ import annotations

import unicodedata

__all__ = ["ZWJ", "ZWNJ", "WORD_BOUNDARY_MARKER", "default_pretokenize", "is_word_like"]

# Invisible joiners that control conjunct/ligature formation in Devanagari,
# Telugu and Tamil -- must be kept attached to the word they occur within,
# never treated as punctuation.
ZWJ = "\u200d"
ZWNJ = "\u200c"
_JOINERS = (ZWJ, ZWNJ)

# SentencePiece-style word-boundary marker. Chosen because U+2581 ("LOWER
# ONE EIGHTH BLOCK") does not occur in ordinary English/Hindi/Telugu/Tamil
# text, so it cannot collide with real content.
WORD_BOUNDARY_MARKER = "\u2581"


def _char_kind(ch: str) -> str:
    """Classify a character as "word", "digit", "space" or "punct".

    Unlike a naive ``\\w``/``\\d`` regex, this explicitly folds Unicode
    *mark* categories (Mn/Mc -- combining vowel signs, virama, etc.) into
    "word", because in Devanagari/Telugu/Tamil these marks are integral
    parts of a word's characters, not separate punctuation. Python's
    built-in ``\\w`` does *not* include them, which would otherwise shatter
    every Indic word at each vowel sign.
    """
    if ch in _JOINERS:
        return "word"
    category = unicodedata.category(ch)
    if category[0] == "N":
        return "digit"
    if category[0] in ("L", "M"):
        return "word"
    if category[0] == "Z" or ch in "\t\n\r\f\v":
        return "space"
    return "punct"


def default_pretokenize(text: str) -> list[str]:
    """Split text into word-run / digit-run / single-punctuation tokens,
    marking each pretoken that follows whitespace with a leading ``▁``.

    This is a deliberately simple scanner (one linear pass, no external
    regex/Unicode-segmentation library) that both the BPE trainer and the
    BPE tokenizer's inference path use, so that words are always split
    identically at train time and at inference time. Letters and their
    combining marks stay together as one word; digit runs are separated
    from letters; punctuation is emitted one character at a time -- see
    :func:`_char_kind`. See the module docstring for the boundary-marker
    convention that makes lossless-ish decoding possible.

    Callers that need a different splitting strategy can pass their own
    function wherever this one is used as a default (e.g.
    ``build_word_frequencies``/``train_bpe_from_texts`` in ``trainer.py``,
    or ``BPETokenizer`` in ``tokenizer.py``) via their ``pretokenize``
    parameter.
    """
    tokens: list[str] = []
    current: list[str] = []
    current_kind: str | None = None
    at_boundary = False  # True once whitespace has been consumed and the
    # next emitted pretoken must be marker-prefixed.

    def flush() -> None:
        if current:
            tokens.append("".join(current))
            current.clear()

    for ch in text:
        kind = _char_kind(ch)
        if kind == "space":
            flush()
            current_kind = None
            at_boundary = True
            continue
        if kind == "punct":
            flush()
            current_kind = None
            marker = WORD_BOUNDARY_MARKER if at_boundary else ""
            tokens.append(marker + ch)
            at_boundary = False
            continue
        # kind is "word" or "digit"
        if current_kind is not None and kind != current_kind:
            flush()
        if not current and at_boundary:
            current.append(WORD_BOUNDARY_MARKER)
            at_boundary = False
        current.append(ch)
        current_kind = kind

    flush()
    return tokens


def is_word_like(pretoken: str) -> bool:
    """True if ``pretoken`` is a word/digit-class pretoken, as opposed to
    a single-character punctuation pretoken -- i.e. exactly the pretokens
    :func:`default_pretokenize` would classify as "word" or "digit" kind,
    identified after stripping the boundary marker if present.

    Used by :mod:`bpe.evaluation` to compute fertility as tokens *per
    real word*: punctuation marks should not inflate the word count the
    way they would if every pretoken were counted indiscriminately.
    """
    stripped = pretoken[len(WORD_BOUNDARY_MARKER) :] if pretoken.startswith(WORD_BOUNDARY_MARKER) else pretoken
    if not stripped:
        return False
    return _char_kind(stripped[0]) in ("word", "digit")
