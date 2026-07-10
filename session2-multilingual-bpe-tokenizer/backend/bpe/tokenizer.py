"""BPE inference: ``encode()`` / ``decode()`` using trained artifacts.

Pipeline position
------------------
    vocab.json + merges.json  --[this module]-->  token ids  <-->  text

This module is intentionally decoupled from ``trainer.py``: it only reads
the JSON artifacts training produces (plus the one pure, stateless helper
function, :func:`bpe.trainer.merge_symbols`, that both training and
inference need -- see its docstring for why sharing that one function
instead of duplicating it is the right call). Nothing here retrains
anything or depends on the trainer's internal search/heap machinery.

Algorithm
---------
``encode`` mirrors how the merges were learned: split text into pretokens
with the *same* pretokenizer used at training time (:mod:`bpe.pretokenizer`
-- this consistency is what makes inference match training at all), then
for each pretoken repeatedly find the *highest-priority* (lowest-rank)
mergeable pair still present and merge it, until no known pair remains.
This is the standard, deterministic BPE encode algorithm: given a fixed
``merges.json``, it always produces the same token sequence for the same
input.

``decode`` reverses the id -> token mapping and joins tokens back into
text, using the same ``▁`` boundary-marker convention documented in
``pretokenizer.py`` to reconstruct spacing.

No-UNK guarantee, honestly stated
----------------------------------
The base vocabulary (see ``trainer.build_base_vocabulary``) covers every
Unicode codepoint seen during training. For text in the same four target
languages that training was performed on, this makes ``<unk>`` usage rare
to nonexistent in practice -- but it is not a byte-level fallback, so a
truly novel codepoint (a script never seen in training) has no vocab
entry and is mapped to ``<unk>`` rather than crashing. The evaluation
pipeline (``bpe.evaluation``) measures this rate empirically on real
held-out text instead of merely asserting it.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from bpe.pretokenizer import WORD_BOUNDARY_MARKER, default_pretokenize
from bpe.trainer import Merge, merge_symbols

__all__ = [
    "BPETokenizer",
    "load_vocab",
    "load_merges",
    "normalize_for_roundtrip",
]

# Special tokens that represent structure, not recoverable content -- they
# are dropped (not rendered literally) on decode. "<unk>" is deliberately
# excluded: it stands in for real content that could not be represented,
# so hiding it on decode would be misleading.
_SILENT_SPECIAL_TOKENS = frozenset({"<pad>", "<bos>", "<eos>"})

_WHITESPACE_RUN_RE = re.compile(r"\s+")


def normalize_for_roundtrip(text: str) -> str:
    """Canonicalize text the same way ``encode``/``decode`` implicitly do.

    ``decode(encode(x)) == x`` only holds up to whitespace normalization
    (every run of whitespace collapses to one space; trailing whitespace
    is dropped entirely) -- this is a direct, documented consequence of
    the pretokenizer's boundary-marker convention, not a bug. Tests and
    the evaluation pipeline's roundtrip metric compare against
    ``normalize_for_roundtrip(x)`` rather than ``x`` itself, so that
    metric reflects a real, well-defined contract instead of silently
    failing on the one input property (exact whitespace) this design
    never claimed to preserve.
    """
    return _WHITESPACE_RUN_RE.sub(" ", text).rstrip()


def load_vocab(path: str | Path) -> tuple[dict[str, int], dict[str, int]]:
    """Load ``vocab.json`` (see trainer.save_vocab for the schema).

    Returns ``(vocab, special_tokens)`` where ``vocab`` maps every token
    string (including special tokens) to its id.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    vocab: dict[str, int] = {token: int(idx) for idx, token in data["tokens"].items()}
    special_tokens: dict[str, int] = dict(data["special_tokens"])
    vocab.update(special_tokens)
    return vocab, special_tokens


def load_merges(path: str | Path) -> list[Merge]:
    """Load ``merges.json`` (see trainer.save_merges for the schema)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        Merge(rank=entry["rank"], pair=tuple(entry["pair"]), result=entry["result"])
        for entry in data["merges"]
    ]


@dataclass
class BPETokenizer:
    """Encode/decode text using a trained vocabulary and merge list."""

    vocab: dict[str, int]
    special_tokens: dict[str, int]
    merges: list[Merge]
    pretokenize: Callable[[str], list[str]] = default_pretokenize

    def __post_init__(self) -> None:
        self._merge_rank: dict[tuple[str, str], int] = {m.pair: m.rank for m in self.merges}
        self._id_to_token: dict[int, str] = {idx: token for token, idx in self.vocab.items()}
        self._unk_id: int | None = self.special_tokens.get("<unk>")

    @classmethod
    def from_files(cls, vocab_path: str | Path, merges_path: str | Path) -> "BPETokenizer":
        vocab, special_tokens = load_vocab(vocab_path)
        merges = load_merges(merges_path)
        return cls(vocab=vocab, special_tokens=special_tokens, merges=merges)

    # -- encoding ----------------------------------------------------------

    def _apply_merges_to_word(self, symbols: list[str]) -> list[str]:
        """Repeatedly apply the highest-priority (lowest-rank) mergeable
        pair still present, until none remain -- the standard BPE encode
        algorithm for a single pretoken.
        """
        if len(symbols) < 2:
            return symbols
        while True:
            best_rank: int | None = None
            best_pair: tuple[str, str] | None = None
            for pair in zip(symbols, symbols[1:]):
                rank = self._merge_rank.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_pair = pair
            if best_pair is None:
                break
            merged = best_pair[0] + best_pair[1]
            symbols = merge_symbols(symbols, best_pair, merged)
        return symbols

    def encode_pretokens(self, text: str) -> list[tuple[str, list[str]]]:
        """Return each pretoken alongside the BPE subtokens it was split
        into, e.g. ``[("▁unbelievable", ["▁un", "believ", "able"])]``.

        This is the structure :func:`encode_as_tokens` flattens away.
        Evaluation metrics need it back: fertility must count tokens
        *per real word*, which means excluding punctuation pretokens
        from both the word count and the token count -- impossible to
        do correctly from a flat token list alone, since a flattened
        list has no record of which original pretoken each token came
        from. It is also what a frontend merge-step visualizer would
        want, one word at a time.
        """
        return [
            (pretoken, self._apply_merges_to_word(list(pretoken)))
            for pretoken in self.pretokenize(text)
        ]

    def encode_as_tokens(self, text: str) -> list[str]:
        """Split ``text`` into the final token strings ``encode`` would
        produce ids for -- useful for the frontend's merge-step display
        and for evaluation metrics that need token text, not just ids.
        """
        tokens: list[str] = []
        for _pretoken, subtokens in self.encode_pretokens(text):
            tokens.extend(subtokens)
        return tokens

    def encode(self, text: str) -> list[int]:
        """Encode ``text`` into a flat list of token ids.

        Any symbol with no vocabulary entry (see the module docstring's
        "No-UNK guarantee, honestly stated") is mapped to ``<unk>``'s id
        if one is configured, otherwise raises -- there is deliberately
        no silent data loss.
        """
        ids: list[int] = []
        for token in self.encode_as_tokens(text):
            token_id = self.vocab.get(token)
            if token_id is None:
                if self._unk_id is None:
                    raise KeyError(
                        f"Token {token!r} not in vocabulary and no '<unk>' "
                        "special token is configured to fall back to."
                    )
                token_id = self._unk_id
            ids.append(token_id)
        return ids

    # -- decoding ------------------------------------------------------------

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token ids back into text.

        Structural special tokens (``<pad>``/``<bos>``/``<eos>``) are
        dropped silently; ``<unk>`` is rendered literally, since it
        represents real content that could not be reconstructed -- see
        the module docstring. Spacing is reconstructed via the ``▁``
        boundary-marker convention from ``pretokenizer.py``; see
        :func:`normalize_for_roundtrip` for the exact contract this
        provides.
        """
        pieces: list[str] = []
        for token_id in ids:
            token = self._id_to_token.get(token_id)
            if token is None or token in _SILENT_SPECIAL_TOKENS:
                continue
            pieces.append(token)
        return "".join(pieces).replace(WORD_BOUNDARY_MARKER, " ")
