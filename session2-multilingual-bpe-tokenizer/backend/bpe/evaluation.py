"""Tokenizer evaluation metrics: fertility, UNK rate, compression,
roundtrip correctness, and vocab utilization.

Pipeline position
------------------
    trained BPETokenizer + held-out text  --[this module]-->  eval_report.json

Every metric here is measured empirically against real text (typically
``data/eval/<lang>/*.txt``, loaded the same way ``bpe.corpus`` loads
training data) rather than assumed from the training design -- the whole
point of this module is to *check* that the design goals from
docs/BPE_ALGORITHM.md actually hold, not to restate them:

- **Fertility** (tokens per real word) is the primary metric the
  assignment cares about: English should be <= 1.2, and the four
  languages should be as close to each other as the language-weighting
  in ``bpe.corpus`` can make them (see :func:`fertility_balance_score`).
- **UNK rate** should measure as (very close to) zero for in-domain text
  -- this module measures it rather than assuming it, since the
  tokenizer's "no-UNK guarantee" is honestly a base-vocabulary-coverage
  argument, not an absolute byte-level guarantee (see tokenizer.py's
  module docstring).
- **Roundtrip correctness** checks ``decode(encode(x)) == normalize(x)``
  for every document, using the exact same whitespace-normalization
  contract ``tokenizer.py`` documents (not silently comparing against a
  looser or stricter definition).
- **Vocab utilization** shows whether the shared vocabulary is genuinely
  multilingual or secretly English-dominated, even if fertility looks
  balanced.

This module also provides :func:`compute_assignment_ratios`, which
re-shapes the per-language fertility (tokens per word) into the exact
``Xi`` / ``assignment_score = 1000 / (max Xi - min Xi)`` form the
assignment rubric/widget asks for -- see that function's docstring.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from bpe.pretokenizer import is_word_like
from bpe.tokenizer import BPETokenizer, normalize_for_roundtrip

__all__ = [
    "LanguageEvalReport",
    "EvaluationReport",
    "AssignmentLanguageRatio",
    "AssignmentRatioReport",
    "evaluate_language",
    "evaluate_tokenizer",
    "fertility_balance_score",
    "save_report",
    "compute_assignment_ratios",
]


@dataclass
class LanguageEvalReport:
    """Everything measured for one language's held-out text."""

    language: str
    num_documents: int
    num_words: int
    num_tokens: int
    fertility: float
    num_unk_tokens: int
    unk_rate: float
    num_chars: int
    compression_ratio: float
    roundtrip_mismatches: int
    roundtrip_ok: bool
    vocab_utilization: float


@dataclass
class EvaluationReport:
    """The full ``eval_report.json`` payload."""

    vocab_size: int
    per_language: dict[str, LanguageEvalReport]
    fertility_balance_score: float
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def evaluate_language(
    tokenizer: BPETokenizer,
    texts: list[str],
    language: str,
) -> LanguageEvalReport:
    """Compute every metric for one language's list of held-out documents."""
    num_words = 0
    num_tokens_for_words = 0
    num_tokens_total = 0
    num_unk = 0
    num_chars = 0
    roundtrip_mismatches = 0
    used_token_ids: set[int] = set()

    unk_id = tokenizer.special_tokens.get("<unk>")

    for text in texts:
        num_chars += len(text)

        for pretoken, subtokens in tokenizer.encode_pretokens(text):
            if is_word_like(pretoken):
                num_words += 1
                num_tokens_for_words += len(subtokens)

        ids = tokenizer.encode(text)
        num_tokens_total += len(ids)
        used_token_ids.update(ids)
        if unk_id is not None:
            num_unk += sum(1 for token_id in ids if token_id == unk_id)

        if tokenizer.decode(ids) != normalize_for_roundtrip(text):
            roundtrip_mismatches += 1

    return LanguageEvalReport(
        language=language,
        num_documents=len(texts),
        num_words=num_words,
        num_tokens=num_tokens_for_words,
        fertility=_safe_divide(num_tokens_for_words, num_words),
        num_unk_tokens=num_unk,
        unk_rate=_safe_divide(num_unk, num_tokens_total),
        num_chars=num_chars,
        compression_ratio=_safe_divide(num_tokens_total, num_chars),
        roundtrip_mismatches=roundtrip_mismatches,
        roundtrip_ok=roundtrip_mismatches == 0,
        vocab_utilization=_safe_divide(len(used_token_ids), len(tokenizer.vocab)),
    )


def fertility_balance_score(per_language: dict[str, LanguageEvalReport]) -> float:
    """``max(fertility) - min(fertility)`` across languages that actually
    had eval text (``num_words > 0``). Lower is more balanced; 0.0 if
    fewer than two languages have data to compare.
    """
    fertilities = [r.fertility for r in per_language.values() if r.num_words > 0]
    if len(fertilities) < 2:
        return 0.0
    return max(fertilities) - min(fertilities)


def evaluate_tokenizer(
    tokenizer: BPETokenizer,
    texts_by_language: dict[str, list[str]],
) -> EvaluationReport:
    """Evaluate ``tokenizer`` against held-out text for every language."""
    per_language = {
        lang: evaluate_language(tokenizer, texts, lang)
        for lang, texts in texts_by_language.items()
    }
    return EvaluationReport(
        vocab_size=len(tokenizer.vocab),
        per_language=per_language,
        fertility_balance_score=fertility_balance_score(per_language),
    )


def save_report(report: EvaluationReport, path: str | Path) -> None:
    """Write the evaluation report as ``eval_report.json``."""
    payload = asdict(report)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Assignment-rubric ratio and score.
#
# The assignment defines, per language, a ratio
#     Xi = (total tokens) / (total words)
# -- i.e. fertility, tokens per real word (the same word-like counting used
# by evaluate_language above: word/digit pretokens only, punctuation
# excluded from both counts). The overall self-score is
#     assignment_score = 1000 / (max(Xi) - min(Xi))
# so a *smaller spread* of fertility across languages yields a higher
# score. This is a thin re-shaping of the fertility numbers evaluate_language
# already computes, exposed in the exact shape the assignment widget/rubric
# asks for.
# ---------------------------------------------------------------------------


@dataclass
class AssignmentLanguageRatio:
    """One language's entry in the assignment ratio report.

    ``ratio`` is ``Xi = total_tokens / total_words`` (fertility). Counts
    are word-like only (punctuation excluded), matching
    :func:`evaluate_language`'s ``num_tokens``/``num_words``.
    """

    language: str
    total_tokens: int
    total_words: int
    ratio: float


@dataclass
class AssignmentRatioReport:
    """``Xi = total_tokens / total_words`` (fertility) per language, plus
    the largest/smallest ratio, their difference, and the derived
    ``assignment_score = 1000 / difference``.
    """

    vocab_size: int
    languages: list[AssignmentLanguageRatio]
    largest_ratio: float
    smallest_ratio: float
    difference: float
    assignment_score: float | str


def compute_assignment_ratios(
    tokenizer: BPETokenizer,
    texts_by_language: dict[str, list[str]],
) -> AssignmentRatioReport:
    """Compute, for every language, ``Xi = total_tokens / total_words``
    (fertility -- tokens per real word) over its ``texts``, then derive
    the largest/smallest ratio, their difference, and
    ``assignment_score = 1000 / difference``.

    ``total_tokens`` and ``total_words`` are counted the same way
    :func:`evaluate_language` counts fertility: only word/digit pretokens
    contribute (standalone punctuation is excluded from both numerator
    and denominator), so ``ratio`` here is exactly that language's
    ``fertility`` field. This is the metric the assignment's "English
    ratio must be <= 1.2" refers to.

    Guards against divide-by-zero: a language with zero words (e.g. no
    text provided) gets ``ratio = 0.0`` and is excluded from the
    max/min; and if every language with data has an identical ratio (so
    ``difference == 0``, including the degenerate all-empty case),
    ``assignment_score`` is the string ``"Infinity"`` rather than raising
    or emitting a non-JSON-standard bare ``Infinity`` token.
    """
    languages: list[AssignmentLanguageRatio] = []
    for lang, texts in texts_by_language.items():
        report = evaluate_language(tokenizer, texts, lang)
        languages.append(
            AssignmentLanguageRatio(
                language=lang,
                total_tokens=report.num_tokens,
                total_words=report.num_words,
                ratio=report.fertility,
            )
        )

    ratios_with_data = [entry.ratio for entry in languages if entry.total_words > 0]
    largest_ratio = max(ratios_with_data) if ratios_with_data else 0.0
    smallest_ratio = min(ratios_with_data) if ratios_with_data else 0.0
    difference = largest_ratio - smallest_ratio
    assignment_score: float | str = "Infinity" if difference == 0 else 1000.0 / difference

    return AssignmentRatioReport(
        vocab_size=len(tokenizer.vocab),
        languages=languages,
        largest_ratio=largest_ratio,
        smallest_ratio=smallest_ratio,
        difference=difference,
        assignment_score=assignment_score,
    )
