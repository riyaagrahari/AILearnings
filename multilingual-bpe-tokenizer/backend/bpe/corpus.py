"""Corpus loading and language-balancing for multilingual BPE training.

Pipeline position
------------------
    data/raw/<lang>/*.txt  --[this module]-->  weighted, combined word frequencies

This module is the only place that touches the filesystem layout the user
provides (``data/raw/en/``, ``data/raw/hi/``, ``data/raw/te/``,
``data/raw/ta/`` -- see ``data/README.md`` for the exact contract). It:

1. Reads every ``*.txt`` file in each language folder and runs it through
   :func:`bpe.preprocess.clean_text` (harmless whether the source is plain
   prose or Wikipedia-style markup).
2. Reduces each language's cleaned text to a `word -> frequency` table via
   :mod:`bpe.pretokenizer` (the same pretokenizer training uses elsewhere).
3. **Weights each language's contribution** before combining them into one
   shared table, via exponential smoothing on total word mass -- this is
   the "prevent English from dominating the vocabulary" lever from
   docs/BPE_ALGORITHM.md, §8: without it, whichever language has the most
   raw text would dominate merge selection purely by data volume, starving
   the others of learned subwords (visible as abnormally high fertility
   for the starved languages).

This module never downloads or generates data -- if a language folder is
missing or empty, it simply contributes zero words; it is the caller's
(the CLI's) job to decide whether that is acceptable or should raise.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from bpe.pretokenizer import default_pretokenize
from bpe.preprocess import clean_text
from bpe.trainer import build_word_frequencies

__all__ = [
    "DEFAULT_LANGUAGES",
    "LanguageCorpusStats",
    "discover_txt_files",
    "load_language_texts",
    "load_raw_corpus",
    "compute_language_weights",
    "apply_language_weight",
    "combine_word_frequencies",
    "build_multilingual_word_frequencies",
]

DEFAULT_LANGUAGES: tuple[str, ...] = ("en", "hi", "te", "ta")


@dataclass(frozen=True)
class LanguageCorpusStats:
    """Per-language corpus/weighting metadata, for tokenizer_config.json."""

    language: str
    num_documents: int
    raw_word_count: int
    weight: float


# ---------------------------------------------------------------------------
# Reading data/raw/<lang>/*.txt
# ---------------------------------------------------------------------------


def discover_txt_files(directory: Path) -> list[Path]:
    """List every ``*.txt`` file directly inside ``directory``, sorted for
    determinism (directory iteration order is filesystem-dependent).
    """
    if not directory.is_dir():
        return []
    return sorted(directory.glob("*.txt"))


def load_language_texts(lang_dir: Path) -> list[str]:
    """Read and clean every ``*.txt`` file in ``lang_dir``.

    Each file becomes one cleaned document string (whole-file, not
    per-line) so that :func:`bpe.preprocess.clean_text`'s multi-line-aware
    rules (headings, list markers, multi-line templates/tables) see full
    file context. Files that clean down to nothing (e.g. pure markup) are
    dropped. ``utf-8-sig`` is used so an optional leading BOM (common in
    Windows-authored files) doesn't leak a stray character into the corpus.
    """
    documents: list[str] = []
    for path in discover_txt_files(lang_dir):
        raw = path.read_text(encoding="utf-8-sig")
        cleaned = clean_text(raw)
        if cleaned:
            documents.append(cleaned)
    return documents


def load_raw_corpus(
    data_dir: str | Path,
    languages: Iterable[str] = DEFAULT_LANGUAGES,
) -> dict[str, list[str]]:
    """Load cleaned documents for every language folder under ``data_dir``.

    A missing language folder yields an empty list for that language
    rather than raising -- see the module docstring on why that decision
    is deferred to the caller.
    """
    base = Path(data_dir)
    return {lang: load_language_texts(base / lang) for lang in languages}


# ---------------------------------------------------------------------------
# Language weighting (prevents the largest corpus from dominating merges)
# ---------------------------------------------------------------------------


def compute_language_weights(word_counts: dict[str, int], alpha: float = 0.5) -> dict[str, float]:
    """Exponential-smoothing weight per language from raw word counts.

    The *target* share of the combined corpus that language ``l`` should
    end up with is the smoothed distribution
    ``target_share_l = count_l**alpha / sum(count_k**alpha for k)``.
    The weight to multiply ``l``'s raw counts by, to actually reach that
    target share once combined with every other language, is
    ``target_share_l / raw_share_l`` where
    ``raw_share_l = count_l / sum(count_k)``.

    ``alpha`` in ``[0, 1]`` controls how aggressively imbalance is
    corrected:
      - ``alpha=1`` -> ``target_share == raw_share`` for every language,
        i.e. weight is exactly 1.0 everywhere -- no correction at all.
      - ``alpha=0`` -> every language gets an equal target share
        (``1/n``) regardless of its size -- full equalization, which
        over-weights a tiny corpus's idiosyncrasies.
      - A middle value (e.g. 0.3-0.5, the recommendation in
        docs/BPE_ALGORITHM.md §8) softens size imbalance without fully
        erasing it: the language with more raw data still ends up with a
        larger share of the combined corpus, just a less dominant one.

    Languages with zero words get weight 0.0 (nothing to weight).
    """
    nonzero = {lang: count for lang, count in word_counts.items() if count > 0}
    if not nonzero:
        return {lang: 0.0 for lang in word_counts}

    total_raw = sum(nonzero.values())
    smoothed = {lang: count**alpha for lang, count in nonzero.items()}
    total_smoothed = sum(smoothed.values())

    weights: dict[str, float] = {}
    for lang, count in nonzero.items():
        raw_share = count / total_raw
        target_share = smoothed[lang] / total_smoothed
        weights[lang] = target_share / raw_share
    for lang in word_counts:
        weights.setdefault(lang, 0.0)
    return weights


def apply_language_weight(word_freqs: dict[str, int], weight: float) -> dict[str, int]:
    """Scale one language's word-frequency table by ``weight``.

    Rounded to the nearest integer (keeping the combined table's counts
    integral, matching ``trainer.train_bpe``'s existing contract) with a
    floor of 1 for any word that had nonzero frequency -- so a language
    given a very small weight still keeps every word it actually has
    *some* representation in the shared vocabulary's training signal,
    rather than being rounded away to zero and effectively erased.
    """
    scaled: dict[str, int] = {}
    for word, freq in word_freqs.items():
        if freq <= 0:
            continue
        scaled[word] = max(1, round(freq * weight))
    return scaled


def combine_word_frequencies(per_language: dict[str, dict[str, int]]) -> dict[str, int]:
    """Sum weighted per-language word-frequency tables into one shared
    table. Languages are combined in sorted-key order so the result is
    deterministic regardless of dict iteration order.
    """
    combined: dict[str, int] = {}
    for lang in sorted(per_language):
        for word, freq in per_language[lang].items():
            combined[word] = combined.get(word, 0) + freq
    return combined


def build_multilingual_word_frequencies(
    raw_texts_by_language: dict[str, list[str]],
    alpha: float = 0.5,
    pretokenize: Callable[[str], list[str]] = default_pretokenize,
    language_boosts: dict[str, float] | None = None,
) -> tuple[dict[str, int], dict[str, LanguageCorpusStats]]:
    """Full corpus -> weighted, combined `word -> frequency` table.

    Returns ``(combined_word_freqs, stats_by_language)``; the stats are
    everything ``scripts/train_tokenizer.py`` needs to record in
    ``tokenizer_config.json`` (document counts, raw word counts, and the
    weight actually applied to each language).

    ``language_boosts`` is an optional per-language multiplier applied *on
    top of* the exponential-smoothing weight from ``alpha`` (default 1.0
    for any language not listed). It lets the caller deliberately dedicate
    more of the shared vocabulary budget to a language -- e.g. boosting
    English so its fertility meets a required threshold -- accepting that,
    with a fixed vocab size, this raises the other languages' fertility.
    The boosted weight is what gets recorded in the per-language stats, so
    ``tokenizer_config.json`` reflects the weights actually used.
    """
    boosts = language_boosts or {}
    per_language_freqs: dict[str, dict[str, int]] = {}
    raw_word_counts: dict[str, int] = {}
    for lang, texts in raw_texts_by_language.items():
        freqs = build_word_frequencies(texts, pretokenize=pretokenize)
        per_language_freqs[lang] = freqs
        raw_word_counts[lang] = sum(freqs.values())

    weights = compute_language_weights(raw_word_counts, alpha=alpha)
    weights = {lang: weight * boosts.get(lang, 1.0) for lang, weight in weights.items()}
    weighted_freqs = {
        lang: apply_language_weight(freqs, weights[lang])
        for lang, freqs in per_language_freqs.items()
    }
    combined = combine_word_frequencies(weighted_freqs)

    stats = {
        lang: LanguageCorpusStats(
            language=lang,
            num_documents=len(raw_texts_by_language[lang]),
            raw_word_count=raw_word_counts[lang],
            weight=weights[lang],
        )
        for lang in raw_texts_by_language
    }
    return combined, stats
