"""Unit tests for bpe.corpus.

Uses small literal text snippets written directly in tmp_path fixtures
(not downloaded, not procedurally generated) purely to exercise the
file-discovery, cleaning and language-weighting logic in isolation.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bpe.corpus import (  # noqa: E402
    apply_language_weight,
    build_multilingual_word_frequencies,
    combine_word_frequencies,
    compute_language_weights,
    discover_txt_files,
    load_language_texts,
    load_raw_corpus,
)


# ---------------------------------------------------------------------------
# discover_txt_files / load_language_texts / load_raw_corpus
# ---------------------------------------------------------------------------


def test_discover_txt_files_returns_sorted_txt_files_only(tmp_path):
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "notes.md").write_text("ignore me", encoding="utf-8")
    files = discover_txt_files(tmp_path)
    assert [f.name for f in files] == ["a.txt", "b.txt"]


def test_discover_txt_files_missing_directory_returns_empty_list(tmp_path):
    assert discover_txt_files(tmp_path / "does_not_exist") == []


def test_load_language_texts_cleans_each_file(tmp_path):
    (tmp_path / "doc1.txt").write_text(
        "<!-- note -->'''Delhi''' is a city.", encoding="utf-8"
    )
    (tmp_path / "doc2.txt").write_text("Second document.", encoding="utf-8")
    docs = load_language_texts(tmp_path)
    assert docs == ["Delhi is a city.", "Second document."]


def test_load_language_texts_drops_documents_that_clean_to_nothing(tmp_path):
    (tmp_path / "empty_after_clean.txt").write_text("<!-- only a comment -->", encoding="utf-8")
    (tmp_path / "real.txt").write_text("Real content.", encoding="utf-8")
    docs = load_language_texts(tmp_path)
    assert docs == ["Real content."]


def test_load_language_texts_handles_utf8_bom(tmp_path):
    (tmp_path / "bom.txt").write_bytes("Hello.".encode("utf-8-sig"))
    docs = load_language_texts(tmp_path)
    assert docs == ["Hello."]
    assert "\ufeff" not in docs[0]


def test_load_raw_corpus_missing_language_folder_yields_empty_list(tmp_path):
    (tmp_path / "en").mkdir()
    (tmp_path / "en" / "a.txt").write_text("English text.", encoding="utf-8")
    corpus = load_raw_corpus(tmp_path, languages=["en", "hi"])
    assert corpus["en"] == ["English text."]
    assert corpus["hi"] == []


def test_load_raw_corpus_reads_all_four_target_languages(tmp_path):
    samples = {
        "en": "The price is high.",
        "hi": "मूल्य अधिक है।",
        "te": "ధర ఎక్కువ.",
        "ta": "விலை அதிகம்.",
    }
    for lang, text in samples.items():
        lang_dir = tmp_path / lang
        lang_dir.mkdir()
        (lang_dir / "doc.txt").write_text(text, encoding="utf-8")

    corpus = load_raw_corpus(tmp_path, languages=samples.keys())
    for lang, text in samples.items():
        assert corpus[lang] == [text]


# ---------------------------------------------------------------------------
# compute_language_weights
# ---------------------------------------------------------------------------


def test_compute_language_weights_equal_counts_gives_equal_weights():
    weights = compute_language_weights({"en": 100, "hi": 100}, alpha=0.5)
    assert weights["en"] == weights["hi"] == 1.0


def test_compute_language_weights_boosts_smaller_language():
    # en has 10x the raw word mass of hi -- with alpha < 1, hi's weight
    # should end up larger than en's, correcting the imbalance.
    weights = compute_language_weights({"en": 10_000, "hi": 1_000}, alpha=0.5)
    assert weights["hi"] > weights["en"]


def test_compute_language_weights_alpha_one_means_no_correction():
    # alpha=1 -> target share equals raw share for every language, i.e.
    # weight is exactly 1.0 everywhere (no reweighting applied at all).
    weights = compute_language_weights({"en": 9_000, "hi": 1_000}, alpha=1.0)
    assert weights["en"] == 1.0
    assert weights["hi"] == 1.0


def test_compute_language_weights_handles_zero_count_language():
    weights = compute_language_weights({"en": 100, "hi": 0}, alpha=0.5)
    assert weights["hi"] == 0.0
    assert weights["en"] > 0.0


def test_compute_language_weights_all_zero_returns_all_zero():
    weights = compute_language_weights({"en": 0, "hi": 0}, alpha=0.5)
    assert weights == {"en": 0.0, "hi": 0.0}


# ---------------------------------------------------------------------------
# apply_language_weight / combine_word_frequencies
# ---------------------------------------------------------------------------


def test_apply_language_weight_scales_and_rounds():
    scaled = apply_language_weight({"cat": 10, "dog": 3}, weight=2.0)
    assert scaled == {"cat": 20, "dog": 6}


def test_apply_language_weight_floors_nonzero_words_at_one():
    scaled = apply_language_weight({"rare": 1}, weight=0.1)
    assert scaled["rare"] == 1


def test_apply_language_weight_drops_zero_frequency_words():
    scaled = apply_language_weight({"gone": 0, "kept": 5}, weight=1.0)
    assert scaled == {"kept": 5}


def test_combine_word_frequencies_sums_shared_words_across_languages():
    combined = combine_word_frequencies({"en": {"42": 3}, "hi": {"42": 2}})
    assert combined == {"42": 5}


def test_combine_word_frequencies_is_deterministic_regardless_of_dict_order():
    a = combine_word_frequencies({"en": {"x": 1}, "hi": {"y": 1}})
    b = combine_word_frequencies({"hi": {"y": 1}, "en": {"x": 1}})
    assert a == b


# ---------------------------------------------------------------------------
# build_multilingual_word_frequencies (end-to-end)
# ---------------------------------------------------------------------------


def test_build_multilingual_word_frequencies_reports_stats_per_language():
    texts_by_lang = {
        "en": ["the cat sat"] * 10,
        "hi": ["बिल्ली बैठी"],
    }
    combined, stats = build_multilingual_word_frequencies(texts_by_lang, alpha=0.5)
    assert combined  # non-empty
    assert set(stats.keys()) == {"en", "hi"}
    assert stats["en"].num_documents == 10
    assert stats["hi"].num_documents == 1
    assert stats["en"].raw_word_count > stats["hi"].raw_word_count
    # en has far more raw data -- hi's weight should partially compensate.
    assert stats["hi"].weight > stats["en"].weight


def test_build_multilingual_word_frequencies_empty_language_contributes_nothing():
    texts_by_lang = {"en": ["hello world"], "hi": []}
    combined, stats = build_multilingual_word_frequencies(texts_by_lang)
    assert stats["hi"].raw_word_count == 0
    assert stats["hi"].weight == 0.0
    # every word in `combined` must be explainable by the English text.
    for word in combined:
        assert word.lstrip("\u2581") in ("hello", "world")
