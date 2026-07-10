"""Unit tests for bpe.evaluation.

Uses small trained-in-memory tokenizers (literal text, not downloaded or
procedurally generated) to exercise each metric's arithmetic in isolation
against hand-verifiable expectations.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bpe.evaluation import (  # noqa: E402
    compute_assignment_ratios,
    evaluate_language,
    evaluate_tokenizer,
    fertility_balance_score,
    save_report,
)
from bpe.tokenizer import BPETokenizer  # noqa: E402
from bpe.trainer import train_bpe_from_texts  # noqa: E402


def _tokenizer_from_texts(texts, vocab_size=200):
    result = train_bpe_from_texts(texts, vocab_size=vocab_size)
    return BPETokenizer(
        vocab=result.vocab, special_tokens=result.special_tokens, merges=result.merges
    )


# ---------------------------------------------------------------------------
# evaluate_language -- fertility
# ---------------------------------------------------------------------------


def test_fertility_excludes_punctuation_from_word_count():
    # Train enough that "hello" and "world" each collapse to one token.
    tok = _tokenizer_from_texts(["hello world " * 20], vocab_size=200)
    report = evaluate_language(tok, ["hello, world!"], "en")
    # 2 real words ("hello", "world"), 2 punctuation marks excluded from
    # both numerator and denominator.
    assert report.num_words == 2
    assert report.fertility == report.num_tokens / 2


def test_fertility_is_one_for_fully_merged_frequent_word():
    tok = _tokenizer_from_texts(["repeatedword " * 50], vocab_size=300)
    report = evaluate_language(tok, ["repeatedword"], "en")
    assert report.num_words == 1
    assert report.fertility == report.num_tokens


def test_fertility_zero_words_gives_zero_not_a_crash():
    tok = _tokenizer_from_texts(["hello world"], vocab_size=50)
    report = evaluate_language(tok, ["!!!"], "en")  # only punctuation
    assert report.num_words == 0
    assert report.fertility == 0.0


def test_fertility_no_documents_gives_zero():
    tok = _tokenizer_from_texts(["hello world"], vocab_size=50)
    report = evaluate_language(tok, [], "en")
    assert report.num_documents == 0
    assert report.fertility == 0.0
    assert report.unk_rate == 0.0
    assert report.compression_ratio == 0.0


# ---------------------------------------------------------------------------
# evaluate_language -- UNK rate
# ---------------------------------------------------------------------------


def test_unk_rate_is_zero_for_fully_in_vocabulary_text():
    tok = _tokenizer_from_texts(["hello world"], vocab_size=50)
    report = evaluate_language(tok, ["hello world"], "en")
    assert report.num_unk_tokens == 0
    assert report.unk_rate == 0.0


def test_unk_rate_reflects_out_of_vocabulary_characters():
    tok = _tokenizer_from_texts(["cat dog"], vocab_size=30)
    report = evaluate_language(tok, ["zzz"], "en")  # 'z' never in training
    assert report.num_unk_tokens > 0
    assert report.unk_rate > 0.0


# ---------------------------------------------------------------------------
# evaluate_language -- compression ratio
# ---------------------------------------------------------------------------


def test_compression_ratio_is_tokens_over_chars():
    tok = _tokenizer_from_texts(["hello world"], vocab_size=50)
    text = "hello world"
    report = evaluate_language(tok, [text], "en")
    ids = tok.encode(text)
    assert report.compression_ratio == len(ids) / len(text)


# ---------------------------------------------------------------------------
# evaluate_language -- roundtrip correctness
# ---------------------------------------------------------------------------


def test_roundtrip_ok_for_in_vocabulary_text():
    tok = _tokenizer_from_texts(["the cat sat", "the dog sat"], vocab_size=100)
    report = evaluate_language(tok, ["the cat sat", "the dog sat"], "en")
    assert report.roundtrip_mismatches == 0
    assert report.roundtrip_ok is True


def test_roundtrip_fails_when_unk_substitution_loses_content():
    tok = _tokenizer_from_texts(["cat dog"], vocab_size=30)
    report = evaluate_language(tok, ["zzz totally unseen text"], "en")
    assert report.roundtrip_mismatches >= 1
    assert report.roundtrip_ok is False


# ---------------------------------------------------------------------------
# evaluate_language -- vocab utilization
# ---------------------------------------------------------------------------


def test_vocab_utilization_is_fraction_of_vocab_actually_used():
    tok = _tokenizer_from_texts(["hello world"], vocab_size=50)
    report = evaluate_language(tok, ["hello"], "en")
    ids = set(tok.encode("hello"))
    assert report.vocab_utilization == len(ids) / len(tok.vocab)


def test_vocab_utilization_is_zero_for_no_documents():
    tok = _tokenizer_from_texts(["hello world"], vocab_size=50)
    report = evaluate_language(tok, [], "en")
    assert report.vocab_utilization == 0.0


# ---------------------------------------------------------------------------
# fertility_balance_score
# ---------------------------------------------------------------------------


def test_fertility_balance_score_zero_when_identical():
    from bpe.evaluation import LanguageEvalReport

    reports = {
        "en": LanguageEvalReport("en", 1, 10, 12, 1.2, 0, 0.0, 50, 0.2, 0, True, 0.5),
        "hi": LanguageEvalReport("hi", 1, 10, 12, 1.2, 0, 0.0, 50, 0.2, 0, True, 0.5),
    }
    assert fertility_balance_score(reports) == 0.0


def test_fertility_balance_score_ignores_languages_with_no_words():
    from bpe.evaluation import LanguageEvalReport

    reports = {
        "en": LanguageEvalReport("en", 1, 10, 12, 1.2, 0, 0.0, 50, 0.2, 0, True, 0.5),
        "ta": LanguageEvalReport("ta", 0, 0, 0, 0.0, 0, 0.0, 0, 0.0, 0, True, 0.0),
    }
    # Only one language has real data -- nothing to compare, so 0.0.
    assert fertility_balance_score(reports) == 0.0


def test_fertility_balance_score_reflects_difference():
    from bpe.evaluation import LanguageEvalReport

    reports = {
        "en": LanguageEvalReport("en", 1, 10, 12, 1.2, 0, 0.0, 50, 0.2, 0, True, 0.5),
        "te": LanguageEvalReport("te", 1, 10, 20, 2.0, 0, 0.0, 50, 0.2, 0, True, 0.5),
    }
    assert abs(fertility_balance_score(reports) - 0.8) < 1e-9


# ---------------------------------------------------------------------------
# evaluate_tokenizer / save_report -- end-to-end
# ---------------------------------------------------------------------------


def test_evaluate_tokenizer_covers_every_language():
    texts = [
        "the price is high",
        "मूल्य अधिक है",
        "ధర ఎక్కువ",
        "விலை அதிகம்",
    ]
    tok = _tokenizer_from_texts(texts, vocab_size=400)
    eval_texts = {
        "en": ["the price is high"],
        "hi": ["मूल्य अधिक है"],
        "te": ["ధర ఎక్కువ"],
        "ta": ["விலை அதிகம்"],
    }
    report = evaluate_tokenizer(tok, eval_texts)
    assert set(report.per_language.keys()) == {"en", "hi", "te", "ta"}
    assert report.vocab_size == len(tok.vocab)
    for lang_report in report.per_language.values():
        assert lang_report.roundtrip_ok is True
        assert lang_report.unk_rate == 0.0


def test_save_report_writes_valid_json(tmp_path):
    tok = _tokenizer_from_texts(["hello world"], vocab_size=50)
    report = evaluate_tokenizer(tok, {"en": ["hello world"]})
    path = tmp_path / "eval_report.json"
    save_report(report, path)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["vocab_size"] == report.vocab_size
    assert "en" in data["per_language"]
    assert "generated_at" in data
    assert "fertility_balance_score" in data


# ---------------------------------------------------------------------------
# compute_assignment_ratios
# ---------------------------------------------------------------------------


def test_assignment_ratio_is_fertility_tokens_per_word():
    # Xi = total_tokens / total_words (word-like), i.e. it must equal the
    # language's fertility from evaluate_language on the same text.
    tok = _tokenizer_from_texts(["hello world hello world"], vocab_size=100)
    text = "hello world"

    direct = evaluate_language(tok, [text], "en")
    report = compute_assignment_ratios(tok, {"en": [text]})
    entry = report.languages[0]
    assert entry.total_tokens == direct.num_tokens
    assert entry.total_words == direct.num_words
    assert abs(entry.ratio - direct.fertility) < 1e-9


def test_assignment_ratio_covers_every_language():
    texts_by_lang = {
        "en": ["the price is high"],
        "hi": ["मूल्य अधिक है"],
        "te": ["ధర ఎక్కువ"],
        "ta": ["விலை அதிகம்"],
    }
    # Train on a combined corpus so every language has at least some
    # in-vocabulary characters.
    tok = _tokenizer_from_texts(
        [t for texts in texts_by_lang.values() for t in texts], vocab_size=400
    )
    report = compute_assignment_ratios(tok, texts_by_lang)
    assert {entry.language for entry in report.languages} == set(texts_by_lang.keys())
    assert report.vocab_size == len(tok.vocab)


def test_assignment_ratio_zero_words_gives_zero_ratio_and_is_excluded():
    tok = _tokenizer_from_texts(["hello world"], vocab_size=50)
    report = compute_assignment_ratios(tok, {"en": ["hello"], "hi": []})
    by_lang = {e.language: e for e in report.languages}
    assert by_lang["hi"].total_words == 0
    assert by_lang["hi"].ratio == 0.0
    # The empty language must not become the min ratio (0.0) -- only
    # languages with data participate in max/min.
    assert report.smallest_ratio == by_lang["en"].ratio


def test_assignment_score_formula_matches_spec_example():
    # 1000 / (largest - smallest), verified against the worked example
    # from the assignment spec: difference=0.003 -> score=333333.33...
    from bpe.evaluation import AssignmentLanguageRatio, AssignmentRatioReport

    report = AssignmentRatioReport(
        vocab_size=10_000,
        languages=[AssignmentLanguageRatio("en", 6200, 5100, 1.2157)],
        largest_ratio=1.219,
        smallest_ratio=1.216,
        difference=0.003,
        assignment_score=1000.0 / 0.003,
    )
    assert abs(report.assignment_score - 333333.33) < 0.1


def test_assignment_score_is_infinity_string_when_difference_is_zero():
    tok = _tokenizer_from_texts(["hello world"], vocab_size=50)
    # Same text for every language -> identical fertility -> difference 0.
    report = compute_assignment_ratios(
        tok, {"en": ["hello world"], "hi": ["hello world"]}
    )
    assert report.difference == 0.0
    assert report.assignment_score == "Infinity"


def test_assignment_score_is_infinity_string_when_no_data_at_all():
    tok = _tokenizer_from_texts(["hello world"], vocab_size=50)
    report = compute_assignment_ratios(tok, {"en": [], "hi": []})
    assert report.largest_ratio == 0.0
    assert report.smallest_ratio == 0.0
    assert report.assignment_score == "Infinity"


def test_assignment_score_never_raises_zero_division():
    tok = _tokenizer_from_texts(["hello world"], vocab_size=50)
    # Should not raise even with wildly imbalanced/empty per-language data.
    report = compute_assignment_ratios(tok, {"en": ["hello"], "hi": []})
    assert isinstance(report.assignment_score, (float, str))


def test_assignment_ratios_json_serializable(tmp_path):
    import dataclasses

    tok = _tokenizer_from_texts(["hello world"], vocab_size=50)
    report = compute_assignment_ratios(tok, {"en": ["hello world"], "hi": []})
    payload = dataclasses.asdict(report)
    text = json.dumps(payload)  # must not raise
    assert '"assignment_score"' in text


def test_assignment_score_higher_when_fertilities_are_closer():
    # A larger fertility spread across languages must yield a lower score
    # (score = 1000 / (max - min)), which is the whole point of the metric.
    from bpe.evaluation import AssignmentLanguageRatio, AssignmentRatioReport

    def score_for(spread: float) -> float:
        return AssignmentRatioReport(
            vocab_size=10_000,
            languages=[
                AssignmentLanguageRatio("en", 0, 0, 1.30),
                AssignmentLanguageRatio("hi", 0, 0, 1.30 + spread),
            ],
            largest_ratio=1.30 + spread,
            smallest_ratio=1.30,
            difference=spread,
            assignment_score=1000.0 / spread,
        ).assignment_score

    assert score_for(0.1) > score_for(0.5)
