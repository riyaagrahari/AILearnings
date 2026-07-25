"""The eight Session-3 data-cleaning strategies, implemented from scratch.

Session 3 (`session3-report-training`) defines an eight-stage cleaning pipeline
in ``src/data/cleaning.ts``:

    1. Raw Documents (ingest)          5. Quality Scoring
    2. Language Identification         6. Safety & Toxicity Filtering
    3. Unicode Normalization           7. Code-Repository Filtering
    4. Deduplication                   8. Training Corpus (assembly)

Stages 1 and 8 are ingest/ship bookends; stages 2-7 are the six active
"cleanups". Every function here is pure-Python/NumPy (no NLP libraries) and
returns a ``StageResult`` carrying the surviving documents plus real, countable
statistics that the Session-4 widget renders verbatim.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import regex  # Unicode-property regex (\p{L}\p{M}\p{N}) for correct Indic tokenizing

# A word = a run of letters, numbers and (crucially) combining MARKS. Python's
# stdlib ``\w`` excludes Mn/Mc marks, which would shatter every Devanagari/
# Telugu word at each vowel-sign/virama -- exactly the trap Session 2 flagged.
_word_re = regex.compile(r"[\p{L}\p{N}\p{M}_]+")
_letter_re = regex.compile(r"[\p{L}\p{M}]")

# --------------------------------------------------------------------------- #
# Shared document type + stage result container
# --------------------------------------------------------------------------- #


@dataclass
class Doc:
    id: str
    url: str
    title: str
    text: str
    src_lang: str  # language label from the source dump (hi / te / mr)
    # populated as the doc flows through the pipeline:
    content_hash: str = ""
    det_script: str = ""
    quality: float = 0.0
    flags: set[str] = field(default_factory=set)


@dataclass
class StageResult:
    docs: list[Doc]
    stats: dict[str, Any]


# --------------------------------------------------------------------------- #
# Unicode script ranges (letters only) used by language identification
# --------------------------------------------------------------------------- #

_SCRIPT_RANGES = {
    "devanagari": (0x0900, 0x097F),  # Hindi, Marathi, Sanskrit, ...
    "telugu": (0x0C00, 0x0C7F),
    "tamil": (0x0B80, 0x0BFF),
}
_EXPECTED_SCRIPT = {"hi": "devanagari", "mr": "devanagari", "te": "telugu"}

# Marathi-specific markers (letter ळ + very common function words) that let us
# spot Hindi text mislabelled as Marathi (both use Devanagari, so script alone
# cannot separate them).
_MARATHI_MARKERS = ("ळ", " आणि ", " आहे ", " होते ", " व ", " यांनी ", " मराठी ")
_HINDI_MARKERS = (" है ", " और ", " के ", " में ", " था ", " हैं ", " हिंदी ")


def _letter_script_counts(text: str) -> tuple[Counter, int, int]:
    """Return (script->count, total_letters, latin_letters)."""
    counts: Counter = Counter()
    total = 0
    latin = 0
    for ch in text:
        cp = ord(ch)
        if (0x41 <= cp <= 0x5A) or (0x61 <= cp <= 0x7A):
            latin += 1
            total += 1
            continue
        if not ch.isalpha():
            continue
        total += 1
        for name, (lo, hi) in _SCRIPT_RANGES.items():
            if lo <= cp <= hi:
                counts[name] += 1
                break
    return counts, total, latin


def detect(text: str) -> tuple[str, float, float]:
    """Return (dominant_script, script_confidence, latin_ratio)."""
    counts, total, latin = _letter_script_counts(text)
    if total == 0:
        return "none", 0.0, 0.0
    counts["latin"] = latin
    script, n = counts.most_common(1)[0]
    return script, n / total, latin / total


# --------------------------------------------------------------------------- #
# STAGE 1 -- Raw Documents (ingest, provenance, content-hash)
# --------------------------------------------------------------------------- #


def stage_raw(docs: list[Doc]) -> StageResult:
    kept: list[Doc] = []
    empty = 0
    chars = 0
    per_lang: Counter = Counter()
    for d in docs:
        text = d.text or ""
        if not text.strip():
            empty += 1
            continue
        d.content_hash = hashlib.sha1(text.encode("utf-8")).hexdigest()
        kept.append(d)
        chars += len(text)
        per_lang[d.src_lang] += 1
    return StageResult(
        kept,
        {
            "docs_in": len(docs),
            "docs_out": len(kept),
            "removed": empty,
            "removed_reasons": {"empty_document": empty},
            "chars_out": chars,
            "per_language_docs": dict(per_lang),
            "provenance": "wikimedia/wikipedia",
            "license": "CC BY-SA 4.0",
        },
    )


# --------------------------------------------------------------------------- #
# STAGE 2 -- Language Identification (script routing + code-switch)
# --------------------------------------------------------------------------- #


def stage_langid(docs: list[Doc]) -> StageResult:
    kept: list[Doc] = []
    removed_wrong_script = 0
    review_low_conf = 0
    review_code_switch = 0
    mislabelled_hi_mr = 0
    examples: list[dict[str, Any]] = []
    for d in docs:
        script, conf, latin_ratio = detect(d.text)
        d.det_script = script
        expected = _EXPECTED_SCRIPT[d.src_lang]

        # Hard drop: dominant script is clearly not the expected Indic script
        # (e.g. an all-English stub sitting in an Indic dump).
        if script != expected and conf >= 0.5:
            removed_wrong_script += 1
            if len(examples) < 5:
                examples.append({"title": d.title[:80], "src_lang": d.src_lang,
                                 "detected": script, "confidence": round(conf, 2)})
            continue

        # Heavy Latin content inside an Indic doc -> code-switch, route to review.
        if 0.15 <= latin_ratio < 0.5:
            d.flags.add("code_switch")
            review_code_switch += 1

        # Low overall confidence -> keep but flag for the human review pool
        # (we never silently discard low-confidence Indic data).
        if conf < 0.6:
            d.flags.add("low_confidence_langid")
            review_low_conf += 1

        # Hindi vs Marathi share Devanagari -> use marker words to catch swaps.
        if d.src_lang in ("hi", "mr"):
            hay = " " + d.text[:4000] + " "
            hi_hits = sum(hay.count(m) for m in _HINDI_MARKERS)
            mr_hits = sum(hay.count(m) for m in _MARATHI_MARKERS)
            looks = "mr" if mr_hits > hi_hits else "hi"
            if looks != d.src_lang and max(hi_hits, mr_hits) >= 3:
                d.flags.add("possible_language_mismatch")
                mislabelled_hi_mr += 1

        kept.append(d)
    return StageResult(
        kept,
        {
            "docs_in": len(docs),
            "docs_out": len(kept),
            "removed": removed_wrong_script,
            "removed_reasons": {"wrong_dominant_script": removed_wrong_script},
            "review_pool": {
                "low_confidence": review_low_conf,
                "code_switch": review_code_switch,
                "hi_mr_mismatch_flagged": mislabelled_hi_mr,
            },
            "examples_removed": examples,
        },
    )


# --------------------------------------------------------------------------- #
# STAGE 3 -- Unicode Normalization (NFC, ZWJ/ZWNJ-safe, whitespace/quote/digit)
# --------------------------------------------------------------------------- #

ZWJ, ZWNJ = "\u200d", "\u200c"
_SMART_QUOTES = {
    "\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
    "\u00ab": '"', "\u00bb": '"',
}
_WS_RUN = re.compile(r"[ \t\u00a0]+")
_NL_RUN = re.compile(r"\n{3,}")


def _normalize_one(text: str) -> tuple[str, dict[str, int]]:
    counters = {"nfc_changed": 0, "quotes": 0, "fullwidth_digits": 0,
               "whitespace_chars": 0, "zwj_zwnj_preserved": 0}
    counters["zwj_zwnj_preserved"] = text.count(ZWJ) + text.count(ZWNJ)

    nfc = unicodedata.normalize("NFC", text)
    if nfc != text:
        counters["nfc_changed"] = 1
    text = nfc

    out = []
    for ch in text:
        if ch in _SMART_QUOTES:
            out.append(_SMART_QUOTES[ch])
            counters["quotes"] += 1
        elif 0xFF10 <= ord(ch) <= 0xFF19:  # fullwidth ASCII digits -> ASCII
            out.append(chr(ord(ch) - 0xFEE0))
            counters["fullwidth_digits"] += 1
        else:
            out.append(ch)
    text = "".join(out)

    before = len(text)
    text = _WS_RUN.sub(" ", text)
    text = _NL_RUN.sub("\n\n", text)
    text = "\n".join(line.rstrip() for line in text.split("\n")).strip()
    counters["whitespace_chars"] = max(0, before - len(text))
    return text, counters


def _snippet_around_quote(text: str) -> str | None:
    for i, ch in enumerate(text):
        if ch in _SMART_QUOTES:
            return text[max(0, i - 30):i + 30]
    return None


def stage_normalize(docs: list[Doc]) -> StageResult:
    totals = Counter()
    chars_in = chars_out = 0
    example = None
    for d in docs:
        chars_in += len(d.text)
        if example is None:
            before = _snippet_around_quote(d.text)
            if before:
                example = {"before": before, "after": _normalize_one(before)[0]}
        d.text, c = _normalize_one(d.text)
        # re-hash: normalization is what dedup should compare on.
        d.content_hash = hashlib.sha1(d.text.encode("utf-8")).hexdigest()
        chars_out += len(d.text)
        for k, v in c.items():
            totals[k] += v
    return StageResult(
        docs,
        {
            "docs_in": len(docs),
            "docs_out": len(docs),
            "removed": 0,
            "chars_in": chars_in,
            "chars_out": chars_out,
            "chars_normalized_away": chars_in - chars_out,
            "counters": dict(totals),
            "example": example,
        },
    )


# --------------------------------------------------------------------------- #
# STAGE 4 -- Deduplication (exact content-hash + from-scratch MinHash/LSH)
# --------------------------------------------------------------------------- #

_MINHASH_K = 64
_LSH_BANDS = 16          # bands * rows == K
_LSH_ROWS = 4
_NEAR_DUP_JACCARD = 0.80
_SHINGLE_WORDS = 5
_MAX_SHINGLES = 400
_MERSENNE = (1 << 61) - 1


def _shingle_hashes(text: str) -> np.ndarray:
    words = _word_re.findall(text.lower())
    if len(words) < _SHINGLE_WORDS:
        shingles = {" ".join(words)} if words else set()
    else:
        shingles = {
            " ".join(words[i:i + _SHINGLE_WORDS])
            for i in range(len(words) - _SHINGLE_WORDS + 1)
        }
    hs = [int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(),
                         "little") & 0xFFFFFFFFFFFFFFFF for s in shingles]
    if not hs:
        return np.zeros(0, dtype=np.uint64)
    if len(hs) > _MAX_SHINGLES:
        hs = hs[:_MAX_SHINGLES]
    return np.array(hs, dtype=np.uint64)


def _signature(sh: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if sh.size == 0:
        return np.full(_MINHASH_K, np.iinfo(np.uint64).max, dtype=np.uint64)
    # (a[:,None]*sh + b) mod prime  ->  min over shingles
    prod = (a[:, None] * sh[None, :] + b[:, None]) % _MERSENNE
    return prod.min(axis=1)


def stage_dedup(docs: list[Doc]) -> StageResult:
    rng = np.random.default_rng(20240724)
    a = rng.integers(1, _MERSENNE, size=_MINHASH_K, dtype=np.uint64)
    b = rng.integers(0, _MERSENNE, size=_MINHASH_K, dtype=np.uint64)

    kept: list[Doc] = []
    kept_titles: list[str] = []
    seen_hash: set[str] = set()
    exact = 0
    near = 0
    examples: list[dict[str, Any]] = []
    band_tables: list[dict[bytes, list[int]]] = [dict() for _ in range(_LSH_BANDS)]
    kept_sigs: list[np.ndarray] = []

    for d in docs:
        if d.content_hash in seen_hash:
            exact += 1
            continue
        sig = _signature(_shingle_hashes(d.text), a, b)
        is_near = False
        match_j = -1
        cand: set[int] = set()
        for bi in range(_LSH_BANDS):
            band = sig[bi * _LSH_ROWS:(bi + 1) * _LSH_ROWS].tobytes()
            cand.update(band_tables[bi].get(band, ()))
        for j in cand:
            est = float(np.mean(sig == kept_sigs[j]))
            if est >= _NEAR_DUP_JACCARD:
                is_near = True
                match_j = j
                break
        if is_near:
            near += 1
            if len(examples) < 5 and match_j >= 0:
                examples.append({"dropped": d.title[:70],
                                 "kept_as": kept_titles[match_j][:70],
                                 "jaccard_est": round(est, 2)})
            continue

        idx = len(kept_sigs)
        kept_sigs.append(sig)
        kept_titles.append(d.title)
        for bi in range(_LSH_BANDS):
            band = sig[bi * _LSH_ROWS:(bi + 1) * _LSH_ROWS].tobytes()
            band_tables[bi].setdefault(band, []).append(idx)
        seen_hash.add(d.content_hash)
        kept.append(d)

    return StageResult(
        kept,
        {
            "docs_in": len(docs),
            "docs_out": len(kept),
            "removed": exact + near,
            "removed_reasons": {"exact_duplicate": exact, "near_duplicate": near},
            "method": f"exact SHA1 + MinHash(k={_MINHASH_K}) / LSH "
                      f"({_LSH_BANDS}x{_LSH_ROWS}) @ Jaccard>={_NEAR_DUP_JACCARD}",
            "examples_removed": examples,
        },
    )


# --------------------------------------------------------------------------- #
# STAGE 5 -- Quality Scoring (Gopher/C4-style heuristics)
# --------------------------------------------------------------------------- #

_MIN_WORDS = 50
_ALPHA_FLOOR = 0.60
_SYMBOL_CEIL = 0.10
_DUP_LINE_CEIL = 0.30


def _quality(text: str) -> tuple[float, dict[str, float], str | None]:
    words = _word_re.findall(text)
    n = len(words)
    lines = [ln for ln in text.split("\n") if ln.strip()]
    letters = len(_letter_re.findall(text))  # letters + Indic combining marks
    symbols = sum(1 for c in text if c in "#{}[]<>|\\^~`")
    mean_wl = (sum(len(w) for w in words) / n) if n else 0.0
    alpha_ratio = letters / max(1, len(text))
    symbol_ratio = symbols / max(1, n)
    dup_line_frac = 1 - (len(set(lines)) / len(lines)) if lines else 0.0

    feats = {
        "words": float(n),
        "mean_word_len": round(mean_wl, 2),
        "alpha_ratio": round(alpha_ratio, 3),
        "symbol_ratio": round(symbol_ratio, 3),
        "dup_line_frac": round(dup_line_frac, 3),
    }
    reason = None
    if n < _MIN_WORDS:
        reason = "too_short"
    elif not (1.5 <= mean_wl <= 14):
        reason = "bad_mean_word_length"
    elif alpha_ratio < _ALPHA_FLOOR:
        reason = "low_alpha_ratio"
    elif symbol_ratio > _SYMBOL_CEIL:
        reason = "high_symbol_ratio"
    elif dup_line_frac > _DUP_LINE_CEIL:
        reason = "repetitive_lines"

    # bounded 0..1 score (higher = better), used later for curriculum ordering.
    score = max(0.0, min(1.0,
        0.5 * min(1.0, n / 300)
        + 0.2 * min(1.0, alpha_ratio)
        + 0.15 * (1 - min(1.0, symbol_ratio / _SYMBOL_CEIL))
        + 0.15 * (1 - min(1.0, dup_line_frac / _DUP_LINE_CEIL))))
    return round(score, 3), feats, reason


def stage_quality(docs: list[Doc]) -> StageResult:
    kept: list[Doc] = []
    reasons: Counter = Counter()
    examples: list[dict[str, Any]] = []
    score_hist = [0] * 10  # deciles
    for d in docs:
        score, feats, reason = _quality(d.text)
        d.quality = score
        if reason is not None:
            reasons[reason] += 1
            if len(examples) < 6:
                examples.append({"title": d.title[:70], "reason": reason,
                                 "words": int(feats["words"]),
                                 "alpha_ratio": feats["alpha_ratio"]})
            continue
        score_hist[min(9, int(score * 10))] += 1
        kept.append(d)
    return StageResult(
        kept,
        {
            "docs_in": len(docs),
            "docs_out": len(kept),
            "removed": sum(reasons.values()),
            "removed_reasons": dict(reasons),
            "examples_removed": examples,
            "score_deciles": score_hist,
            "thresholds": {
                "min_words": _MIN_WORDS, "alpha_floor": _ALPHA_FLOOR,
                "symbol_ceil": _SYMBOL_CEIL, "dup_line_ceil": _DUP_LINE_CEIL,
            },
        },
    )


# --------------------------------------------------------------------------- #
# STAGE 6 -- Safety & Toxicity Filtering (PII redaction + toxicity drop)
# --------------------------------------------------------------------------- #

_PII_PATTERNS = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "phone_in": re.compile(r"(?<!\d)(?:\+91[\-\s]?)?[6-9]\d{9}(?!\d)"),
    "ipv4": re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)"),
    "aadhaar": re.compile(r"(?<!\d)\d{4}\s?\d{4}\s?\d{4}(?!\d)"),
    "credit_card": re.compile(r"(?<!\d)(?:\d[ -]?){15}\d(?!\d)"),
}
# small, deliberately conservative multilingual slur/toxicity list
_TOXIC_TERMS = [
    "fuck", "shit", "bitch", "bastard", "asshole", "motherfucker",
    "chutiya", "chutiye", "madarchod", "behenchod", "bhosdi", "gaandu",
    "randi", "harami", "kutta", "kutti",
]
_TOXIC_RE = re.compile("|".join(re.escape(t) for t in _TOXIC_TERMS), re.IGNORECASE)
_TOXIC_DENSITY_CEIL = 0.002  # toxic tokens / total words


def stage_safety(docs: list[Doc]) -> StageResult:
    kept: list[Doc] = []
    pii_counts: Counter = Counter()
    docs_with_pii = 0
    toxic_removed = 0
    for d in docs:
        hits = 0
        for name, pat in _PII_PATTERNS.items():
            def _sub(_m, _n=name):
                pii_counts[_n] += 1
                return f"[REDACTED_{_n.upper()}]"
            d.text, k = pat.subn(_sub, d.text)
            hits += k
        if hits:
            docs_with_pii += 1

        words = max(1, len(_word_re.findall(d.text)))
        tox = len(_TOXIC_RE.findall(d.text))
        if tox / words > _TOXIC_DENSITY_CEIL:
            toxic_removed += 1
            continue
        kept.append(d)
    return StageResult(
        kept,
        {
            "docs_in": len(docs),
            "docs_out": len(kept),
            "removed": toxic_removed,
            "removed_reasons": {"toxicity_over_threshold": toxic_removed},
            "pii_redactions": dict(pii_counts),
            "docs_with_pii_redacted": docs_with_pii,
        },
    )


# --------------------------------------------------------------------------- #
# STAGE 7 -- Code-Repository Filtering (transferable secret/license gates)
# --------------------------------------------------------------------------- #
# The corpus is prose, not source repositories, so repo/static-analysis gates
# do not apply. The two *transferable* gates -- secret detection and SPDX
# license allow-listing -- still run and are reported honestly.

_SECRET_PATTERNS = {
    "aws_access_key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "private_key_block": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "generic_api_token": re.compile(
        r"(?i)(?:api[_-]?key|secret|token)\s*[:=]\s*['\"][A-Za-z0-9_\-]{24,}['\"]"),
}
_SPDX_ALLOWLIST = {"CC-BY-SA-4.0", "CC-BY-4.0", "MIT", "Apache-2.0", "BSD-3-Clause"}


def stage_code_gate(docs: list[Doc], source_license: str = "CC-BY-SA-4.0") -> StageResult:
    secret_counts: Counter = Counter()
    docs_with_secret = 0
    for d in docs:
        found = False
        for name, pat in _SECRET_PATTERNS.items():
            def _sub(_m, _n=name):
                secret_counts[_n] += 1
                return f"[REDACTED_SECRET_{_n.upper()}]"
            d.text, k = pat.subn(_sub, d.text)
            if k:
                found = True
        if found:
            docs_with_secret += 1
    license_ok = source_license in _SPDX_ALLOWLIST
    return StageResult(
        docs,  # nothing dropped: license is allow-listed, secrets are redacted
        {
            "docs_in": len(docs),
            "docs_out": len(docs),
            "removed": 0,
            "secret_redactions": dict(secret_counts),
            "docs_with_secret_redacted": docs_with_secret,
            "source_license": source_license,
            "license_allow_listed": license_ok,
            "note": "Repo-level & static-analysis code gates are N/A for a prose "
                    "corpus; secret-scanning and SPDX allow-listing still run.",
        },
    )


# --------------------------------------------------------------------------- #
# STAGE 8 -- Training Corpus (rebalance, curriculum order, frozen snapshot)
# --------------------------------------------------------------------------- #

# Target pre-training mix across the three Indic languages (illustrative,
# roughly speaker-weighted). Rebalancing subsamples the over-represented
# language, dropping its lowest-quality documents first.
_TARGET_MIX = {"hi": 0.50, "mr": 0.25, "te": 0.25}


def stage_corpus(docs: list[Doc]) -> StageResult:
    by_lang: dict[str, list[Doc]] = {}
    for d in docs:
        by_lang.setdefault(d.src_lang, []).append(d)

    total = len(docs)
    # Largest feasible corpus that still hits the target ratios given supply.
    scale = min(len(by_lang.get(l, [])) / share
                for l, share in _TARGET_MIX.items() if l in by_lang)
    rebalanced: list[Doc] = []
    removed_rebalance = 0
    per_lang_final: Counter = Counter()
    for lang, share in _TARGET_MIX.items():
        pool = by_lang.get(lang, [])
        keep_n = int(scale * share)
        pool_sorted = sorted(pool, key=lambda d: d.quality, reverse=True)
        rebalanced.extend(pool_sorted[:keep_n])
        removed_rebalance += len(pool) - keep_n
        per_lang_final[lang] = keep_n

    # Curriculum ordering: easy/low-quality-tail first, best material annealed
    # to the end of the schedule.
    rebalanced.sort(key=lambda d: d.quality)

    corpus_text = "\n".join(d.text for d in rebalanced)
    words = len(_word_re.findall(corpus_text))
    chars = len(corpus_text)
    est_tokens = int(chars / 4)  # rough bf16-tokenizer estimate, clearly labelled
    snapshot_hash = hashlib.sha256(corpus_text.encode("utf-8")).hexdigest()[:16]

    return StageResult(
        rebalanced,
        {
            "docs_in": total,
            "docs_out": len(rebalanced),
            "removed": removed_rebalance,
            "removed_reasons": {"rebalance_subsample": removed_rebalance},
            "target_mix": _TARGET_MIX,
            "per_language_final_docs": dict(per_lang_final),
            "final_words": words,
            "final_chars": chars,
            "estimated_tokens": est_tokens,
            "snapshot_sha256_16": snapshot_hash,
        },
    )
