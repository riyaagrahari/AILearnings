"""Loads real tokenizer artifacts + real evaluation corpus and runs the
actual tokenizer to answer every API request. No statistic is computed,
cached, or approximated independently of the tokenizer -- this module is
a thin orchestration layer over ``bpe.tokenizer``, ``bpe.corpus`` and
``bpe.evaluation``.

Paths (overridable via environment variables, for flexible deployment):

    BPE_ARTIFACTS_DIR  -- directory with vocab.json / merges.json / tokenizer_config.json
                          (default: backend/artifacts, written by scripts/train_tokenizer.py)
    BPE_EVAL_DIR        -- directory with <lang>/*.txt held-out text
                          (default: data/eval, see data/README.md)
    BPE_LANGUAGES       -- comma-separated language codes (default: en,hi,te,ta)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from bpe.corpus import DEFAULT_LANGUAGES, load_raw_corpus
from bpe.evaluation import AssignmentRatioReport, compute_assignment_ratios
from bpe.tokenizer import BPETokenizer

_API_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _API_DIR.parent
_PROJECT_ROOT = _BACKEND_DIR.parent

ARTIFACTS_DIR = Path(os.environ.get("BPE_ARTIFACTS_DIR", _BACKEND_DIR / "artifacts"))
EVAL_DIR = Path(os.environ.get("BPE_EVAL_DIR", _PROJECT_ROOT / "data" / "eval"))
LANGUAGES: list[str] = [
    lang.strip()
    for lang in os.environ.get("BPE_LANGUAGES", ",".join(DEFAULT_LANGUAGES)).split(",")
    if lang.strip()
]

VOCAB_PATH = ARTIFACTS_DIR / "vocab.json"
MERGES_PATH = ARTIFACTS_DIR / "merges.json"
CONFIG_PATH = ARTIFACTS_DIR / "tokenizer_config.json"

LANGUAGE_DISPLAY_NAMES: dict[str, str] = {
    "en": "English",
    "hi": "Hindi",
    "te": "Telugu",
    "ta": "Tamil",
}


def display_name(language_code: str) -> str:
    return LANGUAGE_DISPLAY_NAMES.get(language_code, language_code)


class ArtifactsNotFoundError(RuntimeError):
    """Raised when vocab.json/merges.json don't exist yet.

    Means: the tokenizer has not been trained. The caller (main.py)
    turns this into a 503 with an actionable message -- never a
    fabricated response.
    """


class NoEvaluationDataError(RuntimeError):
    """Raised when no language under BPE_EVAL_DIR has any usable text."""


# ---------------------------------------------------------------------------
# Tokenizer loading, cached and invalidated by artifact mtimes so a fresh
# `train_tokenizer.py` run is picked up without restarting the server.
# ---------------------------------------------------------------------------

_tokenizer_cache: tuple[float, float, BPETokenizer] | None = None


def _require_artifacts() -> None:
    if not VOCAB_PATH.exists() or not MERGES_PATH.exists():
        raise ArtifactsNotFoundError(
            f"No trained tokenizer found at {ARTIFACTS_DIR}. Run "
            "'python scripts/train_tokenizer.py' first (see data/README.md "
            "for the expected data/raw/<lang>/*.txt layout)."
        )


def get_tokenizer() -> BPETokenizer:
    """Return the trained tokenizer, reloading if the artifact files on
    disk have changed since the last call (cheap mtime check).
    """
    global _tokenizer_cache
    _require_artifacts()

    vocab_mtime = VOCAB_PATH.stat().st_mtime
    merges_mtime = MERGES_PATH.stat().st_mtime
    if _tokenizer_cache is not None:
        cached_vocab_mtime, cached_merges_mtime, cached_tokenizer = _tokenizer_cache
        if cached_vocab_mtime == vocab_mtime and cached_merges_mtime == merges_mtime:
            return cached_tokenizer

    tokenizer = BPETokenizer.from_files(VOCAB_PATH, MERGES_PATH)
    _tokenizer_cache = (vocab_mtime, merges_mtime, tokenizer)
    return tokenizer


def get_tokenizer_config() -> dict:
    """Training configuration written by scripts/train_tokenizer.py, if
    it exists -- purely informational (e.g. for display), never used to
    compute a statistic.
    """
    if not CONFIG_PATH.exists():
        return {}
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def get_evaluation_texts() -> dict[str, list[str]]:
    """Real held-out text per language from BPE_EVAL_DIR. Re-read on
    every call (cheap for text files of this scale) so edits to
    data/eval/ are picked up immediately.
    """
    return load_raw_corpus(EVAL_DIR, languages=LANGUAGES)


# ---------------------------------------------------------------------------
# Statistics (GET /api/statistics)
# ---------------------------------------------------------------------------


def compute_statistics() -> AssignmentRatioReport:
    """Run the real tokenizer against the real evaluation corpus and
    compute the assignment's ratio/score metrics -- see
    bpe.evaluation.compute_assignment_ratios for the formula.
    """
    tokenizer = get_tokenizer()
    texts_by_language = get_evaluation_texts()
    if all(not texts for texts in texts_by_language.values()):
        raise NoEvaluationDataError(
            f"No evaluation text found under {EVAL_DIR}. Populate "
            "data/eval/<lang>/*.txt with real held-out text (see "
            "data/README.md) before requesting statistics."
        )
    return compute_assignment_ratios(tokenizer, texts_by_language)


# ---------------------------------------------------------------------------
# Playground (POST /api/tokenize)
# ---------------------------------------------------------------------------


def tokenize_text(text: str) -> dict:
    tokenizer = get_tokenizer()
    pretokens = tokenizer.pretokenize(text)
    tokens = tokenizer.encode_as_tokens(text)
    ids = tokenizer.encode(text)
    decoded_text = tokenizer.decode(ids)
    return {
        "pretokens": pretokens,
        "tokens": tokens,
        "ids": ids,
        "decoded_text": decoded_text,
    }


# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------


def build_combined_tokenizer_json() -> str:
    """Build a single-file ``tokenizer.json`` (vocab + merges + training
    config in one document) purely for convenience -- vocab.json and
    merges.json remain the authoritative artifacts trainer.py writes;
    this is assembled on read, never stored as a separate source of
    truth.
    """
    _require_artifacts()
    vocab_data = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
    merges_data = json.loads(MERGES_PATH.read_text(encoding="utf-8"))
    config_data = get_tokenizer_config()
    combined = {"vocab": vocab_data, "merges": merges_data, "config": config_data}
    return json.dumps(combined, ensure_ascii=False, indent=2)
