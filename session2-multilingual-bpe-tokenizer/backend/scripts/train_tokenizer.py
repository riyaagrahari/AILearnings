#!/usr/bin/env python3
"""CLI: train the shared multilingual BPE tokenizer from real corpus files.

This script reads raw text the user provides -- it never downloads or
generates any data. Expected layout (see ``data/README.md`` for the full
contract):

    data/raw/en/*.txt
    data/raw/hi/*.txt
    data/raw/te/*.txt
    data/raw/ta/*.txt

Each ``*.txt`` file is UTF-8 (a leading BOM is tolerated) plain text or
Wikipedia-style markup; every file in every language folder is read and
cleaned via ``bpe.preprocess.clean_text``.

Usage
-----
    python scripts/train_tokenizer.py
    python scripts/train_tokenizer.py --vocab-size 10000 --alpha 0.5
    python scripts/train_tokenizer.py --data-dir /path/to/data/raw --languages en,hi

Outputs (written to --output-dir, default backend/artifacts/):
    vocab.json             -- see bpe.trainer.save_vocab for the schema
    merges.json            -- see bpe.trainer.save_merges for the schema
    tokenizer_config.json  -- training configuration + per-language corpus stats
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND_DIR))

from bpe.corpus import (  # noqa: E402
    DEFAULT_LANGUAGES,
    build_multilingual_word_frequencies,
    load_raw_corpus,
)
from bpe.trainer import save_merges, save_vocab, train_bpe  # noqa: E402

PROJECT_ROOT = _BACKEND_DIR.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_DIR = _BACKEND_DIR / "artifacts"
DATA_README = PROJECT_ROOT / "data" / "README.md"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Directory containing <lang>/*.txt subfolders (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--languages",
        type=str,
        default=",".join(DEFAULT_LANGUAGES),
        help=f"Comma-separated language codes / subfolder names (default: {','.join(DEFAULT_LANGUAGES)})",
    )
    parser.add_argument("--vocab-size", type=int, default=10_000, help="Target vocabulary size")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Exponential-smoothing exponent for language weighting: "
        "1=no correction, 0=fully equalize languages, 0.3-0.5 recommended (default: 0.5)",
    )
    parser.add_argument(
        "--weight-boost",
        type=str,
        default="",
        help="Optional comma-separated per-language weight multipliers applied on top "
        "of --alpha, e.g. 'en=3.0,te=1.5'. Dedicates more of the shared vocab budget to "
        "a language (raising others' fertility). Default: none (all 1.0).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Where to write vocab.json/merges.json/tokenizer_config.json (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args(argv)


def parse_weight_boosts(spec: str) -> dict[str, float]:
    """Parse ``'en=3.0,te=1.5'`` into ``{'en': 3.0, 'te': 1.5}``."""
    boosts: dict[str, float] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        lang, _, value = item.partition("=")
        boosts[lang.strip()] = float(value)
    return boosts


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    languages = [lang.strip() for lang in args.languages.split(",") if lang.strip()]
    language_boosts = parse_weight_boosts(args.weight_boost)

    print(f"Reading raw corpus from {args.data_dir} for languages: {', '.join(languages)}")
    texts_by_language = load_raw_corpus(args.data_dir, languages=languages)

    empty_languages = [lang for lang, texts in texts_by_language.items() if not texts]
    if empty_languages:
        print(
            f"WARNING: no usable .txt content found for: {', '.join(empty_languages)}. "
            f"See {DATA_README} for the expected folder structure.",
            file=sys.stderr,
        )
    if len(empty_languages) == len(languages):
        print(
            "ERROR: no training data found for any language. This script does not "
            f"download or generate data -- populate {args.data_dir}/<lang>/*.txt with "
            f"real text before training (see {DATA_README}). Nothing was written.",
            file=sys.stderr,
        )
        return 1

    print("Building weighted, combined word-frequency table...")
    if language_boosts:
        print(f"  applying per-language weight boosts: {language_boosts}")
    word_freqs, stats = build_multilingual_word_frequencies(
        texts_by_language, alpha=args.alpha, language_boosts=language_boosts
    )

    print(f"Training BPE (target vocab size = {args.vocab_size})...")
    start = time.time()
    result = train_bpe(word_freqs, vocab_size=args.vocab_size)
    elapsed = time.time() - start

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_vocab(result, args.output_dir / "vocab.json")
    save_merges(result, args.output_dir / "merges.json")

    config = {
        "target_vocab_size": args.vocab_size,
        "achieved_vocab_size": result.vocab_size,
        "languages": languages,
        "alpha": args.alpha,
        "language_boosts": language_boosts,
        "num_merges": len(result.merges),
        "training_seconds": round(elapsed, 2),
        "language_stats": {
            lang: {
                "num_documents": s.num_documents,
                "raw_word_count": s.raw_word_count,
                "weight": round(s.weight, 4),
            }
            for lang, s in stats.items()
        },
    }
    (args.output_dir / "tokenizer_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\nDone in {elapsed:.2f}s. Wrote vocab.json, merges.json, tokenizer_config.json to {args.output_dir}")
    print(f"  achieved vocab size : {result.vocab_size} / {args.vocab_size} target")
    print(f"  merges learned       : {len(result.merges)}")
    print("  per-language corpus stats:")
    for lang, s in stats.items():
        print(
            f"    {lang}: documents={s.num_documents:>5}  raw_words={s.raw_word_count:>9}  "
            f"weight={s.weight:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
