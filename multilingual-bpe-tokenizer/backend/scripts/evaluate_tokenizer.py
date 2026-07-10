#!/usr/bin/env python3
"""CLI: evaluate a trained tokenizer against real held-out text.

Expected layout (mirrors data/raw/, see data/README.md):

    data/eval/en/*.txt
    data/eval/hi/*.txt
    data/eval/te/*.txt
    data/eval/ta/*.txt

This script never downloads or generates evaluation data either -- it
only reads whatever ``*.txt`` files exist under ``--eval-dir``.

Usage
-----
    python scripts/train_tokenizer.py            # first, to produce artifacts
    python scripts/evaluate_tokenizer.py          # then, to evaluate them

Outputs eval_report.json (see bpe.evaluation for the schema) into
--artifacts-dir, and prints a human-readable summary table, including
whether the assignment's English fertility <= 1.2 target is met.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_BACKEND_DIR))

from bpe.corpus import DEFAULT_LANGUAGES, load_raw_corpus  # noqa: E402
from bpe.evaluation import evaluate_tokenizer, save_report  # noqa: E402
from bpe.tokenizer import BPETokenizer  # noqa: E402

PROJECT_ROOT = _BACKEND_DIR.parent
DEFAULT_ARTIFACTS_DIR = _BACKEND_DIR / "artifacts"
DEFAULT_EVAL_DIR = PROJECT_ROOT / "data" / "eval"
DATA_README = PROJECT_ROOT / "data" / "README.md"

ENGLISH_FERTILITY_TARGET = 1.2


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=DEFAULT_ARTIFACTS_DIR,
        help=f"Directory containing vocab.json/merges.json (default: {DEFAULT_ARTIFACTS_DIR})",
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=DEFAULT_EVAL_DIR,
        help=f"Directory containing <lang>/*.txt held-out text (default: {DEFAULT_EVAL_DIR})",
    )
    parser.add_argument(
        "--languages",
        type=str,
        default=",".join(DEFAULT_LANGUAGES),
        help=f"Comma-separated language codes / subfolder names (default: {','.join(DEFAULT_LANGUAGES)})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write eval_report.json (default: <artifacts-dir>/eval_report.json)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    languages = [lang.strip() for lang in args.languages.split(",") if lang.strip()]
    output_path = args.output or (args.artifacts_dir / "eval_report.json")

    vocab_path = args.artifacts_dir / "vocab.json"
    merges_path = args.artifacts_dir / "merges.json"
    if not vocab_path.exists() or not merges_path.exists():
        print(
            f"ERROR: {vocab_path} / {merges_path} not found. Run "
            "scripts/train_tokenizer.py first.",
            file=sys.stderr,
        )
        return 1

    print(f"Loading tokenizer from {args.artifacts_dir}")
    tokenizer = BPETokenizer.from_files(vocab_path, merges_path)

    print(f"Reading held-out corpus from {args.eval_dir} for languages: {', '.join(languages)}")
    texts_by_language = load_raw_corpus(args.eval_dir, languages=languages)

    empty_languages = [lang for lang, texts in texts_by_language.items() if not texts]
    if empty_languages:
        print(
            f"WARNING: no usable .txt content found for: {', '.join(empty_languages)}. "
            f"See {DATA_README} for the expected folder structure. These languages will "
            "show as all-zero in the report.",
            file=sys.stderr,
        )
    if len(empty_languages) == len(languages):
        print(
            "ERROR: no evaluation data found for any language. This script does not "
            f"download or generate data -- populate {args.eval_dir}/<lang>/*.txt with "
            f"real held-out text before evaluating (see {DATA_README}). Nothing was written.",
            file=sys.stderr,
        )
        return 1

    report = evaluate_tokenizer(tokenizer, texts_by_language)

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    save_report(report, output_path)

    print(f"\nWrote {output_path}\n")
    print(f"{'language':<10} {'docs':>6} {'words':>8} {'fertility':>10} {'unk_rate':>9} "
          f"{'compression':>12} {'roundtrip':>10} {'vocab_use':>10}")
    for lang, r in report.per_language.items():
        print(
            f"{lang:<10} {r.num_documents:>6} {r.num_words:>8} {r.fertility:>10.3f} "
            f"{r.unk_rate:>9.3%} {r.compression_ratio:>12.3f} "
            f"{'OK' if r.roundtrip_ok else 'FAIL':>10} {r.vocab_utilization:>10.3%}"
        )

    print(f"\nfertility_balance_score (max-min across languages): {report.fertility_balance_score:.3f}")

    english_report = report.per_language.get("en")
    if english_report is not None and english_report.num_words > 0:
        met = english_report.fertility <= ENGLISH_FERTILITY_TARGET
        status = "MET" if met else "NOT MET"
        print(
            f"English fertility target (<= {ENGLISH_FERTILITY_TARGET}): "
            f"{english_report.fertility:.3f} -- {status}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
