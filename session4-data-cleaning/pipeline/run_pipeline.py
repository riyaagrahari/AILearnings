"""Run the eight Session-3 cleaning strategies over the real Wikipedia corpus.

Reads a bounded character budget out of the downloaded parquet shards (so the
working set sits comfortably in the assignment's 10-100M-character range),
streams every document through the eight stages in ``strategies.py``, and writes
``artifacts/stats.json`` -- the single source of truth the Session-4 widget
renders. Nothing is fabricated: every number comes from actually processing the
text.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from strategies import (
    Doc,
    stage_raw,
    stage_langid,
    stage_normalize,
    stage_dedup,
    stage_quality,
    stage_safety,
    stage_code_gate,
    stage_corpus,
)

HERE = Path(__file__).parent
RAW_DIR = HERE / "data" / "raw"
ART_DIR = HERE / "artifacts"

LANGUAGES = ["hi", "te", "mr"]
CHAR_BUDGET_PER_LANG = 20_000_000  # 3 x 20M = ~60M chars, inside 10-100M

LANG_NAMES = {"hi": "Hindi", "te": "Telugu", "mr": "Marathi"}


def load_docs() -> list[Doc]:
    docs: list[Doc] = []
    for lang in LANGUAGES:
        path = RAW_DIR / f"{lang}.parquet"
        if not path.exists():
            raise SystemExit(f"missing {path} -- run `python3 download.py` first")
        budget = CHAR_BUDGET_PER_LANG
        got = 0
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=512, columns=["id", "url", "title", "text"]):
            for row in batch.to_pylist():
                text = row["text"] or ""
                docs.append(Doc(id=str(row["id"]), url=row["url"] or "",
                                title=row["title"] or "", text=text, src_lang=lang))
                got += len(text)
            if got >= budget:
                break
        print(f"[{lang}] loaded {got/1e6:.1f}M chars")
    return docs


STRATEGY_META = [
    ("raw", "Raw Documents", "Ingest", False),
    ("langid", "Language Identification", "Detect", True),
    ("normalize", "Unicode Normalization", "Normalize", True),
    ("dedup", "Deduplication", "Dedup", True),
    ("quality", "Quality Scoring", "Score", True),
    ("safety", "Safety & Toxicity Filtering", "Filter", True),
    ("code", "Code-Repository Filtering", "Code gate", True),
    ("corpus", "Training Corpus", "Ship", False),
]


def per_language(docs: list[Doc]) -> dict[str, int]:
    c: Counter = Counter()
    for d in docs:
        c[d.src_lang] += 1
    return {l: c.get(l, 0) for l in LANGUAGES}


def main() -> int:
    t0 = time.time()
    ART_DIR.mkdir(parents=True, exist_ok=True)

    docs = load_docs()
    raw_count = len(docs)
    raw_chars = sum(len(d.text) for d in docs)
    lang_before = per_language(docs)
    print(f"loaded {raw_count:,} docs / {raw_chars/1e6:.1f}M chars")

    runners = {
        "raw": stage_raw, "langid": stage_langid, "normalize": stage_normalize,
        "dedup": stage_dedup, "quality": stage_quality, "safety": stage_safety,
        "code": stage_code_gate, "corpus": stage_corpus,
    }

    stages_out = []
    for sid, title, short, is_cleanup in STRATEGY_META:
        st = time.time()
        result = runners[sid](docs)
        docs = result.docs
        dt = time.time() - st
        stats = result.stats
        stats.update({
            "id": sid, "title": title, "short": short,
            "is_active_cleanup": is_cleanup, "seconds": round(dt, 2),
        })
        stages_out.append(stats)
        print(f"[{sid:9}] {stats['docs_in']:>7,} -> {stats['docs_out']:>7,} "
              f"(removed {stats.get('removed', 0):>6,})  {dt:5.1f}s")

    final_docs = docs
    final_chars = sum(len(d.text) for d in final_docs)
    lang_after = per_language(final_docs)

    report = {
        "meta": {
            "session": 4,
            "title": "India-First Multilingual Corpus Cleaning",
            "dataset": "wikimedia/wikipedia (20231101 dumps)",
            "dataset_url": "https://huggingface.co/datasets/wikimedia/wikipedia",
            "license": "CC BY-SA 4.0",
            "languages": [{"code": l, "name": LANG_NAMES[l]} for l in LANGUAGES],
            "char_budget_per_language": CHAR_BUDGET_PER_LANG,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "runtime_seconds": round(time.time() - t0, 1),
        },
        "strategy_count": {
            "total_stages": len(STRATEGY_META),
            "active_cleanups": sum(1 for *_ , c in STRATEGY_META if c),
        },
        "totals": {
            "raw_documents": raw_count,
            "raw_characters": raw_chars,
            "final_documents": len(final_docs),
            "final_characters": final_chars,
            "documents_removed": raw_count - len(final_docs),
            "document_retention_pct": round(100 * len(final_docs) / max(1, raw_count), 2),
            "character_retention_pct": round(100 * final_chars / max(1, raw_chars), 2),
        },
        "per_language": {
            "before": lang_before,
            "after": lang_after,
            "names": LANG_NAMES,
        },
        "stages": stages_out,
    }

    payload = json.dumps(report, ensure_ascii=False, indent=2)
    out = ART_DIR / "stats.json"
    out.write_text(payload, encoding="utf-8")
    print(f"\nwrote {out}  ({out.stat().st_size/1024:.0f} KB)")

    # keep the widget's data source in sync with the canonical run output.
    widget_copy = HERE.parent / "src" / "data" / "stats.json"
    if widget_copy.parent.exists():
        widget_copy.write_text(payload, encoding="utf-8")
        print(f"synced  {widget_copy}")

    print(f"done in {report['meta']['runtime_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
