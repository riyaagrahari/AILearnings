from pathlib import Path
from .utils import sha256_obj, write_json, read_json

def build_demo_documents():
    rows = []
    examples = {
        "general": [
            "India has a diverse geography and a long history.",
            "Distributed systems require deterministic coordination.",
            "Good engineering makes assumptions explicit and testable.",
        ],
        "coding": [
            "def add(a, b): return a + b",
            "Use a hash to identify immutable content.",
            "A retry must not duplicate a committed batch.",
        ],
        "reasoning": [
            "If A implies B and A is true, B follows.",
            "Compare evidence before selecting a conclusion.",
        ],
        "agentic": [
            "An agent observes state, chooses an action, and verifies the result.",
            "Tool use should record inputs, outputs, and provenance.",
        ],
        "indic": [
            "भारत विविध भाषाओं और संस्कृतियों का देश है।",
            "తెలుగు భారతదేశంలోని ప్రధాన భాషలలో ఒకటి.",
        ],
        "math": [
            "A triangle has three sides and the angles sum to 180 degrees.",
            "If x is 4 and y is 5, x plus y equals 9.",
        ],
        "science": [
            "Water freezes near zero degrees Celsius at standard pressure.",
            "Plants convert light energy into chemical energy through photosynthesis.",
        ],
    }
    i = 0
    for lane, texts in examples.items():
        for j, text in enumerate(texts):
            rows.append({
                "document_id": f"{lane}-{j:03d}",
                "lane": lane,
                "language": "hi" if lane == "indic" and j == 0 else ("te" if lane == "indic" else "en"),
                "split": "train",
                "text": text,
            })
            i += 1
    rows.append({
        "document_id": "eval-000",
        "lane": "general",
        "language": "en",
        "split": "eval",
        "text": "Distributed systems require deterministic coordination.",
    })
    return rows

def create_shards(documents, tokenizer, out_dir: Path, shard_size=3):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Shards are content-addressed outputs: an existing shard cannot be silently overwritten.
    train = [d for d in documents if d["split"] == "train"]
    evals = [d for d in documents if d["split"] == "eval"]
    manifests = []
    for group_name, docs in (("train", train), ("eval", evals)):
        for start in range(0, len(docs), shard_size):
            chunk = docs[start:start+shard_size]
            shard_id = f"{group_name}-{start//shard_size:03d}"
            records = []
            for d in chunk:
                tokens = tokenizer.encode(d["text"])
                records.append({
                    "document_id": d["document_id"],
                    "lane": d["lane"],
                    "language": d["language"],
                    "split": d["split"],
                    "tokens": tokens,
                    "token_count": len(tokens),
                })
            content_hash = sha256_obj(records)
            manifest = {
                "schema_version": 1,
                "shard_id": shard_id,
                "split": group_name,
                "document_count": len(records),
                "token_count": sum(r["token_count"] for r in records),
                "tokenizer_hash": tokenizer.tokenizer_hash,
                "content_hash": content_hash,
                "document_ids": [r["document_id"] for r in records],
            }
            shard_path = out_dir / f"{shard_id}.json"
            manifest_path = out_dir / f"{shard_id}.manifest.json"
            if shard_path.exists() or manifest_path.exists():
                existing = read_json(manifest_path)
                if existing["content_hash"] != content_hash:
                    raise RuntimeError(f"immutable shard conflict: {shard_id}")
            else:
                write_json(shard_path, records)
                write_json(manifest_path, manifest)
            manifests.append(manifest)
    return manifests

def validate_manifests(out_dir, tokenizer):
    manifests = []
    for p in sorted(out_dir.glob("*.manifest.json")):
        m = read_json(p)
        records = read_json(out_dir / f"{m['shard_id']}.json")
        assert m["tokenizer_hash"] == tokenizer.tokenizer_hash
        assert m["content_hash"] == sha256_obj(records)
        assert m["token_count"] == sum(r["token_count"] for r in records)
        manifests.append(m)
    return manifests
