from pathlib import Path
from src.tokenizer import FrozenTokenizer
from src.shards import build_demo_documents, create_shards, validate_manifests

def test_manifest_hashes(tmp_path):
    tok = FrozenTokenizer()
    out = tmp_path / "shards"
    create_shards(build_demo_documents(), tok, out)
    ms = validate_manifests(out, tok)
    assert ms
    assert all(m["tokenizer_hash"] == tok.tokenizer_hash for m in ms)
