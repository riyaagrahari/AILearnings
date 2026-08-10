from src.tokenizer import FrozenTokenizer
from src.shards import build_demo_documents, create_shards, validate_manifests
from src.utils import read_json

def test_content_hash_detects_token_change(tmp_path):
    tok = FrozenTokenizer()
    out = tmp_path / "s"
    create_shards(build_demo_documents(), tok, out)
    m = validate_manifests(out, tok)[0]
    p = out / f"{m['shard_id']}.json"
    records = read_json(p)
    records[0]["tokens"][0] += 1
    from src.utils import sha256_obj
    assert sha256_obj(records) != m["content_hash"]
