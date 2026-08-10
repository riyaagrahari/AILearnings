from src.tokenizer import FrozenTokenizer
from src.shards import build_demo_documents, create_shards

def test_same_shards_are_idempotent(tmp_path):
    out=tmp_path/'shards'; tok=FrozenTokenizer()
    ms1=create_shards(build_demo_documents(),tok,out)
    ms2=create_shards(build_demo_documents(),tok,out)
    assert ms1 == ms2
