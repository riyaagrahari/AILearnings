import pytest
from src.firewall import consume_shard

def test_eval_blocked():
    with pytest.raises(PermissionError):
        consume_shard({"shard_id":"eval-0","split":"eval"})
