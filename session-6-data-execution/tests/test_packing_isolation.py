from src.packing import pack_sequence, validate_packed_sequence

def test_block_diagonal_attention_and_local_positions():
    samples = [
        {"document_id":"a","tokens":[1,10,11,2]},
        {"document_id":"b","tokens":[1,20,21,2]},
    ]
    p = pack_sequence(samples, 16)
    assert validate_packed_sequence(p)
    assert p["position_ids"][:3] == [0,1,2]
    assert p["position_ids"][3:6] == [0,1,2]
    # Last token of A cannot attend to B.
    assert p["attention_mask"][2][3] == 0
    # B can attend causally within B.
    assert p["attention_mask"][5][4] == 1
