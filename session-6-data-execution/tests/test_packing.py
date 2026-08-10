from src.packing import pack_sample, validate_packed

def test_masks_and_positions():
    p = pack_sample({"document_id":"x","tokens":[1,10,11,2]}, 8)
    assert validate_packed(p)
    assert len(p["input_ids"]) == 7
    assert p["position_ids"] == list(range(7))
    assert p["loss_mask"][2] == 0
