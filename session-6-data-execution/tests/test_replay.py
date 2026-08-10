from src.recovery import replay

def test_replay_preserves_identity():
    batches = [
        {"batch_id":"b0","batch_hash":"h0","samples":[{"source":{"sample_id":"x","token_start":0,"token_end":4}}]},
        {"batch_id":"b1","batch_hash":"h1","samples":[{"source":{"sample_id":"y","token_start":0,"token_end":5}}]},
    ]
    assert replay(batches, 0, 2) == [
        {"batch_id":"b0","batch_hash":"h0","spans":[{"sample_id":"x","token_start":0,"token_end":4}]},
        {"batch_id":"b1","batch_hash":"h1","spans":[{"sample_id":"y","token_start":0,"token_end":5}]},
    ]
