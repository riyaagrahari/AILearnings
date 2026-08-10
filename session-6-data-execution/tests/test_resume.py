from src.recovery import compare_batch

def test_resume_exact_match():
    assert compare_batch(
        {"batch_id":"b1","batch_hash":"h1"},
        {"batch_id":"b1","batch_hash":"h1"}
    )
    assert not compare_batch(
        {"batch_id":"b1","batch_hash":"h1"},
        {"batch_id":"b2","batch_hash":"h1"}
    )
