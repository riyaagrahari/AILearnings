from src.recovery import compare_batch

def test_resume_rejects_wrong_hash():
    assert not compare_batch(
        {"batch_id":"b1","batch_hash":"expected"},
        {"batch_id":"b1","batch_hash":"wrong"},
    )

def test_resume_rejects_wrong_id():
    assert not compare_batch(
        {"batch_id":"b1","batch_hash":"h"},
        {"batch_id":"b2","batch_hash":"h"},
    )
