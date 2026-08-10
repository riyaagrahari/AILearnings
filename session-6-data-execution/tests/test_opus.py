from src.mixture import opus_decide

def test_all_opus_outcomes():
    assert opus_decide({"quality":.9}, .2, .1)[0] == "ACCEPT"
    assert opus_decide({"quality":.1}, .2, .1)[0] == "REJECT"
    assert opus_decide({"quality":.5}, .2, .1)[0] == "DEFER"
    assert opus_decide({"quality":.9}, .01, .1)[0] == "FLOOR_OVERRIDE"
