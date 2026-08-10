from src.mixture import compile_schedule

def test_weights_and_floors():
    stages = compile_schedule()
    for s in stages:
        assert abs(sum(s.weights.values()) - 1) < 1e-9
        for lane, floor in s.floors.items():
            assert s.weights[lane] >= floor
