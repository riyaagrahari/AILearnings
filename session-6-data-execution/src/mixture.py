from dataclasses import dataclass
from .utils import sha256_obj

@dataclass(frozen=True)
class Stage:
    name: str
    weights: dict
    floors: dict

STAGES = [
    Stage("foundation", {
        "general": .35, "coding": .18, "reasoning": .12, "agentic": .08,
        "indic": .12, "math": .08, "science": .07
    }, {"coding": .15, "indic": .10}),
    Stage("coding_stem", {
        "general": .25, "coding": .30, "reasoning": .12, "agentic": .10,
        "indic": .10, "math": .08, "science": .05
    }, {"coding": .20, "indic": .08}),
]

def compile_schedule():
    for s in STAGES:
        assert abs(sum(s.weights.values()) - 1.0) < 1e-9
        for lane, floor in s.floors.items():
            assert s.weights[lane] >= floor
    return STAGES

def choose_lane(step, stage, lanes):
    # Deterministic weighted round-robin using the cumulative deficit.
    total = 1000
    targets = {k: int(v * total) for k, v in stage.weights.items()}
    counts = {k: 0 for k in lanes}
    for i in range(step + 1):
        lane = max(lanes, key=lambda k: targets.get(k, 0) * (i + 1) - counts[k] * total)
        counts[lane] += 1
    return lane

def planned_share(stage):
    return sha256_obj({"stage": stage.name, "weights": stage.weights, "floors": stage.floors})


def opus_decide(candidate, current_share, floor):
    if candidate.get("quality", 0) < 0.40:
        return "REJECT", "quality_below_threshold"
    if candidate.get("quality", 0) < 0.60:
        return "DEFER", "quality_needs_review"
    if current_share < floor:
        return "FLOOR_OVERRIDE", "protected_floor"
    return "ACCEPT", "quality_pass"
