def opus_decide(candidate, current_share, floor):
    quality = candidate.get("quality", 0)
    if quality < 0.40:
        return "REJECT", "quality_below_threshold"
    if quality < 0.60:
        return "DEFER", "quality_needs_review"
    if current_share < floor:
        return "FLOOR_OVERRIDE", "protected_floor"
    return "ACCEPT", "quality_pass"
