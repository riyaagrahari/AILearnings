from .packing import batch_hash, pack_sample, validate_packed
from .utils import sha256_obj


def build_batches(samples, seq_len, batch_size, policy="standard"):
    batches = []
    for i in range(0, len(samples), batch_size):
        packed = [pack_sample(s, seq_len, policy=policy) for s in samples[i:i+batch_size]]
        for p in packed:
            validate_packed(p)
        base = {"sequence_count": len(packed), "samples": packed, "packing_policy": policy}
        h = batch_hash(base)
        batches.append({"batch_id": f"batch-{len(batches):04d}", "batch_hash": h, **base})
    return batches


def compare_batch(expected, actual):
    return expected["batch_id"] == actual["batch_id"] and expected["batch_hash"] == actual["batch_hash"]


def replay(batches, start, end):
    return [{
        "batch_id": b["batch_id"],
        "batch_hash": b["batch_hash"],
        "spans": [p.get("source", {}) for p in b["samples"]],
    } for b in batches[start:end]]


def stream_hash(batches):
    return sha256_obj([(b["batch_id"], b["batch_hash"]) for b in batches])
