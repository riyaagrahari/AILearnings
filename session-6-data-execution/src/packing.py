from .utils import sha256_obj

def pack_sample(sample, seq_len, policy="standard"):
    if policy not in ("standard", "packed"):
        raise ValueError(f"unknown packing policy: {policy}")
    toks = sample["tokens"][:seq_len]
    input_ids = toks[:-1]
    labels = toks[1:]
    loss_mask = [1] * len(labels)
    if labels and labels[-1] == 2:
        loss_mask[-1] = 0
    pad = seq_len - 1 - len(input_ids)
    input_ids += [0] * pad
    labels += [-100] * pad
    loss_mask += [0] * pad
    attention_mask = [1 if i < len(toks)-1 else 0 for i in range(seq_len-1)]
    position_ids = list(range(seq_len-1))
    return {
        "input_ids": input_ids,
        "labels": labels,
        "loss_mask": loss_mask,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "source": {
            "sample_id": sample["document_id"],
            "lane": sample.get("lane"),
            "token_start": 0,
            "token_end": len(toks),
        }
    }

def pack_sequence(samples, seq_len):
    """Pack multiple samples with block-diagonal causal attention isolation."""
    flat = []
    sources = []
    for sample in samples:
        toks = sample["tokens"]
        if len(flat) + len(toks) > seq_len:
            break
        start = len(flat)
        flat.extend(toks)
        sources.append((start, len(flat), sample["document_id"]))

    flat = flat[:seq_len]
    n = len(flat)
    input_ids = flat[:-1]
    labels = flat[1:]
    loss_mask = [1] * (n - 1)
    position_ids = []
    segment_ids = [None] * (n - 1)

    for start, end, _sid in sources:
        for pos in range(start, max(start, end - 1)):
            if pos < n - 1:
                position_ids.append(pos - start)
                segment_ids[pos] = _sid

    # Fill any unassigned position ids deterministically.
    while len(position_ids) < n - 1:
        position_ids.append(0)

    # EOS prediction is excluded from loss.
    for i, y in enumerate(labels):
        if y == 2:
            loss_mask[i] = 0

    attention_mask = [[0] * (n - 1) for _ in range(n - 1)]
    for i in range(n - 1):
        for j in range(i + 1):
            if segment_ids[i] is not None and segment_ids[i] == segment_ids[j]:
                attention_mask[i][j] = 1

    return {
        "input_ids": input_ids,
        "labels": labels,
        "loss_mask": loss_mask,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "source_segments": [
            {"sample_id": sid, "start": s, "end": e}
            for s, e, sid in sources
        ],
    }

def batch_hash(batch):
    return sha256_obj(batch)

def validate_packed(p):
    assert len(p["input_ids"]) == len(p["labels"]) == len(p["loss_mask"])
    assert len(p["attention_mask"]) == len(p["position_ids"]) == len(p["input_ids"])
    for i, m in enumerate(p["loss_mask"]):
        if m == 0:
            assert p["labels"][i] == -100 or p["labels"][i] == 2
    return True

def validate_packed_sequence(p):
    n = len(p["input_ids"])
    assert len(p["labels"]) == n
    assert len(p["loss_mask"]) == n
    assert len(p["position_ids"]) == n
    assert len(p["attention_mask"]) == n
    assert all(len(row) == n for row in p["attention_mask"])
    # Causal + same-segment only.
    segments = [None] * n
    for seg in p["source_segments"]:
        for i in range(max(0, seg["start"]), min(n, seg["end"] - 1)):
            segments[i] = seg["sample_id"]
    for i in range(n):
        for j in range(n):
            allowed = j <= i and segments[i] is not None and segments[i] == segments[j]
            assert p["attention_mask"][i][j] == int(allowed)
    return True
