from .utils import write_json, read_json, sha256_obj

def save_checkpoint(path, step, consumption_offset, learning_offset, next_batch, stage, rng_state):
    obj = {
        "checkpoint_id": f"ckpt-{step:04d}",
        "step": step,
        "consumption_ledger_offset": consumption_offset,
        "learning_ledger_offset": learning_offset,
        "next_batch_id": next_batch["batch_id"],
        "next_batch_hash": next_batch["batch_hash"],
        "mixture_stage": stage,
        "rng_state_hash": sha256_obj(rng_state),
    }
    write_json(path, obj)
    return obj

def load_checkpoint(path):
    return read_json(path)
