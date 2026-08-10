# V5 Training Data Execution System

A small deterministic Training Data Execution System demonstrating:

- immutable tokenized shards + manifests
- frozen tokenizer/content hashes
- train/eval firewall
- curriculum + lane weights + protected floors
- OPUS accept/reject/defer/floor override
- deterministic packing, loss masks, attention masks and position ids
- consumption + learning ledgers
- checkpoints tied to ledger offsets
- deliberate crash + exact resume
- historical replay
- checkpoint forking
- generated evidence and performance reports

## Run

```bash
python run_demo.py
```

The command regenerates `submission_artifacts/` and runs the full demonstration.

## Tests

```bash
pytest -q
```

The implementation intentionally uses a tiny deterministic tokenizer and toy model.
The goal is data-system correctness, lineage, reproducibility and auditability,
not model quality or scale.

## Final hardening

The demo also exercises a true multi-sample packed sequence policy with
causal, same-sample-only block attention and local position IDs. Tests verify
packing isolation, tokenizer content-hash detection, all OPUS outcomes,
ledger offsets, resume mismatch rejection, and generated evidence.
